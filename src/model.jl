##############   Fourier Feature Positional Encoding (state holds random matrix B)   ##############

"""
Fourier feature mapping: x (3×N) -> φ(x) (D×N), where
φ(x) = [x; sin(2π Bx); cos(2π Bx)] if include_input=true
B is sampled once at initialization and stored in the state (non-trainable).
"""
struct FourierFeatures{T} <: LuxCore.AbstractLuxLayer
    num_features::Int   # number of random frequencies
    scale::T            # frequency scale
end

function LuxCore.initialparameters(rng::AbstractRNG, ::FourierFeatures)
    return (;)
end

function LuxCore.initialstates(rng::AbstractRNG, layer::FourierFeatures)
    noise = randn(rng, Float32, layer.num_features, 3)   # num_features × 3
    B = layer.scale .* noise                             # num_features × 3
    return (; B)
end

function (layer::FourierFeatures)(x::AbstractMatrix, params::NamedTuple, states::NamedTuple)
    # x is 3×N
    B = states.B
    two_pi = 2.0f0 * π
    proj = B * x                  # num_features × N
    sins = sin.(two_pi .* proj)   # num_features × N
    coss = cos.(two_pi .* proj)   # num_features × N
    y = [x; sins; coss]           # (3 + 2*num_features) × N
    return (y, states)
end

##################   Custom linear layers that inject p without explicit tiling   ##################

"""
Linear layer that takes (x, p) and computes: Wx*x + Wp*p + b
- x: (Dx × N)
- p: (Dp × 1)
- output: (Do × N)   (Wp*p is Do × 1 and broadcasts across N)
"""
struct LinearXP <: LuxCore.AbstractLuxLayer
    dim_x_in::Int
    dim_p_in::Int
    dim_out::Int
end

function LuxCore.initialparameters(rng::AbstractRNG, layer::LinearXP)
    dim_out = layer.dim_out
    Wx = glorot_uniform(rng, dim_out, layer.dim_x_in)
    Wp = glorot_uniform(rng, dim_out, layer.dim_p_in)
    b = zeros(Float32, layer.dim_out)
    return (; Wx, Wp, b)
end

function LuxCore.initialstates(::AbstractRNG, ::LinearXP)
    return (;)
end

function (layer::LinearXP)(
    input::Tuple{AbstractArray{Float32},AbstractArray{Float32}},
    params::NamedTuple,
    states::NamedTuple
)
    (x, p) = input
    y = params.Wx * x .+ params.Wp * p .+ params.b
    return (y, states)
end

"""
Skip-connection linear layer that takes (h, x_enc, p) and computes:
Wh*h + Wx*x_enc + Wp*p + b

- h:      (Dh×N)
- x_enc:  (Dx×N)
- p:      (Dp×1)
- output: (Do×N)
"""
struct LinearHXP <: LuxCore.AbstractLuxLayer
    dim_h_in::Int
    dim_x_in::Int
    dim_p_in::Int
    dim_out::Int
end

function LuxCore.initialparameters(rng::AbstractRNG, layer::LinearHXP)
    dim_out = layer.dim_out
    Wh = glorot_uniform(rng, dim_out, layer.dim_h_in)
    Wx = glorot_uniform(rng, dim_out, layer.dim_x_in)
    Wp = glorot_uniform(rng, dim_out, layer.dim_p_in)
    b = zeros(Float32, layer.dim_out)
    return (; Wh, Wx, Wp, b)
end

function LuxCore.initialstates(::AbstractRNG, ::LinearHXP)
    return (;)
end

function (layer::LinearHXP)(
    input::Tuple{AbstractArray{Float32},AbstractArray{Float32},AbstractArray{Float32}},
    params::NamedTuple,
    states::NamedTuple
)
    (h, x_enc, p) = input
    y = params.Wh * h .+ params.Wx * x_enc .+ params.Wp * p .+ params.b
    return (y, states)
end

###################   FiLM-conditioned SDF Network (8 layers, skip at layer 5)   ###################

struct FiLMBlock{N} end

function (film_block::FiLMBlock{N})(
    h, params::AbstractArray{Float32,3}, scale::Float32, activation
) where {N}
    # FiLM params has shape (dim_hidden, 2, num_hidden)
    γ_raw = params[:, 1:1, N]   # dim_hidden × 1
    β_raw = params[:, 2:2, N]   # dim_hidden × 1

    # "near-identity" initialization: gamma = 1 + scale*γ_raw, beta = scale*β_raw
    γ = 1.0f0 .+ scale .* γ_raw
    β = scale .* β_raw

    y = h .* γ .+ β
    out = activation.(y)
    return out
end

"""
Condition FiLM SDF network:
- input: (x::3×N, p::4×1)
- positional encoding: FourierFeatures(x)
- conditioning: FiLM scales/shifts for each hidden layer from p via a small "hyper" MLP
- skip connection after 4th hidden layer (layer 5 mixes h + x_enc + p via LinearHXP)
- output: sdf values (1×N)
"""
struct ConditionalSDF{A,PE,FiLM,L1,L2,L3,L4,L5,L6,L7,L8,Out} <: LuxCore.AbstractLuxContainerLayer{
    (:pos_encoder, :film, :layer_1, :layer_2, :layer_3, :layer_4, :layer_5, :layer_6, :layer_7, :layer_8, :out)
}
    dim_hidden::Int
    num_hidden::Int
    scale_film::Float32
    activation::A

    pos_encoder::PE
    film::FiLM
    layer_1::L1
    layer_2::L2
    layer_3::L3
    layer_4::L4
    layer_5::L5
    layer_6::L6
    layer_7::L7
    layer_8::L8
    out::Out
end

"""
Build a standard 8 layer FiLM SDF network.
"""
function ConditionalSDF(;
    activation=swish,
    num_fourier::Int=128,
    fourier_scale::Float32=10.0f0,
    scale_film::Float32=0.1f0,
    dim_p::Int=4,
    dim_hidden::Int=512,
    dim_film::Int=128
)
    # Fourier feature output dimension
    num_hidden = 8
    dim_x_enc = 3 + 2 * num_fourier
    pos_encoder = FourierFeatures(num_fourier, fourier_scale)

    # FiLM hypernetwork: p -> (γ, β) dim_hidden-dimensional conditioning vectors for each of 8 hidden layers
    dim_film_out = dim_hidden * 2 * num_hidden # outputs: 2 * dim_hidden * num_hidden values
    film = Chain(
        Dense(dim_p => dim_film, swish),
        Dense(dim_film => dim_film, swish),
        Dense(dim_film => dim_film_out) # no activation
    )

    # hidden layers
    # layer_1 consumes (x_enc, p) without explicitly tiling p
    layer_1 = LinearXP(dim_x_enc, dim_p, dim_hidden)
    # middle layers are standard dense (identity activation, as we apply activation ourselves after FiLM)
    layer_2 = Dense(dim_hidden => dim_hidden)
    layer_3 = Dense(dim_hidden => dim_hidden)
    layer_4 = Dense(dim_hidden => dim_hidden)
    # skip layer mixes (h, x_enc, p) again
    layer_5 = LinearHXP(dim_hidden, dim_x_enc, dim_p, dim_hidden)
    # final layers
    layer_6 = Dense(dim_hidden => dim_hidden)
    layer_7 = Dense(dim_hidden => dim_hidden)
    layer_8 = Dense(dim_hidden => dim_hidden)

    out = Dense(dim_hidden => 1)
    film_sdf = ConditionalSDF(
        dim_hidden,
        num_hidden,
        scale_film,
        activation,
        pos_encoder,
        film,
        layer_1,
        layer_2,
        layer_3,
        layer_4,
        layer_5,
        layer_6,
        layer_7,
        layer_8,
        out
    )
    return film_sdf
end

function (model::ConditionalSDF)(
    input::Tuple{AbstractArray{Float32},AbstractArray{Float32}},
    params::NamedTuple,
    states::NamedTuple
)
    activation = model.activation
    scale_film = model.scale_film
    (x, p) = input   # x: 3×N, p: 4×1

    # positional encoding
    (x_enc, state_enc) = model.pos_encoder(x, params.pos_encoder, states.pos_encoder)

    # FiLM params (γ, β pair per hidden layer)
    (out_film, state_film) = model.film(p, params.film, states.film)   # (dim_hidden*2*num_hidden) × 1
    params_film = reshape(out_film, model.dim_hidden, 2, model.num_hidden)

    # layer 1: (x_enc, p)
    film_1 = FiLMBlock{1}()
    (h_1, state_1) = model.layer_1((x_enc, p), params.layer_1, states.layer_1)
    x_2 = film_1(h_1, params_film, scale_film, activation)

    # layer 2
    film_2 = FiLMBlock{2}()
    (h_2, state_2) = model.layer_2(x_2, params.layer_2, states.layer_2)
    x_3 = film_2(h_2, params_film, scale_film, activation)

    # layer 3
    film_3 = FiLMBlock{3}()
    (h_3, state_3) = model.layer_3(x_3, params.layer_3, states.layer_3)
    x_4 = film_3(h_3, params_film, scale_film, activation)

    # layer 4
    film_4 = FiLMBlock{4}()
    (h_4, state_4) = model.layer_4(x_4, params.layer_4, states.layer_4)
    x_5 = film_4(h_4, params_film, scale_film, activation)

    # skip layer 5: (h, x_enc, p)
    film_5 = FiLMBlock{5}()
    (h_5, state_5) = model.layer_5((x_5, x_enc, p), params.layer_5, states.layer_5)
    x_6 = film_5(h_5, params_film, scale_film, activation)

    # layer 6
    film_6 = FiLMBlock{6}()
    (h_6, state_6) = model.layer_6(x_6, params.layer_6, states.layer_6)
    x_7 = film_6(h_6, params_film, scale_film, activation)

    # layer 7
    film_7 = FiLMBlock{7}()
    (h_7, state_7) = model.layer_7(x_7, params.layer_7, states.layer_7)
    x_8 = film_7(h_7, params_film, scale_film, activation)

    # layer 8
    film_8 = FiLMBlock{8}()
    (h_8, state_8) = model.layer_8(x_8, params.layer_8, states.layer_8)
    x_out = film_8(h_8, params_film, scale_film, activation)

    # output
    (sdf, state_out) = model.out(x_out, params.out, states.out)

    states_out = (;
        pos_encoder=state_enc,
        film=state_film,
        layer_1=state_1,
        layer_2=state_2,
        layer_3=state_3,
        layer_4=state_4,
        layer_5=state_5,
        layer_6=state_6,
        layer_7=state_7,
        layer_8=state_8,
        out=state_out
    )
    return (sdf, states_out)
end

#######################   Loss: Truncated L₁ SDF + Eikonal Regularization   #######################

struct SDFEikonalLoss
    trunc::Float32        # truncation distance (e.g. 0.05..0.2 in normalized units)
    weight_eik::Float32   # weight for eikonal regularization term
end

"""
Loss function callable for Lux.Training.single_train_step!

Expected data tuple: (x_sdf, sdf_clamped, x_eik, p)
- x_sdf:        3 × n_sdf  points with ground-truth SDF values
- sdf_clamped:  1 × n_sdf  clamped ground-truth SDF values
- x_eik:        3 × n_eik  points for eikonal term (no GT needed)
- p:            4 × 1      shape parameters for the object
"""
function (loss::SDFEikonalLoss)(
    model::ConditionalSDF,
    params::NamedTuple,
    states::NamedTuple,
    data::Tuple{AbstractArray{Float32},AbstractArray{Float32},AbstractArray{Float32},AbstractArray{Float32}}
)
    δ = loss.trunc
    λ = loss.weight_eik
    (x_sdf, sdf_clamped, x_eik, p) = data
    n_eik = size(x_eik, 2)

    # SDF L₁ regression pass
    (sdf_hat, states_out) = Lux.apply(model, (x_sdf, p), params, states)
    loss_sdf = mean(abs.(clamp.(sdf_hat, -δ, δ) .- sdf_clamped))

    # Eikonal regularization term via Vector-Jacobian Product (VJP) AD
    # freeze states as we don't want to update it during eikonal regularization pass
    states_eik = Lux.testmode(states_out)

    # f(x) must return only the primal output (no state threading here)
    f_x = function (x)
        (y, _) = Lux.apply(model, (x, p), params, states_eik)   # ignore returned state
        return y
    end
    # u must have the same structure/shape/type as the output of f(x_eik), which is (1 × n_eik)
    # allocate u on the same device as the model outputs
    u = similar(sdf_hat, 1, n_eik)
    uno = one(eltype(u))
    fill!(u, uno)

    # VJP: v = u ⋅ (∂f/∂x)  -> has same shape as x_eik (3 × n_eik)
    ∇ₓf = Lux.vector_jacobian_product(f_x, AutoEnzyme(), x_eik, u)   # 3 × n_eik
    norm∇ₓf² = sum(abs2, ∇ₓf; dims=1)           #  ‖∇ₓf‖²              1 × n_eik
    ε = 1.0f-8
    loss_eik = mean(abs2.(sqrt.(norm∇ₓf² .+ ε) .- 1.0f0))    # (√(‖∇ₓf‖² + ε) - 1)²

    Σloss = loss_sdf + λ * loss_eik
    stats = (; loss_sdf, loss_eik)
    return (Σloss, states_out, stats)
end

function evaluate_dataset_loss(
    model::ConditionalSDF,
    params::NamedTuple,
    states::NamedTuple,
    mesh_samplers::Vector{MeshSDFSampler},
    sampling_params::SamplingParameters
)
    δ = sampling_params.threshold_clamp
    loss = 0.0f0
    states_val = Lux.testmode(states)
    for mesh_sampler in mesh_samplers
        (xs, sdf_clamped, p) = sample_sdf_points(mesh_sampler, sampling_params)
        (sdf_hat, _) = model((xs, p), params, states_val)
        # SDF L₁ regression pass
        loss_sdf = mean(abs.(clamp.(sdf_hat, -δ, δ) .- sdf_clamped))
        loss += loss_sdf
    end
    loss /= length(mesh_samplers)
    return loss
end
