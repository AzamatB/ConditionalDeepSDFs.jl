using Random, Statistics, Printf

using Lux
using LuxCore
using Optimisers
using Enzyme
using Reactant
using WeightInitializers
using NNlib: swish

# --------------------------
# Utilities
# --------------------------

"""
Smooth softplus with a "beta" sharpness parameter, implemented in a numerically stable way.
Common choice for SDF MLPs is beta ~ 50..200.
"""
struct SoftplusBeta{T}
    beta::T
end

@inline function (s::SoftplusBeta)(x)
    β = s.beta
    z = β * x
    # stable softplus(z) = log1p(exp(-abs(z))) + max(z, 0)
    return (log1p(exp(-abs(z))) + max(z, zero(z))) / β
end


# --------------------------
# Fourier Feature Positional Encoding (state holds random matrix B)
# --------------------------

"""
Fourier feature mapping: x (3×N) -> φ(x) (D×N), where
φ(x) = [x; sin(2π Bx); cos(2π Bx)] if include_input=true
B is sampled once at initialization and stored in the *state* (non-trainable).
"""
struct FourierFeatures{T} <: LuxCore.AbstractLuxLayer
    nfeat::Int               # number of random frequencies
    sigma::T                 # frequency scale
    include_input::Bool
end

LuxCore.initialparameters(::AbstractRNG, ::FourierFeatures) = NamedTuple()

function LuxCore.initialstates(rng::AbstractRNG, ff::FourierFeatures)
    # B: (nfeat × 3)
    B = ff.sigma .* randn(rng, Float32, ff.nfeat, 3)
    return (B = B,)
end

function (ff::FourierFeatures)(x::AbstractMatrix, ps, st::NamedTuple)
    # x is 3×N
    B = st.B
    two_pi = 2f0 * Float32(pi)

    proj = B * x                      # (nfeat×N)
    s = sin.(two_pi .* proj)          # (nfeat×N)
    c = cos.(two_pi .* proj)          # (nfeat×N)

    y = ff.include_input ? vcat(x, s, c) : vcat(s, c)
    return y, st
end


# --------------------------
# Custom linear layers that inject p without explicit tiling
# --------------------------

"""
Linear layer that takes (x, p) and computes: Wx*x + Wp*p + b
- x: (Dx×N)
- p: (Dp×1)
- output: (Do×N)   (Wp*p is Do×1 and broadcasts across N)
"""
struct LinearXP{F1,F2} <: LuxCore.AbstractLuxLayer
    x_in::Int
    p_in::Int
    out::Int
    init_weight::F1
    init_bias::F2
end

function LinearXP(x_in::Int, p_in::Int, out::Int;
                  init_weight=glorot_uniform, init_bias=zeros32)
    return LinearXP{typeof(init_weight), typeof(init_bias)}(
        x_in, p_in, out, init_weight, init_bias
    )
end

function LuxCore.initialparameters(rng::AbstractRNG, l::LinearXP)
    return (
        Wx = l.init_weight(rng, l.out, l.x_in),
        Wp = l.init_weight(rng, l.out, l.p_in),
        b  = l.init_bias(rng, l.out, 1),
    )
end

LuxCore.initialstates(::AbstractRNG, ::LinearXP) = NamedTuple()

function (l::LinearXP)(input::Tuple, ps, st::NamedTuple)
    x, p = input
    y = ps.Wx * x .+ ps.Wp * p .+ ps.b
    return y, st
end


"""
Skip-connection linear layer that takes (h, xenc, p) and computes:
Wh*h + Wx*xenc + Wp*p + b

- h:    (Dh×N)
- xenc: (Dx×N)
- p:    (Dp×1)
- output: (Do×N)
"""
struct LinearHXP{F1,F2} <: LuxCore.AbstractLuxLayer
    h_in::Int
    x_in::Int
    p_in::Int
    out::Int
    init_weight::F1
    init_bias::F2
end

function LinearHXP(h_in::Int, x_in::Int, p_in::Int, out::Int;
                   init_weight=glorot_uniform, init_bias=zeros32)
    return LinearHXP{typeof(init_weight), typeof(init_bias)}(
        h_in, x_in, p_in, out, init_weight, init_bias
    )
end

function LuxCore.initialparameters(rng::AbstractRNG, l::LinearHXP)
    return (
        Wh = l.init_weight(rng, l.out, l.h_in),
        Wx = l.init_weight(rng, l.out, l.x_in),
        Wp = l.init_weight(rng, l.out, l.p_in),
        b  = l.init_bias(rng, l.out, 1),
    )
end

LuxCore.initialstates(::AbstractRNG, ::LinearHXP) = NamedTuple()

function (l::LinearHXP)(input::Tuple, ps, st::NamedTuple)
    h, xenc, p = input
    y = ps.Wh * h .+ ps.Wx * xenc .+ ps.Wp * p .+ ps.b
    return y, st
end


# --------------------------
# FiLM-conditioned SDF Network (8×512, skip at layer 5)
# --------------------------

"""
FiLM SDF network:
- input: (x::3×N, p::4×1)
- positional encoding: FourierFeatures(x)
- conditioning: FiLM scales/shifts for each hidden layer from p via a small "hyper" MLP
- skip connection after 4th hidden layer (layer 5 mixes h + xenc + p via LinearHXP)
- output: sdf values (1×N)
"""
struct FiLMSDF{
    PE, Film,
    L1,L2,L3,L4,L5,L6,L7,L8,Out,
    A
} <: LuxCore.AbstractLuxContainerLayer{
    (:posenc,:film,:l1,:l2,:l3,:l4,:l5,:l6,:l7,:l8,:out)
}
    posenc::PE
    film::Film

    l1::L1
    l2::L2
    l3::L3
    l4::L4
    l5::L5
    l6::L6
    l7::L7
    l8::L8
    out::Out

    act::A
    width::Int
    n_layers::Int
    film_scale::Float32
end

"""
Build a standard 8×512 FiLM SDF network.
"""
function build_film_sdf(;
    n_fourier::Int = 64,
    fourier_sigma::Float32 = 10f0,
    include_xyz::Bool = true,
    p_dim::Int = 4,
    width::Int = 512,
    n_hidden::Int = 8,                 # fixed at 8 in this implementation
    film_hidden::Int = 128,
    film_scale::Float32 = 0.1f0,
    act = SoftplusBeta(100f0),
)
    @assert n_hidden == 8 "This implementation is wired for 8 hidden layers (as requested)."

    # Fourier feature output dimension
    xenc_dim = (include_xyz ? 3 : 0) + 2 * n_fourier

    posenc = FourierFeatures(n_fourier, fourier_sigma, include_xyz)

    # FiLM hypernetwork: p -> (gamma,beta) for each of 8 hidden layers
    # outputs: 2 * width * n_hidden values
    film = Chain(
        Dense(p_dim => film_hidden, swish),
        Dense(film_hidden => film_hidden, swish),
        Dense(film_hidden => 2 * width * n_hidden) # no activation
    )

    # Hidden layers:
    # l1 consumes (xenc, p) without explicitly tiling p
    l1 = LinearXP(xenc_dim, p_dim, width)

    # middle layers are standard dense (identity activation, we apply act ourselves after FiLM)
    l2 = Dense(width => width, identity)
    l3 = Dense(width => width, identity)
    l4 = Dense(width => width, identity)

    # skip layer mixes (h, xenc, p) again
    l5 = LinearHXP(width, xenc_dim, p_dim, width)

    l6 = Dense(width => width, identity)
    l7 = Dense(width => width, identity)
    l8 = Dense(width => width, identity)

    out = Dense(width => 1, identity)

    return FiLMSDF(posenc, film, l1,l2,l3,l4,l5,l6,l7,l8,out, act, width, n_hidden, film_scale)
end

@inline function _apply_film!(h, film_params, layer_idx::Int, act, film_scale::Float32)
    # film_params has shape: (2, width, n_layers, Bp)
    γraw = film_params[1, :, layer_idx, :]  # (width×Bp)
    βraw = film_params[2, :, layer_idx, :]  # (width×Bp)

    # "near-identity" initialization: gamma = 1 + s*γraw, beta = s*βraw
    γ = 1f0 .+ film_scale .* γraw
    β = film_scale .* βraw

    h = h .* γ .+ β
    h = act.(h)
    return h
end

function (m::FiLMSDF)(input::Tuple, ps, st::NamedTuple)
    x, p = input                 # x: 3×N, p: 4×1

    # positional encoding
    xenc, st_pe = m.posenc(x, ps.posenc, st.posenc)

    # FiLM params (gamma/beta per hidden layer)
    film_out, st_film = m.film(p, ps.film, st.film)     # (2*width*n_layers)×Bp
    Bp = size(film_out, 2)
    film_params = reshape(film_out, 2, m.width, m.n_layers, Bp)

    # layer 1: (xenc, p)
    h, st_l1 = m.l1((xenc, p), ps.l1, st.l1)
    h = _apply_film!(h, film_params, 1, m.act, m.film_scale)

    # layer 2
    h, st_l2 = m.l2(h, ps.l2, st.l2)
    h = _apply_film!(h, film_params, 2, m.act, m.film_scale)

    # layer 3
    h, st_l3 = m.l3(h, ps.l3, st.l3)
    h = _apply_film!(h, film_params, 3, m.act, m.film_scale)

    # layer 4
    h, st_l4 = m.l4(h, ps.l4, st.l4)
    h = _apply_film!(h, film_params, 4, m.act, m.film_scale)

    # skip layer 5: (h, xenc, p)
    h, st_l5 = m.l5((h, xenc, p), ps.l5, st.l5)
    h = _apply_film!(h, film_params, 5, m.act, m.film_scale)

    # layer 6
    h, st_l6 = m.l6(h, ps.l6, st.l6)
    h = _apply_film!(h, film_params, 6, m.act, m.film_scale)

    # layer 7
    h, st_l7 = m.l7(h, ps.l7, st.l7)
    h = _apply_film!(h, film_params, 7, m.act, m.film_scale)

    # layer 8
    h, st_l8 = m.l8(h, ps.l8, st.l8)
    h = _apply_film!(h, film_params, 8, m.act, m.film_scale)

    # output
    y, st_out = m.out(h, ps.out, st.out)

    st_new = (
        posenc = st_pe,
        film   = st_film,
        l1 = st_l1, l2 = st_l2, l3 = st_l3, l4 = st_l4,
        l5 = st_l5, l6 = st_l6, l7 = st_l7, l8 = st_l8,
        out = st_out
    )
    return y, st_new
end


# --------------------------
# Objective: Truncated L1 SDF + Eikonal
# --------------------------

"""
Objective callable for Lux.Training.single_train_step!

Expected data tuple: (x_sdf, d_sdf, x_eik, p)
- x_sdf: 3×Nsdf   points with ground-truth SDF values
- d_sdf: 1×Nsdf
- x_eik: 3×Neik   points for eikonal term (no GT needed)
- p:     4×1      shape parameters for the object
"""
struct SDFEikonalObjective{T}
    trunc::T        # truncation distance (e.g. 0.05..0.2 in your normalized units)
    λ_eik::T        # weight for eikonal term
end

function (obj::SDFEikonalObjective)(model, ps, st, data)
    x_sdf, d_sdf, x_eik, p = data

    smodel = Lux.StatefulLuxLayer(model, ps, st)

    # SDF regression (truncated L1 via clamping)
    d̂ = smodel((x_sdf, p))                      # 1×Nsdf
    τ = obj.trunc
    d̂c = clamp.(d̂, -τ, τ)
    dc  = clamp.(d_sdf, -τ, τ)
    sdf_loss = mean(abs, d̂c .- dc)

    # Eikonal: ||∇_x f|| ≈ 1
    # Gradient wrt x of sum(f(x)) gives 3×Neik
    g = Enzyme.gradient(Enzyme.Reverse,
                        sum ∘ (x -> smodel((x, p))),
                        x_eik)[1]

    # Norm per point
    # dims=1 => (1×Neik)
    epsv = eps(eltype(g))
    grad_norm = sqrt.(sum(abs2, g; dims=1) .+ epsv)
    eik_loss = mean(abs2, grad_norm .- one(eltype(grad_norm)))

    loss = sdf_loss + obj.λ_eik * eik_loss
    return loss, smodel.st, (; sdf_loss, eik_loss)
end


# --------------------------
# Training helper (skeleton)
# --------------------------

"""
Train for `nsteps` iterations.

You must provide `sample_batch()` which returns:
    x_sdf::Matrix{Float32} (3×Nsdf),
    d_sdf::Matrix{Float32} (1×Nsdf),
    x_eik::Matrix{Float32} (3×Neik),
    p::Matrix{Float32}     (4×1)

Important for Reactant compilation:
- Keep Nsdf and Neik constant (static shapes), otherwise XLA will recompile or fail.
"""
function train!(
    model,
    train_state::Lux.Training.TrainState,
    sample_batch::Function;
    nsteps::Int = 10_000,
    log_every::Int = 100,
    trunc::Float32 = 0.1f0,
    λ_eik::Float32 = 0.1f0,
    device = reactant_device(; force=true),
)
    obj = SDFEikonalObjective(trunc, λ_eik)

    for step in 1:nsteps
        # Sample on CPU (your code), then move to Reactant device
        data_cpu = sample_batch()
        data = data_cpu |> device

        # One compiled training step (AutoEnzyme + Reactant)
        _, loss, stats, train_state = Lux.Training.single_train_step!(
            Lux.AutoEnzyme(),
            obj,
            data,
            train_state;
            return_gradients = Val(false),
        )

        if (step % log_every == 0) || (step == 1)
            @printf("step %6d | loss %.6e | sdf %.6e | eik %.6e\n",
                    step, loss, stats.sdf_loss, stats.eik_loss)
        end
    end

    return train_state
end


# --------------------------
# Convenience: create TrainState on Reactant device
# --------------------------

"""
Initialize model parameters/states and construct a TrainState on a Reactant device.
"""
function init_train_state(model;
    rng = Random.default_rng(),
    lr::Float32 = 1e-4f0,
    device = reactant_device(; force=true),
)
    ps, st = Lux.setup(rng, model)
    ps = ps |> device
    st = st |> device

    opt = Optimisers.Adam(lr)
    return Lux.Training.TrainState(model, ps, st, opt)
end


end # module
