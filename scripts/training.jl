# main script to train the ConditionalSDF model
using Pkg
Pkg.activate(@__DIR__)

using Reactant
using ConditionalDeepSDFs
using ConditionalDeepSDFs: ConditionalSDF, MeshSDFSampler, SamplingParameters,
    SDFSamplingBuffer, EikonalSamplingBuffer, SDFEikonalLoss,
    mean_and_std_parameters, sample_sdf_points!, sample_sdf_and_eikonal_points!
using GeometryBasics
using JLD2
using Lux
using Optimisers
using Printf
using Random

Reactant.set_default_backend("gpu")            # "gpu" = CUDA backend (via XLA/PJRT)
const device = reactant_device(; force=true)   # error if no functional Reactant GPU device
const cpu = cpu_device()                       # move results back to host for inspection

# set random seed for reproducibility
const rng = Random.default_rng()
Random.seed!(rng, 42)

function load_mesh_samplers!(
    dataset_path::String, rng::AbstractRNG; do_shuffle::Bool=true, num::Val{N}=Val(8)
) where {N}
    mesh_samplers = load_object(dataset_path)
    do_shuffle && shuffle!(rng, mesh_samplers)
    n = length(mesh_samplers)
    m = n - N
    k = m - N

    mesh_samplers_tv = @view mesh_samplers[1:m]
    (μ, σ) = mean_and_std_parameters(mesh_samplers_tv)

    mesh_samplers_train = mesh_samplers
    mesh_samplers_val = ntuple(i -> mesh_samplers[k+i], num)
    mesh_samplers_test = ntuple(i -> mesh_samplers[m+i], num)
    resize!(mesh_samplers_train, k)

    (base, ext) = splitext(dataset_path)
    dataset_path_test = base * "_test.jld2"
    save_object(dataset_path_test, mesh_samplers_test)
    return (mesh_samplers_train, mesh_samplers_val, mesh_samplers_test, μ, σ)
end

function save_checkpoint(train_state::Training.TrainState, save_dir::String, epoch::Int)
    model = train_state.model
    params = train_state.parameters |> cpu
    states = Lux.testmode(train_state.states) |> cpu
    trained_model = (; model, params, states)
    model_path = joinpath(save_dir, "trained_model_epoch_$(epoch).jld2")
    # delete previously saved model parameters
    rm(save_dir; recursive=true, force=true)
    mkpath(save_dir)
    # save the current model parameters
    save_object(model_path, trained_model)
    return model_path
end

function load_checkpoint(model_path::String)
    trained_model = load_object(model_path)
    return (; trained_model.model, trained_model.params, trained_model.states)
end

function train_model(
    rng::AbstractRNG,
    dataset_path::String;
    model_path::Union{String,Nothing}=nothing,
    model_save_dir::String=normpath(joinpath(@__DIR__, "..", "trained_models")),
    weight_eikonal::Float32=0.1f0,
    # set training hyperparameters
    num_epochs::Integer,
    learning_rate::Float32=1f-3,
    weight_decay::Float32=1f-4
)
    sampling_params = SamplingParameters(
        rng;
        num_samples=131_072,
        grid_resolution=256,
        ratio_eikonal=0.25f0,
        clamp_voxel_threshold=16,
        eikonal_voxel_threshold=2,
        splits=(; surface=0.2f0, band=0.7f0, volume=0.1f0),
        splits_band=(0.35f0, 0.3f0, 0.2f0, 0.15f0),
        voxel_σs=(1, 4, 8, 12)
    )
    threshold_clamp = sampling_params.threshold_clamp

    # pre-allocate sampling buffers
    # double-buffering for asynchronous data generation to prevent data races
    eikonal_buffers = (EikonalSamplingBuffer(sampling_params), EikonalSamplingBuffer(sampling_params))
    sdf_buffer = SDFSamplingBuffer(sampling_params)

    # load dataset into CPU memory
    @time (mesh_samplers_train, mesh_samplers_val, _, μ, σ) = load_mesh_samplers!(dataset_path, rng)
    num_meshes_train = length(mesh_samplers_train)
    @info "Number of meshes in training set: $num_meshes_train"

    # initialize model instance together with its parameters and states
    if isnothing(model_path)
        model = ConditionalSDF(
            μ,
            σ;
            num_fourier=128,
            fourier_scale=10.0f0,
            scale_film=0.1f0,
            dim_hidden=512,
            dim_film=128
        )
        # setup model parameters and states
        (ps, st) = Lux.setup(rng, model)
    else # warm start training from a pretrained model
        (model, ps, st) = load_checkpoint(model_path)
    end
    display(model)
    params = device(ps)
    states = device(Lux.trainmode(st))

    # instantiate optimiser
    optimiser = AdamW(eta=learning_rate, lambda=weight_decay)
    # instantiate training state
    train_state = Training.TrainState(model, params, states, optimiser)
    loss_func = SDFEikonalLoss(threshold_clamp, weight_eikonal)
    ad_engine = AutoEnzyme()

    # precompile model for validation evaluation
    # each sample_sdf_points! call overwrites sdf_buffer, so send to device before next call
    samples_batch = map(mesh_samplers_val) do sampler
        sample_sdf_points!(sdf_buffer, sampler, sampling_params) |> device
    end
    evaluate_dataset_loss_compiled = @compile ConditionalDeepSDFs.evaluate_dataset_loss(
        model, params, states, samples_batch, threshold_clamp
    )
    function evaluate_dataset_loss(model, params, states, samples_batch, threshold_clamp)
        loss = evaluate_dataset_loss_compiled(model, params, states, samples_batch, threshold_clamp)
        return Reactant.to_number(loss)
    end

    loss_val_min = evaluate_dataset_loss(model, params, states, samples_batch, threshold_clamp)
    @printf "Validation loss before training:  %4.6f\n" loss_val_min

    @info "Training..."
    for epoch in 1:num_epochs
        loss_train = 0.0f0

        # shuffle training data each epoch
        shuffle!(rng, mesh_samplers_train)

        # asynchronous data loading pipeline
        channel = Channel(2; spawn=true) do ch
            for (i, sampler) in enumerate(mesh_samplers_train)
                buffer = eikonal_buffers[(i%2)+1]
                samples_cpu = sample_sdf_and_eikonal_points!(buffer, sampler, sampling_params)
                samples_gpu = samples_cpu |> device
                put!(ch, samples_gpu)
            end
        end

        for samples in channel
            _, loss, _, train_state = Training.single_train_step!(
                ad_engine, loss_func, samples, train_state
            )
            loss_train += Reactant.to_number(loss)
        end
        loss_train /= num_meshes_train
        @printf "Epoch [%3d]: Training Loss  %4.6f\n" epoch loss_train

        # evaluate the model on validation set
        samples_batch = map(mesh_samplers_val) do sampler
            sample_sdf_points!(sdf_buffer, sampler, sampling_params) |> device
        end
        loss_val = evaluate_dataset_loss(
            model, train_state.parameters, train_state.states, samples_batch, threshold_clamp
        )
        @printf "Epoch [%3d]: Validation loss  %4.6f\n" epoch loss_val
        if loss_val < loss_val_min
            loss_val_min = loss_val
            @info "Saving pretrained model weights with validation loss  $loss_val_min"
            save_checkpoint(train_state, model_save_dir, epoch)
        end
    end
    @info "Training completed."
    output = (;
        model=train_state.model,
        params=train_state.parameters,
        states=Lux.testmode(train_state.states)
    )
    return output
end

####################################################################################################

const num_epochs = 700

const dataset_path = normpath(joinpath(@__DIR__, "..", "data/preprocessed/mesh_samplers.jld2"))
const model_save_dir = normpath(joinpath(@__DIR__, "..", "trained_models"))
# model_path = nothing
model_path = joinpath(model_save_dir, "trained_model_epoch_458.jld2")

(model, params, states) = train_model(rng, dataset_path; model_path, num_epochs, model_save_dir)
