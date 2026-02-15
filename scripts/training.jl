# main script to train the ConditionalSDF model
using Pkg
Pkg.activate(@__DIR__)

using Reactant
using ConditionalDeepSDFs: ConditionalSDF, MeshSDFSampler, SamplingParameters, SDFEikonalLoss,
    evaluate_dataset_loss, load_datasets, sample_sdf_and_eikonal_points
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

function save_checkpoint(train_state::Training.TrainState, save_dir::String, epoch::Int)
    model = train_state.model
    params = train_state.parameters |> cpu
    states = Lux.testmode(train_state.states) |> cpu
    trained_model = (; model, params, states)
    model_path = joinpath(save_dir, "model_epoch_$(epoch).jld2")
    # delete previously saved model parameters
    rm(save_dir; recursive=true, force=true)
    mkpath(save_dir)
    # save the current model parameters
    save_object(model_path; trained_model)
    return model_path
end

function train_model(
    rng::AbstractRNG,
    dataset_dir::String;
    model_save_dir="trained_model",
    weight_eikonal::Float32=0.1f0,
    # set training hyperparameters
    num_epochs::Integer,
    learning_rate::Float32=1f-3,
    weight_decay::Float32=1f-4
)
    sampling_params = SamplingParameters(;
        num_samples=262_144,
        grid_resolution=256,
        ratio_eikonal=0.3f0,
        clamp_voxel_threshold=16,
        eikonal_voxel_threshold=2,
        splits=(; surface=0.2f0, band=0.7f0, volume=0.1f0),
        splits_band=(0.35f0, 0.3f0, 0.2f0, 0.15f0),
        voxel_σs=(1, 4, 8, 12)
    )

    model = ConditionalSDF(;
        num_fourier=128,
        fourier_scale=10.0f0,
        scale_film=0.1f0,
        dim_p=4,
        dim_hidden=512,
        dim_film=128
    )
    display(model)

    # setup model parameters and states
    (ps, st) = Lux.setup(rng, model)
    params = device(ps)
    states = device(st)

    # load dataset into CPU memory
    (mesh_samplers_train, mesh_samplers_val, _) = load_datasets(dataset_dir)
    num_meshes_train = length(mesh_samplers_train)

    # instantiate optimiser
    optimiser = AdamW(eta=learning_rate, lambda=weight_decay)
    # instantiate training state
    train_state = Training.TrainState(model, params, states, optimiser)
    loss_func = SDFEikonalLoss(sampling_params.threshold_clamp, weight_eikonal)
    ad_engine = AutoEnzyme()

    # precompile model for validation evaluation
    evaluate_dataset_loss_compiled = @compile evaluate_dataset_loss(
        model, params, states, mesh_samplers_val, sampling_params
    )
    loss_val_min = evaluate_dataset_loss_compiled(
        model, params, states, mesh_samplers_val, sampling_params
    )
    @printf "Validation loss before training:  %4.6f\n" loss_val_min

    @info "Training..."
    for epoch in 1:num_epochs
        loss_train = 0.0f0
        for sampler in mesh_samplers_train
            data = sample_sdf_and_eikonal_points(sampler, sampling_params)
            _, loss, _, train_state = Training.single_train_step!(
                ad_engine, loss_func, data, train_state
            )
            loss_train += loss
        end
        loss_train /= num_meshes_train
        @printf "Epoch [%3d]: Training Loss  %4.6f\n" epoch loss_train

        # evaluate the model on validation set
        loss_val = evaluate_dataset_loss_compiled(
            model, train_state.parameters, train_state.states, mesh_samplers_val, sampling_params
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

const num_epochs = 300
const dataset_dir = normpath(joinpath(@__DIR__, "..", "datasets/ShapeNet-Car"))
const model_save_dir = normpath(joinpath(@__DIR__, "trained_model"))

@time (model, params, states) = train_model(rng, dataset_dir; num_epochs, model_save_dir)
