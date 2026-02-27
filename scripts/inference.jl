using Pkg
Pkg.activate(@__DIR__)

using ConditionalDeepSDFs
using ConditionalDeepSDFs: ConditionalSDF, MeshSampler, ODCKernels, construct_mesh, visualize
using Reactant
using JLD2
using Lux

Reactant.set_default_backend("gpu")            # "gpu" = CUDA backend (via XLA/PJRT)
const device = reactant_device(; force=true)   # error if no functional Reactant GPU device
const cpu = cpu_device()                       # move results back to host for inspection

model_path = normpath(joinpath(@__DIR__, "..", "trained_models/trained_model_epoch_458.jld2"))
dataset_path = normpath(joinpath(@__DIR__, "..", "data/preprocessed/mesh_samplers_test.jld2"))

# load the trained model
trained_model = load_object(model_path)
model = trained_model.model
params = device(trained_model.params)
states = device(Lux.testmode(trained_model.states))
display(model)

# load the mesh samplers
mesh_samplers = load_object(dataset_path)

# compile the model
mesh_sampler = first(mesh_samplers)
mesh_params = device(mesh_sampler.parameters)

function make_signed_distance(model, mesh_params, params, states)
    function signed_distance(points)
        input = (points, mesh_params)
        return first(model(input, params, states))
    end
    return signed_distance
end

signed_dist = make_signed_distance(model, mesh_params, params, states)
odc_kernels = ODCKernels(signed_dist)
bounding_box = (mesh_sampler.bbox_min, mesh_sampler.bbox_max)
mesh = construct_mesh(odc_kernels, bounding_box)
# visualize the mesh
visualize(mesh)
