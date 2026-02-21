using Pkg
Pkg.activate(@__DIR__)

using ConditionalDeepSDFs
using ConditionalDeepSDFs: ConditionalSDF, GridSlabs, LazyUnitCubeGrid, MeshSampler,
    construct_mesh, point_indices, slab_points, visualize
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
slab_size = 8   # or 4

# compile the model
mesh_sampler = first(mesh_samplers)
n = mesh_sampler.resolution
# slab_size must evenly divide n
grid_slabs = GridSlabs(n, slab_size)
points = device(slab_points(grid_slabs, 1))
mesh_params = device(mesh_sampler.parameters)
model_compiled = @compile model((points, mesh_params), params, states)

# run full inference on a single mesh
mesh_params = device(mesh_sampler.parameters)
sdf_flat = Array{Float32}(undef, n^3)
for idx in eachindex(grid_slabs)
    points = device(slab_points(grid_slabs, idx))
    indices = point_indices(grid_slabs, idx)
    (signed_dists, _) = model_compiled((points, mesh_params), params, states)
    copyto!(view(sdf_flat, indices), cpu(signed_dists))
end
# reshape into a signed distance field over the grid
sdf = reshape(sdf_flat, n, n, n)
# construct a mesh from the signed distance field
mesh = construct_mesh(sdf)
# visualize the mesh
visualize(mesh)
