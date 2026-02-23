using Pkg
Pkg.activate(@__DIR__)

using ConditionalDeepSDFs
using ConditionalDeepSDFs: GridSlabs, MeshSampler, construct_mesh, point_indices, slab_points,
    visualize, compute_signed_distance!
using JLD2

dataset_path = normpath(joinpath(@__DIR__, "..", "data/preprocessed/mesh_samplers.jld2"))
resolution = 256
slab_size = 8

# load the mesh samplers
mesh_samplers = load_object(dataset_path)

# compile the model
mesh_sampler = first(mesh_samplers)
sdm = mesh_sampler.sdm
# slab_size must evenly divide resolution
grid_slabs = GridSlabs(resolution, slab_size)

# run full inference on a single mesh
sdf_flat = Array{Float32}(undef, resolution^3)
for idx in eachindex(grid_slabs)
    local points = slab_points(grid_slabs, idx)
    indices = point_indices(grid_slabs, idx)
    out = view(sdf_flat, indices)
    compute_signed_distance!(out, sdm, points)
end
# reshape into a signed distance field over the grid
sdf = reshape(sdf_flat, resolution, resolution, resolution)
# construct a mesh from the signed distance field
mesh = construct_mesh(sdf)
# visualize the mesh
visualize(mesh)
