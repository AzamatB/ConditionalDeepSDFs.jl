using FileIO
using MeshIO

include("canonicalization.jl")
include("sdf.jl")

for n in 28:32
    mesh_path = "data/stl_files/$n.stl"
    mesh = load(mesh_path)
    @time mesh = canonicalize(mesh)
    @time sdf = construct_sdf(mesh, 256)
    @time mesh = construct_mesh(sdf)
    figure = visualize(mesh)
end
