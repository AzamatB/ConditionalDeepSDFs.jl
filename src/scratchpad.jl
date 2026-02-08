using FileIO
using MeshIO

include("canonicalization.jl")
include("sdf.jl")

mesh_path = "data/stl_files/32.stl"
mesh = load(mesh_path)

@time mesh = canonicalize(mesh)
@time sdf = construct_sdf(mesh, 256);
@time mesh = construct_mesh(sdf)
visualize(mesh)
