include("canonicalization.jl")
include("sdf.jl")

using FileIO
using MeshIO

mesh_path = "data/stl_files/28.stl"
mesh = load(mesh_path)

@time mesh = canonicalize(mesh)

@time sdf = construct_sdf(mesh, 256);
@time mesh = construct_mesh(sdf)
visualize(mesh)
