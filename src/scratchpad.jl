include("normalization.jl")
include("sdf.jl")

using FileIO
using MeshIO

mesh_path = "data/stl_files/28.stl"
mesh = load(mesh_path)

mesh = normalize_mesh(mesh)

sdf = compute_sdf(mesh)
