include("normalization.jl")
include("sdf.jl")

using FileIO
using MeshIO

mesh_path = "data/stl_files/28.stl"
mesh = load(mesh_path)

mesh = normalize_mesh(mesh)

using GLMakie

fig = Figure()
ax = LScene(fig[1, 1])  # LScene for full 3D camera control
mesh!(ax, mesh, color=:coral)
fig



sdf = compute_sdf(mesh, 128);

sdf_cpu = Array(sdf);

using Meshing

points, faces = isosurface(sdf_cpu, MarchingCubes())
points, faces = isosurface(sdf_cpu, MarchingTetrahedra())

ps = reinterpret(Point3{Float64}, points)
fs = reinterpret(TriangleFace{Int}, faces)

msh = Mesh(ps, fs)

using GLMakie

fig = Figure()
ax = LScene(fig[1, 1])  # LScene for full 3D camera control
mesh!(ax, msh, color=:coral)
fig
