include("canonicalization.jl")
include("sdf.jl")

using FileIO
using MeshIO

mesh_path = "data/stl_files/28.stl"
mesh = load(mesh_path)

@time mesh = canonicalize(mesh)

@time sdf = compute_sdf(mesh, 256);
sdf_cpu = Array(sdf);

using Meshing

points, fcs = isosurface(sdf_cpu, MarchingCubes())
points, fcs = isosurface(sdf_cpu, MarchingTetrahedra())

ps = reinterpret(Point3{Float64}, points)
fs = reinterpret(TriangleFace{Int}, fcs)
msh = GeometryBasics.Mesh(ps, fs)

using GLMakie

figure = Figure()
lscene = LScene(figure[1, 1])  # LScene for full 3D camera control
mesh!(lscene, msh, color = :lightblue, shading = true)
figure
