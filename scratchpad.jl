using FileIO
using MeshIO
using CUDA
using Statistics
using LinearAlgebra

mesh_path = "data/stl_files/28.stl"
mesh = load(mesh_path)

vertices = mesh.position

matrix_cpu = reinterpret(reshape, Float32, vertices)

points = CuMatrix(matrix_cpu)


mean(points; dims=2)


"""
    normalize_pointcloud!(points::CuMatrix{T}) where T

Normalize a 3D point cloud in-place on the GPU:
- Centers at origin
- Scales to fit within unit sphere
- Aligns principal axes (largest variance along x, then y, then z)

Points should be 3×N matrix where columns are 3D points.
Returns the normalized points and transformation parameters (μ, scale, rotation).
"""
# function normalize_pointcloud!(points::CuMatrix{T}) where T
@assert size(points, 1) == 3 "Expected 3×N matrix with columns as points"
n = size(points, 2)
# step 1: center at origin
μ = mean(points, dims=2)
points .-= μ

# step 2: compute covariance matrix for PCA (on GPU, then transfer small 3×3)
cov_mat = (points * points') ./ n
cov_cpu = Matrix(cov_mat)  # 3×3, cheap transfer

# step 3: perform SVD for principal axes alignment
F = svd(cov_cpu) # eigenvectors of covariance give principal directions
rotation = F.U  # rotation matrix (principal axes)
# ensure proper rotation (det = +1), not reflection
flip = det(rotation) < 0
c = 1.0f0 - 2.0f0 * flip
rotation[:, 3] .*= c

# step 4: invert rotation via transpose to align principal axes
rot_inv = CuMatrix(rotation')
points .= rot_inv * points

# step 5: scale to unit sphere
distances² = sum(abs2, points; dims=1)
radius_max = √(maximum(distances²))
scale = 1.0f0 / radius_max
points .*= scale
return points
# end

"""
    normalize_pointcloud(points::CuMatrix{T}) where T

Non-mutating version that returns a normalized copy.
"""
function normalize_pointcloud(points::CuMatrix{T}) where T
    return normalize_pointcloud!(copy(points))
end
