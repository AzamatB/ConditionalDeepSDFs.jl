using CUDA
using Statistics
using LinearAlgebra

using GeometryBasics: Mesh, Point3

function vec_to_matrix(points::Vector{Point3{T}}) where {T<:Real}
    matrix = reinterpret(reshape, T, points)
    return matrix
end

function matrix_to_vec(matrix::Matrix{T}) where {T<:Real}
    @assert size(matrix, 1) == 3 "Expected 3×N matrix with columns as points"
    vector = Point3.(eachcol(matrix))
    return vector::Vector{Point3{T}}
end

"""
    normalize_pointcloud(points::CuMatrix{Float32})

Normalize a 3D point cloud on the GPU:
- Centers at origin
- Scales to fit within unit sphere
- Aligns principal axes (largest variance along x, then y, then z)

Points should be 3×N matrix where columns are 3D points.
Returns a new normalized matrix and transformation parameters (centroid, scale, rotation).
"""
function normalize_pointcloud(points::CuMatrix{Float32})
    @assert size(points, 1) == 3 "Expected 3×N matrix with columns as points"
    n = size(points, 2)
    # step 1: center at origin
    μ = mean(points, dims=2)
    points = points .- μ
    centroid = vec(Array(μ))

    # step 2: compute covariance matrix for PCA (on GPU, then transfer small 3×3)
    cov_mat = (points * points') ./ n
    cov_cpu = Matrix(cov_mat)  # 3×3, cheap transfer

    # step 3: perform SVD for principal axes alignment
    F = svd(cov_cpu) # eigenvectors of covariance give principal directions
    rotation = F.U  # rotation matrix (principal axes)

    # canonical sign convention: ensure diagonal elements are positive
    # this makes the principal axes point in consistent directions
    for col in eachcol(rotation)
        (_, idx_max) = findmax(abs, col)
        flip = col[idx_max] < 0
        c = 1 - 2flip
        col .*= c
    end

    # ensure proper rotation (det = +1), not reflection
    flip = det(rotation) < 0
    c = 1 - 2flip
    rotation[:, 3] .*= c

    # step 4: invert rotation via transpose to align principal axes
    rot_inv = CuMatrix(rotation')
    points = rot_inv * points

    # step 5: scale to unit sphere
    distances² = sum(abs2, points; dims=1)
    radius_max = √(maximum(distances²))
    scale = 1 / radius_max
    points = scale .* points

    params = (; centroid, scale=radius_max, rotation)
    return (points, params)
end

function normalize_pointcloud(points::Vector{Point3{Float32}})
    points_mat = CuMatrix(vec_to_matrix(points))
    (points_normalized_mat, params) = normalize_pointcloud(points_mat)
    points_normalized = matrix_to_vec(Matrix(points_normalized_mat))
    return (points_normalized, params)
end

function normalize_mesh(mesh::Mesh{3,Float32})
    vertices = coordinates(mesh)
    faces = faces(mesh)
    (vertices_normalized, _) = normalize_pointcloud(vertices)
    mesh_normalized = Mesh(vertices_normalized, faces)
    return mesh_normalized::Mesh{3,Float32}
end
