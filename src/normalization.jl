using CUDA
using GeometryBasics
using LinearAlgebra
using Statistics

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
    triangles = faces(mesh)
    (vertices_normalized, _) = normalize_pointcloud(vertices)
    mesh_normalized = Mesh(vertices_normalized, triangles)
    return mesh_normalized::Mesh{3,Float32}
end

"""
    merge_vertices(mesh::Mesh{3, Float32, GLTriangleFace}; ε::Float64=1e-7)

Deterministic vertex welding using lexicographic sort (x,y,z) + sweep line window on x.

- Merges vertices connected by edges of length ≤ ε (single-linkage via union-find).
- Sets welded vertex positions to the centroid (mean) of their cluster.
- Remaps faces and removes degenerate and duplicate triangles (repeated indices).

Returns: `Mesh{3, Float32, TriangleFace{Int32}}`.

Note: Worst-case can be O(n^2) if many vertices fall into the same x-slab of width ε.
"""
function merge_vertices(mesh::Mesh{3,Float32,GLTriangleFace}; ε::Float64=1e-7)
    @assert ε > 0 "ε tolerance must be positive"
    vertices = coordinates(mesh)
    faces_old = faces(mesh)
    num_vertices = length(vertices)
    num_faces = length(faces_old)

    (num_vertices == 0) && return Mesh(Point{3,Float32}[], TriangleFace{Int32}[])

    ε² = ε * ε
    # sort indices lexicographically by (x, y, z)
    @inline @inbounds function is_less(i::Int, j::Int)
        pi = vertices[i]
        pj = vertices[j]
        Δ = pi - pj
        (Δx, Δy, Δz) = Δ
        # prioritized comparison: x, then y, then z
        cmp = ifelse(Δx == 0f0, Δy, Δx)
        cmp = ifelse(cmp == 0f0, Δz, cmp)
        return cmp < 0f0
    end
    perm = collect(1:num_vertices)
    sort!(perm, lt=is_less)

    # union-find (disjoint set union)
    parents = collect(1:num_vertices)
    sizes = ones(Int32, num_vertices)

    @inline function find(x::Int)
        @inbounds while true
            px = parents[x]
            (px == x) && return x
            parents[x] = parents[px]
            x = parents[x]
        end
    end

    @inline @inbounds function unite(a::Int, b::Int)
        ra = find(a)
        rb = find(b)
        (ra == rb) && return ra
        if sizes[ra] < sizes[rb]
            (ra, rb) = (rb, ra)
        end
        parents[rb] = ra
        sizes[ra] += sizes[rb]
        return ra
    end

    # sweep with sliding x-window: only compare to earlier vertices with x within ε
    lo = 1
    @inbounds for hi in 1:num_vertices
        ii = perm[hi]
        pi = vertices[ii]
        xi = Float64(pi[1])
        yi = Float64(pi[2])
        zi = Float64(pi[3])

        while lo < hi
            jj = perm[lo]
            pj = vertices[jj]
            xj = Float64(pj[1])
            Δx = xi - xj
            (Δx > ε) || break
            lo += 1
        end

        # compare against all candidates in [lo, hi-1]
        for k in lo:(hi-1)
            jj = perm[k]
            pj = vertices[jj]

            Δy = yi - Float64(pj[2])
            (Δy * Δy > ε²) && continue
            Δz = zi - Float64(pj[3])
            (Δz * Δz > ε²) && continue
            Δx = xi - Float64(pj[1]) # >= 0 because of sorting; and Δx <= ε due to window
            dist² = Δx * Δx + Δy * Δy + Δz * Δz
            if dist² <= ε²
                unite(ii, jj)
            end
        end
    end

    # compute centroids per component
    Σx = zeros(Float64, num_vertices)
    Σy = zeros(Float64, num_vertices)
    Σz = zeros(Float64, num_vertices)
    count = zeros(Int32, num_vertices)

    @inbounds for i in 1:num_vertices
        r = find(i)
        p = vertices[i]
        Σx[r] += p[1]
        Σy[r] += p[2]
        Σz[r] += p[3]
        count[r] += 1
    end

    P = eltype(vertices)
    vertices_new = Vector{P}()
    sizehint!(vertices_new, num_vertices)
    root2new = zeros(Int32, num_vertices)
    @inbounds for r in 1:num_vertices
        c = count[r]
        if c != 0
            weight = inv(c)
            x = Float32(Σx[r] * weight)
            y = Float32(Σy[r] * weight)
            z = Float32(Σz[r] * weight)
            push!(vertices_new, P(x, y, z))
            root2new[r] = length(vertices_new)
        end
    end

    old2new = Vector{Int32}(undef, num_vertices)
    @inbounds for i in 1:num_vertices
        old2new[i] = root2new[find(i)]
    end
    # remap faces and drop degenerate ones
    seen = Dict{NTuple{3,Int32},Int}()
    sizehint!(seen, num_faces)
    faces_new = Vector{TriangleFace{Int32}}(undef, num_faces)
    k = 0
    @inbounds for index in 1:num_faces
        face = faces_old[index]
        a = old2new[face[1]]
        b = old2new[face[2]]
        c = old2new[face[3]]
        # drop degenerate faces
        ((a == b) | (b == c) | (a == c)) && continue
        # dedup regardless of order (handles (1,2,3) vs (2,3,1) vs (1,3,2))
        key = sort_triplet(a, b, c)
        # "must be newly inserted" otherwise it's a duplicate face -> skip
        get!(seen, key, index) == index || continue
        k += 1
        faces_new[k] = TriangleFace{Int32}(a, b, c)  # keep the first ordering encountered
    end
    resize!(faces_new, k)
    return Mesh(vertices_new, faces_new)
end

@inline function sort_triplet(a::Int32, b::Int32, c::Int32)
    # compare-swap (a, b)
    gt = a > b
    aa = ifelse(gt, b, a)
    bb = ifelse(gt, a, b)
    a = aa
    b = bb

    # compare-swap (b, c)
    gt = b > c
    bb = ifelse(gt, c, b)
    cc = ifelse(gt, b, c)
    b = bb
    c = cc

    # compare-swap (a, b)
    gt = a > b
    aa = ifelse(gt, b, a)
    bb = ifelse(gt, a, b)

    return (aa, bb, c)
end
