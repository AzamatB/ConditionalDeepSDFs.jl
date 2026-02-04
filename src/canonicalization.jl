using CUDA
using GeometryBasics
using LinearAlgebra

function canonicalize(mesh::Mesh{3,Float32,GLTriangleFace})
    # weld vertices that are within distance ε (single-linkage via union-find)
    mesh_repaired = weld_vertices(mesh)
    # shift the mesh so that its vertex centroid is at the origin
    shift_to_origin!(mesh_repaired)
    # reorient the mesh so that its faces are oriented outward
    mesh_oriented = reorient_outward(mesh_repaired)
    # align the mesh so that its principal axes are aligned with the coordinate axes
    # and rescale it to fit the unit sphere (runs on GPU)
    mesh_normalized = align_and_rescale(mesh_oriented)
    return mesh_normalized
end

###############################   Welding of Mesh Vertices   ###############################

"""
    weld_vertices(mesh::Mesh{3, Float32, GLTriangleFace}; ε::Float64=1e-7)

Deterministic vertex welding using lexicographic sort (x,y,z) + sweep line window on x.

- Welds vertices that are within distance ε (single-linkage via union-find).
- Sets welded vertex positions to the centroid (mean) of their cluster.
- Remaps faces and removes degenerate triangle faces (repeated vertex indices) and
  duplicate triangle faces (same 3 vertices up to permutation).

Returns: `Mesh{3,Float32,GLTriangleFace}`.

Note: Worst-case can be O(n^2) if many vertices fall into the same x-slab of width ε.
"""
function weld_vertices(mesh::Mesh{3,Float32,GLTriangleFace}; ε::Float64=1e-7)
    @assert ε > 0 "ε tolerance must be positive"
    vertices = coordinates(mesh)
    faces_old = faces(mesh)
    num_vertices = length(vertices)
    num_faces = length(faces_old)

    (num_vertices == 0) && return Mesh(Point{3,Float32}[], GLTriangleFace[])

    ε² = ε * ε
    # sort indices lexicographically by (x, y, z)
    @inline @inbounds function is_less(i::Int, j::Int)
        pi = vertices[i]
        pj = vertices[j]
        Δx = pi[1] - pj[1]
        Δy = pi[2] - pj[2]
        Δz = pi[3] - pj[3]
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

    VT = eltype(vertices)
    vertices_new = Vector{VT}()
    sizehint!(vertices_new, num_vertices)
    root2new = zeros(Int32, num_vertices)
    @inbounds for r in 1:num_vertices
        c = count[r]
        if c != 0
            weight = inv(c)
            x = Float32(Σx[r] * weight)
            y = Float32(Σy[r] * weight)
            z = Float32(Σz[r] * weight)
            push!(vertices_new, VT(x, y, z))
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
    FT = eltype(faces_old)  # GLTriangleFace
    faces_new = Vector{FT}(undef, num_faces)
    k = 0
    @inbounds for index in 1:num_faces
        face = faces_old[index]
        a = old2new[face[1]]
        b = old2new[face[2]]
        c = old2new[face[3]]
        # drop degenerate faces
        ((a == b) | (b == c) | (a == c)) && continue
        # dedup faces (considering all vertex orderings as same face)
        key = sort_triplet(a, b, c)
        # "must be newly inserted" otherwise it's a duplicate face -> skip
        get!(seen, key, index) == index || continue
        k += 1
        faces_new[k] = FT(a, b, c)  # keep the first ordering encountered
    end
    resize!(faces_new, k)
    num_vertices_new = length(vertices_new)
    ratio_v = round(100 * num_vertices_new / num_vertices; digits=2)
    ratio_f = round(100 * k / num_faces; digits=2)
    @info "$ratio_v% of the original $num_vertices vertices remain after performing the vertex welding."
    @info "$ratio_f% of the original $num_faces faces remain after performing the vertex welding."

    return Mesh(vertices_new, faces_new)
end

@inline function sort_triplet(a::Int32, b::Int32, c::Int32)::NTuple{3,Int32}
    # compare-swap (a, b)
    gt = a > b
    a, b = ifelse(gt, b, a), ifelse(gt, a, b)
    # compare-swap (b, c)
    gt = b > c
    b, c = ifelse(gt, c, b), ifelse(gt, b, c)
    # compare-swap (a, b)
    gt = a > b
    a, b = ifelse(gt, b, a), ifelse(gt, a, b)
    return (a, b, c)
end

##############################   Mesh Outward Reorientation   ##############################

# compact, isbits edge record for sort-based pairing
struct EdgeEntry
    key::UInt64      # packed (min, max) vertex ids
    face_id::Int32   # face index
    edge_id::Int8    # edge index: 1=(v₁,v₂), 2=(v₂,v₃), 3=(v₃,v₁)
    sign::Int8       # +1 if directed edge matches min->max, -1 otherwise
end

# sort by edge key only (fast comparisons)
@inline function Base.isless(a::EdgeEntry, b::EdgeEntry)
    return a.key < b.key
end

@inline function edge_key_sign(u::GLIndex, v::GLIndex)
    uu = UInt64(GeometryBasics.value(u))
    vv = UInt64(GeometryBasics.value(v))
    if uu <= vv
        return (uu << 32) | vv, Int8(1)
    else
        return (vv << 32) | uu, Int8(-1)
    end
end

@inline function signed_volume(
    a::Point3f, b::Point3f, c::Point3f,
    centroid_x::Float64, centroid_y::Float64, centroid_z::Float64
)
    # 6 × signed volume of tetrahedron with vertices a, b, c and origin
    ax = Float64(a[1]) - centroid_x
    ay = Float64(a[2]) - centroid_y
    az = Float64(a[3]) - centroid_z
    bx = Float64(b[1]) - centroid_x
    by = Float64(b[2]) - centroid_y
    bz = Float64(b[3]) - centroid_z
    cx = Float64(c[1]) - centroid_x
    cy = Float64(c[2]) - centroid_y
    cz = Float64(c[3]) - centroid_z
    # a ⋅ (b × c)
    vol6 = ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)
    return vol6
end

"""
    reorient_outward(mesh::Mesh{3,Float32,GLTriangleFace}; check=true) -> Mesh

Reorients triangle faces of the mesh so that:
  1) all faces are consistently oriented across shared edges, and
  2) each connected component is oriented outward (positive signed volume).

Assumes a watertight, closed triangle 2-manifold surface. If `check=true`, throws if
boundary or non-manifold edges are detected.
"""
function reorient_outward(mesh::Mesh{3,Float32,GLTriangleFace}; check::Bool=true)
    fcs = faces(mesh)
    num_faces = length(fcs)
    num_faces == 0 && return mesh
    num_faces > typemax(Int32) && throw(ArgumentError("Too many faces ($num_faces) for Int32 adjacency indexing."))

    vertices = coordinates(mesh)
    # build adjacency using sort-based edge pairing
    edges = Vector{EdgeEntry}(undef, 3num_faces)
    @inbounds for index in 1:num_faces
        face = fcs[index]
        (a, b, c) = face
        idx = Int32(index)

        (key, s) = edge_key_sign(a, b)
        edges[3index-2] = EdgeEntry(key, idx, Int8(1), s)
        (key, s) = edge_key_sign(b, c)
        edges[3index-1] = EdgeEntry(key, idx, Int8(2), s)
        (key, s) = edge_key_sign(c, a)
        edges[3index] = EdgeEntry(key, idx, Int8(3), s)
    end
    sort!(edges)

    # neighbors[edge_id, face_id] -> adjacent face_id (0 if none)
    neighbors = zeros(Int32, 3, num_faces)
    # toggle[edge_id, face_id] == true means neighbor flip must differ (xor) to satisfy consistency.
    toggles = fill(false, 3, num_faces)

    boundary_edges = 0
    nonmanifold_edges = 0

    m = length(edges)
    ptr = 1
    @inbounds while ptr <= m
        key = edges[ptr].key
        j = ptr + 1
        while j <= m && edges[j].key == key
            j += 1
        end
        run = j - ptr

        if run == 2
            edge₁ = edges[ptr]
            edge₂ = edges[ptr+1]

            face_id₁ = edge₁.face_id
            edge_id₁ = edge₁.edge_id
            face_id₂ = edge₂.face_id
            edge_id₂ = edge₂.edge_id

            # If both faces traverse the shared edge in the same direction (same sign),
            # then the faces must have opposite flip states to make the shared edge anti-parallel.
            toggle = (edge₁.sign == edge₂.sign)

            neighbors[edge_id₁, face_id₁] = face_id₂
            toggles[edge_id₁, face_id₁] = toggle
            neighbors[edge_id₂, face_id₂] = face_id₁
            toggles[edge_id₂, face_id₂] = toggle
        elseif run == 1
            boundary_edges += 1
        else
            nonmanifold_edges += 1
        end
        ptr = j
    end

    if check
        boundary_edges == 0 || throw(ArgumentError("Mesh is not closed/watertight: $boundary_edges boundary edges."))
        nonmanifold_edges == 0 || throw(ArgumentError("Mesh is non-manifold: $nonmanifold_edges edges shared by >2 faces."))
    end

    # BFS for consistent orientation + per-component outward fix
    flip = fill(false, num_faces)   # vector{Bool} for fast indexing
    seen = fill(false, num_faces)

    # preallocated BFS queue (also serves as the component face list: queue[1:tail])
    queue = Vector{Int}(undef, num_faces)
    # reference point for numerically stable signed volume accumulation.
    centroid = sum(vertices; init=zero(Point3d)) / length(vertices)
    (cx, cy, cz) = centroid

    @inbounds for start in 1:num_faces
        seen[start] && continue
        head = 1
        tail = 1
        queue[1] = start
        seen[start] = true
        vol6 = 0.0

        while head <= tail
            index = queue[head]
            head += 1

            # accumulate signed volume contribution for this oriented triangle
            face = fcs[index]
            (i1, i2, i3) = face
            (i2, i3) = ifelse(flip[index], (i3, i2), (i2, i3))
            # indexing with GLIndex directly (no conversion)
            vertex1 = vertices[i1]
            vertex2 = vertices[i2]
            vertex3 = vertices[i3]
            vol6 += signed_volume(vertex1, vertex2, vertex3, cx, cy, cz)

            # propagate constraints to neighbors
            for k in 1:3
                neighbor_id = neighbors[k, index]
                (neighbor_id == 0) && continue
                neighbor_id = Int(neighbor_id)

                required = flip[index] ⊻ toggles[k, index]

                if !seen[neighbor_id]
                    seen[neighbor_id] = true
                    flip[neighbor_id] = required
                    tail += 1
                    queue[tail] = neighbor_id
                elseif flip[neighbor_id] != required
                    throw(ArgumentError("Mesh is not consistently orientable (conflicting constraints)."))
                end
            end
        end

        # if component volume is negative, the whole component is inside-out: invert all flips
        if vol6 < 0
            for t in 1:tail
                index = queue[t]
                flip[index] = !flip[index]
            end
        end
    end

    num_reoriented = count(flip)
    ratio = round(100 * num_reoriented / num_faces; digits=2)
    @info "Reoriented $ratio% of the total $num_faces faces."

    FT = eltype(fcs)  # GLTriangleFace
    faces_new = Vector{FT}(undef, num_faces)
    @inbounds for index in 1:num_faces
        face = fcs[index]
        faces_new[index] = ifelse(flip[index], FT(face[1], face[3], face[2]), face)
    end
    return Mesh(vertices, faces_new)
end

############################   Normalization of Mesh Vertices   ############################

function shift_to_origin!(mesh::Mesh{3,Float32,GLTriangleFace})
    vertices = coordinates(mesh)
    isempty(vertices) && return mesh
    centroidd = sum(vertices; init=zero(Point3d)) / length(vertices)
    centroid = Point3f(centroidd)
    vertices .-= centroid
    return mesh
end

"""
    align_and_rescale(points::CuMatrix{Float32})

Normalize a 3D point cloud on the GPU assuming that it is already centered at the origin:
- Scales to fit within unit sphere
- Aligns principal axes (largest variance along x, then y, then z)

Points should be 3×N matrix where columns are 3D points.
Returns a new normalized matrix and transformation parameters (scale, rotation).
"""
function align_and_rescale(mesh::Mesh{3,Float32})
    vertices = coordinates(mesh)
    fcs = faces(mesh)
    (vertices_normalized, _) = align_and_rescale(vertices)
    mesh_normalized = Mesh(vertices_normalized, fcs)
    return mesh_normalized::Mesh{3,Float32}
end

function align_and_rescale(points::Vector{Point3f})
    points_mat = CuMatrix(vector_to_matrix(points))
    (points_normalized_mat, params) = align_and_rescale(points_mat)
    points_normalized = matrix_to_vector(Matrix(points_normalized_mat))
    return (points_normalized, params)
end

function align_and_rescale(points::CuMatrix{Float32})
    @assert size(points, 1) == 3 "Expected 3×N matrix with columns as points"
    n = size(points, 2)
    # step 1: compute covariance matrix for PCA (on GPU, then transfer small 3×3)
    cov_mat = (points * points') ./ n
    cov_cpu = Matrix(cov_mat)  # 3×3, cheap transfer

    # step 2: perform SVD for principal axes alignment
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

    # step 3: invert rotation via transpose to align principal axes
    rot_inv = CuMatrix(rotation')
    points = rot_inv * points

    # step 4: scale to unit sphere
    distances² = sum(abs2, points; dims=1)
    radius_max = √(maximum(distances²))
    is_zero = iszero(radius_max)
    scale = inv(radius_max + is_zero)
    points = scale .* points

    params = (; scale=radius_max, rotation)
    return (points, params)
end

function vector_to_matrix(points::Vector{Point3{T}}) where {T<:Real}
    matrix = reinterpret(reshape, T, points)
    return matrix
end

function matrix_to_vector(matrix::Matrix{T}) where {T<:Real}
    @assert size(matrix, 1) == 3 "Expected 3×N matrix with columns as points"
    vector = Point3{T}.(eachcol(matrix))
    return vector::Vector{Point3{T}}
end
