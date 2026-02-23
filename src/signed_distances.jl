function construct_mesh(
    sdf::Array{Float32,3}, method::M=MarchingTetrahedra{Float32,Float32}()
) where {M<:Union{MarchingCubes{Float32},MarchingTetrahedra{Float32,Float32}}}
    (vertices_t, faces_t) = isosurface(sdf, method)
    vertices = Point3f.(vertices_t)
    fcs = GLTriangleFace.(faces_t)
    mesh = Mesh(vertices, fcs)
    return mesh::Mesh{3,Float32,GLTriangleFace}
end

function visualize(mesh::Mesh{3,Float32})
    screen = Screen()
    figure = Figure()
    # LScene for full 3D camera control
    lscene = LScene(figure[1, 1])
    mesh!(lscene, mesh; color=:lightblue, shading=true)
    display(screen, figure)
    return figure
end

##################################   Feature Codes & Utilities   ##################################

# feature codes double as tuple indices into TriangleNormals.normals,
# enabling O(1) branchless pseudonormal lookup: normals[feat].
const FEAT_V1 = UInt8(1)     # vertex 1 (a)
const FEAT_V2 = UInt8(2)     # vertex 2 (b)
const FEAT_V3 = UInt8(3)     # vertex 3 (c)
const FEAT_E12 = UInt8(4)    # edge AB  (v1–v2)
const FEAT_E23 = UInt8(5)    # edge BC  (v2–v3)
const FEAT_E31 = UInt8(6)    # edge CA  (v3–v1)
const FEAT_FACE = UInt8(7)   # face interior

@inline function norm²(point::Point3{T}) where {T<:AbstractFloat}
    n² = point ⋅ point
    return n²::T
end

@inline function normalize(point::Point3{T}) where {T<:AbstractFloat}
    ε = nextfloat(zero(T))
    n² = norm²(point)
    c = ifelse(n² > ε, inv(√(n²)), zero(T))
    point_n = c * point
    return point_n::Point3{T}
end

# stable angle via atan(‖ × ‖, ⋅ ) — avoids division, clamp, and acos instability near 0°/180°
@inline function angle_between(u::Point3{T}, v::Point3{T}) where {T<:AbstractFloat}
    x = u ⋅ v
    y = √(norm²(u × v))
    α = atan(y, x)
    return α::T
end

#######################################   Data Structures   #######################################

# packed triangle vertices (contiguous by BVH leaf order for cache locality)
struct TriangleGeometry{T<:AbstractFloat}
    a::Point3{T}
    ab::Point3{T}
    ac::Point3{T}
end

# all 7 pseudonormals packed per-triangle.
# tuple indices match feature codes for O(1) lookup: normals[feat]
#   [1]=v1  [2]=v2  [3]=v3  [4]=e12  [5]=e23  [6]=e31  [7]=face
struct TriangleNormals{T<:AbstractFloat}
    normals::NTuple{7,Point3{T}}
end


# Fast generalized winding number (Barill et al. 2018) precomputation per BVH node.
# We store a 0th-order (dipole) far-field expansion:
#   w̃(q) = (n_sum ⋅ (c - q)) / (4π‖c - q‖³)
# where n_sum = Σ_t area_t * n̂_t (vector area) and c is the area-weighted centroid.
struct FastWindingData{T<:AbstractFloat}
    cm_x::Vector{T}
    cm_y::Vector{T}
    cm_z::Vector{T}
    # radius bound for the Barnes-Hut admissibility test: ‖q-c‖ > β r
    r::Vector{T}
    # n_sum = Σ area_t * n̂_t (vector area) for all triangles in node
    n_x::Vector{T}
    n_y::Vector{T}
    n_z::Vector{T}
    # Barnes–Hut accuracy parameter (β). Larger => more accurate, slower. Typical value: 2.
    beta::T
end


# AoS BVH node with overlapped integer fields for cache efficiency.
# for T=Float32 this is 6×4 + 2×4 = 32 bytes — exactly two nodes per 64-byte cache line.
# internal and leaf nodes overlap integer fields via sign-bit discriminant:
#   internal: index = left child,  child_or_size = right child  (both > 0)
#   leaf:     index = leaf_start,  child_or_size = -leaf_size   (< 0)
struct BVHNode{T<:AbstractFloat}
    lb_x::T
    lb_y::T
    lb_z::T
    ub_x::T
    ub_y::T
    ub_z::T
    index::Int32
    child_or_size::Int32
end

struct BoundingVolumeHierarchy{T<:AbstractFloat}
    nodes::Vector{BVHNode{T}}
    leaf_capacity::Int32
    num_nodes::Int32
end

# stack element carrying both node id and its already-computed AABB lower bound.
# this avoids recomputing aabb_dist² again when the node is popped.
struct NodeDist{T<:AbstractFloat}
    node_id::Int32
    dist²::T
end


# Per-chunk scratch space: BVH traversal stacks for distance and winding-number queries.
# (Allocated outside the hot point-loop, so there are no allocations per query.)
struct QueryStacks{T<:AbstractFloat}
    dist::Vector{NodeDist{T}}
    wind::Vector{Int32}
end


# Tg: geometry/distance type (Float32 recommended)
# Ts: pseudonormal/sign type (Float64 recommended)
struct SignedDistanceMesh{Tg<:AbstractFloat,Ts<:AbstractFloat}
    tri_geometries::Vector{TriangleGeometry{Tg}}   # packed by BVH leaf order
    tri_normals::Vector{TriangleNormals{Ts}}       # packed by BVH leaf order
    bvh::BoundingVolumeHierarchy{Tg}
    # face_to_packed[f] = packed triangle index for original face id f
    # (used to exploit your “source triangle” hints)
    face_to_packed::Vector{Int32}
    fwn::FastWindingData{Tg}
end

#######################################   BVH Construction   #######################################

# partial sort of indices by centroid along axis (build-time only)
function median_split_sort!(
    indices::Vector{Int32}, lo::Int, mid::Int, hi::Int, centroids::NTuple{3,Vector{Tg}}, axis::Int
) where {Tg<:AbstractFloat}
    sub_indices = @view indices[lo:hi]
    centroids_axis = centroids[axis]
    mid_relative = mid - lo + 1
    partialsort!(sub_indices, mid_relative; by=tri_idx -> centroids_axis[tri_idx])
    return nothing
end

mutable struct BVHBuilder{T}
    const nodes::Vector{BVHNode{T}}
    const leaf_capacity::Int32
    next_node::Int32
end

function build_node!(
    builder::BVHBuilder{T},
    tri_indices::Vector{Int32}, lo::Int, hi::Int, centroids::NTuple{3,Vector{T}},
    lb_x_t::Vector{T}, lb_y_t::Vector{T}, lb_z_t::Vector{T},
    ub_x_t::Vector{T}, ub_y_t::Vector{T}, ub_z_t::Vector{T},
) where {T}
    node_id = builder.next_node
    builder.next_node += 1

    # compute node bounds
    min_x = T(Inf)
    min_y = T(Inf)
    min_z = T(Inf)
    max_x = -T(Inf)
    max_y = -T(Inf)
    max_z = -T(Inf)
    @inbounds for i in lo:hi
        idx_face = tri_indices[i]
        min_x = min(min_x, lb_x_t[idx_face])
        min_y = min(min_y, lb_y_t[idx_face])
        min_z = min(min_z, lb_z_t[idx_face])
        max_x = max(max_x, ub_x_t[idx_face])
        max_y = max(max_y, ub_y_t[idx_face])
        max_z = max(max_z, ub_z_t[idx_face])
    end

    count = hi - lo + 1
    if count <= builder.leaf_capacity
        @inbounds builder.nodes[node_id] = BVHNode{T}(
            min_x, min_y, min_z, max_x, max_y, max_z, Int32(lo), -Int32(count)
        )
        return node_id
    end

    # split axis = longest centroid extent
    centroid_min_x = T(Inf)
    centroid_min_y = T(Inf)
    centroid_min_z = T(Inf)
    centroid_max_x = -T(Inf)
    centroid_max_y = -T(Inf)
    centroid_max_z = -T(Inf)
    @inbounds for i in lo:hi
        t = tri_indices[i]
        centroid_min_x = min(centroid_min_x, centroids[1][t])
        centroid_min_y = min(centroid_min_y, centroids[2][t])
        centroid_min_z = min(centroid_min_z, centroids[3][t])
        centroid_max_x = max(centroid_max_x, centroids[1][t])
        centroid_max_y = max(centroid_max_y, centroids[2][t])
        centroid_max_z = max(centroid_max_z, centroids[3][t])
    end
    spread_x = centroid_max_x - centroid_min_x
    spread_y = centroid_max_y - centroid_min_y
    spread_z = centroid_max_z - centroid_min_z
    (spread_max, axis) = findmax((spread_x, spread_y, spread_z))

    mid = (lo + hi) >>> 1   # (lo + hi) ÷ 2
    # median split via partial sort (skip if all centroids identical along all axes)
    (spread_max > 0) && median_split_sort!(tri_indices, lo, mid, hi, centroids, axis)

    node_left = build_node!(
        builder, tri_indices, lo, mid, centroids, lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t
    )
    node_right = build_node!(
        builder, tri_indices, mid + 1, hi, centroids, lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t
    )
    @inbounds builder.nodes[node_id] = BVHNode{T}(
        min_x, min_y, min_z, max_x, max_y, max_z, node_left, node_right
    )
    return node_id
end

function build_bvh(
    centroids::NTuple{3,Vector{Tg}},
    lb_x_t::Vector{Tg}, lb_y_t::Vector{Tg}, lb_z_t::Vector{Tg},
    ub_x_t::Vector{Tg}, ub_y_t::Vector{Tg}, ub_z_t::Vector{Tg};
    leaf_capacity::Int=8
) where {Tg}
    num_faces = length(first(centroids))
    tri_indices = Int32.(1:num_faces)

    max_nodes = 2 * num_faces
    builder = BVHBuilder{Tg}(Vector{BVHNode{Tg}}(undef, max_nodes), Int32(leaf_capacity), Int32(1))
    build_node!(
        builder, tri_indices, 1, num_faces, centroids, lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t
    )
    num_nodes = builder.next_node - 1
    bvh = BoundingVolumeHierarchy{Tg}(builder.nodes[1:num_nodes], builder.leaf_capacity, num_nodes)
    return (bvh, tri_indices)  # return triangle order for packing
end

######################################   Mesh Preprocessing   ######################################

@inline function edge_key(a::Int32, b::Int32)
    (lo, hi) = minmax(a, b)
    key = (UInt64(lo) << 32) | UInt64(hi)
    return key
end

"""
    preprocess_mesh(mesh::Mesh{3,Float32,GLTriangleFace}, [sign_type=Float64]; leaf_capacity=8, winding_beta=2.0)

Build the preprocessing data needed for fast signed-distance queries:

- A BVH for **unsigned** distance queries (closest-point distance).
- A fast **generalized winding number** hierarchy for a robust inside/outside **sign**
  (Barnes–Hut-style acceleration, Barill et al. 2018).

Arguments:
- `mesh`: closed, consistently-oriented triangle mesh.
- `sign_type`: floating-point type used during preprocessing of face normals/pseudonormals.
  (The final sign returned by `compute_signed_distance!` is determined by winding numbers.)
- `leaf_capacity`: BVH leaf size (distance & winding traversal share the same BVH).
- `winding_beta` (β): Barnes–Hut admissibility parameter. Larger ⇒ more accurate winding numbers,
  but more tree traversal work. A common default is `2.0`.

Returns a `SignedDistanceMesh{Tg,Ts}` ready for `compute_signed_distance!` calls.
"""
function preprocess_mesh(
    mesh::Mesh{3,Float32,GLTriangleFace}, sign_type::Type{Ts}=Float64; leaf_capacity::Int=8, winding_beta::Real=2.0
) where {Ts<:AbstractFloat}
    vertices = GeometryBasics.coordinates(mesh)
    tri_faces = GeometryBasics.faces(mesh)
    faces = NTuple{3,Int32}.(tri_faces)
    return preprocess_mesh(vertices, faces, sign_type; leaf_capacity, winding_beta)
end

function preprocess_mesh(
    vertices::Vector{Point3f},
    faces::Vector{NTuple{3,Int32}},
    sign_type::Type{Ts}=Float64;
    leaf_capacity::Int=8,
    winding_beta::Real=2.0
) where {Ts<:AbstractFloat}
    Tg = Float32
    num_vertices = length(vertices)
    num_faces = length(faces)

    # face unit normals computed in Ts (Float64 recommended)
    normals = Vector{Point3{Ts}}(undef, num_faces)
    @inbounds for idx_face in eachindex(faces)
        (idx_v1, idx_v2, idx_v3) = faces[idx_face]
        v1 = Point3{Ts}(vertices[idx_v1])
        v2 = Point3{Ts}(vertices[idx_v2])
        v3 = Point3{Ts}(vertices[idx_v3])
        normal = (v2 - v1) × (v3 - v1)
        normals[idx_face] = normalize(normal)
    end

    # face_adjacency[edge, face] = index of the face sharing local `edge` of `face` (0 if boundary)
    face_adjacency = zeros(Int32, 3, num_faces)
    neighbors = Dict{UInt64,Tuple{Int32,Int32}}()
    sizehint!(neighbors, 3 * num_faces)
    @inbounds for idx_face in eachindex(faces)
        (idx_v1, idx_v2, idx_v3) = faces[idx_face]
        for (edge, vertex_a, vertex_b) in ((Int32(1), idx_v1, idx_v2), (Int32(2), idx_v2, idx_v3), (Int32(3), idx_v3, idx_v1))
            key = edge_key(vertex_a, vertex_b)
            pair = get(neighbors, key, nothing)
            if pair === nothing
                neighbors[key] = (Int32(idx_face), edge)
            else
                (face_adjacent, edge_common) = pair
                face_adjacency[edge, idx_face] = face_adjacent
                face_adjacency[edge_common, face_adjacent] = Int32(idx_face)
                delete!(neighbors, key)
            end
        end
    end
    isempty(neighbors) || throw(ArgumentError("Mesh is not watertight: $(length(neighbors)) boundary edges"))

    # edge pseudonormals: sum of adjacent unit face normals (unnormalized as only sign matters)
    pns_edge = Matrix{Point3{Ts}}(undef, 3, num_faces)
    for idx_face₁ in eachindex(normals)
        normal₁ = normals[idx_face₁]
        for edge in 1:3
            idx_face₂ = face_adjacency[edge, idx_face₁]
            normal₂ = normals[idx_face₂]
            pns_edge[edge, idx_face₁] = normal₁ + normal₂
        end
    end

    # vertex pseudonormals (angle-weighted, unnormalized) in Ts
    pns_vertex = zeros(Point3{Ts}, num_vertices)
    @inbounds for idx_face in eachindex(faces)
        face = faces[idx_face]
        (idx_v1, idx_v2, idx_v3) = face
        v1 = Point3{Ts}(vertices[idx_v1])
        v2 = Point3{Ts}(vertices[idx_v2])
        v3 = Point3{Ts}(vertices[idx_v3])
        normal = normals[idx_face]

        α1 = angle_between(v2 - v1, v3 - v1)
        α2 = angle_between(v3 - v2, v1 - v2)
        α3 = angle_between(v1 - v3, v2 - v3)

        # intentionally unnormalized: only sign(rvec ⋅ pn) matters for Bærentzen signing
        pns_vertex[idx_v1] += α1 * normal
        pns_vertex[idx_v2] += α2 * normal
        pns_vertex[idx_v3] += α3 * normal
    end

    # build BVH (in Tg)
    lb_x_t = Vector{Tg}(undef, num_faces)
    lb_y_t = Vector{Tg}(undef, num_faces)
    lb_z_t = Vector{Tg}(undef, num_faces)
    ub_x_t = Vector{Tg}(undef, num_faces)
    ub_y_t = Vector{Tg}(undef, num_faces)
    ub_z_t = Vector{Tg}(undef, num_faces)
    centroids_x = Vector{Tg}(undef, num_faces)
    centroids_y = Vector{Tg}(undef, num_faces)
    centroids_z = Vector{Tg}(undef, num_faces)

    @inbounds for idx_face in eachindex(faces)
        face = faces[idx_face]
        (idx_v1, idx_v2, idx_v3) = face
        v1 = vertices[idx_v1]
        v2 = vertices[idx_v2]
        v3 = vertices[idx_v3]
        (x1, y1, z1) = v1
        (x2, y2, z2) = v2
        (x3, y3, z3) = v3

        lb_x_t[idx_face] = min(x1, x2, x3)
        lb_y_t[idx_face] = min(y1, y2, y3)
        lb_z_t[idx_face] = min(z1, z2, z3)
        ub_x_t[idx_face] = max(x1, x2, x3)
        ub_y_t[idx_face] = max(y1, y2, y3)
        ub_z_t[idx_face] = max(z1, z2, z3)

        centroid = (v1 + v2 + v3) / 3
        centroids_x[idx_face] = centroid[1]
        centroids_y[idx_face] = centroid[2]
        centroids_z[idx_face] = centroid[3]
    end
    centroids = (centroids_x, centroids_y, centroids_z)
    (bvh, tri_order) = build_bvh(
        centroids, lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t; leaf_capacity
    )

    # pack triangle geometry & normals contiguously by BVH leaf order
    tri_geometries = Vector{TriangleGeometry{Tg}}(undef, num_faces)
    tri_normals = Vector{TriangleNormals{Ts}}(undef, num_faces)

    # map original face index → packed index (for triangle-hint acceleration)
    face_to_packed = Vector{Int32}(undef, num_faces)

    @inbounds for j in eachindex(faces)
        idx_face = tri_order[j]   # original face index
        face_to_packed[idx_face] = Int32(j)

        (idx_v1, idx_v2, idx_v3) = faces[idx_face]
        v1 = vertices[idx_v1]
        v2 = vertices[idx_v2]
        v3 = vertices[idx_v3]
        tri_geometries[j] = TriangleGeometry{Tg}(v1, v2 - v1, v3 - v1)

        # tuple indices match feature codes for direct normals[feat] lookup
        tri_normals[j] = TriangleNormals{Ts}((
            pns_vertex[idx_v1],      # [1] = FEAT_V1
            pns_vertex[idx_v2],      # [2] = FEAT_V2
            pns_vertex[idx_v3],      # [3] = FEAT_V3
            pns_edge[1, idx_face],   # [4] = FEAT_E12
            pns_edge[2, idx_face],   # [5] = FEAT_E23
            pns_edge[3, idx_face],   # [6] = FEAT_E31
            normals[idx_face],       # [7] = FEAT_FACE
        ))
    end


    # Precompute fast generalized winding-number (Barill et al. 2018) data for robust sign queries.
    # This adds a Barnes–Hut BVH traversal to the signed distance query, but is O(log m) per point
    # and typically robust around sharp features where pseudonormal methods can fail.
    fwn = precompute_fast_winding_data(bvh, tri_geometries; beta=Tg(winding_beta))

    return SignedDistanceMesh{Tg,Ts}(tri_geometries, tri_normals, bvh, face_to_packed, fwn)
end


#################################   Fast Winding Number Precompute   #################################

# Precompute node-wise data for the fast generalized winding number of a triangle soup.
# We use a Barnes–Hut style far-field approximation (0th order / "single dipole" per node)
# as described in Barill et al. 2018.
function precompute_fast_winding_data(
    bvh::BoundingVolumeHierarchy{Tg},
    tri_geometries::Vector{TriangleGeometry{Tg}};
    beta::Tg=Tg(2)
) where {Tg<:AbstractFloat}

    num_nodes = Int(bvh.num_nodes)

    # Stored per node (geometry type Tg for compactness/cache-efficiency)
    cm_x = Vector{Tg}(undef, num_nodes)
    cm_y = Vector{Tg}(undef, num_nodes)
    cm_z = Vector{Tg}(undef, num_nodes)
    r = Vector{Tg}(undef, num_nodes)
    n_x = Vector{Tg}(undef, num_nodes)
    n_y = Vector{Tg}(undef, num_nodes)
    n_z = Vector{Tg}(undef, num_nodes)

    # Temporary aggregation buffers in Float64 for stable summation during preprocessing
    area_sum = Vector{Float64}(undef, num_nodes)
    cent_sum_x = Vector{Float64}(undef, num_nodes)
    cent_sum_y = Vector{Float64}(undef, num_nodes)
    cent_sum_z = Vector{Float64}(undef, num_nodes)
    n_sum_x = Vector{Float64}(undef, num_nodes)
    n_sum_y = Vector{Float64}(undef, num_nodes)
    n_sum_z = Vector{Float64}(undef, num_nodes)

    @inbounds for node_id in num_nodes:-1:1
        node = bvh.nodes[node_id]
        child_or_size = node.child_or_size

        if child_or_size < 0
            leaf_start = Int(node.index)
            leaf_size = Int(-child_or_size)
            leaf_end = leaf_start + leaf_size - 1

            a_sum = 0.0
            c_x_sum = 0.0
            c_y_sum = 0.0
            c_z_sum = 0.0
            nn_x_sum = 0.0
            nn_y_sum = 0.0
            nn_z_sum = 0.0

            for idx in leaf_start:leaf_end
                tri = tri_geometries[idx]
                ax = Float64(tri.a[1])
                ay = Float64(tri.a[2])
                az = Float64(tri.a[3])

                abx = Float64(tri.ab[1])
                aby = Float64(tri.ab[2])
                abz = Float64(tri.ab[3])

                acx = Float64(tri.ac[1])
                acy = Float64(tri.ac[2])
                acz = Float64(tri.ac[3])

                # cross = ab × ac (oriented)
                cx = aby * acz - abz * acy
                cy = abz * acx - abx * acz
                cz = abx * acy - aby * acx

                # vector area = 0.5 * cross  (= area * n̂)
                vax = 0.5 * cx
                vay = 0.5 * cy
                vaz = 0.5 * cz

                # scalar area = 0.5 * ‖cross‖
                area = 0.5 * sqrt(cx * cx + cy * cy + cz * cz)

                # centroid = a + (ab + ac)/3  (more flops-friendly than (a+b+c)/3)
                if area > 0.0
                    centx = ax + (abx + acx) / 3.0
                    centy = ay + (aby + acy) / 3.0
                    centz = az + (abz + acz) / 3.0

                    a_sum += area
                    c_x_sum += area * centx
                    c_y_sum += area * centy
                    c_z_sum += area * centz
                end

                nn_x_sum += vax
                nn_y_sum += vay
                nn_z_sum += vaz
            end

            area_sum[node_id] = a_sum
            cent_sum_x[node_id] = c_x_sum
            cent_sum_y[node_id] = c_y_sum
            cent_sum_z[node_id] = c_z_sum
            n_sum_x[node_id] = nn_x_sum
            n_sum_y[node_id] = nn_y_sum
            n_sum_z[node_id] = nn_z_sum
        else
            left = Int(node.index)
            right = Int(child_or_size)

            a_sum = area_sum[left] + area_sum[right]
            area_sum[node_id] = a_sum
            cent_sum_x[node_id] = cent_sum_x[left] + cent_sum_x[right]
            cent_sum_y[node_id] = cent_sum_y[left] + cent_sum_y[right]
            cent_sum_z[node_id] = cent_sum_z[left] + cent_sum_z[right]
            n_sum_x[node_id] = n_sum_x[left] + n_sum_x[right]
            n_sum_y[node_id] = n_sum_y[left] + n_sum_y[right]
            n_sum_z[node_id] = n_sum_z[left] + n_sum_z[right]
        end

        # far-field expansion center = area-weighted centroid of elements in the node
        a_sum = area_sum[node_id]
        if a_sum > 0.0
            c_x = cent_sum_x[node_id] / a_sum
            c_y = cent_sum_y[node_id] / a_sum
            c_z = cent_sum_z[node_id] / a_sum
        else
            # degenerate cluster: fall back to AABB center to avoid NaNs
            c_x = 0.5 * (Float64(node.lb_x) + Float64(node.ub_x))
            c_y = 0.5 * (Float64(node.lb_y) + Float64(node.ub_y))
            c_z = 0.5 * (Float64(node.lb_z) + Float64(node.ub_z))
        end

        cm_x[node_id] = Tg(c_x)
        cm_y[node_id] = Tg(c_y)
        cm_z[node_id] = Tg(c_z)

        n_x[node_id] = Tg(n_sum_x[node_id])
        n_y[node_id] = Tg(n_sum_y[node_id])
        n_z[node_id] = Tg(n_sum_z[node_id])

        # radius bound: farthest AABB corner from (c_x, c_y, c_z)
        dx = max(abs(c_x - Float64(node.lb_x)), abs(c_x - Float64(node.ub_x)))
        dy = max(abs(c_y - Float64(node.lb_y)), abs(c_y - Float64(node.ub_y)))
        dz = max(abs(c_z - Float64(node.lb_z)), abs(c_z - Float64(node.ub_z)))
        r[node_id] = Tg(sqrt(dx * dx + dy * dy + dz * dz))
    end

    return FastWindingData{Tg}(cm_x, cm_y, cm_z, r, n_x, n_y, n_z, beta)
end

function calculate_tree_height(num_faces::Integer, leaf_capacity::Integer)
    num_leaves = max(ceil(num_faces / leaf_capacity), 1.0)
    tree_height = ceil(Int, log2(num_leaves))
    return tree_height::Int
end

# allocate one traversal stack per chunk
# each stack is tiny (~256 bytes for 100k triangles), so per-call allocation is negligible
function allocate_stacks(sdm::SignedDistanceMesh{Tg,Ts}, num_points::Int) where {Tg,Ts}
    num_faces = length(sdm.tri_geometries)
    leaf_capacity = sdm.bvh.leaf_capacity
    tree_height = calculate_tree_height(num_faces, leaf_capacity)
    stack_capacity = 2 * tree_height + 4
    n_threads = Threads.nthreads()
    # enforce a minimum chunk size to prevent false sharing and guarantee cache locality.
    min_chunk_size = 512
    num_chunks_max = max(1, num_points ÷ min_chunk_size)

    if num_chunks_max < n_threads
        # array is too small to use all threads optimally, limit the chunks
        num_chunks = num_chunks_max
    else # array is large enough to target ideal 8x oversubscription
        factor_max = 8
        factor = min(factor_max, num_chunks_max ÷ n_threads)
        # set num_chunks to an exact multiple of n_threads for even load distribution over all cores
        num_chunks = n_threads * factor
    end


    stacks = [QueryStacks{Tg}(
        Vector{NodeDist{Tg}}(undef, stack_capacity),   # distance BVH stack
        Vector{Int32}(undef, stack_capacity)           # winding BVH stack
    ) for _ in 1:num_chunks]
    return stacks
end

##############################   High-Performance Hot Loop Routines   ##############################

# AABB squared distance: single node load + branchless clamp
@inline function aabb_dist²(
    point::Point3{Tg}, bvh::BoundingVolumeHierarchy{Tg}, node_id::Int32
) where {Tg}
    @fastmath begin
        zer = zero(Tg)
        (point_x, point_y, point_z) = point
        @inbounds node = bvh.nodes[node_id]  # 32 bytes for Float32 — two nodes per cache line
        Δx = max(node.lb_x - point_x, point_x - node.ub_x, zer)
        Δy = max(node.lb_y - point_y, point_y - node.ub_y, zer)
        Δz = max(node.lb_z - point_z, point_z - node.ub_z, zer)
        dist² = muladd(Δx, Δx, muladd(Δy, Δy, Δz * Δz))
    end
    return dist²
end

# exact closest-point-on-triangle (Ericson-style) but returns Δ = p - closest_point.
# This avoids computing and storing closest point, and makes the sign test use Δ directly.
@inline function closest_diff_triangle(p::Point3{Tg}, triangle::TriangleGeometry{Tg}) where {Tg}
    a = triangle.a
    ab = triangle.ab
    ac = triangle.ac

    ap = p - a
    d1 = ab ⋅ ap
    d2 = ac ⋅ ap
    if (d1 <= 0) && (d2 <= 0)
        return (norm²(ap), ap, FEAT_V1)
    end

    bp = ap - ab
    d3 = ab ⋅ bp
    d4 = ac ⋅ bp
    if (d3 >= 0) && (d4 <= d3)
        return (norm²(bp), bp, FEAT_V2)
    end

    ε = nextfloat(zero(Tg))
    vc = d1 * d4 - d3 * d2
    d13 = d1 - d3
    if (vc <= 0) && (d1 >= 0) && (d3 <= 0) && (d13 > ε)
        v = d1 / d13   # bary: (1-v, v, 0)
        Δ = ap - v * ab      # p - (a + v*ab)
        return (norm²(Δ), Δ, FEAT_E12)
    end

    cp = ap - ac
    d5 = ab ⋅ cp
    d6 = ac ⋅ cp
    if (d6 >= 0) && (d5 <= d6)
        return (norm²(cp), cp, FEAT_V3)
    end

    vb = d5 * d2 - d1 * d6
    d26 = d2 - d6
    if (vb <= 0) && (d2 >= 0) && (d6 <= 0) && (d26 > ε)
        w = d2 / d26   # bary: (1-w, 0, w)
        Δ = ap - w * ac      # p - (a + w*ac)
        return (norm²(Δ), Δ, FEAT_E31)
    end

    va = d3 * d6 - d5 * d4
    d43 = d4 - d3
    d56 = d5 - d6
    denom_sum = d43 + d56
    if (va <= 0) && (d43 >= 0) && (d56 >= 0) && (denom_sum > ε)
        w = d43 / denom_sum  # bary: (0, 1-w, w)
        bc = ac - ab
        Δ = bp - w * bc
        return (norm²(Δ), Δ, FEAT_E23)
    end

    denom_sum = va + vb + vc
    # fallback for degenerate triangles to avoid NaN
    (denom_sum > ε) || return (norm²(ap), ap, FEAT_V1)
    denom = inv(denom_sum)
    v = vb * denom
    w = vc * denom
    Δ = ap - v * ab - w * ac
    return (norm²(Δ), Δ, FEAT_FACE)
end


#################################   Fast Winding Number Query   #################################

# 1/(4π) in Float64 for winding-number normalization.
const INV4PI64 = 1.0 / (4.0 * π)

# Signed solid angle of a single oriented triangle as seen from q, normalized by 4π.
# Formula from Van Oosterom & Strackee (1983), widely used in generalized winding number code.
@inline function solid_angle_over_4pi(q::Point3{Tg}, tri::TriangleGeometry{Tg}) where {Tg<:AbstractFloat}
    # triangle vertices: a, b = a+ab, c = a+ac
    ax0 = Float64(tri.a[1])
    ay0 = Float64(tri.a[2])
    az0 = Float64(tri.a[3])
    bx0 = ax0 + Float64(tri.ab[1])
    by0 = ay0 + Float64(tri.ab[2])
    bz0 = az0 + Float64(tri.ab[3])
    cx0 = ax0 + Float64(tri.ac[1])
    cy0 = ay0 + Float64(tri.ac[2])
    cz0 = az0 + Float64(tri.ac[3])

    qx = Float64(q[1])
    qy = Float64(q[2])
    qz = Float64(q[3])

    # vectors from q to vertices
    ax = ax0 - qx
    ay = ay0 - qy
    az = az0 - qz
    bx = bx0 - qx
    by = by0 - qy
    bz = bz0 - qz
    cx = cx0 - qx
    cy = cy0 - qy
    cz = cz0 - qz

    # lengths
    la = sqrt(ax * ax + ay * ay + az * az)
    lb = sqrt(bx * bx + by * by + bz * bz)
    lc = sqrt(cx * cx + cy * cy + cz * cz)

    # scalar triple product: a · (b × c)
    tpx = by * cz - bz * cy
    tpy = bz * cx - bx * cz
    tpz = bx * cy - by * cx
    det = ax * tpx + ay * tpy + az * tpz

    # denominator
    ab = ax * bx + ay * by + az * bz
    ac = ax * cx + ay * cy + az * cz
    bc = bx * cx + by * cy + bz * cz
    denom = la * lb * lc + ab * lc + ac * lb + bc * la

    Ω = 2.0 * atan(det, denom)
    return Ω * INV4PI64
end

# Fast generalized winding number evaluation at a point using the precomputed BVH data.
# Returns a scalar winding number, typically ~1 inside and ~0 outside (for oriented closed meshes).
@inline function winding_number_point_kernel(
    sdm::SignedDistanceMesh{Tg,Ts},
    point::Point3{Tg},
    stack::Vector{Int32}
)::Float64 where {Tg<:AbstractFloat,Ts<:AbstractFloat}

    fwn = sdm.fwn
    bvh = sdm.bvh
    tri_geometries = sdm.tri_geometries

    β = Float64(fwn.beta)
    β2 = β * β

    qx = Float64(point[1])
    qy = Float64(point[2])
    qz = Float64(point[3])

    wn = 0.0
    stack_top = 1
    @inbounds stack[1] = Int32(1)

    @inbounds while stack_top > 0
        node_id_i32 = stack[stack_top]
        stack_top -= 1
        node_id = Int(node_id_i32)

        # Load precomputed node data
        cmx = Float64(fwn.cm_x[node_id])
        cmy = Float64(fwn.cm_y[node_id])
        cmz = Float64(fwn.cm_z[node_id])

        rx = cmx - qx
        ry = cmy - qy
        rz = cmz - qz

        dist2 = rx * rx + ry * ry + rz * rz
        r_node = Float64(fwn.r[node_id])

        if dist2 > β2 * (r_node * r_node)
            # Far field: 0th order (dipole) approximation
            nx = Float64(fwn.n_x[node_id])
            ny = Float64(fwn.n_y[node_id])
            nz = Float64(fwn.n_z[node_id])

            dot = rx * nx + ry * ny + rz * nz
            inv_denom = inv(dist2 * sqrt(dist2))  # 1/‖r‖^3
            wn += dot * INV4PI64 * inv_denom
        else
            node = bvh.nodes[node_id]
            child_or_size = node.child_or_size

            if child_or_size < 0
                # Near field: exact sum of triangle solid angles for this leaf
                leaf_start = Int(node.index)
                leaf_size = Int(-child_or_size)
                leaf_end = leaf_start + leaf_size - 1
                for tri_id in leaf_start:leaf_end
                    wn += solid_angle_over_4pi(point, tri_geometries[tri_id])
                end
            else
                # Recurse into children
                stack_top += 1
                stack[stack_top] = node.index
                stack_top += 1
                stack[stack_top] = child_or_size
            end
        end
    end

    return wn
end

######################################   Single-Point Query   ######################################

function signed_distance_point(
    sdm::SignedDistanceMesh{Tg,Ts}, point::Point3{Tg}, hint_face::Int32, stacks::QueryStacks{Tg}
) where {Tg<:AbstractFloat,Ts<:AbstractFloat}
    # tighten initial bound using the provided triangle hint (packed index).
    # this is especially effective for near-surface samples.
    tri_best = hint_face
    @inbounds triangle = sdm.tri_geometries[tri_best]
    (dist²_best, Δ_best, feat_best) = closest_diff_triangle(point, triangle)
    (dist²_best <= 0) && return zero(Tg)

    signed_distance = signed_distance_point_kernel(sdm, point, dist²_best, Δ_best, feat_best, tri_best, stacks)
    return signed_distance::Tg
end

function signed_distance_point(
    sdm::SignedDistanceMesh{Tg,Ts}, point::Point3{Tg}, stacks::QueryStacks{Tg}
) where {Tg<:AbstractFloat,Ts<:AbstractFloat}
    dist²_best = Tg(Inf)
    Δ_best = zero(Point3{Tg})
    feat_best = UInt8(0)
    tri_best = Int32(0)
    signed_distance = signed_distance_point_kernel(sdm, point, dist²_best, Δ_best, feat_best, tri_best, stacks)
    return signed_distance::Tg
end

function signed_distance_point_kernel(
    sdm::SignedDistanceMesh{Tg,Ts},
    point::Point3{Tg},
    dist²_best::Tg,
    Δ_best::Point3{Tg},
    feat_best::UInt8,
    tri_best::Int32,
    stacks::QueryStacks{Tg}
) where {Tg<:AbstractFloat,Ts<:AbstractFloat}
    bvh = sdm.bvh
    tri_geometries = sdm.tri_geometries
    stack = stacks.dist
    wind_stack = stacks.wind
    stack_top = 1
    dist²_root = aabb_dist²(point, bvh, Int32(1))
    @inbounds stack[1] = NodeDist{Tg}(Int32(1), dist²_root)

    @inbounds while stack_top > 0
        node_dist = stack[stack_top]
        stack_top -= 1

        (node_dist.dist² > dist²_best) && continue

        node = bvh.nodes[node_dist.node_id]
        child_or_size = node.child_or_size

        if child_or_size > 0 # internal node
            # compute child AABB bounds once, push with stored distances
            child_l = node.index
            child_r = child_or_size

            dist²_l = aabb_dist²(point, bvh, child_l)
            dist²_r = aabb_dist²(point, bvh, child_r)

            # sort so near child is pushed last (popped first)
            if dist²_l > dist²_r
                (child_l, child_r) = (child_r, child_l)
                (dist²_l, dist²_r) = (dist²_r, dist²_l)
            end

            # push far, then near
            if dist²_r <= dist²_best
                stack_top += 1
                stack[stack_top] = NodeDist{Tg}(child_r, dist²_r)
            end
            if dist²_l <= dist²_best
                stack_top += 1
                stack[stack_top] = NodeDist{Tg}(child_l, dist²_l)
            end
        else # leaf: test triangles (data is contiguous in packed arrays)
            leaf_start = node.index
            leaf_size = -child_or_size
            leaf_end = leaf_start + leaf_size - Int32(1)
            for idx in leaf_start:leaf_end
                (dist², Δ, feat) = closest_diff_triangle(point, tri_geometries[idx])
                if dist² < dist²_best
                    dist²_best = dist²
                    Δ_best = Δ
                    feat_best = feat
                    tri_best = idx
                end
            end
        end
    end

    iszero(tri_best) && error("No triangle found for point $point")
    dist = √(dist²_best)
    iszero(dist) && return zero(Tg)


    # Robust inside/outside sign via (fast) generalized winding number.
    # For a consistently oriented closed mesh: wn ≈ 1 inside, wn ≈ 0 outside.
    wn = winding_number_point_kernel(sdm, point, wind_stack)

    uno = one(Tg)
    sgn = ifelse(abs(wn) > 0.5, -uno, uno)  # inside => negative signed distance
    signed_distance = sgn * dist
    return signed_distance::Tg
end

##########################################   Public API   ##########################################

"""
    compute_signed_distance!(out, sdm, points_mat, [hint_faces])

In-place batch signed distance query. Writes results into `out`.
- `out`:            length-n vector to store the output signed distances.
- `sdm`:            a [`SignedDistanceMesh`] built once via `preprocess_mesh`.
- `points_mat`:     `3 × n` matrix of query points (Float32 recommended).
- `hint_faces`:     length-n vector of *original* face indices (1-based, matching the input `faces`)
                    for each point. This uses a single exact triangle check to tighten the upper
                    bound before BVH traversal, which can substantially speed up near-surface queries.

Positive = outside, negative = inside.

Notes:
- The unsigned distance is computed in the geometry type `Tg` (Float32).
- The sign is determined by the (fast) generalized winding number (wn>0.5 ⇒ inside).
  The winding number accumulation is done in Float64.
"""
function compute_signed_distance!(
    out::AbstractVector{Tg},
    sdm::SignedDistanceMesh{Tg,Ts},
    points::StridedMatrix{Tg},
    hint_faces::Vector{Int32}
) where {Tg<:AbstractFloat,Ts<:AbstractFloat}
    num_points = size(points, 2)
    @assert length(out) == length(hint_faces) == num_points
    @assert size(points, 1) == 3 "points matrix must be 3×n"

    face_to_packed = sdm.face_to_packed
    stacks = allocate_stacks(sdm, num_points)
    # equipartition the work into chunks, so each chunk gets its own stack.
    # to ensure thread-safety, stacks are allocated per-call (not stored in the struct) so that
    # concurrent callers on the same sdm never share mutable scratch space.
    num_chunks = length(stacks)
    chunk_size = cld(num_points, num_chunks)
    Threads.@threads :dynamic for idx_chunk in 1:num_chunks
        stack = stacks[idx_chunk]
        chunk_start = (idx_chunk - 1) * chunk_size + 1
        chunk_end = min(idx_chunk * chunk_size, num_points)
        for idx in chunk_start:chunk_end
            idx_face = hint_faces[idx]
            @inbounds idx_face_packed = face_to_packed[idx_face]
            @inbounds point = Point3{Tg}(points[1, idx], points[2, idx], points[3, idx])
            @inbounds out[idx] = signed_distance_point(sdm, point, idx_face_packed, stack)
        end
    end
    return out
end

function compute_signed_distance!(
    out::AbstractVector{Tg}, sdm::SignedDistanceMesh{Tg,Ts}, points::StridedMatrix{Tg}
) where {Tg<:AbstractFloat,Ts<:AbstractFloat}
    num_points = size(points, 2)
    @assert length(out) == num_points
    @assert size(points, 1) == 3 "points matrix must be 3×n"

    stacks = allocate_stacks(sdm, num_points)
    # equipartition the work into chunks, so each chunk gets its own stack.
    # to ensure thread-safety, stacks are allocated per-call (not stored in the struct) so that
    # concurrent callers on the same sdm never share mutable scratch space.
    num_chunks = length(stacks)
    chunk_size = cld(num_points, num_chunks)
    Threads.@threads :dynamic for idx_chunk in 1:num_chunks
        stack = stacks[idx_chunk]
        chunk_start = (idx_chunk - 1) * chunk_size + 1
        chunk_end = min(idx_chunk * chunk_size, num_points)
        for idx in chunk_start:chunk_end
            @inbounds point = Point3{Tg}(points[1, idx], points[2, idx], points[3, idx])
            @inbounds out[idx] = signed_distance_point(sdm, point, stack)
        end
    end
    return out
end


########################################   Optional Reusable Scratch   ########################################

"""
    compute_signed_distance!(out, sdm, points, stacks)
    compute_signed_distance!(out, sdm, points, hint_faces, stacks)

Same as the other `compute_signed_distance!` methods, but reuses a preallocated `stacks`
workspace (from `allocate_stacks`) to avoid allocations when calling this function many
times in a hot loop.
"""
# (docstring-only helper; the function is defined above)

function compute_signed_distance!(
    out::AbstractVector{Tg},
    sdm::SignedDistanceMesh{Tg,Ts},
    points::StridedMatrix{Tg},
    hint_faces::Vector{Int32},
    stacks::Vector{QueryStacks{Tg}}
) where {Tg<:AbstractFloat,Ts<:AbstractFloat}
    num_points = size(points, 2)
    @assert length(out) == length(hint_faces) == num_points
    @assert size(points, 1) == 3 "points matrix must be 3×n"
    @assert !isempty(stacks)

    face_to_packed = sdm.face_to_packed

    num_chunks = length(stacks)
    chunk_size = cld(num_points, num_chunks)
    Threads.@threads :dynamic for idx_chunk in 1:num_chunks
        stack = stacks[idx_chunk]
        chunk_start = (idx_chunk - 1) * chunk_size + 1
        chunk_end = min(idx_chunk * chunk_size, num_points)
        for idx in chunk_start:chunk_end
            idx_face = hint_faces[idx]
            @inbounds idx_face_packed = face_to_packed[idx_face]
            @inbounds point = Point3{Tg}(points[1, idx], points[2, idx], points[3, idx])
            @inbounds out[idx] = signed_distance_point(sdm, point, idx_face_packed, stack)
        end
    end
    return out
end

function compute_signed_distance!(
    out::AbstractVector{Tg},
    sdm::SignedDistanceMesh{Tg,Ts},
    points::StridedMatrix{Tg},
    stacks::Vector{QueryStacks{Tg}}
) where {Tg<:AbstractFloat,Ts<:AbstractFloat}
    num_points = size(points, 2)
    @assert length(out) == num_points
    @assert size(points, 1) == 3 "points matrix must be 3×n"
    @assert !isempty(stacks)

    num_chunks = length(stacks)
    chunk_size = cld(num_points, num_chunks)
    Threads.@threads :dynamic for idx_chunk in 1:num_chunks
        stack = stacks[idx_chunk]
        chunk_start = (idx_chunk - 1) * chunk_size + 1
        chunk_end = min(idx_chunk * chunk_size, num_points)
        for idx in chunk_start:chunk_end
            @inbounds point = Point3{Tg}(points[1, idx], points[2, idx], points[3, idx])
            @inbounds out[idx] = signed_distance_point(sdm, point, stack)
        end
    end
    return out
end
