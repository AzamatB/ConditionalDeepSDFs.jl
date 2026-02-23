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

#######################################   Utilities   #######################################

@inline function norm²(point::Point3{T}) where {T<:AbstractFloat}
    n² = point ⋅ point
    return n²::T
end

#######################################   Data Structures   #######################################

# packed triangle vertices (contiguous by BVH leaf order for cache locality)
# perfectly aligned to exactly 64 bytes (16 Floats) for an unbroken L1 cache-line
struct TriangleGeometry{T<:AbstractFloat}
    a::Point3{T}
    ab::Point3{T}
    ac::Point3{T}
    d11::T           # norm²(ab)
    d12::T           # ab ⋅ ac
    d22::T           # norm²(ac)
    inv_denom::T     # 1.0 / (d11*d22 - d12²)
    inv_d11::T       # 1.0 / d11
    inv_d22::T       # 1.0 / d22
    inv_d33::T       # 1.0 / norm²(bc)
end

# 32-byte aligned Array of Structs (AoS) for the Fast Winding Number (Barill et al. 2018)
# We store a 0th-order (dipole) far-field expansion:
#   w̃(q) = (n_sum ⋅ (c - q)) / (4π‖c - q‖³)
# where n_sum = Σ_t area_t * n̂_t (vector area) and c is the area-weighted centroid.
struct FWNNode{T<:AbstractFloat}
    cm_x::T
    cm_y::T
    cm_z::T
    r_beta_sq::T    # Precomputed threshold (β*r)² for admissibility
    n_x::T
    n_y::T
    n_z::T
    _pad::T         # 32-byte alignment for cache line efficiency
end

struct FastWindingData{T<:AbstractFloat}
    nodes::Vector{FWNNode{T}}
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

# Tg: geometry/distance type
struct SignedDistanceMesh{Tg<:AbstractFloat}
    tri_geometries::Vector{TriangleGeometry{Tg}}   # packed by BVH leaf order
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
    builder::BVHBuilder{T}, node_id::Int32,
    tri_indices::Vector{Int32}, lo::Int, hi::Int, centroids::NTuple{3,Vector{T}},
    lb_x_t::Vector{T}, lb_y_t::Vector{T}, lb_z_t::Vector{T},
    ub_x_t::Vector{T}, ub_y_t::Vector{T}, ub_z_t::Vector{T},
) where {T}

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

    # Pre-allocate BOTH children contiguously BEFORE recursing for L1 cache-line pairing.
    child_l = builder.next_node
    child_r = builder.next_node + Int32(1)
    builder.next_node += Int32(2)

    build_node!(
        builder, child_l, tri_indices, lo, mid, centroids, lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t
    )
    build_node!(
        builder, child_r, tri_indices, mid + 1, hi, centroids, lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t
    )
    @inbounds builder.nodes[node_id] = BVHNode{T}(
        min_x, min_y, min_z, max_x, max_y, max_z, child_l, child_r
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

    max_nodes = 2 * num_faces + 1 # +1 for structural padding

    # Start allocating children at index 3 to guarantee sibling alignment per 64-byte lines
    builder = BVHBuilder{Tg}(Vector{BVHNode{Tg}}(undef, max_nodes), Int32(leaf_capacity), Int32(3))

    # Safely initialize node 2 as a dummy to avoid out-of-bounds risks if scanned directly
    if max_nodes >= 2
        builder.nodes[2] = BVHNode{Tg}(
            zero(Tg), zero(Tg), zero(Tg), zero(Tg), zero(Tg), zero(Tg), Int32(0), Int32(0)
        )
    end

    build_node!(
        builder, Int32(1), tri_indices, 1, num_faces, centroids, lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t
    )

    num_nodes = builder.next_node - 1
    # Truncate builder vector precisely to size in-place to avoid deepcopy allocation
    resize!(builder.nodes, num_nodes)
    bvh = BoundingVolumeHierarchy{Tg}(builder.nodes, builder.leaf_capacity, Int32(num_nodes))
    return (bvh, tri_indices)  # return triangle order for packing
end

######################################   Mesh Preprocessing   ######################################

"""
    preprocess_mesh(mesh; leaf_capacity=8, winding_beta=2.0)

Build the preprocessing data needed for fast signed-distance queries:

- A BVH for **unsigned** distance queries (closest-point distance).
- A fast **generalized winding number** hierarchy for a robust inside/outside **sign**
  (Barnes–Hut-style acceleration, Barill et al. 2018).

Arguments:
- `mesh`: a closed or non-watertight "soup" triangle mesh.
- `leaf_capacity`: BVH leaf size (distance & winding traversal share the same BVH).
- `winding_beta` (β): Barnes–Hut admissibility parameter. Larger ⇒ more accurate winding numbers,
  but more tree traversal work. A common default is `2.0`.

Returns a `SignedDistanceMesh{Tg}` ready for `compute_signed_distance!` calls.
"""
function preprocess_mesh(
    mesh::Mesh{3,Tg,GLTriangleFace}; leaf_capacity::Int=8, winding_beta::Real=2.0
) where {Tg<:AbstractFloat}
    vertices = GeometryBasics.coordinates(mesh)
    tri_faces = GeometryBasics.faces(mesh)
    faces = NTuple{3,Int32}.(tri_faces)
    return preprocess_mesh(vertices, faces; leaf_capacity, winding_beta)
end

function preprocess_mesh(
    vertices::AbstractVector{<:Point3{Tg}},
    faces::AbstractVector{NTuple{3,Int32}};
    leaf_capacity::Int=8,
    winding_beta::Real=2.0
) where {Tg<:AbstractFloat}
    num_faces = length(faces)
    num_faces > 0 || error("Mesh must contain at least one face.")

    # Build BVH (in Tg)
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

    # pack triangle geometry contiguously by BVH leaf order
    tri_geometries = Vector{TriangleGeometry{Tg}}(undef, num_faces)

    # map original face index → packed index (for triangle-hint acceleration)
    face_to_packed = Vector{Int32}(undef, num_faces)

    @inbounds for j in eachindex(faces)
        idx_face = tri_order[j]   # original face index
        face_to_packed[idx_face] = Int32(j)

        (idx_v1, idx_v2, idx_v3) = faces[idx_face]
        v1 = vertices[idx_v1]
        v2 = vertices[idx_v2]
        v3 = vertices[idx_v3]

        ab = v2 - v1
        ac = v3 - v1

        d11 = norm²(ab)
        d12 = ab ⋅ ac
        d22 = norm²(ac)

        inv_d11 = d11 > 0 ? inv(d11) : zero(Tg)
        inv_d22 = d22 > 0 ? inv(d22) : zero(Tg)

        # d33 is exactly ||ac - ab||². Computed purely geometrically to avoid catastrophic cancellation on slivers.
        bcx = ac[1] - ab[1]
        bcy = ac[2] - ab[2]
        bcz = ac[3] - ab[3]
        d33 = bcx * bcx + bcy * bcy + bcz * bcz
        inv_d33 = d33 > 0 ? inv(d33) : zero(Tg)

        # denom_sum is exactly ||ab × ac||². Computed purely geometrically to avoid catastrophic cancellation on slivers.
        cx = ab[2] * ac[3] - ab[3] * ac[2]
        cy = ab[3] * ac[1] - ab[1] * ac[3]
        cz = ab[1] * ac[2] - ab[2] * ac[1]
        denom_sum = cx * cx + cy * cy + cz * cz
        inv_denom = denom_sum > 0 ? inv(denom_sum) : zero(Tg)

        tri_geometries[j] = TriangleGeometry{Tg}(
            v1, ab, ac,
            d11, d12, d22,
            inv_denom, inv_d11, inv_d22, inv_d33
        )
    end

    # Precompute fast generalized winding-number (Barill et al. 2018) data
    fwn = precompute_fast_winding_data(bvh, tri_geometries; beta=Tg(winding_beta))

    return SignedDistanceMesh{Tg}(tri_geometries, bvh, face_to_packed, fwn)
end


#################################   Fast Winding Number Precompute   #################################

# Precompute node-wise data for the fast generalized winding number of a triangle soup.
# We use a Barnes–Hut style far-field approximation (0th order / "single dipole" per node)
function precompute_fast_winding_data(
    bvh::BoundingVolumeHierarchy{Tg},
    tri_geometries::Vector{TriangleGeometry{Tg}};
    beta::Tg=Tg(2)
) where {Tg<:AbstractFloat}

    num_nodes = Int(bvh.num_nodes)
    fwn_nodes = Vector{FWNNode{Tg}}(undef, num_nodes)

    # Temporary aggregation buffers in Float64 for stable summation during preprocessing
    area_sum = Vector{Float64}(undef, num_nodes)
    cent_sum_x = Vector{Float64}(undef, num_nodes)
    cent_sum_y = Vector{Float64}(undef, num_nodes)
    cent_sum_z = Vector{Float64}(undef, num_nodes)
    n_sum_x = Vector{Float64}(undef, num_nodes)
    n_sum_y = Vector{Float64}(undef, num_nodes)
    n_sum_z = Vector{Float64}(undef, num_nodes)

    @inbounds for node_id in num_nodes:-1:1
        (node_id == 2) && continue # Skip unused alignment padding

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

        # squared radius bound: farthest AABB corner from (c_x, c_y, c_z)
        dx = max(abs(c_x - Float64(node.lb_x)), abs(c_x - Float64(node.ub_x)))
        dy = max(abs(c_y - Float64(node.lb_y)), abs(c_y - Float64(node.ub_y)))
        dz = max(abs(c_z - Float64(node.lb_z)), abs(c_z - Float64(node.ub_z)))

        r_sq = dx * dx + dy * dy + dz * dz
        r_beta_sq = (Float64(beta)^2) * r_sq

        fwn_nodes[node_id] = FWNNode{Tg}(
            Tg(c_x), Tg(c_y), Tg(c_z),
            Tg(r_beta_sq),
            Tg(n_sum_x[node_id]), Tg(n_sum_y[node_id]), Tg(n_sum_z[node_id]),
            zero(Tg) # Empty space padding for cache-line alignment
        )
    end

    return FastWindingData{Tg}(fwn_nodes)
end

function calculate_tree_height(num_faces::Integer, leaf_capacity::Integer)
    num_leaves = max(cld(num_faces, leaf_capacity), 1)
    tree_height = ceil(Int, log2(num_leaves))
    return tree_height::Int
end

# allocate one traversal stack per chunk
function allocate_stacks(sdm::SignedDistanceMesh{Tg}, num_points::Int) where {Tg}
    num_faces = length(sdm.tri_geometries)
    leaf_capacity = sdm.bvh.leaf_capacity
    tree_height = calculate_tree_height(num_faces, leaf_capacity)
    stack_capacity = 2 * tree_height + 4
    n_threads = Threads.nthreads()
    # enforce a minimum chunk size to prevent false sharing and guarantee cache locality.
    min_chunk_size = 512
    num_chunks_max = max(1, num_points ÷ min_chunk_size)

    if num_chunks_max < n_threads
        num_chunks = num_chunks_max
    else
        factor_max = 8
        factor = min(factor_max, num_chunks_max ÷ n_threads)
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

# Division-free, cancellation-free exact projection logic evaluating purely by orthogonality
@inline function closest_dist²_triangle(p::Point3{Tg}, triangle::TriangleGeometry{Tg}) where {Tg}
    ap = p - triangle.a
    d1 = triangle.ab ⋅ ap
    d2 = triangle.ac ⋅ ap

    (d1 <= 0 && d2 <= 0) && return norm²(ap)

    d11, d12, d22 = triangle.d11, triangle.d12, triangle.d22

    d3 = d1 - d11
    d4 = d2 - d12
    (d3 >= 0 && d4 <= d3) && return norm²(ap - triangle.ab)

    vc = d11 * d2 - d12 * d1
    if vc <= 0 && d1 >= 0 && d3 <= 0
        v = d1 * triangle.inv_d11
        return norm²(ap - v * triangle.ab)
    end

    d5 = d1 - d12
    d6 = d2 - d22
    (d6 >= 0 && d5 <= d6) && return norm²(ap - triangle.ac)

    vb = d22 * d1 - d12 * d2
    if vb <= 0 && d2 >= 0 && d6 <= 0
        w = d2 * triangle.inv_d22
        return norm²(ap - w * triangle.ac)
    end

    va = (d11 * d22 - d12 * d12) - vb - vc
    d43 = d11 - d12 - d1 + d2
    d56 = d22 - d12 + d1 - d2
    if va <= 0 && d43 >= 0 && d56 >= 0
        w = d43 * triangle.inv_d33
        return norm²(ap - (one(Tg) - w) * triangle.ab - w * triangle.ac)
    end

    v = vb * triangle.inv_denom
    w = vc * triangle.inv_denom
    return norm²(ap - v * triangle.ab - w * triangle.ac)
end

#################################   Fast Face-Interior Sign Shortcut   #################################

# If the closest point from `p` to `tri` lies strictly in the *interior* of the triangle face,
# the signed distance sign can be obtained from the oriented face normal without computing a
# (fast) generalized winding number.
#
# Returns:
#   +1  => outside (same side as outward face normal)
#   -1  => inside
#    0  => closest point is on an edge/vertex (or triangle is degenerate) → fall back to winding number
@inline function face_interior_sign_or_zero(
    p::Point3{Tg}, tri::TriangleGeometry{Tg}, bary_tol::Tg=Tg(1e-4)
) where {Tg<:AbstractFloat}
    inv_denom = tri.inv_denom
    iszero(inv_denom) && return zero(Tg)

    ap = p - tri.a
    d1 = tri.ab ⋅ ap
    d2 = tri.ac ⋅ ap

    # Directly evaluate the 2x2 barycentric system projection (Massive reduction in branching overhead)
    vb = tri.d22 * d1 - tri.d12 * d2
    vc = tri.d11 * d2 - tri.d12 * d1

    v = vb * inv_denom
    w = vc * inv_denom
    u = one(Tg) - v - w

    # If the projection is NOT strictly inside the face, fall back to FWN
    (v <= bary_tol || w <= bary_tol || u <= bary_tol) && return zero(Tg)

    # Oriented (outward) face normal (unnormalized)
    ab = tri.ab
    ac = tri.ac
    nx = ab[2] * ac[3] - ab[3] * ac[2]
    ny = ab[3] * ac[1] - ab[1] * ac[3]
    nz = ab[1] * ac[2] - ab[2] * ac[1]

    # Signed plane-side test: sign((p - a) ⋅ n)
    dot = ap[1] * nx + ap[2] * ny + ap[3] * nz
    return ifelse(dot >= zero(Tg), one(Tg), -one(Tg))
end

#################################   Fast Winding Number Query   #################################

const INV4PI64 = 1.0 / (4.0 * π)
const INV2PI64 = 1.0 / (2.0 * π)

# Signed solid angle of a single oriented triangle as seen from q, normalized.
@inline function solid_angle_scaled(q::Point3{Tg}, tri::TriangleGeometry{Tg}) where {Tg<:AbstractFloat}
    qx = Float64(q[1])
    qy = Float64(q[2])
    qz = Float64(q[3])

    # Exploit relative vectors mapping to drop global coordinate additions
    ax = Float64(tri.a[1]) - qx
    ay = Float64(tri.a[2]) - qy
    az = Float64(tri.a[3]) - qz

    bx = ax + Float64(tri.ab[1])
    by = ay + Float64(tri.ab[2])
    bz = az + Float64(tri.ab[3])

    cx = ax + Float64(tri.ac[1])
    cy = ay + Float64(tri.ac[2])
    cz = az + Float64(tri.ac[3])

    la = sqrt(ax * ax + ay * ay + az * az)
    lb = sqrt(bx * bx + by * by + bz * bz)
    lc = sqrt(cx * cx + cy * cy + cz * cz)

    tpx = by * cz - bz * cy
    tpy = bz * cx - bx * cz
    tpz = bx * cy - by * cx
    det = ax * tpx + ay * tpy + az * tpz

    ab = ax * bx + ay * by + az * bz
    ac = ax * cx + ay * cy + az * cz
    bc = bx * cx + by * cy + bz * cz
    denom = la * lb * lc + ab * lc + ac * lb + bc * la

    return atan(det, denom) * INV2PI64
end

@inline function winding_number_point_kernel(
    sdm::SignedDistanceMesh{Tg},
    point::Point3{Tg},
    stack::Vector{Int32}
)::Float64 where {Tg<:AbstractFloat}

    fwn_nodes = sdm.fwn.nodes
    bvh_nodes = sdm.bvh.nodes
    tri_geometries = sdm.tri_geometries

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

        fnode = fwn_nodes[node_id]

        rx = Float64(fnode.cm_x) - qx
        ry = Float64(fnode.cm_y) - qy
        rz = Float64(fnode.cm_z) - qz

        dist2 = rx * rx + ry * ry + rz * rz

        if dist2 > Float64(fnode.r_beta_sq)
            # Far field: 0th order (dipole) approximation
            nx = Float64(fnode.n_x)
            ny = Float64(fnode.n_y)
            nz = Float64(fnode.n_z)

            dot = rx * nx + ry * ny + rz * nz
            inv_denom = inv(dist2 * sqrt(dist2))  # 1/‖r‖^3
            wn += dot * INV4PI64 * inv_denom
        else
            node = bvh_nodes[node_id]
            child_or_size = node.child_or_size

            if child_or_size < 0
                leaf_start = Int(node.index)
                leaf_size = Int(-child_or_size)
                leaf_end = leaf_start + leaf_size - 1
                for tri_id in leaf_start:leaf_end
                    wn += solid_angle_scaled(point, tri_geometries[tri_id])
                end
            else
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
    sdm::SignedDistanceMesh{Tg}, point::Point3{Tg}, hint_face::Int32, stacks::QueryStacks{Tg}
) where {Tg<:AbstractFloat}
    # tighten initial bound using the provided triangle hint (packed index).
    tri_best = hint_face
    @inbounds triangle = sdm.tri_geometries[tri_best]
    dist²_best = closest_dist²_triangle(point, triangle)
    (dist²_best <= 0) && return zero(Tg)

    return signed_distance_point_kernel(sdm, point, dist²_best, tri_best, stacks)
end

function signed_distance_point(
    sdm::SignedDistanceMesh{Tg}, point::Point3{Tg}, stacks::QueryStacks{Tg}
) where {Tg<:AbstractFloat}
    dist²_best = Tg(Inf)
    tri_best = Int32(0)
    return signed_distance_point_kernel(sdm, point, dist²_best, tri_best, stacks)
end

function signed_distance_point_kernel(
    sdm::SignedDistanceMesh{Tg},
    point::Point3{Tg},
    dist²_best::Tg,
    tri_best::Int32,
    stacks::QueryStacks{Tg}
) where {Tg<:AbstractFloat}
    bvh = sdm.bvh
    tri_geometries = sdm.tri_geometries
    stack = stacks.dist
    wind_stack = stacks.wind

    dist²_root = aabb_dist²(point, bvh, Int32(1))
    stack_top = 1
    @inbounds stack[1] = NodeDist{Tg}(Int32(1), dist²_root)

    @inbounds while stack_top > 0
        node_dist = stack[stack_top]
        stack_top -= 1

        (node_dist.dist² > dist²_best) && continue

        node = bvh.nodes[node_dist.node_id]
        child_or_size = node.child_or_size

        if child_or_size > 0 # internal node
            child_l = node.index
            child_r = child_or_size

            dist²_l = aabb_dist²(point, bvh, child_l)
            dist²_r = aabb_dist²(point, bvh, child_r)

            if dist²_l > dist²_r
                (child_l, child_r) = (child_r, child_l)
                (dist²_l, dist²_r) = (dist²_r, dist²_l)
            end

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
                dist² = closest_dist²_triangle(point, tri_geometries[idx])
                if dist² < dist²_best
                    dist²_best = dist²
                    tri_best = idx
                end
            end
        end
    end

    iszero(tri_best) && error("No triangle found for point $point")
    dist = √(dist²_best)
    iszero(dist) && return zero(Tg)

    # Fast sign shortcut:
    # If the closest point lies strictly in the interior of the closest triangle face,
    # we can determine the sign using only the oriented face normal (no winding-number query).
    @inbounds tri = tri_geometries[tri_best]
    sgn_fast = face_interior_sign_or_zero(point, tri)
    if !iszero(sgn_fast)
        return (sgn_fast * dist)::Tg
    end

    # Robust inside/outside sign via (fast) generalized winding number.
    wn = winding_number_point_kernel(sdm, point, wind_stack)

    uno = one(Tg)
    # Removing abs() ensures robust structural orientation alignment (preventing checkerboards on inverted normals)
    sgn = ifelse(wn >= 0.5, -uno, uno)  # inside => negative signed distance
    return (sgn * dist)::Tg
end

##########################################   Public API   ##########################################

"""
    compute_signed_distance!(out, sdm, points_mat, [hint_faces])

In-place batch signed distance query. Writes results into `out`.
- `out`:            length-n vector to store the output signed distances.
- `sdm`:            a [`SignedDistanceMesh`] built once via `preprocess_mesh`.
- `points_mat`:     `3 × n` matrix of query points.
- `hint_faces`:     length-n vector of *original* face indices (1-based) to accelerate surface queries.

Positive = outside, negative = inside.
"""
function compute_signed_distance!(
    out::AbstractVector{Tg},
    sdm::SignedDistanceMesh{Tg},
    points::StridedMatrix{Tg},
    hint_faces::Vector{Int32}
) where {Tg<:AbstractFloat}
    num_points = size(points, 2)
    @assert length(out) == length(hint_faces) == num_points
    @assert size(points, 1) == 3 "points matrix must be 3×n"

    face_to_packed = sdm.face_to_packed
    stacks = allocate_stacks(sdm, num_points)
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
    out::AbstractVector{Tg}, sdm::SignedDistanceMesh{Tg}, points::StridedMatrix{Tg}
) where {Tg<:AbstractFloat}
    num_points = size(points, 2)
    @assert length(out) == num_points
    @assert size(points, 1) == 3 "points matrix must be 3×n"

    stacks = allocate_stacks(sdm, num_points)
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

function compute_signed_distance!(
    out::AbstractVector{Tg},
    sdm::SignedDistanceMesh{Tg},
    points::StridedMatrix{Tg},
    hint_faces::Vector{Int32},
    stacks::Vector{QueryStacks{Tg}}
) where {Tg<:AbstractFloat}
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
    sdm::SignedDistanceMesh{Tg},
    points::StridedMatrix{Tg},
    stacks::Vector{QueryStacks{Tg}}
) where {Tg<:AbstractFloat}
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
