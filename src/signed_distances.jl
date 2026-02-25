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

@inline function norm²(point::Point3{T}) where {T<:AbstractFloat}
    n² = point ⋅ point
    return n²::T
end

#######################################   Data Structures   #######################################

# packed triangle geometry aligned to 64 bytes (16 floats) for L1 cache-line
struct TriangleGeometry{T<:AbstractFloat}
    a::Point3{T}
    ab::Point3{T}         # edge vector: b - a
    ac::Point3{T}         # edge vector: c - a
    n::Point3{T}          # unnormalized face normal: ab × ac
    d11::T                # norm²(ab)
    d12::T                # ab ⋅ ac
    d22::T                # norm²(ac)
    inv_denom::T          # 1 / norm²(n), zero for degenerate triangles
end

# 0th-order dipole expansion for fast winding numbers.
# topology packed into UInt32 to avoid redundant BVH fetches:
#   leaf:     high-bit set, bits 25–30 = count, bits 0–24 = leaf_start
#   internal: bits 0–24 = left child (right child = left + 1)
struct FastWindingNode{T<:AbstractFloat}
    centroid_x::T
    centroid_y::T
    centroid_z::T
    normal_x::T               # area-weighted normal ÷ (4π)
    normal_y::T
    normal_z::T
    radius²_β²::T             # bounding-sphere radius² × β²
    topology::UInt32
end

struct FastWindingData{T<:AbstractFloat}
    nodes::Vector{FastWindingNode{T}}
end

# AoS BVH node with overlapped integer fields for cache efficiency.
# for T=Float32 this is 6×4 + 2×4 = 32 bytes — exactly two nodes per 64-byte cache line.
# internal and leaf nodes overlap integer fields via sign-bit discriminant:
#   internal: index = left child,  child_or_size = right child  (both > 0)
#   leaf:     index = leaf_start,  child_or_size = -leaf_end    (< 0)
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
    max_depth::Int32
    leaf_capacity::Int32
    num_nodes::Int32
end

# isotropic spatial voxel grid over [-1, 1]³ with padding.
# each cell stores the nearest packed triangle index and the signed distance at cell center.
# cell_size and offset are precomputed so center = idx * cell_size + offset (pure FMA).
struct HintGrid{T<:AbstractFloat}
    lb::T
    inv_cell::T
    cell_size::T
    offset::T
    res::Int
    cells::Array{Tuple{Int32,T},3}  # (nearest_tri, signed_distance_at_center)
end

# stack element carrying node topology for inline AABB traversal.
# stores child_or_size alongside dist² to avoid re-fetching the node on pop.
struct NodeDist{T<:AbstractFloat}
    dist²::T
    index::Int32
    child_or_size::Int32
end

struct QueryStacks{T<:AbstractFloat}
    dist::Vector{NodeDist{T}}
    wind::Vector{Int32}
end

struct SignedDistanceMesh{Tg<:AbstractFloat}
    tri_geometries::Vector{TriangleGeometry{Tg}}   # packed by BVH leaf order
    bvh::BoundingVolumeHierarchy{Tg}
    # face_to_packed[f] = packed triangle index for original face id f
    face_to_packed::Vector{Int32}
    fast_winding_data::FastWindingData{Tg}
    hint_grid::HintGrid{Tg}
end

#######################################   BVH Construction   #######################################

# partial sort of indices by centroid along axis (build-time only)
function median_split_sort!(
    indices::Vector{Int32}, lo::Int, mid::Int, hi::Int, centroids::NTuple{3,Vector{Tg}}, axis::Int
) where {Tg<:AbstractFloat}
    sub_indices = @view indices[lo:hi]
    centroids_axis = centroids[axis]
    mid_relative = mid - lo + 1
    partialsort!(sub_indices, mid_relative; by=idx_face -> centroids_axis[idx_face])
    return nothing
end

###################################   SAH (Surface Area Heuristic)   #################################

const NUM_SAH_BINS = 16

mutable struct SAHScratch{T<:AbstractFloat}
    count::Vector{Int32}
    lb_x::Vector{T}
    lb_y::Vector{T}
    lb_z::Vector{T}
    ub_x::Vector{T}
    ub_y::Vector{T}
    ub_z::Vector{T}
    suffix_count::Vector{Int32}
    suffix_lb_x::Vector{T}
    suffix_lb_y::Vector{T}
    suffix_lb_z::Vector{T}
    suffix_ub_x::Vector{T}
    suffix_ub_y::Vector{T}
    suffix_ub_z::Vector{T}
end

function SAHScratch{T}() where {T<:AbstractFloat}
    num_bins = NUM_SAH_BINS
    count = Vector{Int32}(undef, num_bins)
    lb_x = Vector{T}(undef, num_bins)
    lb_y = Vector{T}(undef, num_bins)
    lb_z = Vector{T}(undef, num_bins)
    ub_x = Vector{T}(undef, num_bins)
    ub_y = Vector{T}(undef, num_bins)
    ub_z = Vector{T}(undef, num_bins)

    suffix_count = Vector{Int32}(undef, num_bins + 1)
    suffix_lb_x = Vector{T}(undef, num_bins + 1)
    suffix_lb_y = Vector{T}(undef, num_bins + 1)
    suffix_lb_z = Vector{T}(undef, num_bins + 1)
    suffix_ub_x = Vector{T}(undef, num_bins + 1)
    suffix_ub_y = Vector{T}(undef, num_bins + 1)
    suffix_ub_z = Vector{T}(undef, num_bins + 1)

    scratch = SAHScratch{T}(
        count, lb_x, lb_y, lb_z, ub_x, ub_y, ub_z,
        suffix_count, suffix_lb_x, suffix_lb_y, suffix_lb_z,
        suffix_ub_x, suffix_ub_y, suffix_ub_z
    )
    return scratch::SAHScratch{T}
end

@inline function reset_sah_bins!(scratch::SAHScratch{T}) where {T<:AbstractFloat}
    nb = NUM_SAH_BINS
    fill!(scratch.count, Int32(0))
    inf_val = floatmax(T)
    ninf_val = -floatmax(T)
    @inbounds for b in 1:nb
        scratch.lb_x[b] = inf_val
        scratch.lb_y[b] = inf_val
        scratch.lb_z[b] = inf_val
        scratch.ub_x[b] = ninf_val
        scratch.ub_y[b] = ninf_val
        scratch.ub_z[b] = ninf_val
    end
    return nothing
end

@inline function centroid_bin_index(c::T, centroid_min::T, inv_extent::T) where {T<:AbstractFloat}
    x = (c - centroid_min) * inv_extent
    idx_bin_raw = Int(floor(x))
    nbm1 = NUM_SAH_BINS - 1
    idx_bin_raw = ifelse(idx_bin_raw < 0, 0, ifelse(idx_bin_raw > nbm1, nbm1, idx_bin_raw))
    idx_bin = idx_bin_raw + 1
    return idx_bin::Int
end

@inline function aabb_half_area(lb_x::T, lb_y::T, lb_z::T, ub_x::T, ub_y::T, ub_z::T) where {T<:AbstractFloat}
    zer = zero(T)
    Δx = max(ub_x - lb_x, zer)
    Δy = max(ub_y - lb_y, zer)
    Δz = max(ub_z - lb_z, zer)
    half_area = muladd(Δx, Δy, muladd(Δx, Δz, Δy * Δz))
    return half_area::T
end

# evaluate SAH cost for all bin splits along one axis; returns (best_cost, best_split, inv_extent)
@inline function sah_best_split_axis!(
    scratch::SAHScratch{T},
    tri_indices::Vector{Int32}, lo::Int, hi::Int,
    centroids_axis::Vector{T}, centroid_min::T, centroid_max::T,
    lb_x_t::Vector{T}, lb_y_t::Vector{T}, lb_z_t::Vector{T},
    ub_x_t::Vector{T}, ub_y_t::Vector{T}, ub_z_t::Vector{T}
) where {T<:AbstractFloat}
    extent = centroid_max - centroid_min
    (extent > zero(T)) || return (floatmax(T), 0, zero(T))

    num_bins = NUM_SAH_BINS
    inv_extent = T(num_bins) / extent
    reset_sah_bins!(scratch)

    # bin triangles by centroid position
    @inbounds for i in lo:hi
        idx_face = tri_indices[i]
        idx_bin = centroid_bin_index(centroids_axis[idx_face], centroid_min, inv_extent)

        scratch.count[idx_bin] += Int32(1)
        scratch.lb_x[idx_bin] = min(scratch.lb_x[idx_bin], lb_x_t[idx_face])
        scratch.lb_y[idx_bin] = min(scratch.lb_y[idx_bin], lb_y_t[idx_face])
        scratch.lb_z[idx_bin] = min(scratch.lb_z[idx_bin], lb_z_t[idx_face])
        scratch.ub_x[idx_bin] = max(scratch.ub_x[idx_bin], ub_x_t[idx_face])
        scratch.ub_y[idx_bin] = max(scratch.ub_y[idx_bin], ub_y_t[idx_face])
        scratch.ub_z[idx_bin] = max(scratch.ub_z[idx_bin], ub_z_t[idx_face])
    end

    # suffix scan for right-side bounds
    inf_val = floatmax(T)
    ninf_val = -inf_val
    scratch.suffix_count[num_bins+1] = Int32(0)
    scratch.suffix_lb_x[num_bins+1] = inf_val
    scratch.suffix_lb_y[num_bins+1] = inf_val
    scratch.suffix_lb_z[num_bins+1] = inf_val
    scratch.suffix_ub_x[num_bins+1] = ninf_val
    scratch.suffix_ub_y[num_bins+1] = ninf_val
    scratch.suffix_ub_z[num_bins+1] = ninf_val

    @inbounds for idx_bin in num_bins:-1:1
        scratch.suffix_count[idx_bin] = scratch.suffix_count[idx_bin+1] + scratch.count[idx_bin]
        scratch.suffix_lb_x[idx_bin] = min(scratch.lb_x[idx_bin], scratch.suffix_lb_x[idx_bin+1])
        scratch.suffix_lb_y[idx_bin] = min(scratch.lb_y[idx_bin], scratch.suffix_lb_y[idx_bin+1])
        scratch.suffix_lb_z[idx_bin] = min(scratch.lb_z[idx_bin], scratch.suffix_lb_z[idx_bin+1])
        scratch.suffix_ub_x[idx_bin] = max(scratch.ub_x[idx_bin], scratch.suffix_ub_x[idx_bin+1])
        scratch.suffix_ub_y[idx_bin] = max(scratch.ub_y[idx_bin], scratch.suffix_ub_y[idx_bin+1])
        scratch.suffix_ub_z[idx_bin] = max(scratch.ub_z[idx_bin], scratch.suffix_ub_z[idx_bin+1])
    end

    # prefix scan for left-side bounds, evaluate SAH cost at each split
    left_count = Int32(0)
    left_lb_x = inf_val
    left_lb_y = inf_val
    left_lb_z = inf_val
    left_ub_x = ninf_val
    left_ub_y = ninf_val
    left_ub_z = ninf_val

    best_cost = inf_val
    best_split = 0

    @inbounds for idx_bin in 1:(num_bins-1)
        curr_count = scratch.count[idx_bin]
        if !iszero(curr_count)
            left_count += curr_count
            left_lb_x = min(left_lb_x, scratch.lb_x[idx_bin])
            left_lb_y = min(left_lb_y, scratch.lb_y[idx_bin])
            left_lb_z = min(left_lb_z, scratch.lb_z[idx_bin])
            left_ub_x = max(left_ub_x, scratch.ub_x[idx_bin])
            left_ub_y = max(left_ub_y, scratch.ub_y[idx_bin])
            left_ub_z = max(left_ub_z, scratch.ub_z[idx_bin])
        end

        right_count = scratch.suffix_count[idx_bin+1]
        if (left_count > 0) && (right_count > 0)
            area_l = aabb_half_area(left_lb_x, left_lb_y, left_lb_z, left_ub_x, left_ub_y, left_ub_z)
            area_r = aabb_half_area(
                scratch.suffix_lb_x[idx_bin+1], scratch.suffix_lb_y[idx_bin+1], scratch.suffix_lb_z[idx_bin+1],
                scratch.suffix_ub_x[idx_bin+1], scratch.suffix_ub_y[idx_bin+1], scratch.suffix_ub_z[idx_bin+1]
            )
            cost = area_l * T(left_count) + area_r * T(right_count)
            if cost < best_cost
                best_cost = cost
                best_split = idx_bin
            end
        end
    end

    return (best_cost, best_split, inv_extent)
end

########################################   BVH Builder   #########################################

mutable struct BVHBuilder{T}
    const nodes::Vector{BVHNode{T}}
    const leaf_capacity::Int32
    const sah_scratch::SAHScratch{T}
    next_node::Int32
    max_depth::Int32
end

function build_node!(
    builder::BVHBuilder{T}, node_id::Int32,
    tri_indices::Vector{Int32}, lo::Int, hi::Int, centroids::NTuple{3,Vector{T}},
    lb_x_t::Vector{T}, lb_y_t::Vector{T}, lb_z_t::Vector{T},
    ub_x_t::Vector{T}, ub_y_t::Vector{T}, ub_z_t::Vector{T},
    depth::Int
) where {T<:AbstractFloat}

    builder.max_depth = max(builder.max_depth, Int32(depth))

    count = hi - lo + 1
    if count <= builder.leaf_capacity
        # compute leaf bounds
        min_x = floatmax(T)
        min_y = floatmax(T)
        min_z = floatmax(T)
        max_x = -floatmax(T)
        max_y = -floatmax(T)
        max_z = -floatmax(T)
        @inbounds for i in lo:hi
            idx_face = tri_indices[i]
            min_x = min(min_x, lb_x_t[idx_face])
            min_y = min(min_y, lb_y_t[idx_face])
            min_z = min(min_z, lb_z_t[idx_face])
            max_x = max(max_x, ub_x_t[idx_face])
            max_y = max(max_y, ub_y_t[idx_face])
            max_z = max(max_z, ub_z_t[idx_face])
        end

        @inbounds builder.nodes[node_id] = BVHNode{T}(
            min_x, min_y, min_z, max_x, max_y, max_z, Int32(lo), -Int32(hi)
        )
        return node_id::Int32
    end

    # split axis = longest centroid extent
    centroid_min_x = floatmax(T)
    centroid_min_y = floatmax(T)
    centroid_min_z = floatmax(T)
    centroid_max_x = -floatmax(T)
    centroid_max_y = -floatmax(T)
    centroid_max_z = -floatmax(T)
    @inbounds for i in lo:hi
        idx_face = tri_indices[i]
        centroid_min_x = min(centroid_min_x, centroids[1][idx_face])
        centroid_min_y = min(centroid_min_y, centroids[2][idx_face])
        centroid_min_z = min(centroid_min_z, centroids[3][idx_face])
        centroid_max_x = max(centroid_max_x, centroids[1][idx_face])
        centroid_max_y = max(centroid_max_y, centroids[2][idx_face])
        centroid_max_z = max(centroid_max_z, centroids[3][idx_face])
    end

    # try SAH split on all three axes, pick best
    scratch = builder.sah_scratch
    best_cost = floatmax(T)
    best_axis = 0
    best_split = 0
    best_inv_extent = zero(T)
    best_centroid_min = zero(T)

    (cost_x, split_x, inv_extent_x) = sah_best_split_axis!(
        scratch, tri_indices, lo, hi, centroids[1], centroid_min_x, centroid_max_x,
        lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t
    )
    if (!iszero(split_x)) && (cost_x < best_cost)
        best_cost = cost_x
        best_axis = 1
        best_split = split_x
        best_inv_extent = inv_extent_x
        best_centroid_min = centroid_min_x
    end

    (cost_y, split_y, inv_extent_y) = sah_best_split_axis!(
        scratch, tri_indices, lo, hi, centroids[2], centroid_min_y, centroid_max_y,
        lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t
    )
    if (!iszero(split_y)) && (cost_y < best_cost)
        best_cost = cost_y
        best_axis = 2
        best_split = split_y
        best_inv_extent = inv_extent_y
        best_centroid_min = centroid_min_y
    end

    (cost_z, split_z, inv_extent_z) = sah_best_split_axis!(
        scratch, tri_indices, lo, hi, centroids[3], centroid_min_z, centroid_max_z,
        lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t
    )
    if (!iszero(split_z)) && (cost_z < best_cost)
        best_cost = cost_z
        best_axis = 3
        best_split = split_z
        best_inv_extent = inv_extent_z
        best_centroid_min = centroid_min_z
    end

    # partition triangles by SAH bin boundary
    mid = 0
    if !iszero(best_axis)
        axis = best_axis
        i = lo
        j = hi
        @inbounds while i <= j
            idx_face = tri_indices[i]
            idx_bin = centroid_bin_index(centroids[axis][idx_face], best_centroid_min, best_inv_extent)
            if idx_bin <= best_split
                i += 1
            else
                (tri_indices[i], tri_indices[j]) = (tri_indices[j], tri_indices[i])
                j -= 1
            end
        end
        mid = j
    end

    # fallback to median split if SAH produced a degenerate partition
    if iszero(best_axis) || (mid < lo) || (mid >= hi)
        spread_x = centroid_max_x - centroid_min_x
        spread_y = centroid_max_y - centroid_min_y
        spread_z = centroid_max_z - centroid_min_z
        (spread_max, axis) = findmax((spread_x, spread_y, spread_z))
        mid = (lo + hi) >>> 1
        (spread_max > 0) && median_split_sort!(tri_indices, lo, mid, hi, centroids, axis)
    end

    node_left_id = builder.next_node
    node_right_id = builder.next_node + Int32(1)
    builder.next_node += Int32(2)

    build_node!(
        builder, node_left_id, tri_indices, lo, mid, centroids,
        lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t, depth + 1
    )
    build_node!(
        builder, node_right_id, tri_indices, mid + 1, hi, centroids,
        lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t, depth + 1
    )

    # parent bounds = union of child bounds
    @inbounds begin
        node_l = builder.nodes[node_left_id]
        node_r = builder.nodes[node_right_id]
        builder.nodes[node_id] = BVHNode{T}(
            min(node_l.lb_x, node_r.lb_x), min(node_l.lb_y, node_r.lb_y), min(node_l.lb_z, node_r.lb_z),
            max(node_l.ub_x, node_r.ub_x), max(node_l.ub_y, node_r.ub_y), max(node_l.ub_z, node_r.ub_z),
            node_left_id, node_right_id
        )
    end

    return node_id::Int32
end

function build_bvh(
    centroids::NTuple{3,Vector{Tg}},
    lb_x_t::Vector{Tg}, lb_y_t::Vector{Tg}, lb_z_t::Vector{Tg},
    ub_x_t::Vector{Tg}, ub_y_t::Vector{Tg}, ub_z_t::Vector{Tg};
    leaf_capacity::Int=8
) where {Tg<:AbstractFloat}

    # cap leaf capacity to 63 for 6-bit packing in Fast Winding Number topology field
    leaf_capacity = min(leaf_capacity, 63)
    num_faces = length(first(centroids))
    @assert num_faces <= 33554431 "Mesh exceeds 33.5M faces."

    tri_indices = Int32.(1:num_faces)
    max_nodes = 2 * num_faces + 1

    nodes = Vector{BVHNode{Tg}}(undef, max_nodes)
    sah_scratch = SAHScratch{Tg}()
    builder = BVHBuilder{Tg}(nodes, Int32(leaf_capacity), sah_scratch, Int32(3), Int32(1))

    # sentinel node at index 2 (builder.next_node starts at 3)
    if max_nodes >= 2
        zer = zero(Tg)
        builder.nodes[2] = BVHNode{Tg}(zer, zer, zer, zer, zer, zer, Int32(0), Int32(0))
    end

    build_node!(
        builder, Int32(1), tri_indices, 1, num_faces, centroids,
        lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t, 1
    )

    num_nodes = builder.next_node - 1
    resize!(builder.nodes, num_nodes)
    bvh = BoundingVolumeHierarchy{Tg}(builder.nodes, builder.max_depth, builder.leaf_capacity, Int32(num_nodes))
    return (bvh, tri_indices)  # return triangle order for packing
end

######################################   Mesh Preprocessing   ######################################

# BVH closest-triangle search with optional pre-seeded hint.
# returns (packed_triangle_index, squared_distance).
function closest_triangle_kernel(
    point::Point3{Tg},
    bvh::BoundingVolumeHierarchy{Tg},
    tri_geometries::Vector{TriangleGeometry{Tg}},
    stack::Vector{NodeDist{Tg}},
    tri_best::Int32=Int32(0),
    dist²_best::Tg=floatmax(Tg)
) where {Tg<:AbstractFloat}
    zer = zero(Tg)
    inf_val = floatmax(Tg)

    # tighten initial bound if a hint triangle was provided but not yet evaluated
    if (dist²_best == inf_val) && (tri_best > 0)
        @inbounds triangle = tri_geometries[tri_best]
        dist²_best = closest_dist²_triangle(point, triangle, dist²_best)
    end

    if dist²_best == inf_val
        tri_best = Int32(1)
    end

    (point_x, point_y, point_z) = point
    stack_top = 1
    @inbounds root_node = bvh.nodes[1]
    @inbounds stack[1] = NodeDist{Tg}(zer, root_node.index, root_node.child_or_size)

    @inbounds while stack_top > 0
        node_dist = stack[stack_top]
        stack_top -= 1

        while true
            (node_dist.dist² >= dist²_best) && break
            child_or_size = node_dist.child_or_size

            if child_or_size > 0  # internal node
                child_l_id = node_dist.index
                child_r_id = child_or_size

                node_l = bvh.nodes[child_l_id]
                node_r = bvh.nodes[child_r_id]

                # inline AABB dist² for both children
                @fastmath begin
                    Δx_l = max(node_l.lb_x - point_x, point_x - node_l.ub_x, zer)
                    Δy_l = max(node_l.lb_y - point_y, point_y - node_l.ub_y, zer)
                    Δz_l = max(node_l.lb_z - point_z, point_z - node_l.ub_z, zer)
                    dist²_l = muladd(Δx_l, Δx_l, muladd(Δy_l, Δy_l, Δz_l * Δz_l))

                    Δx_r = max(node_r.lb_x - point_x, point_x - node_r.ub_x, zer)
                    Δy_r = max(node_r.lb_y - point_y, point_y - node_r.ub_y, zer)
                    Δz_r = max(node_r.lb_z - point_z, point_z - node_r.ub_z, zer)
                    dist²_r = muladd(Δx_r, Δx_r, muladd(Δy_r, Δy_r, Δz_r * Δz_r))
                end

                index_l = node_l.index
                child_or_size_l = node_l.child_or_size
                index_r = node_r.index
                child_or_size_r = node_r.child_or_size

                # sort so near child is traversed next (far child pushed to stack)
                swap = dist²_l > dist²_r
                dist²_near = ifelse(swap, dist²_r, dist²_l)
                dist²_far = ifelse(swap, dist²_l, dist²_r)
                idx_near = ifelse(swap, index_r, index_l)
                idx_far = ifelse(swap, index_l, index_r)
                cos_near = ifelse(swap, child_or_size_r, child_or_size_l)
                cos_far = ifelse(swap, child_or_size_l, child_or_size_r)

                if dist²_near < dist²_best
                    if dist²_far < dist²_best
                        stack_top += 1
                        @inbounds stack[stack_top] = NodeDist{Tg}(dist²_far, idx_far, cos_far)
                    end
                    node_dist = NodeDist{Tg}(dist²_near, idx_near, cos_near)
                    continue
                end
                break
            else  # leaf: test triangles
                leaf_start = node_dist.index
                leaf_end = -child_or_size
                @inbounds for idx_face in leaf_start:leaf_end
                    dist² = closest_dist²_triangle(point, tri_geometries[idx_face], dist²_best)
                    if dist² < dist²_best
                        dist²_best = dist²
                        tri_best = Int32(idx_face)
                    end
                end
                break
            end
        end
    end
    return (tri_best, dist²_best)
end

# compute hint grid: for each voxel center, store nearest triangle and signed distance.
# uses winding numbers for sign determination.
function compute_hint_grid(
    bvh::BoundingVolumeHierarchy{Tg},
    tri_geometries::Vector{TriangleGeometry{Tg}},
    fast_winding_data::FastWindingData{Tg},
    grid_res::Int
) where {Tg<:AbstractFloat}

    # queries are sampled from [-1, 1]³; pad slightly to cover boundary
    pad = Tg(0.05)
    lb = Tg(-1.0) - pad
    extent = Tg(2.0) + Tg(2.0) * pad

    inv_cell = Tg(grid_res) / extent
    cell_size = one(Tg) / inv_cell
    offset = lb - Tg(0.5) * cell_size

    cells = Array{Tuple{Int32,Tg},3}(undef, grid_res, grid_res, grid_res)
    stack_capacity = bvh.max_depth + 4

    Threads.@threads :dynamic for idx_z in 1:grid_res
        stack = Vector{NodeDist{Tg}}(undef, stack_capacity)
        wind_stack = Vector{Int32}(undef, stack_capacity)
        for idx_y in 1:grid_res
            hint_face = Int32(0)  # seed horizontal propagation
            for idx_x in 1:grid_res
                # cell-center coordinates via FMA
                center_x = muladd(Tg(idx_x), cell_size, offset)
                center_y = muladd(Tg(idx_y), cell_size, offset)
                center_z = muladd(Tg(idx_z), cell_size, offset)
                point_center = Point3{Tg}(center_x, center_y, center_z)

                # propagate previous triangle as hint to tighten initial BVH bound
                (tri_best, dist²_best) = closest_triangle_kernel(point_center, bvh, tri_geometries, stack, hint_face)
                hint_face = tri_best

                dist = √(max(dist²_best, zero(Tg)))

                @inbounds triangle = tri_geometries[tri_best]
                sgn_fast = face_interior_sign_or_zero(point_center, triangle)

                if !iszero(sgn_fast)
                    sgn_dist = sgn_fast * dist
                else
                    winding_number = winding_number_point_kernel(fast_winding_data.nodes, tri_geometries, point_center, wind_stack)
                    sgn_dist = copysign(dist, Tg(0.5) - Tg(winding_number))
                end

                cells[idx_x, idx_y, idx_z] = (tri_best, sgn_dist)
            end
        end
    end
    hint_grid = HintGrid{Tg}(lb, inv_cell, cell_size, offset, grid_res, cells)
    return hint_grid::HintGrid{Tg}
end

"""
    preprocess_mesh(mesh; leaf_capacity=8, β_wind=2.0, grid_res=128)

Build the acceleration structure for signed-distance queries on a
watertight, consistently-oriented triangle mesh.

- `mesh`:  `Mesh{3,Tg,GLTriangleFace}` — closed, watertight, consistently-oriented triangle mesh.

Returns a `SignedDistanceMesh{Tg}` ready for `compute_signed_distance!` calls.
"""
function preprocess_mesh(
    mesh::Mesh{3,Tg,GLTriangleFace}; leaf_capacity::Int=8, β_wind::Real=2.0, grid_res::Int=128
) where {Tg<:AbstractFloat}
    vertices = GeometryBasics.coordinates(mesh)
    tri_faces = GeometryBasics.faces(mesh)
    faces = NTuple{3,Int32}.(tri_faces)
    sdm = preprocess_mesh(vertices, faces; leaf_capacity, β_wind, grid_res)
    return sdm::SignedDistanceMesh{Tg}
end

function preprocess_mesh(
    vertices::Vector{Point3{Tg}},
    faces::Vector{NTuple{3,Int32}};
    leaf_capacity::Int=8, β_wind::Real=2.0, grid_res::Int=128
) where {Tg<:AbstractFloat}
    num_faces = length(faces)
    (num_faces > 0) || error("Mesh must contain at least one face.")

    # per-triangle bounding boxes and centroids
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
        (idx_v1, idx_v2, idx_v3) = faces[idx_face]
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

        normal_x = ab[2] * ac[3] - ab[3] * ac[2]
        normal_y = ab[3] * ac[1] - ab[1] * ac[3]
        normal_z = ab[1] * ac[2] - ab[2] * ac[1]
        denom_sum = normal_x * normal_x + normal_y * normal_y + normal_z * normal_z
        zer = zero(Tg)

        # guard for degenerate triangles to avoid NaN
        inv_denom = ifelse(denom_sum > floatmin(Tg), inv(denom_sum), zer)

        tri_geometries[j] = TriangleGeometry{Tg}(
            v1, ab, ac, Point3{Tg}(normal_x, normal_y, normal_z), d11, d12, d22, inv_denom
        )
    end

    fast_winding_data = precompute_fast_winding_data(bvh, tri_geometries; β=Tg(β_wind))
    hint_grid = compute_hint_grid(bvh, tri_geometries, fast_winding_data, grid_res)

    sdm = SignedDistanceMesh{Tg}(tri_geometries, bvh, face_to_packed, fast_winding_data, hint_grid)
    return sdm::SignedDistanceMesh{Tg}
end

################################   Fast Winding Number Precompute   ################################

const INV4PI64 = 1.0 / (4.0 * π)
const INV2PI64 = 1.0 / (2.0 * π)

function precompute_fast_winding_data(
    bvh::BoundingVolumeHierarchy{Tg},
    tri_geometries::Vector{TriangleGeometry{Tg}};
    β::Tg=Tg(2)
) where {Tg<:AbstractFloat}

    num_nodes = Int(bvh.num_nodes)
    fast_winding_nodes = Vector{FastWindingNode{Tg}}(undef, num_nodes)
    exact_radius = Vector{Float64}(undef, num_nodes)

    area_sum = Vector{Float64}(undef, num_nodes)
    centroid_sum_x = Vector{Float64}(undef, num_nodes)
    centroid_sum_y = Vector{Float64}(undef, num_nodes)
    centroid_sum_z = Vector{Float64}(undef, num_nodes)
    normal_sum_x = Vector{Float64}(undef, num_nodes)
    normal_sum_y = Vector{Float64}(undef, num_nodes)
    normal_sum_z = Vector{Float64}(undef, num_nodes)

    # bottom-up pass: deepest nodes first due to reverse iteration
    @inbounds for node_id in num_nodes:-1:1
        # sentinel node (index 2) — skip
        if node_id == 2
            fast_winding_nodes[2] = FastWindingNode{Tg}(
                zero(Tg), zero(Tg), zero(Tg),
                zero(Tg), zero(Tg), zero(Tg),
                floatmax(Tg), UInt32(0)
            )
            exact_radius[2] = 0.0
            continue
        end

        node = bvh.nodes[node_id]
        child_or_size = node.child_or_size

        if child_or_size < 0  # leaf: accumulate from triangles
            leaf_start = Int(node.index)
            leaf_end = Int(-child_or_size)
            area_sum_leaf = 0.0
            centroid_x_sum_leaf = 0.0
            centroid_y_sum_leaf = 0.0
            centroid_z_sum_leaf = 0.0
            normal_x_sum_leaf = 0.0
            normal_y_sum_leaf = 0.0
            normal_z_sum_leaf = 0.0

            @inbounds for idx in leaf_start:leaf_end
                triangle = tri_geometries[idx]
                normal_x = Float64(triangle.n[1])
                normal_y = Float64(triangle.n[2])
                normal_z = Float64(triangle.n[3])
                area_vec_x = 0.5 * normal_x
                area_vec_y = 0.5 * normal_y
                area_vec_z = 0.5 * normal_z
                area = 0.5 * √(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z)

                if area > 0.0
                    centroid_x = Float64(triangle.a[1]) + (Float64(triangle.ab[1]) + Float64(triangle.ac[1])) / 3.0
                    centroid_y = Float64(triangle.a[2]) + (Float64(triangle.ab[2]) + Float64(triangle.ac[2])) / 3.0
                    centroid_z = Float64(triangle.a[3]) + (Float64(triangle.ab[3]) + Float64(triangle.ac[3])) / 3.0

                    area_sum_leaf += area
                    centroid_x_sum_leaf += area * centroid_x
                    centroid_y_sum_leaf += area * centroid_y
                    centroid_z_sum_leaf += area * centroid_z
                end
                normal_x_sum_leaf += area_vec_x
                normal_y_sum_leaf += area_vec_y
                normal_z_sum_leaf += area_vec_z
            end

            area_sum[node_id] = area_sum_leaf
            centroid_sum_x[node_id] = centroid_x_sum_leaf
            centroid_sum_y[node_id] = centroid_y_sum_leaf
            centroid_sum_z[node_id] = centroid_z_sum_leaf
            normal_sum_x[node_id] = normal_x_sum_leaf
            normal_sum_y[node_id] = normal_y_sum_leaf
            normal_sum_z[node_id] = normal_z_sum_leaf
        else  # internal: merge children
            child_l = Int(node.index)
            child_r = Int(child_or_size)
            area_sum_internal = area_sum[child_l] + area_sum[child_r]
            area_sum[node_id] = area_sum_internal
            centroid_sum_x[node_id] = centroid_sum_x[child_l] + centroid_sum_x[child_r]
            centroid_sum_y[node_id] = centroid_sum_y[child_l] + centroid_sum_y[child_r]
            centroid_sum_z[node_id] = centroid_sum_z[child_l] + centroid_sum_z[child_r]
            normal_sum_x[node_id] = normal_sum_x[child_l] + normal_sum_x[child_r]
            normal_sum_y[node_id] = normal_sum_y[child_l] + normal_sum_y[child_r]
            normal_sum_z[node_id] = normal_sum_z[child_l] + normal_sum_z[child_r]
        end

        # area-weighted centroid (fallback to AABB center for zero-area nodes)
        area_sum_internal = area_sum[node_id]
        if area_sum_internal > 0.0
            centroid_x = centroid_sum_x[node_id] / area_sum_internal
            centroid_y = centroid_sum_y[node_id] / area_sum_internal
            centroid_z = centroid_sum_z[node_id] / area_sum_internal
        else
            centroid_x = 0.5 * (Float64(node.lb_x) + Float64(node.ub_x))
            centroid_y = 0.5 * (Float64(node.lb_y) + Float64(node.ub_y))
            centroid_z = 0.5 * (Float64(node.lb_z) + Float64(node.ub_z))
        end

        local topology::UInt32

        if child_or_size < 0  # leaf: bounding sphere from vertex distances
            leaf_start = Int(node.index)
            leaf_end = Int(-child_or_size)
            max_radius² = 0.0
            @inbounds for idx in leaf_start:leaf_end
                triangle = tri_geometries[idx]
                for v in (triangle.a, triangle.a + triangle.ab, triangle.a + triangle.ac)
                    Δx_val = Float64(v[1]) - centroid_x
                    Δy_val = Float64(v[2]) - centroid_y
                    Δz_val = Float64(v[3]) - centroid_z
                    dist²_val = Δx_val * Δx_val + Δy_val * Δy_val + Δz_val * Δz_val
                    max_radius² = max(max_radius², dist²_val)
                end
            end
            exact_radius[node_id] = √(max_radius²)

            count = UInt32(leaf_end - leaf_start + 1)
            topology = 0x80000000 | (count << 25) | UInt32(leaf_start)
        else  # internal: hierarchical bounding sphere
            child_l = Int(node.index)
            child_r = Int(child_or_size)

            node_l = fast_winding_nodes[child_l]
            node_r = fast_winding_nodes[child_r]

            Δx_l = Float64(node_l.centroid_x) - centroid_x
            Δy_l = Float64(node_l.centroid_y) - centroid_y
            Δz_l = Float64(node_l.centroid_z) - centroid_z
            radius_l = exact_radius[child_l] + √(Δx_l * Δx_l + Δy_l * Δy_l + Δz_l * Δz_l)

            Δx_r = Float64(node_r.centroid_x) - centroid_x
            Δy_r = Float64(node_r.centroid_y) - centroid_y
            Δz_r = Float64(node_r.centroid_z) - centroid_z
            radius_r = exact_radius[child_r] + √(Δx_r * Δx_r + Δy_r * Δy_r + Δz_r * Δz_r)

            hierarchical_r = max(radius_l, radius_r)

            # tighten with AABB-derived radius
            dx_aabb = max(abs(Float64(node.lb_x) - centroid_x), abs(Float64(node.ub_x) - centroid_x))
            dy_aabb = max(abs(Float64(node.lb_y) - centroid_y), abs(Float64(node.ub_y) - centroid_y))
            dz_aabb = max(abs(Float64(node.lb_z) - centroid_z), abs(Float64(node.ub_z) - centroid_z))
            aabb_r = √(dx_aabb * dx_aabb + dy_aabb * dy_aabb + dz_aabb * dz_aabb)

            exact_radius[node_id] = min(hierarchical_r, aabb_r)

            topology = UInt32(child_l)
        end

        exact_radius_node = exact_radius[node_id]
        beta_val = Float64(β)
        radius²_β² = (exact_radius_node * exact_radius_node) * (beta_val * beta_val)

        fast_winding_nodes[node_id] = FastWindingNode{Tg}(
            Tg(centroid_x), Tg(centroid_y), Tg(centroid_z),
            Tg(normal_sum_x[node_id] * INV4PI64),
            Tg(normal_sum_y[node_id] * INV4PI64),
            Tg(normal_sum_z[node_id] * INV4PI64),
            Tg(radius²_β²),
            topology
        )
    end

    fast_winding_data = FastWindingData{Tg}(fast_winding_nodes)
    return fast_winding_data::FastWindingData{Tg}
end

# allocate one traversal stack pair per chunk.
# each stack is tiny, so per-call allocation is negligible.
function allocate_stacks(sdm::SignedDistanceMesh{Tg}, num_points::Int) where {Tg}
    stack_capacity = Int(sdm.bvh.max_depth) + 4

    n_threads = Threads.nthreads()
    # enforce a minimum chunk size to prevent false sharing and guarantee cache locality
    min_chunk_size = 512
    num_chunks_max = max(1, num_points ÷ min_chunk_size)

    if num_chunks_max < n_threads
        # array is too small to use all threads optimally, limit the chunks
        num_chunks = num_chunks_max
    else  # array is large enough to target ideal 8× oversubscription
        factor_max = 8
        factor = min(factor_max, num_chunks_max ÷ n_threads)
        # set num_chunks to an exact multiple of n_threads for even load distribution
        num_chunks = n_threads * factor
    end

    stacks = [QueryStacks{Tg}(
        Vector{NodeDist{Tg}}(undef, stack_capacity),
        Vector{Int32}(undef, stack_capacity)
    ) for _ in 1:num_chunks]
    return stacks
end

##############################   High-Performance Hot Loop Routines   ##############################

# closest squared distance from point to a line segment defined by ap = point - a, ab = b - a
@inline function closest_dist²_segment(
    ap::Point3{Tg}, ab::Point3{Tg}, norm²_ab::Tg
) where {Tg<:AbstractFloat}
    zer = zero(Tg)
    # guard for degenerate (zero-length) edge
    (norm²_ab <= floatmin(Tg)) && return norm²(ap)::Tg
    @fastmath begin
        t_clamped = clamp((ap ⋅ ab) / norm²_ab, zer, one(Tg))
        Δ = ap - t_clamped * ab
        dist² = norm²(Δ)
        return dist²::Tg
    end
end

# closest squared distance from point to triangle with early-exit plane test.
# returns dist² ≥ dist²_best unchanged if the plane distance alone exceeds the current best.
@inline function closest_dist²_triangle(
    point::Point3{Tg}, triangle::TriangleGeometry{Tg}, dist²_best::Tg
) where {Tg<:AbstractFloat}
    @fastmath begin
        inv_denom = triangle.inv_denom
        ap = point - triangle.a

        # degenerate triangle: test all three edges
        if iszero(inv_denom)
            d2_ab = closest_dist²_segment(ap, triangle.ab, triangle.d11)
            d2_ac = closest_dist²_segment(ap, triangle.ac, triangle.d22)
            bc = triangle.ac - triangle.ab
            d2_bc = closest_dist²_segment(ap - triangle.ab, bc, norm²(bc))

            dist² = d2_ab
            dist² = ifelse(d2_ac < dist², d2_ac, dist²)
            dist² = ifelse(d2_bc < dist², d2_bc, dist²)
            return dist²::Tg
        end

        # early-exit: plane distance alone exceeds current best
        dot_n = triangle.n ⋅ ap
        plane_dist² = (dot_n * dot_n) * inv_denom
        (plane_dist² >= dist²_best) && return plane_dist²::Tg

        d1 = triangle.ab ⋅ ap
        d2 = triangle.ac ⋅ ap

        # vertex a region
        if (d1 <= 0) && (d2 <= 0)
            dist² = norm²(ap)
            return dist²::Tg
        end

        d11 = triangle.d11
        d12 = triangle.d12
        d22 = triangle.d22

        # vertex b region
        d3 = d1 - d11
        d4 = d2 - d12
        if (d3 >= 0) && (d4 <= d3)
            Δ = ap - triangle.ab
            dist² = norm²(Δ)
            return dist²::Tg
        end

        # edge ab region
        vc = muladd(d11, d2, -d12 * d1)
        if (vc <= 0) && (d1 >= 0) && (d3 <= 0)
            zer = zero(Tg)
            v = ifelse(d11 > floatmin(Tg), d1 / d11, zer)
            Δ = ap - v * triangle.ab
            dist² = norm²(Δ)
            return dist²::Tg
        end

        # vertex c region
        d5 = d1 - d12
        d6 = d2 - d22
        if (d6 >= 0) && (d5 <= d6)
            Δ = ap - triangle.ac
            dist² = norm²(Δ)
            return dist²::Tg
        end

        # edge ac region
        vb = muladd(d22, d1, -d12 * d2)
        if (vb <= 0) && (d2 >= 0) && (d6 <= 0)
            zer = zero(Tg)
            w = ifelse(d22 > floatmin(Tg), d2 / d22, zer)
            Δ = ap - w * triangle.ac
            dist² = norm²(Δ)
            return dist²::Tg
        end

        # edge bc region or face interior
        uno = one(Tg)
        v_bary = (vb + vc) * inv_denom
        if v_bary >= uno
            d43 = d11 - d12 - d1 + d2
            zer = zero(Tg)
            d33 = max(zer, d11 + d22 - Tg(2) * d12)

            w = clamp(ifelse(d33 > floatmin(Tg), d43 / d33, zer), zer, uno)
            bp = ap - triangle.ab
            Δ = bp - w * (triangle.ac - triangle.ab)
            dist² = norm²(Δ)
            return dist²::Tg
        else
            # face interior: plane distance is exact
            dist² = plane_dist²
            return dist²::Tg
        end
    end
end

###############################   Fast Face-Interior Sign Shortcut   ###############################

# returns +1 / -1 if the closest point is strictly inside the face interior,
# zero otherwise (edge/vertex proximity — requires full winding number).
@inline function face_interior_sign_or_zero(
    point::Point3{Tg}, triangle::TriangleGeometry{Tg}, bary_tol::Tg=Tg(1e-4)
) where {Tg<:AbstractFloat}
    inv_denom = triangle.inv_denom
    iszero(inv_denom) && return zero(Tg)

    @fastmath begin
        ap = point - triangle.a

        d1 = triangle.ab ⋅ ap
        d2 = triangle.ac ⋅ ap

        v = muladd(triangle.d22, d1, -triangle.d12 * d2) * inv_denom
        w = muladd(triangle.d11, d2, -triangle.d12 * d1) * inv_denom

        if (v <= bary_tol) || (w <= bary_tol) || ((one(Tg) - v - w) <= bary_tol)
            return zero(Tg)
        end

        # reuse precomputed face normal to determine sign
        sgn = ifelse((triangle.n ⋅ ap) >= zero(Tg), one(Tg), -one(Tg))
        return sgn::Tg
    end
end

###################################   Fast Winding Number Query   ##################################

# solid angle subtended by a triangle, scaled by 1/(2π), computed in Float64.
# uses the atan2-based formula for numerical stability.
@inline function solid_angle_scaled(point_f64::NTuple{3,Float64}, triangle::TriangleGeometry{Tg}) where {Tg<:AbstractFloat}
    @fastmath begin
        (point_x, point_y, point_z) = point_f64

        vec_a_x = Float64(triangle.a[1]) - point_x
        vec_a_y = Float64(triangle.a[2]) - point_y
        vec_a_z = Float64(triangle.a[3]) - point_z
        vec_b_x = vec_a_x + Float64(triangle.ab[1])
        vec_b_y = vec_a_y + Float64(triangle.ab[2])
        vec_b_z = vec_a_z + Float64(triangle.ab[3])
        vec_c_x = vec_a_x + Float64(triangle.ac[1])
        vec_c_y = vec_a_y + Float64(triangle.ac[2])
        vec_c_z = vec_a_z + Float64(triangle.ac[3])

        det = vec_a_x * Float64(triangle.n[1]) + vec_a_y * Float64(triangle.n[2]) + vec_a_z * Float64(triangle.n[3])

        norm_a = √(vec_a_x * vec_a_x + vec_a_y * vec_a_y + vec_a_z * vec_a_z)
        norm_b = √(vec_b_x * vec_b_x + vec_b_y * vec_b_y + vec_b_z * vec_b_z)
        norm_c = √(vec_c_x * vec_c_x + vec_c_y * vec_c_y + vec_c_z * vec_c_z)

        ε = 1.0e-30
        (norm_a < ε || norm_b < ε || norm_c < ε) && return 0.0

        dot_ab = vec_a_x * vec_b_x + vec_a_y * vec_b_y + vec_a_z * vec_b_z
        dot_ac = vec_a_x * vec_c_x + vec_a_y * vec_c_y + vec_a_z * vec_c_z
        dot_bc = vec_b_x * vec_c_x + vec_b_y * vec_c_y + vec_b_z * vec_c_z
        denom = norm_a * norm_b * norm_c + dot_ab * norm_c + dot_ac * norm_b + dot_bc * norm_a

        ω = atan(det, denom) * INV2PI64
        return ω::Float64
    end
end

# fast winding number via multipole acceptance criterion (MAC).
# returns the winding number in Float64 (inside ≈ 1, outside ≈ 0).
@inline function winding_number_point_kernel(
    fast_winding_nodes::Vector{FastWindingNode{Tg}},
    tri_geometries::Vector{TriangleGeometry{Tg}},
    point::Point3{Tg},
    stack::Vector{Int32}
) where {Tg<:AbstractFloat}

    point_x = Float64(point[1])
    point_y = Float64(point[2])
    point_z = Float64(point[3])
    point_f64 = (point_x, point_y, point_z)
    wn = 0.0

    # evaluate root node MAC before entering the loop
    @inbounds root_node = fast_winding_nodes[1]
    @fastmath begin
        rx_root = Float64(root_node.centroid_x) - point_x
        ry_root = Float64(root_node.centroid_y) - point_y
        rz_root = Float64(root_node.centroid_z) - point_z
        dist2_root = rx_root * rx_root + ry_root * ry_root + rz_root * rz_root

        if dist2_root > Float64(root_node.radius²_β²)
            dot_val_root = rx_root * Float64(root_node.normal_x) + ry_root * Float64(root_node.normal_y) + rz_root * Float64(root_node.normal_z)
            inv_dist_root = inv(√(dist2_root))
            wn_val = dot_val_root * (inv_dist_root * inv_dist_root * inv_dist_root)
            return wn_val::Float64
        end
    end

    stack_top = 1
    @inbounds stack[1] = Int32(1)

    @inbounds while stack_top > 0
        node_id = Int(stack[stack_top])
        stack_top -= 1

        while true
            fnode = fast_winding_nodes[node_id]
            topology = fnode.topology

            if !iszero(topology & 0x80000000)  # leaf: exact solid angles
                count = (topology >> 25) & 0x3F
                leaf_start = topology & 0x01FFFFFF
                leaf_end = leaf_start + count - 1
                for idx_face in leaf_start:leaf_end
                    wn += solid_angle_scaled(point_f64, tri_geometries[idx_face])
                end
                break
            else  # internal: MAC test on both children
                child_l_id = Int(topology)
                child_r_id = child_l_id + 1

                node_l = fast_winding_nodes[child_l_id]
                node_r = fast_winding_nodes[child_r_id]

                pass_l = false
                pass_r = false

                @fastmath begin
                    # MAC left child
                    rx_l = Float64(node_l.centroid_x) - point_x
                    ry_l = Float64(node_l.centroid_y) - point_y
                    rz_l = Float64(node_l.centroid_z) - point_z
                    dist2_l = muladd(rx_l, rx_l, muladd(ry_l, ry_l, rz_l * rz_l))
                    pass_l = dist2_l > Float64(node_l.radius²_β²)

                    if pass_l
                        dot_val_l = rx_l * Float64(node_l.normal_x) + ry_l * Float64(node_l.normal_y) + rz_l * Float64(node_l.normal_z)
                        inv_dist_l = inv(√(dist2_l))
                        wn += dot_val_l * (inv_dist_l * inv_dist_l * inv_dist_l)
                    end

                    # MAC right child
                    rx_r = Float64(node_r.centroid_x) - point_x
                    ry_r = Float64(node_r.centroid_y) - point_y
                    rz_r = Float64(node_r.centroid_z) - point_z
                    dist2_r = muladd(rx_r, rx_r, muladd(ry_r, ry_r, rz_r * rz_r))
                    pass_r = dist2_r > Float64(node_r.radius²_β²)

                    if pass_r
                        dot_val_r = rx_r * Float64(node_r.normal_x) + ry_r * Float64(node_r.normal_y) + rz_r * Float64(node_r.normal_z)
                        inv_dist_r = inv(√(dist2_r))
                        wn += dot_val_r * (inv_dist_r * inv_dist_r * inv_dist_r)
                    end
                end

                # descend into children that failed MAC
                if (!pass_l) && (!pass_r)
                    stack_top += 1
                    stack[stack_top] = Int32(child_r_id)
                    node_id = child_l_id
                    continue
                elseif !pass_l
                    node_id = child_l_id
                    continue
                elseif !pass_r
                    node_id = child_r_id
                    continue
                else  # both passed MAC — no further traversal needed
                    break
                end
            end
        end
    end

    return wn::Float64
end

######################################   Single-Point Query   ######################################

@inline function get_grid_hint(grid::HintGrid{Tg}, point::Point3{Tg}) where {Tg<:AbstractFloat}
    inv_cell = grid.inv_cell
    lb = grid.lb

    x = (point[1] - lb) * inv_cell
    y = (point[2] - lb) * inv_cell
    z = (point[3] - lb) * inv_cell

    res = grid.res
    @fastmath begin
        x_clamped = clamp(x, zero(Tg), Tg(res - 1))
        y_clamped = clamp(y, zero(Tg), Tg(res - 1))
        z_clamped = clamp(z, zero(Tg), Tg(res - 1))

        idx_x = Base.unsafe_trunc(Int, x_clamped) + 1
        idx_y = Base.unsafe_trunc(Int, y_clamped) + 1
        idx_z = Base.unsafe_trunc(Int, z_clamped) + 1
    end

    # cell-center coordinates via FMA
    center_x = muladd(Tg(idx_x), grid.cell_size, grid.offset)
    center_y = muladd(Tg(idx_y), grid.cell_size, grid.offset)
    center_z = muladd(Tg(idx_z), grid.cell_size, grid.offset)

    Δx = point[1] - center_x
    Δy = point[2] - center_y
    Δz = point[3] - center_z
    dist_to_center² = muladd(Δx, Δx, muladd(Δy, Δy, Δz * Δz))

    @inbounds cell = grid.cells[idx_x, idx_y, idx_z]
    (idx_face_nearest, signed_distance_at_center) = cell
    return (idx_face_nearest::Int32, signed_distance_at_center::Tg, dist_to_center²::Tg)
end

function signed_distance_point(
    sdm::SignedDistanceMesh{Tg}, point::Point3{Tg}, hint_face::Int32, stacks::QueryStacks{Tg}
) where {Tg<:AbstractFloat}
    dist²_best = floatmax(Tg)
    tri_best = Int32(0)

    # tighten initial bound using the provided triangle hint
    if (hint_face > 0) && (hint_face <= length(sdm.tri_geometries))
        tri_best = hint_face
        @inbounds triangle = sdm.tri_geometries[tri_best]
        dist²_best = closest_dist²_triangle(point, triangle, dist²_best)
        (dist²_best <= zero(Tg)) && return zero(Tg)
    end

    (tri_grid, sd_center, dist_to_center²) = get_grid_hint(sdm.hint_grid, point)

    # further tighten with grid hint if it differs from the face hint
    if (tri_grid > 0) && (tri_grid != tri_best)
        @inbounds triangle = sdm.tri_geometries[tri_grid]
        dist² = closest_dist²_triangle(point, triangle, dist²_best)
        if dist² <= zero(Tg)
            return zero(Tg)
        end
        if dist² < dist²_best
            dist²_best = dist²
            tri_best = tri_grid
        end
    end

    signed_distance = signed_distance_point_kernel(
        sdm, point, dist²_best, tri_best, sd_center, dist_to_center², stacks
    )
    return signed_distance::Tg
end

function signed_distance_point(
    sdm::SignedDistanceMesh{Tg}, point::Point3{Tg}, stacks::QueryStacks{Tg}
) where {Tg<:AbstractFloat}
    (tri_grid, sd_center, dist_to_center²) = get_grid_hint(sdm.hint_grid, point)

    dist²_best = floatmax(Tg)
    tri_best = Int32(0)

    if (tri_grid > 0) && (tri_grid <= length(sdm.tri_geometries))
        tri_best = tri_grid
        @inbounds triangle = sdm.tri_geometries[tri_best]
        dist²_best = closest_dist²_triangle(point, triangle, dist²_best)
        (dist²_best <= zero(Tg)) && return zero(Tg)
    end

    signed_distance = signed_distance_point_kernel(
        sdm, point, dist²_best, tri_best, sd_center, dist_to_center², stacks
    )
    return signed_distance::Tg
end

# core kernel: BVH traversal → sign determination (face-interior shortcut → Eikonal → winding number)
function signed_distance_point_kernel(
    sdm::SignedDistanceMesh{Tg},
    point::Point3{Tg},
    dist²_best_in::Tg,
    tri_best_in::Int32,
    sd_center::Tg,
    dist_to_center²::Tg,
    stacks::QueryStacks{Tg}
) where {Tg<:AbstractFloat}

    (tri_best, dist²_best) = closest_triangle_kernel(
        point, sdm.bvh, sdm.tri_geometries, stacks.dist,
        tri_best_in, dist²_best_in
    )

    zer = zero(Tg)
    iszero(tri_best) && error("No triangle found for point $point")
    dist²_best = max(dist²_best, zer)
    dist = √(dist²_best)
    iszero(dist) && return zero(Tg)

    @inbounds triangle = sdm.tri_geometries[tri_best]

    # fast face-interior sign bypass (avoids winding number when clearly inside a face)
    sgn_fast = face_interior_sign_or_zero(point, triangle)
    if !iszero(sgn_fast)
        signed_distance = sgn_fast * dist
        return signed_distance::Tg
    end

    # Eikonal early-out: if the triangle inequality with the grid center holds,
    # the sign must match the cached grid center sign
    abs_sd = abs(sd_center)
    sum_dist = dist + abs_sd

    if (sum_dist * sum_dist) > dist_to_center² * Tg(1.00001)
        signed_distance = copysign(dist, sd_center)
        return signed_distance::Tg
    end

    # fallback: full winding number for sign determination
    wn = winding_number_point_kernel(sdm.fast_winding_data.nodes, sdm.tri_geometries, point, stacks.wind)

    signed_distance = copysign(dist, Tg(0.5) - Tg(wn))
    return signed_distance::Tg
end

##########################################   Public API   ##########################################

"""
    compute_signed_distance!(out, sdm, points, hint_faces)

In-place batch signed distance query with triangle hints. Writes results into `out`.
- `out`:            length-n vector to store the output signed distances.
- `sdm`:            a [`SignedDistanceMesh`] built once via `preprocess_mesh`.
- `points`:         `3 × n` matrix of query points.
- `hint_faces`:     length-n vector of *original* face indices (1-based, matching the input `faces`)
                    for each point. A single exact triangle check tightens the upper
                    bound before BVH traversal, which can substantially speed up near-surface queries.

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
            idx_face_packed = Int32(0)
            if (idx_face > 0) && (idx_face <= length(face_to_packed))
                @inbounds idx_face_packed = face_to_packed[idx_face]
            end
            @inbounds point = Point3{Tg}(points[1, idx], points[2, idx], points[3, idx])
            @inbounds out[idx] = signed_distance_point(sdm, point, idx_face_packed, stack)
        end
    end
    return out
end

"""
    compute_signed_distance!(out, sdm, points)

In-place batch signed distance query without triangle hints. Writes results into `out`.
- `out`:        length-n vector to store the output signed distances.
- `sdm`:        a [`SignedDistanceMesh`] built once via `preprocess_mesh`.
- `points`:     `3 × n` matrix of query points.

Positive = outside, negative = inside.
"""
function compute_signed_distance!(
    out::AbstractVector{Tg}, sdm::SignedDistanceMesh{Tg}, points::StridedMatrix{Tg}
) where {Tg<:AbstractFloat}
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

###################################   Optional Reusable Scratch   ##################################

"""
    compute_signed_distance!(out, sdm, points, hint_faces, stacks)

In-place batch signed distance query with pre-allocated stacks.
Identical to the two-argument variant but reuses `stacks` across calls to avoid allocation.
"""
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
            idx_face_packed = Int32(0)
            if (idx_face > 0) && (idx_face <= length(face_to_packed))
                @inbounds idx_face_packed = face_to_packed[idx_face]
            end
            @inbounds point = Point3{Tg}(points[1, idx], points[2, idx], points[3, idx])
            @inbounds out[idx] = signed_distance_point(sdm, point, idx_face_packed, stack)
        end
    end
    return out
end

"""
    compute_signed_distance!(out, sdm, points, stacks)

In-place batch signed distance query with pre-allocated stacks and no triangle hints.
"""
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
