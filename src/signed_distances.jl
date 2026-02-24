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

###########################################   Utilities   ##########################################

@inline function norm²(point::Point3{T}) where {T<:AbstractFloat}
    n² = point ⋅ point
    return n²::T
end

#######################################   Data Structures   #######################################

# packed triangle geometry aligned to 64 bytes (16 floats) for L1 cache-line
struct TriangleGeometry{T<:AbstractFloat}
    a::Point3{T}
    ab::Point3{T}
    ac::Point3{T}
    n::Point3{T}     # unnormalized face normal: ab × ac
    d11::T           # norm²(ab)
    d12::T           # ab ⋅ ac
    d22::T           # norm²(ac)
    inv_denom::T     # 1.0 / norm²(n)
end

# 0th-order dipole expansion for fast winding numbers.
# Topology packed natively to avoid redundant BVH fetches.
struct FWNNode{T<:AbstractFloat}
    cm_x::T
    cm_y::T
    cm_z::T
    n_x::T
    n_y::T
    n_z::T
    r²β²::T
    topology::UInt32 # High-bit: leaf flag. Bits 25-30: count. Bits 0-24: child_l / leaf_start
end

struct FastWindingData{T<:AbstractFloat}
    nodes::Vector{FWNNode{T}}
end

# AoS BVH node with overlapped integer fields for cache efficiency.
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

# Isotropic spatial voxel grid tailored to [-1, 1]^3.
# Features exact cell-center evaluations for Eikonal Early-Out bypass.
# Unified cells into an 8-byte Tuple block for single-instruction L1 Cache fetching.
struct HintGrid{T<:AbstractFloat}
    lb::T
    inv_cell::T
    res::Int
    cells::Array{Tuple{Int32,T},3}
end

# Stack element carrying node topology.
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
    tri_geometries::Vector{TriangleGeometry{Tg}}
    bvh::BoundingVolumeHierarchy{Tg}
    face_to_packed::Vector{Int32}
    fwn::FastWindingData{Tg}
    hint_grid::HintGrid{Tg}
end

#######################################   BVH Construction   #######################################

function median_split_sort!(
    indices::Vector{Int32}, lo::Int, mid::Int, hi::Int, centroids::NTuple{3,Vector{Tg}}, axis::Int
) where {Tg<:AbstractFloat}
    sub_indices = @view indices[lo:hi]
    centroids_axis = centroids[axis]
    mid_relative = mid - lo + 1
    partialsort!(sub_indices, mid_relative; by=tri_idx -> centroids_axis[tri_idx])
    return nothing
end

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

const SAH_BINS = 16

function SAHScratch{T}() where {T<:AbstractFloat}
    nb = SAH_BINS
    count = Vector{Int32}(undef, nb)
    lb_x = Vector{T}(undef, nb)
    lb_y = Vector{T}(undef, nb)
    lb_z = Vector{T}(undef, nb)
    ub_x = Vector{T}(undef, nb)
    ub_y = Vector{T}(undef, nb)
    ub_z = Vector{T}(undef, nb)

    suffix_count = Vector{Int32}(undef, nb + 1)
    suffix_lb_x = Vector{T}(undef, nb + 1)
    suffix_lb_y = Vector{T}(undef, nb + 1)
    suffix_lb_z = Vector{T}(undef, nb + 1)
    suffix_ub_x = Vector{T}(undef, nb + 1)
    suffix_ub_y = Vector{T}(undef, nb + 1)
    suffix_ub_z = Vector{T}(undef, nb + 1)

    return SAHScratch{T}(
        count, lb_x, lb_y, lb_z, ub_x, ub_y, ub_z,
        suffix_count, suffix_lb_x, suffix_lb_y, suffix_lb_z,
        suffix_ub_x, suffix_ub_y, suffix_ub_z
    )
end

@inline function reset_sah_bins!(scratch::SAHScratch{T}) where {T<:AbstractFloat}
    nb = SAH_BINS
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

@inline function centroid_bin_index(c::T, cmin::T, inv_extent::T) where {T<:AbstractFloat}
    x = (c - cmin) * inv_extent
    b = Int(floor(x))
    nbm1 = SAH_BINS - 1
    b = ifelse(b < 0, 0, ifelse(b > nbm1, nbm1, b))
    return (b + 1)::Int
end

@inline function aabb_half_area(lb_x::T, lb_y::T, lb_z::T, ub_x::T, ub_y::T, ub_z::T) where {T<:AbstractFloat}
    zer = zero(T)
    dx = max(ub_x - lb_x, zer)
    dy = max(ub_y - lb_y, zer)
    dz = max(ub_z - lb_z, zer)
    return muladd(dx, dy, muladd(dx, dz, dy * dz))::T
end

@inline function sah_best_split_axis!(
    scratch::SAHScratch{T},
    tri_indices::Vector{Int32}, lo::Int, hi::Int,
    centroids_axis::Vector{T}, cmin::T, cmax::T,
    lb_x_t::Vector{T}, lb_y_t::Vector{T}, lb_z_t::Vector{T},
    ub_x_t::Vector{T}, ub_y_t::Vector{T}, ub_z_t::Vector{T}
) where {T<:AbstractFloat}
    extent = cmax - cmin
    (extent > zero(T)) || return (floatmax(T), 0, zero(T))

    nb = SAH_BINS
    inv_extent = T(nb) / extent
    reset_sah_bins!(scratch)

    @inbounds for i in lo:hi
        tri = tri_indices[i]
        b = centroid_bin_index(centroids_axis[tri], cmin, inv_extent)

        scratch.count[b] += Int32(1)
        scratch.lb_x[b] = min(scratch.lb_x[b], lb_x_t[tri])
        scratch.lb_y[b] = min(scratch.lb_y[b], lb_y_t[tri])
        scratch.lb_z[b] = min(scratch.lb_z[b], lb_z_t[tri])
        scratch.ub_x[b] = max(scratch.ub_x[b], ub_x_t[tri])
        scratch.ub_y[b] = max(scratch.ub_y[b], ub_y_t[tri])
        scratch.ub_z[b] = max(scratch.ub_z[b], ub_z_t[tri])
    end

    inf_val = floatmax(T)
    ninf_val = -floatmax(T)
    scratch.suffix_count[nb+1] = Int32(0)
    scratch.suffix_lb_x[nb+1] = inf_val
    scratch.suffix_lb_y[nb+1] = inf_val
    scratch.suffix_lb_z[nb+1] = inf_val
    scratch.suffix_ub_x[nb+1] = ninf_val
    scratch.suffix_ub_y[nb+1] = ninf_val
    scratch.suffix_ub_z[nb+1] = ninf_val

    @inbounds for b in nb:-1:1
        scratch.suffix_count[b] = scratch.suffix_count[b+1] + scratch.count[b]
        scratch.suffix_lb_x[b] = min(scratch.lb_x[b], scratch.suffix_lb_x[b+1])
        scratch.suffix_lb_y[b] = min(scratch.lb_y[b], scratch.suffix_lb_y[b+1])
        scratch.suffix_lb_z[b] = min(scratch.lb_z[b], scratch.suffix_lb_z[b+1])
        scratch.suffix_ub_x[b] = max(scratch.ub_x[b], scratch.suffix_ub_x[b+1])
        scratch.suffix_ub_y[b] = max(scratch.ub_y[b], scratch.suffix_ub_y[b+1])
        scratch.suffix_ub_z[b] = max(scratch.ub_z[b], scratch.suffix_ub_z[b+1])
    end

    left_count = Int32(0)
    left_lb_x = inf_val
    left_lb_y = inf_val
    left_lb_z = inf_val
    left_ub_x = ninf_val
    left_ub_y = ninf_val
    left_ub_z = ninf_val

    best_cost = floatmax(T)
    best_split = 0

    @inbounds for b in 1:(nb-1)
        ccount = scratch.count[b]
        if ccount != 0
            left_count += ccount
            left_lb_x = min(left_lb_x, scratch.lb_x[b])
            left_lb_y = min(left_lb_y, scratch.lb_y[b])
            left_lb_z = min(left_lb_z, scratch.lb_z[b])
            left_ub_x = max(left_ub_x, scratch.ub_x[b])
            left_ub_y = max(left_ub_y, scratch.ub_y[b])
            left_ub_z = max(left_ub_z, scratch.ub_z[b])
        end

        right_count = scratch.suffix_count[b+1]
        if (left_count > 0) && (right_count > 0)
            area_l = aabb_half_area(left_lb_x, left_lb_y, left_lb_z, left_ub_x, left_ub_y, left_ub_z)
            area_r = aabb_half_area(
                scratch.suffix_lb_x[b+1], scratch.suffix_lb_y[b+1], scratch.suffix_lb_z[b+1],
                scratch.suffix_ub_x[b+1], scratch.suffix_ub_y[b+1], scratch.suffix_ub_z[b+1]
            )
            cost = area_l * T(left_count) + area_r * T(right_count)
            if cost < best_cost
                best_cost = cost
                best_split = b
            end
        end
    end

    return (best_cost::T, best_split::Int, inv_extent::T)
end

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

    centroid_min_x = floatmax(T)
    centroid_min_y = floatmax(T)
    centroid_min_z = floatmax(T)
    centroid_max_x = -floatmax(T)
    centroid_max_y = -floatmax(T)
    centroid_max_z = -floatmax(T)
    @inbounds for i in lo:hi
        t = tri_indices[i]
        centroid_min_x = min(centroid_min_x, centroids[1][t])
        centroid_min_y = min(centroid_min_y, centroids[2][t])
        centroid_min_z = min(centroid_min_z, centroids[3][t])
        centroid_max_x = max(centroid_max_x, centroids[1][t])
        centroid_max_y = max(centroid_max_y, centroids[2][t])
        centroid_max_z = max(centroid_max_z, centroids[3][t])
    end

    scratch = builder.sah_scratch
    best_cost = floatmax(T)
    best_axis = 0
    best_split = 0
    best_inv_extent = zero(T)
    best_cmin = zero(T)

    (cost_x, split_x, inv_ext_x) = sah_best_split_axis!(
        scratch, tri_indices, lo, hi, centroids[1], centroid_min_x, centroid_max_x,
        lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t
    )
    if (split_x != 0) && (cost_x < best_cost)
        best_cost = cost_x
        best_axis = 1
        best_split = split_x
        best_inv_extent = inv_ext_x
        best_cmin = centroid_min_x
    end

    (cost_y, split_y, inv_ext_y) = sah_best_split_axis!(
        scratch, tri_indices, lo, hi, centroids[2], centroid_min_y, centroid_max_y,
        lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t
    )
    if (split_y != 0) && (cost_y < best_cost)
        best_cost = cost_y
        best_axis = 2
        best_split = split_y
        best_inv_extent = inv_ext_y
        best_cmin = centroid_min_y
    end

    (cost_z, split_z, inv_ext_z) = sah_best_split_axis!(
        scratch, tri_indices, lo, hi, centroids[3], centroid_min_z, centroid_max_z,
        lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t
    )
    if (split_z != 0) && (cost_z < best_cost)
        best_cost = cost_z
        best_axis = 3
        best_split = split_z
        best_inv_extent = inv_ext_z
        best_cmin = centroid_min_z
    end

    mid = 0
    if best_axis != 0
        axis = best_axis
        i = lo
        j = hi
        @inbounds while i <= j
            t = tri_indices[i]
            b = centroid_bin_index(centroids[axis][t], best_cmin, best_inv_extent)
            if b <= best_split
                i += 1
            else
                tri_indices[i], tri_indices[j] = tri_indices[j], tri_indices[i]
                j -= 1
            end
        end
        mid = j
    end

    if (best_axis == 0) || (mid < lo) || (mid >= hi)
        spread_x = centroid_max_x - centroid_min_x
        spread_y = centroid_max_y - centroid_min_y
        spread_z = centroid_max_z - centroid_min_z
        (spread_max, axis) = findmax((spread_x, spread_y, spread_z))
        mid = (lo + hi) >>> 1
        (spread_max > 0) && median_split_sort!(tri_indices, lo, mid, hi, centroids, axis)
    end

    child_l = builder.next_node
    child_r = builder.next_node + Int32(1)
    builder.next_node += Int32(2)

    build_node!(builder, child_l, tri_indices, lo, mid, centroids, lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t, depth + 1)
    build_node!(builder, child_r, tri_indices, mid + 1, hi, centroids, lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t, depth + 1)

    @inbounds begin
        node_l = builder.nodes[child_l]
        node_r = builder.nodes[child_r]
        builder.nodes[node_id] = BVHNode{T}(
            min(node_l.lb_x, node_r.lb_x), min(node_l.lb_y, node_r.lb_y), min(node_l.lb_z, node_r.lb_z),
            max(node_l.ub_x, node_r.ub_x), max(node_l.ub_y, node_r.ub_y), max(node_l.ub_z, node_r.ub_z),
            child_l, child_r
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

    # 🌟 5.B: Topology limit cap for robust 25-bit shift limits
    # Steals an extra bit to map leaf_start indexing safely up to 33.5 million faces.
    leaf_capacity = min(leaf_capacity, 63)
    num_faces = length(first(centroids))
    @assert num_faces <= 33554431 "Mesh exceeds 33.5M faces."

    tri_indices = Int32.(1:num_faces)
    max_nodes = 2 * num_faces + 1

    nodes = Vector{BVHNode{Tg}}(undef, max_nodes)
    sah_scratch = SAHScratch{Tg}()
    builder = BVHBuilder{Tg}(nodes, Int32(leaf_capacity), sah_scratch, Int32(3), Int32(1))

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
    return (bvh, tri_indices)
end

######################################   Mesh Preprocessing   ######################################

function closest_triangle_kernel(
    point::Point3{Tg},
    bvh::BoundingVolumeHierarchy{Tg},
    tri_geometries::Vector{TriangleGeometry{Tg}},
    stack::Vector{NodeDist{Tg}},
    hint_tri::Int32=Int32(0) # 🌟 2: Precomputation Hint Propagation Seed
) where {Tg<:AbstractFloat}
    dist²_best = floatmax(Tg)
    tri_best = hint_tri > 0 ? hint_tri : Int32(1)
    feat_best = FEATURE_DEGENERATE
    zer = zero(Tg)

    if hint_tri > 0
        @inbounds triangle = tri_geometries[hint_tri]
        (dist²_best, feat_best) = closest_dist²_triangle_feature(point, triangle, dist²_best)
    end

    (p_x, p_y, p_z) = point
    stack_top = 1
    @inbounds root_node = bvh.nodes[1]
    @inbounds stack[1] = NodeDist{Tg}(zer, root_node.index, root_node.child_or_size)

    @inbounds while stack_top > 0
        node_dist = stack[stack_top]
        stack_top -= 1

        while true
            (node_dist.dist² >= dist²_best) && break
            child_or_size = node_dist.child_or_size

            if child_or_size > 0
                child_l_id = node_dist.index
                child_r_id = child_or_size

                node_l = bvh.nodes[child_l_id]
                node_r = bvh.nodes[child_r_id]

                @fastmath begin
                    Δx_l = max(node_l.lb_x - p_x, p_x - node_l.ub_x, zer)
                    Δy_l = max(node_l.lb_y - p_y, p_y - node_l.ub_y, zer)
                    Δz_l = max(node_l.lb_z - p_z, p_z - node_l.ub_z, zer)
                    dist²_l = muladd(Δx_l, Δx_l, muladd(Δy_l, Δy_l, Δz_l * Δz_l))

                    Δx_r = max(node_r.lb_x - p_x, p_x - node_r.ub_x, zer)
                    Δy_r = max(node_r.lb_y - p_y, p_y - node_r.ub_y, zer)
                    Δz_r = max(node_r.lb_z - p_z, p_z - node_r.ub_z, zer)
                    dist²_r = muladd(Δx_r, Δx_r, muladd(Δy_r, Δy_r, Δz_r * Δz_r))
                end

                idx_l = node_l.index
                cos_l = node_l.child_or_size
                idx_r = node_r.index
                cos_r = node_r.child_or_size

                swap = dist²_l > dist²_r
                near_dist² = ifelse(swap, dist²_r, dist²_l)
                far_dist² = ifelse(swap, dist²_l, dist²_r)
                near_idx = ifelse(swap, idx_r, idx_l)
                far_idx = ifelse(swap, idx_l, idx_r)
                near_cos = ifelse(swap, cos_r, cos_l)
                far_cos = ifelse(swap, cos_l, cos_r)

                if near_dist² < dist²_best
                    if far_dist² < dist²_best
                        stack_top += 1
                        @inbounds stack[stack_top] = NodeDist{Tg}(far_dist², far_idx, far_cos)
                    end
                    node_dist = NodeDist{Tg}(near_dist², near_idx, near_cos)
                    continue
                end
                break
            else
                leaf_start = node_dist.index
                leaf_end = -child_or_size
                @inbounds for idx in leaf_start:leaf_end
                    (dist², feat) = closest_dist²_triangle_feature(point, tri_geometries[idx], dist²_best)
                    if dist² < dist²_best
                        dist²_best = dist²
                        tri_best = Int32(idx)
                        feat_best = feat
                    end
                end
                break
            end
        end
    end
    return (tri_best::Int32, dist²_best::Tg, feat_best::UInt8)
end

function compute_hint_grid(
    bvh::BoundingVolumeHierarchy{Tg},
    tri_geometries::Vector{TriangleGeometry{Tg}},
    fwn::FastWindingData{Tg},
    grid_res::Int
) where {Tg<:AbstractFloat}

    # -----------------------------------------------------------------------------------
    # Because queries are mathematically guaranteed to be sampled strictly from within
    # the unit cube [-1, 1]^3, we decouple the Hint Grid from the dynamic mesh AABB
    # and lock it to perfectly partition this exact query domain.
    # -----------------------------------------------------------------------------------
    pad = Tg(0.05)
    lb = Tg(-1.0) - pad
    extent = Tg(2.0) + Tg(2.0) * pad
    inv_cell = Tg(grid_res) / extent

    # 🌟 4: Unboxed Tuple Array structures directly fetch identically contiguous 8-byte boundaries
    cells = Array{Tuple{Int32,Tg},3}(undef, grid_res, grid_res, grid_res)
    stack_capacity = bvh.max_depth + 4

    Threads.@threads :dynamic for k in 1:grid_res
        stack = Vector{NodeDist{Tg}}(undef, stack_capacity)
        wind_stack = Vector{Int32}(undef, stack_capacity)
        for j in 1:grid_res
            hint_tri = Int32(0) # 🌟 2: Seed horizontal propagation
            for i in 1:grid_res
                cx = lb + (Tg(i) - Tg(0.5)) / inv_cell
                cy = lb + (Tg(j) - Tg(0.5)) / inv_cell
                cz = lb + (Tg(k) - Tg(0.5)) / inv_cell
                p = Point3{Tg}(cx, cy, cz)

                # 🌟 2: Forward hint propagation shrinks initial BVH search spheres instantly
                tri_best, dist²_best, feat = closest_triangle_kernel(p, bvh, tri_geometries, stack, hint_tri)
                hint_tri = tri_best

                # Eikonal Early-Out Precomputation
                dist = √(max(dist²_best, zero(Tg)))
                if feat == FEATURE_FACE
                    ap = p - tri_geometries[tri_best].a
                    sgn = ifelse((tri_geometries[tri_best].n ⋅ ap) >= zero(Tg), one(Tg), -one(Tg))
                else
                    wn = winding_number_point_kernel(fwn.nodes, tri_geometries, p, wind_stack)
                    sgn = ifelse(wn >= 0.5, -one(Tg), one(Tg))
                end

                cells[i, j, k] = (tri_best, sgn * dist)
            end
        end
    end

    return HintGrid{Tg}(lb, inv_cell, grid_res, cells)
end

function preprocess_mesh(
    mesh::Mesh{3,Tg,GLTriangleFace}; leaf_capacity::Int=8, β_wind::Real=2.0, grid_res::Int=64
) where {Tg<:AbstractFloat}
    vertices = GeometryBasics.coordinates(mesh)
    tri_faces = GeometryBasics.faces(mesh)
    faces = NTuple{3,Int32}.(tri_faces)
    return preprocess_mesh(vertices, faces; leaf_capacity, β_wind, grid_res)
end

function preprocess_mesh(
    vertices::Vector{Point3{Tg}},
    faces::Vector{NTuple{3,Int32}};
    leaf_capacity::Int=8, β_wind::Real=2.0, grid_res::Int=64
) where {Tg<:AbstractFloat}
    num_faces = length(faces)
    (num_faces > 0) || error("Mesh must contain at least one face.")

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

    tri_geometries = Vector{TriangleGeometry{Tg}}(undef, num_faces)
    face_to_packed = Vector{Int32}(undef, num_faces)

    @inbounds for j in eachindex(faces)
        idx_face = tri_order[j]
        face_to_packed[idx_face] = Int32(j)

        face = faces[idx_face]
        (idx_v1, idx_v2, idx_v3) = face
        v1 = vertices[idx_v1]
        v2 = vertices[idx_v2]
        v3 = vertices[idx_v3]

        ab = v2 - v1
        ac = v3 - v1
        d11 = norm²(ab)
        d12 = ab ⋅ ac
        d22 = norm²(ac)

        c_x = ab[2] * ac[3] - ab[3] * ac[2]
        c_y = ab[3] * ac[1] - ab[1] * ac[3]
        c_z = ab[1] * ac[2] - ab[2] * ac[1]
        denom_sum = c_x * c_x + c_y * c_y + c_z * c_z
        zer = zero(Tg)

        # Guard prevents division by subnormal cascades throwing downstream NaNs
        inv_denom = ifelse(denom_sum > floatmin(Tg), inv(denom_sum), zer)

        tri_geometries[j] = TriangleGeometry{Tg}(
            v1, ab, ac, Point3{Tg}(c_x, c_y, c_z), d11, d12, d22, inv_denom
        )
    end

    fwn = precompute_fast_winding_data(bvh, tri_geometries; β=Tg(β_wind))
    hint_grid = compute_hint_grid(bvh, tri_geometries, fwn, grid_res)

    sdm = SignedDistanceMesh{Tg}(tri_geometries, bvh, face_to_packed, fwn, hint_grid)
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
    fwn_nodes = Vector{FWNNode{Tg}}(undef, num_nodes)
    exact_r = Vector{Float64}(undef, num_nodes)

    area_sum = Vector{Float64}(undef, num_nodes)
    cent_sum_x = Vector{Float64}(undef, num_nodes)
    cent_sum_y = Vector{Float64}(undef, num_nodes)
    cent_sum_z = Vector{Float64}(undef, num_nodes)
    n_sum_x = Vector{Float64}(undef, num_nodes)
    n_sum_y = Vector{Float64}(undef, num_nodes)
    n_sum_z = Vector{Float64}(undef, num_nodes)

    # Clean bottom-up reverse iteration inherently evaluates structurally deepest nodes exactly
    @inbounds for node_id in num_nodes:-1:1
        if node_id == 2
            fwn_nodes[2] = FWNNode{Tg}(
                zero(Tg), zero(Tg), zero(Tg),
                zero(Tg), zero(Tg), zero(Tg),
                floatmax(Tg), UInt32(0)
            )
            exact_r[2] = 0.0
            continue
        end

        node = bvh.nodes[node_id]
        child_or_size = node.child_or_size

        if child_or_size < 0
            leaf_start = Int(node.index)
            leaf_end = Int(-child_or_size)
            a_sum = 0.0
            c_x_sum = 0.0
            c_y_sum = 0.0
            c_z_sum = 0.0
            nn_x_sum = 0.0
            nn_y_sum = 0.0
            nn_z_sum = 0.0

            for idx in leaf_start:leaf_end
                tri = tri_geometries[idx]
                n_x = Float64(tri.n[1])
                n_y = Float64(tri.n[2])
                n_z = Float64(tri.n[3])
                va_x = 0.5 * n_x
                va_y = 0.5 * n_y
                va_z = 0.5 * n_z
                area = 0.5 * √(n_x * n_x + n_y * n_y + n_z * n_z)

                if area > 0.0
                    cent_x = Float64(tri.a[1]) + (Float64(tri.ab[1]) + Float64(tri.ac[1])) / 3.0
                    cent_y = Float64(tri.a[2]) + (Float64(tri.ab[2]) + Float64(tri.ac[2])) / 3.0
                    cent_z = Float64(tri.a[3]) + (Float64(tri.ab[3]) + Float64(tri.ac[3])) / 3.0

                    a_sum += area
                    c_x_sum += area * cent_x
                    c_y_sum += area * cent_y
                    c_z_sum += area * cent_z
                end
                nn_x_sum += va_x
                nn_y_sum += va_y
                nn_z_sum += va_z
            end

            area_sum[node_id] = a_sum
            cent_sum_x[node_id] = c_x_sum
            cent_sum_y[node_id] = c_y_sum
            cent_sum_z[node_id] = c_z_sum
            n_sum_x[node_id] = nn_x_sum
            n_sum_y[node_id] = nn_y_sum
            n_sum_z[node_id] = nn_z_sum
        else
            child_l = Int(node.index)
            child_r = Int(child_or_size)
            a_sum = area_sum[child_l] + area_sum[child_r]
            area_sum[node_id] = a_sum
            cent_sum_x[node_id] = cent_sum_x[child_l] + cent_sum_x[child_r]
            cent_sum_y[node_id] = cent_sum_y[child_l] + cent_sum_y[child_r]
            cent_sum_z[node_id] = cent_sum_z[child_l] + cent_sum_z[child_r]
            n_sum_x[node_id] = n_sum_x[child_l] + n_sum_x[child_r]
            n_sum_y[node_id] = n_sum_y[child_l] + n_sum_y[child_r]
            n_sum_z[node_id] = n_sum_z[child_l] + n_sum_z[child_r]
        end

        a_sum = area_sum[node_id]
        if a_sum > 0.0
            c_x = cent_sum_x[node_id] / a_sum
            c_y = cent_sum_y[node_id] / a_sum
            c_z = cent_sum_z[node_id] / a_sum
        else
            c_x = 0.5 * (Float64(node.lb_x) + Float64(node.ub_x))
            c_y = 0.5 * (Float64(node.lb_y) + Float64(node.ub_y))
            c_z = 0.5 * (Float64(node.lb_z) + Float64(node.ub_z))
        end

        local topology::UInt32

        if child_or_size < 0
            leaf_start = Int(node.index)
            leaf_end = Int(-child_or_size)
            max_r² = 0.0
            for idx in leaf_start:leaf_end
                tri = tri_geometries[idx]
                for v in (tri.a, tri.a + tri.ab, tri.a + tri.ac)
                    d² = (Float64(v[1]) - c_x)^2 + (Float64(v[2]) - c_y)^2 + (Float64(v[3]) - c_z)^2
                    max_r² = max(max_r², d²)
                end
            end
            exact_r[node_id] = √(max_r²)

            count = UInt32(leaf_end - leaf_start + 1)
            # 🌟 5.B: Bitwise layout formally extended to gracefully support meshes crossing 16.7M face indices
            topology = 0x80000000 | (count << 25) | UInt32(leaf_start)
        else
            child_l = Int(node.index)
            child_r = Int(child_or_size)

            fL = fwn_nodes[child_l]
            fR = fwn_nodes[child_r]
            r_l = exact_r[child_l] + √((Float64(fL.cm_x) - c_x)^2 + (Float64(fL.cm_y) - c_y)^2 + (Float64(fL.cm_z) - c_z)^2)
            r_r = exact_r[child_r] + √((Float64(fR.cm_x) - c_x)^2 + (Float64(fR.cm_y) - c_y)^2 + (Float64(fR.cm_z) - c_z)^2)
            hierarchical_r = max(r_l, r_r)

            # 🌟 3: Tighter Multipole MAC via formal AABB geometry constraints
            dx = max(abs(Float64(node.lb_x) - c_x), abs(Float64(node.ub_x) - c_x))
            dy = max(abs(Float64(node.lb_y) - c_y), abs(Float64(node.ub_y) - c_y))
            dz = max(abs(Float64(node.lb_z) - c_z), abs(Float64(node.ub_z) - c_z))
            aabb_r = √(dx^2 + dy^2 + dz^2)

            exact_r[node_id] = min(hierarchical_r, aabb_r)

            topology = UInt32(child_l)
        end

        r²β² = exact_r[node_id]^2 * Float64(β)^2

        fwn_nodes[node_id] = FWNNode{Tg}(
            Tg(c_x), Tg(c_y), Tg(c_z),
            Tg(n_sum_x[node_id] * INV4PI64),
            Tg(n_sum_y[node_id] * INV4PI64),
            Tg(n_sum_z[node_id] * INV4PI64),
            Tg(r²β²),
            topology
        )
    end

    fwd = FastWindingData{Tg}(fwn_nodes)
    return fwd::FastWindingData{Tg}
end

function allocate_stacks(sdm::SignedDistanceMesh{Tg}, num_points::Int) where {Tg}
    stack_capacity = Int(sdm.bvh.max_depth) + 4

    n_threads = Threads.nthreads()
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
        Vector{NodeDist{Tg}}(undef, stack_capacity),
        Vector{Int32}(undef, stack_capacity)
    ) for _ in 1:num_chunks]
    return stacks::Vector{QueryStacks{Tg}}
end

##############################   High-Performance Hot Loop Routines   ##############################

const FEATURE_FACE = UInt8(0)
const FEATURE_VERTEX_A = UInt8(1)
const FEATURE_VERTEX_B = UInt8(2)
const FEATURE_VERTEX_C = UInt8(3)
const FEATURE_EDGE_AB = UInt8(4)
const FEATURE_EDGE_AC = UInt8(5)
const FEATURE_EDGE_BC = UInt8(6)
const FEATURE_DEGENERATE = UInt8(7)

# 🌟 6: Degenerate Segment Fallback entirely re-engineered for zero redundant allocations
@inline function closest_dist²_segment_feature(
    ap::Point3{Tg}, ab::Point3{Tg}, ab2::Tg,
    feat_edge::UInt8, feat_a::UInt8, feat_b::UInt8
) where {Tg<:AbstractFloat}
    zer = zero(Tg)
    (ab2 <= zer) && return (norm²(ap)::Tg, feat_a)
    @fastmath begin
        tc = clamp((ap ⋅ ab) / ab2, zer, one(Tg))
        dist² = norm²(ap - tc * ab)
        feat = ifelse(tc <= zer, feat_a, ifelse(tc >= one(Tg), feat_b, feat_edge))
        return (dist²::Tg, feat::UInt8)
    end
end

@inline function closest_dist²_triangle_feature(
    p::Point3{Tg}, triangle::TriangleGeometry{Tg}, dist²_best::Tg
) where {Tg<:AbstractFloat}
    @fastmath begin
        inv_denom = triangle.inv_denom
        ap = p - triangle.a

        if iszero(inv_denom)
            # 🌟 6: Bypassing re-derivation via pure algebraic identities directly routing pre-computed attributes
            (d2_ab, f_ab) = closest_dist²_segment_feature(ap, triangle.ab, triangle.d11, FEATURE_EDGE_AB, FEATURE_VERTEX_A, FEATURE_VERTEX_B)
            (d2_ac, f_ac) = closest_dist²_segment_feature(ap, triangle.ac, triangle.d22, FEATURE_EDGE_AC, FEATURE_VERTEX_A, FEATURE_VERTEX_C)

            bc = triangle.ac - triangle.ab
            (d2_bc, f_bc) = closest_dist²_segment_feature(ap - triangle.ab, bc, norm²(bc), FEATURE_EDGE_BC, FEATURE_VERTEX_B, FEATURE_VERTEX_C)

            dist² = d2_ab
            feat = f_ab
            if d2_ac < dist²
                dist² = d2_ac
                feat = f_ac
            end
            if d2_bc < dist²
                dist² = d2_bc
                feat = f_bc
            end
            return (dist²::Tg, feat::UInt8)
        end

        dot_n = triangle.n ⋅ ap
        plane_dist² = (dot_n * dot_n) * inv_denom
        (plane_dist² >= dist²_best) && return (plane_dist²::Tg, FEATURE_FACE)

        d1 = triangle.ab ⋅ ap
        d2 = triangle.ac ⋅ ap

        if (d1 <= 0) && (d2 <= 0)
            dist² = norm²(ap)
            return (dist²::Tg, FEATURE_VERTEX_A)
        end

        d11 = triangle.d11
        d12 = triangle.d12
        d22 = triangle.d22

        d3 = d1 - d11
        d4 = d2 - d12
        if (d3 >= 0) && (d4 <= d3)
            dist² = norm²(ap - triangle.ab)
            return (dist²::Tg, FEATURE_VERTEX_B)
        end

        vc = muladd(d11, d2, -d12 * d1)
        if (vc <= 0) && (d1 >= 0) && (d3 <= 0)
            zer = zero(Tg)
            v = ifelse(d11 > zer, d1 / d11, zer)
            dist² = norm²(ap - v * triangle.ab)
            return (dist²::Tg, FEATURE_EDGE_AB)
        end

        d5 = d1 - d12
        d6 = d2 - d22
        if (d6 >= 0) && (d5 <= d6)
            dist² = norm²(ap - triangle.ac)
            return (dist²::Tg, FEATURE_VERTEX_C)
        end

        vb = muladd(d22, d1, -d12 * d2)
        if (vb <= 0) && (d2 >= 0) && (d6 <= 0)
            zer = zero(Tg)
            w = ifelse(d22 > zer, d2 / d22, zer)
            dist² = norm²(ap - w * triangle.ac)
            return (dist²::Tg, FEATURE_EDGE_AC)
        end

        uno = one(Tg)
        v_bary = (vb + vc) * inv_denom
        if v_bary >= uno
            d43 = d11 - d12 - d1 + d2
            zer = zero(Tg)
            d33 = max(zer, d11 + d22 - Tg(2) * d12)

            w = clamp(ifelse(d33 > zer, d43 / d33, zer), zer, uno)
            bp = ap - triangle.ab
            dist² = norm²(bp - w * (triangle.ac - triangle.ab))
            return (dist²::Tg, FEATURE_EDGE_BC)
        else
            dist² = plane_dist²
            return (dist²::Tg, FEATURE_FACE)
        end
    end
end

###################################   Fast Winding Number Query   ##################################

@inline function solid_angle_scaled(q_f64::NTuple{3,Float64}, tri::TriangleGeometry{Tg}) where {Tg<:AbstractFloat}
    @fastmath begin
        (qx, qy, qz) = q_f64

        ax = Float64(tri.a[1]) - qx
        ay = Float64(tri.a[2]) - qy
        az = Float64(tri.a[3]) - qz
        bx = ax + Float64(tri.ab[1])
        by = ay + Float64(tri.ab[2])
        bz = az + Float64(tri.ab[3])
        cx = ax + Float64(tri.ac[1])
        cy = ay + Float64(tri.ac[2])
        cz = az + Float64(tri.ac[3])

        det = ax * Float64(tri.n[1]) + ay * Float64(tri.n[2]) + az * Float64(tri.n[3])

        la = √(ax * ax + ay * ay + az * az)
        lb = √(bx * bx + by * by + bz * bz)
        lc = √(cx * cx + cy * cy + cz * cz)

        eps_val = 1.0e-30
        (la < eps_val || lb < eps_val || lc < eps_val) && return 0.0

        ab = ax * bx + ay * by + az * bz
        ac = ax * cx + ay * cy + az * cz
        bc = bx * cx + by * cy + bz * cz
        denom = la * lb * lc + ab * lc + ac * lb + bc * la

        ω = atan(det, denom) * INV2PI64
        return ω::Float64
    end
end

@inline function winding_number_point_kernel(
    fwn_nodes::Vector{FWNNode{Tg}},
    tri_geometries::Vector{TriangleGeometry{Tg}},
    point::Point3{Tg},
    stack::Vector{Int32}
) where {Tg<:AbstractFloat}

    qx = Float64(point[1])
    qy = Float64(point[2])
    qz = Float64(point[3])
    q_f64 = (qx, qy, qz)

    wn = 0.0
    stack_top = 1
    @inbounds stack[1] = Int32(1)

    @inbounds while stack_top > 0
        node_id = Int(stack[stack_top])
        stack_top -= 1

        while true
            fnode = fwn_nodes[node_id]

            @fastmath begin
                rx = Float64(fnode.cm_x) - qx
                ry = Float64(fnode.cm_y) - qy
                rz = Float64(fnode.cm_z) - qz
                dist2 = rx * rx + ry * ry + rz * rz

                if dist2 > Float64(fnode.r²β²)
                    dot_val = rx * Float64(fnode.n_x) + ry * Float64(fnode.n_y) + rz * Float64(fnode.n_z)
                    inv_dist = inv(√(dist2))
                    wn += dot_val * (inv_dist * inv_dist * inv_dist)
                    break
                end
            end

            topology = fnode.topology
            if (topology & 0x80000000) != 0
                count = (topology >> 25) & 0x3F           # 🌟 5.B: Unpack robust 6-bit count limit
                leaf_start = topology & 0x01FFFFFF        # 🌟 5.B: Unpack strict 25-bit face indices
                leaf_end = leaf_start + count - 1
                for tri_id in leaf_start:leaf_end
                    wn += solid_angle_scaled(q_f64, tri_geometries[tri_id])
                end
                break
            else
                child_l_id = Int(topology)
                child_r_id = child_l_id + 1

                stack_top += 1
                stack[stack_top] = Int32(child_r_id)
                node_id = child_l_id
                continue
            end
        end
    end

    return wn::Float64
end

######################################   Single-Point Query   ######################################

@inline function get_grid_hint(grid::HintGrid{Tg}, point::Point3{Tg}) where {Tg<:AbstractFloat}
    inv_c = grid.inv_cell
    lb = grid.lb

    x = (point[1] - lb) * inv_c
    y = (point[2] - lb) * inv_c
    z = (point[3] - lb) * inv_c

    res = grid.res
    @fastmath begin
        # Pre-truncation bounding gracefully intercepts uninitialized query hazards
        x_c = clamp(x, zero(Tg), Tg(res - 1))
        y_c = clamp(y, zero(Tg), Tg(res - 1))
        z_c = clamp(z, zero(Tg), Tg(res - 1))

        ix = Base.unsafe_trunc(Int, x_c) + 1
        iy = Base.unsafe_trunc(Int, y_c) + 1
        iz = Base.unsafe_trunc(Int, z_c) + 1
    end

    cx = lb + (Tg(ix) - Tg(0.5)) / inv_c
    cy = lb + (Tg(iy) - Tg(0.5)) / inv_c
    cz = lb + (Tg(iz) - Tg(0.5)) / inv_c

    dx = point[1] - cx
    dy = point[2] - cy
    dz = point[3] - cz
    dist_to_c² = muladd(dx, dx, muladd(dy, dy, dz * dz))

    # 🌟 4: Access unpacked scalar structures instantly mapped against an unboxed Tuple cache-line
    @inbounds cell = grid.cells[ix, iy, iz]
    return cell[1], cell[2], dist_to_c²
end

function signed_distance_point(
    sdm::SignedDistanceMesh{Tg}, point::Point3{Tg}, hint_face::Int32, stacks::QueryStacks{Tg}
) where {Tg<:AbstractFloat}
    dist²_best = floatmax(Tg)

    if (hint_face > 0) && (hint_face <= length(sdm.tri_geometries))
        tri_best = hint_face
        @inbounds triangle = sdm.tri_geometries[tri_best]
        (dist²_best, feature_best) = closest_dist²_triangle_feature(point, triangle, dist²_best)
        (dist²_best <= zero(Tg)) && return zero(Tg)
    else
        tri_best = Int32(0)
        feature_best = FEATURE_DEGENERATE
    end

    (grid_hint, sd_center, dist_to_c²) = get_grid_hint(sdm.hint_grid, point)

    if (grid_hint > 0) && (grid_hint != tri_best)
        @inbounds triangle = sdm.tri_geometries[grid_hint]
        (d², feat) = closest_dist²_triangle_feature(point, triangle, dist²_best)
        if d² <= zero(Tg)
            return zero(Tg)
        end
        if d² < dist²_best
            dist²_best = d²
            tri_best = grid_hint
            feature_best = feat
        end
    end

    signed_distance = signed_distance_point_kernel(
        sdm, point, dist²_best, tri_best, feature_best, sd_center, dist_to_c², stacks
    )
    return signed_distance::Tg
end

function signed_distance_point(
    sdm::SignedDistanceMesh{Tg}, point::Point3{Tg}, stacks::QueryStacks{Tg}
) where {Tg<:AbstractFloat}
    (grid_hint, sd_center, dist_to_c²) = get_grid_hint(sdm.hint_grid, point)

    dist²_best = floatmax(Tg)
    if (grid_hint > 0) && (grid_hint <= length(sdm.tri_geometries))
        tri_best = grid_hint
        @inbounds triangle = sdm.tri_geometries[tri_best]
        (dist²_best, feature_best) = closest_dist²_triangle_feature(point, triangle, dist²_best)
        (dist²_best <= zero(Tg)) && return zero(Tg)
    else
        tri_best = Int32(0)
        feature_best = FEATURE_DEGENERATE
    end

    signed_distance = signed_distance_point_kernel(
        sdm, point, dist²_best, tri_best, feature_best, sd_center, dist_to_c², stacks
    )
    return signed_distance::Tg
end

function signed_distance_point_kernel(
    sdm::SignedDistanceMesh{Tg},
    point::Point3{Tg},
    dist²_best_in::Tg,
    tri_best_in::Int32,
    feature_best_in::UInt8,
    sd_center::Tg,
    dist_to_c²::Tg,
    stacks::QueryStacks{Tg}
) where {Tg<:AbstractFloat}
    bvh = sdm.bvh
    tri_geometries = sdm.tri_geometries
    stack = stacks.dist
    wind_stack = stacks.wind

    zer = zero(Tg)
    (p_x, p_y, p_z) = point

    dist²_best = dist²_best_in
    tri_best = tri_best_in
    feature_best = feature_best_in

    stack_top = 1
    @inbounds root_node = bvh.nodes[1]
    @inbounds stack[1] = NodeDist{Tg}(zer, root_node.index, root_node.child_or_size)

    @inbounds while stack_top > 0
        node_dist = stack[stack_top]
        stack_top -= 1

        while true
            (node_dist.dist² >= dist²_best) && break
            child_or_size = node_dist.child_or_size

            if child_or_size > 0
                child_l_id = node_dist.index
                child_r_id = child_or_size

                node_l = bvh.nodes[child_l_id]
                node_r = bvh.nodes[child_r_id]

                @fastmath begin
                    Δx_l = max(node_l.lb_x - p_x, p_x - node_l.ub_x, zer)
                    Δy_l = max(node_l.lb_y - p_y, p_y - node_l.ub_y, zer)
                    Δz_l = max(node_l.lb_z - p_z, p_z - node_l.ub_z, zer)
                    dist²_l = muladd(Δx_l, Δx_l, muladd(Δy_l, Δy_l, Δz_l * Δz_l))

                    Δx_r = max(node_r.lb_x - p_x, p_x - node_r.ub_x, zer)
                    Δy_r = max(node_r.lb_y - p_y, p_y - node_r.ub_y, zer)
                    Δz_r = max(node_r.lb_z - p_z, p_z - node_r.ub_z, zer)
                    dist²_r = muladd(Δx_r, Δx_r, muladd(Δy_r, Δy_r, Δz_r * Δz_r))
                end

                idx_l = node_l.index
                cos_l = node_l.child_or_size
                idx_r = node_r.index
                cos_r = node_r.child_or_size

                swap = dist²_l > dist²_r
                near_dist² = ifelse(swap, dist²_r, dist²_l)
                far_dist² = ifelse(swap, dist²_l, dist²_r)
                near_idx = ifelse(swap, idx_r, idx_l)
                far_idx = ifelse(swap, idx_l, idx_r)
                near_cos = ifelse(swap, cos_r, cos_l)
                far_cos = ifelse(swap, cos_l, cos_r)

                if near_dist² < dist²_best
                    if far_dist² < dist²_best
                        stack_top += 1
                        @inbounds stack[stack_top] = NodeDist{Tg}(far_dist², far_idx, far_cos)
                    end
                    node_dist = NodeDist{Tg}(near_dist², near_idx, near_cos)
                    continue
                end
                break
            else
                leaf_start = node_dist.index
                leaf_end = -child_or_size
                @inbounds for idx in leaf_start:leaf_end
                    (dist², feat) = closest_dist²_triangle_feature(point, tri_geometries[idx], dist²_best)
                    if dist² < dist²_best
                        dist²_best = dist²
                        tri_best = Int32(idx)
                        feature_best = feat
                    end
                end
                break
            end
        end
    end

    iszero(tri_best) && error("No triangle found for point $point")
    dist²_best = max(dist²_best, zer)
    dist = √(dist²_best)
    iszero(dist) && return zero(Tg)

    @inbounds tri = tri_geometries[tri_best]
    if feature_best == FEATURE_FACE
        ap = point - tri.a
        sgn = ifelse((tri.n ⋅ ap) >= zer, one(Tg), -one(Tg))
        return (sgn * dist)::Tg
    end

    # 🌟 1: STRONGER EIKONAL EARLY-OUT
    # By strictly leveraging the Triangle Inequality directly connecting our fully evaluated voxel center
    # to the local physical magnitude `dist`, bounding separation instantly clears safely
    abs_sd = abs(sd_center)
    sum_dist = dist + abs_sd

    # Multiply coefficient minutely against invisible floating-point rounding violations
    if (sum_dist * sum_dist) > dist_to_c² * Tg(1.00001)
        sgn = ifelse(sd_center >= zer, one(Tg), -one(Tg))
        return (sgn * dist)::Tg
    end

    wn = winding_number_point_kernel(sdm.fwn.nodes, tri_geometries, point, wind_stack)
    uno = one(Tg)
    sgn = ifelse(wn >= 0.5, -uno, uno)
    signed_distance = sgn * dist
    return signed_distance::Tg
end

##########################################   Public API   ##########################################

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
            @inbounds idx_face_packed = (idx_face > 0 && idx_face <= length(face_to_packed)) ? face_to_packed[idx_face] : Int32(0)
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

###################################   Optional Reusable Scratch   ##################################

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
            @inbounds idx_face_packed = (idx_face > 0 && idx_face <= length(face_to_packed)) ? face_to_packed[idx_face] : Int32(0)
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
