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

# Packed triangle geometries perfectly aligned to exactly 64 bytes (16 Floats) for an unbroken L1 cache-line
struct TriangleGeometry{T<:AbstractFloat}
    a::Point3{T}
    ab::Point3{T}
    ac::Point3{T}
    n::Point3{T}     # Unnormalized face normal: ab × ac
    d11::T           # norm²(ab)
    d12::T           # ab ⋅ ac
    d22::T           # norm²(ac)
    inv_denom::T     # 1.0 / norm²(n)
end

# 32-byte aligned AoS Fast Winding Number (0th-order dipole expansion)
struct FWNNode{T<:AbstractFloat}
    cm_x::T
    cm_y::T
    cm_z::T
    r_beta_sq::T
    n_x::T
    n_y::T
    n_z::T
    _pad::T
end

struct FastWindingData{T<:AbstractFloat}
    nodes::Vector{FWNNode{T}}
end

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

# Upgraded stack element carrying exact topology cache to eradicate redundant
# bvh.nodes memory fetches when elements are popped. (Dense 12-byte layout for Float32)
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

    count = hi - lo + 1
    if count <= builder.leaf_capacity
        # Optimization: O(N) primitive bound limits purely evaluated exclusively for terminal nodes
        min_x, min_y, min_z = T(Inf), T(Inf), T(Inf)
        max_x, max_y, max_z = -T(Inf), -T(Inf), -T(Inf)
        @inbounds for i in lo:hi
            idx_face = tri_indices[i]
            min_x = min(min_x, lb_x_t[idx_face])
            min_y = min(min_y, lb_y_t[idx_face])
            min_z = min(min_z, lb_z_t[idx_face])
            max_x = max(max_x, ub_x_t[idx_face])
            max_y = max(max_y, ub_y_t[idx_face])
            max_z = max(max_z, ub_z_t[idx_face])
        end

        # Store bounds directly so leaf_end doesn't need sequential arithmetic later
        @inbounds builder.nodes[node_id] = BVHNode{T}(
            min_x, min_y, min_z, max_x, max_y, max_z, Int32(lo), -Int32(hi)
        )
        return node_id
    end

    centroid_min_x, centroid_min_y, centroid_min_z = T(Inf), T(Inf), T(Inf)
    centroid_max_x, centroid_max_y, centroid_max_z = -T(Inf), -T(Inf), -T(Inf)
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

    mid = (lo + hi) >>> 1
    (spread_max > 0) && median_split_sort!(tri_indices, lo, mid, hi, centroids, axis)

    child_l = builder.next_node
    child_r = builder.next_node + Int32(1)
    builder.next_node += Int32(2)

    build_node!(builder, child_l, tri_indices, lo, mid, centroids, lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t)
    build_node!(builder, child_r, tri_indices, mid + 1, hi, centroids, lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t)

    # Internal Bound Reduction: O(1) bottom-up exact bound merge natively spanning calculated child depths
    @inbounds begin
        node_l = builder.nodes[child_l]
        node_r = builder.nodes[child_r]
        builder.nodes[node_id] = BVHNode{T}(
            min(node_l.lb_x, node_r.lb_x), min(node_l.lb_y, node_r.lb_y), min(node_l.lb_z, node_r.lb_z),
            max(node_l.ub_x, node_r.ub_x), max(node_l.ub_y, node_r.ub_y), max(node_l.ub_z, node_r.ub_z),
            child_l, child_r
        )
    end
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
    max_nodes = 2 * num_faces + 1

    builder = BVHBuilder{Tg}(Vector{BVHNode{Tg}}(undef, max_nodes), Int32(leaf_capacity), Int32(3))
    if max_nodes >= 2
        builder.nodes[2] = BVHNode{Tg}(zero(Tg), zero(Tg), zero(Tg), zero(Tg), zero(Tg), zero(Tg), Int32(0), Int32(0))
    end

    build_node!(builder, Int32(1), tri_indices, 1, num_faces, centroids, lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t)

    num_nodes = builder.next_node - 1
    resize!(builder.nodes, num_nodes)
    bvh = BoundingVolumeHierarchy{Tg}(builder.nodes, builder.leaf_capacity, Int32(num_nodes))
    return (bvh, tri_indices)
end

######################################   Mesh Preprocessing   ######################################

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
    leaf_capacity::Int=8, winding_beta::Real=2.0
) where {Tg<:AbstractFloat}
    num_faces = length(faces)
    num_faces > 0 || error("Mesh must contain at least one face.")

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
        v1, v2, v3 = vertices[idx_v1], vertices[idx_v2], vertices[idx_v3]
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
    (bvh, tri_order) = build_bvh(centroids, lb_x_t, lb_y_t, lb_z_t, ub_x_t, ub_y_t, ub_z_t; leaf_capacity)

    tri_geometries = Vector{TriangleGeometry{Tg}}(undef, num_faces)
    face_to_packed = Vector{Int32}(undef, num_faces)

    @inbounds for j in eachindex(faces)
        idx_face = tri_order[j]
        face_to_packed[idx_face] = Int32(j)

        (idx_v1, idx_v2, idx_v3) = faces[idx_face]
        v1, v2, v3 = vertices[idx_v1], vertices[idx_v2], vertices[idx_v3]

        ab = v2 - v1
        ac = v3 - v1
        d11 = norm²(ab)
        d12 = ab ⋅ ac
        d22 = norm²(ac)

        cx = ab[2] * ac[3] - ab[3] * ac[2]
        cy = ab[3] * ac[1] - ab[1] * ac[3]
        cz = ab[1] * ac[2] - ab[2] * ac[1]
        denom_sum = cx * cx + cy * cy + cz * cz
        inv_denom = denom_sum > 0 ? inv(denom_sum) : zero(Tg)

        # Precompute & persist identical 64-Byte layout mapping structural normal to memory
        tri_geometries[j] = TriangleGeometry{Tg}(
            v1, ab, ac, Point3{Tg}(cx, cy, cz),
            d11, d12, d22, inv_denom
        )
    end

    fwn = precompute_fast_winding_data(bvh, tri_geometries; beta=Tg(winding_beta))
    return SignedDistanceMesh{Tg}(tri_geometries, bvh, face_to_packed, fwn)
end


#################################   Fast Winding Number Precompute   #################################

const INV4PI64 = 1.0 / (4.0 * π)
const INV2PI64 = 1.0 / (2.0 * π)

function precompute_fast_winding_data(
    bvh::BoundingVolumeHierarchy{Tg},
    tri_geometries::Vector{TriangleGeometry{Tg}};
    beta::Tg=Tg(2)
) where {Tg<:AbstractFloat}

    num_nodes = Int(bvh.num_nodes)
    fwn_nodes = Vector{FWNNode{Tg}}(undef, num_nodes)

    area_sum = Vector{Float64}(undef, num_nodes)
    cent_sum_x = Vector{Float64}(undef, num_nodes)
    cent_sum_y = Vector{Float64}(undef, num_nodes)
    cent_sum_z = Vector{Float64}(undef, num_nodes)
    n_sum_x = Vector{Float64}(undef, num_nodes)
    n_sum_y = Vector{Float64}(undef, num_nodes)
    n_sum_z = Vector{Float64}(undef, num_nodes)

    @inbounds for node_id in num_nodes:-1:1
        (node_id == 2) && continue

        node = bvh.nodes[node_id]
        child_or_size = node.child_or_size

        if child_or_size < 0
            leaf_start = Int(node.index)
            leaf_end = Int(-child_or_size)

            a_sum = c_x_sum = c_y_sum = c_z_sum = nn_x_sum = nn_y_sum = nn_z_sum = 0.0

            for idx in leaf_start:leaf_end
                tri = tri_geometries[idx]

                # Retrieve from pristine L1 cache geometry arrays securely initialized earlier!
                nx, ny, nz = Float64(tri.n[1]), Float64(tri.n[2]), Float64(tri.n[3])
                vax, vay, vaz = 0.5 * nx, 0.5 * ny, 0.5 * nz
                area = 0.5 * sqrt(nx * nx + ny * ny + nz * nz)

                if area > 0.0
                    centx = Float64(tri.a[1]) + (Float64(tri.ab[1]) + Float64(tri.ac[1])) / 3.0
                    centy = Float64(tri.a[2]) + (Float64(tri.ab[2]) + Float64(tri.ac[2])) / 3.0
                    centz = Float64(tri.a[3]) + (Float64(tri.ab[3]) + Float64(tri.ac[3])) / 3.0

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
            cent_sum_x[node_id], cent_sum_y[node_id], cent_sum_z[node_id] = c_x_sum, c_y_sum, c_z_sum
            n_sum_x[node_id], n_sum_y[node_id], n_sum_z[node_id] = nn_x_sum, nn_y_sum, nn_z_sum
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

        dx = max(abs(c_x - Float64(node.lb_x)), abs(c_x - Float64(node.ub_x)))
        dy = max(abs(c_y - Float64(node.lb_y)), abs(c_y - Float64(node.ub_y)))
        dz = max(abs(c_z - Float64(node.lb_z)), abs(c_z - Float64(node.ub_z)))

        r_sq = dx * dx + dy * dy + dz * dz
        r_beta_sq = (Float64(beta)^2) * r_sq

        # Optimization: Pre-absorb `INV4PI64` factor into the normal area sum to prune BVH internal loop logic
        fwn_nodes[node_id] = FWNNode{Tg}(
            Tg(c_x), Tg(c_y), Tg(c_z), Tg(r_beta_sq),
            Tg(n_sum_x[node_id] * INV4PI64), Tg(n_sum_y[node_id] * INV4PI64), Tg(n_sum_z[node_id] * INV4PI64),
            zero(Tg)
        )
    end

    return FastWindingData{Tg}(fwn_nodes)
end

function calculate_tree_height(num_faces::Integer, leaf_capacity::Integer)
    num_leaves = max(cld(num_faces, leaf_capacity), 1)
    tree_height = ceil(Int, log2(num_leaves))
    return tree_height::Int
end

function allocate_stacks(sdm::SignedDistanceMesh{Tg}, num_points::Int) where {Tg}
    num_faces = length(sdm.tri_geometries)
    leaf_capacity = sdm.bvh.leaf_capacity
    tree_height = calculate_tree_height(num_faces, leaf_capacity)

    # Absolute bounds safety padding against heavily degenerate meshes
    stack_capacity = max(128, 2 * tree_height + 4)
    n_threads = Threads.nthreads()
    min_chunk_size = 512
    num_chunks_max = max(1, num_points ÷ min_chunk_size)

    if num_chunks_max < n_threads
        num_chunks = num_chunks_max
    else
        factor = min(8, num_chunks_max ÷ n_threads)
        num_chunks = n_threads * factor
    end

    stacks = [QueryStacks{Tg}(
        Vector{NodeDist{Tg}}(undef, stack_capacity),
        Vector{Int32}(undef, stack_capacity)
    ) for _ in 1:num_chunks]
    return stacks
end

##############################   High-Performance Hot Loop Routines   ##############################

# Optimized O(1) Fast Rejection and Division-Free Exact Algebraic Projection
@inline function closest_dist²_triangle(p::Point3{Tg}, triangle::TriangleGeometry{Tg}, dist²_best::Tg) where {Tg}
    @fastmath begin
        ap = p - triangle.a

        # Early rejection isolating out-of-range primitives before computationally heavy FMA logic!
        # Because theoretical lower-bound distance to the full triangle maps strictly to the face plane distance
        dot_n = triangle.n ⋅ ap
        plane_dist² = (dot_n * dot_n) * triangle.inv_denom
        (plane_dist² >= dist²_best) && return plane_dist²

        d1 = triangle.ab ⋅ ap
        d2 = triangle.ac ⋅ ap

        (d1 <= 0 && d2 <= 0) && return norm²(ap)

        d11, d12, d22 = triangle.d11, triangle.d12, triangle.d22

        d3 = d1 - d11
        d4 = d2 - d12
        (d3 >= 0 && d4 <= d3) && return norm²(ap - triangle.ab)

        vc = d11 * d2 - d12 * d1
        if vc <= 0 && d1 >= 0 && d3 <= 0
            v = d11 > zero(Tg) ? d1 / d11 : zero(Tg)
            return norm²(ap - v * triangle.ab)
        end

        d5 = d1 - d12
        d6 = d2 - d22
        (d6 >= 0 && d5 <= d6) && return norm²(ap - triangle.ac)

        vb = d22 * d1 - d12 * d2
        if vb <= 0 && d2 >= 0 && d6 <= 0
            w = d22 > zero(Tg) ? d2 / d22 : zero(Tg)
            return norm²(ap - w * triangle.ac)
        end

        # Deferred Edge Arithmetic (Branchless): heavily prevents unneeded allocations mapped out to pure scalar bounds
        if (vb + vc) * triangle.inv_denom >= one(Tg)
            d43 = d11 - d12 - d1 + d2
            # Fast algebraic scalar shortcut dynamically maps vectors outside the norm checks
            d33 = max(zero(Tg), d11 + d22 - Tg(2) * d12)

            # Unconditionally clamp the algebraic weight to bypass CPU branches
            w = clamp(d33 > zero(Tg) ? d43 / d33 : zero(Tg), zero(Tg), one(Tg))
            bp = ap - triangle.ab
            return norm²(bp - w * (triangle.ac - triangle.ab))
        end

        # Fast algebraic interior plane mapping bypassing 15 interpolation vector arithmetic operations
        return (triangle.inv_denom > zero(Tg)) ? plane_dist² : norm²(ap)
    end
end

#################################   Fast Face-Interior Sign Shortcut   #################################

@inline function face_interior_sign_or_zero(
    p::Point3{Tg}, tri::TriangleGeometry{Tg}, bary_tol::Tg=Tg(1e-4)
) where {Tg<:AbstractFloat}
    inv_denom = tri.inv_denom
    iszero(inv_denom) && return zero(Tg)

    ap = p - tri.a
    v = (tri.d22 * (tri.ab ⋅ ap) - tri.d12 * (tri.ac ⋅ ap)) * inv_denom
    w = (tri.d11 * (tri.ac ⋅ ap) - tri.d12 * (tri.ab ⋅ ap)) * inv_denom

    (v <= bary_tol || w <= bary_tol || (one(Tg) - v - w) <= bary_tol) && return zero(Tg)

    # Re-used exact structural normal entirely avoids the cross product calculation
    return ifelse((tri.n ⋅ ap) >= zero(Tg), one(Tg), -one(Tg))
end

#################################   Fast Winding Number Query   #################################

@inline function solid_angle_scaled(q_f64::NTuple{3,Float64}, tri::TriangleGeometry{Tg}) where {Tg<:AbstractFloat}
    @fastmath begin
        qx, qy, qz = q_f64

        ax = Float64(tri.a[1]) - qx
        ay = Float64(tri.a[2]) - qy
        az = Float64(tri.a[3]) - qz

        bx = ax + Float64(tri.ab[1])
        by = ay + Float64(tri.ab[2])
        bz = az + Float64(tri.ab[3])

        cx = ax + Float64(tri.ac[1])
        cy = ay + Float64(tri.ac[2])
        cz = az + Float64(tri.ac[3])

        # Mathematical substitution: (B_q × C_q) ⋅ A_q equates perfectly to A_q ⋅ (AB × AC)
        det = ax * Float64(tri.n[1]) + ay * Float64(tri.n[2]) + az * Float64(tri.n[3])

        la = sqrt(ax * ax + ay * ay + az * az)
        lb = sqrt(bx * bx + by * by + bz * bz)
        lc = sqrt(cx * cx + cy * cy + cz * cz)

        ab = ax * bx + ay * by + az * bz
        ac = ax * cx + ay * cy + az * cz
        bc = bx * cx + by * cy + bz * cz
        denom = la * lb * lc + ab * lc + ac * lb + bc * la

        return atan(det, denom) * INV2PI64
    end
end

@inline function winding_number_point_kernel(
    sdm::SignedDistanceMesh{Tg},
    point::Point3{Tg},
    stack::Vector{Int32}
)::Float64 where {Tg<:AbstractFloat}

    fwn_nodes = sdm.fwn.nodes
    bvh_nodes = sdm.bvh.nodes
    tri_geometries = sdm.tri_geometries

    qx, qy, qz = Float64(point[1]), Float64(point[2]), Float64(point[3])
    q_f64 = (qx, qy, qz)

    wn = 0.0
    stack_top = 1
    @inbounds stack[1] = Int32(1)

    @inbounds while stack_top > 0
        node_id = Int(stack[stack_top])
        stack_top -= 1

        # Tail-Call Winding Number Traversal continuous path loops straight to internal limits
        while true
            fnode = fwn_nodes[node_id]

            @fastmath begin
                rx = Float64(fnode.cm_x) - qx
                ry = Float64(fnode.cm_y) - qy
                rz = Float64(fnode.cm_z) - qz

                dist2 = rx * rx + ry * ry + rz * rz

                if dist2 > Float64(fnode.r_beta_sq)
                    dot = rx * Float64(fnode.n_x) + ry * Float64(fnode.n_y) + rz * Float64(fnode.n_z)
                    inv_dist = inv(sqrt(dist2))
                    wn += dot * (inv_dist * inv_dist * inv_dist)
                    break
                else
                    node = bvh_nodes[node_id]
                    child_or_size = node.child_or_size

                    if child_or_size < 0
                        leaf_start = Int(node.index)
                        leaf_end = Int(-child_or_size)
                        for tri_id in leaf_start:leaf_end
                            wn += solid_angle_scaled(q_f64, tri_geometries[tri_id])
                        end
                        break
                    else
                        # Safely drop the right child into memory and loop straight into left properties
                        stack_top += 1
                        stack[stack_top] = child_or_size
                        node_id = Int(node.index)
                        continue
                    end
                end
            end
        end
    end

    return wn
end

######################################   Single-Point Query   ######################################

function signed_distance_point(
    sdm::SignedDistanceMesh{Tg}, point::Point3{Tg}, hint_face::Int32, stacks::QueryStacks{Tg}
) where {Tg<:AbstractFloat}
    if hint_face > 0 && hint_face <= length(sdm.tri_geometries)
        tri_best = hint_face
        @inbounds triangle = sdm.tri_geometries[tri_best]
        dist²_best = closest_dist²_triangle(point, triangle, Tg(Inf))
        (dist²_best <= 0) && return zero(Tg)
    else
        tri_best = Int32(0)
        dist²_best = Tg(Inf)
    end
    return signed_distance_point_kernel(sdm, point, dist²_best, tri_best, stacks)
end

function signed_distance_point(
    sdm::SignedDistanceMesh{Tg}, point::Point3{Tg}, stacks::QueryStacks{Tg}
) where {Tg<:AbstractFloat}
    return signed_distance_point_kernel(sdm, point, Tg(Inf), Int32(0), stacks)
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

    zer = zero(Tg)
    p_x, p_y, p_z = point[1], point[2], point[3]

    stack_top = 1
    @inbounds root_node = bvh.nodes[1]
    @inbounds stack[1] = NodeDist{Tg}(zer, root_node.index, root_node.child_or_size)

    @inbounds while stack_top > 0
        nd = stack[stack_top]
        stack_top -= 1

        # Tail-Call logic purely traverses bounded limits straight to closer branch L1 cache
        while true
            (nd.dist² >= dist²_best) && break
            child_or_size = nd.child_or_size

            if child_or_size > 0
                child_l_id = nd.index
                child_r_id = child_or_size

                node_l = bvh.nodes[child_l_id]
                node_r = bvh.nodes[child_r_id]

                # Inlining AABB overlap avoids generic `Point` tuple unpacks and speeds up traversal heavily
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

                # Branchless Child Sorting converted natively into hardware cmov instructions via ifelse
                swap = dist²_l > dist²_r
                near_dist² = ifelse(swap, dist²_r, dist²_l)
                far_dist² = ifelse(swap, dist²_l, dist²_r)
                near_idx = ifelse(swap, idx_r, idx_l)
                far_idx = ifelse(swap, idx_l, idx_r)
                near_cos = ifelse(swap, cos_r, cos_l)
                far_cos = ifelse(swap, cos_l, cos_r)

                if near_dist² < dist²_best
                    if far_dist² < dist²_best
                        # Drop STRICTLY the farther node out of bounds to stack arrays
                        stack_top += 1
                        @inbounds stack[stack_top] = NodeDist{Tg}(far_dist², far_idx, far_cos)
                    end
                    # Iterate the tail-called closure inside registers explicitly
                    nd = NodeDist{Tg}(near_dist², near_idx, near_cos)
                    continue
                end
                break
            else
                leaf_start = nd.index
                leaf_end = -child_or_size
                for idx in leaf_start:leaf_end
                    dist² = closest_dist²_triangle(point, tri_geometries[idx], dist²_best)
                    if dist² < dist²_best
                        dist²_best = dist²
                        tri_best = idx
                    end
                end
                break
            end
        end
    end

    iszero(tri_best) && error("No triangle found for point $point")
    dist = √(dist²_best)
    iszero(dist) && return zero(Tg)

    @inbounds tri = tri_geometries[tri_best]
    sgn_fast = face_interior_sign_or_zero(point, tri)
    if !iszero(sgn_fast)
        return (sgn_fast * dist)::Tg
    end

    wn = winding_number_point_kernel(sdm, point, wind_stack)
    uno = one(Tg)
    sgn = ifelse(wn >= 0.5, -uno, uno)
    return (sgn * dist)::Tg
end

##########################################   Public API   ##########################################

function compute_signed_distance!(
    out::AbstractVector{Tg},
    sdm::SignedDistanceMesh{Tg},
    points::AbstractMatrix{Tg},
    hint_faces::AbstractVector{<:Integer}
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
            idx_face = Int32(hint_faces[idx])
            @inbounds idx_face_packed = (idx_face > 0 && idx_face <= length(face_to_packed)) ? face_to_packed[idx_face] : Int32(0)
            @inbounds point = Point3{Tg}(points[1, idx], points[2, idx], points[3, idx])
            @inbounds out[idx] = signed_distance_point(sdm, point, idx_face_packed, stack)
        end
    end
    return out
end

function compute_signed_distance!(
    out::AbstractVector{Tg}, sdm::SignedDistanceMesh{Tg}, points::AbstractMatrix{Tg}
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
    points::AbstractMatrix{Tg},
    hint_faces::AbstractVector{<:Integer},
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
            idx_face = Int32(hint_faces[idx])
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
    points::AbstractMatrix{Tg},
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
