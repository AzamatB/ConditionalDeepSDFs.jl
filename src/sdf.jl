"""
GPU-Accelerated Signed Distance Field (SDF)
══════════════════════════════════════════

This file computes a signed distance field on a uniform grid over the fixed
domain [-1, 1]³. It is designed to be:

• Robust: sign is computed via ray parity (odd/even).
• Fast on GPU: distance uses a narrow-band seed + jump flooding (JFA) over
  triangle indices; sign uses triangle rasterization + a per-column prefix XOR.

Pipeline:
  Phase 1  — Seed:   atomic_min of packed (dist², tri_idx) within a voxel band.
  Phase 1b — Index:  unpack the winning triangle index per voxel.
  Phase 2  — JFA:    propagate triangle indices (26-neighborhood) with exact
                     point-to-triangle distance checks per candidate.
  Phase 3  — Parity: rasterize triangle XY footprint; for each (x,y) cast a jittered +z ray,
                     toggle parity at the hit sample index (half-open convention).
  Phase 4  — Final:  prefix-XOR parity per (x,y) column; recompute exact distance
                     to assigned triangle; apply sign.

Notes:
• Parity sign is orientation-independent (unlike winding-number methods).
• The JFA nearest-triangle assignment is an excellent approximation in practice
  but is not mathematically guaranteed exact for triangle distance; it is,
  however, far faster than brute force for large meshes.
"""

using CUDA
using GeometryBasics

# constants
const SENTINEL_U64 = typemax(UInt64)
const NO_TRIANGLE = Int32(0)   # 1-based triangle indices; 0 = unassigned

##################################   Low-level GPU math   ##################################

@inline function dot3(
    ax::Float32, ay::Float32, az::Float32,
    bx::Float32, by::Float32, bz::Float32
)
    return muladd(ax, bx, muladd(ay, by, az * bz))
end

"""Squared distance from point P to triangle (A, A+AB, A+AC).

Voronoi-region closest-point (Ericson, Real-Time Collision Detection).
"""
@inline function dist²_point_triangle(
    px::Float32, py::Float32, pz::Float32,
    ax::Float32, ay::Float32, az::Float32,
    abx::Float32, aby::Float32, abz::Float32,
    acx::Float32, acy::Float32, acz::Float32,
)
    apx = px - ax
    apy = py - ay
    apz = pz - az
    d1 = dot3(abx, aby, abz, apx, apy, apz)
    d2 = dot3(acx, acy, acz, apx, apy, apz)
    if (d1 <= 0f0) & (d2 <= 0f0)
        return dot3(apx, apy, apz, apx, apy, apz)
    end

    bpx = apx - abx
    bpy = apy - aby
    bpz = apz - abz
    d3 = dot3(abx, aby, abz, bpx, bpy, bpz)
    d4 = dot3(acx, acy, acz, bpx, bpy, bpz)
    if (d3 >= 0f0) & (d4 <= d3)
        return dot3(bpx, bpy, bpz, bpx, bpy, bpz)
    end

    cpx = apx - acx
    cpy = apy - acy
    cpz = apz - acz
    d5 = dot3(abx, aby, abz, cpx, cpy, cpz)
    d6 = dot3(acx, acy, acz, cpx, cpy, cpz)
    if (d6 >= 0f0) & (d5 <= d6)
        return dot3(cpx, cpy, cpz, cpx, cpy, cpz)
    end

    vc = d1 * d4 - d3 * d2
    if (vc <= 0f0) & (d1 >= 0f0) & (d3 <= 0f0)
        v = d1 / (d1 - d3)
        dx = apx - v * abx
        dy = apy - v * aby
        dz = apz - v * abz
        return dot3(dx, dy, dz, dx, dy, dz)
    end

    vb = d5 * d2 - d1 * d6
    if (vb <= 0f0) & (d2 >= 0f0) & (d6 <= 0f0)
        w = d2 / (d2 - d6)
        dx = apx - w * acx
        dy = apy - w * acy
        dz = apz - w * acz
        return dot3(dx, dy, dz, dx, dy, dz)
    end

    va = d3 * d6 - d5 * d4
    if (va <= 0f0) & ((d4 - d3) >= 0f0) & ((d5 - d6) >= 0f0)
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        bcx = acx - abx
        bcy = acy - aby
        bcz = acz - abz
        dx = bpx - w * bcx
        dy = bpy - w * bcy
        dz = bpz - w * bcz
        return dot3(dx, dy, dz, dx, dy, dz)
    end

    denom = inv(va + vb + vc)
    v = vb * denom
    w = vc * denom
    dx = apx - v * abx - w * acx
    dy = apy - v * aby - w * acy
    dz = apz - v * abz - w * acz
    return dot3(dx, dy, dz, dx, dy, dz)
end

# packing (dist², tri_idx) into UInt64 for atomic_min
@inline function pack_dist2_idx(d2::Float32, idx::Int32)
    # High 32 bits: dist² as UInt32 (positive floats preserve ordering)
    # Low  32 bits: triangle index
    return (UInt64(reinterpret(UInt32, d2)) << 32) | UInt64(UInt32(idx))
end

@inline function unpack_idx(p::UInt64)
    return (p & 0xFFFF_FFFF) % Int32
end

# deterministic (x,y) jitter (cheap integer hash)
@inline function u32_hash(x::UInt32)
    y = x
    y ⊻= y >> 16
    y *= UInt32(0x7feb352d)
    y ⊻= y >> 15
    y *= UInt32(0x846ca68b)
    y ⊻= y >> 16
    return y
end

@inline function column_jitter(ix::Int32, iy::Int32, mag::Float32)
    h = u32_hash(UInt32(ix) * 0x9e3779b9 + UInt32(iy) * 0x7f4a7c15)
    jx = (Float32(h & 0xFFFF) / 65535f0 - 0.5f0) * mag
    jy = (Float32((h >> 16) & 0xFFFF) / 65535f0 - 0.5f0) * mag
    return (jx, jy)
end

############################   Phase 1a — narrow-band seeding   ############################

function seed_kernel!(
    packed::CuDeviceArray{UInt64,3},
    v0x::CuDeviceVector{Float32}, v0y::CuDeviceVector{Float32}, v0z::CuDeviceVector{Float32},
    e0x::CuDeviceVector{Float32}, e0y::CuDeviceVector{Float32}, e0z::CuDeviceVector{Float32},
    e1x::CuDeviceVector{Float32}, e1y::CuDeviceVector{Float32}, e1z::CuDeviceVector{Float32},
    origin::Float32, step_val::Float32, inv_step::Float32,
    n::Int32, band::Int32, n_faces::Int32,
)
    fi = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    (fi > n_faces) && return nothing

    @inbounds begin
        ax = v0x[fi]
        ay = v0y[fi]
        az = v0z[fi]
        abx = e0x[fi]
        aby = e0y[fi]
        abz = e0z[fi]
        acx = e1x[fi]
        acy = e1y[fi]
        acz = e1z[fi]
    end

    bx = ax + abx
    by = ay + aby
    bz = az + abz
    cx = ax + acx
    cy = ay + acy
    cz = az + acz

    i0 = max(unsafe_trunc(Int32, (min(ax, bx, cx) - origin) * inv_step) + Int32(1) - band, Int32(1))
    i1 = min(unsafe_trunc(Int32, (max(ax, bx, cx) - origin) * inv_step) + Int32(1) + band, n)
    j0 = max(unsafe_trunc(Int32, (min(ay, by, cy) - origin) * inv_step) + Int32(1) - band, Int32(1))
    j1 = min(unsafe_trunc(Int32, (max(ay, by, cy) - origin) * inv_step) + Int32(1) + band, n)
    k0 = max(unsafe_trunc(Int32, (min(az, bz, cz) - origin) * inv_step) + Int32(1) - band, Int32(1))
    k1 = min(unsafe_trunc(Int32, (max(az, bz, cz) - origin) * inv_step) + Int32(1) + band, n)

    i = i0
    while i <= i1
        px = muladd(Float32(i - Int32(1)), step_val, origin)
        j = j0
        while j <= j1
            py = muladd(Float32(j - Int32(1)), step_val, origin)
            k = k0
            while k <= k1
                pz = muladd(Float32(k - Int32(1)), step_val, origin)
                d2 = dist²_point_triangle(px, py, pz, ax, ay, az,
                    abx, aby, abz, acx, acy, acz)
                lin = i + (j - Int32(1)) * n + (k - Int32(1)) * n * n
                CUDA.atomic_min!(pointer(packed, lin), pack_dist2_idx(d2, fi))
                k += Int32(1)
            end
            j += Int32(1)
        end
        i += Int32(1)
    end
    return nothing
end

##############################   Phase 1b — extract indices   ##############################

function extract_indices_kernel!(
    idx::CuDeviceArray{Int32,3}, packed::CuDeviceArray{UInt64,3}, n::Int32
)
    ix = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    iz = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    ((ix > n) | (iy > n) | (iz > n)) && return nothing

    @inbounds p = packed[ix, iy, iz]
    @inbounds idx[ix, iy, iz] = p == SENTINEL_U64 ? NO_TRIANGLE : unpack_idx(p)
    return nothing
end

###################   Phase 2 — JFA pass (propagate triangle indices)   ###################

function jfa_pass_kernel!(
    grid_out::CuDeviceArray{Int32,3}, grid_in::CuDeviceArray{Int32,3},
    v0x::CuDeviceVector{Float32}, v0y::CuDeviceVector{Float32}, v0z::CuDeviceVector{Float32},
    e0x::CuDeviceVector{Float32}, e0y::CuDeviceVector{Float32}, e0z::CuDeviceVector{Float32},
    e1x::CuDeviceVector{Float32}, e1y::CuDeviceVector{Float32}, e1z::CuDeviceVector{Float32},
    origin::Float32, step_val::Float32, n::Int32, jump::Int32,
)
    ix = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    iz = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    ((ix > n) | (iy > n) | (iz > n)) && return nothing

    px = muladd(Float32(ix - Int32(1)), step_val, origin)
    py = muladd(Float32(iy - Int32(1)), step_val, origin)
    pz = muladd(Float32(iz - Int32(1)), step_val, origin)

    @inbounds best_idx = grid_in[ix, iy, iz]
    best_d2 = Inf32

    @inbounds if best_idx != NO_TRIANGLE
        ax = v0x[best_idx]
        ay = v0y[best_idx]
        az = v0z[best_idx]
        abx = e0x[best_idx]
        aby = e0y[best_idx]
        abz = e0z[best_idx]
        acx = e1x[best_idx]
        acy = e1y[best_idx]
        acz = e1z[best_idx]
        best_d2 = dist²_point_triangle(px, py, pz, ax, ay, az, abx, aby, abz, acx, acy, acz)
    end

    @inbounds for dz in Int32(-1):Int32(1)
        nz = iz + dz * jump
        ((nz < Int32(1)) | (nz > n)) && continue
        for dy in Int32(-1):Int32(1)
            ny = iy + dy * jump
            ((ny < Int32(1)) | (ny > n)) && continue
            for dx in Int32(-1):Int32(1)
                nx = ix + dx * jump
                ((nx < Int32(1)) | (nx > n)) && continue
                ((dx == Int32(0)) & (dy == Int32(0)) & (dz == Int32(0))) && continue

                nb = grid_in[nx, ny, nz]
                ((nb == NO_TRIANGLE) | (nb == best_idx)) && continue

                ax = v0x[nb]
                ay = v0y[nb]
                az = v0z[nb]
                abx = e0x[nb]
                aby = e0y[nb]
                abz = e0z[nb]
                acx = e1x[nb]
                acy = e1y[nb]
                acz = e1z[nb]

                d2 = dist²_point_triangle(px, py, pz, ax, ay, az, abx, aby, abz, acx, acy, acz)
                if d2 < best_d2
                    best_d2 = d2
                    best_idx = nb
                end
            end
        end
    end

    @inbounds grid_out[ix, iy, iz] = best_idx
    return nothing
end

############   Phase 3 — parity rasterization (orientation-independent sign)   ############

function parity_kernel!(
    parity::CuDeviceArray{UInt32,3},
    v0x::CuDeviceVector{Float32}, v0y::CuDeviceVector{Float32}, v0z::CuDeviceVector{Float32},
    e0x::CuDeviceVector{Float32}, e0y::CuDeviceVector{Float32}, e0z::CuDeviceVector{Float32},
    e1x::CuDeviceVector{Float32}, e1y::CuDeviceVector{Float32}, e1z::CuDeviceVector{Float32},
    origin::Float32, step_val::Float32, inv_step::Float32,
    n::Int32, n_faces::Int32,
    jitter_mag::Float32, ε_det::Float32, ε_bary::Float32
)
    fi = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    (fi > n_faces) && return nothing

    @inbounds begin
        ax = v0x[fi]
        ay = v0y[fi]
        az = v0z[fi]
        abx = e0x[fi]
        aby = e0y[fi]
        abz = e0z[fi]
        acx = e1x[fi]
        acy = e1y[fi]
        acz = e1z[fi]
    end

    # Solve barycentric using XY only for vertical rays. We use:
    # det = aby*acx - abx*acy  (=-cross_z)
    det = muladd(aby, acx, -abx * acy)
    (abs(det) <= ε_det) && return nothing
    inv_det = inv(det)

    # XY bounding box (pad by one voxel + jitter)
    bx = ax + abx
    by = ay + aby
    cx = ax + acx
    cy = ay + acy

    # Clamp to domain to keep trunc() behaving like floor (non-negative)
    x_min = max(min(ax, bx, cx) - jitter_mag, origin)
    x_max = min(max(ax, bx, cx) + jitter_mag, origin + step_val * Float32(n - Int32(1)))
    y_min = max(min(ay, by, cy) - jitter_mag, origin)
    y_max = min(max(ay, by, cy) + jitter_mag, origin + step_val * Float32(n - Int32(1)))

    ix0 = max(unsafe_trunc(Int32, (x_min - origin) * inv_step) + Int32(1), Int32(1))
    ix1 = min(unsafe_trunc(Int32, (x_max - origin) * inv_step) + Int32(2), n)
    iy0 = max(unsafe_trunc(Int32, (y_min - origin) * inv_step) + Int32(1), Int32(1))
    iy1 = min(unsafe_trunc(Int32, (y_max - origin) * inv_step) + Int32(2), n)

    z_end = origin + step_val * Float32(n - Int32(1))

    ix = ix0
    while ix <= ix1
        iy = iy0
        while iy <= iy1
            jx, jy = column_jitter(ix, iy, jitter_mag)
            rx = muladd(Float32(ix - Int32(1)), step_val, origin) + jx
            ry = muladd(Float32(iy - Int32(1)), step_val, origin) + jy

            sx = rx - ax
            sy = ry - ay
            u = (muladd(sy, acx, -sx * acy)) * inv_det
            v = (muladd(sx, aby, -sy * abx)) * inv_det

            # Half-open rule: exclude edges/vertices to avoid double-counting.
            if (u <= ε_bary) | (v <= ε_bary) | ((u + v) >= (1f0 - ε_bary))
                iy += Int32(1)
                continue
            end

            z_hit = az + u * abz + v * acz

            # Half-open in z: only hits strictly inside (origin, z_end)
            if (z_hit <= origin) | (z_hit >= z_end)
                iy += Int32(1)
                continue
            end

            # Toggle at the first grid sample strictly above z_hit.
            t = (z_hit - origin) * inv_step  # in (0, n-1)
            z_idx = unsafe_trunc(Int32, t) + Int32(2)

            if z_idx <= n
                lin = ix + (iy - Int32(1)) * n + (z_idx - Int32(1)) * n * n
                CUDA.atomic_xor!(pointer(parity, lin), UInt32(1))
            end
            iy += Int32(1)
        end
        ix += Int32(1)
    end
    return nothing
end

###############   Phase 4 — finalize (prefix XOR parity + exact distance)   ###############

function finalize_kernel!(
    sdf::CuDeviceArray{Float32,3},
    idx_grid::CuDeviceArray{Int32,3},
    parity::CuDeviceArray{UInt32,3},
    v0x::CuDeviceVector{Float32}, v0y::CuDeviceVector{Float32}, v0z::CuDeviceVector{Float32},
    e0x::CuDeviceVector{Float32}, e0y::CuDeviceVector{Float32}, e0z::CuDeviceVector{Float32},
    e1x::CuDeviceVector{Float32}, e1y::CuDeviceVector{Float32}, e1z::CuDeviceVector{Float32},
    origin::Float32, step_val::Float32, n::Int32,
    dist_fallback::Float32,
)
    ix = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    ((ix > n) | (iy > n)) && return nothing

    px = muladd(Float32(ix - Int32(1)), step_val, origin)
    py = muladd(Float32(iy - Int32(1)), step_val, origin)

    parity_acc = UInt32(0)
    iz = Int32(1)
    while iz <= n
        pz = muladd(Float32(iz - Int32(1)), step_val, origin)

        @inbounds parity_acc ⊻= parity[ix, iy, iz]
        is_inside = (parity_acc & UInt32(1)) == UInt32(1)

        @inbounds tri = idx_grid[ix, iy, iz]
        if tri != NO_TRIANGLE
            @inbounds begin
                ax = v0x[tri]
                ay = v0y[tri]
                az = v0z[tri]
                abx = e0x[tri]
                aby = e0y[tri]
                abz = e0z[tri]
                acx = e1x[tri]
                acy = e1y[tri]
                acz = e1z[tri]
            end
            d = sqrt(dist²_point_triangle(px, py, pz, ax, ay, az, abx, aby, abz, acx, acy, acz))
            @inbounds sdf[ix, iy, iz] = is_inside ? -d : d
        else
            # Should not happen for reasonable band; keep sign consistent anyway.
            @inbounds sdf[ix, iy, iz] = is_inside ? -dist_fallback : dist_fallback
        end

        iz += Int32(1)
    end
    return nothing
end

####################   CPU preprocessing: mesh → SoA triangle arrays   ####################

"""Preprocess mesh into SoA Float32 arrays: vertex A + edges AB, AC.

• Edge vectors are computed in Float64 first to reduce cancellation.
• Degenerate triangles are dropped.
"""
function preprocess_geometry(
    vertices::AbstractVector{<:GeometryBasics.Point{3}},
    fcs::AbstractVector;
    ε²::Float64=1e-20
)
    num_faces = length(fcs)
    arrays = ntuple(_ -> sizehint!(Float32[], num_faces), 9)
    v0x, v0y, v0z, e0x, e0y, e0z, e1x, e1y, e1z = arrays

    @inbounds for face in fcs
        # Works for TriangleFace / GLTriangleFace; we only need integer indices.
        a, b, c = Int.(GeometryBasics.value.(face))
        A = Float64.(vertices[a])
        ab = Float64.(vertices[b]) .- A
        ac = Float64.(vertices[c]) .- A
        nx = ab[2] * ac[3] - ab[3] * ac[2]
        ny = ab[3] * ac[1] - ab[1] * ac[3]
        nz = ab[1] * ac[2] - ab[2] * ac[1]
        ((nx * nx + ny * ny + nz * nz) <= ε²) && continue

        push!(v0x, Float32(A[1]))
        push!(v0y, Float32(A[2]))
        push!(v0z, Float32(A[3]))
        push!(e0x, Float32(ab[1]))
        push!(e0y, Float32(ab[2]))
        push!(e0z, Float32(ab[3]))
        push!(e1x, Float32(ac[1]))
        push!(e1y, Float32(ac[2]))
        push!(e1z, Float32(ac[3]))
    end

    n_faces = Int32(length(v0x))
    (n_faces > 0) || error("All faces degenerate after filtering")
    return (;
        v0x, v0y, v0z,
        e0x, e0y, e0z,
        e1x, e1y, e1z,
        n_faces
    )
end

######################################   Public API   ######################################

"""
Compute the signed distance field on a uniform [-1,1]³ grid.

Returns a `CuArray{Float32,3}` of size `(n,n,n)`.

Keyword arguments (good defaults for 256³):
• `band=5`
    Half-width (in voxels) of the seed AABB expansion around each triangle.
    Larger = better JFA coverage, slower seeding.
• `jfa_corrections=2`
    Number of extra JFA passes at `jump=1` after the main pyramid.
• `jitter_scale=1f-3`
    Ray jitter magnitude as a fraction of grid spacing (so physical jitter is
    `jitter_scale * step`). This helps avoid measure-zero edge/vertex cases.
• `ε_det=1f-10`
    Threshold for skipping triangles whose XY projection is nearly degenerate.
• `ε_bary=1f-7`
    Half-open barycentric margin (excludes edges to avoid double-counting).
• `dist_fallback=10f0`
    Used only if some voxels remain unassigned after JFA (rare if `band` is sane).
"""
function compute_sdf(mesh::Mesh{3,Float32,GLTriangleFace}, n::Int=256; kwargs...)
    return compute_sdf(coordinates(mesh), faces(mesh), n; kwargs...)
end

function compute_sdf(
    vertices::AbstractVector{<:GeometryBasics.Point{3}},
    fcs::AbstractVector,
    n::Int=256;
    band::Int=5,
    jfa_corrections::Int=2,
    jitter_scale::Float32=1f-3,
    ε_det::Float32=1f-10,
    ε_bary::Float32=1f-7,
    dist_fallback::Float32=10f0
)
    n32 = Int32(n)
    origin = -1f0
    step_val = 2f0 / Float32(n - 1)
    inv_step = inv(step_val)

    # geometry → SoA → GPU
    geom = preprocess_geometry(vertices, fcs)
    num_faces = geom.n_faces

    d_v0x = CuArray(geom.v0x)
    d_v0y = CuArray(geom.v0y)
    d_v0z = CuArray(geom.v0z)
    d_e0x = CuArray(geom.e0x)
    d_e0y = CuArray(geom.e0y)
    d_e0z = CuArray(geom.e0z)
    d_e1x = CuArray(geom.e1x)
    d_e1y = CuArray(geom.e1y)
    d_e1z = CuArray(geom.e1z)
    soa = (d_v0x, d_v0y, d_v0z, d_e0x, d_e0y, d_e0z, d_e1x, d_e1y, d_e1z)

    # launch configs
    tri_threads = 256
    tri_blocks = cld(Int(num_faces), tri_threads)
    blk3 = (8, 8, 4)
    grd3 = cld.(Int.((n32, n32, n32)), blk3)

    # phase 1: seed
    packed = CUDA.fill(SENTINEL_U64, n32, n32, n32)
    @cuda threads = tri_threads blocks = tri_blocks seed_kernel!(
        packed, soa..., origin, step_val, inv_step, n32, Int32(band), num_faces
    )
    # phase 1b: extract indices
    grid_a = CUDA.zeros(Int32, n32, n32, n32)
    @cuda threads = blk3 blocks = grd3 extract_indices_kernel!(grid_a, packed, n32)

    # phase 2: JFA
    grid_b = CUDA.zeros(Int32, n32, n32, n32)
    curr_in, curr_out = grid_a, grid_b

    jump = Int32(n ÷ 2)
    while jump >= Int32(1)
        @cuda threads = blk3 blocks = grd3 jfa_pass_kernel!(
            curr_out, curr_in, soa..., origin, step_val, n32, jump
        )
        (curr_in, curr_out) = (curr_out, curr_in)
        jump >>= Int32(1)
    end

    for _ in 1:jfa_corrections
        @cuda threads = blk3 blocks = grd3 jfa_pass_kernel!(
            curr_out, curr_in, soa..., origin, step_val, n32, Int32(1)
        )
        (curr_in, curr_out) = (curr_out, curr_in)
    end

    idx_grid = curr_in
    # (curr_out is a scratch buffer we can drop)

    # phase 3: parity rasterization
    parity = CUDA.zeros(UInt32, n32, n32, n32)
    jitter_mag = jitter_scale * step_val

    @cuda threads = tri_threads blocks = tri_blocks parity_kernel!(
        parity, soa..., origin, step_val, inv_step, n32, num_faces, jitter_mag, ε_det, ε_bary
    )
    # phase 4: finalize
    sdf = CUDA.zeros(Float32, n32, n32, n32)
    blk2 = (16, 16)
    grd2 = cld.(Int.((n32, n32)), blk2)

    @cuda threads = blk2 blocks = grd2 finalize_kernel!(
        sdf, idx_grid, parity, soa..., origin, step_val, n32, dist_fallback
    )
    return sdf
end
