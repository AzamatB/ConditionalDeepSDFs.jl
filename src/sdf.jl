"""
GPU-Accelerated Signed Distance Field (SDF)
══════════════════════════════════════════

This file computes a signed distance field on a uniform grid over the fixed
domain [-1, 1]³.

It is designed to be:

• Robust: sign is computed via ray parity (odd/even).
• Fast on GPU: distance uses narrow-band seeds + jump flooding (JFA) over
  triangle indices.

Pipeline (optimized version):
  Phase 1  — Seed:   **tile-binned gather seeding** within a voxel band (no global atomics).
  Phase 2  — JFA:    propagate nearest-triangle assignment (26-neighborhood), carrying
                     both triangle index and **exact dist² for the current winner**.
  Phase 3  — Parity: **tile-binned (x,y) column rasterization** for sign (no global atomics).
  Phase 4  — Final:  prefix-XOR parity per (x,y) column; **sqrt(dist²)**; apply sign.

Notes:
• Parity sign is orientation-independent (unlike winding-number methods).
• The JFA nearest-triangle assignment is an excellent approximation in practice
  but is not mathematically guaranteed exact for triangle distance; it is,
  however, far faster than brute force for large meshes.

Implementation highlights (vs. the original atomic-splat version):
• Seeding and parity both flip from triangle→voxel scatter (variable loops + atomics)
  to tile/voxel→triangle gather using a CPU-built CSR binning structure.
• Distances are stored as dist² and propagated through JFA so finalize does not
  recompute point–triangle distance.
"""

using CUDA
using GeometryBasics
using GLMakie: Figure, LScene, Screen, mesh!
using Meshing: MarchingCubes, MarchingTetrahedra, isosurface

# constants
const NO_TRIANGLE = Int32(0)   # 1-based triangle indices; 0 = unassigned

function construct_mesh(
    sdf::CuArray{Float32,3}, method::M=MarchingTetrahedra{Float32,Float32}()
) where {M<:Union{MarchingCubes{Float32},MarchingTetrahedra{Float32,Float32}}}
    sdf_cpu = Array(sdf)
    mesh = construct_mesh(sdf_cpu, method)
    return mesh::Mesh{3,Float32,TriangleFace{Int}}
end

function construct_mesh(
    sdf::Array{Float32,3}, method::M=MarchingTetrahedra{Float32,Float32}()
) where {M<:Union{MarchingCubes{Float32},MarchingTetrahedra{Float32,Float32}}}
    (vertices_t, faces_t) = isosurface(sdf, method)
    vertices = reinterpret(Point3f, vertices_t)
    faces = reinterpret(TriangleFace{Int}, faces_t)
    mesh = Mesh(vertices, faces)
    return mesh::Mesh{3,Float32,TriangleFace{Int}}
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
    acx::Float32, acy::Float32, acz::Float32
)
    apx = px - ax
    apy = py - ay
    apz = pz - az
    d1 = dot3(abx, aby, abz, apx, apy, apz)
    d2 = dot3(acx, acy, acz, apx, apy, apz)
    if (d1 <= 0.0f0) & (d2 <= 0.0f0)
        return dot3(apx, apy, apz, apx, apy, apz)
    end

    bpx = apx - abx
    bpy = apy - aby
    bpz = apz - abz
    d3 = dot3(abx, aby, abz, bpx, bpy, bpz)
    d4 = dot3(acx, acy, acz, bpx, bpy, bpz)
    if (d3 >= 0.0f0) & (d4 <= d3)
        return dot3(bpx, bpy, bpz, bpx, bpy, bpz)
    end

    cpx = apx - acx
    cpy = apy - acy
    cpz = apz - acz
    d5 = dot3(abx, aby, abz, cpx, cpy, cpz)
    d6 = dot3(acx, acy, acz, cpx, cpy, cpz)
    if (d6 >= 0.0f0) & (d5 <= d6)
        return dot3(cpx, cpy, cpz, cpx, cpy, cpz)
    end

    vc = d1 * d4 - d3 * d2
    if (vc <= 0.0f0) & (d1 >= 0.0f0) & (d3 <= 0.0f0)
        v = d1 / (d1 - d3)
        dx = apx - v * abx
        dy = apy - v * aby
        dz = apz - v * abz
        return dot3(dx, dy, dz, dx, dy, dz)
    end

    vb = d5 * d2 - d1 * d6
    if (vb <= 0.0f0) & (d2 >= 0.0f0) & (d6 <= 0.0f0)
        w = d2 / (d2 - d6)
        dx = apx - w * acx
        dy = apy - w * acy
        dz = apz - w * acz
        return dot3(dx, dy, dz, dx, dy, dz)
    end

    va = d3 * d6 - d5 * d4
    if (va <= 0.0f0) & ((d4 - d3) >= 0.0f0) & ((d5 - d6) >= 0.0f0)
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
    jx = (Float32(h & 0xFFFF) / 65_535.0f0 - 0.5f0) * mag
    jy = (Float32((h >> 16) & 0xFFFF) / 65_535.0f0 - 0.5f0) * mag
    return (jx, jy)
end

# Rasterization top-left ownership rule for a directed 2D edge.
@inline function is_top_left(dx::Float32, dy::Float32)::Bool
    return (dy > 0.0f0) | ((dy == 0.0f0) & (dx < 0.0f0))
end

"""Pack asymmetric parity edge-ownership bits for barycentric boundaries.

Bits:
  0x01 => owns `u = 0` edge (AC)
  0x02 => owns `v = 0` edge (AB)
  0x04 => owns `w = 0` edge (BC), where `w = 1 - u - v`
"""
@inline function pack_parity_edge_flags(
    abx::Float32, aby::Float32,
    acx::Float32, acy::Float32
)::UInt8
    det_xy = muladd(aby, acx, -abx * acy)
    # Choose a consistent CCW edge orientation for top-left ownership.
    s = ifelse(det_xy < 0.0f0, 1.0f0, -1.0f0)

    own_u = is_top_left(-s * acx, -s * acy)                   # AC boundary (u = 0)
    own_v = is_top_left(s * abx, s * aby)                     # AB boundary (v = 0)
    own_w = is_top_left(s * (acx - abx), s * (acy - aby))     # BC boundary (w = 0)

    flags = UInt8(0)
    own_u && (flags |= UInt8(0x01))
    own_v && (flags |= UInt8(0x02))
    own_w && (flags |= UInt8(0x04))
    return flags
end

#################################   CPU tiling / binning   #################################

"""Build a 3D tile→triangles CSR for narrow-band seeding.

Tiles are axis-aligned bricks of size (Tx,Ty,Tz) in voxel index space.
A triangle contributes to every tile overlapped by its (voxel) AABB expanded by `band`.

Returns:
  offsets::Vector{Int32}  (length num_tiles+1, 1-based CSR offsets)
  tris::Vector{Int32}     (flattened triangle indices)
  active_tiles::Vector{Int32} (tile ids with nonzero triangle counts)
  ntx, nty, ntz::Int32    (tile grid dimensions)
"""
function build_seed_tile_csr(
    v0x::Vector{Float32}, v0y::Vector{Float32}, v0z::Vector{Float32},
    e0x::Vector{Float32}, e0y::Vector{Float32}, e0z::Vector{Float32},
    e1x::Vector{Float32}, e1y::Vector{Float32}, e1z::Vector{Float32},
    origin::Float32, inv_step::Float32,
    n::Int32, band::Int32,
    Tx::Int32, Ty::Int32, Tz::Int32
)
    ntx = Int32(cld(Int(n), Int(Tx)))
    nty = Int32(cld(Int(n), Int(Ty)))
    ntz = Int32(cld(Int(n), Int(Tz)))
    num_tiles = Int(ntx) * Int(nty) * Int(ntz)

    counts = zeros(Int32, num_tiles)
    num_faces = length(v0x)
    # Cache per-face tile ranges so we don't repeat heavy AABB/range math in the fill pass.
    # (one contiguous stream is typically friendlier to caches/prefetch than 6 separate arrays)
    ranges = Vector{NTuple{6,Int32}}(undef, num_faces)  # (tx0,tx1,ty0,ty1,tz0,tz1)
    txy = ntx * nty

    @inbounds for fi in 1:num_faces
        ax = v0x[fi]
        ay = v0y[fi]
        az = v0z[fi]
        abx = e0x[fi]
        aby = e0y[fi]
        abz = e0z[fi]
        acx = e1x[fi]
        acy = e1y[fi]
        acz = e1z[fi]

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

        # tile ranges (0-based)
        tx0 = max((i0 - Int32(1)) ÷ Tx, Int32(0))
        tx1 = min((i1 - Int32(1)) ÷ Tx, ntx - Int32(1))
        ty0 = max((j0 - Int32(1)) ÷ Ty, Int32(0))
        ty1 = min((j1 - Int32(1)) ÷ Ty, nty - Int32(1))
        tz0 = max((k0 - Int32(1)) ÷ Tz, Int32(0))
        tz1 = min((k1 - Int32(1)) ÷ Tz, ntz - Int32(1))

        ranges[fi] = (tx0, tx1, ty0, ty1, tz0, tz1)

        for tz in tz0:tz1
            base_tz = tz * txy
            for ty in ty0:ty1
                base_ty = base_tz + ty * ntx
                for tx in tx0:tx1
                    tile = Int(base_ty + tx + Int32(1))
                    counts[tile] += Int32(1)
                end
            end
        end
    end

    # CSR offsets (1-based)
    offsets = Vector{Int32}(undef, num_tiles + 1)
    offsets[1] = Int32(1)
    @inbounds for t in eachindex(counts)
        offsets[t+1] = offsets[t] + counts[t]
    end
    total_refs = Int(offsets[end] - Int32(1))
    tris = Vector{Int32}(undef, total_refs)

    write_ptr = copy(offsets[1:end-1])

    @inbounds for fi in 1:num_faces
        tx0, tx1, ty0, ty1, tz0, tz1 = ranges[fi]

        fi32 = Int32(fi)
        for tz in tz0:tz1
            base_tz = tz * txy
            for ty in ty0:ty1
                base_ty = base_tz + ty * ntx
                for tx in tx0:tx1
                    tile = Int(base_ty + tx + Int32(1))
                    pos = write_ptr[tile]
                    tris[Int(pos)] = fi32
                    write_ptr[tile] = pos + Int32(1)
                end
            end
        end
    end

    # list active tiles (for smaller GPU launch)
    active_tiles = Vector{Int32}()
    sizehint!(active_tiles, num_tiles)
    @inbounds for t in eachindex(counts)
        (counts[t] == Int32(0)) && continue
        push!(active_tiles, Int32(t))
    end

    return (offsets, tris, active_tiles, ntx, nty, ntz)
end

"""Build a 2D tile→triangles CSR for parity rasterization (vertical +z rays).

Tiles are (Tx,Ty) bricks in the XY plane in voxel index space.
Triangles are binned by their XY AABB (expanded by `jitter_mag`).
Triangles whose XY projection is nearly degenerate (|det| <= ε_det) are skipped.

Returns:
  offsets::Vector{Int32}  (length num_tiles+1, 1-based CSR offsets)
  tris::Vector{Int32}     (flattened triangle indices)
  active_tiles::Vector{Int32} (tile ids with nonzero triangle counts)
  ntx, nty::Int32         (tile grid dimensions)
"""
function build_parity_tile_csr(
    v0x::Vector{Float32}, v0y::Vector{Float32}, v0z::Vector{Float32},
    e0x::Vector{Float32}, e0y::Vector{Float32}, e0z::Vector{Float32},
    e1x::Vector{Float32}, e1y::Vector{Float32}, e1z::Vector{Float32},
    origin::Float32, step_val::Float32, inv_step::Float32,
    n::Int32,
    Tx::Int32, Ty::Int32,
    jitter_mag::Float32,
    ε_det::Float32
)
    ntx = Int32(cld(Int(n), Int(Tx)))
    nty = Int32(cld(Int(n), Int(Ty)))
    num_tiles = Int(ntx) * Int(nty)

    counts = zeros(Int32, num_tiles)
    domain_max = origin + step_val * Float32(n - Int32(1))
    num_faces = length(v0x)
    # Cache per-face tile ranges so we don't redo projection/AABB math in the fill pass.
    # Degenerate XY projections are stored as an empty range (so both passes naturally skip them).
    ranges = Vector{NTuple{4,Int32}}(undef, num_faces)  # (tx0,tx1,ty0,ty1)

    empty_range = (Int32(1), Int32(0), Int32(1), Int32(0))

    @inbounds for fi in 1:num_faces
        ax = v0x[fi]
        ay = v0y[fi]
        abx = e0x[fi]
        aby = e0y[fi]
        acx = e1x[fi]
        acy = e1y[fi]

        det = muladd(aby, acx, -abx * acy)
        if abs(det) <= ε_det
            ranges[fi] = empty_range
            continue
        end

        bx = ax + abx
        by = ay + aby
        cx = ax + acx
        cy = ay + acy

        x_min = max(min(ax, bx, cx) - jitter_mag, origin)
        x_max = min(max(ax, bx, cx) + jitter_mag, domain_max)
        y_min = max(min(ay, by, cy) - jitter_mag, origin)
        y_max = min(max(ay, by, cy) + jitter_mag, domain_max)

        # match the kernel's convention
        ix0 = max(unsafe_trunc(Int32, (x_min - origin) * inv_step) + Int32(1), Int32(1))
        ix1 = min(unsafe_trunc(Int32, (x_max - origin) * inv_step) + Int32(2), n)
        iy0 = max(unsafe_trunc(Int32, (y_min - origin) * inv_step) + Int32(1), Int32(1))
        iy1 = min(unsafe_trunc(Int32, (y_max - origin) * inv_step) + Int32(2), n)

        tx0 = max((ix0 - Int32(1)) ÷ Tx, Int32(0))
        tx1 = min((ix1 - Int32(1)) ÷ Tx, ntx - Int32(1))
        ty0 = max((iy0 - Int32(1)) ÷ Ty, Int32(0))
        ty1 = min((iy1 - Int32(1)) ÷ Ty, nty - Int32(1))

        ranges[fi] = (tx0, tx1, ty0, ty1)

        for ty in ty0:ty1
            base_ty = ty * ntx
            for tx in tx0:tx1
                tile = Int(base_ty + tx + Int32(1))
                counts[tile] += Int32(1)
            end
        end
    end

    offsets = Vector{Int32}(undef, num_tiles + 1)
    offsets[1] = Int32(1)
    @inbounds for t in eachindex(counts)
        offsets[t+1] = offsets[t] + counts[t]
    end
    total_refs = Int(offsets[end] - Int32(1))
    tris = Vector{Int32}(undef, total_refs)

    write_ptr = copy(offsets[1:end-1])

    @inbounds for fi in 1:num_faces
        tx0, tx1, ty0, ty1 = ranges[fi]

        fi32 = Int32(fi)
        for ty in ty0:ty1
            base_ty = ty * ntx
            for tx in tx0:tx1
                tile = Int(base_ty + tx + Int32(1))
                pos = write_ptr[tile]
                tris[Int(pos)] = fi32
                write_ptr[tile] = pos + Int32(1)
            end
        end
    end

    active_tiles = Vector{Int32}()
    sizehint!(active_tiles, num_tiles)
    @inbounds for t in eachindex(counts)
        (counts[t] == Int32(0)) && continue
        push!(active_tiles, Int32(t))
    end

    return (offsets, tris, active_tiles, ntx, nty)
end

###############################   Phase 1 — tiled seeding   ###############################

"""Tile-binned narrow-band seed kernel.

Each block processes one active 3D tile. Threads correspond to voxels in the tile.
Each voxel gathers candidate triangles from the tile's CSR list, computes exact
point–triangle dist², and writes the best (idx, dist²).

No atomics; each voxel is owned by exactly one thread.
"""
function seed_tiled_kernel!(
    idx::CuDeviceArray{Int32,3},
    d2::CuDeviceArray{Float32,3},
    active_tiles::CuDeviceVector{Int32},
    tile_offsets::CuDeviceVector{Int32},
    tile_tris::CuDeviceVector{Int32},
    v0x::CuDeviceVector{Float32}, v0y::CuDeviceVector{Float32}, v0z::CuDeviceVector{Float32},
    e0x::CuDeviceVector{Float32}, e0y::CuDeviceVector{Float32}, e0z::CuDeviceVector{Float32},
    e1x::CuDeviceVector{Float32}, e1y::CuDeviceVector{Float32}, e1z::CuDeviceVector{Float32},
    origin::Float32, step_val::Float32,
    n::Int32,
    ntx::Int32, nty::Int32,
    Tx::Int32, Ty::Int32, Tz::Int32
)
    # map block -> global tile id (1-based)
    tile_id = @inbounds active_tiles[blockIdx().x]

    # 0-based tile coordinates
    t0 = tile_id - Int32(1)
    tx = t0 % ntx
    t1 = t0 ÷ ntx
    ty = t1 % nty
    tz = t1 ÷ nty

    ix = tx * Tx + threadIdx().x
    iy = ty * Ty + threadIdx().y
    iz = tz * Tz + threadIdx().z

    ((ix > n) | (iy > n) | (iz > n)) && return nothing

    px = muladd(Float32(ix - Int32(1)), step_val, origin)
    py = muladd(Float32(iy - Int32(1)), step_val, origin)
    pz = muladd(Float32(iz - Int32(1)), step_val, origin)

    best_d2 = Inf32
    best_idx = NO_TRIANGLE

    start = @inbounds tile_offsets[tile_id]
    stop = @inbounds tile_offsets[tile_id+Int32(1)] - Int32(1)

    @inbounds for ptr in start:stop
        fi = tile_tris[ptr]

        ax = v0x[fi]
        ay = v0y[fi]
        az = v0z[fi]
        abx = e0x[fi]
        aby = e0y[fi]
        abz = e0z[fi]
        acx = e1x[fi]
        acy = e1y[fi]
        acz = e1z[fi]

        dd = dist²_point_triangle(px, py, pz, ax, ay, az, abx, aby, abz, acx, acy, acz)
        if dd < best_d2
            best_d2 = dd
            best_idx = fi
        end
    end

    @inbounds begin
        idx[ix, iy, iz] = best_idx
        d2[ix, iy, iz] = best_d2
    end

    return nothing
end

###############   Phase 2 — JFA pass (propagate triangle indices + dist²)   ###############

function jfa_pass_kernel!(
    idx_out::CuDeviceArray{Int32,3}, d2_out::CuDeviceArray{Float32,3},
    idx_in::CuDeviceArray{Int32,3}, d2_in::CuDeviceArray{Float32,3},
    v0x::CuDeviceVector{Float32}, v0y::CuDeviceVector{Float32}, v0z::CuDeviceVector{Float32},
    e0x::CuDeviceVector{Float32}, e0y::CuDeviceVector{Float32}, e0z::CuDeviceVector{Float32},
    e1x::CuDeviceVector{Float32}, e1y::CuDeviceVector{Float32}, e1z::CuDeviceVector{Float32},
    origin::Float32, step_val::Float32, n::Int32, jump::Int32
)
    ix = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    iz = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z
    ((ix > n) | (iy > n) | (iz > n)) && return nothing

    px = muladd(Float32(ix - Int32(1)), step_val, origin)
    py = muladd(Float32(iy - Int32(1)), step_val, origin)
    pz = muladd(Float32(iz - Int32(1)), step_val, origin)

    @inbounds best_idx = idx_in[ix, iy, iz]
    @inbounds best_d2 = d2_in[ix, iy, iz]

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

                nb = idx_in[nx, ny, nz]
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

                dd = dist²_point_triangle(px, py, pz, ax, ay, az, abx, aby, abz, acx, acy, acz)
                if dd < best_d2
                    best_d2 = dd
                    best_idx = nb
                end
            end
        end
    end

    @inbounds begin
        idx_out[ix, iy, iz] = best_idx
        d2_out[ix, iy, iz] = best_d2
    end
    return nothing
end

###############   Phase 3 — parity rasterization (tile-binned, no atomics)   ###############

"""Tile-binned parity rasterization.

Each block processes one active XY tile; each thread owns one (ix,iy) column.
Threads iterate triangles binned to the tile and toggle parity at the hit sample
index along +z.

Because each (ix,iy) column is owned by exactly one thread, toggles can be plain
XOR stores (no global atomics).
"""
function parity_tiled_kernel!(
    parity::CuDeviceArray{UInt8,3},
    active_tiles::CuDeviceVector{Int32},
    tile_offsets::CuDeviceVector{Int32},
    tile_tris::CuDeviceVector{Int32},
    edge_flags::CuDeviceVector{UInt8},
    v0x::CuDeviceVector{Float32}, v0y::CuDeviceVector{Float32}, v0z::CuDeviceVector{Float32},
    e0x::CuDeviceVector{Float32}, e0y::CuDeviceVector{Float32}, e0z::CuDeviceVector{Float32},
    e1x::CuDeviceVector{Float32}, e1y::CuDeviceVector{Float32}, e1z::CuDeviceVector{Float32},
    origin::Float32, step_val::Float32, inv_step::Float32,
    n::Int32,
    ntx::Int32,
    Tx::Int32, Ty::Int32,
    jitter_mag::Float32,
    ε_det::Float32,
    ε_bary::Float32
)
    tile_id = @inbounds active_tiles[blockIdx().x]

    # 0-based tile coords
    t0 = tile_id - Int32(1)
    tx = t0 % ntx
    ty = t0 ÷ ntx

    ix = tx * Tx + threadIdx().x
    iy = ty * Ty + threadIdx().y
    ((ix > n) | (iy > n)) && return nothing

    # jittered ray origin (x,y)
    (jx, jy) = column_jitter(ix, iy, jitter_mag)
    rx = muladd(Float32(ix - Int32(1)), step_val, origin) + jx
    ry = muladd(Float32(iy - Int32(1)), step_val, origin) + jy

    z_end = origin + step_val * Float32(n - Int32(1))

    start = @inbounds tile_offsets[tile_id]
    stop = @inbounds tile_offsets[tile_id+Int32(1)] - Int32(1)

    @inbounds for ptr in start:stop
        fi = tile_tris[ptr]
        flags = edge_flags[fi]

        ax = v0x[fi]
        ay = v0y[fi]
        az = v0z[fi]
        abx = e0x[fi]
        aby = e0y[fi]
        abz = e0z[fi]
        acx = e1x[fi]
        acy = e1y[fi]
        acz = e1z[fi]

        # det = aby*acx - abx*acy
        det = muladd(aby, acx, -abx * acy)
        (abs(det) <= ε_det) && continue
        inv_det = inv(det)

        sx = rx - ax
        sy = ry - ay
        u = (muladd(sy, acx, -sx * acy)) * inv_det
        v = (muladd(sx, aby, -sy * abx)) * inv_det

        # Asymmetric half-open rule via per-edge ownership flags.
        # Boundary points are accepted only by the owning edge.
        w = 1.0f0 - u - v

        own_u = (flags & UInt8(0x01)) != UInt8(0)  # u = 0 edge (AC)
        own_v = (flags & UInt8(0x02)) != UInt8(0)  # v = 0 edge (AB)
        own_w = (flags & UInt8(0x04)) != UInt8(0)  # w = 0 edge (BC)

        abs_u = abs(u)
        abs_v = abs(v)
        abs_w = abs(w)

        in_u = (u > ε_bary) | ((abs_u <= ε_bary) & own_u)
        in_v = (v > ε_bary) | ((abs_v <= ε_bary) & own_v)
        in_w = (w > ε_bary) | ((abs_w <= ε_bary) & own_w)

        if !(in_u & in_v & in_w)
            continue
        end

        z_hit = az + u * abz + v * acz
        isfinite(z_hit) || continue

        # Half-open in z: only hits strictly inside (origin, z_end)
        if (z_hit <= origin) | (z_hit >= z_end)
            continue
        end

        # toggle at the first grid sample strictly above z_hit
        t = (z_hit - origin) * inv_step  # in (0, n-1)
        z_idx = unsafe_trunc(Int32, t) + Int32(2)

        if (z_idx >= Int32(1)) & (z_idx <= n)
            @inbounds parity[ix, iy, z_idx] = parity[ix, iy, z_idx] ⊻ UInt8(1)
        end
    end

    return nothing
end

#################   Phase 4 — finalize (prefix XOR parity + sqrt(dist²))   #################

function finalize_kernel!(
    sdf::CuDeviceArray{Float32,3},
    d2_grid::CuDeviceArray{Float32,3},
    parity::CuDeviceArray{UInt8,3},
    origin::Float32, step_val::Float32, n::Int32,
    dist_fallback::Float32
)
    ix = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    ((ix > n) | (iy > n)) && return nothing

    parity_acc = UInt8(0)
    iz = Int32(1)
    while iz <= n
        @inbounds parity_acc ⊻= parity[ix, iy, iz]
        is_inside = (parity_acc & UInt8(1)) == UInt8(1)

        @inbounds dd = d2_grid[ix, iy, iz]
        d = (dd < Inf32) ? sqrt(dd) : dist_fallback

        @inbounds sdf[ix, iy, iz] = is_inside ? -d : d

        iz += Int32(1)
    end

    return nothing
end

####################   CPU preprocessing: mesh → SoA triangle arrays   ####################

"""Preprocess mesh into SoA Float32 arrays: vertex A + edges AB, AC.

• Edge vectors are computed in Float64 first to reduce cancellation.
• Degenerate triangles are dropped.
• Precomputes parity edge-ownership flags for asymmetric boundary tie-break.
"""
function preprocess_geometry(
    vertices::AbstractVector{Point3f},
    fcs::AbstractVector{GLTriangleFace};
    ε²::Float64=1.0e-20
)
    num_faces = length(fcs)
    arrays = ntuple(_ -> sizehint!(Float32[], num_faces), 9)
    (v0x, v0y, v0z, e0x, e0y, e0z, e1x, e1y, e1z) = arrays
    parity_edge_flags = sizehint!(UInt8[], num_faces)

    @inbounds for face in fcs
        # works for TriangleFace / GLTriangleFace; we only need integer indices
        (a, b, c) = GeometryBasics.value.(face)
        A = Float64.(vertices[a])
        ab = Float64.(vertices[b]) .- A
        ac = Float64.(vertices[c]) .- A
        nx = ab[2] * ac[3] - ab[3] * ac[2]
        ny = ab[3] * ac[1] - ab[1] * ac[3]
        nz = ab[1] * ac[2] - ab[2] * ac[1]
        norm² = muladd(nx, nx, muladd(ny, ny, nz * nz))  # = nx*nx + ny*ny + nz*nz
        (norm² <= ε²) && continue

        ax = Float32(A[1])
        ay = Float32(A[2])
        az = Float32(A[3])
        abx32 = Float32(ab[1])
        aby32 = Float32(ab[2])
        abz32 = Float32(ab[3])
        acx32 = Float32(ac[1])
        acy32 = Float32(ac[2])
        acz32 = Float32(ac[3])

        push!(v0x, ax)
        push!(v0y, ay)
        push!(v0z, az)
        push!(e0x, abx32)
        push!(e0y, aby32)
        push!(e0z, abz32)
        push!(e1x, acx32)
        push!(e1y, acy32)
        push!(e1z, acz32)
        push!(parity_edge_flags, pack_parity_edge_flags(abx32, aby32, acx32, acy32))
    end

    num_faces = Int32(length(v0x))
    (num_faces > 0) || error("All faces degenerate after filtering")
    return (;
        v0x, v0y, v0z,
        e0x, e0y, e0z,
        e1x, e1y, e1z,
        parity_edge_flags,
        num_faces
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
• `jitter_scale=1.0f-3`
    Ray jitter magnitude as a fraction of grid spacing (so physical jitter is
    `jitter_scale * step`). This helps avoid measure-zero edge/vertex cases.
    For meshes that show a rare parity-column artifact, reducing this value
    (or setting it to 0) can be used as a workaround.
• `ε_det=1.0f-10`
    Threshold for skipping triangles whose XY projection is nearly degenerate.
• `ε_bary=5.0f-8`
    Boundary tolerance for asymmetric edge ownership tie-break.
    At `|u|,|v|,|w| <= ε_bary`, only the owner edge accepts the hit.
    Smaller values reduce boundary snapping; larger values increase tie-band width.
• `dist_fallback=10.0f0`
    Used only if some voxels remain unassigned after JFA (rare if `band` is sane).
"""
function construct_sdf(mesh::Mesh{3,Float32,GLTriangleFace}, n::Int=256; kwargs...)
    return construct_sdf(coordinates(mesh), faces(mesh), n; kwargs...)
end

function construct_sdf(
    vertices::AbstractVector{Point3f},
    fcs::AbstractVector{GLTriangleFace},
    n::Int=256;
    band::Int=5,
    jfa_corrections::Int=2,
    jitter_scale::Float32=1.0f-3,
    ε_det::Float32=1.0f-10,
    ε_bary::Float32=5.0f-8,
    dist_fallback::Float32=10.0f0
)
    (7 < n < 1025) || error("Grid size n must be between 7 and 1024!")
    n32 = Int32(n)
    origin = -1.0f0
    step_val = 2.0f0 / Float32(n - 1)
    inv_step = inv(step_val)

    # geometry → SoA (CPU)
    geom = preprocess_geometry(vertices, fcs)

    # phase 1 (CPU): build tile bins
    # seeding tiles (3D)
    seed_blk = (8, 8, 4)
    Tx_s = Int32(seed_blk[1])
    Ty_s = Int32(seed_blk[2])
    Tz_s = Int32(seed_blk[3])

    (seed_offsets, seed_tris, seed_active, seed_ntx, seed_nty, seed_ntz) = build_seed_tile_csr(
        geom.v0x, geom.v0y, geom.v0z,
        geom.e0x, geom.e0y, geom.e0z,
        geom.e1x, geom.e1y, geom.e1z,
        origin, inv_step, n32, Int32(band), Tx_s, Ty_s, Tz_s
    )

    # parity tiles (2D)
    parity_blk = (16, 16)
    Tx_p = Int32(parity_blk[1])
    Ty_p = Int32(parity_blk[2])
    jitter_mag = jitter_scale * step_val

    (parity_offsets, parity_tris, parity_active, parity_ntx, parity_nty) = build_parity_tile_csr(
        geom.v0x, geom.v0y, geom.v0z,
        geom.e0x, geom.e0y, geom.e0z,
        geom.e1x, geom.e1y, geom.e1z,
        origin, step_val, inv_step, n32, Tx_p, Ty_p, jitter_mag, ε_det
    )

    # upload geometry + bins to GPU
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

    d_seed_offsets = CuArray(seed_offsets)
    d_seed_tris = CuArray(seed_tris)
    d_seed_active = CuArray(seed_active)

    d_parity_offsets = CuArray(parity_offsets)
    d_parity_tris = CuArray(parity_tris)
    d_parity_active = CuArray(parity_active)
    d_parity_edge_flags = CuArray(geom.parity_edge_flags)

    # phase 1 (GPU): tiled seeding
    idx_a = CUDA.zeros(Int32, n32, n32, n32)
    d2_a = CUDA.fill(Inf32, n32, n32, n32)

    # blocks = number of active tiles (1D)
    @cuda threads = seed_blk blocks = length(seed_active) seed_tiled_kernel!(
        idx_a, d2_a, d_seed_active, d_seed_offsets, d_seed_tris, soa...,
        origin, step_val, n32, seed_ntx, seed_nty, Tx_s, Ty_s, Tz_s
    )

    # phase 2 (GPU): JFA (propagate idx + dist²)
    idx_b = CUDA.zeros(Int32, n32, n32, n32)
    d2_b = CUDA.fill(Inf32, n32, n32, n32)

    blk3 = seed_blk
    grd3 = cld.(Int.((n32, n32, n32)), blk3)

    (curr_idx_in, curr_d2_in) = (idx_a, d2_a)
    (curr_idx_out, curr_d2_out) = (idx_b, d2_b)

    jump = Int32(n ÷ 2)
    while jump >= Int32(1)
        @cuda threads = blk3 blocks = grd3 jfa_pass_kernel!(
            curr_idx_out, curr_d2_out, curr_idx_in, curr_d2_in, soa...,
            origin, step_val, n32, jump
        )
        (curr_idx_in, curr_idx_out) = (curr_idx_out, curr_idx_in)
        (curr_d2_in, curr_d2_out) = (curr_d2_out, curr_d2_in)
        jump >>= Int32(1)
    end

    for _ in 1:jfa_corrections
        @cuda threads = blk3 blocks = grd3 jfa_pass_kernel!(
            curr_idx_out, curr_d2_out, curr_idx_in, curr_d2_in, soa...,
            origin, step_val, n32, Int32(1)
        )
        (curr_idx_in, curr_idx_out) = (curr_idx_out, curr_idx_in)
        (curr_d2_in, curr_d2_out) = (curr_d2_out, curr_d2_in)
    end
    d2_grid = curr_d2_in

    # phase 3 (GPU): parity rasterization (tile-binned, no atomics)
    parity = CUDA.zeros(UInt8, n32, n32, n32)

    @cuda threads = parity_blk blocks = length(parity_active) parity_tiled_kernel!(
        parity, d_parity_active, d_parity_offsets, d_parity_tris, d_parity_edge_flags, soa...,
        origin, step_val, inv_step, n32, parity_ntx, Tx_p, Ty_p, jitter_mag, ε_det, ε_bary
    )

    # phase 4 (GPU): finalize (prefix XOR parity + sqrt(dist²) + sign)
    sdf = CUDA.zeros(Float32, n32, n32, n32)
    blk2 = (16, 16)
    grd2 = cld.(Int.((n32, n32)), blk2)

    @cuda threads = blk2 blocks = grd2 finalize_kernel!(
        sdf, d2_grid, parity, origin, step_val, n32, dist_fallback
    )
    return Array(sdf)::Array{Float32,3}
end
