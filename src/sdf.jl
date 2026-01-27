using CUDA
using GeometryBasics
using LinearAlgebra

# ----------------------------
# CPU helpers (Float64)
# ----------------------------

function dot3(x1::Float64, y1::Float64, z1::Float64, x2::Float64, y2::Float64, z2::Float64)
    value = x1*x2 + y1*y2 + z1*z2
    return value::Float64
end

function norm3(x::Float64, y::Float64, z::Float64)
    value = sqrt(x*x + y*y + z*z)
    return value::Float64
end

function angle_between(
    u1::Float64, u2::Float64, u3::Float64, v1::Float64, v2::Float64, v3::Float64
)
    nu = norm3(u1, u2, u3)
    nv = norm3(v1, v2, v3)
    if nu == 0.0 || nv == 0.0
        return 0.0
    end

    c = dot3(u1, u2, u3, v1, v2, v3) / (nu * nv)
    c = clamp(c, -1.0, 1.0)
    θ = acos(c)
    return θ::Float64
end

function edge_key(i::Int32, j::Int32)::UInt64
    a = i < j ? i : j
    b = i < j ? j : i
    key = (UInt64(a) << 32) | UInt64(b)
    return key
end

# ----------------------------
# GPU helpers (Float32)
# ----------------------------

function dot3f(ax::Float32, ay::Float32, az::Float32, bx::Float32, by::Float32, bz::Float32)
    value = muladd(ax, bx, muladd(ay, by, az * bz))
    return value::Float32
end

"""
Squared distance from point p to triangle (a, a+ab, a+ac).

This is the Ericson (Real-Time Collision Detection) Voronoi-region test,
implemented without allocating vectors.

Inputs (Float32):
  p = (px,py,pz)
  a = (ax,ay,az)
  ab = (abx,aby,abz) where b = a + ab
  ac = (acx,acy,acz) where c = a + ac
"""
function dist2_point_triangle(
    px::Float32, py::Float32, pz::Float32,
    ax::Float32, ay::Float32, az::Float32,
    abx::Float32, aby::Float32, abz::Float32,
    acx::Float32, acy::Float32, acz::Float32
)
    apx = px - ax
    apy = py - ay
    apz = pz - az

    d1 = dot3f(abx, aby, abz, apx, apy, apz)
    d2 = dot3f(acx, acy, acz, apx, apy, apz)

    if (d1 <= 0f0) & (d2 <= 0f0)
        return dot3f(apx, apy, apz, apx, apy, apz)  # closest to A
    end

    bpx = apx - abx
    bpy = apy - aby
    bpz = apz - abz

    d3 = dot3f(abx, aby, abz, bpx, bpy, bpz)
    d4 = dot3f(acx, acy, acz, bpx, bpy, bpz)

    if (d3 >= 0f0) & (d4 <= d3)
        return dot3f(bpx, bpy, bpz, bpx, bpy, bpz)  # closest to B
    end

    cpx = apx - acx
    cpy = apy - acy
    cpz = apz - acz

    d5 = dot3f(abx, aby, abz, cpx, cpy, cpz)
    d6 = dot3f(acx, acy, acz, cpx, cpy, cpz)

    if (d6 >= 0f0) & (d5 <= d6)
        return dot3f(cpx, cpy, cpz, cpx, cpy, cpz)  # closest to C
    end

    vc = d1*d4 - d3*d2
    if (vc <= 0f0) & (d1 >= 0f0) & (d3 <= 0f0)
        v = d1 / (d1 - d3)                         # on AB
        dx = apx - v*abx
        dy = apy - v*aby
        dz = apz - v*abz
        return dot3f(dx, dy, dz, dx, dy, dz)
    end

    vb = d5*d2 - d1*d6
    if (vb <= 0f0) & (d2 >= 0f0) & (d6 <= 0f0)
        w = d2 / (d2 - d6)                         # on AC
        dx = apx - w*acx
        dy = apy - w*acy
        dz = apz - w*acz
        return dot3f(dx, dy, dz, dx, dy, dz)
    end

    va = d3*d6 - d5*d4
    if (va <= 0f0) & ((d4 - d3) >= 0f0) & ((d5 - d6) >= 0f0)
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))    # on BC
        bcx = acx - abx
        bcy = acy - aby
        bcz = acz - abz
        dx = bpx - w*bcx
        dy = bpy - w*bcy
        dz = bpz - w*bcz
        return dot3f(dx, dy, dz, dx, dy, dz)
    end

    # inside face region
    denom = inv(va + vb + vc)
    v = vb * denom
    w = vc * denom

    dx = apx - v*abx - w*acx
    dy = apy - v*aby - w*acy
    dz = apz - v*abz - w*acz

    result = dot3f(dx, dy, dz, dx, dy, dz)
    return result::Float32
end

"""
Closest point q on triangle and a region code (UInt8) identifying feature:

  0 = face interior
  1 = vertex A
  2 = vertex B
  3 = vertex C
  4 = edge AB
  5 = edge AC
  6 = edge BC
"""
function closest_point_region(
    px::Float32, py::Float32, pz::Float32,
    ax::Float32, ay::Float32, az::Float32,
    abx::Float32, aby::Float32, abz::Float32,
    acx::Float32, acy::Float32, acz::Float32
)
    apx = px - ax
    apy = py - ay
    apz = pz - az

    d1 = dot3f(abx, aby, abz, apx, apy, apz)
    d2 = dot3f(acx, acy, acz, apx, apy, apz)

    if (d1 <= 0f0) & (d2 <= 0f0)
        return ax, ay, az, UInt8(1)
    end

    bpx = apx - abx
    bpy = apy - aby
    bpz = apz - abz

    d3 = dot3f(abx, aby, abz, bpx, bpy, bpz)
    d4 = dot3f(acx, acy, acz, bpx, bpy, bpz)

    if (d3 >= 0f0) & (d4 <= d3)
        return ax + abx, ay + aby, az + abz, UInt8(2)
    end

    cpx = apx - acx
    cpy = apy - acy
    cpz = apz - acz

    d5 = dot3f(abx, aby, abz, cpx, cpy, cpz)
    d6 = dot3f(acx, acy, acz, cpx, cpy, cpz)

    if (d6 >= 0f0) & (d5 <= d6)
        return ax + acx, ay + acy, az + acz, UInt8(3)
    end

    vc = d1*d4 - d3*d2
    if (vc <= 0f0) & (d1 >= 0f0) & (d3 <= 0f0)
        v = d1 / (d1 - d3)
        qx = muladd(v, abx, ax)
        qy = muladd(v, aby, ay)
        qz = muladd(v, abz, az)
        return qx, qy, qz, UInt8(4)
    end

    vb = d5*d2 - d1*d6
    if (vb <= 0f0) & (d2 >= 0f0) & (d6 <= 0f0)
        w = d2 / (d2 - d6)
        qx = muladd(w, acx, ax)
        qy = muladd(w, acy, ay)
        qz = muladd(w, acz, az)
        return qx, qy, qz, UInt8(5)
    end

    va = d3*d6 - d5*d4
    if (va <= 0f0) & ((d4 - d3) >= 0f0) & ((d5 - d6) >= 0f0)
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        bcx = acx - abx
        bcy = acy - aby
        bcz = acz - abz
        bx = ax + abx
        by = ay + aby
        bz = az + abz
        qx = muladd(w, bcx, bx)
        qy = muladd(w, bcy, by)
        qz = muladd(w, bcz, bz)
        return qx, qy, qz, UInt8(6)
    end

    denom = inv(va + vb + vc)
    v = vb * denom
    w = vc * denom

    qx = muladd(w, acx, muladd(v, abx, ax))
    qy = muladd(w, acy, muladd(v, aby, ay))
    qz = muladd(w, acz, muladd(v, abz, az))

    return qx, qy, qz, UInt8(0)
end

# =============================================================================
# CPU preprocessing
# =============================================================================

"""
Precompute packed triangle geometry + pseudo-normals.

Returns a NamedTuple of CPU arrays:
  - indices i0,i1,i2 (Int32)
  - geometry v0 + e0(ab) + e1(ac) (Float32)
  - face normals per tri (Float32)
  - edge normals per tri for AB, AC, BC (Float32)
  - vertex pseudo-normals per vertex (Float32)

Degenerate faces (very small area) are dropped.
"""
function precompute_data_on_cpu(
    vertices::Vector{Point3{Float32}},
    faces::Vector{<:NgonFace{3}};
    degenerate_area2_eps::Float64 = 1e-20
)
    n_verts = length(vertices)

    # Determine whether faces are 0-based or 1-based
    min_idx = typemax(Int)
    @inbounds for f in faces
        min_idx = min(min_idx, Int(f[1]), Int(f[2]), Int(f[3]))
    end
    offset = (min_idx == 0) ? 1 : 0

    # Accumulators for vertex pseudo-normals (Float64)
    vnx_acc = zeros(Float64, n_verts)
    vny_acc = zeros(Float64, n_verts)
    vnz_acc = zeros(Float64, n_verts)

    # Edge normal sums (sum of incident unit face normals)
    edge_acc = Dict{UInt64, NTuple{3,Float64}}()

    # Packed triangles (skip degenerates)
    i0 = Int32[]
    i1 = Int32[]
    i2 = Int32[]

    v0x = Float32[]
    v0y = Float32[]
    v0z = Float32[]
    e0x = Float32[]
    e0y = Float32[]
    e0z = Float32[]
    e1x = Float32[]
    e1y = Float32[]
    e1z = Float32[]

    fnx = Float32[]
    fny = Float32[]
    fnz = Float32[]

    @inbounds for f in faces
        a = Int32(Int(f[1]) + offset)
        b = Int32(Int(f[2]) + offset)
        c = Int32(Int(f[3]) + offset)

        ax = Float64(vertices[a][1])
        ay = Float64(vertices[a][2])
        az = Float64(vertices[a][3])

        bx = Float64(vertices[b][1])
        by = Float64(vertices[b][2])
        bz = Float64(vertices[b][3])

        cx = Float64(vertices[c][1])
        cy = Float64(vertices[c][2])
        cz = Float64(vertices[c][3])

        abx64 = bx - ax
        aby64 = by - ay
        abz64 = bz - az

        acx64 = cx - ax
        acy64 = cy - ay
        acz64 = cz - az

        # Face normal (unit)
        nx = aby64*acz64 - abz64*acy64
        ny = abz64*acx64 - abx64*acz64
        nz = abx64*acy64 - aby64*acx64

        area2 = nx*nx + ny*ny + nz*nz
        if area2 <= degenerate_area2_eps
            continue
        end

        invn = 1.0 / sqrt(area2)
        nux = nx * invn
        nuy = ny * invn
        nuz = nz * invn

        # Angles for vertex pseudo-normals (Float64)
        angA = angle_between(abx64, aby64, abz64, acx64, acy64, acz64)

        bax = -abx64
        bay = -aby64
        baz = -abz64
        bcx = acx64 - abx64
        bcy = acy64 - aby64
        bcz = acz64 - abz64
        angB = angle_between(bax, bay, baz, bcx, bcy, bcz)

        cax = -acx64
        cay = -acy64
        caz = -acz64
        cbx = abx64 - acx64
        cby = aby64 - acy64
        cbz = abz64 - acz64
        angC = angle_between(cax, cay, caz, cbx, cby, cbz)

        vnx_acc[a] += nux * angA
        vny_acc[a] += nuy * angA
        vnz_acc[a] += nuz * angA

        vnx_acc[b] += nux * angB
        vny_acc[b] += nuy * angB
        vnz_acc[b] += nuz * angB

        vnx_acc[c] += nux * angC
        vny_acc[c] += nuy * angC
        vnz_acc[c] += nuz * angC

        # Edge pseudo-normal accumulation
        for (u, v) in ((a, b), (a, c), (b, c))
            key = edge_key(u, v)
            sx, sy, sz = get(edge_acc, key, (0.0, 0.0, 0.0))
            edge_acc[key] = (sx + nux, sy + nuy, sz + nuz)
        end

        # Store packed tri geometry
        push!(i0, a)
        push!(i1, b)
        push!(i2, c)

        push!(v0x, Float32(ax))
        push!(v0y, Float32(ay))
        push!(v0z, Float32(az))

        push!(e0x, Float32(abx64))
        push!(e0y, Float32(aby64))
        push!(e0z, Float32(abz64))

        push!(e1x, Float32(acx64))
        push!(e1y, Float32(acy64))
        push!(e1z, Float32(acz64))

        push!(fnx, Float32(nux))
        push!(fny, Float32(nuy))
        push!(fnz, Float32(nuz))
    end

    n_faces = length(i0)
    if n_faces == 0
        error("All faces were degenerate after filtering")
    end

    # Vertex pseudo-normals (unit) Float64 -> Float32
    vnx = Vector{Float32}(undef, n_verts)
    vny = Vector{Float32}(undef, n_verts)
    vnz = Vector{Float32}(undef, n_verts)

    @inbounds for i in 1:n_verts
        x = vnx_acc[i]
        y = vny_acc[i]
        z = vnz_acc[i]
        n = norm3(x, y, z)

        if n == 0.0
            vnx[i] = 0f0
            vny[i] = 0f0
            vnz[i] = 1f0
        else
            inv = 1.0 / n
            vnx[i] = Float32(x * inv)
            vny[i] = Float32(y * inv)
            vnz[i] = Float32(z * inv)
        end
    end

    # Normalize edge pseudo-normals (dict of UInt64 => Float32 triple)
    edge_unit = Dict{UInt64, NTuple{3,Float32}}()
    for (k, (sx, sy, sz)) in edge_acc
        n = norm3(sx, sy, sz)
        if n == 0.0
            edge_unit[k] = (0f0, 0f0, 0f0)
        else
            inv = 1.0 / n
            edge_unit[k] = (Float32(sx * inv), Float32(sy * inv), Float32(sz * inv))
        end
    end

    # Per-triangle edge normals for AB, AC, BC
    eabx = Vector{Float32}(undef, n_faces)
    eaby = Vector{Float32}(undef, n_faces)
    eabz = Vector{Float32}(undef, n_faces)

    eacx = Vector{Float32}(undef, n_faces)
    eacy = Vector{Float32}(undef, n_faces)
    eacz = Vector{Float32}(undef, n_faces)

    ebcx = Vector{Float32}(undef, n_faces)
    ebcy = Vector{Float32}(undef, n_faces)
    ebcz = Vector{Float32}(undef, n_faces)

    @inbounds for t in 1:n_faces
        a = i0[t]
        b = i1[t]
        c = i2[t]
        fn = (fnx[t], fny[t], fnz[t])

        nab = get(edge_unit, edge_key(a, b), fn)
        nac = get(edge_unit, edge_key(a, c), fn)
        nbc = get(edge_unit, edge_key(b, c), fn)

        eabx[t] = nab[1]
        eaby[t] = nab[2]
        eabz[t] = nab[3]

        eacx[t] = nac[1]
        eacy[t] = nac[2]
        eacz[t] = nac[3]

        ebcx[t] = nbc[1]
        ebcy[t] = nbc[2]
        ebcz[t] = nbc[3]
    end

    result = (
        i0=i0, i1=i1, i2=i2,
        v0x=v0x, v0y=v0y, v0z=v0z,
        e0x=e0x, e0y=e0y, e0z=e0z,
        e1x=e1x, e1y=e1y, e1z=e1z,
        fnx=fnx, fny=fny, fnz=fnz,
        eabx=eabx, eaby=eaby, eabz=eabz,
        eacx=eacx, eacy=eacy, eacz=eacz,
        ebcx=ebcx, ebcy=ebcy, ebcz=ebcz,
        vnx=vnx, vny=vny, vnz=vnz,
        n_faces=Int32(n_faces),
        n_verts=Int32(n_verts)
    )
    return result
end

# =============================================================================
# CUDA kernel
# =============================================================================

function compute_sdf_kernel!(
    out,
    v0x, v0y, v0z, e0x, e0y, e0z, e1x, e1y, e1z,
    i0, i1, i2,
    fnx, fny, fnz,
    eabx, eaby, eabz,
    eacx, eacy, eacz,
    ebcx, ebcy, ebcz,
    vnx, vny, vnz,
    start::Float32, step::Float32, n_grid::Int32, n_faces::Int32, ε²::Float32,
    ::Val{TILE}
) where {TILE}

    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    iz = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (ix > n_grid) | (iy > n_grid) | (iz > n_grid)
        return nothing
    end

    px = muladd(Float32(ix - 1), step, start)
    py = muladd(Float32(iy - 1), step, start)
    pz = muladd(Float32(iz - 1), step, start)

    # Shared geometry tile (9 arrays)
    sh_v0x = CUDA.CuStaticSharedArray(Float32, TILE)
    sh_v0y = CUDA.CuStaticSharedArray(Float32, TILE)
    sh_v0z = CUDA.CuStaticSharedArray(Float32, TILE)
    sh_e0x = CUDA.CuStaticSharedArray(Float32, TILE)
    sh_e0y = CUDA.CuStaticSharedArray(Float32, TILE)
    sh_e0z = CUDA.CuStaticSharedArray(Float32, TILE)
    sh_e1x = CUDA.CuStaticSharedArray(Float32, TILE)
    sh_e1y = CUDA.CuStaticSharedArray(Float32, TILE)
    sh_e1z = CUDA.CuStaticSharedArray(Float32, TILE)

    tx = blockDim().x
    ty = blockDim().y
    tz = blockDim().z
    tid = (threadIdx().z - 1) * (tx*ty) + (threadIdx().y - 1) * tx + threadIdx().x
    stride = tx * ty * tz

    best_d² = Inf32
    best_f  = Int32(1)

    tile_start = Int32(1)
    while tile_start <= n_faces
        cnt = n_faces - tile_start + Int32(1)
        if cnt > Int32(TILE)
            cnt = Int32(TILE)
        end

        # cooperative load
        t = tid
        while t <= cnt
            g = tile_start + Int32(t) - Int32(1)
            @inbounds begin
                sh_v0x[t] = v0x[g]
                sh_v0y[t] = v0y[g]
                sh_v0z[t] = v0z[g]
                sh_e0x[t] = e0x[g]
                sh_e0y[t] = e0y[g]
                sh_e0z[t] = e0z[g]
                sh_e1x[t] = e1x[g]
                sh_e1y[t] = e1y[g]
                sh_e1z[t] = e1z[g]
            end
            t += stride
        end
        CUDA.sync_threads()

        # process tile (dist² only)
        t = Int32(1)
        while t <= cnt
            ax  = sh_v0x[t]
            ay  = sh_v0y[t]
            az  = sh_v0z[t]
            abx = sh_e0x[t]
            aby = sh_e0y[t]
            abz = sh_e0z[t]
            acx = sh_e1x[t]
            acy = sh_e1y[t]
            acz = sh_e1z[t]

            d² = dist2_point_triangle(px, py, pz, ax, ay, az, abx, aby, abz, acx, acy, acz)
            if d² < best_d²
                best_d² = d²
                best_f = tile_start + t - Int32(1)
            end
            t += Int32(1)
        end

        CUDA.sync_threads()
        tile_start += Int32(TILE)
    end

    # robust on-surface handling
    if best_d² <= ε²
        @inbounds out[ix, iy, iz] = 0f0
        return nothing
    end

    # compute closest point + feature only for best face
    @inbounds begin
        ax  = v0x[best_f]
        ay  = v0y[best_f]
        az  = v0z[best_f]
        abx = e0x[best_f]
        aby = e0y[best_f]
        abz = e0z[best_f]
        acx = e1x[best_f]
        acy = e1y[best_f]
        acz = e1z[best_f]
    end

    qx, qy, qz, region = closest_point_region(
        px, py, pz, ax, ay, az, abx, aby, abz, acx, acy, acz
    )

    # choose pseudo-normal (face/edge/vertex)
    nx = 0f0
    ny = 0f0
    nz = 1f0

    @inbounds begin
        if region == UInt8(0)
            nx = fnx[best_f]
            ny = fny[best_f]
            nz = fnz[best_f]
        elseif region == UInt8(1)
            vi = i0[best_f]
            nx = vnx[vi]
            ny = vny[vi]
            nz = vnz[vi]
        elseif region == UInt8(2)
            vi = i1[best_f]
            nx = vnx[vi]
            ny = vny[vi]
            nz = vnz[vi]
        elseif region == UInt8(3)
            vi = i2[best_f]
            nx = vnx[vi]
            ny = vny[vi]
            nz = vnz[vi]
        elseif region == UInt8(4)   # AB
            nx = eabx[best_f]
            ny = eaby[best_f]
            nz = eabz[best_f]
        elseif region == UInt8(5)   # AC
            nx = eacx[best_f]
            ny = eacy[best_f]
            nz = eacz[best_f]
        else                        # BC
            nx = ebcx[best_f]
            ny = ebcy[best_f]
            nz = ebcz[best_f]
        end
    end

    vx = px - qx
    vy = py - qy
    vz = pz - qz

    s = dot3f(vx, vy, vz, nx, ny, nz)
    d = sqrt(best_d²)

    @inbounds out[ix, iy, iz] = (s >= 0f0) ? d : -d

    return nothing
end

# =============================================================================
# Public API (drop-in replacement style)
# =============================================================================

function compute_sdf(mesh::Mesh{3,Float32}, n::Int=128; tile_size::Int=256)
    rng = range(-1f0, 1f0; length=n)
    sdf = compute_sdf(mesh.position, mesh.faces, rng; tile_size)
    return sdf::CuArray{Float32,3}
end

function compute_sdf(
    vertices::Vector{Point3{Float32}},
    faces::Vector{<:NgonFace{3}},
    rng::StepRangeLen{Float32};
    tile_size::Int=256
)
    # - analytic grid coordinate generation (no d_grid loads)
    # - true Float64 CPU preprocessing for normals/angles
    # - precomputed triangle edges (v0,e0,e1) to reduce kernel FLOPs
    # - shared memory holds only geometry (9 floats/tri)
    # - sign computed once from closest feature (face/edge/vertex pseudo-normal)
    # - robust on-surface epsilon (return 0)
    # - generic thread indexing and tunable TILE_SIZE (64/128/256)
    (tile_size ∈ (64, 128, 256)) || error("tile_size must be 64, 128, or 256")

    data_cpu = precompute_data_on_cpu(vertices, faces)

    # Upload to GPU
    d_v0x = CuArray(data_cpu.v0x)
    d_v0y = CuArray(data_cpu.v0y)
    d_v0z = CuArray(data_cpu.v0z)

    d_e0x = CuArray(data_cpu.e0x)
    d_e0y = CuArray(data_cpu.e0y)
    d_e0z = CuArray(data_cpu.e0z)

    d_e1x = CuArray(data_cpu.e1x)
    d_e1y = CuArray(data_cpu.e1y)
    d_e1z = CuArray(data_cpu.e1z)

    d_i0  = CuArray(data_cpu.i0)
    d_i1  = CuArray(data_cpu.i1)
    d_i2  = CuArray(data_cpu.i2)

    d_fnx = CuArray(data_cpu.fnx)
    d_fny = CuArray(data_cpu.fny)
    d_fnz = CuArray(data_cpu.fnz)

    d_eabx = CuArray(data_cpu.eabx)
    d_eaby = CuArray(data_cpu.eaby)
    d_eabz = CuArray(data_cpu.eabz)

    d_eacx = CuArray(data_cpu.eacx)
    d_eacy = CuArray(data_cpu.eacy)
    d_eacz = CuArray(data_cpu.eacz)

    d_ebcx = CuArray(data_cpu.ebcx)
    d_ebcy = CuArray(data_cpu.ebcy)
    d_ebcz = CuArray(data_cpu.ebcz)

    d_vnx = CuArray(data_cpu.vnx)
    d_vny = CuArray(data_cpu.vny)
    d_vnz = CuArray(data_cpu.vnz)

    n_grid = Int32(length(rng))
    start = Float32(first(rng))
    step  = Float32(step(rng))

    # Robust on-surface epsilon
    ε = max(1f-6, 1f-3 * step)
    ε² = ε * ε

    sdf = CUDA.zeros(Float32, n_grid, n_grid, n_grid)

    if tile_size == 64
        val_tile_size = Val(64)
        threads = (32, 2, 1)
    elseif tile_size == 128
        val_tile_size = Val(128)
        threads = (32, 4, 1)
    else
        val_tile_size = Val(256)
        threads = (32, 4, 2)
    end

    tx, ty, tz = threads
    blocks = (cld(Int(n_grid), tx), cld(Int(n_grid), ty), cld(Int(n_grid), tz))

    @cuda threads=threads blocks=blocks compute_sdf_kernel!(
        sdf,
        CUDA.Const(d_v0x), CUDA.Const(d_v0y), CUDA.Const(d_v0z),
        CUDA.Const(d_e0x), CUDA.Const(d_e0y), CUDA.Const(d_e0z),
        CUDA.Const(d_e1x), CUDA.Const(d_e1y), CUDA.Const(d_e1z),
        CUDA.Const(d_i0),  CUDA.Const(d_i1),  CUDA.Const(d_i2),
        CUDA.Const(d_fnx), CUDA.Const(d_fny), CUDA.Const(d_fnz),
        CUDA.Const(d_eabx), CUDA.Const(d_eaby), CUDA.Const(d_eabz),
        CUDA.Const(d_eacx), CUDA.Const(d_eacy), CUDA.Const(d_eacz),
        CUDA.Const(d_ebcx), CUDA.Const(d_ebcy), CUDA.Const(d_ebcz),
        CUDA.Const(d_vnx), CUDA.Const(d_vny), CUDA.Const(d_vnz),
        start, step, n_grid, data_cpu.n_faces, ε², val_tile_size
    )
    return sdf::CuArray{Float32,3}
end
