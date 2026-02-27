# --------------------------------------------
# Public API
# --------------------------------------------

struct ODCKernels{Ksd,Keb,Knf,Kqs}
    edge_batch::Int
    qef_batch::Int
    bs_iters::Int
    fd::Float32
    λ::Float32
    det_eps::Float32
    eval_sd::Ksd
    edge_bisect::Keb
    normals_fd::Knf
    qef_solve::Kqs
end

"""
    ODCKernels(signed_distance; edge_batch, qef_batch, bs_iters, qef_lambda, det_eps, backend)

Pre-compile all Reactant/XLA kernels needed by [`construct_mesh`](@ref).

`signed_distance(pts)` must accept a `(3, n)` matrix and return a length-`n`
vector of signed distances (negative inside, positive outside).
"""
function ODCKernels(
    signed_distance::Function;
    edge_batch::Int=262_144,
    qef_batch::Int=262_144,
    bs_iters::Int=15,
    qef_lambda::Float32=0.05f0,
    det_eps::Float32=1f-12,
    backend::String="gpu"
)
    Reactant.set_default_backend(backend)

    λ = qef_lambda

    # --- eval_sd kernel (3,B) -> (B)
    let B = edge_batch
        pts_ex = Reactant.to_rarray(zeros(Float32, 3, B))
        function eval_sd_kernel(pts)
            return signed_distance(pts)
        end
        eval_sd = @compile eval_sd_kernel(pts_ex)

        # --- edge bisect kernel (3,B),(3,B) -> (3,B)
        p0_ex = Reactant.to_rarray(zeros(Float32, 3, B))
        p1_ex = Reactant.to_rarray(zeros(Float32, 3, B))
        function edge_bisect_kernel(p0, p1)
            return _edge_bisect(p0, p1, signed_distance, Val(bs_iters))
        end
        edge_bisect = @compile edge_bisect_kernel(p0_ex, p1_ex)

        # --- normals kernel (3,B),(1,) -> (3,B)   fd is a runtime scalar
        p_ex = Reactant.to_rarray(zeros(Float32, 3, B))
        fd_ex = Reactant.to_rarray(zeros(Float32, 1))
        function normals_kernel(p, fd_buf)
            return _normals_fd(p, signed_distance, fd_buf[1])
        end
        normals_fd = @compile normals_kernel(p_ex, fd_ex)

        # --- QEF kernel (E,Bqef) -> (3,Bqef)
        E = 12
        Bq = qef_batch
        zerosEB() = Reactant.to_rarray(zeros(Float32, E, Bq))
        zerosB() = Reactant.to_rarray(zeros(Float32, Bq))

        Px_ex = zerosEB()
        Py_ex = zerosEB()
        Pz_ex = zerosEB()
        Nx_ex = zerosEB()
        Ny_ex = zerosEB()
        Nz_ex = zerosEB()
        W_ex = zerosEB()
        Cx_ex = zerosB()
        Cy_ex = zerosB()
        Cz_ex = zerosB()
        minx_ex = zerosB()
        miny_ex = zerosB()
        minz_ex = zerosB()
        maxx_ex = zerosB()
        maxy_ex = zerosB()
        maxz_ex = zerosB()

        function qef_kernel(Px, Py, Pz, Nx, Ny, Nz, W, Cx, Cy, Cz, minx, miny, minz, maxx, maxy, maxz)
            return _qef_solve_batch(Px, Py, Pz, Nx, Ny, Nz, W, Cx, Cy, Cz, minx, miny, minz, maxx, maxy, maxz, λ, det_eps)
        end
        qef_solve = @compile qef_kernel(Px_ex, Py_ex, Pz_ex, Nx_ex, Ny_ex, Nz_ex, W_ex,
            Cx_ex, Cy_ex, Cz_ex, minx_ex, miny_ex, minz_ex, maxx_ex, maxy_ex, maxz_ex)

        return ODCKernels(edge_batch, qef_batch, bs_iters, 0f0, λ, det_eps,
            eval_sd, edge_bisect, normals_fd, qef_solve)
    end
end


"""
    construct_mesh(kernels::ODCKernels, bounding_box::NTuple{2,Point3f}; resolution_max::Int=256)

SDF-only occupancy-based / manifold dual contouring style meshing (multi-vertex per cell).

Inputs
- `kernels`: pre-compiled [`ODCKernels`](@ref) (carries the SDF and all GPU kernels).
- `bounding_box = (pmin, pmax)`.
- `resolution_max`: number of cells along the longest bounding-box axis.
  Shorter axes get proportionally fewer cells, and the bounding box is
  symmetrically padded on each axis so that all voxels are exactly cubic.

Output
- `GeometryBasics.Mesh` with `Point3f` vertices and `TriangleFace{Int32}` faces.

Notes
- Multi-vertex-per-cell via *true primal-face partitioning*: face marching-squares segments → cycles → one QEF vertex per cycle.
- Hermite normals from SDF gradients via finite differences (GPU-batched).
- Intersection-Free Contouring quad splitting uses ODC appendix case1/case2/case3 with Wang (2009) concavity test.
- Manifold post-pass splits non-manifold edges/vertices to guarantee 2-manifold output.
"""
function construct_mesh(
    kernels::ODCKernels, bounding_box::NTuple{2,Point3f}; resolution_max::Int=256
)
    edge_batch = kernels.edge_batch
    qef_batch = kernels.qef_batch

    # --------------------
    # Grid setup: cubic voxels with symmetric bbox padding
    # --------------------
    (δ, nx, ny, nz, xmin, ymin, zmin, xmax, ymax, zmax) = setup_grid(bounding_box, resolution_max)

    fd = 0.5f0 * δ

    # --------------------
    # Runtime fd device buffer (not baked into the compiled graph)
    # --------------------
    fd_device = Reactant.to_rarray(Float32[fd])

    # --------------------
    # 1) Evaluate SDF on grid vertices (GPU-batched)
    # --------------------
    sdf = _eval_grid_sdf!(kernels.eval_sd, xmin, ymin, zmin, δ, nx, ny, nz, edge_batch)

    # --------------------
    # 2) Extract sign-change edges (CPU) + 3) 1D root search (GPU) + normals (GPU)
    # --------------------
    ex_p, ex_n, ex_active = _compute_edge_hermite_data_x!(kernels, sdf, xmin, ymin, zmin, δ, nx, ny, nz, edge_batch, fd_device)
    ey_p, ey_n, ey_active = _compute_edge_hermite_data_y!(kernels, sdf, xmin, ymin, zmin, δ, nx, ny, nz, edge_batch, fd_device)
    ez_p, ez_n, ez_active = _compute_edge_hermite_data_z!(kernels, sdf, xmin, ymin, zmin, δ, nx, ny, nz, edge_batch, fd_device)

    # --------------------
    # 4) True primal-face partitioning per cell → cycles → patches
    #    Build:
    #      - patch constraints (<=12 edges each)
    #      - cell-edge → patch-id lookup (12 edges per cell)
    # --------------------
    cell_edge_patch, patch_data = _build_patches_primal_partition!(
        sdf, ex_p, ex_n, ey_p, ey_n, ez_p, ez_n,
        xmin, ymin, zmin, δ, nx, ny, nz
    )

    n_patches = patch_data.n_patches
    if n_patches == 0
        return Mesh(Point3f[], TriangleFace{Int32}[])
    end

    # --------------------
    # 5) QEF solve per patch (GPU-batched) + clamp (GPU)
    # --------------------
    patch_pos = _solve_qef_patches!(kernels, patch_data, qef_batch)

    # Start vertex list with the patch vertices (index == patch_id)
    vertices = Vector{Point3f}(undef, n_patches)
    @inbounds for pid in 1:n_patches
        vertices[pid] = Point3f(patch_pos.px[pid], patch_pos.py[pid], patch_pos.pz[pid])
    end

    # --------------------
    # 6) Emit quads per sign-change grid edge (dual connectivity) + IC split
    # --------------------
    faces = Vector{TriangleFace{Int32}}()
    sizehint!(faces, 2 * (length(ex_active) + length(ey_active) + length(ez_active)))

    _emit_faces_x!(faces, vertices, cell_edge_patch, sdf, ex_p, ex_active,
        xmin, ymin, zmin, δ, nx, ny, nz)

    _emit_faces_y!(faces, vertices, cell_edge_patch, sdf, ey_p, ey_active,
        xmin, ymin, zmin, δ, nx, ny, nz)

    _emit_faces_z!(faces, vertices, cell_edge_patch, sdf, ez_p, ez_active,
        xmin, ymin, zmin, δ, nx, ny, nz)

    _manifoldize!(vertices, faces)

    return Mesh(vertices, faces)
end


# --------------------------------------------
# Cubic-voxel grid from bounding box
# --------------------------------------------

"""
    setup_grid(bounding_box, resolution_max) -> NamedTuple

Compute a uniform grid with exactly cubic voxels.

`resolution_max` cells are placed along the longest bounding-box axis, fixing
the voxel size `δ`.  Shorter axes get `ceil(extent / δ)` cells.  Each axis is
then symmetrically padded so that `ni * δ` equals the padded extent exactly.

All arithmetic is performed in Float64 to avoid intermediate rounding; the
returned values are converted to Float32 at the very end.

Returns `(; nx, ny, nz, xmin, ymin, zmin, xmax, ymax, zmax, δ)` with Float32 coordinates.
"""
function setup_grid(bounding_box::NTuple{2,Point3f}, resolution_max::Int)
    (pmin, pmax) = bounding_box
    xmin = Float64(pmin[1])
    ymin = Float64(pmin[2])
    zmin = Float64(pmin[3])
    xmax = Float64(pmax[1])
    ymax = Float64(pmax[2])
    zmax = Float64(pmax[3])

    ex = xmax - xmin
    ey = ymax - ymin
    ez = zmax - zmin

    # Voxel size determined by the longest axis
    δ = max(ex, ey, ez) / resolution_max

    # Cell count per axis (at least 1)
    nx = max(ceil(Int, ex / δ), 1)
    ny = max(ceil(Int, ey / δ), 1)
    nz = max(ceil(Int, ez / δ), 1)

    # Symmetric padding: pad_i = (ni * δ - ei) / 2
    # Use muladd(ni, δ, -ei) for ni*δ - ei to avoid intermediate rounding
    pad_x = 0.5 * muladd(nx, δ, -ex)
    pad_y = 0.5 * muladd(ny, δ, -ey)
    pad_z = 0.5 * muladd(nz, δ, -ez)

    # xmin = pmin - pad, xmax = xmin + ni*δ
    # Use muladd for xmax = muladd(ni, δ, xmin) to fuse the multiply-add
    xmin -= pad_x
    ymin -= pad_y
    zmin -= pad_z
    xmax = muladd(nx, δ, xmin)
    ymax = muladd(ny, δ, ymin)
    zmax = muladd(nz, δ, zmin)

    return (
        Float32(δ),
        nx, ny, nz,
        Float32(xmin), Float32(ymin), Float32(zmin),
        Float32(xmax), Float32(ymax), Float32(zmax),
    )
end



# --------------------------------------------
# Reactant kernels (pure array ops)
# --------------------------------------------

function _edge_bisect(p0, p1, signed_distance, ::Val{ITERS}) where {ITERS}
    lo = p0
    hi = p1
    for _ in 1:ITERS
        mid = 0.5f0 .* (lo .+ hi)
        fmid = signed_distance(mid)
        inside = fmid .< 0f0
        # if inside: lo = mid else hi = mid
        lo = ifelse.(inside, mid, lo)
        hi = ifelse.(inside, hi, mid)
    end
    return 0.5f0 .* (lo .+ hi)
end

function _normals_fd(p, signed_distance, fd)
    # p: (3,B)
    B = size(p, 2)
    # build 6B points: (+x,-x,+y,-y,+z,-z)
    px = p[1, :]
    py = p[2, :]
    pz = p[3, :]
    # NOTE: keep everything broadcastable
    pts = similar(p, 3, 6B)

    @inbounds begin
        pts[1, 1:B] .= px .+ fd
        pts[2, 1:B] .= py
        pts[3, 1:B] .= pz
        pts[1, B+1:2B] .= px .- fd
        pts[2, B+1:2B] .= py
        pts[3, B+1:2B] .= pz
        pts[1, 2B+1:3B] .= px
        pts[2, 2B+1:3B] .= py .+ fd
        pts[3, 2B+1:3B] .= pz
        pts[1, 3B+1:4B] .= px
        pts[2, 3B+1:4B] .= py .- fd
        pts[3, 3B+1:4B] .= pz
        pts[1, 4B+1:5B] .= px
        pts[2, 4B+1:5B] .= py
        pts[3, 4B+1:5B] .= pz .+ fd
        pts[1, 5B+1:6B] .= px
        pts[2, 5B+1:6B] .= py
        pts[3, 5B+1:6B] .= pz .- fd
    end

    f = signed_distance(pts)  # length 6B
    fxp = view(f, 1:B)
    fxm = view(f, B+1:2B)
    fyp = view(f, 2B+1:3B)
    fym = view(f, 3B+1:4B)
    fzp = view(f, 4B+1:5B)
    fzm = view(f, 5B+1:6B)

    inv2fd = 1f0 / (2f0 * fd)
    gx = (fxp .- fxm) .* inv2fd
    gy = (fyp .- fym) .* inv2fd
    gz = (fzp .- fzm) .* inv2fd

    nrm = sqrt.(gx .* gx .+ gy .* gy .+ gz .* gz .+ 1f-20)
    nx = gx ./ nrm
    ny = gy ./ nrm
    nz = gz ./ nrm

    out = similar(p)
    out[1, :] .= nx
    out[2, :] .= ny
    out[3, :] .= nz
    return out
end

function _qef_solve_batch(Px, Py, Pz, Nx, Ny, Nz, W, Cx, Cy, Cz,
    minx, miny, minz, maxx, maxy, maxz,
    λ::Float32, det_eps::Float32)

    # dot_i = n_i ⋅ p_i
    dotp = Nx .* Px .+ Ny .* Py .+ Nz .* Pz

    # Weighted sums for A = Σ w n n^T and b = Σ w n (n⋅p)
    wNx = W .* Nx
    wNy = W .* Ny
    wNz = W .* Nz

    Axx = sum(wNx .* Nx, dims=1)
    Axy = sum(wNx .* Ny, dims=1)
    Axz = sum(wNx .* Nz, dims=1)
    Ayy = sum(wNy .* Ny, dims=1)
    Ayz = sum(wNy .* Nz, dims=1)
    Azz = sum(wNz .* Nz, dims=1)

    bx = sum(wNx .* dotp, dims=1)
    by = sum(wNy .* dotp, dims=1)
    bz = sum(wNz .* dotp, dims=1)

    # Drop the singleton dim so everything is 1D length B
    Axx = vec(Axx)
    Axy = vec(Axy)
    Axz = vec(Axz)
    Ayy = vec(Ayy)
    Ayz = vec(Ayz)
    Azz = vec(Azz)
    bx = vec(bx)
    by = vec(by)
    bz = vec(bz)

    # Tikhonov regularization toward cell center
    Axx = Axx .+ λ
    Ayy = Ayy .+ λ
    Azz = Azz .+ λ
    bx = bx .+ λ .* Cx
    by = by .+ λ .* Cy
    bz = bz .+ λ .* Cz

    # Invert symmetric 3x3 matrix analytically
    m11 = Axx
    m12 = Axy
    m13 = Axz
    m22 = Ayy
    m23 = Ayz
    m33 = Azz

    det = m11 .* (m22 .* m33 .- m23 .* m23) .-
          m12 .* (m12 .* m33 .- m23 .* m13) .+
          m13 .* (m12 .* m23 .- m22 .* m13)

    good = abs.(det) .> det_eps
    invdet = 1f0 ./ det

    inv11 = (m22 .* m33 .- m23 .* m23) .* invdet
    inv12 = (m13 .* m23 .- m12 .* m33) .* invdet
    inv13 = (m12 .* m23 .- m13 .* m22) .* invdet
    inv22 = (m11 .* m33 .- m13 .* m13) .* invdet
    inv23 = (m13 .* m12 .- m11 .* m23) .* invdet
    inv33 = (m11 .* m22 .- m12 .* m12) .* invdet

    x = inv11 .* bx .+ inv12 .* by .+ inv13 .* bz
    y = inv12 .* bx .+ inv22 .* by .+ inv23 .* bz
    z = inv13 .* bx .+ inv23 .* by .+ inv33 .* bz

    # fallback if singular/near-singular
    x = ifelse.(good, x, Cx)
    y = ifelse.(good, y, Cy)
    z = ifelse.(good, z, Cz)

    # clamp to cell AABB
    x = min.(max.(x, minx), maxx)
    y = min.(max.(y, miny), maxy)
    z = min.(max.(z, minz), maxz)

    out = Reactant.Ops.stack((x, y, z); axis=1)  # (3,B)
    return out
end


# --------------------------------------------
# Grid SDF evaluation (chunked)
# --------------------------------------------

function _eval_grid_sdf!(eval_sd, xmin, ymin, zmin, δ, nx, ny, nz, batch::Int)
    sx = nx + 1
    sy = ny + 1
    sz = nz + 1
    N = sx * sy * sz

    sdf_flat = Vector{Float32}(undef, N)

    pts_host = zeros(Float32, 3, batch)
    pts_device = Reactant.to_rarray(pts_host)  # pre-allocate device buffer once

    for base in 0:batch:(N-1)
        m = min(batch, N - base)
        @inbounds for t in 1:m
            idx = base + (t - 1)
            i = idx % sx
            j = (idx ÷ sx) % sy
            k = idx ÷ (sx * sy)
            pts_host[1, t] = xmin + δ * Float32(i)
            pts_host[2, t] = ymin + δ * Float32(j)
            pts_host[3, t] = zmin + δ * Float32(k)
        end
        if m < batch
            @inbounds pts_host[:, m+1:batch] .= 0f0
        end

        copyto!(pts_device, pts_host)
        out_r = eval_sd(pts_device)
        out = Array(out_r)
        @inbounds sdf_flat[base+1:base+m] .= out[1:m]
    end

    sdf = reshape(sdf_flat, sx, sy, sz)
    return sdf
end


# --------------------------------------------
# Edge extraction + Hermite data (root + normal)
# --------------------------------------------

# Dense storage:
#   ex_p, ex_n: (3, nx, ny+1, nz+1)
#   ey_p, ey_n: (3, nx+1, ny, nz+1)
#   ez_p, ez_n: (3, nx+1, ny+1, nz)
#
# Active edge lists store linear indices in those arrays (1-based linear indexing).

@inline function _decode_ex(l::Int, nx::Int, ny::Int)
    # ex dims: (nx, ny+1, nz+1) but nz handled outside
    l0 = l - 1
    i = (l0 % nx) + 1
    l0 ÷= nx
    j = (l0 % (ny + 1)) + 1
    k = (l0 ÷ (ny + 1)) + 1
    return i, j, k
end

@inline function _decode_ey(l::Int, nx::Int, ny::Int)
    # ey dims: (nx+1, ny, nz+1)
    l0 = l - 1
    i = (l0 % (nx + 1)) + 1
    l0 ÷= (nx + 1)
    j = (l0 % ny) + 1
    k = (l0 ÷ ny) + 1
    return i, j, k
end

@inline function _decode_ez(l::Int, nx::Int, ny::Int)
    # ez dims: (nx+1, ny+1, nz)
    l0 = l - 1
    i = (l0 % (nx + 1)) + 1
    l0 ÷= (nx + 1)
    j = (l0 % (ny + 1)) + 1
    k = (l0 ÷ (ny + 1)) + 1
    return i, j, k
end

function _compute_edge_hermite_data_x!(kernels::ODCKernels, sdf, xmin, ymin, zmin, δ, nx, ny, nz, batch::Int, fd_device)
    ex_p = zeros(Float32, 3, nx, ny + 1, nz + 1)
    ex_n = zeros(Float32, 3, nx, ny + 1, nz + 1)
    active = Int[]

    # find active x-edges
    @inbounds for k in 1:(nz+1), j in 1:(ny+1), i in 1:nx
        f0 = sdf[i, j, k]
        f1 = sdf[i+1, j, k]
        if (f0 < 0f0) != (f1 < 0f0)
            push!(active, LinearIndices((nx, ny + 1, nz + 1))[i, j, k])
        end
    end

    # chunked GPU root+normal — pre-allocate host AND device buffers once
    p0 = zeros(Float32, 3, batch)
    p1 = zeros(Float32, 3, batch)
    p0_device = Reactant.to_rarray(p0)
    p1_device = Reactant.to_rarray(p1)

    for base in 0:batch:(length(active)-1)
        m = min(batch, length(active) - base)

        @inbounds for t in 1:m
            l = active[base+t]
            i, j, k = _decode_ex(l, nx, ny)
            x0 = xmin + δ * Float32(i - 1)
            y0 = ymin + δ * Float32(j - 1)
            z0 = zmin + δ * Float32(k - 1)

            f0 = sdf[i, j, k]
            # endpoint at (i,j,k) and (i+1,j,k)
            if f0 < 0f0
                # inside at start
                p0[1, t] = x0
                p0[2, t] = y0
                p0[3, t] = z0
                p1[1, t] = x0 + δ
                p1[2, t] = y0
                p1[3, t] = z0
            else
                # inside at end
                p0[1, t] = x0 + δ
                p0[2, t] = y0
                p0[3, t] = z0
                p1[1, t] = x0
                p1[2, t] = y0
                p1[3, t] = z0
            end
        end
        if m < batch
            @inbounds begin
                p0[:, m+1:batch] .= 0f0
                p1[:, m+1:batch] .= 0f0
            end
        end

        copyto!(p0_device, p0)
        copyto!(p1_device, p1)
        root_r = kernels.edge_bisect(p0_device, p1_device)
        root = Array(root_r)

        # reuse p0_device for normals input (same shape)
        copyto!(p0_device, root)
        nor_r = kernels.normals_fd(p0_device, fd_device)
        nor = Array(nor_r)

        @inbounds for t in 1:m
            l = active[base+t]
            i, j, k = _decode_ex(l, nx, ny)
            ex_p[:, i, j, k] .= root[:, t]
            ex_n[:, i, j, k] .= nor[:, t]
        end
    end

    return ex_p, ex_n, active
end

function _compute_edge_hermite_data_y!(kernels::ODCKernels, sdf, xmin, ymin, zmin, δ, nx, ny, nz, batch::Int, fd_device)
    ey_p = zeros(Float32, 3, nx + 1, ny, nz + 1)
    ey_n = zeros(Float32, 3, nx + 1, ny, nz + 1)
    active = Int[]

    @inbounds for k in 1:(nz+1), j in 1:ny, i in 1:(nx+1)
        f0 = sdf[i, j, k]
        f1 = sdf[i, j+1, k]
        if (f0 < 0f0) != (f1 < 0f0)
            push!(active, LinearIndices((nx + 1, ny, nz + 1))[i, j, k])
        end
    end

    p0 = zeros(Float32, 3, batch)
    p1 = zeros(Float32, 3, batch)
    p0_device = Reactant.to_rarray(p0)
    p1_device = Reactant.to_rarray(p1)

    for base in 0:batch:(length(active)-1)
        m = min(batch, length(active) - base)

        @inbounds for t in 1:m
            l = active[base+t]
            i, j, k = _decode_ey(l, nx, ny)
            x0 = xmin + δ * Float32(i - 1)
            y0 = ymin + δ * Float32(j - 1)
            z0 = zmin + δ * Float32(k - 1)

            f0 = sdf[i, j, k]
            if f0 < 0f0
                p0[1, t] = x0
                p0[2, t] = y0
                p0[3, t] = z0
                p1[1, t] = x0
                p1[2, t] = y0 + δ
                p1[3, t] = z0
            else
                p0[1, t] = x0
                p0[2, t] = y0 + δ
                p0[3, t] = z0
                p1[1, t] = x0
                p1[2, t] = y0
                p1[3, t] = z0
            end
        end
        if m < batch
            @inbounds begin
                p0[:, m+1:batch] .= 0f0
                p1[:, m+1:batch] .= 0f0
            end
        end

        copyto!(p0_device, p0)
        copyto!(p1_device, p1)
        root_r = kernels.edge_bisect(p0_device, p1_device)
        root = Array(root_r)

        copyto!(p0_device, root)
        nor_r = kernels.normals_fd(p0_device, fd_device)
        nor = Array(nor_r)

        @inbounds for t in 1:m
            l = active[base+t]
            i, j, k = _decode_ey(l, nx, ny)
            ey_p[:, i, j, k] .= root[:, t]
            ey_n[:, i, j, k] .= nor[:, t]
        end
    end

    return ey_p, ey_n, active
end

function _compute_edge_hermite_data_z!(kernels::ODCKernels, sdf, xmin, ymin, zmin, δ, nx, ny, nz, batch::Int, fd_device)
    ez_p = zeros(Float32, 3, nx + 1, ny + 1, nz)
    ez_n = zeros(Float32, 3, nx + 1, ny + 1, nz)
    active = Int[]

    @inbounds for k in 1:nz, j in 1:(ny+1), i in 1:(nx+1)
        f0 = sdf[i, j, k]
        f1 = sdf[i, j, k+1]
        if (f0 < 0f0) != (f1 < 0f0)
            push!(active, LinearIndices((nx + 1, ny + 1, nz))[i, j, k])
        end
    end

    p0 = zeros(Float32, 3, batch)
    p1 = zeros(Float32, 3, batch)
    p0_device = Reactant.to_rarray(p0)
    p1_device = Reactant.to_rarray(p1)

    for base in 0:batch:(length(active)-1)
        m = min(batch, length(active) - base)

        @inbounds for t in 1:m
            l = active[base+t]
            i, j, k = _decode_ez(l, nx, ny)
            x0 = xmin + δ * Float32(i - 1)
            y0 = ymin + δ * Float32(j - 1)
            z0 = zmin + δ * Float32(k - 1)

            f0 = sdf[i, j, k]
            if f0 < 0f0
                p0[1, t] = x0
                p0[2, t] = y0
                p0[3, t] = z0
                p1[1, t] = x0
                p1[2, t] = y0
                p1[3, t] = z0 + δ
            else
                p0[1, t] = x0
                p0[2, t] = y0
                p0[3, t] = z0 + δ
                p1[1, t] = x0
                p1[2, t] = y0
                p1[3, t] = z0
            end
        end
        if m < batch
            @inbounds begin
                p0[:, m+1:batch] .= 0f0
                p1[:, m+1:batch] .= 0f0
            end
        end

        copyto!(p0_device, p0)
        copyto!(p1_device, p1)
        root_r = kernels.edge_bisect(p0_device, p1_device)
        root = Array(root_r)

        copyto!(p0_device, root)
        nor_r = kernels.normals_fd(p0_device, fd_device)
        nor = Array(nor_r)

        @inbounds for t in 1:m
            l = active[base+t]
            i, j, k = _decode_ez(l, nx, ny)
            ez_p[:, i, j, k] .= root[:, t]
            ez_n[:, i, j, k] .= nor[:, t]
        end
    end

    return ez_p, ez_n, active
end


# --------------------------------------------
# Primal-face partitioning → cycles → patch constraints
# --------------------------------------------

# Local cube vertex ordering (standard MC):
# v0=(0,0,0), v1=(1,0,0), v2=(1,1,0), v3=(0,1,0), v4=(0,0,1), v5=(1,0,1), v6=(1,1,1), v7=(0,1,1)
#
# Local edges (0..11):
# e0 v0-v1 (x)   e1 v1-v2 (y)   e2 v2-v3 (x)   e3 v3-v0 (y)
# e4 v4-v5 (x)   e5 v5-v6 (y)   e6 v6-v7 (x)   e7 v7-v4 (y)
# e8 v0-v4 (z)   e9 v1-v5 (z)   e10 v2-v6 (z)  e11 v3-v7 (z)

@inline function _link!(nbr1::Vector{Int8}, nbr2::Vector{Int8}, ea::Int8, eb::Int8)
    ia = Int(ea) + 1
    ib = Int(eb) + 1
    if nbr1[ia] == -1
        nbr1[ia] = eb
    else
        nbr2[ia] = eb
    end
    if nbr1[ib] == -1
        nbr1[ib] = ea
    else
        nbr2[ib] = ea
    end
    return nothing
end

@inline function _face_segments!(nbr1::Vector{Int8}, nbr2::Vector{Int8},
    f0::Float32, f1::Float32, f2::Float32, f3::Float32,
    e0::Int8, e1::Int8, e2::Int8, e3::Int8)

    s0 = f0 < 0f0
    s1 = f1 < 0f0
    s2 = f2 < 0f0
    s3 = f3 < 0f0

    a0 = s0 != s1
    a1 = s1 != s2
    a2 = s2 != s3
    a3 = s3 != s0

    cnt = (a0 ? 1 : 0) + (a1 ? 1 : 0) + (a2 ? 1 : 0) + (a3 ? 1 : 0)
    if cnt == 0
        return nothing
    elseif cnt == 2
        # connect the two active edges
        if a0 && a1
            _link!(nbr1, nbr2, e0, e1)
        elseif a1 && a2
            _link!(nbr1, nbr2, e1, e2)
        elseif a2 && a3
            _link!(nbr1, nbr2, e2, e3)
        elseif a3 && a0
            _link!(nbr1, nbr2, e3, e0)
        elseif a0 && a2
            _link!(nbr1, nbr2, e0, e2) # rare numeric boundary; still ok
        else
            _link!(nbr1, nbr2, e1, e3)
        end
        return nothing
    elseif cnt == 4
        # Ambiguous marching squares case: s0==s2 != s1==s3.
        # Use asymptotic decider: sign at bilinear saddle.
        a = f0 - f1 - f3 + f2
        Fs = if abs(a) > 1f-20
            (f0 * f2 - f1 * f3) / a
        else
            0.25f0 * (f0 + f1 + f2 + f3)
        end

        diag_inside = s0  # s0==s2 in ambiguous case
        Fs_inside = Fs < 0f0

        if Fs_inside == diag_inside
            # Pair around v1 and v3: (e0,e1) and (e2,e3)
            _link!(nbr1, nbr2, e0, e1)
            _link!(nbr1, nbr2, e2, e3)
        else
            # Pair around v0 and v2: (e1,e2) and (e3,e0)
            _link!(nbr1, nbr2, e1, e2)
            _link!(nbr1, nbr2, e3, e0)
        end
        return nothing
    else
        # Shouldn't happen for a well-formed scalar field on a face
        return nothing
    end
end

@inline function _count_cycles_from_adjacency!(nbr1::Vector{Int8}, nbr2::Vector{Int8}, visited::Vector{Bool})
    fill!(visited, false)
    cycles = 0
    @inbounds for e in 0:11
        if nbr1[e+1] != -1 && !visited[e+1]
            cycles += 1
            start = Int8(e)
            prev = Int8(-1)
            curr = start
            while true
                visited[Int(curr)+1] = true
                n1 = nbr1[Int(curr)+1]
                n2 = nbr2[Int(curr)+1]
                nxt = (n1 != prev) ? n1 : n2
                prev, curr = curr, nxt
                if curr == start
                    break
                end
            end
        end
    end
    return cycles
end

# Pack for GPU QEF solve
struct _PatchData
    n_patches::Int
    Px::Matrix{Float32}
    Py::Matrix{Float32}
    Pz::Matrix{Float32}
    Nx::Matrix{Float32}
    Ny::Matrix{Float32}
    Nz::Matrix{Float32}
    W::Matrix{Float32}   # mask (0/1)
    Cx::Vector{Float32}
    Cy::Vector{Float32}
    Cz::Vector{Float32}
    minx::Vector{Float32}
    miny::Vector{Float32}
    minz::Vector{Float32}
    maxx::Vector{Float32}
    maxy::Vector{Float32}
    maxz::Vector{Float32}
end

function _build_patches_primal_partition!(sdf,
    ex_p, ex_n, ey_p, ey_n, ez_p, ez_n,
    xmin, ymin, zmin, δ, nx, ny, nz)

    # cell_edge_patch: (12, nx, ny, nz) gives patch_id (== vertex index) for each local edge in each cell
    cell_edge_patch = zeros(Int32, 12, nx, ny, nz)

    # ---- pass 1: count cycles per cell
    cycles_per_cell = zeros(UInt8, nx, ny, nz)

    # thread-local buffers
    Tn = nthreads()
    nbr1s = [fill(Int8(-1), 12) for _ in 1:Tn]
    nbr2s = [fill(Int8(-1), 12) for _ in 1:Tn]
    visiteds = [fill(false, 12) for _ in 1:Tn]

    @threads for ck in 1:nz
        tid = threadid()
        nbr1 = nbr1s[tid]
        nbr2 = nbr2s[tid]
        visited = visiteds[tid]
        for cj in 1:ny, ci in 1:nx
            # gather 8 corner values
            f0 = sdf[ci, cj, ck]
            f1 = sdf[ci+1, cj, ck]
            f2 = sdf[ci+1, cj+1, ck]
            f3 = sdf[ci, cj+1, ck]
            f4 = sdf[ci, cj, ck+1]
            f5 = sdf[ci+1, cj, ck+1]
            f6 = sdf[ci+1, cj+1, ck+1]
            f7 = sdf[ci, cj+1, ck+1]

            # quick reject: all same sign
            inside0 = f0 < 0f0
            allsame = ((f1 < 0f0) == inside0) && ((f2 < 0f0) == inside0) && ((f3 < 0f0) == inside0) &&
                      ((f4 < 0f0) == inside0) && ((f5 < 0f0) == inside0) && ((f6 < 0f0) == inside0) && ((f7 < 0f0) == inside0)
            if allsame
                cycles_per_cell[ci, cj, ck] = 0x00
                continue
            end

            fill!(nbr1, -1)
            fill!(nbr2, -1)

            # 6 faces
            _face_segments!(nbr1, nbr2, f0, f1, f2, f3, Int8(0), Int8(1), Int8(2), Int8(3))         # z=0
            _face_segments!(nbr1, nbr2, f4, f5, f6, f7, Int8(4), Int8(5), Int8(6), Int8(7))         # z=1
            _face_segments!(nbr1, nbr2, f0, f1, f5, f4, Int8(0), Int8(9), Int8(4), Int8(8))         # y=0
            _face_segments!(nbr1, nbr2, f3, f2, f6, f7, Int8(2), Int8(10), Int8(6), Int8(11))       # y=1
            _face_segments!(nbr1, nbr2, f0, f3, f7, f4, Int8(3), Int8(11), Int8(7), Int8(8))        # x=0
            _face_segments!(nbr1, nbr2, f1, f2, f6, f5, Int8(1), Int8(10), Int8(5), Int8(9))        # x=1

            c = _count_cycles_from_adjacency!(nbr1, nbr2, visited)
            cycles_per_cell[ci, cj, ck] = UInt8(c)
        end
    end

    n_patches = sum(Int.(cycles_per_cell))
    if n_patches == 0
        return cell_edge_patch, _PatchData(0,
            zeros(Float32, 12, 0), zeros(Float32, 12, 0), zeros(Float32, 12, 0),
            zeros(Float32, 12, 0), zeros(Float32, 12, 0), zeros(Float32, 12, 0),
            zeros(Float32, 12, 0),
            Float32[], Float32[], Float32[],
            Float32[], Float32[], Float32[],
            Float32[], Float32[], Float32[]
        )
    end

    # ---- prefix offsets to assign patch IDs
    patch_offset = zeros(Int32, nx, ny, nz)
    run = Int32(0)
    @inbounds for ck in 1:nz, cj in 1:ny, ci in 1:nx
        c = cycles_per_cell[ci, cj, ck]
        patch_offset[ci, cj, ck] = run
        run += Int32(c)
    end
    @assert run == n_patches

    # ---- allocate patch constraints
    E = 12
    Px = zeros(Float32, E, n_patches)
    Py = zeros(Float32, E, n_patches)
    Pz = zeros(Float32, E, n_patches)
    Nx = zeros(Float32, E, n_patches)
    Ny = zeros(Float32, E, n_patches)
    Nz = zeros(Float32, E, n_patches)
    W = zeros(Float32, E, n_patches)

    Cx = zeros(Float32, n_patches)
    Cy = zeros(Float32, n_patches)
    Cz = zeros(Float32, n_patches)
    minx = zeros(Float32, n_patches)
    miny = zeros(Float32, n_patches)
    minz = zeros(Float32, n_patches)
    maxx = zeros(Float32, n_patches)
    maxy = zeros(Float32, n_patches)
    maxz = zeros(Float32, n_patches)

    # ---- pass 2: fill cycles + constraints + cell_edge_patch
    @threads for ck in 1:nz
        tid = threadid()
        nbr1 = nbr1s[tid]
        nbr2 = nbr2s[tid]
        visited = visiteds[tid]
        for cj in 1:ny, ci in 1:nx
            ccyc = cycles_per_cell[ci, cj, ck]
            if ccyc == 0
                continue
            end

            # corners
            f0 = sdf[ci, cj, ck]
            f1 = sdf[ci+1, cj, ck]
            f2 = sdf[ci+1, cj+1, ck]
            f3 = sdf[ci, cj+1, ck]
            f4 = sdf[ci, cj, ck+1]
            f5 = sdf[ci+1, cj, ck+1]
            f6 = sdf[ci+1, cj+1, ck+1]
            f7 = sdf[ci, cj+1, ck+1]

            fill!(nbr1, -1)
            fill!(nbr2, -1)

            _face_segments!(nbr1, nbr2, f0, f1, f2, f3, Int8(0), Int8(1), Int8(2), Int8(3))
            _face_segments!(nbr1, nbr2, f4, f5, f6, f7, Int8(4), Int8(5), Int8(6), Int8(7))
            _face_segments!(nbr1, nbr2, f0, f1, f5, f4, Int8(0), Int8(9), Int8(4), Int8(8))
            _face_segments!(nbr1, nbr2, f3, f2, f6, f7, Int8(2), Int8(10), Int8(6), Int8(11))
            _face_segments!(nbr1, nbr2, f0, f3, f7, f4, Int8(3), Int8(11), Int8(7), Int8(8))
            _face_segments!(nbr1, nbr2, f1, f2, f6, f5, Int8(1), Int8(10), Int8(5), Int8(9))

            fill!(visited, false)

            # cell bounds and center (same for all patches in this cell)
            cminx = xmin + δ * Float32(ci - 1)
            cminy = ymin + δ * Float32(cj - 1)
            cminz = zmin + δ * Float32(ck - 1)
            cmaxx = cminx + δ
            cmaxy = cminy + δ
            cmaxz = cminz + δ
            ccx = 0.5f0 * (cminx + cmaxx)
            ccy = 0.5f0 * (cminy + cmaxy)
            ccz = 0.5f0 * (cminz + cmaxz)

            local_cycle = 0
            base = patch_offset[ci, cj, ck]

            @inbounds for e in 0:11
                if nbr1[e+1] != -1 && !visited[e+1]
                    local_cycle += 1
                    pid = Int32(base + local_cycle) # 1-based patch id

                    # write cell bounds/center
                    Cx[pid] = ccx
                    Cy[pid] = ccy
                    Cz[pid] = ccz
                    minx[pid] = cminx
                    miny[pid] = cminy
                    minz[pid] = cminz
                    maxx[pid] = cmaxx
                    maxy[pid] = cmaxy
                    maxz[pid] = cmaxz

                    start = Int8(e)
                    prev = Int8(-1)
                    curr = start
                    tslot = 0

                    while true
                        visited[Int(curr)+1] = true
                        tslot += 1

                        # map this local edge to the patch vertex id
                        cell_edge_patch[Int(curr)+1, ci, cj, ck] = pid

                        # fetch Hermite data for this local edge
                        _fill_edge_constraint!(Px, Py, Pz, Nx, Ny, Nz, W,
                            pid, tslot, curr, ci, cj, ck,
                            ex_p, ex_n, ey_p, ey_n, ez_p, ez_n)

                        n1 = nbr1[Int(curr)+1]
                        n2 = nbr2[Int(curr)+1]
                        nxt = (n1 != prev) ? n1 : n2
                        prev, curr = curr, nxt
                        if curr == start
                            break
                        end
                        if tslot == 12
                            # safety: shouldn't exceed 12 in a cube
                            break
                        end
                    end

                    # pad remainder
                    for s in (tslot+1):12
                        W[s, pid] = 0f0
                        Px[s, pid] = 0f0
                        Py[s, pid] = 0f0
                        Pz[s, pid] = 0f0
                        Nx[s, pid] = 0f0
                        Ny[s, pid] = 0f0
                        Nz[s, pid] = 0f0
                    end
                end
            end
        end
    end

    pdata = _PatchData(n_patches, Px, Py, Pz, Nx, Ny, Nz, W, Cx, Cy, Cz, minx, miny, minz, maxx, maxy, maxz)
    return cell_edge_patch, pdata
end

# Lookup table: local edge → (axis_char, di, dj, dk) where axis_char selects
# which edge array (x/y/z) and (di,dj,dk) is the offset from cell corner (ci,cj,ck).
# x-edges: 0,2,4,6   y-edges: 1,3,5,7   z-edges: 8,9,10,11
const _EDGE_LUT = (
    # e0:  x-edge at (ci, cj,   ck  )
    (0x01, 0, 0, 0),
    # e1:  y-edge at (ci+1, cj, ck  )
    (0x02, 1, 0, 0),
    # e2:  x-edge at (ci, cj+1, ck  )
    (0x01, 0, 1, 0),
    # e3:  y-edge at (ci, cj,   ck  )
    (0x02, 0, 0, 0),
    # e4:  x-edge at (ci, cj,   ck+1)
    (0x01, 0, 0, 1),
    # e5:  y-edge at (ci+1, cj, ck+1)
    (0x02, 1, 0, 1),
    # e6:  x-edge at (ci, cj+1, ck+1)
    (0x01, 0, 1, 1),
    # e7:  y-edge at (ci, cj,   ck+1)
    (0x02, 0, 0, 1),
    # e8:  z-edge at (ci, cj,   ck  )
    (0x03, 0, 0, 0),
    # e9:  z-edge at (ci+1, cj, ck  )
    (0x03, 1, 0, 0),
    # e10: z-edge at (ci+1, cj+1,ck )
    (0x03, 1, 1, 0),
    # e11: z-edge at (ci, cj+1, ck  )
    (0x03, 0, 1, 0),
)

@inline function _fill_edge_constraint!(Px, Py, Pz, Nx, Ny, Nz, W,
    pid::Int32, slot::Int,
    e::Int8, ci::Int, cj::Int, ck::Int,
    ex_p, ex_n, ey_p, ey_n, ez_p, ez_n)

    axis, di, dj, dk = _EDGE_LUT[Int(e)+1]
    @inbounds begin
        W[slot, pid] = 1f0
        if axis == 0x01  # x-edge
            gi, gj, gk = ci + di, cj + dj, ck + dk
            Px[slot, pid] = ex_p[1, gi, gj, gk]
            Py[slot, pid] = ex_p[2, gi, gj, gk]
            Pz[slot, pid] = ex_p[3, gi, gj, gk]
            Nx[slot, pid] = ex_n[1, gi, gj, gk]
            Ny[slot, pid] = ex_n[2, gi, gj, gk]
            Nz[slot, pid] = ex_n[3, gi, gj, gk]
        elseif axis == 0x02  # y-edge
            gi, gj, gk = ci + di, cj + dj, ck + dk
            Px[slot, pid] = ey_p[1, gi, gj, gk]
            Py[slot, pid] = ey_p[2, gi, gj, gk]
            Pz[slot, pid] = ey_p[3, gi, gj, gk]
            Nx[slot, pid] = ey_n[1, gi, gj, gk]
            Ny[slot, pid] = ey_n[2, gi, gj, gk]
            Nz[slot, pid] = ey_n[3, gi, gj, gk]
        else  # z-edge
            gi, gj, gk = ci + di, cj + dj, ck + dk
            Px[slot, pid] = ez_p[1, gi, gj, gk]
            Py[slot, pid] = ez_p[2, gi, gj, gk]
            Pz[slot, pid] = ez_p[3, gi, gj, gk]
            Nx[slot, pid] = ez_n[1, gi, gj, gk]
            Ny[slot, pid] = ez_n[2, gi, gj, gk]
            Nz[slot, pid] = ez_n[3, gi, gj, gk]
        end
    end
    return nothing
end


# --------------------------------------------
# QEF solve on patches (chunked GPU)
# --------------------------------------------

struct _PatchPositions
    px::Vector{Float32}
    py::Vector{Float32}
    pz::Vector{Float32}
end

function _solve_qef_patches!(kernels::ODCKernels, pdata::_PatchData, qef_batch::Int)
    n = pdata.n_patches
    px = Vector{Float32}(undef, n)
    py = Vector{Float32}(undef, n)
    pz = Vector{Float32}(undef, n)

    E = 12
    B = qef_batch

    # chunk buffers (host)
    Px = zeros(Float32, E, B)
    Py = zeros(Float32, E, B)
    Pz = zeros(Float32, E, B)
    Nx = zeros(Float32, E, B)
    Ny = zeros(Float32, E, B)
    Nz = zeros(Float32, E, B)
    W = zeros(Float32, E, B)
    Cx = zeros(Float32, B)
    Cy = zeros(Float32, B)
    Cz = zeros(Float32, B)
    minx = zeros(Float32, B)
    miny = zeros(Float32, B)
    minz = zeros(Float32, B)
    maxx = zeros(Float32, B)
    maxy = zeros(Float32, B)
    maxz = zeros(Float32, B)

    # Pre-allocate device buffers once (16 arrays)
    Px_r = Reactant.to_rarray(Px)
    Py_r = Reactant.to_rarray(Py)
    Pz_r = Reactant.to_rarray(Pz)
    Nx_r = Reactant.to_rarray(Nx)
    Ny_r = Reactant.to_rarray(Ny)
    Nz_r = Reactant.to_rarray(Nz)
    W_r = Reactant.to_rarray(W)
    Cx_r = Reactant.to_rarray(Cx)
    Cy_r = Reactant.to_rarray(Cy)
    Cz_r = Reactant.to_rarray(Cz)
    minx_r = Reactant.to_rarray(minx)
    miny_r = Reactant.to_rarray(miny)
    minz_r = Reactant.to_rarray(minz)
    maxx_r = Reactant.to_rarray(maxx)
    maxy_r = Reactant.to_rarray(maxy)
    maxz_r = Reactant.to_rarray(maxz)

    for base in 0:B:(n-1)
        m = min(B, n - base)
        rng = (base+1):(base+m)

        # fill chunk
        @inbounds begin
            Px[:, 1:m] .= pdata.Px[:, rng]
            Py[:, 1:m] .= pdata.Py[:, rng]
            Pz[:, 1:m] .= pdata.Pz[:, rng]
            Nx[:, 1:m] .= pdata.Nx[:, rng]
            Ny[:, 1:m] .= pdata.Ny[:, rng]
            Nz[:, 1:m] .= pdata.Nz[:, rng]
            W[:, 1:m] .= pdata.W[:, rng]
            Cx[1:m] .= pdata.Cx[rng]
            Cy[1:m] .= pdata.Cy[rng]
            Cz[1:m] .= pdata.Cz[rng]
            minx[1:m] .= pdata.minx[rng]
            miny[1:m] .= pdata.miny[rng]
            minz[1:m] .= pdata.minz[rng]
            maxx[1:m] .= pdata.maxx[rng]
            maxy[1:m] .= pdata.maxy[rng]
            maxz[1:m] .= pdata.maxz[rng]
        end
        if m < B
            @inbounds begin
                Px[:, m+1:B] .= 0f0
                Py[:, m+1:B] .= 0f0
                Pz[:, m+1:B] .= 0f0
                Nx[:, m+1:B] .= 0f0
                Ny[:, m+1:B] .= 0f0
                Nz[:, m+1:B] .= 0f0
                W[:, m+1:B] .= 0f0
                Cx[m+1:B] .= 0f0
                Cy[m+1:B] .= 0f0
                Cz[m+1:B] .= 0f0
                minx[m+1:B] .= 0f0
                miny[m+1:B] .= 0f0
                minz[m+1:B] .= 0f0
                maxx[m+1:B] .= 0f0
                maxy[m+1:B] .= 0f0
                maxz[m+1:B] .= 0f0
            end
        end

        # Copy host → pre-allocated device buffers (no allocation)
        copyto!(Px_r, Px)
        copyto!(Py_r, Py)
        copyto!(Pz_r, Pz)
        copyto!(Nx_r, Nx)
        copyto!(Ny_r, Ny)
        copyto!(Nz_r, Nz)
        copyto!(W_r, W)
        copyto!(Cx_r, Cx)
        copyto!(Cy_r, Cy)
        copyto!(Cz_r, Cz)
        copyto!(minx_r, minx)
        copyto!(miny_r, miny)
        copyto!(minz_r, minz)
        copyto!(maxx_r, maxx)
        copyto!(maxy_r, maxy)
        copyto!(maxz_r, maxz)

        out_r = kernels.qef_solve(Px_r, Py_r, Pz_r, Nx_r, Ny_r, Nz_r, W_r, Cx_r, Cy_r, Cz_r,
            minx_r, miny_r, minz_r, maxx_r, maxy_r, maxz_r)
        out = Array(out_r) # (3,B)
        @inbounds begin
            px[rng] .= out[1, 1:m]
            py[rng] .= out[2, 1:m]
            pz[rng] .= out[3, 1:m]
        end
    end

    return _PatchPositions(px, py, pz)
end


# --------------------------------------------
# IC quad split (ODC appendix + Wang concavity)
# --------------------------------------------

@inline _sub(a::Point3f, b::Point3f) = Point3f(a.x - b.x, a.y - b.y, a.z - b.z)
@inline _dot(a::Point3f, b::Point3f) = a.x * b.x + a.y * b.y + a.z * b.z
@inline function _cross(a::Point3f, b::Point3f)
    return Point3f(a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x)
end

"""
Wang-style concavity test used by ODC Appendix A.1:

For a candidate envelope vertex q relative to diagonal endpoints (a,b) and edge endpoints p_in/p_out:
- q is concave if:
    (q - p_out) ⋅ ((a - p_out) × (b - p_out)) < 0   OR
    (q - p_in ) ⋅ ((a - p_in ) × (b - p_in )) > 0
"""
@inline function _is_concave(q::Point3f, a::Point3f, b::Point3f, p_in::Point3f, p_out::Point3f)
    t1 = _dot(_sub(q, p_out), _cross(_sub(a, p_out), _sub(b, p_out))) < 0f0
    t2 = _dot(_sub(q, p_in), _cross(_sub(a, p_in), _sub(b, p_in))) > 0f0
    return t1 || t2
end

@inline function _push_tri!(faces::Vector{TriangleFace{Int32}}, i1::Int32, i2::Int32, i3::Int32)
    push!(faces, TriangleFace{Int32}(i1, i2, i3))
    return nothing
end

function _triangulate_quad_ic!(faces::Vector{TriangleFace{Int32}},
    vertices::Vector{Point3f},
    i1::Int32, i2::Int32, i3::Int32, i4::Int32,
    p_in::Point3f, p_out::Point3f,
    p_edge::Point3f)

    p1 = vertices[Int(i1)]
    p2 = vertices[Int(i2)]
    p3 = vertices[Int(i3)]
    p4 = vertices[Int(i4)]

    # case1 uses diagonal (p1,p3): invalid if p2 or p4 concave wrt triangle (p1,p3, p_in/p_out)
    bad1 = _is_concave(p2, p1, p3, p_in, p_out) || _is_concave(p4, p1, p3, p_in, p_out)

    # case2 uses diagonal (p2,p4): invalid if p1 or p3 concave wrt (p2,p4, p_in/p_out)
    bad2 = _is_concave(p1, p2, p4, p_in, p_out) || _is_concave(p3, p2, p4, p_in, p_out)

    if !bad1 && bad2
        # case1: (1,2,3) + (1,3,4)
        _push_tri!(faces, i1, i2, i3)
        _push_tri!(faces, i1, i3, i4)
    elseif !bad2 && bad1
        # case2: (1,2,4) + (2,3,4)
        _push_tri!(faces, i1, i2, i4)
        _push_tri!(faces, i2, i3, i4)
    elseif !bad1 && !bad2
        # both enclosed: pick deterministic (case1)
        _push_tri!(faces, i1, i2, i3)
        _push_tri!(faces, i1, i3, i4)
    else
        # case3: fan from edge vertex p
        pe_idx = Int32(length(vertices) + 1)
        push!(vertices, p_edge)
        _push_tri!(faces, pe_idx, i1, i2)
        _push_tri!(faces, pe_idx, i2, i3)
        _push_tri!(faces, pe_idx, i3, i4)
        _push_tri!(faces, pe_idx, i4, i1)
    end
    return nothing
end


# --------------------------------------------
# Dual connectivity emission: per sign-change grid edge, connect 4 adjacent cells
# --------------------------------------------

# For each axis edge, we:
#  - identify the 4 incident cells in cyclic order around the edge for +axis direction
#  - decide edge direction by which endpoint is inside (<0) to orient p_in -> p_out
#  - reverse order if direction is negative
#  - look up per-cell patch id (vertex index) via cell_edge_patch[local_edge+1, cell]
#  - apply IC quad split.

function _emit_faces_x!(faces, vertices, cell_edge_patch, sdf, ex_p, active_edges,
    xmin, ymin, zmin, δ, nx, ny, nz)

    # local edge indices for the x-edge in the 4 surrounding cells (base order for +x)
    # cells: A(i, j-1,k-1), B(i, j,k-1), C(i, j,k), D(i, j-1,k)
    local_edges = Int8[6, 4, 0, 2]

    for l in active_edges
        i, j, k = _decode_ex(l, nx, ny)
        # Need 4 cells around edge → skip boundary
        if j == 1 || j == ny + 1 || k == 1 || k == nz + 1
            continue
        end
        # edge endpoints (grid vertices): (i,j,k) and (i+1,j,k)
        f0 = sdf[i, j, k]
        # direction: p_in -> p_out (inside -> outside)
        dirpos = (f0 < 0f0)  # if inside at lower x, direction is +x; else -x

        # cells (1-based cell coords)
        cA = (i, j - 1, k - 1)
        cB = (i, j, k - 1)
        cC = (i, j, k)
        cD = (i, j - 1, k)

        cells = dirpos ? (cA, cB, cC, cD) : (cD, cC, cB, cA)
        les = dirpos ? (local_edges[1], local_edges[2], local_edges[3], local_edges[4]) :
              (local_edges[4], local_edges[3], local_edges[2], local_edges[1])

        # patch vertex indices
        v = ntuple(t -> begin
                ci, cj, ck = cells[t]
                le = les[t]
                cell_edge_patch[Int(le)+1, ci, cj, ck]
            end, 4)

        if any(x -> x == 0, v)
            continue
        end

        i1 = Int32(v[1])
        i2 = Int32(v[2])
        i3 = Int32(v[3])
        i4 = Int32(v[4])

        # p_in/p_out
        x0 = xmin + δ * Float32(i - 1)
        y0 = ymin + δ * Float32(j - 1)
        z0 = zmin + δ * Float32(k - 1)

        p_lo = Point3f(x0, y0, z0)
        p_hi = Point3f(x0 + δ, y0, z0)
        p_in = dirpos ? p_lo : p_hi
        p_out = dirpos ? p_hi : p_lo

        p_edge = Point3f(ex_p[1, i, j, k], ex_p[2, i, j, k], ex_p[3, i, j, k])

        _triangulate_quad_ic!(faces, vertices, i1, i2, i3, i4, p_in, p_out, p_edge)
    end
end

function _emit_faces_y!(faces, vertices, cell_edge_patch, sdf, ey_p, active_edges,
    xmin, ymin, zmin, δ, nx, ny, nz)

    # y-edge around order for +y:
    # cells: A(i-1,j,k-1), B(i-1,j,k), C(i,j,k), D(i,j,k-1)
    local_edges = Int8[5, 1, 3, 7]  # matches those cells: A->e5, B->e1, C->e3, D->e7

    for l in active_edges
        i, j, k = _decode_ey(l, nx, ny)
        if i == 1 || i == nx + 1 || k == 1 || k == nz + 1
            continue
        end

        f0 = sdf[i, j, k]              # vertex (i,j,k)
        dirpos = (f0 < 0f0)            # if inside at lower y, direction +y else -y

        cA = (i - 1, j, k - 1)
        cB = (i - 1, j, k)
        cC = (i, j, k)
        cD = (i, j, k - 1)

        cells = dirpos ? (cA, cB, cC, cD) : (cD, cC, cB, cA)
        les = dirpos ? (local_edges[1], local_edges[2], local_edges[3], local_edges[4]) :
              (local_edges[4], local_edges[3], local_edges[2], local_edges[1])

        v = ntuple(t -> begin
                ci, cj, ck = cells[t]
                le = les[t]
                cell_edge_patch[Int(le)+1, ci, cj, ck]
            end, 4)
        if any(x -> x == 0, v)
            continue
        end

        i1 = Int32(v[1])
        i2 = Int32(v[2])
        i3 = Int32(v[3])
        i4 = Int32(v[4])

        x0 = xmin + δ * Float32(i - 1)
        y0 = ymin + δ * Float32(j - 1)
        z0 = zmin + δ * Float32(k - 1)

        p_lo = Point3f(x0, y0, z0)
        p_hi = Point3f(x0, y0 + δ, z0)
        p_in = dirpos ? p_lo : p_hi
        p_out = dirpos ? p_hi : p_lo

        p_edge = Point3f(ey_p[1, i, j, k], ey_p[2, i, j, k], ey_p[3, i, j, k])

        _triangulate_quad_ic!(faces, vertices, i1, i2, i3, i4, p_in, p_out, p_edge)
    end
end

function _emit_faces_z!(faces, vertices, cell_edge_patch, sdf, ez_p, active_edges,
    xmin, ymin, zmin, δ, nx, ny, nz)

    # z-edge around order for +z:
    # cells: A(i-1,j-1,k), B(i,j-1,k), C(i,j,k), D(i-1,j,k)
    local_edges = Int8[10, 11, 8, 9]  # A->e10, B->e11, C->e8, D->e9

    for l in active_edges
        i, j, k = _decode_ez(l, nx, ny)
        if i == 1 || i == nx + 1 || j == 1 || j == ny + 1
            continue
        end

        f0 = sdf[i, j, k]
        dirpos = (f0 < 0f0)  # if inside at lower z, direction +z else -z

        cA = (i - 1, j - 1, k)
        cB = (i, j - 1, k)
        cC = (i, j, k)
        cD = (i - 1, j, k)

        cells = dirpos ? (cA, cB, cC, cD) : (cD, cC, cB, cA)
        les = dirpos ? (local_edges[1], local_edges[2], local_edges[3], local_edges[4]) :
              (local_edges[4], local_edges[3], local_edges[2], local_edges[1])

        v = ntuple(t -> begin
                ci, cj, ck = cells[t]
                le = les[t]
                cell_edge_patch[Int(le)+1, ci, cj, ck]
            end, 4)
        if any(x -> x == 0, v)
            continue
        end

        i1 = Int32(v[1])
        i2 = Int32(v[2])
        i3 = Int32(v[3])
        i4 = Int32(v[4])

        x0 = xmin + δ * Float32(i - 1)
        y0 = ymin + δ * Float32(j - 1)
        z0 = zmin + δ * Float32(k - 1)

        p_lo = Point3f(x0, y0, z0)
        p_hi = Point3f(x0, y0, z0 + δ)
        p_in = dirpos ? p_lo : p_hi
        p_out = dirpos ? p_hi : p_lo

        p_edge = Point3f(ez_p[1, i, j, k], ez_p[2, i, j, k], ez_p[3, i, j, k])

        _triangulate_quad_ic!(faces, vertices, i1, i2, i3, i4, p_in, p_out, p_edge)
    end
end


# --------------------------------------------
# Manifold post-pass (ported from SDFODC)
# Splits non-manifold edges and vertices to guarantee 2-manifold output.
# --------------------------------------------

@inline function _edgekey(u::Int32, v::Int32)::UInt64
    a = UInt64(min(u, v))
    b = UInt64(max(u, v))
    return (a << 32) | b
end

@inline function _unpack_edgekey(key::UInt64)::Tuple{Int32,Int32}
    u = Int32(key >> 32)
    v = Int32(key & 0xffffffff)
    return u, v
end

@inline function _third_vertex(f::TriangleFace{Int32}, u::Int32, v::Int32)::Int32
    a, b, c = f[1], f[2], f[3]
    if a != u && a != v
        return a
    elseif b != u && b != v
        return b
    else
        return c
    end
end

function _manifoldize!(verts::Vector{Point3f}, faces::Vector{TriangleFace{Int32}})
    # ---- Pass 1: Split non-manifold edges (>2 incident faces) ----
    edge_faces = Dict{UInt64,Vector{Int32}}()
    for (fi, f) in enumerate(faces)
        a, b, c = f[1], f[2], f[3]
        for (u, v) in ((a, b), (b, c), (c, a))
            key = _edgekey(u, v)
            push!(get!(edge_faces, key, Int32[]), Int32(fi))
        end
    end

    for (key, flist) in edge_faces
        if length(flist) <= 2
            continue
        end

        u, v = _unpack_edgekey(key)
        pu = verts[Int(u)]
        pv = verts[Int(v)]
        d = _sub(pv, pu)
        inv_len = 1f0 / (sqrt(_dot(d, d)) + 1f-12)
        dn = Point3f(d[1] * inv_len, d[2] * inv_len, d[3] * inv_len)

        # Build a stable perpendicular basis (b1, b2) around edge axis dn
        ax = abs(dn[1]) < 0.9f0 ? Point3f(1, 0, 0) : Point3f(0, 1, 0)
        b1 = _cross(dn, ax)
        inv_b1 = 1f0 / (sqrt(_dot(b1, b1)) + 1f-12)
        b1 = Point3f(b1[1] * inv_b1, b1[2] * inv_b1, b1[3] * inv_b1)
        b2 = _cross(dn, b1)

        mid = Point3f(0.5f0 * (pu[1] + pv[1]), 0.5f0 * (pu[2] + pv[2]), 0.5f0 * (pu[3] + pv[3]))

        angles = Vector{Float32}(undef, length(flist))
        for (ii, fid) in enumerate(flist)
            f = faces[Int(fid)]
            w = _third_vertex(f, u, v)
            pw = verts[Int(w)]
            vec = _sub(pw, mid)
            x = _dot(vec, b1)
            y = _dot(vec, b2)
            angles[ii] = atan(y, x)
        end

        perm = sortperm(angles)
        sorted_f = flist[perm]

        # Keep first two faces on original (u,v); for the rest, duplicate u in pairs
        for start in 3:2:length(sorted_f)
            newu = Int32(length(verts) + 1)
            push!(verts, verts[Int(u)]) # duplicate position

            for fid in sorted_f[start:min(start + 1, length(sorted_f))]
                f = faces[Int(fid)]
                a, b, c = f[1], f[2], f[3]
                a = (a == u) ? newu : a
                b = (b == u) ? newu : b
                c = (c == u) ? newu : c
                faces[Int(fid)] = TriangleFace{Int32}(a, b, c)
            end
        end
    end

    # ---- Pass 2: Split non-manifold vertices by connected components ----
    faces_of_v = [Int32[] for _ in 1:length(verts)]
    for (fi, f) in enumerate(faces)
        a, b, c = f[1], f[2], f[3]
        push!(faces_of_v[Int(a)], Int32(fi))
        push!(faces_of_v[Int(b)], Int32(fi))
        push!(faces_of_v[Int(c)], Int32(fi))
    end

    for v in Int32(1):Int32(length(verts))
        fl = faces_of_v[Int(v)]
        if length(fl) <= 1
            continue
        end

        # Local union-find over incident faces
        m = length(fl)
        parent = collect(1:m)

        find(i) = (parent[i] == i ? i : (parent[i] = find(parent[i])))
        function unite(i, j)
            ri, rj = find(i), find(j)
            if ri != rj
                parent[rj] = ri
            end
        end

        neighbor_face = Dict{Int32,Int}()  # neighbor vertex -> local face idx
        for (li, fid) in enumerate(fl)
            f = faces[Int(fid)]
            a, b, c = f[1], f[2], f[3]
            n1::Int32 = 0
            n2::Int32 = 0
            if a == v
                n1, n2 = b, c
            elseif b == v
                n1, n2 = a, c
            else
                n1, n2 = a, b
            end

            for n in (n1, n2)
                if haskey(neighbor_face, n)
                    unite(li, neighbor_face[n])
                else
                    neighbor_face[n] = li
                end
            end
        end

        # Group by root
        groups = Dict{Int,Vector{Int32}}()
        for (li, fid) in enumerate(fl)
            r = find(li)
            push!(get!(groups, r, Int32[]), fid)
        end

        if length(groups) <= 1
            continue
        end

        first_root = first(keys(groups))
        for (r, gfaces) in groups
            if r == first_root
                continue
            end
            newv = Int32(length(verts) + 1)
            push!(verts, verts[Int(v)])

            for fid in gfaces
                f = faces[Int(fid)]
                a, b, c = f[1], f[2], f[3]
                a = (a == v) ? newv : a
                b = (b == v) ? newv : b
                c = (c == v) ? newv : c
                faces[Int(fid)] = TriangleFace{Int32}(a, b, c)
            end
        end
    end

    return nothing
end
