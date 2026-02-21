########################################   MeshSDFSampler   ########################################

"""
Area-weighted surface sampler.
"""
struct MeshSDFSampler
    vertices::Vector{Point3f}
    triangles::Vector{GLTriangleFace}
    # area-weighted categorical distribution over triangle faces stored as an alias table for O(1) sampling
    distribution::AliasTable{UInt32,Int}
    sdf::Array{Float32,3}
    resolution::Int
    # sdf grid mapping
    bbox_min::Point3f
    bbox_max::Point3f
    # shape parameters
    parameters::Vector{Float32}
end

function MeshSDFSampler(mesh::Mesh, parameters::Vector{Float32}; resolution::Int=256)
    mesh = canonicalize(mesh)
    vertices = coordinates(mesh)
    triangles = faces(mesh)
    sdf = construct_sdf(mesh, resolution)
    distribution = construct_triangle_distribution(vertices, triangles)
    (bbox_min, bbox_max) = compute_bounding_box(vertices)
    mesh_sampler = MeshSDFSampler(
        vertices, triangles, distribution, sdf, resolution, bbox_min, bbox_max, parameters
    )
    return mesh_sampler
end

function construct_triangle_distribution(
    vertices::Vector{Point3f}, triangles::Vector{GLTriangleFace}
)
    num_faces = length(triangles)
    weights = Vector{Float32}(undef, num_faces)

    @inbounds for index in eachindex(triangles)
        (i, j, k) = triangles[index]
        vertex_a = vertices[i]
        edge_1 = vertices[j] - vertex_a
        edge_2 = vertices[k] - vertex_a
        double_area = norm(edge_1 × edge_2)
        weights[index] = double_area
    end

    Σweights = sum(weights)
    @assert (Σweights > 0.0f0) "Mesh has zero total surface area"
    weights ./= Σweights
    distribution = AliasTable{UInt32,Int}(weights)
    return distribution
end

function compute_bounding_box(vertices::Vector{Point3f})
    @assert !isempty(vertices)
    vertex = first(vertices)
    x_min = x_max = vertex[1]
    y_min = y_max = vertex[2]
    z_min = z_max = vertex[3]

    @inbounds for vertex in vertices
        (x, y, z) = vertex
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)
        z_min = min(z_min, z)
        z_max = max(z_max, z)
    end

    bbox_min = Point3f(x_min, y_min, z_min)
    bbox_max = Point3f(x_max, y_max, z_max)
    return (bbox_min, bbox_max)
end

function Broadcast.broadcastable(mesh_sampler::MeshSDFSampler)
    return Ref(mesh_sampler)
end

function mean_and_std_parameters(mesh_samplers::AbstractVector{MeshSDFSampler})
    num_meshes = length(mesh_samplers)
    dim = length(first(mesh_samplers).parameters)
    μ = zeros(Float32, dim)

    for mesh_sampler in mesh_samplers
        parameters = mesh_sampler.parameters
        μ .+= parameters
    end
    μ ./= num_meshes

    σ = zeros(Float32, dim)
    for mesh_sampler in mesh_samplers
        parameters = mesh_sampler.parameters
        @. σ += (parameters - μ)^2
    end
    @. σ = sqrt(σ / (num_meshes - 1))
    return (μ, σ)
end

########################################   Sampling APIs   ########################################

struct SamplingParameters{N}
    rng::TaskLocalRNG
    num_samples::Int
    num_eikonal::Int
    voxel_size::Float32
    threshold_clamp::Float32
    threshold_eikonal::Float32
    slice_surface::UnitRange{Int}
    slice_band::UnitRange{Int}
    slice_volume::UnitRange{Int}
    subslices_band::NTuple{N,UnitRange{Int}}
    σs::NTuple{N,Float32}
end

function SamplingParameters(
    rng::TaskLocalRNG;
    num_samples::Int=131_072,
    grid_resolution::Int=256,
    ratio_eikonal::Float32=0.25f0,
    clamp_voxel_threshold::Int=16,
    eikonal_voxel_threshold::Int=2,
    splits::NamedTuple{(:surface, :band, :volume),NTuple{3,Float32}}=(; surface=0.2f0, band=0.7f0, volume=0.1f0),
    splits_band::NTuple{N,Float32}=(0.35f0, 0.3f0, 0.2f0, 0.15f0),
    voxel_σs::NTuple{N,Int}=(1, 4, 8, 12)
) where {N}
    (box_min, box_max) = (-1.0f0, 1.0f0)
    num_eikonal = round(Int, num_samples * ratio_eikonal)

    Δ = box_max - box_min
    voxel_size = Δ / (grid_resolution - 1)
    threshold_clamp = voxel_size * clamp_voxel_threshold
    threshold_eikonal = voxel_size * eikonal_voxel_threshold
    σs = voxel_size .* voxel_σs

    (slice_surface, slice_band, slice_volume) = partition_slice(1:num_samples, splits)
    subslices_band = partition_slice(slice_band, splits_band)
    return SamplingParameters{N}(
        rng,
        num_samples,
        num_eikonal,
        voxel_size,
        threshold_clamp,
        threshold_eikonal,
        slice_surface,
        slice_band,
        slice_volume,
        subslices_band,
        σs
    )
end

function Broadcast.broadcastable(params::SamplingParameters)
    return Ref(params)
end

function partition_slice(
    slice::AbstractUnitRange{Int},
    splits::Union{NamedTuple{<:Any,NTuple{N,Float32}},NTuple{N,Float32}}
) where {N}
    @assert isone(sum(splits))
    @assert all(>(0.0f0), splits)
    n = length(slice)
    Σsplits = cumsum(Tuple(splits))
    endpoints = round.(Int, Σsplits .* n)
    # new tuple with last = n
    endpoints = Base.setindex(endpoints, n, N)
    starts = ntuple(i -> i == 1 ? 1 : endpoints[i-1] + 1, Val(N))
    fn = (s, e) -> slice[s:e]
    partition = fn.(starts, endpoints)
    return partition
end

#######################################   Sampling Buffers   #######################################

struct SDFSamplingBuffer
    points::Matrix{Float32}
    signed_dists::Vector{Float32}
end

function SDFSamplingBuffer(params::SamplingParameters)
    points = Matrix{Float32}(undef, 3, params.num_samples)
    signed_dists = Vector{Float32}(undef, params.num_samples)
    return SDFSamplingBuffer(points, signed_dists)
end

struct EikonalSamplingBuffer
    sdf_buffer::SDFSamplingBuffer
    indices_eikonal::Vector{Int}
    points_eikonal::Vector{Float32}
end

function EikonalSamplingBuffer(params::SamplingParameters)
    sdf_buffer = SDFSamplingBuffer(params)
    num_off_surface = params.num_samples - last(params.slice_surface)
    indices_eikonal = Vector{Int}(undef, num_off_surface)
    points_eikonal = Vector{Float32}(undef, 3 * params.num_eikonal)
    return EikonalSamplingBuffer(sdf_buffer, indices_eikonal, points_eikonal)
end

######################################   Sampling Functions   ######################################

@inline function sample(rng::AbstractRNG, cat_distr::AliasTable{T,Int}) where {T}
    seed = rand(rng, T)
    return AliasTables.sample(seed, cat_distr)
end

@inline function sample_mesh_surface(
    vertices::Vector{Point3f},
    triangles::Vector{GLTriangleFace},
    cat_distr::AliasTable{UInt32,Int},
    rng::AbstractRNG
)
    # sample a triangle face
    index = sample(rng, cat_distr)
    @inbounds begin
        (i, j, k) = triangles[index]
        vertex_a = vertices[i]
        vertex_b = vertices[j]
        vertex_c = vertices[k]
    end
    u = √(rand(rng, Float32))
    v = rand(rng, Float32)

    w_a = 1.0f0 - u
    w_b = u * (1.0f0 - v)
    w_c = u * v
    point = w_a * vertex_a + w_b * vertex_b + w_c * vertex_c
    return point
end

@inline function sample_mesh_surface!(
    points::AbstractMatrix{Float32},
    vertices::Vector{Point3f},
    triangles::Vector{GLTriangleFace},
    cat_distr::AliasTable{UInt32,Int},
    rng::AbstractRNG
)
    @assert size(points, 1) == 3
    @inbounds for j in axes(points, 2)
        (x, y, z) = sample_mesh_surface(vertices, triangles, cat_distr, rng)
        points[1, j] = x
        points[2, j] = y
        points[3, j] = z
    end
    return points
end

@inline function sample_near_mesh_surface!(
    points::AbstractMatrix{Float32},
    vertices::Vector{Point3f},
    triangles::Vector{GLTriangleFace},
    cat_distr::AliasTable{UInt32,Int},
    σ::Float32,
    rng::AbstractRNG
)
    @assert size(points, 1) == 3
    @inbounds for j in axes(points, 2)
        (x, y, z) = sample_mesh_surface(vertices, triangles, cat_distr, rng)
        points[1, j] = perturb(x, σ, rng)
        points[2, j] = perturb(y, σ, rng)
        points[3, j] = perturb(z, σ, rng)
    end
    return points
end

# add Gaussian noise to the x, while keeping it inside [-1, 1]
@inline function perturb(x::Float32, σ::Float32, rng::AbstractRNG)
    while true
        x̃ = x + σ * randn(rng, Float32)
        (abs(x̃) > 1.0f0) || return x̃
    end
end

# populate `points` with samples uniformly distributed in [-1, 1]
@inline function sample_globally!(points::AbstractMatrix{Float32}, rng::AbstractRNG)
    @assert size(points, 1) == 3
    rand!(rng, points)                   # U[0,1)
    @. points = 2.0f0 * points - 1.0f0   # U[-1,1)
    return points
end

function sample_sdf_and_eikonal_points!(
    buffer::EikonalSamplingBuffer, sampler::MeshSDFSampler, params::SamplingParameters{N}
) where {N}
    rng = params.rng
    num_eikonal = params.num_eikonal
    threshold_eikonal = params.threshold_eikonal
    slice_off_surface = first(params.slice_band):params.num_samples

    (points, signed_dists, mesh_params) = sample_sdf_points!(buffer.sdf_buffer, sampler, params)
    points_off_surface = @view points[:, slice_off_surface]
    signed_dists_off_surface = @view signed_dists[slice_off_surface]

    # find eikonal candidate indices (into the off-surface view) without allocating
    indices = buffer.indices_eikonal
    candidates = find_eikonal_indices!(indices, signed_dists_off_surface, threshold_eikonal)
    indices_eikonal = take_random_subset!(candidates, num_eikonal, rng)

    # gather eikonal points into pre-allocated buffer
    num_eikonal = length(indices_eikonal)
    memory = buffer.points_eikonal
    resize!(memory, 3 * num_eikonal)
    points_eikonal = reshape(memory, 3, num_eikonal)
    @inbounds for (col_dst, col_src) in enumerate(indices_eikonal)
        points_eikonal[1, col_dst] = points_off_surface[1, col_src]
        points_eikonal[2, col_dst] = points_off_surface[2, col_src]
        points_eikonal[3, col_dst] = points_off_surface[3, col_src]
    end
    return (points, signed_dists, points_eikonal, mesh_params)
end

function sample_sdf_points!(
    buffer::SDFSamplingBuffer, sampler::MeshSDFSampler, params::SamplingParameters{N}
) where {N}
    rng = params.rng
    num_samples = params.num_samples
    voxel_size = params.voxel_size
    slice_surface = params.slice_surface
    slice_band = params.slice_band
    slice_volume = params.slice_volume
    slice_off_surface = first(slice_band):num_samples
    subslices_band = params.subslices_band
    σs = params.σs

    vertices = sampler.vertices
    triangles = sampler.triangles
    cat_distr = sampler.distribution
    sdf = sampler.sdf

    points = buffer.points
    signed_dists = buffer.signed_dists

    @inbounds begin
        points_surface = @view points[:, slice_surface]
        points_volume = @view points[:, slice_volume]

        sample_mesh_surface!(points_surface, vertices, triangles, cat_distr, rng)
        sample_globally!(points_volume, rng)
        for i in 1:N
            σ = σs[i]
            slice = subslices_band[i]
            points_band = @view points[:, slice]
            sample_near_mesh_surface!(points_band, vertices, triangles, cat_distr, σ, rng)
        end

        # compute corresponding signed distances via trilinear interpolation
        (n_x, n_y, n_z) = size(sdf) .- 1
        signed_dists[slice_surface] .= 0.0f0
        @batch for j in slice_off_surface
            signed_dists[j] = trilerp_sdf(
                sdf, points[1, j], points[2, j], points[3, j],
                voxel_size, voxel_size, voxel_size,
                n_x, n_y, n_z
            )
        end
    end
    return (points, signed_dists, sampler.parameters)
end

###########################################   Helpers   ###########################################

"""
Scan `signed_dists` for entries exceeding `threshold` in absolute value,
writing their indices into `indices` in-place. Replaces `findall` to avoid allocation.
"""
@inline function find_eikonal_indices!(
    indices::Vector{Int}, signed_dists::AbstractVector{Float32}, threshold::Float32
)
    count = 0
    @inbounds for j in eachindex(signed_dists)
        if abs(signed_dists[j]) > threshold
            count += 1
            indices[count] = j
        end
    end
    subindices = @view indices[1:count]
    return subindices
end

# partial Fisher–Yates shuffle on a view — returns @view indices[1:k]
@inline function take_random_subset!(
    indices::T, k::Int, rng::AbstractRNG
) where {T<:SubArray{Int,1,Vector{Int}}}
    n = length(indices)
    (k < n) || return indices::T
    @inbounds for i in 1:k
        j = rand(rng, i:n)
        (indices[i], indices[j]) = (indices[j], indices[i])
    end
    subindices = @view indices[1:k]
    return subindices::T
end
