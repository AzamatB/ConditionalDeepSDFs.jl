###   Lazy representation of the cubic n-grid [-1, 1]³ as a 4D array of shape (3 × n × n × n)   ###

struct LazyUnitCubeGrid <: AbstractArray{Float32,4}
    coords::StepRangeLen{Float32,Float64,Float64,Int}
    n::Int
end

function LazyUnitCubeGrid(n::Int)
    return LazyUnitCubeGrid(range(-1.0f0, 1.0f0; length=n), n)
end

function Base.size(grid::LazyUnitCubeGrid)
    n = grid.n
    return (3, n, n, n)
end

Base.@propagate_inbounds function Base.getindex(
    grid::LazyUnitCubeGrid, d::Int, i::Int, j::Int, k::Int
)
    @boundscheck checkbounds(grid, d, i, j, k)
    coords = grid.coords
    @inbounds (coords[i], coords[j], coords[k])[d]
end

# intercept your exact chunk-slicing syntax
Base.@propagate_inbounds function Base.getindex(
    grid::LazyUnitCubeGrid, ::Colon, ::Colon, ::Colon, slice_k::AbstractUnitRange{<:Integer}
)
    coords = grid.coords
    @boundscheck checkbounds(coords, slice_k)
    n = grid.n
    n_k = length(slice_k)
    out = Array{Float32}(undef, 3, n, n, n_k)
    @inbounds for (idx_k, k) in enumerate(slice_k)
        z = coords[k]
        for j in 1:n
            y = coords[j]
            for i in 1:n
                x = coords[i]
                out[1, i, j, idx_k] = x
                out[2, i, j, idx_k] = y
                out[3, i, j, idx_k] = z
            end
        end
    end
    return out::Array{Float32,4}
end

function fill_grid_slab!(
    out::AbstractArray{Float32,4}, grid::LazyUnitCubeGrid, slice_k::AbstractUnitRange{<:Integer}
)
    coords = grid.coords
    n = grid.n
    @boundscheck begin
        checkbounds(coords, slice_k)
        expected = (3, n, n, length(slice_k))
        size(out) == expected || throw(
            DimensionMismatch("Output buffer has size $(size(out)), expected $expected")
        )
    end

    @inbounds for (idx_k, k) in enumerate(slice_k)
        z = coords[k]
        for j in 1:n
            y = coords[j]
            for i in 1:n
                x = coords[i]
                out[1, i, j, idx_k] = x
                out[2, i, j, idx_k] = y
                out[3, i, j, idx_k] = z
            end
        end
    end
    return out
end

#######################   Contiguous slabs iterator over LazyUnitCubeGrid   #######################

"""
    GridSlabs(grid, slab_size)

Indexable iterator over contiguous k-slabs of a `LazyUnitCubeGrid`. Each element is a
`(3, n, n, slab_size)` `Array{Float32,4}`. `slab_size` must evenly divide grid resolution `grid.n`.
"""
struct GridSlabs
    grid::LazyUnitCubeGrid
    slab_size::Int
    len::Int
    buffer::Array{Float32,4}

    function GridSlabs(grid_size::Int, slab_size::Int)
        grid = LazyUnitCubeGrid(grid_size)
        iszero(grid_size % slab_size) || throw(
            ArgumentError("slab_size=$slab_size does not divide grid_size=$grid_size")
        )
        len = grid_size ÷ slab_size
        buffer = Array{Float32,4}(undef, 3, grid_size, grid_size, slab_size)
        return new(grid, slab_size, len, buffer)
    end
end

function Base.length(grid_slabs::GridSlabs)
    return grid_slabs.len
end
function Base.size(grid_slabs::GridSlabs)
    return (length(grid_slabs),)
end
function Base.eltype(::Type{GridSlabs})
    return Array{Float32,4}
end
function Base.eachindex(grid_slabs::GridSlabs)
    return 1:length(grid_slabs)
end

function Base.getindex(grid_slabs::GridSlabs, i::Int)
    @boundscheck (1 <= i <= length(grid_slabs)) || throw(BoundsError(grid_slabs, i))
    slab_size = grid_slabs.slab_size
    start_k = (i - 1) * slab_size + 1
    end_k = start_k + slab_size - 1
    slice_k = start_k:end_k
    out = grid_slabs.buffer
    fill_grid_slab!(out, grid_slabs.grid, slice_k)
    return out
end

function Base.iterate(grid_slabs::GridSlabs, i::Int=1)
    i > length(grid_slabs) && return nothing
    return (grid_slabs[i], i + 1)
end

"""
    slab_points(grid_slabs::GridSlabs, i::Int) -> Matrix{Float32}

Return points of slab `i` flattened into a `(3, n² × slab_size)` matrix.
"""
function slab_points(grid_slabs::GridSlabs, i::Int)
    return reshape(grid_slabs[i], 3, :)
end

"""
    point_indices(grid_slabs::GridSlabs, i::Int) -> UnitRange{Int}

Return the range of linear indices of the grid points covered by slab `i`.
Points are numbered `1:n³` in column-major order over the spatial dimensions `(n, n, n)`.
"""
function point_indices(grid_slabs::GridSlabs, i::Int)
    @boundscheck (1 <= i <= length(grid_slabs)) || throw(BoundsError(grid_slabs, i))
    n = grid_slabs.grid.n
    stride = n * n * grid_slabs.slab_size
    idx_end = i * stride
    idx_start = idx_end - stride + 1
    return idx_start:idx_end
end
