using Test

using CUDA
using FileIO
using MeshIO

include(joinpath(@__DIR__, "..", "src", "canonicalization.jl"))
include(joinpath(@__DIR__, "..", "src", "sdf.jl"))

function count_negative_columns(sdf_cpu::Array{Float32,3}, z_idx::Int)
    return count(<(0.0f0), @view sdf_cpu[:, :, z_idx])
end

function load_canonical_mesh(stl_name::AbstractString)
    mesh_path = joinpath(@__DIR__, "..", "data", "stl_files", stl_name)
    return canonicalize(load(mesh_path))
end

@testset "SDF parity regression" begin
    if !CUDA.functional()
        @info "CUDA is not functional; skipping GPU SDF regression checks."
        @test true
    else
        @testset "Dataset sanity for 28-31" begin
            for id in 28:31
                mesh = load_canonical_mesh("$(id).stl")
                sdf = construct_sdf(mesh, 256)
                sdf_cpu = Array(sdf)

                @test count_negative_columns(sdf_cpu, size(sdf_cpu, 3)) == 0
                @test count_negative_columns(sdf_cpu, 1) == 0
            end
        end

        @testset "No top-column artifact for 32" begin
            mesh = load_canonical_mesh("32.stl")
            sdf = construct_sdf(mesh, 256)
            sdf_cpu = Array(sdf)

            @test count_negative_columns(sdf_cpu, size(sdf_cpu, 3)) == 0
            @test count_negative_columns(sdf_cpu, 1) == 0

            reconstructed = construct_mesh(sdf_cpu)
            z_max = maximum(p[3] for p in coordinates(reconstructed))
            @test z_max < 0.4f0
        end
    end
end
