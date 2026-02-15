using Pkg
Pkg.activate(@__DIR__)

using Base.Order: Forward
using ConditionalDeepSDFs: MeshSDFSampler
using CSV
using DataStructures
using FileIO
using GeometryBasics
using JLD2
using MeshIO

const MeshType = Mesh{3,Float32,GLTriangleFace,(:position, :normal),Tuple{Vector{Point3f},Vector{Vec3f}},Vector{GLTriangleFace}}

function extract_id(id::Int)
    return id
end
function extract_id(str::AbstractString)
    m = match(r"\d+$", str)
    id = isnothing(m) ? 0 : parse(Int, m.match)
    return id
end

function read_csv(csv_path::String; header::Bool=false, delim::Char, ignorerepeated::Bool)
    isfile(csv_path) || error("CSV file not found: $csv_path")
    csv_data = CSV.File(csv_path; header, delim, ignorerepeated)
    num_rows = length(csv_data)
    num_cols = length(csv_data.names)
    display(csv_data)
    return (csv_data, num_rows, num_cols)
end

function read_data_batch(
    csv_path::String,
    mesh_dir::String,
    mesh_file::String,
    folder_pattern::Regex;
    delim::Char,
    ignorerepeated::Bool
)
    (csv_data, num_rows, num_cols) = read_csv(csv_path; delim, ignorerepeated)
    (num_cols == 5) || error("Expected 5 columns in the CSV file, found $num_cols")

    folders = readdir(mesh_dir; sort=true)
    filter!(str -> occursin(folder_pattern, str), folders)
    num_meshes = length(folders)
    (num_rows == num_meshes) || @warn "Number of rows in CSV file does not match number of meshes"

    mesh_dict = SortedDict{Int,Tuple{MeshType,Point4f}}(Forward)
    sizehint!(mesh_dict, num_meshes)

    for folder in folders
        id = extract_id(folder)
        mesh_path = joinpath(mesh_dir, folder, mesh_file)
        mesh = load(mesh_path)

        index = id + 1
        row = csv_data[index]
        tuple_row = Tuple(row)
        id_csv = extract_id(tuple_row[1])
        @assert id == id_csv "ID mismatch: $id != $id_csv"
        params = Point4f(tuple_row[2:end])
        mesh_dict[id] = (mesh, params)
    end
    return mesh_dict
end

save_path = normpath(joinpath(@__DIR__, "..", "data/preprocessed/mesh_samplers.jld2"))

# batch 1
csv_path = normpath(joinpath(@__DIR__, "..", "data/batch_1/coldplate_variants_batch_1.csv"))
mesh_dir = normpath(joinpath(@__DIR__, "..", "data/batch_1/coldplate_41_variants_stl"))
mesh_file = "Body_6.stl"
folder_pattern = r"^run_id\d*$"i
mesh_dict_1 = read_data_batch(csv_path, mesh_dir, mesh_file, folder_pattern; delim=',', ignorerepeated=false)

# batch 2
csv_path = normpath(joinpath(@__DIR__, "..", "data/batch_2/coldplate_variants_batch_2.txt"))
mesh_dir = normpath(joinpath(@__DIR__, "..", "data/batch_2/coldplate_inst_stl"))
mesh_file = "Body_6.stl"
folder_pattern = r"^coldplate_inst\d*$"i
mesh_dict_2 = read_data_batch(csv_path, mesh_dir, mesh_file, folder_pattern; delim=' ', ignorerepeated=true)

offset = length(mesh_dict_1)
len = offset + length(mesh_dict_2)
mesh_samplers = Vector{MeshSDFSampler}(undef, len)

# batch 1
for (i, (_, (mesh, params))) in enumerate(mesh_dict_1)
    mesh_samplers[i] = MeshSDFSampler(mesh, params)
end
# batch 2
for (i, (_, (mesh, params))) in enumerate(mesh_dict_2)
    mesh_samplers[offset+i] = MeshSDFSampler(mesh, params)
end

all(index -> isassigned(mesh_samplers, index), eachindex(mesh_samplers)) || error("Not all mesh samplers have been constructed.")

save_object(save_path, mesh_samplers)
