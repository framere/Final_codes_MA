using LinearAlgebra
using JLD2

function load_matrix(system::String)
    N = 11994

    filename = "../H2_molecule/gamma_VASP_" * system * ".dat"
    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    return Hermitian(A)
end

names = ["HFbasis", "RNDbasis1"] # , "RNDbasis2", "RNDbasis3"]

function diagonalize_and_save(system::String)    
    A = load_matrix(system)
    println("Diagonalizing the matrix for system: $system")
    @time F = eigen(A)  # F.values, F.vectors

    output_file = "../Eigenvalues_folder/eigenres_H2_$system.jld2"
    println("Saving results to $output_file")

    jldsave(output_file; 
        eigenvalues = F.values, 
        eigenvectors = F.vectors
    )

    println("Done saving eigenvalues and eigenvectors.")
end


# for system in names
#     diagonalize_and_save(system)
# end

# open the file and read the eigenvalues
function read_eigen_data(system::String; with_vectors::Bool=false)
    output_file = "../Eigenvalues_folder/eigenres_H2_$system.jld2"
    println("Reading eigen data from $output_file")

    data = jldopen(output_file, "r")
    eigenvalues = data["eigenvalues"]
    eigenvectors = with_vectors ? data["eigenvectors"] : nothing
    close(data)

    return eigenvalues, eigenvectors
end


# Example usage
for system in names
    eigenvalues, _ = read_eigen_data(system)
    println("Top 10 eigenvalues for system $system: ", eigenvalues[1:10])
end

