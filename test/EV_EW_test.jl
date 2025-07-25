using LinearAlgebra
using JLD2

function load_matrix(filename::String)
    N = 20000  

    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    # A = -A
    return Hermitian(A)
end

function diagonalize_and_save(filename::String)    
    A = load_matrix(filename)
    println("Diagonalizing the matrix ...")
    @time F = eigen(A)  # F.values, F.vectors

    output_file = "test_EW_results_1.jld2"
    println("Saving results to $output_file")

    jldsave(output_file; 
        eigenvalues = F.values, 
        eigenvectors = F.vectors
    )

    println("Done saving eigenvalues and eigenvectors.")
end


diagonalize_and_save("large_sparse_matrix_1.dat")

# open the file and read the eigenvalues
# function read_eigen_data(system::String; with_vectors::Bool=false)
#     output_file = "Eigenvalues_folder/eigen_results_$system.jld2"
#     println("Reading eigen data from $output_file")

#     data = jldopen(output_file, "r")
#     eigenvalues = data["eigenvalues"]
#     eigenvectors = with_vectors ? data["eigenvectors"] : nothing
#     close(data)

#     return eigenvalues, eigenvectors
# end


# # Example usage
# for system in names
#     eigenvalues, _ = read_eigen_data(system)
#     println("Top 10 eigenvalues for system $system: ", eigenvalues[1:10])
# end

