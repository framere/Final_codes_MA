using LinearAlgebra
using JLD2

function load_matrix(number::Int)
    filename = "large_sparse_matrix_$number.dat"
    println("read ", filename)
    file = open(filename, "r")
    if number % 2 == 0
        N = 20000
    else
        N = 10000
    end
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    A = -A
    return Hermitian(A)
end

function diagonalize_and_save(number::Int)
    A = load_matrix(number)
    println("Diagonalizing the matrix for system: $number")
    @time Σexact, Uexact = eigen(A)
    output_file="Eigenvalues_folder/eigenresults_matrix_$(number).jld2"
    println("Saving results to $output_file")
    jldsave(output_file; Σexact, Uexact)  # JLD2 format
    println("Done!")
end

for system in 1:6
    println("Processing system: $system")
    diagonalize_and_save(system)
end

#------------------------- function to open and read the saved eigenvalues and eigenvectors -------------------------
# function load_eigenresults(output_file="eigen_results.jld2")
#     # Unpack directly into variables
#     data = load(output_file)  # Returns a Dict-like object
#     Σexact = data["Σexact"]  # Access by key
#     Uexact = data["Uexact"]
#     return Σexact, Uexact
# end