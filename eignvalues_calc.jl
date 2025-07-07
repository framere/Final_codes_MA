using LinearAlgebra
using JLD2

function load_matrix(system::String)
    N = 27643

    filename = "formaldehyde/gamma_VASP_" * system * ".dat"
    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    A = -A
    return Hermitian(A)
end

names = ["HFbasis", "RNDbasis1", "RNDbasis2", "RNDbasis3"]

function diagonalize_and_save(system::String)
    A = load_matrix(system)
    println("Diagonalizing the matrix for system: $system")
    @time F = eigen(A)
    true_EVs = -sqrt.(abs.(F.values))
    output_file="../Eigenvalues_folder/eigen_results_$system.jld2"
    println("Saving results to $output_file")
    jldsave(output_file; true_EVs)  # JLD2 format
    println("Done!")
end


for system in names
    diagonalize_and_save(system)
end

