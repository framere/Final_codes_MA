using LinearAlgebra
using JLD2
using Printf

function load_matrix(filename::String)
    N = 11994
    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    # A = -A
    return Hermitian(A)
end

function off_diagonal_fronenius(A::Hermitian{T}) where T
    D = diag(A)
    diagonal_matrix = Diagonal(D) |> Matrix
    diagonal_matrix = Hermitian(diagonal_matrix)  # Ensure it's Hermitian
    off_diagonal_matrix = A - diagonal_matrix
    offdiagonal_norm = norm(off_diagonal_matrix, 2)  # Use 2-norm for Frobenius norm
    return offdiagonal_norm
end

systems = ["HFbasis", "RNDbasis1"] #, "RNDbasis1" , "RNDbasis2", "RNDbasis3"]

function main(system::String)
    filename = "../H2_molecule/gamma_VASP_" * system * ".dat"
    println("Reading eigenvalues from $system")
    A = load_matrix(filename)

    # Calculate off-diagonal Frobenius norm
    off_diag_norm = off_diagonal_fronenius(A)
    @printf("Off-diagonal Frobenius norm for system %s: %.6f\n", system, off_diag_norm)
    
end

for system in systems
    main(system)
end