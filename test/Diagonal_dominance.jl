using LinearAlgebra
using JLD2
using Printf

function load_matrix(filename::String, molecule::String)
    if molecule == "H2"
        N = 11994
    else
        N = 27643
    end
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
    println("Calculating diagonal elements")
    @time diagonal_matrix = Diagonal(D) |> Matrix
    diagonal_matrix = Hermitian(diagonal_matrix)  # Ensure it's Hermitian
    off_diagonal_matrix = A - diagonal_matrix
    println("Calculating off-diagonal norm")
    @time offdiagonal_norm = norm(off_diagonal_matrix, Frobenius())  # Corrected: use Frobenius norm
    return offdiagonal_norm
end

systems = ["HFbasis", "RNDbasis1"] #, "RNDbasis1" , "RNDbasis2", "RNDbasis3"]
molecules = ["H2", "formaldehyde"]

function main(molecule::String, system::String)
    filename = "../" *molecule* "/gamma_VASP_" * system * ".dat"
    println("Reading eigenvalues from $system")
    A = load_matrix(filename, molecule)
    println("Matrix loaded from $filename, size: $(size(A))")
    # Calculate off-diagonal Frobenius norm
    off_diag_norm = off_diagonal_fronenius(A)
    @printf("Off-diagonal Frobenius norm for system %s: %.6f\n", system, off_diag_norm)
end

for molecule in molecules
    println("Processing molecule: $molecule")
    for system in systems
        main(molecule, system)
    end
end
