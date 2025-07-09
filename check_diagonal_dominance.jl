using LinearAlgebra
using Printf
using JLD2

function load_matrix(filename::String)
    N = 27643
    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    # A = -A
    return Hermitian(A)
end

function check_diagonal_dominance(A::AbstractMatrix{T}) where T<:Number
    N = size(A, 1)
    
    count_non_diago_dominant_rows = 0
    for i in 1:N
        diag_element = abs(A[i, i])
        off_diag_sum = sum(abs(A[i, j]) for j in 1:N if j != i)

        if diag_element <= off_diag_sum
            count_non_diago_dominant_rows += 1            
        end
    end

    return count_non_diago_dominant_rows
end

function main(system::String)
    filename = "formaldehyde/gamma_VASP_" * system * ".dat"
    println("Loading matrix from: $filename")
    A = load_matrix(filename)

    non_dominant_count = check_diagonal_dominance(A)
    if non_dominant_count > 0
        println("Matrix is not diagonally dominant in $non_dominant_count rows.")
    else
        println("Matrix is diagonally dominant in all rows.")
    end
end

systems = ["HFbasis", "RNDbasis1"] 

for system in systems
    main(system)
end
