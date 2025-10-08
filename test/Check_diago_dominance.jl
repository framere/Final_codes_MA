using LinearAlgebra
using Printf
using Random

function analyze_diagonal_dominance(A::AbstractMatrix{T}, output_filename::String) where T<:Number
    N = size(A, 1)
    
    # Open output file
    output_file = open(output_filename, "w")
    
    count_non_diago_dominant_rows = 0
    for i in 1:N
        diag_element = abs(A[i, i])
        off_diag_sum = sum(abs(A[i, j]) for j in 1:N if j != i)

        # Write to file
        @printf(output_file, "%.15e %.15e\n", diag_element, off_diag_sum)

        if diag_element <= off_diag_sum
            count_non_diago_dominant_rows += 1            
        end
    end
    
    close(output_file)
    
    return count_non_diago_dominant_rows
end

function load_sparse_matrix(filename::String, number::Int)
    if number % 2 == 0
        N = 20000
    elseif number % 2 == 1
        N = 10000
    else
        error("Unknown molecule: $number")
    end
    # println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    return Hermitian(A)
end

function load_matrix(filename::String, molecule::String)
    if molecule == "H2"
        N = 11994
    elseif molecule == "formaldehyde"
        N = 27643
    elseif molecule == "uracil"
        N = 32416
    else    
        error("Unknown molecule: $molecule")
    end
    # println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    return Hermitian(A)
end

number = 1:14
folder = "diagonalization_data/"

for i in number
    filename = "large_sparse_matrix_" * string(i) * ".dat"
    A = load_sparse_matrix(filename, i)
    output_filename = folder * "diagonal_dominance_sparse_matrix_" * string(i) * ".dat"
    count = analyze_diagonal_dominance(A, output_filename)
    println("Molecule $i: $count non-diagonally dominant rows")
end
