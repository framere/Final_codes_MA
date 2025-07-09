using LinearAlgebra
using Printf

function load_matrix(filename::String)
    N = 27643
    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    return Hermitian(A)
end

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

# function main(system::String)
#     filename = "formaldehyde/gamma_VASP_" * system * ".dat"
#     output_filename = "diagonal_analysis_" * system * ".txt"
    
#     println("Loading matrix from: $filename")
#     A = load_matrix(filename)

#     non_dominant_count = analyze_diagonal_dominance(A, output_filename)
#     if non_dominant_count > 0
#         println("Matrix is not diagonally dominant in $non_dominant_count rows.")
#         println("Results written to $output_filename")
#     else
#         println("Matrix is diagonally dominant in all rows.")
#         println("Results written to $output_filename")
#     end
# end

# systems = ["HFbasis", "RNDbasis1"] 

# for system in systems
#     main(system)
# end

function define_stochastic_matrix(size::Int)
    """Generates a random stochastic matrix of given size."""
    matrix = rand(size, size)
    matrix ./= sum(matrix, dims=2)  # Normalize rows to sum to 1
    return matrix
end 

function main()
    size = 2000  # Example size
    A = define_stochastic_matrix(size)
    output_filename = "stochastic_matrix.txt"
    non_dominant_count = analyze_diagonal_dominance(A, output_filename)
    if non_dominant_count > 0
        println("Stochastic matrix is not diagonally dominant in $non_dominant_count rows.")
    else
        println("Stochastic matrix is diagonally dominant in all rows.")
    end    
    println("Stochastic matrix saved to $output_filename")
end

main()