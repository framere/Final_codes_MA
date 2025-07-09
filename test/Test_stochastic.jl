using LinearAlgebra
using Printf
using Random

function diagonal_matrix(N::Int)
    # Create a diagonal matrix with random values
    A = Matrix{Float64}(undef, N, N)

    for i in 1:N
        for j in 1:N
            if i == j
                A[i, j] = -1.0 /(1.0 +i^2)  # Random value on the diagonal
            else
                A[i, j] = 0.0000001  # Zero elsewhere
            end
        end
    end
    return Hermitian(A)  # Return as a Hermitian matrix
end

function stochastic_rotation(A::Hermitian{Float64, Matrix{Float64}}, N::Int)
    # Generate a random orthogonal matrix
    Urand = rand(N, N) .- 0.5
    qr_decomp = qr(Urand)
    U = Matrix(qr_decomp.Q)

    # Transform the matrix into the new basis
    A_new = U' * (A * U)  # Equivalent to A' = Qᵀ * A * Q
    return Hermitian(A_new)
end


N = 1000
A = diagonal_matrix(N)
A_stochastic = stochastic_rotation(A, N)

# println("Diagonal Matrix A:")
# display(A)
# println("Stochastic Matrix A_stochastic:")
# display(A_stochastic)


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

names = ["diagonal", "stochastic"]

for name in names
    filename = "analysis_" * name * ".txt"
    if name == "diagonal"
        count = analyze_diagonal_dominance(A, filename)
    else
        count = analyze_diagonal_dominance(A_stochastic, filename)
    end
    if count > 0
        println("Matrix is not diagonally dominant in $count rows for $name matrix.")
        println("Results written to $filename")
    else
        println("Matrix is diagonally dominant in all rows for $name matrix.")
        println("Results written to $filename")
    end
end