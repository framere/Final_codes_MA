using LinearAlgebra
using Random

function generate_random_matrix(N::Int, factor::Float64 = 100.0)
    """    generate_random_matrix(N::Int, factor::Float64 = 100.0) -> Matrix{Float64}
    Generates a random NxN matrix with diagonal elements scaled by `factor` and small off-diagonal elements.        
    The diagonal elements are uniformly distributed between 0 and `factor`, while off-diagonal elements are small random values.
    """
    α = -500.7142494929915
    β = 0.10931396203607915
    A = Matrix{Float64}(undef, N, N)
    for i in 1:N
        for j in 1:N
            if i == j
                A[i, j] = α / (β * i^2)  # Diagonal elements
            else
                a = rand()
                if a < 0.5
                    A[i, j] = - abs(rand()) / factor  # Small off-diagonal elements
                else
                    A[i, j] = 0  # Even smaller off-diagonal elements
                end
            end
        end
    end
    return A
end

function save_matrix_to_file(A::Matrix{Float64}, filename::String)
    """    save_matrix_to_file(A::Matrix{Float64}, filename::String)
    Saves the Hermitian matrix `A` to a file in a flattened format.
    """
    A_vec = vec(Matrix(A))  # Flatten to 1D
    open(filename, "w") do file
        write(file, A_vec)
    end
end

# === MAIN USAGE ===
N = 12000  # Size of the matrix
factor = 100.0  # Scaling factor for diagonal elements
println("Generating a random Hermitian matrix of size $N x $N with diagonal scaling factor $factor...")
A = generate_random_matrix(N, factor)
println("Saving the generated matrix to 'large_sparse_matrix_3.dat'...")
save_matrix_to_file(A, "large_sparse_matrix_3.dat")
