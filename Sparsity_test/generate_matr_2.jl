using LinearAlgebra
using Random

function new_curve(x, gamma, delta, epsilon)
    return gamma .* exp.(- delta .* x.^ epsilon)
end

function generate_random_matrix(N::Int, factor::Float64 = 100.0)
    """    generate_random_matrix(N::Int, factor::Float64 = 100.0) -> Matrix{Float64}
    Generates a random NxN matrix with diagonal elements scaled by `factor` and small off-diagonal elements.        
    The diagonal elements are uniformly distributed between 0 and `factor`, while off-diagonal elements are small random values.
    """
    α = -500.7142494929915
    β = 0.10931396203607915
    A = Matrix{Float64}(undef, N, N)
    for i in 1:N
        a = 31.2458198825
        b = 0.6473512253
        c = 0.3556968463
        for j in 1:N
            if i == j
                A[i, j] = - new_curve(i, a,b,c)  # Diagonal elements
            else
                a = rand()
                if a < 0.5
                    A[i, j] = - new_curve(i, a,b,c) / factor  # Small off-diagonal elements
                else
                    A[i, j] = 0  # Even smaller off-diagonal elements
                end
            end
        end
    end
    return -A
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
Ns = [30000]  # Different sizes for testing
factors = 10 .^ (2:2:10)  # [1e10, 1e2]

global counter = 1
for (N, factor) in Iterators.product(Ns, factors)
    println("Generating a random Hermitian matrix of size $N x $N with diagonal scaling factor $factor...")
    A = generate_random_matrix(N, factor)
    filename = "large_sparse_matrix_$(counter).dat"
    println("Saving the generated matrix to '$filename'...")
    save_matrix_to_file(A, filename)
    global counter += 1
end