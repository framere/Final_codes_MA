using LinearAlgebra
using Random

function load_matrix(system::String)
    N = 27643
    filename = "formaldehyde/gamma_VASP_" * system * ".dat"
    println("Reading matrix from ", filename)

    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    A = Hermitian(A)  # Assuming the file stores a full Hermitian matrix
    return A
end

function transform_and_save_matrix(A::Hermitian{Float64, Matrix{Float64}}, out_filename::String)
    N = size(A, 1)
    
    println("Generating random orthogonal matrix...")
    @time Urand = rand(N, N) .- 0.5

    println("Performing QR decomposition to obtain orthogonal matrix...")
    @time qr_decomp = qr(Urand)
    U = Matrix(qr_decomp.Q)

    println("Transforming matrix into new basis...")
    @time A_new = U' * (A * U)  # Equivalent to A' = Qᵀ * A * Q

    println("Saving transformed matrix to ", out_filename)
    A_new_vec = vec(Matrix(A_new))  # Flatten to 1D
    open(out_filename, "w") do file
        write(file, A_new_vec)
    end
end

# === MAIN USAGE ===
system = "RNDbasis1"  # Replace with your actual system name
A = load_matrix(system)
transform_and_save_matrix(A, "transformed_RND1.dat")
