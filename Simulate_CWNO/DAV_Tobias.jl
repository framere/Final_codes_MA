using LinearAlgebra
using Printf
using JLD2

# === Global FLOP counter and helpers ===
global NFLOPs = 0

include("../FLOP_count.jl")

function load_matrix(filename::String)
    N = 29791  

    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    A = Hermitian(A)
    return -A
end

function load_vector_from_file(filename::String)
    # read raw bytes and reinterpret as Float64
    bytes = read(filename)
    return reinterpret(Float64, bytes)
end


function read_eigenresults(number::Integer)
    output_file = "./CWNO_final_tilde_results.jld2"
    println("Reading eigenvalues from $output_file")
    data = jldopen(output_file, "r")
    eigenvalues = data["eigenvalues"]
    close(data)
    return sort(eigenvalues)
end

function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Integer,
    thresh::Float64, 
    max_iter::Integer,
    min_number_iter::Integer
)::Tuple{Vector{T},Matrix{T}} where T<:Number
    
    global NFLOPs

    Nlow = size(V, 2)
    if Naux < Nlow
        println("ERROR: auxiliary basis must not be smaller than number of target eigenvalues")
    end

    D = diag(A)
    iter = 0

    while true
        iter += 1

        # Orthogonalize against occupied orbital
        phi_occ = load_vector_from_file("occupied_orbital.dat")
        norm_phi_occ = phi_occ' * phi_occ  # scalar normalization

        for j in 1:size(V, 2)
            V[:, j] -= phi_occ * (phi_occ' * V[:, j]) / norm_phi_occ
        end

        # QR-Orthogonalisierung
        count_qr_flops(size(V,1), size(V,2))
        qr_decomp = qr(V)
        V = Matrix(qr_decomp.Q)
        
        # Rayleigh-Matrix: H = V' * (A * V)
        temp = A * V
        count_matmul_flops(size(A,1), size(A,2), size(V,2))  # A*V
        H = V' * temp
        count_matmul_flops(size(V,2), size(V,1), size(temp,2))  # V'*temp
        
        H = Hermitian(H)
        Σ, U = eigen(H, 1:Nlow)
        count_diag_flops(size(H,1))  # kleine Diagonalisierung
        
        X = V * U
        count_matmul_flops(size(V,1), size(V,2), size(U,2))  # V*U
        
        if iter > max_iter
            println("Maximum iterations reached without convergence.")
            return (Σ, X)
        end
        
        # R = X*Σ' - A*X
        R = X .* Σ'  # Skalierung
        temp2 = A * X
        count_matmul_flops(size(A,1), size(A,2), size(X,2))  # A*X
        R .-= temp2
        count_vec_add_flops(length(R))

        # Count norm calculation
        Rnorm = norm(R, 2)
        rel_Rnorm = Rnorm / norm(X, 2)

        count_norm_flops(length(R))

        output = @sprintf("iter=%6d  rel‖R‖=%11.3e  size(V,2)=%6d\n", iter, rel_Rnorm, size(V,2))
        print(output)

        if rel_Rnorm < thresh
            println("converged!")
            return (Σ, X)
        end

        # Preconditioning
        t = similar(R)
        for i = 1:size(t,2)
            C = 1.0 ./ (Σ[i] .- D)
            t[:,i] = C .* R[:,i]
            count_vec_add_flops(length(D))       # For Σ[i] .- D
            count_vec_scaling_flops(length(D))   # For the division
            count_vec_scaling_flops(length(D))   # For the multiplication
        end

        # Update V
        if size(V,2) <= Naux - Nlow
            V = hcat(V, t)
        else
            V = hcat(X, t)
        end
    end
end

function main(number::Integer, l::Integer, alpha::Integer, min_number_iter::Integer = 10)
    global NFLOPs
    NFLOPs = 0  # reset for each run

    filename = "CWNO_final_tilde.dat"

    A = load_matrix(filename)
    N = size(A, 1)

    Nlow = l
    Naux = Nlow * alpha

    if Naux > 0.35 * N
        println("Skipping: Naux ($Naux) is larger than 15% of the matrix size ($N).")
        return
    end

    # largest diagonal elements as initial guess  
    D = diag(A)
    idxs = sortperm(abs.(D), rev = true)[1:Nlow]
    V0_rr = A[:, idxs]
    
    # phi_occ = load_vector_from_file("occupied_orbital.dat")
    # norm_phi_occ = phi_occ' * phi_occ  # scalar normalization

    # for j in 1:size(V0_rr, 2)
    #     V0_rr[:, j] -= phi_occ * (phi_occ' * V0_rr[:, j]) / norm_phi_occ
    # end
    # V0_rr = randn(N, Nlow).- 0.5

    # davidson algorithm
    println("Davidson")
    @time Σ, U = davidson(A, V0_rr, Naux, 1e-5, 100, min_number_iter)

    Σ = abs.(Σ)  # Take absolute value of eigenvalues
    idx = sortperm(Σ, rev=true)
    Σ = Σ[idx]
    U = U[:, idx]
    
    println("Number of FLOPs: $NFLOPs")

    # Perform exact diagonalization as reference
    println("\nReading exact Eigenvalues...")
    Σexact = read_eigenresults(number)
    Σexact = abs.(Σexact)
    idx_exact = sortperm(Σexact, rev=true)
    Σexact = Σexact[idx_exact]

    # Display difference
    r = min(length(Σ), l)
    println("\nCompute the difference between computed and exact eigenvalues:")

    display("text/plain", (Σ[1:r] - Σexact[1:r])')
    # difference = (Σ[1:r] .- Σexact[1:r])
    # for i in 1:r
    #     println(@sprintf("%3d: %.10f (computed) - %.10f (exact) = % .4e", i, Σ[i], Σexact[i], difference[i]))
    # end
    println("$r Eigenvalues converges, out of $l requested.")
end

alpha = [4]
numbers = 1
ls = [10, 50, 100, 200]
for number in numbers
    println("Processing molecule: $number")
    for a in alpha
        println("Running with alpha = $a")
        for l in ls
            println("Running with l = $l")
            main(number, l * 10, a)
        end
    end
    println("Finished processing molecule: $number")
end
