using LinearAlgebra
using Printf
using JLD2

# === Global FLOP counter and helpers ===
global NFLOPs = 0

include("../FLOP_count.jl")


function occupied_orbitals(molecule::String)
    if molecule == "He"
        return 1
    elseif molecule == "hBN"
        return 36
    elseif molecule == "Si"
        return 72
    else
        error("Unknown molecule: $molecule")
    end
end

function load_matrix(filename::String, molecule::String)
    if molecule == "He"
        N = 4488
    elseif molecule == "hBN"
        N = 6863
    elseif molecule == "Si"
        N = 6201
    else
        error("Unknown molecule: $molecule")
    end
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    A = -A
    return Hermitian(A)
end

function read_eigenresults(molecule::String)
    output_file = "../Eigenvalues_folder/eigen_results_" * molecule * ".jld2"
    println("Reading eigenvalues from $output_file")
    data = jldopen(output_file, "r")
    eigenvalues = data["Σexact"]
    close(data)
    return sort(eigenvalues)
end

function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Integer,
    thresh::Float64,
    max_iter::Integer
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

function main(molecule::String, l::Integer, alpha::Integer)
    global NFLOPs
    NFLOPs = 0  # reset for each run

    filename = "../../Master_arbeit/Davidson_algorithm/m_pp_" * molecule * ".dat"
    
    A = load_matrix(filename,molecule)
    N = size(A, 1)

    Nlow = l
    Naux = Nlow * alpha

    #if Naux > 0.15 * N
    #    println("Skipping: Naux ($Naux) is larger than 15% of the matrix size ($N).")
    #    return
    #end

    # initial guess (naiv)
    V = zeros(N, Nlow)
    for i = 1:Nlow
       V[i,i] = 1.0
    end

    println("Davidson")
    @time Σ, U = davidson(A, V, Naux, 9e-5, 5000)
    idx = sortperm(Σ)
    Σ = Σ[idx]
    U = U[:, idx]

    println("Total estimated FLOPs: $(NFLOPs)")

    # Perform exact diagonalization as reference
    println("\nReading exact Eigenvalues...")
    Σexact = read_eigenresults(molecule)
    # println("Exact Eigenvalues: ", Σexact[1:l])

    # Display difference
    # println("\nDifference between Davidson and exact eigenvalues:")
    # display("text/plain", (Σ[1:l] - Σexact[1:l])')
    println("\nDifference between Davidson and exact eigenvalues:")
    rel_dev = (Σ[1:l] .- Σexact[1:l])
    display("text/plain", rel_dev')
end

alpha = [4,8,10]

molecules = ["He", "hBN", "Si"]

ls_hBN = [5, 10, 50, 100]
ls_Si = [5, 10, 20, 50, 70]
ls_He = [5, 10, 30, 50, 100, 200]
for molecule in molecules
    println("Processing molecule: $molecule")
    for a in alpha
        println("Running with alpha = $a")
        if molecule == "He"
            ls = ls_He
        elseif molecule == "hBN"
            ls = ls_hBN
        elseif molecule == "Si"
            ls = ls_Si
        else
            error("Unknown molecule: $molecule")
        end
        for l in ls
            println("Running with l = $l")
            main(molecule, l * occupied_orbitals(molecule), a)
        end
    end
    println("Finished processing molecule: $molecule")
end
