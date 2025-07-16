using LinearAlgebra
using Printf
using JLD2

# === Global FLOP counter and helpers ===
global NFLOPs = 0

include("../FLOP_count.jl")

function occupied_orbitals(molecule::String)
    if molecule == "H2"
        return 1
    elseif molecule == "formaldehyde"
        return 6
    else
        error("Unknown molecule: $molecule")
    end
end

function load_matrix(filename::String, molecule::String)
    if molecule == "H2"
        N = 11994
    elseif molecule == "formaldehyde"
        N = 27643
    else
        error("Unknown molecule: $molecule")
    end
    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    return Hermitian(A)
end

function read_eigenresults(molecule::String)
    output_file = "../Eigenvalues_folder/eigenres_" * molecule * "_RNDbasis1.jld2"
    println("Reading eigenvalues from $output_file")
    data = jldopen(output_file, "r")
    eigenvalues = data["eigenvalues"]
    close(data)
    return sort(eigenvalues)
end

function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Nauxiliary::Integer,
    thresh::Float64,
    target_nev::Int,
    deflation_eps::Float64, 
    max_iter::Integer
)::Tuple{Vector{T},Matrix{T}} where T<:Number

    global NFLOPs

    Nlow = size(V,2)
    Ritz_vecs = Vector{Vector{Float64}}()
    Eigenvalues = Float64[]
    Xconv = Matrix{T}(undef, size(A,1), 0)

    block = 0
    iter = 0
    n_converged = 0
    while length(Eigenvalues) < target_nev
        block += 1
        println("Block ", block)
        D = diag(A)
        Naux = copy(Nauxiliary)
        println("Initial size of V for block ", block, " is ", size(V, 2))
        println("Number of eigenvalues to find in this block: ", Nlow)
        println("Number of auxiliary vectors: ", Naux)
        
        while true
            iter += 1
            
            # Count QR factorization
            count_qr_flops(size(V,1), size(V,2))
            qr_decomp = qr(V)
            V = Matrix(qr_decomp.Q)


            if iter > max_iter
                println("Max iterations ($max_iter) reached without convergence. Returning what converged so far.")
                return (Eigenvalues, length(Ritz_vecs) > 0 ? hcat(Ritz_vecs...) : Matrix{T}(undef, size(A,1), 0))
            end 

            if size(Xconv, 2) > 0
                # Count orthogonalization against Xconv
                temp = Xconv' * V
                count_matmul_flops(size(Xconv,2), size(Xconv,1), size(V,2))
                V = V - Xconv * temp
                count_matmul_flops(size(Xconv,1), size(Xconv,2), size(temp,2))
                
                # Count second QR
                count_qr_flops(size(V,1), size(V,2))
                V = Matrix(qr(V).Q)
            end

            # Rayleigh-Ritz procedure
            temp_AV = A * V
            count_matmul_flops(size(A,1), size(A,2), size(V,2))
            H = V' * temp_AV
            count_matmul_flops(size(V,2), size(V,1), size(temp_AV,2))

            H = Hermitian(H)
            Σ, U = eigen(H, 1:Nlow)
            count_diag_flops(size(H,1))

            X = V * U
            count_matmul_flops(size(V,1), size(V,2), size(U,2))

            # Residual calculation
            temp_AX = A * X
            count_matmul_flops(size(A,1), size(A,2), size(X,2))
            R = X .* Σ' .- temp_AX
            count_vec_add_flops(length(R))

            # Count norm calculation
            Rnorm = norm(R, 2)
            count_norm_flops(length(R))
            
            output = @sprintf("iter=%6d  Rnorm=%11.3e  size(V,2)=%6d\n", iter, Rnorm, size(V,2))
            print(output)

            if Rnorm < thresh
                if size(Xconv, 2) > 0
                    proj_norm = norm(Xconv' * X, 2)
                    count_matmul_flops(size(Xconv,2), size(Xconv,1), size(X,2))
                else
                    proj_norm = 0.0
                end

                if proj_norm < 1e-1
                    println("converged block ", block, " with Rnorm ", Rnorm)
                    for i = 1:Nlow
                        if abs.(Σ[i] - Σ[end]) .> deflation_eps * abs(Σ[end])
                            push!(Ritz_vecs, X[:, i])
                            push!(Eigenvalues, Σ[i])
                            n_converged += 1
                            @printf("Converged eigenvalue %.10f with norm %.2e (EV %d)\n", Σ[i], norm(R[:, i]), n_converged)
                            
                            # Count orthogonalization of converged vector
                            q = X[:, i]
                            if size(Xconv, 2) > 0
                                temp_q = Xconv' * q
                                count_matmul_flops(size(Xconv,2), size(Xconv,1), 1)
                                q -= Xconv * temp_q
                                count_matmul_flops(size(Xconv,1), size(Xconv,2), 1)
                            end
                            q /= norm(q)
                            count_norm_flops(length(q))
                            count_vec_scaling_flops(length(q))
                            Xconv = hcat(Xconv, q)
                        else
                            @printf("Deflation eigenvalue %.3f: cutting through degenerate eigenvalues\n", Σ[i])
                        end
                    end
                end
                break
            end
            
            # Preconditioning step
            t = zero(similar(R))
            for i = 1:size(t,2)
                C = 1.0 ./ (Σ[i] .- D)
                t[:, i] = C .* R[:, i]
                count_vec_add_flops(length(D))       # For Σ[i] .- D
                count_vec_scaling_flops(length(D))    # For the division
                count_vec_scaling_flops(length(D))    # For the multiplication
            end

            # Update search space
            if size(V,2) <= Naux - Nlow
                V = hcat(V, t)
            else
                V = hcat(X, t)
            end
        end
    end

    return (Eigenvalues, hcat(Ritz_vecs...))
end

function main(molecule::String, target_nev::Int, max_iter::Int, alpha::Int = 12)
    global NFLOPs
    NFLOPs = 0

    filename = "../" * molecule *"/gamma_VASP_RNDbasis1.dat"
    
    A = load_matrix(filename,molecule)
    N = size(A, 1)
    Nlow = 10 + 2
    Naux = Nlow * alpha

    if Naux > 0.35 * N
        println("Skipping: Naux ($Naux) is larger than 15% of the matrix size ($N).")
        return
    end

    # initial guess (naiv)
    V = zeros(N, Nlow)
    for i = 1:Nlow
       V[i,i] = 1.0
    end

    println("Davidson")
    @time Σ, U = davidson(A, V, Naux, 1e-3, target_nev, 1e-2, max_iter)

    idx = sortperm(Σ)
    Σ = Σ[idx]
    U = U[:, idx]
    Σ = sqrt.(abs.(Σ))  # Take square root of eigenvalues   
    
    # Perform exact diagonalization as reference
    println("\nReading exact Eigenvalues...")
    Σexact_squared = read_eigenresults(molecule)

    idx_exact = sortperm(Σexact_squared)
    Σexact_squared = Σexact_squared[idx_exact]
    Σexact = sqrt.(abs.(Σexact_squared))  # Take square root of exact eigenvalues

    # Display difference
    r = min(length(Σ), l)
    println("\nCompute the difference between computed and exact eigenvalues:")
    difference = (Σ[1:r] .- Σexact[1:r])
    for i in 1:r
        println(@sprintf("%3d: %.10f (computed) - %.10f (exact) = % .4e", i, Σ[i], Σexact[i], difference[i]))
    end

    println("$r Eigenvalues converges, out of $l requested.")
end

systems = ["H2"]
targets = [10, 50, 100, 200]  # Example values for N 10, 50, 100, 200
alphas = [6, 8, 10]  # Example values for alpha

for system in systems
    for target in targets
        for alpha in alphas
            println("Running for nev target: $target, alpha: $alpha")
            main(system, target, 8000, alpha)
        end
    end
end

println("All calculations completed.")

