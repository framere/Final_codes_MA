using LinearAlgebra
using Printf
using JLD2

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
            
            qr_decomp = qr(V)
            V = Matrix(qr_decomp.Q)

            if iter > max_iter
                println("Max iterations ($max_iter) reached without convergence. Returning what converged so far.")
                return (Eigenvalues, length(Ritz_vecs) > 0 ? hcat(Ritz_vecs...) : Matrix{T}(undef, size(A,1), 0))
            end 

            if size(Xconv, 2) > 0
                temp = Xconv' * V
                V = V - Xconv * temp
                V = Matrix(qr(V).Q)
            end

            # Rayleigh-Ritz procedure
            temp_AV = A * V
            H = V' * temp_AV
            H = Hermitian(H)
            Σ, U = eigen(H, 1:Nlow)
            X = V * U

            # Residual calculation
            temp_AX = A * X
            R = X .* Σ' .- temp_AX

            Rnorm = norm(R, 2)
            
            output = @sprintf("iter=%6d  Rnorm=%11.3e  size(V,2)=%6d\n", iter, Rnorm, size(V,2))
            print(output)

            if Rnorm < thresh
                if size(Xconv, 2) > 0
                    proj_norm = norm(Xconv' * X, 2)
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
                            
                            # Orthogonalization of converged vector
                            q = X[:, i]
                            if size(Xconv, 2) > 0
                                temp_q = Xconv' * q
                                q -= Xconv * temp_q
                            end
                            q /= norm(q)
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
    filename = "../" * molecule *"/gamma_VASP_RNDbasis1.dat"
    
    A = load_matrix(filename,molecule)
    N = size(A, 1)
    Nlow = 60 + 2
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
    @time Σ, U = davidson(A, V, Naux, 1e-4, target_nev, 1e-2, max_iter)
    idx = sortperm(Σ)
    Σ = Σ[idx]

    Σexact = read_eigenresults(molecule)

    r = length(Σ)
    println("\nDifference between Davidson and exact eigenvalues:")
    display("text/plain", (Σ - Σexact[1:r])')
end

systems = ["formaldehyde"]
targets = [60, 300, 600, 1200]  # Example values for N
alphas = [6, 8, 10]  # Example values for alpha

for system in systems
    for target in targets
        for alpha in alphas
            println("Running for nev target: $target, alpha: $alpha")
            main(system, target, 7000, alpha)
        end
    end
end

println("All calculations completed.")