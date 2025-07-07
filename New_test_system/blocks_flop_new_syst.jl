using LinearAlgebra
using Printf

# === Global FLOP counter and helpers ===
global NFLOPs = 0

include("../FLOP_count.jl")

function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Nauxiliary::Integer,
    thresh::Float64,
    target_nev::Int,
    deflation_eps::Float64
)::Tuple{Vector{T},Matrix{T}} where T<:Number

    global NFLOPs

    Nlow = size(V,2)
    Ritz_vecs = []
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

function define_matrix(system::String)
    Nlow = 25
    Naux = Nlow * 12
    N = 27643

    filename = "formaldehyde/gamma_VASP_" * system * ".dat"
    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    A = -A
    A = Hermitian(A)
    return A, N, Nlow, Naux
end

function main(system::String, target_nev::Int)
    global NFLOPs
    NFLOPs = 0

    A, N, Nlow, Naux = define_matrix(system)

    V = zeros(N, Nlow)
    for i = 1:Nlow
        V[i, i] = 1.0
    end

    println("Davidson")
    Naux = 12 * Nlow
    @time Σ, U = davidson(A, V, Naux, 1e-2, target_nev, 1e-2)
    idx = sortperm(Σ)
    Σ = Σ[idx]

    # println("Full diagonalization")
    # @time Σexact, Uexact = eigen(A)
    # count_diag_flops(N)

    # display("text/plain", (Σ - Σexact[1:length(Σ)])')
    println("Total estimated FLOPs: $(NFLOPs)")
end

target_nevs = [60]

for target_nev in target_nevs
    println("Running for target_nev = ", target_nev)
    main("HFbasis", target_nev)
    println("\n")
end