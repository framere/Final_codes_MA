using LinearAlgebra
using Printf
using Random
using JLD2

# === Global FLOP counter and helpers ===
global NFLOPs = 0

include("../FLOP_count.jl")

function select_corrections_ORTHO(t_candidates, V, V_lock, η, droptol; maxorth=2)
    ν = size(t_candidates, 2)
    n_b = 0
    T_hat = Matrix{eltype(t_candidates)}(undef, size(t_candidates, 1), ν)

    for i in 1:ν
        t_i = t_candidates[:, i]
        old_norm = norm(t_i)
        count_norm_flops(length(t_i))
        k = 0

        while k < maxorth
            k += 1

            # Count orthogonalization against V
            count_orthogonalization_flops(1, size(V,2), size(V,1))
            for j in 1:size(V, 2)
                t_i -= V[:, j] * (V[:, j]' * t_i)
            end

            new_norm = norm(t_i)
            count_norm_flops(length(t_i))
            if new_norm > η * old_norm
                break
            end
            old_norm = new_norm
        end

        if norm(t_i) > droptol
            count_norm_flops(length(t_i))
            n_b += 1
            T_hat[:, n_b] = t_i / norm(t_i)
            count_vec_scaling_flops(length(t_i))
        end
    end

    return T_hat[:, 1:n_b], n_b
end

function load_matrix(filename::String)
    N = 27643
    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    # A = -A
    return Hermitian(A)
end

function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    n_aux::Integer,
    l::Integer,
    thresh::Float64,
    system::String = "",
    stable_thresh::Integer = 3
)::Tuple{Vector{T}, Matrix{T}} where T<:Number

    global NFLOPs

    n_b = size(V, 2)
    nu_0 = max(l, n_b)
    nevf = 0

    D = diag(A)
    Eigenvalues = Float64[]
    Ritz_vecs = Matrix{T}(undef, size(A, 1), 0)
    V_lock = Matrix{T}(undef, size(A, 1), 0)

    iter = 0
    convergence_tracker = Dict{Int, Tuple{Float64, Int, Float64, Vector{T}}}()

    while nevf < l
        iter += 1

        if size(V_lock, 2) > 0
            count_orthogonalization_flops(size(V,2), size(V_lock,2), size(V,1))
            for i in 1:size(V_lock, 2)
                v_lock = V_lock[:, i]
                for j in 1:size(V, 2)
                    V[:, j] -= v_lock * (v_lock' * V[:, j])
                end
            end
        end
        
        count_qr_flops(size(V,1), size(V,2))
        V = Matrix(qr(V).Q)

        # Rayleigh-Ritz
        AV = A * V
        count_matmul_flops(size(A, 1), size(A, 2), size(V, 2))
        H = Hermitian(V' * AV)
        count_matmul_flops(size(V, 2), size(V, 1), size(AV, 2))

        nu = min(size(H, 2), nu_0 - nevf)
        Σ, U = eigen(H, 1:nu)
        count_diag_flops(size(H, 1))

        X = V * U
        count_matmul_flops(size(V, 1), size(V, 2), size(U, 2))

        R = X .* Σ' .- A * X
        count_matmul_flops(size(A, 1), size(A, 2), size(X, 2))
        count_vec_add_flops(length(R))  # For the subtraction

        norms = vec(norm.(eachcol(R)))
        for _ in eachcol(R)
            count_norm_flops(size(R,1))
        end

        conv_indices = Int[]
        for i = 1:size(R, 2)
            λ = Σ[i]
            rnorm = norms[i]

            if haskey(convergence_tracker, i)
                λ_prev, count, _, _ = convergence_tracker[i]
                if abs(λ - λ_prev) < 1e-6 && rnorm < thresh
                    convergence_tracker[i] = (λ, count + 1, rnorm, X[:, i])
                else
                    convergence_tracker[i] = (λ, 1, rnorm, X[:, i])
                end
            elseif rnorm < thresh
                convergence_tracker[i] = (λ, 1, rnorm, X[:, i])
            end

            if haskey(convergence_tracker, i)
                λ, count, rnorm, xvec = convergence_tracker[i]
                if count >= stable_thresh
                    push!(conv_indices, i)
                    push!(Eigenvalues, λ)
                    Ritz_vecs = hcat(Ritz_vecs, xvec)
                    V_lock = hcat(V_lock, xvec)
                    delete!(convergence_tracker, i)
                    nevf += 1
                    println(@sprintf("EV %3d converged λ = %.10f, ‖r‖ = %.2e, stable for %d iters", nevf, λ, rnorm, count))
                    if nevf >= l
                        println("Converged all eigenvalues.")
                        return (Eigenvalues, Ritz_vecs)
                    end
                end
            end
        end

        non_conv_indices = setdiff(1:size(R, 2), conv_indices)
        X_nc = X[:, non_conv_indices]
        Σ_nc = Σ[non_conv_indices]
        R_nc = R[:, non_conv_indices]

        t = Matrix{T}(undef, size(A, 1), length(non_conv_indices))
        ϵ = 1e-6
        for (i, idx) in enumerate(non_conv_indices)
            denom = clamp.(Σ_nc[i] .- D, ϵ, Inf)
            t[:, i] = R_nc[:, i] ./ denom
            count_vec_add_flops(length(D))
            count_vec_scaling_flops(length(D))
        end

        T_hat, n_b_hat = select_corrections_ORTHO(t, V, V_lock, 0.1, 1e-10)

        if size(V, 2) + n_b_hat > n_aux || length(conv_indices) > 0 || n_b_hat == 0
            # Restart with Ritz vectors + corrections, but enforce n_aux limit
            max_new_vectors = n_aux - size(X_nc, 2)  # Space left after keeping X_nc
            T_hat = T_hat[:, 1:min(n_b_hat, max_new_vectors)]  # Truncate if needed
            V = hcat(X_nc, T_hat)
            n_b = size(V, 2)
        else
            V = hcat(V, T_hat)
            n_b += n_b_hat
        end

        println("Iter $iter: V_size = $n_b, Converged = $nevf, Min ‖r‖ = $(minimum(norms))")
    end

    return (Eigenvalues, Ritz_vecs)
end

function read_eigenresults(system::String)
    output_file = system
    println("Reading eigenvalues from $output_file")
    data = jldopen(output_file, "r")
    eigenvalues = data["values"]
    close(data)
    return sort(eigenvalues; rev=true)
end


function main(system::String, l::Integer, factor::Integer)
    global NFLOPs
    NFLOPs = 0  # reset for each run

    filename = "../H2_molecule/gamma_VASP_" * system * ".dat"

    Nlow = 25
    Naux = Nlow * 12 * factor

    A = load_matrix(filename)
    N = size(A, 1)
    
    V = zeros(N, Nlow)
    for i = 1:Nlow
        V[i, i] = 1.0
    end

    println("Davidson")
    @time Σ, U = davidson(A, V, Naux, l, 1.5e-2, system, 3)

    idx = sortperm(Σ)
    Σ = Σ[idx]
    U = U[:, idx]

    println("Total estimated FLOPs: $(NFLOPs)")

    # Perform exact diagonalization as reference
    # println("\nPerforming full diagonalization for reference...")
    # Σexact,_ = read_eigenresults("../../MA_best/Eigenvalues_folder/eigen_results_$system.jld2")
    # Σexact = read_eigenresults("../Eigenvalues_folder/eigen_results_$system.jld2")
    

    # Display difference
    # println("\nDifference between Davidson and exact eigenvalues:")
    # display("text/plain", (Σ + Σexact[1:l])')
end

systems = ["HFbasis", "RNDbasis1"] # , "RNDbasis2", "RNDbasis3"
ls = [60, 90] # , 120, 160, 200, 340
for system in systems
    println("Running for system: $system")
    for (i, l) in enumerate(ls)
        println("Nlow = $l, factor = $i")
        main(system, l, i)  # Replace F with loop index i
    end
end
