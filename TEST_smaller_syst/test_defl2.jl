using LinearAlgebra
using JLD2
using Printf

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
    n_aux::Integer,
    l::Integer,
    thresh::Float64,
    max_iter::Integer,
    stable_thresh::Integer = 3
)::Tuple{Vector{T}, Matrix{T}} where T<:Number

    global NFLOPs

    n_b = size(V, 2)
    l_buffer = l * 1.95
    l_buffer = Integer(round(l_buffer))
    nu_0 = max(l_buffer, n_b)
    nevf = 0

    println("Starting Davidson with n_aux = $n_aux, l_buffer = $l_buffer, thresh = $thresh, max_iter = $max_iter")

    D = diag(A)
    Eigenvalues = Float64[]
    Ritz_vecs = Matrix{T}(undef, size(A, 1), 0)
    V_lock = Matrix{T}(undef, size(A, 1), 0)

    iter = 0
    convergence_tracker = Dict{Int, Tuple{Float64, Int, Float64, Vector{T}}}()

    while nevf < l_buffer
        iter += 1

        if iter > max_iter
            println("Max iterations ($max_iter) reached without convergence. Skipping this case.")
            return (Eigenvalues, Ritz_vecs)
        end        

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
            λ_est = abs(λ)
            adaptive_thresh = 2 *λ_est* thresh
            rnorm = norms[i]

            if haskey(convergence_tracker, i)
                λ_prev, count, _, _ = convergence_tracker[i]
                if abs(λ - λ_prev) < 1e-6 && rnorm < adaptive_thresh
                    convergence_tracker[i] = (λ, count + 1, rnorm, X[:, i])
                else
                    convergence_tracker[i] = (λ, 1, rnorm, X[:, i])
                end
            elseif rnorm < adaptive_thresh
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
                end
            end
        end

        # === MODIFIED PART STARTS HERE ===
        if length(Eigenvalues) >= l
            Σ_sorted = sort(Eigenvalues)
            λ_lth = Σ_sorted[l]

            # Check if any current Ritz value is smaller than λ_lth
            if all(σ ≥ λ_lth for σ in Σ)
                println("Lowest $l eigenvalues are isolated and converged. Stopping.")
                idx = sortperm(Eigenvalues)[1:l]
                return (Σ_sorted[1:l], Ritz_vecs[:, idx])
            end
        end
        # === MODIFIED PART ENDS HERE ===

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

        if size(V, 2) + n_b_hat > n_aux || n_b_hat == 0
            max_new_vectors = n_aux - size(X_nc, 2)
            T_hat = T_hat[:, 1:min(n_b_hat, max_new_vectors)]
            V = hcat(X_nc, T_hat)
            n_b = size(V, 2)
        else
            V = hcat(V, T_hat)
            n_b += n_b_hat
        end
        
        i_max = argmax(Σ)
        norm_largest_Ritz = norms[i_max]
        println("Iter $iter: V_size = $n_b, Converged = $nevf, ‖r‖ (largest λ) = $norm_largest_Ritz")
    end

    return (Eigenvalues, Ritz_vecs)
end



function main(molecule::String, l::Integer, beta::Integer, factor::Integer, max_iter::Integer)
    global NFLOPs
    NFLOPs = 0  # reset for each run

    filename = "../../Master_arbeit/Davidson_algorithm/m_pp_" * molecule * ".dat"

    Nlow = max(round(Int, 0.1*l), 16)
    Naux = Nlow * beta
    A = load_matrix(filename, molecule)
    N = size(A, 1)

    V = zeros(N, Nlow)
    for i = 1:Nlow
        V[i, i] = 1.0
    end

    @time Σ, U = davidson(A, V, Naux, l, 1e-3 + 0.5e-3 * factor, max_iter)

    idx = sortperm(Σ)
    Σ = Σ[idx]
    U = U[:, idx]
    Σ = abs.(Σ)  # No sqrt

    println("Number of FLOPs: $NFLOPs")

    # Perform exact diagonalization as reference
    println("\nReading exact Eigenvalues...")
    Σexact_squared = read_eigenresults(molecule)

    idx_exact = sortperm(Σexact_squared)
    Σexact_squared = Σexact_squared[idx_exact]
    Σexact = abs.(Σexact_squared)  # No sqrt

    # Display difference
    r = min(length(Σ), l)
    println("\nCompute the difference between computed and exact eigenvalues:")
    difference = (Σ[1:r] .- Σexact[1:r])
    for i in 1:r
        println(@sprintf("%3d: %.10f (computed) - %.10f (exact) = % .4e", i, Σ[i], Σexact[i], difference[i]))
    end
    println("$r Eigenvalues converges, out of $l requested.")
end

# === MAIN LOOP ===
betas = [32, 64]
molecules = ["He", "hBN", "Si"]
ls_hBN = [5, 10, 50, 100]
ls_Si = [5, 10, 20, 50, 70]
ls_He = [5, 10, 30, 50, 100, 200]

for molecule in molecules
    println("Processing molecule: $molecule")
    for beta in betas
        println("Running with beta = $beta")
        if molecule == "hBN"
            ls = ls_hBN
        elseif molecule == "Si"
            ls = ls_Si
        elseif molecule == "He"
            ls = ls_He
        else
            error("Unknown molecule: $molecule")
        end
        for (i, l) in enumerate(ls)
            nev = l * occupied_orbitals(molecule)
            println("Running with l = $nev")
            main(molecule, nev, beta, i, 1000)
        end
    end
    println("Finished processing molecule: $molecule")
end
