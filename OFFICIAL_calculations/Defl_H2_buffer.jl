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
    n_aux::Integer,
    l::Integer,
    thresh::Float64,
    max_iter::Integer,
    stable_thresh::Integer = 3
)::Tuple{Vector{T}, Matrix{T}} where T<:Number


    global NFLOPs

    n_b = size(V, 2)
    l_buffer = l * 1.85
    l_buffer = Integer(round(l_buffer))
    nu_0 = max(l_buffer, n_b)
    nevf = 0

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
            λ_est = sqrt(abs(λ))
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

        T_hat, n_b_hat = select_corrections_ORTHO(t, V, V_lock, 0.1, 1e-8)

        if size(V, 2) + n_b_hat > n_aux  #|| n_b_hat == 0 || length(conv_indices) > 0
            # Restart with Ritz vectors + corrections, but enforce n_aux limit
            max_new_vectors = n_aux - size(X_nc, 2)  # Space left after keeping X_nc
            T_hat = T_hat[:, 1:min(n_b_hat, max_new_vectors)]  # Truncate if needed
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


function main(molecule::String, l::Integer, beta::Integer, max_iter::Integer)
    global NFLOPs
    NFLOPs = 0  # reset for each run

    filename = "../" * molecule *"/gamma_VASP_RNDbasis1.dat"

    Nlow = max(round(Int, 0.1*l), 16)
    Naux = Nlow * beta

    A = load_matrix(filename,molecule)
    N = size(A, 1)
    
    V = zeros(N, Nlow)
    for i = 1:Nlow
        V[i, i] = 1.0
    end

    println("Davidson")
    @time Σ, U = davidson(A, V, Naux, l, 1e-4, max_iter)

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




betas = [8,16,32,64]
molecules = ["H2"]
ls = [10, 50, 100, 200]
for molecule in molecules
    println("Processing molecule: $molecule")
    for beta in betas
        println("Running with beta = $beta")
        for l in ls
            println("Running with l = $l")
            main(molecule, l*occupied_orbitals(molecule), beta, 5000)
        end
    end
    println("Finished processing molecule: $molecule")
end

