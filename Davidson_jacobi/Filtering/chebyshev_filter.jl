using LinearAlgebra
using Printf
using JLD2
using IterativeSolvers
using LinearMaps


# === Global FLOP counter and helpers ===
global NFLOPs = 0

include("../../FLOP_count.jl")

function correction_equations_minres(A, U, lambdas, R; tol=1e-1, maxiter=100)
    global NFLOPs
    n, k = size(U)
    S = zeros(eltype(A), n, k)
    total_iter = 0

    for j in 1:k
        λ, r = lambdas[j], R[:, j]

        M_apply = function(x)
            x_perp = x - (U * (U' * x)); 
            count_matmul_flops(k,1,n); count_matmul_flops(n,1,k); count_vec_add_flops(n)

            tmp = (A * x_perp) - λ * x_perp; 
            count_matmul_flops(n,1,n); count_vec_scaling_flops(n); count_vec_add_flops(n)

            res = tmp - (U * (U' * tmp)); 
            count_matmul_flops(k,1,n); count_matmul_flops(n,1,k); count_vec_add_flops(n)
            return res
        end

        M_op = LinearMap{eltype(A)}(M_apply, n, n; ishermitian=true)

        rhs = r - (U * (U' * r)); 
        count_matmul_flops(k,1,n); count_matmul_flops(n,1,k); count_vec_add_flops(n)
        rhs = -rhs; count_vec_scaling_flops(n)

        s_j, msg = minres(M_op, rhs; reltol=tol, maxiter=maxiter, log=true)
        m = match(r"(\d+)\s+iterations", string(msg))
        if m !== nothing
            niter = parse(Int, m.captures[1])
            total_iter += niter
            # println("Number of iterations: ", niter)
        else
            println("No iteration number found in message: ", msg)
        end

        s_j = s_j - (U * (U' * s_j)); 
        count_matmul_flops(k,1,n); count_matmul_flops(n,1,k); count_vec_add_flops(n)

        S[:, j] = s_j
    end
    println("Total MINRES iterations: ", total_iter)
    NFLOPs += total_iter * (2*n^2 + 4*n*k)  
    return S
end


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
    elseif molecule == "uracil"
        return 21
    else
        error("Unknown molecule: $molecule")
    end
end

function load_matrix(filename::String, molecule::String)
    if molecule == "H2"
        N = 11994
    elseif molecule == "formaldehyde"
        N = 27643
    elseif molecule == "uracil"
        N = 32416
    else
        error("Unknown molecule: $molecule")
    end
    # println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    return Hermitian(A)
end


function read_eigenresults(molecule::String)
    output_file = "../../Eigenvalues_folder/eigenres_" * molecule * "_RNDbasis1.jld2"
    println("Reading eigenvalues from $output_file")
    data = jldopen(output_file, "r")
    eigenvalues = data["eigenvalues"]
    close(data)
    return sort(eigenvalues)
end


function chebyshev_filter(
    A::AbstractMatrix{T},
    V::Matrix{T},
    a::Real,
    b::Real,
    degree::Int
)::Matrix{T} where T<:Number
    global NFLOPs
    
    # Map interval [a,b] to [-1,1] for Chebyshev polynomials
    # We want to damp eigenvalues near b and amplify those near a
    e = (b - a) / 2
    c = (b + a) / 2
    σ = e / (a - c)  # Damping factor
    σ_prime = σ
    
    # Initialize: Y₀ = V, Y₁ = (A*V - c*V) / e
    Y_prev = copy(V)
    AV = A * V
    count_matmul_flops(size(A, 1), size(A, 2), size(V, 2))
    
    Y_curr = (AV .- c .* V) ./ e
    count_vec_add_flops(length(V))
    count_vec_scaling_flops(length(V))
    
    # Apply Chebyshev recursion: Y_{k+1} = 2(A - cI)/e * Y_k - Y_{k-1}
    for k in 2:degree
        AY = A * Y_curr
        count_matmul_flops(size(A, 1), size(A, 2), size(Y_curr, 2))
        
        Y_next = (2 .* (AY .- c .* Y_curr) ./ e) .- Y_prev
        count_vec_add_flops(2 * length(Y_curr))
        count_vec_scaling_flops(length(Y_curr))
        
        Y_prev = Y_curr
        Y_curr = Y_next
    end
    
    # Final filtered vectors
    return Y_curr
end

"""
    estimate_spectral_bounds(A, D, Σ_converged, lc)

Estimate tight spectral bounds for Chebyshev filtering.

# Returns
- `a`: Upper bound for wanted eigenvalues (just above lc-th eigenvalue)
- `b`: Upper bound of spectrum
"""
function estimate_spectral_bounds(
    A::AbstractMatrix{T},
    D::Vector{T},
    Σ_converged::Vector{Float64},
    lc::Int
)::Tuple{Float64, Float64} where T<:Number
    
    # Lower bound a: use last converged eigenvalue + safety margin
    if !isempty(Σ_converged)
        a = maximum(Σ_converged) * 1.15  # 15% safety margin
    else
        # Estimate from diagonal (rough approximation)
        D_sorted = sort(real.(D))
        a = length(D_sorted) >= lc ? D_sorted[min(lc + 5, length(D_sorted))] : D_sorted[end]
    end
    
    # Upper bound b: use Gershgorin circle theorem or largest diagonal
    # For more accuracy, could use power iteration
    b = maximum(real.(D)) * 1.2  # Conservative estimate
    
    return (Float64(a), Float64(b))
end

function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    n_aux::Integer,
    l::Integer,
    thresh::Float64,
    max_iter::Integer,
    stable_thresh::Integer = 3;
    use_chebyshev::Bool = true,
    cheb_degree::Int = 10,
    cheb_restart_freq::Int = 5
)::Tuple{Vector{T}, Matrix{T}} where T<:Number
    global NFLOPs
    
    n_b = size(V, 2)
    l_buffer = round(Int, l * 1.8)
    lc = round(Int, 1.01 * l)
    nu_0 = max(l_buffer, n_b)
    nevf = 0
    
    println("Starting Davidson with n_aux = $n_aux, l_buffer = $l_buffer, lc = $lc, thresh = $thresh")
    println("Chebyshev filtering: $(use_chebyshev ? "enabled (degree=$cheb_degree, restart_freq=$cheb_restart_freq)" : "disabled")")
    
    D = diag(A)
    Eigenvalues = Float64[]
    Ritz_vecs = Matrix{T}(undef, size(A, 1), 0)
    V_lock = Matrix{T}(undef, size(A, 1), 0)
    iter = 0
    convergence_tracker = Dict{Int, Tuple{Float64, Int, Float64, Vector{T}}}()
    residual_history = Dict{Int, Vector{Float64}}()
    
    function is_stagnating(hist::Vector{Float64}; tol=0.1, window=2)
        length(hist) < window && return false
        r_old, r_new = hist[end-window+1], hist[end]
        return abs(r_old - r_new) / r_old < tol
    end
    
    while nevf < l_buffer
        iter += 1
        if iter > max_iter
            println("Max iterations ($max_iter) reached without convergence.")
            return (Eigenvalues, Ritz_vecs)
        end
        
        # Apply Chebyshev filtering at restart points
        if use_chebyshev && iter > 1 && (iter - 1) % cheb_restart_freq == 0 && nevf < lc
            println("  → Applying Chebyshev filter at iteration $iter")
            a, b = estimate_spectral_bounds(A, D, Eigenvalues, lc)
            println("    Spectral bounds: a = $a, b = $b")
            
            # Filter the current search space
            V = chebyshev_filter(A, V, a, b, cheb_degree)
            
            # Re-orthonormalize after filtering
            count_qr_flops(size(V, 1), size(V, 2))
            V = Matrix(qr(V).Q)
        end
        
        # Orthogonalize against locked vectors
        if size(V_lock, 2) > 0
            count_orthogonalization_flops(size(V, 2), size(V_lock, 2), size(V, 1))
            for i in 1:size(V_lock, 2)
                v_lock = V_lock[:, i]
                for j in 1:size(V, 2)
                    V[:, j] -= v_lock * (v_lock' * V[:, j])
                end
            end
        end
        
        # Orthonormalize the basis
        count_qr_flops(size(V, 1), size(V, 2))
        V = Matrix(qr(V).Q)
        
        # Rayleigh-Ritz projection
        AV = A * V
        count_matmul_flops(size(A, 1), size(A, 2), size(V, 2))
        H = Hermitian(V' * AV)
        count_matmul_flops(size(V, 2), size(V, 1), size(AV, 2))
        
        # Compute approximate eigenvalues
        nu = min(size(H, 2), nu_0 - nevf)
        Σ, U = eigen(H, 1:nu)
        count_diag_flops(size(H, 1))
        
        # Compute Ritz vectors
        X = V * U
        count_matmul_flops(size(V, 1), size(V, 2), size(U, 2))
        
        # Compute residuals
        R = X .* Σ' .- A * X
        count_matmul_flops(size(A, 1), size(A, 2), size(X, 2))
        count_vec_add_flops(length(R))
        
        # Compute residual norms
        norms = vec(norm.(eachcol(R)))
        for _ in eachcol(R)
            count_norm_flops(size(R, 1))
        end
        
        # Sort eigenvalues
        sorted_indices = sortperm(Σ)
        Σ_sorted = Σ[sorted_indices]
        X_sorted = X[:, sorted_indices]
        norms_sorted = norms[sorted_indices]
        
        # Check convergence
        current_cutoff = min(lc - nevf, length(Σ_sorted))
        conv_indices = Int[]
        
        for (sorted_i, original_i) in enumerate(sorted_indices[1:current_cutoff])
            λ = Σ_sorted[sorted_i]
            λ_est = sqrt(abs(λ))
            adaptive_thresh = 2 * λ_est * thresh
            rnorm = norms_sorted[sorted_i]
            
            # Update convergence tracking
            if haskey(convergence_tracker, original_i)
                λ_prev, count, _, _ = convergence_tracker[original_i]
                if abs(λ - λ_prev) < 1e-5 && rnorm < adaptive_thresh
                    convergence_tracker[original_i] = (λ, count + 1, rnorm, X_sorted[:, sorted_i])
                else
                    convergence_tracker[original_i] = (λ, 1, rnorm, X_sorted[:, sorted_i])
                end
            elseif rnorm < adaptive_thresh
                convergence_tracker[original_i] = (λ, 1, rnorm, X_sorted[:, sorted_i])
            end
            
            # Check if converged
            if haskey(convergence_tracker, original_i)
                λ, count, rnorm, xvec = convergence_tracker[original_i]
                if count >= stable_thresh
                    push!(conv_indices, original_i)
                    push!(Eigenvalues, λ)
                    Ritz_vecs = hcat(Ritz_vecs, xvec)
                    V_lock = hcat(V_lock, xvec)
                    delete!(convergence_tracker, original_i)
                    nevf += 1
                    if nevf >= lc
                        println("Converged all required eigenvalues.")
                        return (Eigenvalues, Ritz_vecs)
                    end
                end
            end
        end
        
        # Prepare for next iteration
        all_indices = 1:size(R, 2)
        non_conv_indices = setdiff(all_indices, conv_indices)
        non_conv_sorted = sort(non_conv_indices, by=i->Σ[i])
        keep_indices = non_conv_sorted[1:min(length(non_conv_sorted), l_buffer - nevf)]
        
        X_nc = X[:, keep_indices]
        Σ_nc = Σ[keep_indices]
        R_nc = R[:, keep_indices]
        
        # Update residual histories
        for idx in keep_indices
            push!(get!(residual_history, idx, Float64[]), norms[idx])
            if length(residual_history[idx]) > 3
                popfirst!(residual_history[idx])
            end
        end
        
        # Compute correction vectors
        ϵ = 1e-6
        if iter < 10
            t = Matrix{T}(undef, size(A, 1), length(keep_indices))
            for (i, idx) in enumerate(keep_indices)
                denom = clamp.(Σ_nc[i] .- D, ϵ, Inf)
                t[:, i] = R_nc[:, i] ./ denom
                count_vec_add_flops(length(D))
                count_vec_scaling_flops(length(D))
            end
        else
            println("Hybrid Davidson-JD corrections at iteration $iter")
            dav_indices = Int[]
            jd_indices = Int[]
            
            for (i, idx) in enumerate(keep_indices)
                if idx > current_cutoff
                    push!(dav_indices, i)
                else
                    r = norms[idx]
                    hist = get(residual_history, idx, Float64[])
                    if r >= 1e-2 || is_stagnating(hist; tol=0.1, window=2)
                        push!(jd_indices, i)
                    else
                        push!(dav_indices, i)
                    end
                end
            end
            
            t_dav = Matrix{T}(undef, size(A, 1), length(dav_indices))
            for (j, i) in enumerate(dav_indices)
                idx = keep_indices[i]
                denom = clamp.(Σ_nc[i] .- D, ϵ, Inf)
                t_dav[:, j] = R_nc[:, i] ./ denom
                count_vec_add_flops(length(D))
                count_vec_scaling_flops(length(D))
            end
            
            t_jd = correction_equations_minres(
                A, X_nc[:, jd_indices], Σ_nc[jd_indices], R_nc[:, jd_indices];
                tol=1e-1, maxiter=25
            )
            
            t = hcat(t_dav, t_jd)
        end
        
        # Orthogonalize and select correction vectors
        T_hat, n_b_hat = select_corrections_ORTHO(t, V, V_lock, 0.1, 1e-10)
        
        # Update search space
        if size(V, 2) + n_b_hat > n_aux || n_b_hat == 0
            max_new_vectors = n_aux - size(X_nc, 2)
            T_hat = T_hat[:, 1:min(n_b_hat, max_new_vectors)]
            V = hcat(X_nc, T_hat)
            n_b = size(V, 2)
        else
            V = hcat(V, T_hat)
            n_b += n_b_hat
        end
        
        # Print iteration info
        i_max = argmax(Σ)
        norm_largest_Ritz = norms[i_max]
        println("Iter $iter: V_size = $n_b, Converged = $nevf, ‖r‖ (largest λ) = $norm_largest_Ritz")
    end
    
    return (Eigenvalues, Ritz_vecs)
end


function main(molecule::String, l::Integer, beta::Integer, factor::Integer, max_iter::Integer)
    global NFLOPs
    NFLOPs = 0  # reset for each run

    filename = "../../" * molecule *"/gamma_VASP_RNDbasis1.dat"

    Nlow = max(round(Int, 0.1*l), 16)
    Naux = Nlow * beta
    A = load_matrix(filename,molecule)
    N = size(A, 1)

    V = zeros(N, Nlow)
    for i = 1:Nlow
        V[i, i] = 1.0
    end

    @time Σ, U = davidson(A, V, Naux, l, 1e-3 + 0.5e-3 * factor, max_iter)

    idx = sortperm(Σ)
    Σ = Σ[idx]
    U = U[:, idx]
    Σ = sqrt.(abs.(Σ))  # Take square root of eigenvalues   
    
    println("Number of FLOPs: $NFLOPs")

    # Perform exact diagonalization as reference
    println("\nReading exact Eigenvalues...")
    Σexact_squared = read_eigenresults(molecule)

    idx_exact = sortperm(Σexact_squared)
    Σexact_squared = Σexact_squared[idx_exact]
    Σexact = sqrt.(abs.(Σexact_squared))  # Take square root of exact eigenvalues

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



betas = [20] #8,16,32,64, 8,16
molecules = ["formaldehyde"] #, "uracil"
ls = [30] #10, 50, 100, 200
for molecule in molecules
    println("Processing molecule: $molecule")
    for beta in betas
        println("Running with beta = $beta")
        for (i, l) in enumerate(ls)
	    nev = l*occupied_orbitals(molecule)
            println("Running with l = $nev")
            main(molecule, nev, beta, i, 100)
        end
    end
    println("Finished processing molecule: $molecule")
end


