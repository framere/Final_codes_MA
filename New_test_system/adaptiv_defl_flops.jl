using LinearAlgebra
using Printf
using JLD2

# Initialize global FLOP counter
global NFLOPs = 0

include("../FLOP_count.jl")

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


function select_corrections_ORTHO(t_candidates, V, V_lock, η, droptol; maxorth=2)
    ν = size(t_candidates, 2)
    n_b = 0
    T_hat = Matrix{eltype(t_candidates)}(undef, size(t_candidates, 1), ν)

    for i in 1:ν
        t_i = t_candidates[:, i]
        
        # Count initial norm
        count_norm_flops(length(t_i))
        old_norm = norm(t_i)
        k = 0

        while k < maxorth
            k += 1

            for j in 1:size(V, 2)
                # Count dot product and vector update
                count_dot_product_flops(length(t_i))
                count_vec_scaling_flops(length(t_i))
                count_vec_add_flops(length(t_i))
                t_i -= V[:, j] * (V[:, j]' * t_i)
            end

            # Count norm after orthogonalization
            count_norm_flops(length(t_i))
            new_norm = norm(t_i)
            
            if new_norm > η * old_norm
                break
            end
            old_norm = new_norm
        end

        # Count final norm check
        count_norm_flops(length(t_i))
        if norm(t_i) > droptol
            n_b += 1
            # Count normalization
            count_vec_scaling_flops(length(t_i))
            T_hat[:, n_b] = t_i / norm(t_i)
        end
    end

    return T_hat[:, 1:n_b], n_b
end

function davidson(A::AbstractMatrix{T},
    V::Matrix{T},
    n_aux::Integer,
    l::Integer,
    thresh::Float64,
    deg_thresh::Float64 = 1e-3,  # Threshold for considering eigenvalues degenerate
    system::String = ""
)::Tuple{Vector{T}, Matrix{T}} where T<:Number

    n_b = size(V, 2)
    nu_0 = max(l, n_b)
    nevf = 0

    D = diag(A)
    count_diag_flops(size(A, 1))  # Count FLOPs for diagonal extraction
    
    Eigenvalues = Float64[]
    Ritz_vecs = Matrix{T}(undef, size(A, 1), 0)
    V_lock = Matrix{T}(undef, size(A, 1), 0)

    iter = 0

    while nevf < l
        iter += 1

        # Orthogonalize V against locked vectors
        if size(V_lock, 2) > 0
            for i in 1:size(V_lock, 2)
                v_lock = V_lock[:, i]
                for j in 1:size(V, 2)
                    # Count dot product and vector update
                    count_dot_product_flops(size(v_lock, 1))
                    count_vec_scaling_flops(size(v_lock, 1))
                    count_vec_add_flops(size(V, 1))
                    V[:, j] -= v_lock * (v_lock' * V[:, j])
                end
            end
        end
        
        # Count QR factorization
        count_qr_flops(size(V, 1), size(V, 2))
        V = Matrix(qr(V).Q)

        # Rayleigh-Ritz
        # Count matrix multiplication FLOPs
        count_matmul_flops(size(V, 2), size(V, 2), size(A, 1))  # V' * (A*V)
        count_matmul_flops(size(A, 1), size(V, 2), size(A, 2))  # A * V
        H = Hermitian(V' * (A * V))
        
        nu = min(size(H, 2), nu_0 - nevf)
        Σ, U = eigen(H, 1:nu)
        count_diag_flops(size(H, 1))  # Count FLOPs for diagonalization
        
        # Count Ritz vector computation
        count_matmul_flops(size(V, 1), size(U, 2), size(V, 2))
        X = V * U  # Ritz vectors

        # Compute residuals
        # Count matrix-vector multiplications and vector operations
        count_matmul_flops(size(A, 1), size(X, 2), size(A, 2))  # A * X
        count_vec_scaling_flops(size(X, 1) * size(X, 2))  # X .* Σ'
        R = X .* Σ' - A * X
        count_vec_add_flops(size(R, 1) * size(R, 2))  # R = X.*Σ' - A*X
        
        # Count norm computations
        count_norm_flops(size(R, 1) * size(R, 2))
        norms = vec(norm.(eachcol(R)))

        # Find all indices that meet the convergence threshold
        potential_conv_indices = findall(norms .<= thresh)
        
        # Group degenerate eigenvalues
        degenerate_groups = Vector{Vector{Int}}()
        remaining_indices = collect(1:length(Σ))
        
        while !isempty(remaining_indices)
            current_idx = popfirst!(remaining_indices)
            group = [current_idx]
            
            # Find all eigenvalues close to the current one
            for (i, idx) in enumerate(remaining_indices)
                if abs(Σ[idx] - Σ[current_idx]) < deg_thresh
                    push!(group, idx)
                end
            end
            
            # Remove found indices from remaining
            filter!(x -> !(x in group[2:end]), remaining_indices)
            push!(degenerate_groups, group)
        end

        # Determine which vectors to lock
        conv_indices = Int[]
        for group in degenerate_groups
            # Check if all in group are converged
            all_converged = all(in(potential_conv_indices), group)
            
            if all_converged
                append!(conv_indices, group)
                append!(Eigenvalues, Σ[group])
                Ritz_vecs = hcat(Ritz_vecs, X[:, group])
                V_lock = hcat(V_lock, X[:, group])
                nevf += length(group)
                
                for idx in group
                    println(@sprintf("Converged eigenvalue %.10f with norm %.2e (EV %d)", Σ[idx], norms[idx], nevf - length(group) + findfirst(==(idx), group)))
                end
                
                if nevf >= l
                    println("Converged all eigenvalues.")
                    return (Eigenvalues, Ritz_vecs)
                end
            end
        end

        non_conv_indices = setdiff(1:size(R, 2), conv_indices)
        X_nc = X[:, non_conv_indices]
        Σ_nc = Σ[non_conv_indices]
        R_nc = R[:, non_conv_indices]

        # Correction vectors
        t = Matrix{T}(undef, size(A, 1), length(non_conv_indices))
        ϵ = 1e-6  # small value to avoid division by zero
        for (i, idx) in enumerate(non_conv_indices)
            # Count vector operations
            count_vec_add_flops(length(D))  # Σ_nc[i] .- D
            count_vec_scaling_flops(length(R_nc[:, i]))  # division
            denom = clamp.(Σ_nc[i] .- D, ϵ, Inf)
            t[:, i] = R_nc[:, i] ./ denom
        end

        # Orthogonalize corrections
        T_hat, n_b_hat = select_corrections_ORTHO(t, V, V_lock, 0.1, 1e-10)
        # Note: You should also add FLOP counting inside select_corrections_ORTHO function

        # Update subspace
        if size(V, 2) + n_b_hat > n_aux || length(conv_indices) > 0 || n_b_hat == 0 
            max_new_vectors = n_aux - size(X_nc, 2)  # Space left after keeping X_nc
            T_hat = T_hat[:, 1:min(n_b_hat, max_new_vectors)]  # Truncate if needed
            V = hcat(X_nc, T_hat)
            n_b = size(V, 2)
        else
            V = hcat(V, T_hat)
            n_b += n_b_hat
        end

        println("Iter $iter: V_size = $n_b, Converged = $nevf, Norm = $(minimum(norms))")
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
    # Reset FLOP counter at start of main
    global NFLOPs = 0
    
    # the two test systems He and hBN are hardcoded
    system = system
    filename = "../formaldehyde/gamma_VASP_" * system * ".dat"

    Nlow = 25
    Naux = Nlow * 12 * factor

    A = load_matrix(filename)
    N = size(A, 1)

    ## initial guess vectors (naive guess)
    V = zeros(N, Nlow)
    for i = 1:Nlow
       V[i,i] = 1.0
    end

    # perform Davidson algorithm
    println("Davidson")
    @time Σ, U = davidson(A, V, Naux, l, 1e-2, 7e-2, system)

    # Print total FLOP count at end
    println("\nTotal FLOP count: $NFLOPs")

    # sort
    idx = sortperm(Σ)
    Σ = Σ[idx] # sort eigenvalues
    U = U[:,idx] # sort the converged eigenvectors

    # Perform exact diagonalization as reference
    println("\nPerforming full diagonalization for reference...")
    # Σexact,_ = read_eigenresults("../../MA_best/Eigenvalues_folder/eigen_results_$system.jld2")
    Σexact = read_eigenresults("../Eigenvalues_folder/eigen_results_$system.jld2")
    

    # Display difference
    println("\nDifference between Davidson and exact eigenvalues:")
    display("text/plain", (Σ + Σexact[1:l])')
end

systems = ["RNDbasis1"] # "HFbasis", , "RNDbasis2", "RNDbasis3"
ls = [60, 90] #, 120, 160, 200
for system in systems
    println("Running for system: $system")
    for (i, l) in enumerate(ls)
        println("Nlow = $l, factor = $i")
        main(system, l, i)  # Replace F with loop index i
    end
end
