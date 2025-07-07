using LinearAlgebra
using Printf
using JLD2

# Orthogonalize correction vectors against current and locked vectors
function select_corrections_ORTHO(t_candidates, V, V_lock, η, droptol; maxorth=2)
    ν = size(t_candidates, 2)
    n_b = 0
    T_hat = Matrix{eltype(t_candidates)}(undef, size(t_candidates, 1), ν)

    for i in 1:ν
        t_i = t_candidates[:, i]
        old_norm = norm(t_i)
        k = 0

        while k < maxorth
            k += 1

            for j in 1:size(V, 2)
                t_i -= V[:, j] * (V[:, j]' * t_i)
            end

            new_norm = norm(t_i)
            if new_norm > η * old_norm
                break
            end
            old_norm = new_norm
        end

        if norm(t_i) > droptol
            n_b += 1
            T_hat[:, n_b] = t_i / norm(t_i)
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

function davidson(A::AbstractMatrix{T},
    V::Matrix{T},
    n_aux::Integer,
    l::Integer,
    thresh::Float64,
    system::String = ""
)::Tuple{Vector{T}, Matrix{T}} where T<:Number

    n_b = size(V, 2)
    nu_0 = max(l, n_b)
    nevf = 0

    D = diag(A)
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
                    V[:, j] -= v_lock * (v_lock' * V[:, j])
                end
            end
        end
        V = Matrix(qr(V).Q)

        # Rayleigh-Ritz
        H = Hermitian(V' * (A * V))
        nu = min(size(H, 2), nu_0 - nevf)
        Σ, U = eigen(H, 1:nu)
        X = V * U  # Ritz vectors

        # Compute residuals
        R = X .* Σ' - A * X
        norms = vec(norm.(eachcol(R)))

        # Group eigenvalues that are closer than 0.01*value
        groups = Vector{Vector{Int}}()
        if length(Σ) > 0
            current_group = [1]
            for i = 2:length(Σ)
                if abs(Σ[i] - Σ[i-1]) < 0.01 * abs(Σ[i])
                    push!(current_group, i)
                else
                    push!(groups, current_group)
                    current_group = [i]
                end
            end
            push!(groups, current_group)
        end

        # Determine which eigenvalues to lock
        to_lock = falses(length(Σ))
        for group in groups
            # Check convergence for all in group
            group_converged = all(norms[group] .<= thresh)
            
            # Check if group is isolated from others
            isolated = true
            group_idx = findfirst(x -> x == group, groups)
            if group_idx > 1  # check previous group
                prev_group = groups[group_idx-1]
                if abs(Σ[group[1]] - Σ[prev_group[end]]) < 0.01 * abs(Σ[group[1]])
                    isolated = false
                end
            end
            
            if group_converged && isolated
                to_lock[group] .= true
            end
        end

        # Lock the selected eigenvalues
        for i = 1:length(Σ)
            if to_lock[i]
                push!(Eigenvalues, Σ[i])
                Ritz_vecs = hcat(Ritz_vecs, X[:, i])
                V_lock = hcat(V_lock, X[:, i])
                nevf += 1
                println(@sprintf("Converged and locked eigenvalue %.10f with norm %.2e (EV %d)", Σ[i], norms[i], nevf))
                if nevf >= l
                    println("Converged all eigenvalues.")
                    return (Eigenvalues, Ritz_vecs)
                end
            end
        end

        # Find non-converged indices (including converged but not locked)
        non_conv_indices = findall(.!to_lock)
        X_nc = X[:, non_conv_indices]
        Σ_nc = Σ[non_conv_indices]
        R_nc = R[:, non_conv_indices]

        # Correction vectors
        t = Matrix{T}(undef, size(A, 1), length(non_conv_indices))
        ϵ = 1e-6  # small value to avoid division by zero
        for (i, idx) in enumerate(non_conv_indices)
            denom = clamp.(Σ_nc[i] .- D, ϵ, Inf)
            t[:, i] = R_nc[:, i] ./ denom
        end

        # Orthogonalize corrections
        T_hat, n_b_hat = select_corrections_ORTHO(t, V, V_lock, 0.1, 1e-10)

        # Update subspace
        if size(V, 2) + n_b_hat > n_aux || any(to_lock) || n_b_hat == 0 
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

# function load_eigenresults(output_file="eigen_results.jld2")
#     # Unpack directly into variables
#     data = load(output_file)  # Returns a Dict-like object
#     Σexact = data["Σexact"]  # Access by key
#     Uexact = data["Uexact"]
#     return Σexact, Uexact
# end

function main(system::String, l::Integer = 200)
    println("\nRunning Davidson algorithm for system $system seeking $l eigenvalues")
    
    # Matrix parameters
    filename = "../formaldehyde/gamma_VASP_" * system * ".dat"

    Nlow = 25
    Naux = Nlow * 12
    A = load_matrix(filename)
    N = size(A, 1)

    # Initial guess vectors
    V = zeros(N, Nlow)
    for i = 1:Nlow
       V[i,i] = 1.0
    end

    # Perform Davidson algorithm
    println("\nStarting Davidson iteration...")
    @time Σ, U = davidson(A, V, Naux, l, 1e-2, system)

    # Sort results
    idx = sortperm(Σ)
    Σ = Σ[idx]
    U = U[:,idx]

    # Perform exact diagonalization as reference
    # println("\nPerforming full diagonalization for reference...")
    # Σexact, Uexact = load_eigenresults("../../MA_best/Eigenvalues_folder/eigen_results_$system.jld2")

    # # Display difference
    # println("\nDifference between Davidson and exact eigenvalues:")
    # display("text/plain", (Σ - Σexact[1:l])')
end


systems = ["HFbasis","RNDbasis1", "RNDbasis2", "RNDbasis3"]
ls = [60, 90, 120]
for system in systems
    println("Running for system: $system")
    for (i, l) in enumerate(ls)
        println("Nlow = $l, factor = $i")
        main(system, l)  # Replace F with loop index i
    end
end