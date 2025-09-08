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

        # Estimate FLOPs for MINRES solve (approx)
        NFLOPs += maxiter * (2*n^2 + 4*n*k)
        s_j = minres(M_op, rhs; reltol=tol, maxiter=maxiter)

        s_j = s_j - (U * (U' * s_j)); 
        count_matmul_flops(k,1,n); count_matmul_flops(n,1,k); count_vec_add_flops(n)

        S[:, j] = s_j
    end
    return S
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


function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Integer,
    thresh::Float64,
    solver::Symbol,  # Choose between :cg and :minres
    max_iter = 100
)::Tuple{Vector{T},Matrix{T}} where T<:Number


    global NFLOPs
    Nlow = size(V,2) # number of eigenvalues we are interested in
    if Naux < Nlow
        println("ERROR: auxiliary basis must not be smaller than number of target eigenvalues")
    end

    # iterations
    iter = 0
    while true
        iter = iter + 1
        
        # orthogonalize guess orbitals (using QR decomposition)
        count_qr_flops(size(V,1), size(V,2))
        qr_decomp = qr(V)
        V = Matrix(qr_decomp.Q)
        
        # construct and diagonalize Rayleigh matrix
        temp = A * V
        count_matmul_flops(size(A,1), size(A,2), size(V,2))  # A*V
        H = V' * temp
        count_matmul_flops(size(V,2), size(V,1), size(temp,2)) 
        
        H = Hermitian(H)
        Σ, U = eigen(H, 1:Nlow)
        count_diag_flops(size(H,1))
        
        X = V * U
        count_matmul_flops(size(V,1), size(V,2), size(U,2)) 

        if iter > max_iter
            println("Reached maximum iterations ($max_iter) without convergence.")
            return (Σ, X)  # Return the best found so far
        end
        
        # R = X*Σ' - A*X
        R = X .* Σ'  # Skalierung
        temp2 = A * X
        count_matmul_flops(size(A,1), size(A,2), size(X,2))  # A*X
        R .-= temp2
        count_vec_add_flops(length(R))

        # Count norm calculation
        Rnorm = norm(R, 2)

        count_norm_flops(length(R))

        # status output
        output = @sprintf("iter=%6d  Rnorm=%11.3e  size(V,2)=%6d\n", iter, Rnorm, size(V,2))
        print(output)
        
        if Rnorm < thresh
            println("converged!")
            return (Σ, X)
        end
        
        # Solve correction equations using chosen solver
        if solver == :cg
            t = correction_equations_cg(A, X, Σ, R; tol=1e-2, maxiter=100)
        elseif solver == :minres
            t = correction_equations_minres(A, X, Σ, R; tol=1e-2, maxiter=100)
        else
            error("Unknown solver: $solver. Choose :cg or :minres")
        end
        
        # update guess basis
        if size(V,2) + size(t,2) <= Naux
            V = hcat(V, t) # concatenate V and correction
        else
            # Restart: keep only the current Ritz vectors and new corrections
            V = hcat(X, t)
        end
        
        # Optional: limit the basis size to prevent excessive memory usage
        if size(V,2) > Naux
            # Keep the most recent vectors
            V = V[:, end-Naux+1:end]
        end
    end
end


function main(molecule::String, l::Integer, alpha::Integer; solver::Symbol = :cg)
    global NFLOPs
    NFLOPs = 0  # reset for each run

    filename ="../../uracil/uracil/gamma_VASP_RNDbasis.dat"

    A = load_matrix(filename,molecule)
    N = size(A, 1)

    Nlow = l
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

    println("Davidson with $(solver == :cg ? "CG" : "MINRES") solver")
    @time Σ, U = davidson(A, V, Naux, 8e-5, solver, 50)

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
    difference = (Σ[1:r] .- Σexact[1:r])
    for i in 1:r
        println(@sprintf("%3d: %.10f (computed) - %.10f (exact) = % .4e", i, Σ[i], Σexact[i], difference[i]))
    end
    println("$r Eigenvalues converges, out of $l requested.")
end



alpha = [8, 10, 4]
molecules = ["uracil"]
# ls = [50, 75, 100, 200] #10, 50, 100, 200
ls = [10, 15, 25]
for molecule in molecules
    println("Processing molecule: $molecule")
    for a in alpha
        println("Running with alpha = $a")
        for l in ls
            println("Running with l = $l")
            # main(molecule, l * occupied_orbitals(molecule), a, solver=:cg)
            main(molecule, l * occupied_orbitals(molecule), a, solver=:minres)
        end
    end
    println("Finished processing molecule: $molecule")
end
