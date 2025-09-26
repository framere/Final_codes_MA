using LinearAlgebra
using Printf
using JLD2
using Random

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


"""
Estimate largest eigenvalue (in magnitude) of Hermitian A using power method.
Returns (λ_max, λ_min) estimates by running power on A and -A.
"""
function estimate_spectral_bounds(A::AbstractMatrix{T}; iters::Integer=20) where T<:Number
    n = size(A,1)
    x = randn(T, n)
    for i = 1:iters
        x .= A * x
        x ./= norm(x)
    end
    λmax = dot(x, A * x)

    y = randn(T, n)
    for i = 1:iters
        y .= (-A) * y
        y ./= norm(y)
    end
    λmin = -dot(y, A * y)

    # safety padding
    pad = 1e-6 * max(abs(λmax), abs(λmin)) + 1e-8
    return (λmin - pad, λmax + pad)
end

"""
Apply affine-scaled A (Ahat) to a block of vectors X without forming Ahat:
Ahat = (2/(b-a)) * A  -  ((b+a)/(b-a)) * I
So Ahat_mul(X) = (2/(b-a))*(A*X) - ((b+a)/(b-a))*X
"""
function Ahat_mul!(Y::AbstractMatrix, A::AbstractMatrix, X::AbstractMatrix, a::Float64, b::Float64)
    α = 2.0/(b-a)
    β = (b+a)/(b-a)
    mul!(Y, A, X)                # Y = A * X
    count_matmul_flops(size(A,1), size(A,2), size(X,2))  # your flop accounting
    @. Y = α * Y - β * X
    # count vector ops: scaling and axpy-like: approximate with your counters if needed
    return Y
end

"""
Chebyshev filter: computes T_m(Ahat) * V using recurrence.
- A: original matrix
- V: matrix of vectors to filter (each column a vector)
- a,b: estimated spectral bounds of A (real)
- m: degree of Chebyshev polynomial
Returns filtered vectors (same size as V)
"""
function chebyshev_filter(A::AbstractMatrix{T}, V::Matrix{T}, a::Float64, b::Float64, m::Integer) where T<:Number
    n, k = size(V)
    if m == 0
        return copy(V)
    end

    # We'll reuse workspace to avoid allocations
    W0 = copy(V)                 # T_0(Ahat) V = V
    W1 = similar(V)              # T_1(Ahat) V = Ahat * V
    Ahat_mul!(W1, A, V, a, b)

    if m == 1
        return W1
    end

    W2 = similar(V)
    for j = 2:m
        # W2 = 2*Ahat*W1 - W0
        Ahat_mul!(W2, A, W1, a, b)    # temp = Ahat * W1  (stored in W2 temporarily)
        @. W2 = 2.0 * W2 - W0
        # rotate
        W0, W1 = W1, W2
    end

    # Optionally orthonormalize/filter normalization per column (keeps numeric stability)
    # Normalize columns to unit length (so scale doesn't explode)
    for i = 1:size(W1,2)
        nrm = norm(view(W1,:,i))
        if nrm > 0
            @. W1[:,i] /= nrm
        end
    end

    return W1
end


# Modified Davidson with Chebyshev filtering for computing smallest eigenpairs
function davidson_cheb(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Integer,
    thresh::Float64,
    max_iter::Integer;
    cheb_degree::Integer = 20,
    cheb_apply_to::Symbol = :t,       # :t (apply to correction vectors) or :V (apply to V)
    spec_est_iters::Integer = 20
)::Tuple{Vector{T},Matrix{T}} where T<:Number

    global NFLOPs

    Nlow = size(V, 2)
    if Naux < Nlow
        error("ERROR: auxiliary basis must not be smaller than number of target eigenvalues")
    end

    # estimate spectrum bounds (a = λ_min, b = λ_max)
    a, b = estimate_spectral_bounds(A; iters=spec_est_iters)
    @printf("Estimated spectrum bounds: a=%12.6e  b=%12.6e\n", a, b)

    D = diag(A)
    iter = 0

    while true
        iter += 1

        # Optionally filter V at the start of the iteration
        if cheb_apply_to == :V && cheb_degree > 0
            V = chebyshev_filter(A, V, a, b, cheb_degree)
            # re-orthonormalize below
        end

        # QR-Orthogonalisierung
        count_qr_flops(size(V,1), size(V,2))
        qr_decomp = qr(V)
        V = Matrix(qr_decomp.Q)

        # Rayleigh-Matrix: H = V' * (A * V)
        temp = A * V
        count_matmul_flops(size(A,1), size(A,2), size(V,2))  # A*V
        H = V' * temp
        count_matmul_flops(size(V,2), size(V,1), size(temp,2))  # V'*temp

        H = Hermitian(H)
        Σ, U = eigen(H, 1:Nlow)
        count_diag_flops(size(H,1))  # kleine Diagonalisierung

        X = V * U
        count_matmul_flops(size(V,1), size(V,2), size(U,2))  # V*U

        if iter > max_iter
            println("Maximum iterations reached without convergence.")
            return (Σ, X)
        end

        # R = X*Σ' - A*X
        R = X .* Σ'  # Skalierung
        temp2 = A * X
        count_matmul_flops(size(A,1), size(A,2), size(X,2))  # A*X
        R .-= temp2
        count_vec_add_flops(length(R))

        # Count norm calculation
        Rnorm = norm(R, 2)
        rel_Rnorm = Rnorm / norm(X, 2)
        count_norm_flops(length(R))

        @printf("iter=%6d  rel‖R‖=%11.3e  size(V,2)=%6d\n", iter, rel_Rnorm, size(V,2))

        if rel_Rnorm < thresh
            println("converged!")
            return (Σ, X)
        end

        # Preconditioning
        t = similar(R)
        for i = 1:size(t,2)
            C = 1.0 ./ (Σ[i] .- D)
            t[:,i] = C .* R[:,i]
            count_vec_add_flops(length(D))       # For Σ[i] .- D
            count_vec_scaling_flops(length(D))   # For the division
            count_vec_scaling_flops(length(D))   # For the multiplication
        end

        # Apply Chebyshev filter to the correction vectors (or V depending on cheb_apply_to)
        if cheb_degree > 0 && cheb_apply_to == :t
            t = chebyshev_filter(A, t, a, b, cheb_degree)
        end

        # Ensure orthogonality of new vectors (small QR)
        # Optionally orthonormalize t against current V
        # simple classical Gram-Schmidt against V (could replace with more robust orthonormalization)
        for i = 1:size(t,2)
            # project out V
            coeffs = V' * t[:,i]
            t[:,i] .-= V * coeffs
            # normalize
            nrm = norm(t[:,i])
            if nrm > 0
                t[:,i] ./= nrm
            end
        end

        # Update V (restart logic)
        if size(V,2) <= Naux - Nlow
            V = hcat(V, t)
        else
            # simple restart: keep only current approximations
            V = hcat(X, t)
        end
    end
end


function main(molecule::String, l::Integer, alpha::Integer)
    global NFLOPs
    NFLOPs = 0  # reset for each run

    filename = "../" * molecule *"/gamma_VASP_RNDbasis1.dat"
    
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

    println("Davidson")
    @time Σ, U = davidson_cheb(A, V, Naux, 8e-5, 8000)
    idx = sortperm(Σ)
    Σ = Σ[idx]
    U = U[:, idx]
    Σ = sqrt.(abs.(Σ))  # Take square root of eigenvalues   

    println("Total estimated FLOPs: $(NFLOPs)")

    # Perform exact diagonalization as reference
    println("\nReading exact Eigenvalues...")
    Σexact = read_eigenresults(molecule)
    idx_exact = sortperm(Σexact)
    Σexact = Σexact[idx_exact]
    Σexact = sqrt.(abs.(Σexact))  # Take square root of exact eigenvalues

    # Display difference
    # println("\nDifference between Davidson and exact eigenvalues:")
    # display("text/plain", (Σ[1:l] - Σexact[1:l])')
    r = min(length(Σ), l)
    println("\nDifference between Davidson and exact eigenvalues:")
    rel_dev = (Σ[1:r] .- Σexact[1:r])
    display("text/plain", rel_dev')
    println("$r Eigenvalues converges, out of $l requested.")
end

alpha = [8, 10]

molecules = ["formaldehyde"]

ls = [10, 15, 25]
for molecule in molecules
    println("Processing molecule: $molecule")
    for a in alpha
        println("Running with alpha = $a")
        for l in ls
            println("Running with l = $l")
            main(molecule, l * occupied_orbitals(molecule), a)
        end
    end
    println("Finished processing molecule: $molecule")
end
