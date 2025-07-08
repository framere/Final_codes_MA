using LinearAlgebra
using Printf
using Random

# === Global FLOP counter and helpers ===
global NFLOPs = 0

include("../FLOP_count.jl")

function load_matrix(system::String)
    N = 27643

    filename = "../formaldehyde/gamma_VASP_" * system * ".dat"
    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    # A = -A
    A = Hermitian(A)
    return A
end

function davidson(
    A::AbstractMatrix{T},
    V::Matrix{T},
    Naux::Integer,
    thresh::Float64,
    system::String
)::Tuple{Vector{T},Matrix{T}} where T<:Number
    
    global NFLOPs

    Nlow = size(V, 2)
    if Naux < Nlow
        println("ERROR: auxiliary basis must not be smaller than number of target eigenvalues")
    end

    D = diag(A)
    iter = 0

    while true
        iter += 1

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

        # R = X*Σ' - A*X
        R = X .* Σ'  # Skalierung
        temp2 = A * X
        count_matmul_flops(size(A,1), size(A,2), size(X,2))  # A*X
        R .-= temp2
        count_vec_add_flops(length(R))

        # Count norm calculation
        Rnorm = norm(R, 2)
        count_norm_flops(length(R))

        output = @sprintf("iter=%6d  Rnorm=%11.3e  size(V,2)=%6d\n", iter, Rnorm, size(V,2))
        print(output)

        if Rnorm < thresh
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

        # Update V
        if size(V,2) <= Naux - Nlow
            V = hcat(V, t)
        else
            V = hcat(X, t)
        end
    end
end

function main(system::String, Nlow::Int)
    global NFLOPs
    NFLOPs = 0  # Reset FLOP counter
    
    Naux = Nlow * 7
    A = load_matrix(system)
    N = size(A, 1)


    Urand = rand(N,N) .- 0.5
    println("Randomizing the basis")
    @time qr_decomp = qr(Urand)
    Urand = Matrix(qr_decomp.Q)
    A = A*Urand # transform to random basis
    A = Hermitian(A)  # Ensure A is Hermitian after transformation

    # initial guess (naiv)
    V = zeros(N, Nlow)
    for i = 1:Nlow
       V[i,i] = 1.0
    end

    println("Davidson")
    @time Σ, U = davidson(A, V, Naux, 1.3e-2, system)

    # println("Full diagonalization")
    # @time Σexact, Uexact = eigen(A)
    # count_diag_flops(N)

    display("text/plain", Σ')
    # display("text/plain", (Σ - Σexact[1:Nlow])')

    println("Total estimated FLOPs: $(NFLOPs)")
end

systems = ["RNDbasis1"] #"HFbasis", "RNDbasis2", "RNDbasis3"
N_lows = [60, 90, 120, 160, 200]  # Example values for Nlow

for system in systems
    println("Running for system: $system")
    for Nlow in N_lows
        println("Nlow = $Nlow, factor = 1")
        main(system, Nlow)  # Replace F with loop index i
    end
end