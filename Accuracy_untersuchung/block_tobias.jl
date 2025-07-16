using LinearAlgebra
using Printf
using JLD2

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
    Naux::Integer,
    thresh::Float64, 
    max_iter::Integer
)::Tuple{Vector{T},Matrix{T}} where T<:Number
    
    Nlow = size(V, 2)
    if Naux < Nlow
        println("ERROR: auxiliary basis must not be smaller than number of target eigenvalues")
    end

    D = diag(A)
    iter = 0

    while true
        iter += 1
        
        # QR-Orthogonalisierung
        qr_decomp = qr(V)
        V = Matrix(qr_decomp.Q)
        
        # Rayleigh-Matrix: H = V' * (A * V)
        temp = A * V
        H = V' * temp
        
        H = Hermitian(H)
        Σ, U = eigen(H, 1:Nlow)
        X = V * U
        # Verify orthonormality of X
        XtX = X' * X
        err = norm(XtX - I, Inf)
        println(@sprintf("‖XᵀX - I‖_∞ = %.2e (should be close to 0)", err))        
        if iter > max_iter
            println("Maximum iterations reached without convergence.")
            return (Σ, X)
        end
        
        # R = X*Σ' - A*X
        R = X .* Σ'  # Skalierung
        temp2 = A * X
        R .-= temp2
        R .-= X * (X' * R)

        Rnorm = norm(R, 2)

        output = @sprintf("iter=%6d  norm_R=%11.3e  size(V,2)=%6d\n", iter, Rnorm, size(V,2))
        print(output)

        if Rnorm/sqrt(Nlow) < thresh
            println("converged!")
            return (Σ, X)
        end

        # Preconditioning
        t = similar(R)
        for i = 1:size(t,2)
            C = 1.0 ./ (Σ[i] .- D)
            t[:,i] = C .* R[:,i]
        end

        # Update V
        if size(V,2) <= Naux - Nlow
            V = hcat(V, t)
        else
            V = hcat(X, t)
        end
    end
end

function main(molecule::String, l::Integer, alpha::Integer)
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
    @time Σ, U = davidson(A, V, Naux, 1.5e-2, 500)
    idx = sortperm(Σ)
    Σ = Σ[idx]
    U = U[:, idx]

    # Perform exact diagonalization as reference
    println("\nReading exact Eigenvalues...")
    Σexact = read_eigenresults(molecule)

    # Display difference
    r = min(length(Σ), l)
    println("\nDifference between Davidson and exact eigenvalues:")
    difference = (Σ[1:r] .- Σexact[1:r])
    display("text/plain", difference')

    difference_root = sqrt.(abs.(Σ[1:r])) .- sqrt.(abs.(Σexact[1:r]))
    println("\nSquare root of Eigenvalues difference:")
    display("text/plain", difference_root')
    
    println("$r Eigenvalues converges, out of $l requested.")
end

alpha = [4] # ,8,16

molecules = ["formaldehyde"]

ls = [10] # 50, 100, 200
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
