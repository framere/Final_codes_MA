using LinearAlgebra
using JLD2
using Printf

function read_eigenresults(system::String)
    output_file = "../Eigenvalues_folder/eigen_results_$system.jld2"
    println("Reading eigenvalues from $system")
    data = jldopen(output_file, "r")
    eigenvalues = data["eigenvalues"]
    eigenvectors = data["eigenvectors"]
    close(data)
    println("Eigenvalues and eigenvectors loaded from $output_file")
    return sort(eigenvalues; rev=true), eigenvectors
end

systems = ["HFbasis", "RNDbasis1"] # , "RNDbasis2", "RNDbasis3"]

function geodesic_distance(U::AbstractMatrix{<:Number})  # Accepts both real and complex matrices
    # Ensure U is unitary (optional but recommended for correctness)
    dot_product = U' * U
    println("Checking unitarity: norm of U'U = $(norm(dot_product - I, Inf))")

    # Force complex arithmetic for stability (log of a unitary matrix is skew-Hermitian)
    println("Computing geodesic distance for matrix of size $(size(U))")
    L = log(Matrix{ComplexF64}(U))  # Ensures Complex output even for real U
    
    println("Matrix logarithm computed, size of L: $(size(L))")
    d = opnorm(L)
    return d
end

for system in systems
    eigenvalues, eigenvectors = read_eigenresults(system)
    distance = geodesic_distance(eigenvectors)
    println("Geodesic distance for system $system: ", distance)  # Should be 0 for the same system
end

