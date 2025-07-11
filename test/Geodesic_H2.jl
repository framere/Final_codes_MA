using LinearAlgebra
using JLD2
using Printf

function read_eigenresults(system::String)
    output_file = "../Eigenvalues_folder/eigenres_H2_$system.jld2"
    println("Reading eigenvalues from $system")
    data = jldopen(output_file, "r")
    eigenvalues = data["eigenvalues"]
    eigenvectors = data["eigenvectors"]
    close(data)
    println("Eigenvalues and eigenvectors loaded from $output_file")
    return sort(eigenvalues; rev=true), eigenvectors
end

systems = ["HFbasis", "RNDbasis1"] #, "RNDbasis1" , "RNDbasis2", "RNDbasis3"]

function geodesic_distance(U::AbstractMatrix{<:Number})
    # Check unitarity (optional but recommended)
    # unitary_check = norm(U' * U - I, 2)
    # @printf("Unitarity check: ‖U'U - I‖₂ = %.2e\n", unitary_check)

    # Compute matrix logarithm (will be complex if needed automatically)
    println("Computing geodesic distance for matrix of size $(size(U))")
    L = log(U)
    
    println("Matrix logarithm computed, size of L: $(size(L))")
    d = norm(L, 2)  # Equivalent to opnorm(L)
    return d
end

for system in systems
    eigenvalues, eigenvectors = read_eigenresults(system)
    distance = geodesic_distance(eigenvectors)
    @printf("Geodesic distance for system %s: %.6f\n", system, distance)
end