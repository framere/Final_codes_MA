using LinearAlgebra
using JLD2
using Printf
using ExponentialUtilities: logm

function read_eigenresults(system::String)
    output_file = "../Eigenvalues_folder/eigen_results_$system.jld2"
    println("Reading eigenvalues from $system")
    data = jldopen(output_file, "r")
    eigenvalues = data["eigenvalues"]
    eigenvectors = data["eigenvectors"]
    close(data)
    return sort(eigenvalues; rev=true), eigenvectors
end

systems = ["HFbasis", "RNDbasis1"] # , "RNDbasis2", "RNDbasis3"]

function geodesic_distance(eigenvectors)
    # Calculate the geometric distance between two sets of eigenvalues
    l = logm(eigenvectors)
    return norm(l, 2)
end

for system in systems
    eigenvalues, eigenvectors = read_eigenresults(system)
    println("Top 10 eigenvalues for system $system: ", eigenvalues[1:10])
    println("Geodesic distance for system $system: ", geodesic_distance(eigenvectors))  # Should be 0 for the same system
end

