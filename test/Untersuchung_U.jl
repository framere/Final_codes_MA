using LinearAlgebra
using JLD2
using Printf

function read_eigenresults(system::String)
    output_file = "../Eigenvalues_folder/eigenres_formaldehyde_$system.jld2"
    println("Reading eigenvalues from $system")
    data = jldopen(output_file, "r")
    eigenvalues = data["eigenvalues"]
    eigenvectors = data["eigenvectors"]
    close(data)
    println("Eigenvalues and eigenvectors loaded from $output_file")
    return sort(eigenvalues; rev=true), eigenvectors
end

systems = ["HFbasis", "RNDbasis1"] #, "RNDbasis1" , "RNDbasis2", "RNDbasis3"]

for system in systems
    eigenvalues, eigenvectors = read_eigenresults(system)
    # Ensure eigenvectors are unitary
    S = svdvals(eigenvectors)
    for i in 1:length(S)
        if abs(S[i]) < 0.95
            @printf("Warning: Singular value %d is differnt to 1, may indicate non-unitarity.\n", i)
        end
    end
end