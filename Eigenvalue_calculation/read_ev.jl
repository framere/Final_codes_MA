using JLD2
using LinearAlgebra

function read_eigenresults(molecule::String)
    output_file = "Eigenvalues_folder/eigenres_" * molecule * "_RNDbasis1.jld2"
    println("Reading eigenvalues from $output_file")
    data = jldopen(output_file, "r")
    eigenvalues = data["eigenvalues"]
    close(data)
    return sort(eigenvalues)
end

function load_matrix(filename::String, molecule::String)
    if molecule == "H2"
        N = 11994
    elseif molecule == "formaldehyde"
        N = 27643
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

molecules = ["H2", "formaldehyde"]
N = [200, 1200]

for (molecule, n) in zip(molecules, N)
    # println("Processing molecule: $molecule")
    
    # Load the matrix
    filename = molecule * "/gamma_VASP_RNDbasis1.dat"
    A = load_matrix(filename, molecule)

    # println("Matrix loaded for $molecule with size $(size(A))")
    # Read eigenvalues
    eigenvalues = read_eigenresults(molecule)
    sorted_root = sort(sqrt.(abs.(eigenvalues)), rev=true)

    println("Eigenvalues for $molecule: ", sorted_root[n])
    println("Squared eigenvalue: ", eigenvalues[n])
end