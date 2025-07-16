using LinearAlgebra


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



function main(molecule::String)
    # Load the matrix
    filename = "../" * molecule * "/gamma_VASP_RNDbasis1.dat"
    A = load_matrix(filename, molecule)
    nnz_A = count(!iszero, A)             # Number of nonzero elements
    total_elements = prod(size(A))        # Total number of elements
    sparsity_ratio = nnz_A / total_elements
    println("Sparsity ratio: ", sparsity_ratio)
    println("Nonzero percentage: ", 100 * sparsity_ratio, "%")
end


molecule = ["H2", "formaldehyde"]
for mol in molecule
    main(mol)
end