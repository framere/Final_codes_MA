using JLD2
using LinearAlgebra
using Printf

function load_sparse_matrix(filename::String, number::Int)
    if number % 2 == 0
        N = 20000
    elseif number % 2 == 1
        N = 10000
    else
        error("Unknown molecule: $number")
    end
    # println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    return Hermitian(A)
end

function read_eigenresults(number::Int)
    output_file = "Eigenvalues_folders/eigenresults_matrix_" * string(number) * "_2.jld2"
    println("Reading eigenvalues from $output_file")
    data = jldopen(output_file, "r")
    eigenvalues = data["eigenvalues"]
    close(data)
    return sort(eigenvalues)
end

"""
Idea: 
For every column of the matrix save the value of the quotient of the diagonal element and every single off-diagonal element.
This will give a more detailed picture of how far the matrix is from being diagonally dominant.
"""

function analyze_diagonal_dominance_detailed(A::AbstractMatrix{T}, output_filename::String) where T<:Number
    N = size(A, 1)
    println("Matrix size: $N x $N")
    
    # Open output file
    output_file = open(output_filename, "w")
    
    for i in 1:N
        diag_element = abs(A[i, i])
        for j in 1:N
            if j != i
                off_diag_element = abs(A[i, j])
                ratio = diag_element / off_diag_element
                @printf(output_file, "%.15e\n", ratio)
            end
        end
    end
    
    close(output_file)

    println("Detailed diagonal dominance analysis completed and written to $output_filename")
end


"""
The other concept I have is to sort the diagonal elements and then see the difference between the diagonal element and the exact eigenvalue.
"""
function analyze_diagonal_dominance_eigenvalue(A::AbstractMatrix{T}, molecule::String, output_filename::String) where T<:Number
    N = size(A, 1)
    println("Matrix size: $N x $N")

    # extract eigenvalues
    eigenvalues = read_eigenresults(molecule)
    println("Number of eigenvalues read: ", length(eigenvalues))

    # extract diagonal elements
    diag_elements = [abs(A[i, i]) for i in 1:N]
    sorted_diag_elements = sort(diag_elements)

    # compute the absolute and relative differences
    differences = [abs(abs(sorted_diag_elements[i]) - abs(eigenvalues[i])) for i in 1:N]
    relative_differences = [abs(differences[i]) / abs(eigenvalues[i]) for i in 1:N]

    println("Computed differences and relative differences. Writing to file...")

    # Open output file
    open(output_filename, "w") do output_file
        for i in 1:N
            @printf(output_file, "%.15e %.15e\n", differences[i], relative_differences[i])
        end
    end
end



numbers = 1:14

for molecule in molecules
    filename = "../" * molecule *"/gamma_VASP_RNDbasis1.dat"
    output_filename_detailed = "diagonal_analysis_detailed_" * molecule * ".txt"
    output_filename_eigen = "eigenvalues_data/diagonal_analysis_eigenvalue_" * molecule * "_1.txt"
    
    println("Loading matrix from: $filename")
    A = load_matrix(filename, molecule)


    analyze_diagonal_dominance_eigenvalue(A, molecule, output_filename_eigen)
    println("Eigenvalue diagonal dominance analysis written to $output_filename_eigen")

    analyze_diagonal_dominance_detailed(A, output_filename_detailed)
    println("Detailed diagonal dominance analysis written to $output_filename_detailed")
end