using LinearAlgebra
using JLD2

function load_matrix(system::String)
    if system == "He"
        N = 4488
    elseif system == "hBN"
        N = 6863        
    elseif system == "Si"
        N = 6201
    else
        error("Unknown system: $system")
    end

    filename = "../Master_arbeit/Davidson_algorithm/m_pp_" * system * ".dat"
    # filename = "../../../OneDrive - Students RWTH Aachen University/Master_arbeit/Davidson_algorithm/m_pp_" * system * ".dat"
    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    A = -A
    return Hermitian(A)
end

function diagonalize_and_save(system::String)
    A = load_matrix(system)
    println("Diagonalizing the matrix for system: $system")
    @time Σexact, Uexact = eigen(A)
    output_file="eigen_results_$system.jld2"
    println("Saving results to $output_file")
    jldsave(output_file; Σexact, Uexact)  # JLD2 format
    println("Done!")
end

systems = ["He", "hBN", "Si"]
for system in systems
    println("Processing system: $system")
    diagonalize_and_save(system)
end

#------------------------- function to open and read the saved eigenvalues and eigenvectors -------------------------
function load_eigenresults(output_file="eigen_results.jld2")
    # Unpack directly into variables
    data = load(output_file)  # Returns a Dict-like object
    Σexact = data["Σexact"]  # Access by key
    Uexact = data["Uexact"]
    return Σexact, Uexact
end

# # Usage
# Σexact, Uexact = load_eigenresults()
# println("Loaded Σexact (first 5 eigenvalues): ", Σexact[1:5])