using JLD2

function read_eigenresults(number::Integer)
    output_file = "./CWNO_$(number)_results.jld2"
    println("Reading eigenvalues from $output_file")
    data = jldopen(output_file, "r")
    eigenvalues = data["eigenvalues"]
    close(data)
    return sort(eigenvalues)
end

function read_eigenresults_mol(molecule::String)
    output_file = "../Eigenvalues_folder/eigenres_" * molecule * "_RNDbasis1.jld2"
    println("Reading eigenvalues from $output_file")
    data = jldopen(output_file, "r")
    eigenvalues = data["eigenvalues"]
    close(data)
    return sort(eigenvalues)
end

numbers = 1
molecules = ["formaldehyde", "H2", "uracil"]
for num in numbers
    Σexact = read_eigenresults(num)
    Σexact = abs.(Σexact)
    idx_exact = sortperm(Σexact, rev=true)
    Σexact = Σexact[idx_exact]
    println("Printing first and last 5 eigenvalues for CWNO_$(num):")
    println("First 5 eigenvalues: ", Σexact[1:5])
    println("Last 5 eigenvalues: ", Σexact[end-4:end])
end

for molecules in molecules
    Σexact = read_eigenresults_mol(molecules)
    Σexact = abs.(Σexact)
    idx_exact = sortperm(Σexact, rev=true)
    Σexact = Σexact[idx_exact]
    println("Printing first and last 5 eigenvalues for molecule $(molecules):")
    println("First 5 eigenvalues: ", Σexact[1:5])
    println("Last 5 eigenvalues: ", Σexact[end-4:end])
end