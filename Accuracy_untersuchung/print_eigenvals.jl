using JLD2

function read_eigenresults(molecule::String)
    output_file = "../Eigenvalues_folder/eigenres_" * molecule * "_RNDbasis1.jld2"
    println("Reading eigenvalues from $output_file")
    data = jldopen(output_file, "r")
    eigenvalues = data["eigenvalues"]
    close(data)
    return sort(eigenvalues)
end

eigenvalues = read_eigenresults("formaldehyde")
nev = 1200
println("Print the first $nev eigenvalues: %{eigenvalues[1:nev]}")