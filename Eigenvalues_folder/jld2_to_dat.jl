using JLD2, DelimitedFiles

function read_eigenresults(name::String)
    println("Reading eigenvalues from $name")
    data = jldopen(name, "r")
    eigenvalues = data["eigenvalues"]
    close(data)
    return sort(eigenvalues)
end

function save_eigenvalues_to_dat(name::String, output_file::String)
    EV = read_eigenresults(name)
    N = length(EV)
    
    # Create a matrix with index and eigenvalue
    data = [1:N EV]
    
    println("Saving eigenvalues to $output_file")
    # Save to .dat file
    writedlm(output_file, data)
    
    println("Saved $N eigenvalues to $output_file")
    println("First few eigenvalues:")
    for i in 1:min(5, N)
        println("  EV[$i] = $(EV[i])")
    end
    return EV
end


molecules = ["formaldehyde", "uracil", "H2"]

for molecule in molecules
    jld2_file = "eigenres_$(molecule)_RNDbasis1.jld2"
    dat_file = "$(molecule)_eigenvalues.dat"
    EV = save_eigenvalues_to_dat(jld2_file, dat_file)
    number = 400
    println("Exact eigenvalue number $number: ", EV[number])
end
