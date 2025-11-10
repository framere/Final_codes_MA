using JLD2

function read_eigenresults(number::Integer)
    output_file = "./CWNO_$(number)_results.jld2"
    println("Reading eigenvalues from $output_file")
    data = jldopen(output_file, "r")
    eigenvalues = data["eigenvalues"]
    close(data)
    return sort(eigenvalues)
end

numbers = 1

for num in numbers
    eigenvalues = read_eigenresults(num)
    println("Printing first and last 5 eigenvalues for CWNO_$(num):")
    println("First 5 eigenvalues: ", eigenvalues[1:5])
    println("Last 5 eigenvalues: ", eigenvalues[end-4:end])
end

