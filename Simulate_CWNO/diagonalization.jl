using LinearAlgebra
using JLD2

function load_matrix(filename::String)
    N = 29791  

    println("read ", filename)
    file = open(filename, "r")
    A = Array{Float64}(undef, N * N)
    read!(file, A)
    close(file)

    A = reshape(A, N, N)
    # A = -A
    return Hermitian(A)
end

function load_vector_from_file(filename::String)
    # read raw bytes and reinterpret as Float64
    bytes = read(filename)
    return reinterpret(Float64, bytes)
end

function project_out(x::AbstractVector, y::AbstractVector)
    proj = (dot(x, y) / dot(y, y)) * y
    return x - proj
end

function diagonalize_and_save(filename::String)    
    A = load_matrix(filename)
    println("Diagonalizing the matrix ...")
    @time F = eigen(A)  # F.values, F.vectors
    occ_orbital = load_vector_from_file("occupied_orbital.dat")

    # project out the occupied orbital from eigenvectors
    println("Projecting out the occupied orbital from eigenvectors ...")
    for i in 1:size(F.vectors, 2)
        F.vectors[:, i] = project_out(F.vectors[:, i], occ_orbital)
        F.vectors[:, i] ./= norm(F.vectors[:, i])  # normalize
    end

    output_file = "CWNO_final_results.jld2"
    println("Saving results to $output_file")

    jldsave(output_file; 
        eigenvalues = F.values, 
        eigenvectors = F.vectors
    )

    println("Done saving eigenvalues and eigenvectors.")
end

filename = "CWNO_final_1.dat"
diagonalize_and_save(filename)
