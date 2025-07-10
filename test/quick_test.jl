using LinearAlgebra
using Random


function calculate_transform_matrix(N::Int)
    Urand = rand(N, N) .- 0.5
    A = Hermitian(rand(N, N) .- 0.5)
    result = Urand' * (A * Urand)
    return result
end


Ns = [100, 300, 1000, 4000, 5000, 6000]

for N in Ns
    println("Calculating transform matrix for N = $N")
    @time result = calculate_transform_matrix(N)   
end