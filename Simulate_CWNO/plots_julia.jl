using Plots
using LinearAlgebra
using JLD2

"""
    load_matrix(filename::String; N::Int=29791, symmetrize::Bool=true)

Reads a binary matrix of Float64 values from file and reshapes it into an NxN matrix.
Optionally symmetrizes it to make it Hermitian.
"""
function load_matrix(filename::String; N::Int=29791, symmetrize::Bool=true)
    println("Reading $filename ...")
    open(filename, "r") do io
        data = read!(io, Array{Float64}(undef, N * N))
        A = reshape(data, N, N)
        if symmetrize
            A = 0.5 * (A + A')
        end
        return Hermitian(A)
    end
end

# read matrix MIC
# A = load_matrix("./CWNO_MIC1.dat"; N=29791)
# D = diag(A)
# plot(1:length(D), D,
#         title="Diagonal Elements Scatter Plot",
#         xlabel="Index",
#         ylabel="Diagonal Elements",
#         markerstrokewidth=0,
#         markersize=4,
#         yscale = :log10,
#         size=(600, 600))

# savefig("diagonal_elements_MIC1.png")


# B = load_matrix("./CWNO_1.dat"; N=29791)
# D2 = diag(B)
# plot(1:length(D2), D2,
#         title="Diagonal Elements Scatter Plot",
#         xlabel="Index",
#         ylabel="Diagonal Elements",
#         markerstrokewidth=0,
#         markersize=4,
#         yscale = :log10,
#         size=(600, 600))    

# savefig("diagonal_elements_CWNO1.png")

C = load_matrix("./CWNO_final_1.dat"; N=29791)
D3 = abs.(diag(C))
plot(1:length(D3), D3,
        title="Diagonal Elements Scatter Plot",
        xlabel="Index",     
        ylabel="Diagonal Elements",
        markerstrokewidth=0,
        markersize=4,
        yscale = :log10,
        size=(600, 600))    

savefig("diagonal_elements_CWNO_final.png") 

# Visualization (optional)
using Plots

heatmap(C[8000:9000, 8000:9000], title="γ-matrix (first 1000x1000 block)")
savefig("gamma_matrix_heatmap_middle.png")
