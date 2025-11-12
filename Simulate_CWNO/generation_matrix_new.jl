using LinearAlgebra
using Random
using Printf

# --- PARAMETERS ---
grid_min = -5.0       # Å
grid_max =  5.0       # Å
spacing  =  0.33      # Å (grid step)
alpha    =  1.0       # 1/Å exponential decay constant --> roughly 10/L for L=10 Å
A0       =  1.0       # overall amplitude
eps      = spacing/2  # small regularizer to avoid singularity
noise_level = 0.01    # small random noise for realism

# --- BOX LENGTHS (for periodic boundary conditions) ---
Lx = grid_max - grid_min
Ly = grid_max - grid_min
Lz = grid_max - grid_min

xs = collect(grid_min:spacing:grid_max)
ys = collect(grid_min:spacing:grid_max)
zs = collect(grid_min:spacing:grid_max)
coords = [[x, y, z] for x in xs for y in ys for z in zs]
n = length(coords)
@printf("Grid points: %d\n", n)

# --- MIC DISTANCE MATRIX ---
function distance_matrix_MIC(coords, Lx, Ly, Lz)
    """
    Computes the distance matrix using the Minimum Image Convention (MIC)
    for periodic boundary conditions in a rectangular box of lengths Lx, Ly, Lz.
    """
    n = length(coords)
    R = zeros(Float64, n, n)
    L = [Lx, Ly, Lz]

    for i in 1:n
        ri = coords[i]
        for j in i:n
            rj = coords[j]
            d = ri .- rj                     # direct difference
            d_MIC = d .- L .* round.(d ./ L) # apply MIC
            dist = sqrt(sum(d_MIC.^2))
            R[i, j] = dist
            R[j, i] = dist
        end
    end
    return R
end

@printf("Building distance matrix with periodic boundaries (MIC)...\n")
R = distance_matrix_MIC(coords, Lx, Ly, Lz)

# --- γ-MATRIX CALCULATION ---
@printf("Building γ-matrix (Coulomb-weighted kernel)...\n")
γ = zeros(Float64, n, n)

# Precompute |r| for all points
abs_r = [sqrt(sum(c.^2)) for c in coords]

for i in 1:n
    for j in i:n
        denom = R[i, j] + eps
        value = A0 * exp(-alpha * (abs_r[i] + abs_r[j])) / denom
        γ[i, j] = value
        γ[j, i] = value
    end
end

# Add small symmetric random noise for realism
if noise_level > 0
    Random.seed!(1234)
    noise = noise_level * (rand(n, n) .- 0.5)
    γ .+= (noise .+ noise') ./ 2
end

function save_matrix_to_file(A::Matrix{Float64}, filename::String)
    """
    Saves the Hermitian matrix A to a file in flattened binary format.
    The file can later be read with read!(io, Array{Float64,1}) and reshaped.
    """
    A_vec = vec(Matrix(A)) # flatten to 1D
    open(filename, "w") do file
        write(file, A_vec)
    end
end

output_filename = "CWNO_final.dat"
save_matrix_to_file(γ, output_filename)

@printf("\nComputation complete!\n")

# --- Optional: visualize shape or save ---
using Plots
heatmap(γ[1:50, 1:50], title="γ-matrix (first 50x50 block)")
savefig("gamma_matrix_heatmap.png")

