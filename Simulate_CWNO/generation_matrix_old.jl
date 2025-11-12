using LinearAlgebra
using Random
using Printf

# --- PARAMETERS ---

grid_min = -5.0       # Å
grid_max =  5.0       # Å
spacing  =  0.33      # Å (grid step)
alpha    =  1.0       # 1/Å exponential decay constant --> alpha = L/10 for L = 10 Å
A0       =  1.0       # overall amplitude
eps      = spacing/2  # small regularizer to avoid singularity
noise_level = 0.01    # small random noise for realism

# --- BOX LENGTHS (for periodic boundary conditions) ---
Lx = grid_max - grid_min
Ly = grid_max - grid_min
Lz = grid_max - grid_min

# --- MOLECULE GEOMETRY (approximate formaldehyde) ---

atoms = Dict(
    "C"  => [0.0,  0.0,  0.0],
    "O"  => [1.21, 0.0,  0.0],
    "H1" => [-0.63,  0.93,  0.0],
    "H2" => [-0.63, -0.93,  0.0],
    "H3" => [1.97,  0.0,  0.90]
)
sigmas = Dict("C"=>0.6, "O"=>0.5, "H1"=>0.8, "H2"=>0.8, "H3"=>0.8)

# --- BUILD 3D GRID ---

xs = collect(grid_min:spacing:grid_max)
ys = collect(grid_min:spacing:grid_max)
zs = collect(grid_min:spacing:grid_max)
coords = [[x,y,z] for x in xs for y in ys for z in zs]
n = length(coords)
@printf("Grid points: %d\n", n)

# --- HELPER: Gaussian atomic envelope ---

function atomic_density(r::Vector{Float64}, atoms::Dict, sigmas::Dict)
    dens = 0.0
    for (label, R0) in atoms
        σ = sigmas[label]
        d2 = sum((r .- R0).^2)
        dens += exp(-d2 / (2σ^2))
    end
    return dens
end

# --- COMPUTE LOCAL DENSITY ENVELOPE ON GRID ---

dens = [atomic_density(r, atoms, sigmas) for r in coords]
dens = dens ./ maximum(dens)  # normalize to 1 at max

# --- BUILD AMPLITUDE MATRIX A_ij = A0 * sqrt(dens_i * dens_j) ---

sqrt_dens = sqrt.(dens)
A = A0 .* (sqrt_dens * sqrt_dens')  # outer product

# --- BUILD DISTANCE MATRIX WITH PERIODIC BOUNDARIES (MIC) ---

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
            R[i,j] = dist
            R[j,i] = dist
        end
    end
    return R
end

@printf("Building distance matrix with periodic boundaries (MIC)...\n")
R = distance_matrix_MIC(coords, Lx, Ly, Lz)

# --- BUILD rho(r,r') = A * exp(-α r) * (1 + noise) ---

Random.seed!(1)
rho = A .* exp.(-alpha .* R) .* (1 .+ noise_level .* randn(n,n))
rho = 0.5 .* (rho + rho')  # symmetrize

# --- BUILD gamma(r,r') = rho / (|r - r'| + eps) ---

gamma = rho ./ (R .+ eps)
for i in 1:n
    gamma[i,i] = rho[i,i] / eps
end

# # --- SPARSITY STATISTICS ---

# absγ = abs.(gamma)
# threshold = 1e-12
# sparse_pct = 100 * count(<(threshold), absγ) / length(absγ)
# @printf("Percent |γ| < %.0e : %.2f%%\n", threshold, sparse_pct)

# # --- PRINT SAMPLE VALUES ---

# center_idx = argmin([sum(abs2, r) for r in coords])  # closest to origin
# @printf("Center point (≈ C atom): %s\n", string(coords[center_idx]))
# @printf("ρ(center,center) = %.3f\n", rho[center_idx,center_idx])
# @printf("γ(center,center) = %.3f\n", gamma[center_idx,center_idx])

# --- SAVE MATRIX TO FILE ---

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

counter = 1 # increment manually or via loop if needed
filename = "CWNO_MIC$(counter).dat"
println("Saving the generated γ matrix to '$filename'...")
save_matrix_to_file(gamma, filename)
println("Matrix saved successfully with dimensions $(size(gamma)).")
