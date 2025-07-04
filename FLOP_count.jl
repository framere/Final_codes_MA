using LinearAlgebra

function count_matmul_flops(M::Int, N::Int, K::Int)
    global NFLOPs += 2 * M * N * K
end

function count_diag_flops(N::Int)
    global NFLOPs += 20 * N^3
end

function count_qr_flops(M::Int, N::Int)
    global NFLOPs += 2 * M * N^2
end

function count_norm_flops(N::Int)
    global NFLOPs += 2 * N
end

function count_vec_scaling_flops(N::Int)
    global NFLOPs += N
end

function count_vec_add_flops(N::Int)
    global NFLOPs += N
end

function count_dot_product_flops(N::Int)
    global NFLOPs += 2 * N
end

function count_orthogonalization_flops(M::Int, N::Int, vec_length::Int)
    global NFLOPs += 2 * M * N * vec_length  # dot products
    global NFLOPs += M * N * vec_length      # vector updates
end
