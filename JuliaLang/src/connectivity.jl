# connectivity.jl — RMT E/I weight matrix generation (port of RMTMatrix.m + RMTConnectivity.m)
#
# Generates dense weight matrices with E/I structure per Harris et al. (2023).
# These serve as initialization for W, which becomes a trainable parameter.

using LinearAlgebra, SparseArrays, Random

"""
    generate_rmt_matrix(n, indegree, f; level_of_chaos=1.0, E_W=0.0,
                        zrs_mode=:none, rng=Random.default_rng())

Generate a recurrent weight matrix W using Random Matrix Theory (Harris 2023).

# Arguments
- `n::Int`: Number of neurons
- `indegree::Int`: Expected in-degree per neuron (connections received)
- `f::Float64`: Fraction of excitatory neurons (0 < f < 1)
- `level_of_chaos::Float64`: Scaling factor for W (default 1.0)
- `E_W::Float64`: Mean offset added to both E and I means (default 0.0)
- `zrs_mode::Symbol`: Zero-row-sum mode (:none, :ZRS, :SZRS, :Partial_SZRS)
- `rng`: Random number generator

# Returns
- `W::Matrix{Float64}`: Weight matrix (n × n), dense, ready for training
- `E_indices::Vector{Int}`: Indices of excitatory neurons
- `I_indices::Vector{Int}`: Indices of inhibitory neurons
- `info::NamedTuple`: Spectral radius, theoretical R, etc.
"""
function generate_rmt_matrix(n::Int, indegree::Int, f::Float64;
                             level_of_chaos::Float64=1.0,
                             E_W::Float64=0.0,
                             zrs_mode::Symbol=:none,
                             rng=Random.default_rng())

    @assert 0 < f < 1 "f must be between 0 and 1"
    @assert 0 < indegree <= n "indegree must be between 1 and n"

    n_E = round(Int, f * n)
    n_I = n - n_E
    E_indices = collect(1:n_E)
    I_indices = collect(n_E+1:n)

    # Connection probability
    α = indegree / n

    # Normalization factor (Harris 2023)
    F = 1.0 / sqrt(n * α * (2.0 - α))

    # Default tilde parameters (matching MATLAB RMTConnectivity defaults)
    μ_E_tilde = 3.0 * F
    μ_I_tilde = -4.0 * F
    σ_E_tilde = F
    σ_I_tilde = F

    # Apply mean offset
    μ_E_eff = μ_E_tilde + E_W
    μ_I_eff = μ_I_tilde + E_W

    # Build base random matrix A (n × n Gaussian)
    A = randn(rng, n, n)

    # Build sparsity mask S
    S = rand(rng, n, n) .< α

    # Build diagonal variance matrix D
    D_vec = zeros(n)
    D_vec[E_indices] .= σ_E_tilde
    D_vec[I_indices] .= σ_I_tilde
    D = Diagonal(D_vec)

    # Build low-rank mean structure M = u * v'
    v = zeros(n)
    v[E_indices] .= μ_E_eff
    v[I_indices] .= μ_I_eff
    M = ones(n) * v'  # rank-1

    # Construct W based on ZRS mode
    if zrs_mode == :none
        W_dense = A * D + M
        W = S .* W_dense
    elseif zrs_mode == :SZRS
        W_base = S .* (A * D + M)
        row_sums = sum(W_base, dims=2)
        row_counts = sum(S, dims=2)
        row_counts = max.(row_counts, 1)  # avoid division by zero
        W_bar = row_sums ./ row_counts
        B = S .* W_bar
        W = W_base - B
    elseif zrs_mode == :Partial_SZRS
        J_base = S .* (A * D)
        M_base = S .* M
        J_row_sums = sum(J_base, dims=2)
        row_counts = sum(S, dims=2)
        row_counts = max.(row_counts, 1)
        J_bar = J_row_sums ./ row_counts
        B_partial = S .* J_bar
        W = (J_base - B_partial) + M_base
    else
        error("Unknown zrs_mode: $zrs_mode. Valid: :none, :SZRS, :Partial_SZRS")
    end

    # Scale by level_of_chaos
    W = level_of_chaos .* W

    # Compute spectral properties
    eigs = eigvals(W)
    spectral_radius = maximum(abs.(eigs))
    abscissa = maximum(real.(eigs))

    # Theoretical spectral radius (Harris 2023 Eq 18)
    σ_se_sq = α * (1 - α) * μ_E_eff^2 + α * σ_E_tilde^2
    σ_si_sq = α * (1 - α) * μ_I_eff^2 + α * σ_I_tilde^2
    R_theoretical = sqrt(n * (f * σ_se_sq + (1 - f) * σ_si_sq)) * level_of_chaos

    info = (spectral_radius=spectral_radius,
            abscissa=abscissa,
            R_theoretical=R_theoretical,
            n_E=n_E, n_I=n_I, α=α)

    return W, E_indices, I_indices, info
end
