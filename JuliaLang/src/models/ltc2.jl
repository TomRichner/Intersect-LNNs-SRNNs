# ltc2.jl — LTC ODE layer faithful to Hasani's MATLAB trajectory analysis code
#
# Faithful port of:
#   liquid_time_constant_networks/trajectory_length_analysis/ltc_def.m
#
# Key differences from ltc.jl (simplified form):
#   1. Separate sensory (feedforward) and recurrent dendritic pathways
#   2. N×N reversal potential matrices E_sens, E_rec (not a vector A)
#   3. abs(f) in decay term (matches Hasani's MATLAB code)
#   4. Multi-layer support: n_layers layers of k neurons each
#      Layer 1 receives external input; layer j≥2 receives layer j-1's state.
#      Each layer is self-recurrent but no backward inter-layer connections.
#
# Dynamics for neuron i in layer j:
#   f_ff_i  = act(W_ff[j]' * source + b_ff[j])    (feedforward dendrite)
#   f_rec_i = act(W_rec[j]' * x_layer_j + b_rec[j]) (recurrent dendrite)
#
#   dx_i/dt = -x_i * (1/τ_i + |f_ff_i| + |f_rec_i|)
#             + Σ_m E_ff[j][m,i_local] * f_ff_i
#             + Σ_m E_rec[j][m,i_local] * f_rec_i

using Lux, Random, NNlib

"""
    LTCODE2(k, n_in; n_layers=1, activation=tanh)

LTC ODE layer faithful to Hasani's MATLAB trajectory analysis code (ltc_def.m).
Two-dendrite model with separate feedforward and recurrent pathways per neuron.

# Arguments
- `k::Int`: Neurons per layer
- `n_in::Int`: External input dimension
- `n_layers::Int`: Number of layers (default: 1)
- `activation`: Activation function (default: tanh)

Total neurons: `N = n_layers * k`

# Trainable Parameters (per layer, stored as vectors of matrices)
- `W_ff[j]`: Feedforward weights — (n_in × k) for layer 1, (k × k) for j ≥ 2
- `b_ff[j]`: Feedforward bias (k)
- `E_ff[j]`: Feedforward reversal potential matrix (k × k)
- `W_rec[j]`: Recurrent weights (k × k)
- `b_rec[j]`: Recurrent bias (k)
- `E_rec[j]`: Recurrent reversal potential matrix (k × k)
- `tau`: Time constants (N) — positive via abs + clamp

# Usage
```julia
layer = LTCODE2(25, 2; n_layers=1)
ps, st = Lux.setup(rng, layer)
st_driven = merge(st, (input=u_t,))
dxdt, st = layer(x, ps, st_driven)
```
"""
struct LTCODE2{F} <: Lux.AbstractLuxLayer
    k::Int          # neurons per layer
    n_in::Int       # external input dimension
    n_layers::Int   # number of layers
    activation::F
end

function LTCODE2(k::Int, n_in::Int; n_layers::Int=1, activation=tanh)
    LTCODE2(k, n_in, n_layers, activation)
end

# Total neuron count
_total_n(layer::LTCODE2) = layer.n_layers * layer.k

function Lux.initialparameters(rng::AbstractRNG, layer::LTCODE2)
    k, n_in, n_layers = layer.k, layer.n_in, layer.n_layers
    N = _total_n(layer)

    # Weight distribution: σ_w = sqrt(2) scaled by k (matching main.m)
    σ_w = Float32(sqrt(2.0))
    σ_b = Float32(1.0)
    w_scale = Float32(σ_w * k / k)  # weight_dist_variance / k

    # Build per-layer parameter tuples stored as NamedTuples in a Tuple
    # For Lux compatibility, flatten into a single NamedTuple with indexed names

    params = Dict{Symbol, Any}()

    for j in 1:n_layers
        ff_in = (j == 1) ? n_in : k  # layer 1 gets external input, rest get k-dim

        params[Symbol("W_ff_$j")]  = w_scale .* randn(rng, Float32, ff_in, k)
        params[Symbol("b_ff_$j")]  = σ_b .* randn(rng, Float32, k)
        params[Symbol("E_ff_$j")]  = w_scale .* randn(rng, Float32, k, k)
        params[Symbol("W_rec_$j")] = w_scale .* randn(rng, Float32, k, k)
        params[Symbol("b_rec_$j")] = σ_b .* randn(rng, Float32, k)
        params[Symbol("E_rec_$j")] = w_scale .* randn(rng, Float32, k, k)
    end

    # Time constants: abs(Gaussian) + clamp, stored as raw values
    params[:tau] = abs.(σ_b .* randn(rng, Float32, N)) .+ 1.0f-4

    return NamedTuple(params)
end

Lux.initialstates(::AbstractRNG, ::LTCODE2) = (input = nothing,)

function Lux.parameterlength(layer::LTCODE2)
    k, n_in, n_layers = layer.k, layer.n_in, layer.n_layers
    N = _total_n(layer)
    # Layer 1: n_in*k + k + k*k + k*k + k + k*k = n_in*k + 2k + 3k²
    # Layer j≥2: k*k + k + k*k + k*k + k + k*k = 2k + 4k²
    layer1 = n_in * k + 2 * k + 3 * k * k
    layerj = (n_layers > 1) ? (n_layers - 1) * (2 * k + 4 * k * k) : 0
    return layer1 + layerj + N  # + N for tau
end

Lux.statelength(::LTCODE2) = 0

function (layer::LTCODE2)(x::AbstractVector, ps, st)
    k = layer.k
    n_layers = layer.n_layers
    act = layer.activation
    N = _total_n(layer)

    # Get input (zero if not provided)
    u = st.input
    if isnothing(u)
        u = zeros(eltype(x), layer.n_in)
    end

    # Time constants (ensure positive)
    τ = max.(ps.tau, 1.0f-4)

    # Output: dx/dt for all N neurons
    dxdt = similar(x, N)

    for j in 1:n_layers
        # Index range for this layer's neurons
        idx = ((j - 1) * k + 1):(j * k)
        x_layer = x[idx]

        # Get per-layer parameters
        W_ff  = ps[Symbol("W_ff_$j")]
        b_ff  = ps[Symbol("b_ff_$j")]
        E_ff  = ps[Symbol("E_ff_$j")]
        W_rec = ps[Symbol("W_rec_$j")]
        b_rec = ps[Symbol("b_rec_$j")]
        E_rec = ps[Symbol("E_rec_$j")]
        τ_layer = τ[idx]

        # Feedforward dendrite
        if j == 1
            # Layer 1: external input
            source = u
        else
            # Layer j≥2: previous layer's state
            idx_prev = ((j - 2) * k + 1):((j - 1) * k)
            source = x[idx_prev]
        end
        f_ff = act.(W_ff' * source .+ b_ff)

        # Recurrent dendrite
        f_rec = act.(W_rec' * x_layer .+ b_rec)

        # Reversal potential drive terms: Σ_m E[m,i] * f[i]
        # (each neuron's own activation times sum of its E column)
        # Matching ltc_def.m: sum(reshape(E_l(j,:,i), k, 1) * f(i))
        # = f(i) * sum(E(:,i))  for each neuron i
        drive_ff  = f_ff .* vec(sum(E_ff, dims=1))
        drive_rec = f_rec .* vec(sum(E_rec, dims=1))

        # Dynamics
        dxdt[idx] = -x_layer .* (1.0f0 ./ τ_layer .+ abs.(f_ff) .+ abs.(f_rec)) .+
                     drive_ff .+ drive_rec
    end

    return dxdt, st
end
