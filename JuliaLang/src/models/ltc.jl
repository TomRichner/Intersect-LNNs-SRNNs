# ltc.jl — Liquid Time-Constant (LTC) Neural ODE layer (port of LNN.m)
#
# Implements the LTC ODE from Hasani et al. (2021):
#   dx/dt = -(1/τ + |f|) .* x + f .* A
#   where f = activation(W * x + W_in * u(t) + μ)
#
# All parameters (W, W_in, μ, τ, A) are trainable via Lux.

using Lux, Random, NNlib

"""
    LTCODE(n, n_in; activation=tanh)

Liquid Time-Constant ODE layer. Defines the right-hand side of the LTC ODE
as a Lux layer with trainable recurrent weights, input weights, bias,
time constants, and reversal potentials.

# Arguments
- `n::Int`: Number of hidden neurons
- `n_in::Int`: Input dimension
- `activation`: Activation function (default: tanh)

# Trainable Parameters (in `ps`)
- `W`: Recurrent weight matrix (n × n)
- `W_in`: Input weight matrix (n × n_in)
- `mu`: Bias vector (n)
- `log_tau`: Log time constants (n) — τ = softplus(log_tau) ensures positivity
- `A`: Reversal potential vector (n)

# Usage
```julia
layer = LTCODE(64, 28)
ps, st = Lux.setup(rng, layer)
dxdt, st = layer(x, ps, st)  # autonomous mode
```

For driven mode, store input in `st.input`:
```julia
st_driven = merge(st, (input=u_t,))
dxdt, st_driven = layer(x, ps, st_driven)
```
"""
struct LTCODE{F} <: Lux.AbstractLuxLayer
    n::Int
    n_in::Int
    activation::F
end

LTCODE(n::Int, n_in::Int; activation=tanh) = LTCODE(n, n_in, activation)

function Lux.initialparameters(rng::AbstractRNG, layer::LTCODE)
    n, n_in = layer.n, layer.n_in
    σ_w = Float32(sqrt(2.0 / n))
    return (
        W      = σ_w .* randn(rng, Float32, n, n),
        W_in   = Float32(0.1) .* randn(rng, Float32, n, n_in),
        mu     = zeros(Float32, n),
        log_tau = zeros(Float32, n),   # softplus(0) ≈ 0.693 → τ ≈ 0.7s
        A      = zeros(Float32, n),
    )
end

Lux.initialstates(::AbstractRNG, ::LTCODE) = (input = nothing,)
Lux.parameterlength(layer::LTCODE) = layer.n * layer.n + layer.n * layer.n_in + 3 * layer.n
Lux.statelength(::LTCODE) = 0

function (layer::LTCODE)(x::AbstractVector, ps, st)
    # Get input (zero if not provided)
    u = st.input
    if isnothing(u)
        u = zeros(eltype(x), layer.n_in)
    end

    # Compute nonlinearity f = activation(W * x + W_in * u + μ)
    z = ps.W * x .+ ps.W_in * u .+ ps.mu
    f = layer.activation.(z)

    # Time constants (positive via softplus)
    τ = NNlib.softplus.(ps.log_tau)

    # LTC dynamics: dx/dt = -(1/τ + |f|) * x + f * A
    dxdt = -(1.0f0 ./ τ .+ abs.(f)) .* x .+ f .* ps.A

    return dxdt, st
end
