# srnn.jl — Stable Recurrent Neural Network ODE layer (port of SRNNModel2.m)
#
# Implements the SRNN dynamics with optional Spike-Frequency Adaptation (SFA)
# and Short-Term Depression (STD). With n_a=0, n_b=0, this reduces to a
# vanilla Hopfield/rate RNN baseline.
#
# State layout: S = [a_E; a_I; b_E; b_I; x]
# All key parameters are trainable via Lux.
#
# SRNNCell wraps SRNN_ODE with fused Euler sub-stepping for BPTT training,
# matching the LTCODE1 interface for drop-in use in train_har.jl.

using Lux, Random, NNlib, Zygote

include(joinpath(@__DIR__, "..", "activations.jl"))

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

"""
    _make_tau_range(lo, hi, n_steps)

Reconstruct log-spaced time constant range from trainable endpoints.
- lo/hi: scalar `(1,)` or per-neuron `(n,)` vector
- Returns: `(length(lo), n_steps)` — each row is linspace(lo[i], hi[i])
Differentiable through Zygote.
"""
function _make_tau_range(lo, hi, n_steps::Int)
    # Build constant interpolation weights [0, 1/(n-1), ..., 1]
    # Wrapped in Zygote.ignore since t doesn't depend on any trainable params
    t = Zygote.ignore() do
        if n_steps == 1
            return Float32[0.5]'
        end
        Float32.([(i - 1) / (n_steps - 1) for i in 1:n_steps])'   # (1, n_steps)
    end
    return lo .+ (hi .- lo) .* t   # (1, n_steps) or (n, n_steps)
end

# ── State unpacking ─────────────────────────────────────────────────────

"""
    _unpack_state(layer::SRNN_ODE, S) → (a_E, a_I, b_E, b_I, x)

Unpack the augmented state vector/matrix into named components.
For vector S: a_E is (n_E, n_a_E), x is (n,), etc.
For matrix S (state_dim, B): a_E is (n_E, n_a_E, B), x is (n, B), etc.
Returns `nothing` for disabled components.
"""
function _unpack_state(layer, S::AbstractVector)
    n, n_E, n_I = layer.n, layer.n_E, layer.n_I
    n_a_E, n_a_I, n_b_E, n_b_I = layer.n_a_E, layer.n_a_I, layer.n_b_E, layer.n_b_I

    idx = 0
    len_a_E = n_E * n_a_E
    a_E = len_a_E > 0 ? reshape(S[idx+1:idx+len_a_E], n_E, n_a_E) : nothing
    idx += len_a_E

    len_a_I = n_I * n_a_I
    a_I = len_a_I > 0 ? reshape(S[idx+1:idx+len_a_I], n_I, n_a_I) : nothing
    idx += len_a_I

    len_b_E = n_E * n_b_E
    b_E = len_b_E > 0 ? S[idx+1:idx+len_b_E] : nothing
    idx += len_b_E

    len_b_I = n_I * n_b_I
    b_I = len_b_I > 0 ? S[idx+1:idx+len_b_I] : nothing
    idx += len_b_I

    x = S[idx+1:idx+n]
    return (a_E=a_E, a_I=a_I, b_E=b_E, b_I=b_I, x=x)
end

function _unpack_state(layer, S::AbstractMatrix)
    n, n_E, n_I = layer.n, layer.n_E, layer.n_I
    n_a_E, n_a_I, n_b_E, n_b_I = layer.n_a_E, layer.n_a_I, layer.n_b_E, layer.n_b_I
    B = size(S, 2)

    idx = 0
    len_a_E = n_E * n_a_E
    a_E = len_a_E > 0 ? reshape(S[idx+1:idx+len_a_E, :], n_E, n_a_E, B) : nothing
    idx += len_a_E

    len_a_I = n_I * n_a_I
    a_I = len_a_I > 0 ? reshape(S[idx+1:idx+len_a_I, :], n_I, n_a_I, B) : nothing
    idx += len_a_I

    len_b_E = n_E * n_b_E
    b_E = len_b_E > 0 ? S[idx+1:idx+len_b_E, :] : nothing
    idx += len_b_E

    len_b_I = n_I * n_b_I
    b_I = len_b_I > 0 ? S[idx+1:idx+len_b_I, :] : nothing
    idx += len_b_I

    x = S[idx+1:idx+n, :]
    return (a_E=a_E, a_I=a_I, b_E=b_E, b_I=b_I, x=x)
end

"""
    _compute_x_eff(layer, st_parts, ps) → x_eff

Apply SFA to dendritic state x. Returns x_eff with same shape as x.
Works for both vector (n,) and matrix (n, B) x.
"""
function _compute_x_eff(layer, st_parts, ps)
    x = st_parts.x
    n_E, n_a_E, n_a_I = layer.n_E, layer.n_a_E, layer.n_a_I
    x_eff = copy(x)

    if n_a_E > 0 && !isnothing(st_parts.a_E)
        c_E = NNlib.softplus.(ps.log_c_E)
        a_E_sum = _sum_adaptation(st_parts.a_E)
        x_eff_E = _rows(x_eff, 1:n_E) .- c_E .* a_E_sum
        x_eff = vcat(x_eff_E, _rows(x_eff, n_E+1:size(x_eff, 1)))
    end

    if n_a_I > 0 && !isnothing(st_parts.a_I)
        c_I = NNlib.softplus.(ps.log_c_I)
        a_I_sum = _sum_adaptation(st_parts.a_I)
        x_eff_I = _rows(x_eff, n_E+1:size(x_eff, 1)) .- c_I .* a_I_sum
        x_eff = vcat(_rows(x_eff, 1:n_E), x_eff_I)
    end

    return x_eff
end

# Dimension-agnostic row selection (works for vectors and matrices)
_rows(v::AbstractVector, idx) = v[idx]
_rows(m::AbstractMatrix, idx) = m[idx, :]

# Sum adaptation over timescale dimension
_sum_adaptation(a::AbstractMatrix) = vec(sum(a, dims=2))              # (n_E, n_a) → (n_E,)
_sum_adaptation(a::AbstractArray{T,3}) where T = dropdims(sum(a, dims=2), dims=2)  # (n_E, n_a, B) → (n_E, B)

# ── Extract b (STD variable) ───────────────────────────────────────────

"""
    _extract_b(layer, st_parts) → b

Extract the full STD vectorb = [b_E; b_I] or ones if no STD.
"""
function _extract_b(layer, st_parts, ::AbstractVector)
    n, n_E = layer.n, layer.n_E
    b = ones(Float32, n)
    if layer.n_b_E > 0 && !isnothing(st_parts.b_E)
        b = vcat(st_parts.b_E, b[n_E+1:end])
    end
    if layer.n_b_I > 0 && !isnothing(st_parts.b_I)
        b = vcat(b[1:n_E], st_parts.b_I)
    end
    return b
end

function _extract_b(layer, st_parts, S::AbstractMatrix)
    n, n_E = layer.n, layer.n_E
    B = size(S, 2)
    b = ones(Float32, n, B)
    if layer.n_b_E > 0 && !isnothing(st_parts.b_E)
        b = vcat(st_parts.b_E, b[n_E+1:end, :])
    end
    if layer.n_b_I > 0 && !isnothing(st_parts.b_I)
        b = vcat(b[1:n_E, :], st_parts.b_I)
    end
    return b
end

# ═══════════════════════════════════════════════════════════════════════
# SRNN_ODE — ODE right-hand side (returns dS/dt)
# ═══════════════════════════════════════════════════════════════════════

"""
    SRNN_ODE(n, n_in, n_E; n_a_E=0, n_a_I=0, n_b_E=0, n_b_I=0,
             per_neuron=false, activation=make_piecewise_sigmoid(S_c=0.0))

Stable Recurrent Neural Network ODE layer. Defines the RHS of the SRNN ODE.

# Arguments
- `n`: Total neurons
- `n_in`: Input dimension
- `n_E`: Number of excitatory neurons (n_I = n - n_E)
- `per_neuron`: If true, dynamics params (τ_d, c, a_0, τ_a endpoints, τ_b) are per-neuron vectors.
               If false (default), they are shared scalars.

# Trainable Parameters
- `W`: Recurrent weights (n × n)
- `W_in`: Input weights (n × n_in)
- `a_0`: Activation threshold — scalar or (n,)
- `log_tau_d`: Dendritic time constant — scalar or (n,)
- `log_tau_a_E_lo/hi`: SFA time constant endpoints (if n_a_E > 0)
- `log_c_E`: SFA coupling (if n_a_E > 0) — scalar or (n_E,)
- `log_tau_b_E_rec/rel`: STD time constants (if n_b_E > 0) — scalar or (n_E,)
- (and analogous I-neuron params)
"""
struct SRNN_ODE{F} <: Lux.AbstractLuxLayer
    n::Int
    n_in::Int
    n_E::Int
    n_I::Int
    n_a_E::Int
    n_a_I::Int
    n_b_E::Int
    n_b_I::Int
    activation::F
    state_dim::Int
    per_neuron::Bool
end

function SRNN_ODE(n::Int, n_in::Int, n_E::Int;
                  n_a_E::Int=0, n_a_I::Int=0,
                  n_b_E::Int=0, n_b_I::Int=0,
                  per_neuron::Bool=false,
                  activation=make_piecewise_sigmoid(S_c=0.0))
    n_I = n - n_E
    state_dim = n_E * n_a_E + n_I * n_a_I + n_E * n_b_E + n_I * n_b_I + n
    return SRNN_ODE(n, n_in, n_E, n_I, n_a_E, n_a_I, n_b_E, n_b_I,
                    activation, state_dim, per_neuron)
end

function Lux.initialparameters(rng::AbstractRNG, layer::SRNN_ODE)
    n, n_in, n_E, n_I = layer.n, layer.n_in, layer.n_E, layer.n_I
    pn = layer.per_neuron
    σ_w = Float32(sqrt(2.0 / n))

    # Helper: scalar or per-neuron/per-pop vector
    _s_or_v(val, dim) = pn ? fill(Float32(val), dim) : Float32[Float32(val)]

    # Inverse softplus: log(exp(x) - 1), so that softplus(result) = x
    _inv_sp(x) = Float32(log(exp(x) - 1))

    ps = Dict{Symbol, Any}(
        :W         => σ_w .* randn(rng, Float32, n, n),
        :W_in      => Float32(0.1) .* randn(rng, Float32, n, n_in),
        :a_0       => _s_or_v(0.35, n),               # MATLAB: S_c = 0.35
        :log_tau_d => _s_or_v(_inv_sp(0.1), n),        # τ_d = 0.1s
    )

    # SFA parameters for E neurons
    if layer.n_a_E > 0
        ps[:log_tau_a_E_lo] = _s_or_v(_inv_sp(0.25), n_E)  # τ = 0.25s
        ps[:log_tau_a_E_hi] = _s_or_v(_inv_sp(10.0), n_E)  # τ = 10.0s
        ps[:log_c_E] = _s_or_v(-3.0, n_E)     # softplus(-3) ≈ 0.049 ≈ MATLAB 0.15/3
    end

    # SFA parameters for I neurons
    if layer.n_a_I > 0
        ps[:log_tau_a_I_lo] = _s_or_v(_inv_sp(0.25), n_I)  # τ = 0.25s
        ps[:log_tau_a_I_hi] = _s_or_v(_inv_sp(10.0), n_I)  # τ = 10.0s
        ps[:log_c_I] = _s_or_v(-3.0, n_I)     # softplus(-3) ≈ 0.049
    end

    # STD parameters for E neurons
    if layer.n_b_E > 0
        ps[:log_tau_b_E_rec] = _s_or_v(_inv_sp(1.0), n_E)    # τ_rec = 1.0s
        ps[:log_tau_b_E_rel] = _s_or_v(_inv_sp(0.25), n_E)   # τ_rel = 0.25s
    end

    # STD parameters for I neurons
    if layer.n_b_I > 0
        ps[:log_tau_b_I_rec] = _s_or_v(_inv_sp(1.0), n_I)    # τ_rec = 1.0s
        ps[:log_tau_b_I_rel] = _s_or_v(_inv_sp(0.25), n_I)   # τ_rel = 0.25s
    end

    return NamedTuple(ps)
end

Lux.initialstates(::AbstractRNG, ::SRNN_ODE) = (input = nothing,)

# ── SRNN_ODE forward pass (vector — single sample) ─────────────────────

function (layer::SRNN_ODE)(S::AbstractVector, ps, st)
    n, n_E, n_I = layer.n, layer.n_E, layer.n_I
    n_a_E, n_a_I = layer.n_a_E, layer.n_a_I
    n_b_E, n_b_I = layer.n_b_E, layer.n_b_I

    # Unpack state
    parts = _unpack_state(layer, S)
    x = parts.x

    # Input
    u_raw = st.input
    if isnothing(u_raw)
        u_raw = zeros(eltype(S), layer.n_in)
    end
    u = ps.W_in * u_raw

    # Compute x_eff (with SFA) and apply activation with trainable a_0
    x_eff = _compute_x_eff(layer, parts, ps)
    r = layer.activation.(x_eff .- ps.a_0)

    # Apply STD
    b = _extract_b(layer, parts, S)
    br = b .* r

    # Compute derivatives
    τ_d = NNlib.softplus.(ps.log_tau_d)
    dx_dt = (-x .+ ps.W * br .+ u) ./ τ_d

    # Adaptation derivatives
    da_E_dt = if n_a_E > 0 && !isnothing(parts.a_E)
        τ_a_E = NNlib.softplus.(_make_tau_range(ps.log_tau_a_E_lo, ps.log_tau_a_E_hi, n_a_E))
        (r[1:n_E] .- parts.a_E) ./ τ_a_E
    else
        Float32[]
    end

    da_I_dt = if n_a_I > 0 && !isnothing(parts.a_I)
        τ_a_I = NNlib.softplus.(_make_tau_range(ps.log_tau_a_I_lo, ps.log_tau_a_I_hi, n_a_I))
        (r[n_E+1:n] .- parts.a_I) ./ τ_a_I
    else
        Float32[]
    end

    # STD derivatives
    db_E_dt = if n_b_E > 0 && !isnothing(parts.b_E)
        τ_b_rec = NNlib.softplus.(ps.log_tau_b_E_rec)
        τ_b_rel = NNlib.softplus.(ps.log_tau_b_E_rel)
        (1.0f0 .- parts.b_E) ./ τ_b_rec .- (parts.b_E .* r[1:n_E]) ./ τ_b_rel
    else
        Float32[]
    end

    db_I_dt = if n_b_I > 0 && !isnothing(parts.b_I)
        τ_b_rec = NNlib.softplus.(ps.log_tau_b_I_rec)
        τ_b_rel = NNlib.softplus.(ps.log_tau_b_I_rel)
        (1.0f0 .- parts.b_I) ./ τ_b_rec .- (parts.b_I .* r[n_E+1:n]) ./ τ_b_rel
    else
        Float32[]
    end

    dS_dt = vcat(vec(da_E_dt), vec(da_I_dt), db_E_dt, db_I_dt, dx_dt)
    return dS_dt, st
end

# ── SRNN_ODE forward pass (batched — S is state_dim × B) ───────────────

function (layer::SRNN_ODE)(S::AbstractMatrix, ps, st)
    n, n_E, n_I = layer.n, layer.n_E, layer.n_I
    n_a_E, n_a_I = layer.n_a_E, layer.n_a_I
    n_b_E, n_b_I = layer.n_b_E, layer.n_b_I
    B = size(S, 2)

    # Unpack state
    parts = _unpack_state(layer, S)
    x = parts.x

    # Input
    u_raw = st.input
    if isnothing(u_raw)
        u_raw = zeros(eltype(S), layer.n_in, B)
    end
    u = ps.W_in * u_raw   # (n, n_in) × (n_in, B) → (n, B)

    # Compute x_eff (with SFA) and apply activation with trainable a_0
    x_eff = _compute_x_eff(layer, parts, ps)
    r = layer.activation.(x_eff .- ps.a_0)   # a_0 broadcasts: (1,) or (n,) vs (n, B)

    # Apply STD
    b = _extract_b(layer, parts, S)
    br = b .* r

    # Compute derivatives
    τ_d = NNlib.softplus.(ps.log_tau_d)    # (1,) or (n,)
    dx_dt = (-x .+ ps.W * br .+ u) ./ τ_d

    # Adaptation derivatives for E neurons
    da_E_dt = if n_a_E > 0 && !isnothing(parts.a_E)
        # τ_a_E: (1, n_a_E) or (n_E, n_a_E), need (…, 1) for batch broadcast
        τ_a_E_2d = NNlib.softplus.(_make_tau_range(ps.log_tau_a_E_lo, ps.log_tau_a_E_hi, n_a_E))
        τ_a_E_3 = reshape(τ_a_E_2d, size(τ_a_E_2d, 1), size(τ_a_E_2d, 2), 1)
        r_E = reshape(r[1:n_E, :], n_E, 1, B)   # (n_E, 1, B)
        (r_E .- parts.a_E) ./ τ_a_E_3            # (n_E, n_a_E, B)
    else
        zeros(Float32, 0, B)
    end

    da_I_dt = if n_a_I > 0 && !isnothing(parts.a_I)
        τ_a_I_2d = NNlib.softplus.(_make_tau_range(ps.log_tau_a_I_lo, ps.log_tau_a_I_hi, n_a_I))
        τ_a_I_3 = reshape(τ_a_I_2d, size(τ_a_I_2d, 1), size(τ_a_I_2d, 2), 1)
        r_I = reshape(r[n_E+1:n, :], n_I, 1, B)
        (r_I .- parts.a_I) ./ τ_a_I_3
    else
        zeros(Float32, 0, B)
    end

    # STD derivatives
    db_E_dt = if n_b_E > 0 && !isnothing(parts.b_E)
        τ_b_rec = NNlib.softplus.(ps.log_tau_b_E_rec)
        τ_b_rel = NNlib.softplus.(ps.log_tau_b_E_rel)
        (1.0f0 .- parts.b_E) ./ τ_b_rec .- (parts.b_E .* r[1:n_E, :]) ./ τ_b_rel
    else
        zeros(Float32, 0, B)
    end

    db_I_dt = if n_b_I > 0 && !isnothing(parts.b_I)
        τ_b_rec = NNlib.softplus.(ps.log_tau_b_I_rec)
        τ_b_rel = NNlib.softplus.(ps.log_tau_b_I_rel)
        (1.0f0 .- parts.b_I) ./ τ_b_rec .- (parts.b_I .* r[n_E+1:n, :]) ./ τ_b_rel
    else
        zeros(Float32, 0, B)
    end

    # Pack: flatten 3D adaptation arrays to (n_E*n_a_E, B) before vcat
    da_E_flat = n_a_E > 0 ? reshape(da_E_dt, n_E * n_a_E, B) : zeros(Float32, 0, B)
    da_I_flat = n_a_I > 0 ? reshape(da_I_dt, n_I * n_a_I, B) : zeros(Float32, 0, B)

    dS_dt = vcat(da_E_flat, da_I_flat, db_E_dt, db_I_dt, dx_dt)
    return dS_dt, st
end

# ═══════════════════════════════════════════════════════════════════════
# INITIAL STATE
# ═══════════════════════════════════════════════════════════════════════

"""
    srnn_initial_state(layer::SRNN_ODE; rng=Random.default_rng())

Generate a default initial state vector for the SRNN ODE.
"""
function srnn_initial_state(layer::SRNN_ODE; rng=Random.default_rng())
    a_E_0 = zeros(Float32, layer.n_E * layer.n_a_E)
    a_I_0 = zeros(Float32, layer.n_I * layer.n_a_I)
    b_E_0 = ones(Float32, layer.n_E * layer.n_b_E)
    b_I_0 = ones(Float32, layer.n_I * layer.n_b_I)
    x_0   = Float32(0.1) .* randn(rng, Float32, layer.n)
    return vcat(a_E_0, a_I_0, b_E_0, b_I_0, x_0)
end

"""
    srnn_initial_state(layer::SRNN_ODE, B::Int; rng=Random.default_rng())

Batched initial state: (state_dim, B) matrix.
"""
function srnn_initial_state(layer::SRNN_ODE, B::Int; rng=Random.default_rng())
    a_E_0 = zeros(Float32, layer.n_E * layer.n_a_E, B)
    a_I_0 = zeros(Float32, layer.n_I * layer.n_a_I, B)
    b_E_0 = ones(Float32, layer.n_E * layer.n_b_E, B)
    b_I_0 = ones(Float32, layer.n_I * layer.n_b_I, B)
    x_0   = Float32(0.1) .* randn(rng, Float32, layer.n, B)
    return vcat(a_E_0, a_I_0, b_E_0, b_I_0, x_0)
end

# ═══════════════════════════════════════════════════════════════════════
# SRNNCell — fused Euler sub-stepping wrapper
# ═══════════════════════════════════════════════════════════════════════

"""
    SRNNCell(n, n_in, n_E; ode_solver_unfolds=6, h=0.1f0,
             readout=:synaptic, kwargs...)

Wrapper around SRNN_ODE that applies N Euler sub-steps and returns next state.
Drop-in replacement for LTCODE1 in training pipelines.

# Readout modes (used by `readout(cell, S, ps)`)
- `:dendritic` — raw dendritic potential x
- `:rate`      — firing rate r = φ(x_eff - a_0)
- `:synaptic`  — synaptic output b·r (default)
"""
struct SRNNCell{F} <: Lux.AbstractLuxLayer
    ode::SRNN_ODE{F}
    ode_solver_unfolds::Int
    h::Float32
    readout_mode::Symbol
end

function SRNNCell(n::Int, n_in::Int, n_E::Int;
                  ode_solver_unfolds::Int=6, h::Float32=0.1f0,
                  readout::Symbol=:synaptic, kwargs...)
    ode = SRNN_ODE(n, n_in, n_E; kwargs...)
    SRNNCell(ode, ode_solver_unfolds, h, readout)
end

# Delegate Lux interface
Lux.initialparameters(rng::AbstractRNG, cell::SRNNCell) = Lux.initialparameters(rng, cell.ode)
Lux.initialstates(rng::AbstractRNG, cell::SRNNCell) = Lux.initialstates(rng, cell.ode)

# Property accessors for interface compatibility
function Base.getproperty(c::SRNNCell, s::Symbol)
    if s in (:n, :n_in, :n_E, :n_I, :n_a_E, :n_a_I, :n_b_E, :n_b_I, :state_dim, :per_neuron, :activation)
        return getfield(getfield(c, :ode), s)
    else
        return getfield(c, s)
    end
end

# Delegate initial state
srnn_initial_state(cell::SRNNCell, args...; kwargs...) = srnn_initial_state(cell.ode, args...; kwargs...)

# ── SRNNCell call — fused Euler sub-stepping ────────────────────────────
# Works for both vector S and batched matrix S (dispatch handled by SRNN_ODE)

function (cell::SRNNCell)(S, ps, st)
    S_out = S
    for _ in 1:cell.ode_solver_unfolds
        dS, _ = cell.ode(S_out, ps, st)
        S_out = S_out .+ cell.h .* dS
    end
    return S_out, st
end

# ═══════════════════════════════════════════════════════════════════════
# READOUT — extract observation from full state
# ═══════════════════════════════════════════════════════════════════════

"""
    readout(cell::SRNNCell, S, ps) → (n,) or (n, B)

Extract the observation from the full SRNN state S.
- `:dendritic` → x (dendritic potential)
- `:rate`      → r = φ(x_eff - a_0) (firing rate after adaptation)
- `:synaptic`  → b·r (synaptic output after STD) [DEFAULT]
"""
function readout(cell::SRNNCell, S, ps)
    layer = cell.ode
    parts = _unpack_state(layer, S)

    if cell.readout_mode == :dendritic
        return parts.x
    end

    # Compute firing rate
    x_eff = _compute_x_eff(layer, parts, ps)
    r = layer.activation.(x_eff .- ps.a_0)

    if cell.readout_mode == :rate
        return r
    end

    # :synaptic → b .* r
    b = _extract_b(layer, parts, S)
    return b .* r
end
