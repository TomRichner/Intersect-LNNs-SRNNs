# ltc1.jl — LTC ODE layer faithful to Hasani's Python training code
#
# Faithful port of:
#   liquid_time_constant_networks/experiments_with_ltcs/ltc_model.py
#
# Key differences from ltc.jl (simplified form):
#   1. Per-synapse sigmoid gates: σ(σ_ji · (v_j − μ_ji))
#   2. Non-negative weights via softplus(W_raw)
#   3. Separate sensory (input) and recurrent pathways
#   4. N×N reversal potential matrices (±1 init)
#   5. Conductance-based model: cm, gleak, vleak
#   6. Input mapping: learnable affine transform per channel
#
# Continuous ODE:
#   cm_i · dv_i/dt = g_leak_i · (v_leak_i − v_i)
#                   + Σ_j g_rec_ji · (E_rec_ji − v_i)
#                   + Σ_k g_sens_ki · (E_sens_ki − v_i)
#
# where g_rec_ji = softplus(W_raw_ji) · σ(σ_ji · (v_j − μ_ji))
#
# Fused semi-implicit solver:
#   v_i ← (cm_i·v_i + g_leak_i·v_leak_i + Σ g_ji·E_ji) /
#          (cm_i + g_leak_i + Σ g_ji)

using Lux, Random, NNlib

"""
    LTCODE1(n, n_in; ode_solver_unfolds=6, solver=:semi_implicit)

LTC ODE layer faithful to Hasani's Python training code (ltc_model.py).
Per-synapse conductance-based model with separate sensory and recurrent pathways.

# Arguments
- `n::Int`: Number of hidden neurons
- `n_in::Int`: Input dimension (before affine mapping)
- `ode_solver_unfolds::Int`: Number of ODE sub-steps per call (default: 6)
- `solver::Symbol`: `:semi_implicit` (default), `:explicit`, or `:runge_kutta`

# Trainable Parameters
- `W_raw`: Recurrent weights (N×N) — W = softplus(W_raw) ≥ 0
- `mu`: Recurrent gate centers (N×N)
- `sigma`: Recurrent gate widths (N×N)
- `erev`: Recurrent reversal potentials (N×N)
- `sensory_W_raw`: Sensory weights (M×N) — sensory_W = softplus(...) ≥ 0
- `sensory_mu`: Sensory gate centers (M×N)
- `sensory_sigma`: Sensory gate widths (M×N)
- `sensory_erev`: Sensory reversal potentials (M×N)
- `vleak`: Leak reversal potential (N)
- `gleak_raw`: Leak conductance (N) — gleak = softplus(...)
- `cm_raw`: Membrane capacitance (N) — cm = softplus(...)
- `input_w`: Input affine weight (M)
- `input_b`: Input affine bias (M)

# Usage
```julia
layer = LTCODE1(32, 4)
ps, st = Lux.setup(rng, layer)
st_driven = merge(st, (input=u_t,))
# Fused step returns next state (not dxdt!)
v_next, st = layer(v, ps, st_driven)
```

Note: Unlike LTCODE/LTCODE2, this layer returns the **next state** (fused solver),
not dx/dt. For continuous ODE integration, use `ltc1_ode_rhs` instead.
"""
struct LTCODE1{S} <: Lux.AbstractLuxLayer
    n::Int
    n_in::Int
    ode_solver_unfolds::Int
    solver::S
end

function LTCODE1(n::Int, n_in::Int; ode_solver_unfolds::Int=6, solver::Symbol=:semi_implicit)
    LTCODE1(n, n_in, ode_solver_unfolds, solver)
end

# Helper: per-synapse sigmoid gate
# σ(σ_ji · (v_j − μ_ji))
# v_pre: (n,) or (n, batch)   →  reshaped to (n, 1) for broadcasting
# mu:    (n, n) or (m, n)
# sigma: (n, n) or (m, n)
# Returns: (n, n) or (m, n) matrix of gate activations
function _per_synapse_sigmoid(v_pre::AbstractVector, mu, sigma)
    # v_pre is (n,), reshape to (n, 1) for broadcasting against (n, n)
    v = reshape(v_pre, :, 1)  # (n, 1)
    return NNlib.sigmoid.(sigma .* (v .- mu))  # (n, n) or (m, n)
end

# ── Batched: v_pre is (P, B) → result (P, N, B) ─────────────────────────
function _per_synapse_sigmoid(v_pre::AbstractMatrix, mu, sigma)
    # v_pre: (P, B), mu/sigma: (P, N)
    # reshape for 3D broadcast: v → (P, 1, B), mu/sigma → (P, N, 1)
    P, B = size(v_pre)
    v3 = reshape(v_pre, P, 1, B)           # (P, 1, B)
    mu3 = reshape(mu, size(mu, 1), size(mu, 2), 1)     # (P, N, 1)
    sig3 = reshape(sigma, size(sigma, 1), size(sigma, 2), 1) # (P, N, 1)
    return NNlib.sigmoid.(sig3 .* (v3 .- mu3))  # (P, N, B)
end

function Lux.initialparameters(rng::AbstractRNG, layer::LTCODE1)
    n, n_in = layer.n, layer.n_in

    # Initialize in unconstrained space where needed (softplus inverse)
    # softplus(x) ≈ x for x >> 0, inv_softplus(y) = log(exp(y) - 1)
    _inv_sp(y) = y > 20.0 ? Float32(y) : Float32(log(exp(y) - 1.0))

    # Recurrent weights: uniform [0.01, 1.0] → softplus space
    W_init = Float32.(rand(rng, n, n) .* 0.99 .+ 0.01)
    W_raw = _inv_sp.(W_init)

    # Sensory weights: same
    sensory_W_init = Float32.(rand(rng, n_in, n) .* 0.99 .+ 0.01)
    sensory_W_raw = _inv_sp.(sensory_W_init)

    # Reversal potentials: random ±1
    erev = Float32.(2 .* (rand(rng, Int8, n, n) .% 2) .- 1)
    # Fix: use proper random ±1
    erev = Float32.([rand(rng, Bool) ? 1.0 : -1.0 for _ in 1:n, _ in 1:n])
    sensory_erev = Float32.([rand(rng, Bool) ? 1.0 : -1.0 for _ in 1:n_in, _ in 1:n])

    return (
        # Recurrent
        W_raw   = W_raw,
        mu      = Float32.(rand(rng, n, n) .* 0.5 .+ 0.3),      # [0.3, 0.8]
        sigma   = Float32.(rand(rng, n, n) .* 5.0 .+ 3.0),      # [3.0, 8.0]
        erev    = erev,
        # Sensory
        sensory_W_raw   = sensory_W_raw,
        sensory_mu      = Float32.(rand(rng, n_in, n) .* 0.5 .+ 0.3),
        sensory_sigma   = Float32.(rand(rng, n_in, n) .* 5.0 .+ 3.0),
        sensory_erev    = sensory_erev,
        # Leak
        vleak     = Float32.(rand(rng, n) .* 0.4 .- 0.2),       # [-0.2, 0.2]
        gleak_raw = Float32[_inv_sp(1.0f0) for _ in 1:n],       # gleak ≈ 1.0
        cm_raw    = Float32[_inv_sp(0.5f0) for _ in 1:n],       # cm ≈ 0.5
        # Input mapping
        input_w   = ones(Float32, n_in),
        input_b   = zeros(Float32, n_in),
    )
end

Lux.initialstates(::AbstractRNG, ::LTCODE1) = (input = nothing,)
Lux.statelength(::LTCODE1) = 0

function Lux.parameterlength(layer::LTCODE1)
    n, m = layer.n, layer.n_in
    # Recurrent: W(n²) + mu(n²) + sigma(n²) + erev(n²) = 4n²
    # Sensory: W(mn) + mu(mn) + sigma(mn) + erev(mn) = 4mn
    # Leak: vleak(n) + gleak(n) + cm(n) = 3n
    # Input: w(m) + b(m) = 2m
    return 4 * n * n + 4 * m * n + 3 * n + 2 * m
end

# ── Map inputs through affine transform ─────────────────────────────────
function _map_inputs(inputs::AbstractVector, ps)
    return ps.input_w .* inputs .+ ps.input_b
end

# Batched: inputs is (M, B)
function _map_inputs(inputs::AbstractMatrix, ps)
    # ps.input_w is (M,), broadcasts against (M, B)
    return ps.input_w .* inputs .+ ps.input_b
end

# ── Semi-implicit fused solver (one step) ────────────────────────────────
function _fused_step(v_pre::AbstractVector, inputs_mapped, ps)
    # Constrain parameters
    W = NNlib.softplus.(ps.W_raw)
    sensory_W = NNlib.softplus.(ps.sensory_W_raw)
    gleak = NNlib.softplus.(ps.gleak_raw)
    cm = NNlib.softplus.(ps.cm_raw)

    # Sensory pathway: per-synapse gates (M × N)
    sensory_gate = _per_synapse_sigmoid(inputs_mapped, ps.sensory_mu, ps.sensory_sigma)
    sensory_w_act = sensory_W .* sensory_gate                # (M × N)
    sensory_rev_act = sensory_w_act .* ps.sensory_erev       # (M × N)

    w_num_sensory = vec(sum(sensory_rev_act, dims=1))        # (N,)
    w_den_sensory = vec(sum(sensory_w_act, dims=1))          # (N,)

    # Recurrent pathway: per-synapse gates (N × N)
    w_gate = _per_synapse_sigmoid(v_pre, ps.mu, ps.sigma)
    w_act = W .* w_gate                                       # (N × N)
    rev_act = w_act .* ps.erev                                # (N × N)

    w_num = vec(sum(rev_act, dims=1)) .+ w_num_sensory       # (N,)
    w_den = vec(sum(w_act, dims=1)) .+ w_den_sensory         # (N,)

    # Fused update: v ← (cm·v + gleak·vleak + w_num) / (cm + gleak + w_den)
    numerator = cm .* v_pre .+ gleak .* ps.vleak .+ w_num
    denominator = cm .+ gleak .+ w_den

    return numerator ./ denominator
end

# ── Batched semi-implicit fused solver ───────────────────────────────────
# v_pre: (N, B), inputs_mapped: (M, B)
# Returns: (N, B)
function _fused_step(v_pre::AbstractMatrix, inputs_mapped::AbstractMatrix, ps)
    W = NNlib.softplus.(ps.W_raw)             # (N, N)
    sensory_W = NNlib.softplus.(ps.sensory_W_raw) # (M, N)
    gleak = NNlib.softplus.(ps.gleak_raw)     # (N,)
    cm = NNlib.softplus.(ps.cm_raw)           # (N,)

    # Sensory pathway: gates are (M, N, B)
    sensory_gate = _per_synapse_sigmoid(inputs_mapped, ps.sensory_mu, ps.sensory_sigma)
    # sensory_W: (M, N) → (M, N, 1) for broadcast with (M, N, B)
    sW3 = reshape(sensory_W, size(sensory_W, 1), size(sensory_W, 2), 1)
    sensory_w_act = sW3 .* sensory_gate                                # (M, N, B)
    sE3 = reshape(ps.sensory_erev, size(ps.sensory_erev, 1), size(ps.sensory_erev, 2), 1)
    sensory_rev_act = sensory_w_act .* sE3                             # (M, N, B)

    w_num_sensory = dropdims(sum(sensory_rev_act, dims=1), dims=1)     # (N, B)
    w_den_sensory = dropdims(sum(sensory_w_act, dims=1), dims=1)       # (N, B)

    # Recurrent pathway: gates are (N, N, B)
    w_gate = _per_synapse_sigmoid(v_pre, ps.mu, ps.sigma)
    W3 = reshape(W, size(W, 1), size(W, 2), 1)
    w_act = W3 .* w_gate                                               # (N, N, B)
    E3 = reshape(ps.erev, size(ps.erev, 1), size(ps.erev, 2), 1)
    rev_act = w_act .* E3                                               # (N, N, B)

    w_num = dropdims(sum(rev_act, dims=1), dims=1) .+ w_num_sensory   # (N, B)
    w_den = dropdims(sum(w_act, dims=1), dims=1) .+ w_den_sensory     # (N, B)

    # Fused update — (N,) params broadcast against (N, B)
    numerator = cm .* v_pre .+ gleak .* ps.vleak .+ w_num
    denominator = cm .+ gleak .+ w_den

    return numerator ./ denominator
end

# ── Explicit Euler (one step) ────────────────────────────────────────────
function _f_prime(v_pre::AbstractVector, inputs_mapped, ps)
    W = NNlib.softplus.(ps.W_raw)
    sensory_W = NNlib.softplus.(ps.sensory_W_raw)
    gleak = NNlib.softplus.(ps.gleak_raw)
    cm = NNlib.softplus.(ps.cm_raw)

    # Sensory
    sensory_gate = _per_synapse_sigmoid(inputs_mapped, ps.sensory_mu, ps.sensory_sigma)
    sensory_w_act = sensory_W .* sensory_gate
    w_reduced_sensory = vec(sum(sensory_w_act, dims=1))
    sensory_in = ps.sensory_erev .* sensory_w_act

    # Recurrent
    w_gate = _per_synapse_sigmoid(v_pre, ps.mu, ps.sigma)
    w_act = W .* w_gate
    w_reduced_synapse = vec(sum(w_act, dims=1))
    synapse_in = ps.erev .* w_act

    sum_in = vec(sum(sensory_in, dims=1)) .- v_pre .* w_reduced_synapse .+
             vec(sum(synapse_in, dims=1)) .- v_pre .* w_reduced_sensory

    f_prime = (1.0f0 ./ cm) .* (gleak .* (ps.vleak .- v_pre) .+ sum_in)
    return f_prime
end

# ── Batched Explicit Euler ───────────────────────────────────────────────
# v_pre: (N, B), inputs_mapped: (M, B)
# Returns: (N, B)
function _f_prime(v_pre::AbstractMatrix, inputs_mapped::AbstractMatrix, ps)
    W = NNlib.softplus.(ps.W_raw)
    sensory_W = NNlib.softplus.(ps.sensory_W_raw)
    gleak = NNlib.softplus.(ps.gleak_raw)
    cm = NNlib.softplus.(ps.cm_raw)

    # Sensory: (M, N, B)
    sensory_gate = _per_synapse_sigmoid(inputs_mapped, ps.sensory_mu, ps.sensory_sigma)
    sW3 = reshape(sensory_W, size(sensory_W, 1), size(sensory_W, 2), 1)
    sensory_w_act = sW3 .* sensory_gate
    w_reduced_sensory = dropdims(sum(sensory_w_act, dims=1), dims=1)    # (N, B)
    sE3 = reshape(ps.sensory_erev, size(ps.sensory_erev, 1), size(ps.sensory_erev, 2), 1)
    sensory_in = sE3 .* sensory_w_act                                   # (M, N, B)

    # Recurrent: (N, N, B)
    w_gate = _per_synapse_sigmoid(v_pre, ps.mu, ps.sigma)
    W3 = reshape(W, size(W, 1), size(W, 2), 1)
    w_act = W3 .* w_gate
    w_reduced_synapse = dropdims(sum(w_act, dims=1), dims=1)            # (N, B)
    E3 = reshape(ps.erev, size(ps.erev, 1), size(ps.erev, 2), 1)
    synapse_in = E3 .* w_act                                            # (N, N, B)

    sum_in = dropdims(sum(sensory_in, dims=1), dims=1) .- v_pre .* w_reduced_synapse .+
             dropdims(sum(synapse_in, dims=1), dims=1) .- v_pre .* w_reduced_sensory

    f_prime = (1.0f0 ./ cm) .* (gleak .* (ps.vleak .- v_pre) .+ sum_in)
    return f_prime
end

# ── Main call: dispatches to solver (single sample) ─────────────────────
function (layer::LTCODE1)(v::AbstractVector, ps, st)
    u = st.input
    if isnothing(u)
        u = zeros(eltype(v), layer.n_in)
    end

    inputs_mapped = _map_inputs(u, ps)

    if layer.solver === :semi_implicit
        v_out = v
        for _ in 1:layer.ode_solver_unfolds
            v_out = _fused_step(v_out, inputs_mapped, ps)
        end
    elseif layer.solver === :explicit
        v_out = v
        h = 0.1f0
        for _ in 1:layer.ode_solver_unfolds
            v_out = v_out .+ h .* _f_prime(v_out, inputs_mapped, ps)
        end
    elseif layer.solver === :runge_kutta
        v_out = v
        h = 0.1f0
        for _ in 1:layer.ode_solver_unfolds
            k1 = h .* _f_prime(v_out, inputs_mapped, ps)
            k2 = h .* _f_prime(v_out .+ 0.5f0 .* k1, inputs_mapped, ps)
            k3 = h .* _f_prime(v_out .+ 0.5f0 .* k2, inputs_mapped, ps)
            k4 = h .* _f_prime(v_out .+ k3, inputs_mapped, ps)
            v_out = v_out .+ (1.0f0 / 6.0f0) .* (k1 .+ 2.0f0 .* k2 .+ 2.0f0 .* k3 .+ k4)
        end
    else
        error("Unknown solver: $(layer.solver). Use :semi_implicit, :explicit, or :runge_kutta")
    end

    return v_out, st
end

# ── Main call: batched (v is N×B matrix) ─────────────────────────────────
function (layer::LTCODE1)(v::AbstractMatrix, ps, st)
    u = st.input   # (M, B) or nothing
    if isnothing(u)
        B = size(v, 2)
        u = zeros(eltype(v), layer.n_in, B)
    end

    inputs_mapped = _map_inputs(u, ps)

    if layer.solver === :semi_implicit
        v_out = v
        for _ in 1:layer.ode_solver_unfolds
            v_out = _fused_step(v_out, inputs_mapped, ps)
        end
    elseif layer.solver === :explicit
        v_out = v
        h = 0.1f0
        for _ in 1:layer.ode_solver_unfolds
            v_out = v_out .+ h .* _f_prime(v_out, inputs_mapped, ps)
        end
    elseif layer.solver === :runge_kutta
        v_out = v
        h = 0.1f0
        for _ in 1:layer.ode_solver_unfolds
            k1 = h .* _f_prime(v_out, inputs_mapped, ps)
            k2 = h .* _f_prime(v_out .+ 0.5f0 .* k1, inputs_mapped, ps)
            k3 = h .* _f_prime(v_out .+ 0.5f0 .* k2, inputs_mapped, ps)
            k4 = h .* _f_prime(v_out .+ k3, inputs_mapped, ps)
            v_out = v_out .+ (1.0f0 / 6.0f0) .* (k1 .+ 2.0f0 .* k2 .+ 2.0f0 .* k3 .+ k4)
        end
    else
        error("Unknown solver: $(layer.solver). Use :semi_implicit, :explicit, or :runge_kutta")
    end

    return v_out, st
end

"""
    ltc1_ode_rhs(v, ps, inputs_mapped)

Continuous ODE right-hand side for LTCODE1, for use with DiffEqFlux/OrdinaryDiffEq.
Returns dv/dt (not next state).
"""
function ltc1_ode_rhs(v::AbstractVector, ps, inputs_mapped)
    return _f_prime(v, inputs_mapped, ps)
end

"""
    readout(::LTCODE1, v, ps) → v

Identity readout for LTCODE1 — the hidden state IS the readout.
Provides a uniform interface with SRNNCell's readout function.
"""
readout(::LTCODE1, v, ps) = v
