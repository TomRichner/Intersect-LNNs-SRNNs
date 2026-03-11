# srnn.jl — Stable Recurrent Neural Network ODE layer (port of SRNNModel2.m)
#
# Implements the SRNN dynamics with optional Spike-Frequency Adaptation (SFA)
# and Short-Term Depression (STD). With n_a=0, n_b=0, this reduces to a
# vanilla Hopfield/rate RNN baseline.
#
# State layout: S = [a_E; a_I; b_E; b_I; x]
# All key parameters are trainable via Lux.

using Lux, Random, NNlib

include(joinpath(@__DIR__, "..", "activations.jl"))

"""
    SRNN_ODE(n, n_in, n_E; n_a_E=0, n_a_I=0, n_b_E=0, n_b_I=0,
             activation=make_piecewise_sigmoid())

Stable Recurrent Neural Network ODE layer. Defines the right-hand side of the
SRNN ODE system as a Lux layer with trainable parameters.

# State layout
The augmented state vector is: `S = [a_E(:); a_I(:); b_E(:); b_I(:); x]`
- `a_E`: SFA adaptation for E neurons (n_E × n_a_E)
- `a_I`: SFA adaptation for I neurons (n_I × n_a_I)
- `b_E`: STD depression for E neurons (n_E × n_b_E)
- `b_I`: STD depression for I neurons (n_I × n_b_I)
- `x`: Dendritic state (n)

# Hopfield mode
Set `n_a_E=0, n_a_I=0, n_b_E=0, n_b_I=0` and use `activation=tanh` for a
vanilla rate RNN baseline.

# Trainable Parameters (in `ps`)
- `W`: Recurrent weights (n × n)
- `W_in`: Input weights (n × n_in)
- `log_tau_d`: Log dendritic time constant (scalar) — τ_d = softplus(log_tau_d)
- `log_tau_a_E`: Log adaptation time constants for E (n_a_E) (if n_a_E > 0)
- `log_tau_a_I`: Log adaptation time constants for I (n_a_I) (if n_a_I > 0)
- `log_c_E`: Log adaptation scaling for E (scalar) (if n_a_E > 0)
- `log_c_I`: Log adaptation scaling for I (scalar) (if n_a_I > 0)
- `log_tau_b_E_rec`, `log_tau_b_E_rel`: Log STD time constants for E (if n_b_E > 0)
- `log_tau_b_I_rec`, `log_tau_b_I_rel`: Log STD time constants for I (if n_b_I > 0)
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
    state_dim::Int  # total state dimension
end

function SRNN_ODE(n::Int, n_in::Int, n_E::Int;
                  n_a_E::Int=0, n_a_I::Int=0,
                  n_b_E::Int=0, n_b_I::Int=0,
                  activation=make_piecewise_sigmoid())
    n_I = n - n_E
    state_dim = n_E * n_a_E + n_I * n_a_I + n_E * n_b_E + n_I * n_b_I + n
    return SRNN_ODE(n, n_in, n_E, n_I, n_a_E, n_a_I, n_b_E, n_b_I,
                    activation, state_dim)
end

function Lux.initialparameters(rng::AbstractRNG, layer::SRNN_ODE)
    n, n_in = layer.n, layer.n_in
    σ_w = Float32(sqrt(2.0 / n))

    ps = Dict{Symbol, Any}(
        :W         => σ_w .* randn(rng, Float32, n, n),
        :W_in      => Float32(0.1) .* randn(rng, Float32, n, n_in),
        :log_tau_d => Float32[0.0],  # softplus(0) ≈ 0.693
    )

    # SFA parameters for E neurons
    if layer.n_a_E > 0
        # Default τ_a_E: log-spaced from ~0.2 to ~2.0 seconds
        ps[:log_tau_a_E] = Float32.(range(-0.5, 1.0, length=layer.n_a_E))
        ps[:log_c_E] = Float32[-3.0]  # softplus(-3) ≈ 0.049 ≈ c_E default
    end

    # SFA parameters for I neurons
    if layer.n_a_I > 0
        ps[:log_tau_a_I] = Float32.(range(-0.5, 1.0, length=layer.n_a_I))
        ps[:log_c_I] = Float32[-3.0]
    end

    # STD parameters for E neurons
    if layer.n_b_E > 0
        ps[:log_tau_b_E_rec] = Float32[0.0]   # τ_rec ≈ 0.7s
        ps[:log_tau_b_E_rel] = Float32[-1.0]   # τ_rel ≈ 0.3s
    end

    # STD parameters for I neurons
    if layer.n_b_I > 0
        ps[:log_tau_b_I_rec] = Float32[0.0]
        ps[:log_tau_b_I_rel] = Float32[-1.0]
    end

    return NamedTuple(ps)
end

Lux.initialstates(::AbstractRNG, ::SRNN_ODE) = (input = nothing,)

function (layer::SRNN_ODE)(S::AbstractVector, ps, st)
    n = layer.n
    n_E = layer.n_E
    n_I = layer.n_I
    n_a_E = layer.n_a_E
    n_a_I = layer.n_a_I
    n_b_E = layer.n_b_E
    n_b_I = layer.n_b_I

    # --- Unpack state ---
    idx = 0

    # Adaptation states for E neurons
    len_a_E = n_E * n_a_E
    a_E = len_a_E > 0 ? reshape(S[idx+1:idx+len_a_E], n_E, n_a_E) : nothing
    idx += len_a_E

    # Adaptation states for I neurons
    len_a_I = n_I * n_a_I
    a_I = len_a_I > 0 ? reshape(S[idx+1:idx+len_a_I], n_I, n_a_I) : nothing
    idx += len_a_I

    # STD states for E neurons
    len_b_E = n_E * n_b_E
    b_E = len_b_E > 0 ? S[idx+1:idx+len_b_E] : nothing
    idx += len_b_E

    # STD states for I neurons
    len_b_I = n_I * n_b_I
    b_I = len_b_I > 0 ? S[idx+1:idx+len_b_I] : nothing
    idx += len_b_I

    # Dendritic state
    x = S[idx+1:idx+n]

    # --- Get input ---
    u_raw = st.input
    if isnothing(u_raw)
        u_raw = zeros(eltype(S), layer.n_in)
    end
    u = ps.W_in * u_raw

    # --- Compute effective x (with adaptation) ---
    x_eff = copy(x)

    # Apply SFA to E neurons
    if n_a_E > 0 && !isnothing(a_E)
        c_E = NNlib.softplus(ps.log_c_E[1])
        x_eff_E = x_eff[1:n_E] .- c_E .* vec(sum(a_E, dims=2))
        x_eff = vcat(x_eff_E, x_eff[n_E+1:end])
    end

    # Apply SFA to I neurons
    if n_a_I > 0 && !isnothing(a_I)
        c_I = NNlib.softplus(ps.log_c_I[1])
        x_eff_I = x_eff[n_E+1:n] .- c_I .* vec(sum(a_I, dims=2))
        x_eff = vcat(x_eff[1:n_E], x_eff_I)
    end

    # --- Compute firing rates ---
    r = layer.activation.(x_eff)

    # Apply STD: effective rate is b .* r
    b = ones(eltype(S), n)
    if n_b_E > 0 && !isnothing(b_E)
        b = vcat(b_E, b[n_E+1:end])
    end
    if n_b_I > 0 && !isnothing(b_I)
        b = vcat(b[1:n_E], b_I)
    end

    br = b .* r  # synaptic output

    # --- Compute derivatives ---
    τ_d = NNlib.softplus(ps.log_tau_d[1])
    dx_dt = (-x .+ ps.W * br .+ u) ./ τ_d

    # Adaptation derivatives
    da_E_dt = if n_a_E > 0 && !isnothing(a_E)
        τ_a_E = NNlib.softplus.(ps.log_tau_a_E)'  # 1 × n_a_E for broadcasting
        (r[1:n_E] .- a_E) ./ τ_a_E
    else
        Float32[]
    end

    da_I_dt = if n_a_I > 0 && !isnothing(a_I)
        τ_a_I = NNlib.softplus.(ps.log_tau_a_I)'
        (r[n_E+1:n] .- a_I) ./ τ_a_I
    else
        Float32[]
    end

    # STD derivatives
    db_E_dt = if n_b_E > 0 && !isnothing(b_E)
        τ_b_rec = NNlib.softplus(ps.log_tau_b_E_rec[1])
        τ_b_rel = NNlib.softplus(ps.log_tau_b_E_rel[1])
        (1.0f0 .- b_E) ./ τ_b_rec .- (b_E .* r[1:n_E]) ./ τ_b_rel
    else
        Float32[]
    end

    db_I_dt = if n_b_I > 0 && !isnothing(b_I)
        τ_b_rec = NNlib.softplus(ps.log_tau_b_I_rec[1])
        τ_b_rel = NNlib.softplus(ps.log_tau_b_I_rel[1])
        (1.0f0 .- b_I) ./ τ_b_rec .- (b_I .* r[n_E+1:n]) ./ τ_b_rel
    else
        Float32[]
    end

    # --- Pack derivatives (same order as state) ---
    dS_dt = vcat(vec(da_E_dt), vec(da_I_dt), db_E_dt, db_I_dt, dx_dt)

    return dS_dt, st
end

"""
    srnn_initial_state(layer::SRNN_ODE; rng=Random.default_rng())

Generate a default initial state vector for the SRNN ODE.
- Adaptation states `a` initialized to 0
- STD states `b` initialized to 1 (no depression)
- Dendritic states `x` initialized to small Gaussian noise
"""
function srnn_initial_state(layer::SRNN_ODE; rng=Random.default_rng())
    a_E_0 = zeros(Float32, layer.n_E * layer.n_a_E)
    a_I_0 = zeros(Float32, layer.n_I * layer.n_a_I)
    b_E_0 = ones(Float32, layer.n_E * layer.n_b_E)
    b_I_0 = ones(Float32, layer.n_I * layer.n_b_I)
    x_0   = Float32(0.1) .* randn(rng, Float32, layer.n)
    return vcat(a_E_0, a_I_0, b_E_0, b_I_0, x_0)
end
