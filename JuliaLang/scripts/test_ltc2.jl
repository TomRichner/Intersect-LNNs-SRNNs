# test_ltc2.jl — LTC2 time series simulation with 4-panel plot
#
# Tests LTCODE2 (Hasani MATLAB-faithful) with sinusoidal input.
# Single-layer and multi-layer modes.

using Random, LinearAlgebra
using OrdinaryDiffEq, Plots, NNlib
using Lux

# ── Include source modules ──────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "src", "activations.jl"))
include(joinpath(@__DIR__, "..", "src", "models", "ltc2.jl"))

# ── Configuration ───────────────────────────────────────────────────────
const K       = 50        # neurons per layer
const N_IN    = 2
const N_LAYERS = 1        # single layer first
const T_SPAN  = (0.0f0, 10.0f0)
const DT      = 0.001f0

rng = Random.MersenneTwister(42)

# ── Sinusoidal input ────────────────────────────────────────────────────
sinusoidal_input(t) = Float32[sinpi(2.0f0 * t), cospi(2.0f0 * t)]

# ══════════════════════════════════════════════════════════════════════════
#  Single-Layer Test
# ══════════════════════════════════════════════════════════════════════════
println("=== LTCODE2 Single-Layer (k=$K) ===")

layer = LTCODE2(K, N_IN; n_layers=N_LAYERS)
ps, st = Lux.setup(rng, layer)
N_total = _total_n(layer)

function rhs_ltc2!(du, x, p, t)
    st_d = merge(st, (input = sinusoidal_input(t),))
    dxdt, _ = layer(x, p, st_d)
    du .= dxdt
    return nothing
end

x0 = 0.01f0 .* randn(rng, Float32, N_total)
prob = ODEProblem(rhs_ltc2!, x0, T_SPAN, ps)
sol = solve(prob, Tsit5(); dt=DT, saveat=0.01f0, abstol=1e-6, reltol=1e-6)
println("  Integration: $(length(sol.t)) steps")

# Extract trajectories
t_sol = sol.t
X = reduce(hcat, sol.u)'   # (nt × n)
nt = length(t_sol)

# Compute derived quantities
U = zeros(Float32, nt, N_IN)
F_ff  = zeros(Float32, nt, N_total)
F_rec = zeros(Float32, nt, N_total)
τ_sys = zeros(Float32, nt, N_total)

for k_idx in 1:nt
    u_k = sinusoidal_input(t_sol[k_idx])
    U[k_idx, :] .= u_k

    for j in 1:N_LAYERS
        idx = ((j - 1) * K + 1):(j * K)
        x_layer = X[k_idx, idx]

        source = (j == 1) ? u_k : X[k_idx, ((j-2)*K+1):((j-1)*K)]
        W_ff  = ps[Symbol("W_ff_$j")]
        b_ff  = ps[Symbol("b_ff_$j")]
        W_rec = ps[Symbol("W_rec_$j")]
        b_rec = ps[Symbol("b_rec_$j")]

        f_ff  = tanh.(W_ff' * source .+ b_ff)
        f_rec = tanh.(W_rec' * x_layer .+ b_rec)

        F_ff[k_idx, idx]  .= f_ff
        F_rec[k_idx, idx] .= f_rec

        τ_layer = ps.tau[idx]
        τ_sys[k_idx, idx] .= τ_layer ./ (1.0f0 .+ τ_layer .* (abs.(f_ff) .+ abs.(f_rec)))
    end
end

# Plot
neuron_colors = [HSL(h, 0.7, 0.5) for h in range(0, 306, length=N_total)]

p1 = plot(t_sol, U;
    label=["sin(2πt)" "cos(2πt)"], title="External Input I(t)",
    ylabel="Amplitude", lw=1.2, legend=:topright)

p2 = plot(t_sol, X;
    label=false, title="Neuron States x(t)",
    ylabel="State", lw=0.4, palette=neuron_colors, alpha=0.7)

p3 = plot(t_sol, τ_sys;
    label=false, title="τ_sys = τ / (1 + τ(|f_ff| + |f_rec|))",
    ylabel="τ_sys", lw=0.4, palette=neuron_colors, alpha=0.7)

p4 = plot(t_sol, F_ff;
    label=false, title="Feedforward Activation f_ff(t)",
    ylabel="f_ff", xlabel="Time (s)", lw=0.4, palette=neuron_colors, alpha=0.7)

fig1 = plot(p1, p2, p3, p4;
    layout=(4, 1), size=(1000, 800),
    plot_title="LTCODE2 (Hasani MATLAB-faithful) — Single Layer, k=$K")

display(fig1)

# ══════════════════════════════════════════════════════════════════════════
#  Multi-Layer Test
# ══════════════════════════════════════════════════════════════════════════
println("\n=== LTCODE2 Multi-Layer (2 layers × 25 neurons) ===")

const K_ML = 25
layer_ml = LTCODE2(K_ML, N_IN; n_layers=2)
ps_ml, st_ml = Lux.setup(rng, layer_ml)
N_ml = _total_n(layer_ml)

function rhs_ml!(du, x, p, t)
    st_d = merge(st_ml, (input = sinusoidal_input(t),))
    dxdt, _ = layer_ml(x, p, st_d)
    du .= dxdt
    return nothing
end

x0_ml = 0.01f0 .* randn(rng, Float32, N_ml)
prob_ml = ODEProblem(rhs_ml!, x0_ml, T_SPAN, ps_ml)
sol_ml = solve(prob_ml, Tsit5(); dt=DT, saveat=0.01f0, abstol=1e-6, reltol=1e-6)
println("  Integration: $(length(sol_ml.t)) steps")

X_ml = reduce(hcat, sol_ml.u)'

p_l1 = plot(sol_ml.t, X_ml[:, 1:K_ML];
    label=false, title="Layer 1 States",
    ylabel="x", lw=0.4, alpha=0.7)

p_l2 = plot(sol_ml.t, X_ml[:, (K_ML+1):N_ml];
    label=false, title="Layer 2 States",
    ylabel="x", xlabel="Time (s)", lw=0.4, alpha=0.7)

fig2 = plot(p_l1, p_l2;
    layout=(2, 1), size=(1000, 500),
    plot_title="LTCODE2 Multi-Layer — 2 × $K_ML neurons")

display(fig2)

println("\nDone. Press Enter to close.")
readline()
