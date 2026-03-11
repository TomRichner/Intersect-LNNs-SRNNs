# test_ltc.jl — LTC time series simulation with 4-panel plot
#
# Analogous to Matlab/LNN/scripts/test_LNN.m
# Builds an LTC network with RMT connectivity, drives it with a sinusoidal
# 2-channel input, integrates the ODE, and produces a 4-panel figure.

using Random, LinearAlgebra
using OrdinaryDiffEq, Plots, NNlib
using Lux

# ── Include source modules ──────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "src", "connectivity.jl"))
include(joinpath(@__DIR__, "..", "src", "models", "ltc.jl"))

# ── Configuration ───────────────────────────────────────────────────────
const N      = 50
const N_IN   = 2
const F_FRAC = 0.8
const INDEG  = 25
const T_SPAN = (0.0f0, 10.0f0)
const DT     = 0.001f0

rng = Random.MersenneTwister(42)

# ── Build network ───────────────────────────────────────────────────────
println("Building LTC network (n=$N, n_in=$N_IN)...")

# Generate RMT weight matrix
W_rmt, E_idx, I_idx, info = generate_rmt_matrix(N, INDEG, Float64(F_FRAC);
                                                  level_of_chaos=1.0, rng=rng)
println("  RMT spectral radius: $(round(info.spectral_radius, digits=3))")

# Create Lux layer and initialize parameters
layer = LTCODE(N, N_IN; activation=tanh)
ps, st = Lux.setup(rng, layer)

# Replace W with RMT matrix
ps = merge(ps, (W = Float32.(W_rmt),))

# ── Sinusoidal input (matching MATLAB SinusoidalStimulus defaults) ──────
function sinusoidal_input(t)
    return Float32[sinpi(2.0f0 * t), cospi(2.0f0 * t)]
end

# ── ODE right-hand side ────────────────────────────────────────────────
function ltc_rhs!(du, u_state, p, t)
    # Set input in state
    st_driven = merge(st, (input = sinusoidal_input(t),))
    dxdt, _ = layer(u_state, p, st_driven)
    du .= dxdt
    return nothing
end

# ── Integrate ──────────────────────────────────────────────────────────
println("Integrating ODE over t=$(T_SPAN)...")
x0 = 0.01f0 .* randn(rng, Float32, N)
prob = ODEProblem(ltc_rhs!, x0, T_SPAN, ps)
sol = solve(prob, Tsit5(); dt=DT, saveat=DT, abstol=1e-6, reltol=1e-6)
println("  Integration complete: $(length(sol.t)) time steps")

# ── Extract trajectories ───────────────────────────────────────────────
t_sol = sol.t
X = reduce(hcat, sol.u)'   # (nt × n)
nt = length(t_sol)

# Compute derived quantities at each time step
U     = zeros(Float32, nt, N_IN)
F_val = zeros(Float32, nt, N)
τ_sys = zeros(Float32, nt, N)

τ = NNlib.softplus.(ps.log_tau)

for k in 1:nt
    u_k = sinusoidal_input(t_sol[k])
    U[k, :] .= u_k
    z = ps.W * X[k, :] .+ ps.W_in * u_k .+ ps.mu
    f_k = tanh.(z)
    F_val[k, :] .= f_k
    τ_sys[k, :] .= τ ./ (1.0f0 .+ τ .* abs.(f_k))
end

# ── Plot ───────────────────────────────────────────────────────────────
println("Generating 4-panel figure...")

# Color palette for neuron traces
neuron_colors = [HSL(h, 0.7, 0.5) for h in range(0, 306, length=N)]

p1 = plot(t_sol, U;
    label=["sin(2πt)" "cos(2πt)"], title="External Input I(t)",
    ylabel="Amplitude", lw=1.2, legend=:topright)

p2 = plot(t_sol, X;
    label=false, title="Neuron States x(t)",
    ylabel="State", lw=0.4, palette=neuron_colors, alpha=0.7)

p3 = plot(t_sol, τ_sys;
    label=false, title="Effective τ_sys(t) = τ / (1 + τ·|f|)",
    ylabel="τ_sys", lw=0.4, palette=neuron_colors, alpha=0.7)

p4 = plot(t_sol, F_val;
    label=false, title="Nonlinearity f(t) = tanh(W·x + W_in·I + μ)",
    ylabel="f", xlabel="Time (s)", lw=0.4, palette=neuron_colors, alpha=0.7)

fig = plot(p1, p2, p3, p4;
    layout=(4, 1), size=(1000, 800),
    plot_title="LTC Simulation (Julia) — n=$N, n_in=$N_IN")

display(fig)
println("Done. Press Enter to close.")
readline()
