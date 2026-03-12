# test_ltc1.jl — LTCODE1 (Hasani Python-faithful) simulation test
#
# Tests all three solver modes: semi-implicit, explicit, runge-kutta.
# Produces a 3-row × 2-col comparison figure.

using Random, LinearAlgebra
using Plots, NNlib
using Lux

# ── Include source modules ──────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "src", "models", "ltc1.jl"))

# ── Configuration ───────────────────────────────────────────────────────
const N      = 32
const N_IN   = 2
const T_END  = 10.0f0
const FS     = 400
const DT     = 1.0f0 / FS
const NT     = Int(T_END * FS) + 1

rng = Random.MersenneTwister(42)

sinusoidal_input(t) = Float32[sinpi(2.0f0 * t), cospi(2.0f0 * t)]

# ── Run a solver mode ──────────────────────────────────────────────────
function run_mode(solver_sym)
    layer = LTCODE1(N, N_IN; solver=solver_sym, ode_solver_unfolds=6)
    ps, st = Lux.setup(rng, layer)

    # Time step through
    t_vec = collect(range(0.0f0, T_END, length=NT))
    X = zeros(Float32, NT, N)
    v = zeros(Float32, N)
    X[1, :] .= v

    for k in 2:NT
        u_k = sinusoidal_input(t_vec[k-1])
        st_d = merge(st, (input = u_k,))
        v, _ = layer(v, ps, st_d)
        X[k, :] .= v
    end

    return t_vec, X, ps
end

# ── Run all three solvers ───────────────────────────────────────────────
println("=== LTCODE1 (semi_implicit) ===")
t_si, X_si, _ = run_mode(:semi_implicit)
println("  Final state range: [$(minimum(X_si[end,:])), $(maximum(X_si[end,:]))]")

println("\n=== LTCODE1 (explicit) ===")
t_ex, X_ex, _ = run_mode(:explicit)
println("  Final state range: [$(minimum(X_ex[end,:])), $(maximum(X_ex[end,:]))]")

println("\n=== LTCODE1 (runge_kutta) ===")
t_rk, X_rk, _ = run_mode(:runge_kutta)
println("  Final state range: [$(minimum(X_rk[end,:])), $(maximum(X_rk[end,:]))]")

# ── Decimate for plotting ──────────────────────────────────────────────
deci = max(1, round(Int, FS / 10))
idx_d = 1:deci:NT

t_d = t_si[idx_d]
X_si_d = X_si[idx_d, :]
X_ex_d = X_ex[idx_d, :]
X_rk_d = X_rk[idx_d, :]

# ── Input at decimated times ────────────────────────────────────────────
U_d = reduce(hcat, [sinusoidal_input(t) for t in t_d])'

# ── Plot: 3 rows (one per solver) × 2 cols (input, states) ─────────────
neuron_colors = [HSL(h, 0.7, 0.5) for h in range(0, 306, length=N)]

# Row 1: Semi-implicit
p1a = plot(t_d, U_d; label=["sin" "cos"], ylabel="I(t)", title="Input", lw=1.2)
p1b = plot(t_d, X_si_d; label=false, title="Semi-Implicit (fused)",
    ylabel="v", lw=0.4, palette=neuron_colors, alpha=0.7)

# Row 2: Explicit
p2a = plot(t_d, U_d; label=false, ylabel="I(t)", lw=1.2)
p2b = plot(t_d, X_ex_d; label=false, title="Explicit Euler",
    ylabel="v", lw=0.4, palette=neuron_colors, alpha=0.7)

# Row 3: RK4
p3a = plot(t_d, U_d; label=false, ylabel="I(t)", xlabel="Time (s)", lw=1.2)
p3b = plot(t_d, X_rk_d; label=false, title="Runge-Kutta 4",
    ylabel="v", xlabel="Time (s)", lw=0.4, palette=neuron_colors, alpha=0.7)

fig = plot(p1a, p1b, p2a, p2b, p3a, p3b;
    layout=(3, 2), size=(1200, 900),
    plot_title="LTCODE1 (Hasani Python-faithful) — n=$N, 3 Solver Comparison")

display(fig)

# ── Solver agreement check ──────────────────────────────────────────────
println("\n=== Solver Agreement ===")
l2_si_rk = sqrt(mean((X_si .- X_rk).^2))
l2_si_ex = sqrt(mean((X_si .- X_ex).^2))
l2_ex_rk = sqrt(mean((X_ex .- X_rk).^2))
println("  RMS(semi_implicit - rk4):     $l2_si_rk")
println("  RMS(semi_implicit - explicit): $l2_si_ex")
println("  RMS(explicit - rk4):          $l2_ex_rk")

using Statistics

println("\nDone. Press Enter to close.")
readline()
