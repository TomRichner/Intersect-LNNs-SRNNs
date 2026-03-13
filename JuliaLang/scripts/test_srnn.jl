# test_srnn.jl — SRNN time series simulation with multi-panel plot
#
# Analogous to Matlab/SRNN/scripts/test_SRNN2_defaults.m
# Builds an SRNN with RMT connectivity and SFA+STD, drives it with
# sparse step-function stimulus, integrates the ODE, and plots.

using Random, LinearAlgebra, Statistics
using OrdinaryDiffEq, Plots, NNlib
using Lux

# ── Include source modules ──────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "src", "connectivity.jl"))
include(joinpath(@__DIR__, "..", "src", "models", "srnn.jl"))

# ── Configuration ───────────────────────────────────────────────────────
const N       = 300
const INDEG   = 100
const F_FRAC  = 0.5
const N_E     = round(Int, F_FRAC * N)
const N_I     = N - N_E
const N_IN    = N          # W_in = I(n), so input dim = n
const N_A_E   = 3
const N_B_E   = 1
const T_SPAN  = (0.0f0, 50.0f0)
const DT      = 0.0025f0   # fs = 400 Hz (matching MATLAB)
const FS      = 1.0f0 / DT

rng = Random.MersenneTwister(42)

# ── Build network ───────────────────────────────────────────────────────
println("Building SRNN network (n=$N, n_a_E=$N_A_E, n_b_E=$N_B_E)...")

# Generate RMT weight matrix
W_rmt, E_idx, I_idx, info = generate_rmt_matrix(N, INDEG, Float64(F_FRAC);
                                                  level_of_chaos=1.0, rng=rng)
println("  RMT spectral radius: $(round(info.spectral_radius, digits=3))")

# Create Lux layer
layer = SRNN_ODE(N, N_IN, N_E; n_a_E=N_A_E, n_b_E=N_B_E)
ps, st = Lux.setup(rng, layer)

# Replace W with RMT matrix, W_in with identity
ps = merge(ps, (
    W    = Float32.(W_rmt),
    W_in = Float32.(Matrix(I, N, N)),
))

println("  State dimension: $(layer.state_dim)")

# ── Step stimulus (matching MATLAB StepStimulus defaults) ───────────────
# 3 steps, 15% E density, no I stim, amplitude 0.5
# no_stim_pattern = [true, false, true] (odd steps silent)

function generate_step_stimulus(rng, n, n_E, T_end, fs;
                                n_steps=3, step_density_E=0.15,
                                step_density_I=0.0, amp=0.5,
                                no_stim_pattern=[true, false, true])
    dt = 1.0f0 / fs
    t_stim = collect(0.0f0:dt:T_end)
    nt = length(t_stim)

    step_period = T_end / n_steps
    step_length = round(Int, step_period * fs)

    # Random amplitudes
    random_amp = amp .* randn(rng, Float32, n, n_steps)

    # Sparse mask with separate E/I densities
    sparse_mask = falses(n, n_steps)
    sparse_mask[1:n_E, :] .= rand(rng, n_E, n_steps) .< step_density_E
    sparse_mask[n_E+1:end, :] .= rand(rng, n - n_E, n_steps) .< step_density_I

    random_amp .*= sparse_mask
    for (i, ns) in enumerate(no_stim_pattern)
        if ns
            random_amp[:, i] .= 0.0f0
        end
    end

    # Build u_ex matrix (n × nt)
    u_ex = zeros(Float32, n, nt)
    for step_idx in 1:n_steps
        start_i = (step_idx - 1) * step_length + 1
        end_i = min(step_idx * step_length, nt)
        if start_i > nt
            break
        end
        u_ex[:, start_i:end_i] .= random_amp[:, step_idx]
    end

    return t_stim, u_ex
end

stim_rng = Random.MersenneTwister(42)
t_stim, u_ex = generate_step_stimulus(stim_rng, N, N_E, T_SPAN[2], FS)
println("  Stimulus built: $(length(t_stim)) time points")

# Build linear interpolation for input lookup
# Pre-compute for fast access during ODE integration
function make_input_interpolant(t_stim, u_ex)
    # Return a closure that does nearest-neighbor lookup by index
    dt = t_stim[2] - t_stim[1]
    t0 = t_stim[1]
    nt = length(t_stim)
    return function(t)
        idx = clamp(round(Int, (t - t0) / dt) + 1, 1, nt)
        return @view u_ex[:, idx]
    end
end

u_interp = make_input_interpolant(t_stim, u_ex)

# ── ODE right-hand side ────────────────────────────────────────────────
function srnn_rhs!(dS, S, p, t)
    st_driven = merge(st, (input = u_interp(t),))
    dS_dt, _ = layer(S, p, st_driven)
    dS .= dS_dt
    return nothing
end

# ── Integrate ──────────────────────────────────────────────────────────
println("Integrating ODE over t=$(T_SPAN) (this may take a moment)...")
S0 = srnn_initial_state(layer; rng=Random.MersenneTwister(42))
prob = ODEProblem(srnn_rhs!, S0, T_SPAN, ps)

# Use coarser save interval for memory efficiency
save_dt = 0.01f0  # save every 10ms (matching MATLAB plot_deci ~ 10)
sol = solve(prob, Tsit5(); dt=DT, saveat=save_dt, abstol=1e-6, reltol=1e-6)
println("  Integration complete: $(length(sol.t)) saved time steps")

# ── Unpack and plot using plot_srnn module ─────────────────────────────
include(joinpath(@__DIR__, "..", "src", "plot_srnn.jl"))

t_sol = sol.t
S_all = reduce(hcat, sol.u)   # (state_dim, nt) — columns are state vectors
nt = length(t_sol)

# Unpack state trajectory into named components
data = unpack_trajectory(layer, S_all, ps)

# Compute input at saved times for plotting
n_show = min(20, N_E)
U_E_plot = zeros(Float32, n_show, nt)
U_I_plot = zeros(Float32, N_I, nt)
for k in 1:nt
    u_k = u_interp(t_sol[k])
    U_E_plot[:, k] .= u_k[1:n_show]
    U_I_plot[:, k] .= u_k[N_E+1:end]
end

# Generate multi-panel figure
println("Generating multi-panel figure...")
fig = plot_srnn_tseries(t_sol, layer, data;
    u_E=U_E_plot, u_I=U_I_plot,
    title_str="SRNN Simulation (Julia) — n=$N, n_a_E=$N_A_E, n_b_E=$N_B_E")

display(fig)
println("Done. Press Enter to close.")
readline()
