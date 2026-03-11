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
const F_FRAC  = 0.8
const N_E     = round(Int, F_FRAC * N)
const N_I     = N - N_E
const N_IN    = N          # W_in = I(n), so input dim = n
const N_A_E   = 3
const N_B_E   = 1
const T_SPAN  = (0.0f0, 50.0f0)
const DT      = 0.001f0
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

# ── Unpack state trajectories ──────────────────────────────────────────
t_sol = sol.t
S_all = reduce(hcat, sol.u)'   # (nt × state_dim)
nt = length(t_sol)

# Unpack state: S = [a_E(:); a_I(:); b_E(:); b_I(:); x]
len_a_E = N_E * N_A_E
len_b_E = N_E * N_B_E

# Extract x (dendritic states)
x_offset = len_a_E + 0 + len_b_E + 0  # n_a_I=0, n_b_I=0
X = S_all[:, x_offset+1:x_offset+N]  # (nt × n)

# Extract a_E (adaptation, n_E × n_a_E × nt)
A_E = reshape(S_all[:, 1:len_a_E]', N_E, N_A_E, nt)

# Extract b_E (STD, n_E × nt)
B_E = S_all[:, len_a_E+1:len_a_E+len_b_E]'  # (n_E × nt)

# Compute firing rates and synaptic output
R = zeros(Float32, N, nt)
BR = zeros(Float32, N, nt)

c_E = NNlib.softplus(ps.log_c_E[1])

for k in 1:nt
    x_k = X[k, :]

    # Compute x_eff with SFA
    x_eff = copy(x_k)
    sum_a_E = vec(sum(A_E[:, :, k], dims=2))  # (n_E,)
    x_eff[1:N_E] .-= c_E .* sum_a_E

    # Firing rate
    r_k = layer.activation.(x_eff)
    R[:, k] .= r_k

    # STD multiplicative factor
    b_k = ones(Float32, N)
    b_k[1:N_E] .= B_E[:, k]
    BR[:, k] .= b_k .* r_k
end

# Compute input at saved times for plotting
U_plot = zeros(Float32, nt, N)
for k in 1:nt
    U_plot[k, :] .= u_interp(t_sol[k])
end

# ── Plot ───────────────────────────────────────────────────────────────
println("Generating multi-panel figure...")

e_color = :steelblue
i_color = :firebrick

# Panel 1: External input (E neurons only, first 20 for readability)
n_show = min(20, N_E)
p1 = plot(t_sol, U_plot[:, 1:n_show];
    label=false, title="External Input (E neurons, first $n_show)",
    ylabel="u", lw=0.5, alpha=0.6, color=e_color)

# Panel 2: Dendritic states x(t)
p2 = plot(t_sol, X[:, 1:N_E];
    label=false, title="Dendritic States x(t)",
    ylabel="x", lw=0.3, alpha=0.4, color=e_color)
plot!(p2, t_sol, X[:, N_E+1:end];
    label=false, lw=0.3, alpha=0.4, color=i_color)

# Panel 3: Firing rates r(t)
p3 = plot(t_sol, R[1:N_E, :]';
    label=false, title="Firing Rates r(t)",
    ylabel="r", lw=0.3, alpha=0.4, color=e_color)
plot!(p3, t_sol, R[N_E+1:end, :]';
    label=false, lw=0.3, alpha=0.4, color=i_color)

# Panel 4: Mean adaptation across timescales
mean_a_E = dropdims(mean(A_E, dims=2), dims=2)  # (n_E × nt)
p4 = plot(t_sol, mean_a_E';
    label=false, title="Mean SFA Adaptation a(t) (E neurons)",
    ylabel="a", lw=0.3, alpha=0.5, color=e_color)

# Panel 5: STD b(t)
p5 = plot(t_sol, B_E';
    label=false, title="STD b(t) (E neurons)",
    ylabel="b", lw=0.3, alpha=0.5, color=e_color)

# Panel 6: Synaptic output br(t)
p6 = plot(t_sol, BR[1:N_E, :]';
    label=false, title="Synaptic Output br(t)",
    ylabel="br", xlabel="Time (s)", lw=0.3, alpha=0.4, color=e_color)
plot!(p6, t_sol, BR[N_E+1:end, :]';
    label=false, lw=0.3, alpha=0.4, color=i_color)

fig = plot(p1, p2, p3, p4, p5, p6;
    layout=(6, 1), size=(1200, 1200),
    plot_title="SRNN Simulation (Julia) — n=$N, n_a_E=$N_A_E, n_b_E=$N_B_E")

display(fig)
println("Done. Press Enter to close.")
readline()
