# test_lyapunov.jl — Verification tests for Benettin LLE
#
# Five tests with time series + local LLE plots.
#
# Usage:
#   julia --project=JuliaLang JuliaLang/scripts/test_lyapunov.jl

using Random, LinearAlgebra
using Lux, NNlib, Plots

include(joinpath(@__DIR__, "..", "src", "models", "srnn.jl"))
include(joinpath(@__DIR__, "..", "src", "lyapunov.jl"))
include(joinpath(@__DIR__, "..", "src", "plotting", "plot_srnn.jl"))
include(joinpath(@__DIR__, "..", "src", "plotting", "plot_weights.jl"))

"""
Plot SRNN trajectory + local LLE for a test case.
Conditionally includes stimulus, adaptation, and STD panels.
Returns the Plots.Plot object.
"""
function plot_lya_test(title_str, cell, ps, S_traj, local_lya, LLE_val, tau_interval;
    u=nothing, n_show_input=5)
    ode = cell isa SRNNCell ? cell.ode : cell
    c = cell isa SRNNCell ? cell : nothing
    nt = size(S_traj, 2)
    dt_macro = isnothing(c) ? 1.0f0 : c.h * c.ode_solver_unfolds
    t_sec = range(0, step=dt_macro, length=nt)  # time in seconds

    # Unpack trajectory for plotting
    data = unpack_trajectory(ode, S_traj, ps)

    panels = Plots.Plot[]

    # External stimulus (first n_show_input channels)
    if !isnothing(u)
        n_ch = min(n_show_input, size(u, 1))
        t_u = range(0, step=dt_macro, length=size(u, 2))
        p_u = plot(; ylabel="stimulus", legend=false, title=title_str)
        for i in 1:n_ch
            plot!(p_u, t_u, u[i, :]; lw=0.8, alpha=0.5, color=:gray40, label=false)
        end
        push!(panels, p_u)
    end

    # Dendritic states
    has_title = !isnothing(u)  # title already on stimulus panel
    p_x = plot(; ylabel="dendrite", legend=false,
        title=(has_title ? "" : title_str))
    _plot_ei_lines!(p_x, t_sec, data.x_E, data.x_I)
    push!(panels, p_x)

    # Firing rates
    p_r = plot(; ylabel="rate", legend=false, ylim=(0, 1))
    _plot_ei_lines!(p_r, t_sec, data.r_E, data.r_I)
    push!(panels, p_r)

    # Adaptation Σa(t) — only if SFA active
    if !isnothing(data.a_E) || !isnothing(data.a_I)
        p_a = plot(; ylabel="adaptation", legend=false)
        if !isnothing(data.a_E)
            a_E_sum = dropdims(sum(data.a_E, dims=2), dims=2)
            for i in 1:size(a_E_sum, 1)
                plot!(p_a, t_sec, a_E_sum[i, :]; lw=1.0, alpha=0.5,
                    color=_palette_color(E_PALETTE, i), label=false)
            end
        end
        if !isnothing(data.a_I)
            a_I_sum = dropdims(sum(data.a_I, dims=2), dims=2)
            for i in 1:size(a_I_sum, 1)
                plot!(p_a, t_sec, a_I_sum[i, :]; lw=1.0, alpha=0.5,
                    color=_palette_color(I_PALETTE, i), label=false)
            end
        end
        push!(panels, p_a)
    end

    # STD b(t) — only if STD active
    if !isnothing(data.b_E) || !isnothing(data.b_I)
        p_b = plot(; ylabel="depression", legend=false, ylim=(0, 1))
        if !isnothing(data.b_E)
            for i in 1:size(data.b_E, 1)
                plot!(p_b, t_sec, data.b_E[i, :]; lw=1.0, alpha=0.5,
                    color=_palette_color(E_PALETTE, i), label=false)
            end
        end
        if !isnothing(data.b_I)
            for i in 1:size(data.b_I, 1)
                plot!(p_b, t_sec, data.b_I[i, :]; lw=1.0, alpha=0.5,
                    color=_palette_color(I_PALETTE, i), label=false)
            end
        end
        push!(panels, p_b)
    end

    # Local LLE
    n_intervals = length(local_lya)
    t_lya = range(0, stop=(n_intervals - 1) * tau_interval, length=n_intervals)
    p_l = plot(t_lya, local_lya; ylabel="λ₁", xlabel="time (s)", legend=false,
        lw=0.8, alpha=0.7, color=:steelblue)
    hline!(p_l, [0.0]; ls=:dash, color=:black, lw=1)
    hline!(p_l, [LLE_val]; ls=:dash, color=:red, lw=1.5)
    annotate!(p_l, [(t_lya[end], LLE_val,
        text("λ₁=$(round(LLE_val; digits=3))", 8, :red, :right))])
    push!(panels, p_l)

    n_panels = length(panels)
    fig = plot(panels...; layout=(n_panels, 1), size=(900, n_panels * 180), link=:x)
    return fig
end

# ═══════════════════════════════════════════════════════════════════════
# Shared network setup (matching MATLAB SRNNModel2 defaults)
# ═══════════════════════════════════════════════════════════════════════

include(joinpath(@__DIR__, "..", "src", "connectivity.jl"))

N = 300
N_IN = 10             # W_in = identity-like, matching MATLAB convention
INDEG = 100
F_FRAC = 0.5
N_E = round(Int, F_FRAC * N)
T_steps = 3000
steps_per_int = 5
H = 0.001f0           # Euler sub-step size

rng = Random.MersenneTwister(42)

# Generate RMT weight matrix with Dale's law (shared across tests)
W_rmt, E_idx, I_idx, info = generate_rmt_matrix(N, INDEG, F_FRAC; rng=rng)
W_rmt = Float32.(W_rmt)
println("RMT spectral radius: $(round(info.spectral_radius; digits=3))")

# Plot W heatmap
fig_W = plot_weight_matrix(W_rmt;
    title_str="RMT W (N=$N, indeg=$INDEG, f=$F_FRAC, ρ=$(round(info.spectral_radius; digits=2)))")
display(fig_W)

# Shared random inputs
inputs = 0.1f0 .* randn(Random.MersenneTwister(55), Float32, N_IN, T_steps)

# ═══════════════════════════════════════════════════════════════════════
# Test 1: Identity stepper (LLE ≈ 0)
# ═══════════════════════════════════════════════════════════════════════
println("=== Test 1: Identity stepper (LLE ≈ 0) ===")

state_dim = 10

# Trajectory is constant (identity dynamics)
S0 = randn(Random.MersenneTwister(42), Float32, state_dim)
S_traj = repeat(S0, 1, T_steps + 1)

# Identity stepper: S → S (no dynamics)
identity_stepper = (S, idx) -> S

LLE_id, local_id, finite_id = benettin_lle(
    S_traj, identity_stepper, steps_per_int;
    tau_interval=1f0, d0=1f-3,
    rng=Random.MersenneTwister(99),
)

println("  LLE = $LLE_id")
println("  Expected: ≈ 0")
if abs(LLE_id) < 0.01f0
    println("  ✓ PASS — |LLE| < 0.01")
else
    println("  ✗ FAIL — |LLE| = $(abs(LLE_id)) too large")
end

# ═══════════════════════════════════════════════════════════════════════
# Test 2: Contractive system (LLE < 0) — RMT W scaled down
# ═══════════════════════════════════════════════════════════════════════
println("\n=== Test 2: Contractive SRNNCell (LLE < 0) ===")

cell_c = SRNNCell(N, N_IN, N_E; n_a_E=0, n_b_E=0, h=H,
    activation=tanh, readout=:dendritic)
ps_c, st_c = Lux.setup(Random.MersenneTwister(99), cell_c)

# Scale RMT W down to make system contractive
W_small = W_rmt .* 0.1f0
ps_c = merge(ps_c, (W=W_small,))

inputs_c = 0.5f0 .* ones(Float32, N_IN, T_steps)

S0_c = srnn_initial_state(cell_c; rng=Random.MersenneTwister(99))
S_traj_c = collect_trajectory(cell_c, ps_c, st_c, inputs_c, S0_c)

stepper_c = (S, idx) -> begin
    u_t = @view inputs_c[:, idx]
    st_d = merge(st_c, (input=u_t,))
    S_next, _ = cell_c(S, ps_c, st_d)
    return S_next
end

tau_c = Float32(steps_per_int * cell_c.ode_solver_unfolds * cell_c.h)

LLE_c, local_c, finite_c = benettin_lle(
    S_traj_c, stepper_c, steps_per_int;
    tau_interval=tau_c, d0=1f-3,
    rng=Random.MersenneTwister(77),
)

println("  LLE = $LLE_c")
println("  tau_interval = $tau_c s")
println("  Expected: < 0 (contractive)")
if LLE_c < 0
    println("  ✓ PASS — LLE < 0")
else
    println("  ✗ FAIL — LLE = $LLE_c (expected < 0)")
end

fig2 = plot_lya_test("Test 2: Contractive (RMT W×0.02)", cell_c, ps_c, S_traj_c, local_c, LLE_c, tau_c; u=inputs_c)
display(fig2)

# ═══════════════════════════════════════════════════════════════════════
# Test 3: Bare chaotic system (LLE > 0) — RMT W, no SFA/STD
# ═══════════════════════════════════════════════════════════════════════
println("\n=== Test 3: Chaotic SRNNCell (LLE > 0) ===")

cell_ch = SRNNCell(N, N_IN, N_E; n_a_E=0, n_b_E=0, h=H, readout=:dendritic)
ps_ch, st_ch = Lux.setup(Random.MersenneTwister(99), cell_ch)

# Use full RMT W (already at spectral radius ~4-5)
ps_ch = merge(ps_ch, (W=W_rmt,))

S0_ch = srnn_initial_state(cell_ch; rng=Random.MersenneTwister(99))
S_traj_ch = collect_trajectory(cell_ch, ps_ch, st_ch, inputs, S0_ch)

stepper_ch = (S, idx) -> begin
    u_t = @view inputs[:, idx]
    st_d = merge(st_ch, (input=u_t,))
    S_next, _ = cell_ch(S, ps_ch, st_d)
    return S_next
end

tau_ch = Float32(steps_per_int * cell_ch.ode_solver_unfolds * cell_ch.h)

LLE_ch, local_ch, finite_ch = benettin_lle(
    S_traj_ch, stepper_ch, steps_per_int;
    tau_interval=tau_ch, d0=1f-3,
    rng=Random.MersenneTwister(77),
)

println("  LLE = $LLE_ch")
println("  tau_interval = $tau_ch s")
println("  Expected: > 0 (chaotic)")
if LLE_ch > 0
    println("  ✓ PASS — LLE > 0")
else
    println("  ✗ FAIL — LLE = $LLE_ch (expected > 0)")
end

fig3 = plot_lya_test("Test 3: Chaotic (RMT W, no SFA/STD)", cell_ch, ps_ch, S_traj_ch, local_ch, LLE_ch, tau_ch; u=inputs)
display(fig3)

# ═══════════════════════════════════════════════════════════════════════
# Test 4: STD stabilization (same RMT W, LLE should decrease)
# ═══════════════════════════════════════════════════════════════════════
println("\n=== Test 4: STD stabilization effect ===")

cell_std = SRNNCell(N, N_IN, N_E; n_a_E=0, n_b_E=1, h=H, readout=:dendritic)
ps_std, st_std = Lux.setup(Random.MersenneTwister(99), cell_std)

# Inject same RMT W
ps_std = merge(ps_std, (W=W_rmt,))

S0_std = srnn_initial_state(cell_std; rng=Random.MersenneTwister(99))
S_traj_std = collect_trajectory(cell_std, ps_std, st_std, inputs, S0_std)

stepper_std = (S, idx) -> begin
    u_t = @view inputs[:, idx]
    st_d = merge(st_std, (input=u_t,))
    S_next, _ = cell_std(S, ps_std, st_d)
    return S_next
end

tau_std = Float32(steps_per_int * cell_std.ode_solver_unfolds * cell_std.h)

LLE_std, local_std, finite_std = benettin_lle(
    S_traj_std, stepper_std, steps_per_int;
    tau_interval=tau_std, d0=1f-3,
    rng=Random.MersenneTwister(77),
)

println("  LLE (bare):    $LLE_ch")
println("  LLE (STD):     $LLE_std")
if LLE_std < LLE_ch
    println("  ✓ PASS — STD reduced the LLE (stabilizing effect)")
else
    println("  ✗ FAIL — STD did not reduce LLE (unexpected)")
end

if length(local_std) > 0
    fig4 = plot_lya_test("Test 4: STD only (RMT W)", cell_std, ps_std, S_traj_std, local_std, LLE_std, tau_std; u=inputs)
    display(fig4)
end

# ═══════════════════════════════════════════════════════════════════════
# Test 5: SFA+STD stabilization (same RMT W, LLE closer to 0)
# ═══════════════════════════════════════════════════════════════════════
println("\n=== Test 5: SFA+STD stabilization (edge of chaos) ===")

cell_both = SRNNCell(N, N_IN, N_E; n_a_E=3, n_b_E=1, h=H, readout=:dendritic)
ps_both, st_both = Lux.setup(Random.MersenneTwister(99), cell_both)

# Inject same RMT W
ps_both = merge(ps_both, (W=W_rmt,))

S0_both = srnn_initial_state(cell_both; rng=Random.MersenneTwister(99))
S_traj_both = collect_trajectory(cell_both, ps_both, st_both, inputs, S0_both)

stepper_both = (S, idx) -> begin
    u_t = @view inputs[:, idx]
    st_d = merge(st_both, (input=u_t,))
    S_next, _ = cell_both(S, ps_both, st_d)
    return S_next
end

tau_both = Float32(steps_per_int * cell_both.ode_solver_unfolds * cell_both.h)

LLE_both, local_both, finite_both = benettin_lle(
    S_traj_both, stepper_both, steps_per_int;
    tau_interval=tau_both, d0=1f-3,
    rng=Random.MersenneTwister(77),
)

println("  LLE (bare):      $LLE_ch")
println("  LLE (STD):       $LLE_std")
println("  LLE (SFA+STD):   $LLE_both")
if abs(LLE_both) < abs(LLE_ch)
    println("  ✓ PASS — SFA+STD pushed LLE closer to 0 (edge of chaos)")
else
    println("  ✗ FAIL — SFA+STD did not push LLE closer to 0")
end

if length(local_both) > 0
    fig5 = plot_lya_test("Test 5: SFA+STD (RMT W)", cell_both, ps_both, S_traj_both, local_both, LLE_both, tau_both; u=inputs)
    display(fig5)
end

println("\n═══ ALL TESTS COMPLETE ═══")

