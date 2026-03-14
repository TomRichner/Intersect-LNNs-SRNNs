# test_solver_comparison.jl — Side-by-side: semi-implicit vs explicit Euler
#
# Two-column figures: LEFT = fused (semi-implicit), RIGHT = explicit Euler.
# Each column has the standard SRNN panels: stimulus, x, r, a, b, br.
#
# Three scenarios:
#   1. Full model (SFA+STD), small h (both stable, should agree)
#   2. Full model, moderate h (beginning to diverge)
#   3. Full model, large h (stability stress test)
#
# Usage:
#   julia --project=JuliaLang JuliaLang/scripts/test_solver_comparison.jl

using Random, LinearAlgebra
using Lux, NNlib, Plots

include(joinpath(@__DIR__, "..", "src", "models", "srnn.jl"))
include(joinpath(@__DIR__, "..", "src", "connectivity.jl"))
include(joinpath(@__DIR__, "..", "src", "plotting", "plot_srnn.jl"))

# ═══════════════════════════════════════════════════════════════════════
# Shared setup
# ═══════════════════════════════════════════════════════════════════════

N        = 50
N_IN     = 10
INDEG    = 25
F_FRAC   = 0.5
N_E      = round(Int, F_FRAC * N)
N_I      = N - N_E

rng = Random.MersenneTwister(42)

# Generate shared RMT weight matrix with Dale's law
W_rmt, E_idx, I_idx, info = generate_rmt_matrix(N, INDEG, F_FRAC; rng=rng)
W_rmt = Float32.(W_rmt)
println("RMT spectral radius: $(round(info.spectral_radius; digits=3))")

"""
Run SRNN with a given solver and return (S_traj, cell).
S_traj is (state_dim, T+1): column 1 = S0, column t+1 = state after step t.
"""
function run_solver(; solver::Symbol, n_a_E, n_b_E, ps, st, inputs, S0, h, unfolds)
    cell = SRNNCell(N, N_IN, N_E;
        n_a_E=n_a_E, n_b_E=n_b_E,
        ode_solver_unfolds=unfolds, h=h,
        solver=solver, readout=:synaptic)

    T = size(inputs, 2)
    S_traj = zeros(Float32, cell.state_dim, T + 1)
    S_traj[:, 1] .= S0

    S = copy(S0)
    for t in 1:T
        u_t = @view inputs[:, t]
        st_d = merge(st, (input = u_t,))
        S, _ = cell(S, ps, st_d)
        S_traj[:, t + 1] .= S
    end
    return S_traj, cell
end

# ─────────────────────────────────────────────────────────────────────
# Plotting: one column of standard SRNN panels
# ─────────────────────────────────────────────────────────────────────

"""
Build a vector of subplot panels for one solver run.
Returns panels in order: [stimulus, x, r, Σa, b, br].
"""
function build_panels(t, cell, ps, S_traj, inputs;
                      title_str="", n_show_input=5)
    ode = cell.ode
    data = unpack_trajectory(cell, S_traj, ps)

    panels = Plots.Plot[]

    # ── Stimulus ──
    n_ch = min(n_show_input, size(inputs, 1))
    t_in = range(t[1], step=step(t), length=size(inputs, 2))
    p_u = plot(; ylabel="stimulus", legend=false, title=title_str)
    for i in 1:n_ch
        plot!(p_u, t_in, inputs[i, :]; lw=0.8, alpha=0.5, color=:gray40, label=false)
    end
    push!(panels, p_u)

    # ── Dendritic states x ──
    p_x = plot(; ylabel="dendrite x", legend=false)
    _plot_ei_lines!(p_x, t, data.x_E, data.x_I)
    push!(panels, p_x)

    # ── Firing rates r ──
    p_r = plot(; ylabel="rate r", legend=false, ylim=(0, 1))
    _plot_ei_lines!(p_r, t, data.r_E, data.r_I)
    push!(panels, p_r)

    # ── Adaptation Σa ──
    if !isnothing(data.a_E) || !isnothing(data.a_I)
        p_a = plot(; ylabel="adaptation Σa", legend=false)
        if !isnothing(data.a_E)
            a_E_sum = dropdims(sum(data.a_E, dims=2), dims=2)
            for i in 1:size(a_E_sum, 1)
                plot!(p_a, t, a_E_sum[i, :]; lw=1.0, alpha=0.5,
                    color=_palette_color(E_PALETTE, i), label=false)
            end
        end
        if !isnothing(data.a_I)
            a_I_sum = dropdims(sum(data.a_I, dims=2), dims=2)
            for i in 1:size(a_I_sum, 1)
                plot!(p_a, t, a_I_sum[i, :]; lw=1.0, alpha=0.5,
                    color=_palette_color(I_PALETTE, i), label=false)
            end
        end
        push!(panels, p_a)
    end

    # ── STD b ──
    if !isnothing(data.b_E) || !isnothing(data.b_I)
        p_b = plot(; ylabel="depression b", legend=false, ylim=(0, 1.05))
        if !isnothing(data.b_E)
            for i in 1:size(data.b_E, 1)
                plot!(p_b, t, data.b_E[i, :]; lw=1.0, alpha=0.5,
                    color=_palette_color(E_PALETTE, i), label=false)
            end
        end
        if !isnothing(data.b_I)
            for i in 1:size(data.b_I, 1)
                plot!(p_b, t, data.b_I[i, :]; lw=1.0, alpha=0.5,
                    color=_palette_color(I_PALETTE, i), label=false)
            end
        end
        push!(panels, p_b)
    end

    # ── Synaptic output br ──
    if !isnothing(data.b_E) || !isnothing(data.b_I)
        p_br = plot(; ylabel="synaptic br", xlabel="time (s)", legend=false, ylim=(0, 1))
        _plot_ei_lines!(p_br, t, data.br_E, data.br_I)
        push!(panels, p_br)
    end

    return panels
end

"""
Create a side-by-side figure: LEFT = fused, RIGHT = explicit.
Two columns of matched panels.
"""
function plot_side_by_side(t, cell_si, cell_ex, ps, S_si, S_ex, inputs;
                           suptitle="")
    panels_si = build_panels(t, cell_si, ps, S_si, inputs;
        title_str="Semi-Implicit (Fused)")
    panels_ex = build_panels(t, cell_ex, ps, S_ex, inputs;
        title_str="Explicit Euler")

    n_rows = length(panels_si)
    @assert length(panels_ex) == n_rows "Panel count mismatch"

    # Interleave: [si_row1, ex_row1, si_row2, ex_row2, ...]
    interleaved = Plots.Plot[]
    for i in 1:n_rows
        push!(interleaved, panels_si[i])
        push!(interleaved, panels_ex[i])
    end

    fig = plot(interleaved...;
        layout=grid(n_rows, 2),
        size=(1400, n_rows * 200),
        plot_title=suptitle,
        link=:x)

    return fig
end


# ═══════════════════════════════════════════════════════════════════════
# Build shared cell & params
# ═══════════════════════════════════════════════════════════════════════

N_A_E = 3
N_B_E = 1
H_DEFAULT = Float32(0.001)

cell_ref = SRNNCell(N, N_IN, N_E; n_a_E=N_A_E, n_b_E=N_B_E, h=H_DEFAULT,
    ode_solver_unfolds=4, solver=:semi_implicit)
ps_ref, st_ref = Lux.setup(MersenneTwister(99), cell_ref)
ps_ref = merge(ps_ref, (W=W_rmt,))

# Shared input & initial state
T_STEPS = 2000
inputs_all = 0.2f0 .* randn(MersenneTwister(55), Float32, N_IN, T_STEPS)
S0 = srnn_initial_state(cell_ref; rng=MersenneTwister(99))


# ═══════════════════════════════════════════════════════════════════════
# Scenario 1: Small step (h=0.001, 4 unfolds) — solvers should agree
# ═══════════════════════════════════════════════════════════════════════
println("\n" * "="^70)
println("  Scenario 1: h=0.001, 4 unfolds — expect close agreement")
println("="^70)

h1 = Float32(0.001); uf1 = 4

S_si_1, c_si_1 = run_solver(; solver=:semi_implicit, n_a_E=N_A_E, n_b_E=N_B_E,
    ps=ps_ref, st=st_ref, inputs=inputs_all, S0=S0, h=h1, unfolds=uf1)
S_ex_1, c_ex_1 = run_solver(; solver=:explicit, n_a_E=N_A_E, n_b_E=N_B_E,
    ps=ps_ref, st=st_ref, inputs=inputs_all, S0=S0, h=h1, unfolds=uf1)

dt1 = h1 * uf1
t1 = range(0, step=dt1, length=T_STEPS + 1)
max_d1 = maximum(abs.(S_si_1 .- S_ex_1))
println("  Max |fused - explicit| = $(round(max_d1; sigdigits=4))")

fig1 = plot_side_by_side(t1, c_si_1, c_ex_1, ps_ref, S_si_1, S_ex_1, inputs_all;
    suptitle="Scenario 1: h=$(h1), unfolds=$(uf1) — max Δ=$(round(max_d1; sigdigits=3))")
display(fig1)


# ═══════════════════════════════════════════════════════════════════════
# Scenario 2: Moderate step (h=0.01, 6 unfolds)
# ═══════════════════════════════════════════════════════════════════════
println("\n" * "="^70)
println("  Scenario 2: h=0.01, 6 unfolds — moderate step")
println("="^70)

h2 = Float32(0.01); uf2 = 6

S_si_2, c_si_2 = run_solver(; solver=:semi_implicit, n_a_E=N_A_E, n_b_E=N_B_E,
    ps=ps_ref, st=st_ref, inputs=inputs_all, S0=S0, h=h2, unfolds=uf2)
S_ex_2, c_ex_2 = run_solver(; solver=:explicit, n_a_E=N_A_E, n_b_E=N_B_E,
    ps=ps_ref, st=st_ref, inputs=inputs_all, S0=S0, h=h2, unfolds=uf2)

dt2 = h2 * uf2
t2 = range(0, step=dt2, length=T_STEPS + 1)
max_d2 = maximum(abs.(S_si_2 .- S_ex_2))
println("  Max |fused - explicit| = $(round(max_d2; sigdigits=4))")

fig2 = plot_side_by_side(t2, c_si_2, c_ex_2, ps_ref, S_si_2, S_ex_2, inputs_all;
    suptitle="Scenario 2: h=$(h2), unfolds=$(uf2) — max Δ=$(round(max_d2; sigdigits=3))")
display(fig2)


# ═══════════════════════════════════════════════════════════════════════
# Scenario 3: Large step (h=0.05, 4 unfolds) — stability stress test
# ═══════════════════════════════════════════════════════════════════════
println("\n" * "="^70)
println("  Scenario 3: h=0.05, 4 unfolds — large step stress test")
println("="^70)

T_STEPS_3 = 500
h3 = Float32(0.05); uf3 = 4
inputs_3 = inputs_all[:, 1:T_STEPS_3]

S_si_3, c_si_3 = run_solver(; solver=:semi_implicit, n_a_E=N_A_E, n_b_E=N_B_E,
    ps=ps_ref, st=st_ref, inputs=inputs_3, S0=S0, h=h3, unfolds=uf3)
S_ex_3, c_ex_3 = run_solver(; solver=:explicit, n_a_E=N_A_E, n_b_E=N_B_E,
    ps=ps_ref, st=st_ref, inputs=inputs_3, S0=S0, h=h3, unfolds=uf3)

dt3 = h3 * uf3
t3 = range(0, step=dt3, length=T_STEPS_3 + 1)

si_ok = all(isfinite, S_si_3)
ex_ok = all(isfinite, S_ex_3)
println("  Fused solver finite:    $si_ok")
println("  Explicit solver finite: $ex_ok")

if si_ok && ex_ok
    max_d3 = maximum(abs.(S_si_3 .- S_ex_3))
    println("  Max |fused - explicit| = $(round(max_d3; sigdigits=4))")
    fig3 = plot_side_by_side(t3, c_si_3, c_ex_3, ps_ref, S_si_3, S_ex_3, inputs_3;
        suptitle="Scenario 3: h=$(h3), unfolds=$(uf3) — max Δ=$(round(max_d3; sigdigits=3))")
    display(fig3)
elseif si_ok && !ex_ok
    println("  ✓ Fused solver stayed stable where explicit diverged!")
    # Show fused-only figure
    panels_si = build_panels(t3, c_si_3, ps_ref, S_si_3, inputs_3;
        title_str="Fused (stable)")
    fig3 = plot(panels_si...; layout=(length(panels_si), 1),
        size=(800, length(panels_si) * 200),
        plot_title="Scenario 3: Explicit DIVERGED (h=$h3) — Fused stable")
    display(fig3)
else
    println("  Both solvers diverged at this step size")
end


println("\n═══ COMPARISON COMPLETE ═══")
