# test_lyapunov.jl — Verification tests for Benettin LLE
#
# Three tests: contractive (LLE < 0), identity (LLE ≈ 0), chaotic (LLE > 0).
#
# Usage:
#   julia --project=JuliaLang JuliaLang/scripts/test_lyapunov.jl

using Random, LinearAlgebra
using Lux, NNlib

include(joinpath(@__DIR__, "..", "src", "models", "srnn.jl"))
include(joinpath(@__DIR__, "..", "src", "lyapunov.jl"))

# ═══════════════════════════════════════════════════════════════════════
# Test 1: Identity stepper (LLE ≈ 0)
# ═══════════════════════════════════════════════════════════════════════
println("=== Test 1: Identity stepper (LLE ≈ 0) ===")

state_dim = 10
T = 200
steps_per_interval = 10

# Trajectory is constant (identity dynamics)
S0 = randn(Random.MersenneTwister(42), Float32, state_dim)
S_traj = repeat(S0, 1, T + 1)

# Identity stepper: S → S (no dynamics)
identity_stepper = (S, idx) -> S

LLE_id, local_id, finite_id = benettin_lle(
    S_traj, identity_stepper, steps_per_interval;
    tau_interval = 1f0, d0 = 1f-3,
    rng = Random.MersenneTwister(99),
)

println("  LLE = $LLE_id")
println("  Expected: ≈ 0")
if abs(LLE_id) < 0.01f0
    println("  ✓ PASS — |LLE| < 0.01")
else
    println("  ✗ FAIL — |LLE| = $(abs(LLE_id)) too large")
end

# ═══════════════════════════════════════════════════════════════════════
# Test 2: Contractive system (LLE < 0)
# ═══════════════════════════════════════════════════════════════════════
println("\n=== Test 2: Contractive SRNNCell (LLE < 0) ===")

N = 16
N_IN = 4
N_E = 12
T_steps = 500
steps_per_int = 20

rng = Random.MersenneTwister(42)

# Build SRNNCell with no SFA/STD (Hopfield mode), tanh activation
cell_c = SRNNCell(N, N_IN, N_E; n_a_E=0, n_b_E=0,
                  activation=tanh, readout=:dendritic)
ps_c, st_c = Lux.setup(rng, cell_c)

# Scale W down to make system contractive (spectral radius ≪ 1)
W_small = ps_c.W .* 0.1f0
ps_c = merge(ps_c, (W = W_small,))

# Generate constant input (mild)
inputs_c = 0.01f0 .* ones(Float32, N_IN, T_steps)

# Collect reference trajectory
S0_c = srnn_initial_state(cell_c; rng=Random.MersenneTwister(99))
S_traj_c = collect_trajectory(cell_c, ps_c, st_c, inputs_c, S0_c)

# Build stepper closure
stepper_c = (S, idx) -> begin
    u_t = @view inputs_c[:, idx]
    st_d = merge(st_c, (input = u_t,))
    S_next, _ = cell_c(S, ps_c, st_d)
    return S_next
end

tau_c = Float32(steps_per_int * cell_c.ode_solver_unfolds * cell_c.h)

LLE_c, local_c, finite_c = benettin_lle(
    S_traj_c, stepper_c, steps_per_int;
    tau_interval = tau_c, d0 = 1f-3,
    rng = Random.MersenneTwister(77),
)

println("  LLE = $LLE_c")
println("  tau_interval = $tau_c s")
println("  Expected: < 0 (contractive)")
if LLE_c < 0
    println("  ✓ PASS — LLE < 0")
else
    println("  ✗ FAIL — LLE = $LLE_c (expected < 0)")
end

# ═══════════════════════════════════════════════════════════════════════
# Test 3: Chaotic system (LLE > 0)
# ═══════════════════════════════════════════════════════════════════════
println("\n=== Test 3: Chaotic SRNNCell (LLE > 0) ===")

# Scale W up to make system chaotic (spectral radius > 1)
cell_ch = SRNNCell(N, N_IN, N_E; n_a_E=0, n_b_E=0,
                   activation=tanh, readout=:dendritic)
ps_ch, st_ch = Lux.setup(rng, cell_ch)

# Amplify weights significantly
W_big = ps_ch.W .* 3.0f0
ps_ch = merge(ps_ch, (W = W_big,))

# Gentle random inputs
inputs_ch = 0.1f0 .* randn(Random.MersenneTwister(55), Float32, N_IN, T_steps)

S0_ch = srnn_initial_state(cell_ch; rng=Random.MersenneTwister(99))
S_traj_ch = collect_trajectory(cell_ch, ps_ch, st_ch, inputs_ch, S0_ch)

stepper_ch = (S, idx) -> begin
    u_t = @view inputs_ch[:, idx]
    st_d = merge(st_ch, (input = u_t,))
    S_next, _ = cell_ch(S, ps_ch, st_d)
    return S_next
end

tau_ch = Float32(steps_per_int * cell_ch.ode_solver_unfolds * cell_ch.h)

LLE_ch, local_ch, finite_ch = benettin_lle(
    S_traj_ch, stepper_ch, steps_per_int;
    tau_interval = tau_ch, d0 = 1f-3,
    rng = Random.MersenneTwister(77),
)

println("  LLE = $LLE_ch")
println("  tau_interval = $tau_ch s")
println("  Expected: > 0 (chaotic)")
if LLE_ch > 0
    println("  ✓ PASS — LLE > 0")
else
    println("  ✗ FAIL — LLE = $LLE_ch (expected > 0)")
end

# ═══════════════════════════════════════════════════════════════════════
# Test 4: SRNNCell with SFA (LLE should be more negative than without)
# ═══════════════════════════════════════════════════════════════════════
println("\n=== Test 4: SFA stabilization effect ===")

# Same large W but with SFA enabled — should reduce LLE
cell_sfa = SRNNCell(N, N_IN, N_E; n_a_E=3, n_b_E=0,
                    activation=tanh, readout=:dendritic)
ps_sfa, st_sfa = Lux.setup(rng, cell_sfa)

# Use same large W
W_sfa = ps_sfa.W .* 3.0f0
ps_sfa = merge(ps_sfa, (W = W_sfa,))

S0_sfa = srnn_initial_state(cell_sfa; rng=Random.MersenneTwister(99))
S_traj_sfa = collect_trajectory(cell_sfa, ps_sfa, st_sfa, inputs_ch, S0_sfa)

stepper_sfa = (S, idx) -> begin
    u_t = @view inputs_ch[:, idx]
    st_d = merge(st_sfa, (input = u_t,))
    S_next, _ = cell_sfa(S, ps_sfa, st_d)
    return S_next
end

tau_sfa = Float32(steps_per_int * cell_sfa.ode_solver_unfolds * cell_sfa.h)

LLE_sfa, _, _ = benettin_lle(
    S_traj_sfa, stepper_sfa, steps_per_int;
    tau_interval = tau_sfa, d0 = 1f-3,
    rng = Random.MersenneTwister(77),
)

println("  LLE (no SFA, W×3):   $LLE_ch")
println("  LLE (with SFA, W×3): $LLE_sfa")
if LLE_sfa < LLE_ch
    println("  ✓ PASS — SFA reduced the LLE (stabilizing effect)")
else
    println("  ✗ FAIL — SFA did not reduce LLE (unexpected)")
end

println("\n═══ ALL TESTS COMPLETE ═══")
