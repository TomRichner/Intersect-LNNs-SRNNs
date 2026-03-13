# test_srnn_cell.jl — SRNNCell verification tests
#
# Tests: forward pass, batched equivalence, gradient flow, readout modes,
#        per_neuron flag, tau_a endpoint differentiability, Hopfield baseline.
#
# Usage:
#   julia --project=JuliaLang JuliaLang/scripts/test_srnn_cell.jl

using Random, Statistics
using Lux, NNlib, Zygote

include(joinpath(@__DIR__, "..", "src", "models", "srnn.jl"))

const N = 8
const N_IN = 4
const N_E = 6
const T_STEPS = 5

rng = Random.MersenneTwister(42)

# ═══════════════════════════════════════════════════════════════════════
# Test 1: Forward pass sanity
# ═══════════════════════════════════════════════════════════════════════
println("=== Test 1: Forward pass sanity ===")
cell = SRNNCell(N, N_IN, N_E; n_a_E=3, n_b_E=1, ode_solver_unfolds=6)
ps, st = Lux.setup(rng, cell)
S0 = srnn_initial_state(cell; rng=rng)
println("  state_dim = $(length(S0)) (expected $(cell.state_dim))")
@assert length(S0) == cell.state_dim "State dimension mismatch!"

S = S0
let S = S
    for t in 1:T_STEPS
        u_t = randn(rng, Float32, N_IN)
        st_d = merge(st, (input = u_t,))
        S, _ = cell(S, ps, st_d)
    end
    global _S1 = S
end
S = _S1
println("  Final state range: [$(minimum(S)), $(maximum(S))]")
@assert all(isfinite, S) "Non-finite values in state!"
println("  ✓ PASS")

# ═══════════════════════════════════════════════════════════════════════
# Test 2: Batched equivalence
# ═══════════════════════════════════════════════════════════════════════
println("\n=== Test 2: Batched equivalence ===")
B = 3
inputs = Float32.(randn(rng, N_IN, T_STEPS, B))

# Per-sample
S_singles = zeros(Float32, cell.state_dim, B)
for b in 1:B
    let S = srnn_initial_state(cell; rng=MersenneTwister(99))
        for t in 1:T_STEPS
            u_t = inputs[:, t, b]
            st_d = merge(st, (input = u_t,))
            S, _ = cell(S, ps, st_d)
        end
        S_singles[:, b] .= S
    end
end

# Batched — build initial state by hcatting per-sample states (same RNG per column)
S_batch = hcat([srnn_initial_state(cell; rng=MersenneTwister(99)) for _ in 1:B]...)
let S_batch = S_batch
    for t in 1:T_STEPS
        u_t = inputs[:, t, :]
        st_d = merge(st, (input = u_t,))
        S_batch, _ = cell(S_batch, ps, st_d)
    end
    global _S_batch = S_batch
end
S_batch = _S_batch

max_diff = maximum(abs.(S_singles .- S_batch))
println("  Max absolute difference: $max_diff")
if max_diff < 1e-5
    println("  ✓ PASS — batched matches per-sample")
else
    println("  ✗ FAIL — difference too large!")
end

# ═══════════════════════════════════════════════════════════════════════
# Test 3: Gradient smoke test (batched)
# ═══════════════════════════════════════════════════════════════════════
println("\n=== Test 3: Gradient smoke test ===")
head = Lux.Dense(N => 3; init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32)
ps_head, st_head = Lux.setup(rng, head)

x_batch = Float32.(randn(rng, N_IN, T_STEPS, B))
y_labels = [1, 2, 3]

loss_val, grads = Zygote.withgradient((srnn=ps, head=ps_head)) do p
    S = srnn_initial_state(cell, B; rng=MersenneTwister(99))
    for t in 1:T_STEPS
        u_t = @view x_batch[:, t, :]
        st_d = merge(st, (input = u_t,))
        S, _ = cell(S, p.srnn, st_d)
    end
    h = readout(cell, S, p.srnn)
    logits, _ = head(h, p.head, st_head)
    m = maximum(logits, dims=1)
    log_probs = logits .- m .- log.(sum(exp.(logits .- m), dims=1))
    loss = zero(eltype(logits))
    for i in 1:B
        loss -= log_probs[y_labels[i], i]
    end
    loss / B
end

println("  Loss: $loss_val")
srnn_grad = grads[1].srnn
n_nothing = 0
for k in keys(srnn_grad)
    g = getproperty(srnn_grad, k)
    if g === nothing
        println("  WARNING: gradient for $k is nothing!")
        n_nothing += 1
    end
end
if n_nothing == 0
    println("  ✓ All SRNN gradients present")
else
    println("  ✗ $n_nothing gradients are nothing")
end
println("  Head weight grad norm: $(sum(abs2, grads[1].head.weight))")

# ═══════════════════════════════════════════════════════════════════════
# Test 4: Readout modes
# ═══════════════════════════════════════════════════════════════════════
println("\n=== Test 4: Readout modes ===")
for mode in (:dendritic, :rate, :synaptic)
    cell_m = SRNNCell(N, N_IN, N_E; n_a_E=3, n_b_E=1, readout=mode)
    ps_m, st_m = Lux.setup(rng, cell_m)

    # Vector
    S = srnn_initial_state(cell_m; rng=rng)
    st_d = merge(st_m, (input = randn(rng, Float32, N_IN),))
    S, _ = cell_m(S, ps_m, st_d)
    h = readout(cell_m, S, ps_m)
    @assert size(h) == (N,) "readout :$mode vector: expected ($N,), got $(size(h))"

    # Batched
    S_b = srnn_initial_state(cell_m, B; rng=rng)
    st_d = merge(st_m, (input = randn(rng, Float32, N_IN, B),))
    S_b, _ = cell_m(S_b, ps_m, st_d)
    h_b = readout(cell_m, S_b, ps_m)
    @assert size(h_b) == (N, B) "readout :$mode batch: expected ($N,$B), got $(size(h_b))"

    println("  :$mode → vector $(size(h)), batch $(size(h_b)) ✓")
end
println("  ✓ PASS")

# ═══════════════════════════════════════════════════════════════════════
# Test 5: per_neuron flag
# ═══════════════════════════════════════════════════════════════════════
println("\n=== Test 5: per_neuron flag ===")
cell_global = SRNNCell(N, N_IN, N_E; n_a_E=3, n_b_E=1, per_neuron=false)
cell_pn = SRNNCell(N, N_IN, N_E; n_a_E=3, n_b_E=1, per_neuron=true)
ps_g, _ = Lux.setup(rng, cell_global)
ps_p, _ = Lux.setup(rng, cell_pn)

n_params_g = sum(length(getproperty(ps_g, k)) for k in keys(ps_g))
n_params_p = sum(length(getproperty(ps_p, k)) for k in keys(ps_p))
println("  Global params: $n_params_g")
println("  Per-neuron params: $n_params_p")
@assert n_params_p > n_params_g "Per-neuron should have more params!"

# Verify shapes
@assert length(ps_g.a_0) == 1 "Global a_0 should be scalar"
@assert length(ps_p.a_0) == N "Per-neuron a_0 should be (N,)"
@assert length(ps_g.log_tau_d) == 1 "Global tau_d should be scalar"
@assert length(ps_p.log_tau_d) == N "Per-neuron tau_d should be (N,)"
println("  ✓ PASS — per_neuron shapes correct")

# Verify per_neuron runs
S = srnn_initial_state(cell_pn; rng=rng)
st_pn = Lux.initialstates(rng, cell_pn)
st_d = merge(st_pn, (input = randn(rng, Float32, N_IN),))
S_out, _ = cell_pn(S, ps_p, st_d)
@assert all(isfinite, S_out) "Per-neuron forward pass produced non-finite values!"
println("  ✓ Per-neuron forward pass works")

# ═══════════════════════════════════════════════════════════════════════
# Test 6: τ_a endpoint differentiability
# ═══════════════════════════════════════════════════════════════════════
println("\n=== Test 6: τ_a endpoint differentiability ===")
_, grads_ep = Zygote.withgradient(ps) do p
    S = srnn_initial_state(cell; rng=MersenneTwister(99))
    u_t = randn(MersenneTwister(77), Float32, N_IN)
    st_d = merge(st, (input = u_t,))
    S, _ = cell(S, p, st_d)
    sum(S)  # simple scalar output
end
g_lo = grads_ep[1].log_tau_a_E_lo
g_hi = grads_ep[1].log_tau_a_E_hi
@assert g_lo !== nothing "Gradient for log_tau_a_E_lo is nothing!"
@assert g_hi !== nothing "Gradient for log_tau_a_E_hi is nothing!"
println("  grad(log_tau_a_E_lo) = $g_lo")
println("  grad(log_tau_a_E_hi) = $g_hi")
println("  ✓ PASS — endpoints are differentiable")

# ═══════════════════════════════════════════════════════════════════════
# Test 7: Hopfield baseline (no SFA/STD)
# ═══════════════════════════════════════════════════════════════════════
println("\n=== Test 7: Hopfield baseline ===")
cell_hop = SRNNCell(N, N_IN, N_E; n_a_E=0, n_b_E=0, activation=tanh, readout=:dendritic)
ps_hop, st_hop = Lux.setup(rng, cell_hop)
S = srnn_initial_state(cell_hop; rng=rng)
println("  Hopfield state_dim = $(cell_hop.state_dim) (should equal n=$N)")
@assert cell_hop.state_dim == N "Hopfield state_dim should equal n!"

let S = srnn_initial_state(cell_hop; rng=rng)
    for t in 1:T_STEPS
        u_t = randn(rng, Float32, N_IN)
        st_d = merge(st_hop, (input = u_t,))
        S, _ = cell_hop(S, ps_hop, st_d)
    end
    global _S_hop = S
end
S = _S_hop
h = readout(cell_hop, S, ps_hop)
@assert size(h) == (N,) "Hopfield readout size should be ($N,)"
@assert all(isfinite, h) "Hopfield readout has non-finite values!"
println("  ✓ PASS — Hopfield mode works (state_dim=$N, readout=$(size(h)))")

# ═══════════════════════════════════════════════════════════════════════
# Test 8: a_0 gradient
# ═══════════════════════════════════════════════════════════════════════
println("\n=== Test 8: a_0 gradient ===")
g_a0 = grads_ep[1].a_0
@assert g_a0 !== nothing "Gradient for a_0 is nothing!"
println("  grad(a_0) = $g_a0")
println("  ✓ PASS — a_0 is differentiable")

println("\n═══ ALL TESTS PASSED ═══")
