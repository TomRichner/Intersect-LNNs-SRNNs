# test_batched_ltc1.jl — Verify batched path matches per-sample path
#
# Creates an LTCODE1 with fixed params, runs 3 samples individually
# (vector path) and as a batch (matrix path), asserts ≈.

using Random, Statistics
using Lux, NNlib

include(joinpath(@__DIR__, "..", "src", "models", "ltc1.jl"))

const N = 8
const N_IN = 4
const T_STEPS = 5

rng = Random.MersenneTwister(42)

# Test all three solver modes
for solver_sym in (:semi_implicit, :explicit, :runge_kutta)
    println("\n=== Testing solver: $solver_sym ===")
    layer = LTCODE1(N, N_IN; solver=solver_sym, ode_solver_unfolds=6)
    ps, st = Lux.setup(rng, layer)

    B = 3
    # Generate random input sequences: (N_IN, T_STEPS, B)
    inputs = Float32.(randn(rng, N_IN, T_STEPS, B))

    # ── Per-sample (vector) path ────────────────────────────────────
    v_singles = zeros(Float32, N, B)
    for b in 1:B
        v = zeros(Float32, N)
        for t in 1:T_STEPS
            u_t = inputs[:, t, b]       # (N_IN,)
            st_d = merge(st, (input = u_t,))
            v, _ = layer(v, ps, st_d)
        end
        v_singles[:, b] .= v
    end

    # ── Batched (matrix) path ───────────────────────────────────────
    v_batch = zeros(Float32, N, B)
    for t in 1:T_STEPS
        u_t = inputs[:, t, :]           # (N_IN, B)
        st_d = merge(st, (input = u_t,))
        v_batch, _ = layer(v_batch, ps, st_d)
    end

    # ── Compare ─────────────────────────────────────────────────────
    max_diff = maximum(abs.(v_singles .- v_batch))
    println("  Max absolute difference: $max_diff")

    if max_diff < 1e-5
        println("  ✓ PASS — batched matches per-sample (atol=1e-5)")
    else
        println("  ✗ FAIL — difference too large!")
        println("  Per-sample final states:")
        display(v_singles)
        println("\n  Batched final states:")
        display(v_batch)
    end
end

# ── Test gradient flow through batched path ─────────────────────────────
println("\n=== Gradient smoke test (batched) ===")
using Zygote

layer = LTCODE1(N, N_IN; solver=:semi_implicit, ode_solver_unfolds=6)
ps, st = Lux.setup(rng, layer)
head = Lux.Dense(N => 3; init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32)
ps_head, st_head = Lux.setup(rng, head)

# Random batch
B = 4
x_batch = Float32.(randn(rng, N_IN, 5, B))
y_labels = [1, 2, 3, 1]

loss_val, grads = Zygote.withgradient((ltc=ps, head=ps_head)) do p
    v = zeros(Float32, N, B)
    for t in 1:5
        u_t = @view x_batch[:, t, :]
        st_d = merge(st, (input = u_t,))
        v, _ = layer(v, p.ltc, st_d)
    end
    logits, _ = head(v, p.head, st_head)
    # Simple cross-entropy
    m = maximum(logits, dims=1)
    log_probs = logits .- m .- log.(sum(exp.(logits .- m), dims=1))
    loss = zero(eltype(logits))
    for i in 1:B
        loss -= log_probs[y_labels[i], i]
    end
    loss / B
end

println("  Loss: $loss_val")
ltc_grad = grads[1].ltc
n_nothing = sum(g === nothing for g in values(ltc_grad))
if n_nothing == 0
    println("  ✓ All LTC gradients present")
else
    println("  ✗ $n_nothing LTC gradients are nothing")
end
println("  Head weight grad norm: $(sum(abs2, grads[1].head.weight))")
println("\nAll tests complete.")
