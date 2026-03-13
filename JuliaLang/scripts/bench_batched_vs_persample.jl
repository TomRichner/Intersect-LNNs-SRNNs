# bench_batched_vs_persample.jl — Time comparison: per-sample vs batched BPTT
#
# Runs 2 epochs (epoch 0 & 1) with both approaches on identical data/params.
#
# Usage:
#   julia --project=JuliaLang JuliaLang/scripts/bench_batched_vs_persample.jl

using Random, Statistics, DelimitedFiles, Printf
using Lux, NNlib, Zygote, Optimisers

include(joinpath(@__DIR__, "..", "src", "models", "ltc1.jl"))

const SEQ_LEN    = 16
const N_FEATURES = 561
const N_CLASSES  = 6
const BATCH_SIZE = 16
const MODEL_SIZE = 32

# ═══════════════════════════════════════════════════════════════════════
# DATA (3D arrays)
# ═══════════════════════════════════════════════════════════════════════
function cut_in_sequences_3d(x, y, seq_len; inc=1)
    n_samples = size(x, 1)
    n_seqs = length(0:inc:(n_samples - seq_len - 1))
    seqs_x = Array{Float32, 3}(undef, size(x, 2), seq_len, n_seqs)
    seqs_y = Matrix{Int}(undef, seq_len, n_seqs)
    idx = 0
    for s in 0:inc:(n_samples - seq_len - 1)
        idx += 1
        start = s + 1
        stop = start + seq_len - 1
        seqs_x[:, :, idx] .= Float32.(x[start:stop, :]')
        seqs_y[:, idx] .= y[start:stop]
    end
    return seqs_x, seqs_y
end

data_dir = joinpath(@__DIR__, "..", "data", "har", "UCI HAR Dataset")
println("Loading data...")
train_x_raw = readdlm(joinpath(data_dir, "train", "X_train.txt"), Float64)
train_y_raw = Int.(vec(readdlm(joinpath(data_dir, "train", "y_train.txt"), Int)))
train_x, train_y = cut_in_sequences_3d(train_x_raw, train_y_raw, SEQ_LEN; inc=1)

n_total = size(train_x, 3)
perm_data = randperm(MersenneTwister(893429), n_total)
n_valid = div(n_total, 10)
train_idx = perm_data[n_valid+1:end]
tx = train_x[:, :, train_idx]
ty = train_y[:, train_idx]
n_train = size(tx, 3)
println("Training sequences: $n_train")

# ═══════════════════════════════════════════════════════════════════════
# MODEL SETUP (shared params)
# ═══════════════════════════════════════════════════════════════════════
rng = MersenneTwister(42)
ltc = LTCODE1(MODEL_SIZE, N_FEATURES; solver=:semi_implicit, ode_solver_unfolds=6)
head = Lux.Dense(MODEL_SIZE => N_CLASSES; init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32)
ps_ltc, st_ltc = Lux.setup(rng, ltc)
ps_head, st_head = Lux.setup(rng, head)

# ── logsumexp helpers ───────────────────────────────────────────────────
logsumexp_vec(x) = (m = maximum(x); m + log(sum(exp.(x .- m))))
function logsumexp_mat(x)
    m = maximum(x, dims=1)
    m .+ log.(sum(exp.(x .- m), dims=1))
end

# ═══════════════════════════════════════════════════════════════════════
# OLD: PER-SAMPLE APPROACH
# ═══════════════════════════════════════════════════════════════════════
function train_persample_epoch!(ltc, head, params, opt_state, st_ltc, st_head, tx, ty)
    n_train = size(tx, 3)
    perm = randperm(n_train)
    n_batches = div(n_train, BATCH_SIZE)

    for b in 1:n_batches
        batch_idx = perm[((b-1)*BATCH_SIZE + 1):(b*BATCH_SIZE)]

        loss_val, grads = Zygote.withgradient(params) do p
            batch_loss = 0.0f0
            for i in batch_idx
                # Per-sample forward through time
                v = zeros(Float32, ltc.n)
                for t in 1:SEQ_LEN
                    u_t = @view tx[:, t, i]
                    st_d = merge(st_ltc, (input = u_t,))
                    v, _ = ltc(v, p.ltc, st_d)
                end
                logits, _ = head(v, p.head, st_head)
                label = ty[end, i]
                log_probs = logits .- logsumexp_vec(logits)
                batch_loss += -log_probs[label]
            end
            batch_loss / BATCH_SIZE
        end
        opt_state, params = Optimisers.update(opt_state, params, grads[1])
    end
    return params, opt_state
end

# ═══════════════════════════════════════════════════════════════════════
# NEW: BATCHED APPROACH
# ═══════════════════════════════════════════════════════════════════════
function train_batched_epoch!(ltc, head, params, opt_state, st_ltc, st_head, tx, ty)
    n_train = size(tx, 3)
    perm = randperm(n_train)
    n_batches = div(n_train, BATCH_SIZE)

    for b in 1:n_batches
        b_start = (b - 1) * BATCH_SIZE + 1
        b_end = b * BATCH_SIZE
        batch_idx = perm[b_start:b_end]
        x_batch = tx[:, :, batch_idx]
        y_batch = ty[end, batch_idx]

        loss_val, grads = Zygote.withgradient(params) do p
            v = zeros(Float32, ltc.n, BATCH_SIZE)
            for t in 1:SEQ_LEN
                u_t = @view x_batch[:, t, :]
                st_d = merge(st_ltc, (input = u_t,))
                v, _ = ltc(v, p.ltc, st_d)
            end
            logits, _ = head(v, p.head, st_head)
            log_probs = logits .- logsumexp_mat(logits)
            loss = zero(eltype(logits))
            for i in 1:BATCH_SIZE
                loss -= log_probs[y_batch[i], i]
            end
            loss / BATCH_SIZE
        end
        opt_state, params = Optimisers.update(opt_state, params, grads[1])
    end
    return params, opt_state
end

# ═══════════════════════════════════════════════════════════════════════
# WARM-UP (compile both paths)
# ═══════════════════════════════════════════════════════════════════════
println("\n─── Warm-up (compiling both paths, 5 batches each) ───")
tx_warmup = tx[:, :, 1:80]
ty_warmup = ty[:, 1:80]

params_w = (ltc = deepcopy(ps_ltc), head = deepcopy(ps_head))
opt_w = Optimisers.setup(Optimisers.Adam(0.01f0), params_w)
print("  Per-sample warmup... ")
@time train_persample_epoch!(ltc, head, params_w, opt_w, st_ltc, st_head, tx_warmup, ty_warmup)

params_w2 = (ltc = deepcopy(ps_ltc), head = deepcopy(ps_head))
opt_w2 = Optimisers.setup(Optimisers.Adam(0.01f0), params_w2)
print("  Batched warmup... ")
@time train_batched_epoch!(ltc, head, params_w2, opt_w2, st_ltc, st_head, tx_warmup, ty_warmup)

# ═══════════════════════════════════════════════════════════════════════
# BENCHMARK: 2 epochs each
# ═══════════════════════════════════════════════════════════════════════
println("\n═══ BENCHMARK: 2 training epochs on $n_train sequences, batch=$BATCH_SIZE ═══\n")

# Per-sample
params_old = (ltc = deepcopy(ps_ltc), head = deepcopy(ps_head))
opt_old = Optimisers.setup(Optimisers.Adam(0.01f0), params_old)

println("Per-sample (old):")
t_old = @elapsed let params_old=params_old, opt_old=opt_old
    for ep in 0:1
        params_old, opt_old = train_persample_epoch!(ltc, head, params_old, opt_old, st_ltc, st_head, tx, ty)
        println("  Epoch $ep done")
    end
end
@printf("  Total time: %.1f seconds (%.1f s/epoch)\n\n", t_old, t_old/2)

# Batched
params_new = (ltc = deepcopy(ps_ltc), head = deepcopy(ps_head))
opt_new = Optimisers.setup(Optimisers.Adam(0.01f0), params_new)

println("Batched (new):")
t_new = @elapsed let params_new=params_new, opt_new=opt_new
    for ep in 0:1
        params_new, opt_new = train_batched_epoch!(ltc, head, params_new, opt_new, st_ltc, st_head, tx, ty)
        println("  Epoch $ep done")
    end
end
@printf("  Total time: %.1f seconds (%.1f s/epoch)\n\n", t_new, t_new/2)

# Summary
speedup = t_old / t_new
@printf("═══ SPEEDUP: %.1f× (%.1f s → %.1f s for 2 epochs) ═══\n", speedup, t_old, t_new)
