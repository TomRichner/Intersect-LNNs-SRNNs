# train_ozone_srnn.jl — Ozone Level Detection with SRNNCell (batched BPTT via Zygote)
#
# Adapted from: ozone.py (Hasani et al.)
# Binary classification with F1 metric and weighted cross-entropy.
# Uses SRNNCell with fused semi-implicit or explicit Euler solver.
#
# Usage:
#   julia --project=JuliaLang JuliaLang/scripts/train_ozone_srnn.jl [--epochs 200] [--size 32] [--lr 0.01] [--bs 256]
#
# Extra SRNN-specific flags:
#   --ne <int>         Excitatory neuron count (default: n÷2)
#   --n_a <int>        SFA timescale count (default: 3, 0 = no SFA)
#   --n_b <int>        STD timescale count (default: 0, 0 = no STD)
#   --unfolds <int>    ODE solver sub-steps (default: 6)
#   --h <float>        ODE step size (default: 0.02 = 1/50)
#   --readout <sym>    Readout mode: synaptic, rate, dendritic (default: synaptic)
#   --solver <sym>     Solver: semi_implicit, explicit (default: semi_implicit)
#   --per_neuron        Per-neuron dynamics params (default: shared scalars)
#
# Checkpoint flags:
#   --seed <int>       Random seed for reproducibility (default: 42)
#   --save <dir>       Checkpoint directory (default: checkpoints/)
#   --resume <path>    Resume from checkpoint file
#   --save_every <int> Save periodic checkpoint every N epochs (default: 5)
#   --warmup <int>     LR warmup epochs: ramp from lr/10 to lr (default: 0 = off)

using Random, Statistics, Printf
using Lux, NNlib, Zygote, Optimisers
using JLD2

# ── Include SRNNCell ────────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "src", "models", "srnn.jl"))

# ── Configuration ───────────────────────────────────────────────────────
const SEQ_LEN    = 32
const N_FEATURES = 72
const N_CLASSES  = 2      # binary: 0 = normal, 1 = ozone day

# Parse simple command-line args
function parse_args()
    epochs = 200
    model_size = 32
    lr = 0.01f0
    batch_size = 256
    n_E = -1
    n_a = 3
    n_b = 0
    unfolds = 6
    h = Float32(1 / 50)
    readout_mode = :synaptic
    solver = :semi_implicit
    per_neuron = false
    seed = 42
    save_dir = joinpath(@__DIR__, "..", "checkpoints")
    resume_path = ""
    save_every = 5
    warmup_epochs = 0

    for i in eachindex(ARGS)
        if ARGS[i] == "--epochs" && i < length(ARGS)
            epochs = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--size" && i < length(ARGS)
            model_size = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--lr" && i < length(ARGS)
            lr = parse(Float32, ARGS[i+1])
        elseif ARGS[i] == "--bs" && i < length(ARGS)
            batch_size = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--ne" && i < length(ARGS)
            n_E = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--n_a" && i < length(ARGS)
            n_a = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--n_b" && i < length(ARGS)
            n_b = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--unfolds" && i < length(ARGS)
            unfolds = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--h" && i < length(ARGS)
            h = parse(Float32, ARGS[i+1])
        elseif ARGS[i] == "--readout" && i < length(ARGS)
            readout_mode = Symbol(ARGS[i+1])
        elseif ARGS[i] == "--solver" && i < length(ARGS)
            solver = Symbol(ARGS[i+1])
        elseif ARGS[i] == "--per_neuron"
            per_neuron = true
        elseif ARGS[i] == "--seed" && i < length(ARGS)
            seed = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--save" && i < length(ARGS)
            save_dir = ARGS[i+1]
        elseif ARGS[i] == "--resume" && i < length(ARGS)
            resume_path = ARGS[i+1]
        elseif ARGS[i] == "--save_every" && i < length(ARGS)
            save_every = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--warmup" && i < length(ARGS)
            warmup_epochs = parse(Int, ARGS[i+1])
        end
    end

    if n_E < 0
        n_E = model_size ÷ 2
    end

    return (; epochs, model_size, lr, batch_size, n_E, n_a, n_b,
              unfolds, h, readout_mode, solver, per_neuron, seed,
              save_dir, resume_path, save_every, warmup_epochs)
end

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

"""
    load_ozone_trace(filepath) → (x::Matrix{Float64}, y::Vector{Int})

Load the ozone eighthr.data file.
- Skips date column (col 1)
- Reads 72 feature columns (cols 2-73), replacing '?' with 0.0
- Reads label (col 74): 0 or 1
Returns features (N×72) and labels (N,).
"""
function load_ozone_trace(filepath::String)
    all_x = Vector{Vector{Float64}}()
    all_y = Vector{Int}()
    miss = 0
    total = 0

    open(filepath) do f
        for line in eachline(f)
            parts = split(strip(line), ',')
            length(parts) == 74 || continue
            total += 1

            # Check for missing values
            has_missing = false
            for i in 2:73
                if parts[i] == "?"
                    has_missing = true
                    break
                end
            end
            if has_missing
                miss += 1
            end

            # Parse features (cols 2-73), replace '?' with 0.0
            feats = Float64[parts[i] == "?" ? 0.0 : parse(Float64, parts[i]) for i in 2:73]
            # Parse label (col 74) — handle trailing period
            label_str = rstrip(parts[74], '.')
            label = Int(round(parse(Float64, label_str)))

            push!(all_x, feats)
            push!(all_y, label)
        end
    end

    @printf("Missing features in %d out of %d samples (%.2f%%)\n", miss, total, 100 * miss / total)
    println("Read $(length(all_x)) lines")

    x = reduce(hcat, all_x)'  # N × 72
    y = all_y

    imbalance = mean(y)
    @printf("Imbalance: %.2f%% positive\n", 100 * imbalance)

    # Global z-score normalization (across ALL values, matching Python)
    global_mean = mean(x)
    global_std = std(x)
    x = (x .- global_mean) ./ global_std

    return Float64.(x), y
end

"""
    cut_in_sequences_ozone(x, y, seq_len; inc=1)

Sliding window with stride `inc`. Returns 3D arrays:
- x_out: (features, seq_len, N_seqs)
- y_out: (seq_len, N_seqs)  — per-timestep labels
"""
function cut_in_sequences_ozone(x::Matrix{Float64}, y::Vector{Int}, seq_len::Int; inc::Int=1)
    n_samples = size(x, 1)
    n_seqs = div(n_samples - seq_len, inc)

    x_out = Array{Float32, 3}(undef, N_FEATURES, seq_len, n_seqs)
    y_out = Matrix{Int}(undef, seq_len, n_seqs)

    for s in 1:n_seqs
        start = (s - 1) * inc + 1
        stop = start + seq_len - 1
        x_out[:, :, s] .= Float32.(x[start:stop, :]')   # transpose: rows→features, cols→time
        y_out[:, s] .= y[start:stop]
    end

    return x_out, y_out
end

struct OzoneData
    train_x::Array{Float32, 3}    # (features, seq_len, N_train)
    train_y::Matrix{Int}          # (seq_len, N_train) — per-timestep labels
    valid_x::Array{Float32, 3}
    valid_y::Matrix{Int}
    test_x::Array{Float32, 3}
    test_y::Matrix{Int}
end

function load_ozone_data(; data_dir=joinpath(@__DIR__, "..", "data", "ozone"))
    println("Loading Ozone data from: $data_dir")

    x, y = load_ozone_trace(joinpath(data_dir, "eighthr.data"))

    # Sliding window (inc=4, seq_len=32)
    all_x, all_y = cut_in_sequences_ozone(x, y, SEQ_LEN; inc=4)
    N_total = size(all_x, 3)
    println("  Total sequences: $N_total")

    # 3-way split (seed 23489, 10% valid, 15% test, 75% train)
    perm = randperm(MersenneTwister(23489), N_total)
    n_valid = div(N_total * 10, 100)
    n_test  = div(N_total * 15, 100)

    valid_idx = perm[1:n_valid]
    test_idx  = perm[n_valid+1:n_valid+n_test]
    train_idx = perm[n_valid+n_test+1:end]

    println("  Split: $(length(train_idx)) train, $(length(valid_idx)) valid, $(length(test_idx)) test")

    # Convert labels: 0-based → 1-based for Julia
    all_y_1 = all_y .+ 1   # 1 = normal, 2 = ozone day

    return OzoneData(
        all_x[:, :, train_idx], all_y_1[:, train_idx],
        all_x[:, :, valid_idx], all_y_1[:, valid_idx],
        all_x[:, :, test_idx],  all_y_1[:, test_idx],
    )
end

# ═══════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════

function build_model(args, rng)
    cell = SRNNCell(args.model_size, N_FEATURES, args.n_E;
        n_a_E=args.n_a, n_b_E=args.n_b,
        ode_solver_unfolds=args.unfolds,
        h=args.h,
        readout=args.readout_mode,
        solver=args.solver,
        per_neuron=args.per_neuron,
    )
    head = Lux.Dense(args.model_size => N_CLASSES;
        init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32)

    ps_cell, st_cell = Lux.setup(rng, cell)
    ps_head, st_head = Lux.setup(rng, head)

    return cell, head, ps_cell, st_cell, ps_head, st_head
end

# ── Batched forward pass (last-step logits, for non-per-timestep use) ──
function forward_batch(cell, head, ps_cell, ps_head, st_cell, st_head, x_batch)
    B = size(x_batch, 3)
    S = srnn_initial_state(cell, B)

    for t in 1:size(x_batch, 2)
        u_t = @view x_batch[:, t, :]
        st_d = merge(st_cell, (input = u_t,))
        S, _ = cell(S, ps_cell, st_d)
    end

    obs = readout(cell, S, ps_cell)
    logits, _ = head(obs, ps_head, st_head)
    return logits
end

# ── Weighted cross-entropy loss (per-timestep, fused — no mutable arrays) ──
function batch_loss(cell, head, ps_cell, ps_head, st_cell, st_head, x_batch, y_labels)
    # y_labels: (T, B) — 1-indexed per-timestep labels
    B = size(x_batch, 3)
    T = size(x_batch, 2)
    S = srnn_initial_state(cell, B)

    total_loss = zero(eltype(x_batch))
    for t in 1:T
        u_t = @view x_batch[:, t, :]
        st_d = merge(st_cell, (input = u_t,))
        S, _ = cell(S, ps_cell, st_d)

        obs = readout(cell, S, ps_cell)
        logits_t, _ = head(obs, ps_head, st_head)  # (N_CLASSES, B)
        log_probs_t = logits_t .- logsumexp_batch(logits_t)
        for i in 1:B
            label = y_labels[t, i]
            # Weight: positive class (2 in Julia 1-indexed) gets 1.5x + 0.1 baseline
            w = (label == 2) ? 1.6f0 : 0.1f0
            total_loss -= w * log_probs_t[label, i]
        end
    end
    return total_loss / (T * B)
end

function logsumexp_batch(x::AbstractMatrix)
    m = maximum(x, dims=1)
    return m .+ log.(sum(exp.(x .- m), dims=1))
end


# ═══════════════════════════════════════════════════════════════════════
# F1 METRIC
# ═══════════════════════════════════════════════════════════════════════

"""
Compute F1 score from per-timestep predictions.
Positive class = 2 (Julia 1-indexed, i.e., original label 1).
"""
function compute_f1(preds::Vector{Int}, labels::Vector{Int})
    tp = sum((preds .== 2) .& (labels .== 2))
    fp = sum((preds .== 2) .& (labels .== 1))
    fn = sum((preds .== 1) .& (labels .== 2))

    prec = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    f1 = 2 * prec * recall / (prec + recall + 1e-6)
    return f1, prec, recall
end

# ═══════════════════════════════════════════════════════════════════════
# EVALUATION (batched, F1)
# ═══════════════════════════════════════════════════════════════════════

function evaluate(cell, head, ps_cell, ps_head, st_cell, st_head,
                  data_x::Array{Float32, 3}, data_y::Matrix{Int};
                  eval_batch_size::Int=128)
    n = size(data_x, 3)
    T = size(data_x, 2)
    total_loss = 0.0f0
    all_preds = Int[]
    all_labels = Int[]

    n_batches = cld(n, eval_batch_size)
    for b in 1:n_batches
        b_start = (b - 1) * eval_batch_size + 1
        b_end = min(b * eval_batch_size, n)
        batch_x = @view data_x[:, :, b_start:b_end]
        batch_y = @view data_y[:, b_start:b_end]
        B = b_end - b_start + 1

        S = srnn_initial_state(cell, B)
        for t in 1:T
            u_t = @view batch_x[:, t, :]
            st_d = merge(st_cell, (input = u_t,))
            S, _ = cell(S, ps_cell, st_d)

            obs = readout(cell, S, ps_cell)
            logits_t, _ = head(obs, ps_head, st_head)
            log_probs_t = logits_t .- logsumexp_batch(logits_t)
            for i in 1:B
                label = batch_y[t, i]
                w = (label == 2) ? 1.6f0 : 0.1f0
                total_loss -= w * log_probs_t[label, i]
            end
            preds_t = vec(getindex.(argmax(logits_t, dims=1), 1))
            append!(all_preds, preds_t)
            append!(all_labels, batch_y[t, 1:B])
        end
    end

    f1, prec, recall = compute_f1(all_preds, all_labels)
    return total_loss / (n * T), f1, prec, recall
end

# ═══════════════════════════════════════════════════════════════════════
# CHECKPOINTING
# ═══════════════════════════════════════════════════════════════════════

function save_checkpoint(path, params, opt_state, epoch, best_valid_f1, args)
    mkpath(dirname(path))
    jldsave(path;
        params = params,
        opt_state = opt_state,
        epoch = epoch,
        best_valid_f1 = best_valid_f1,
        args = args,
    )
    println("  💾 Checkpoint saved: $path (epoch $epoch, valid F1 $(round(best_valid_f1; digits=4)))")
end

function load_checkpoint(path)
    data = jldopen(path, "r") do f
        (
            params = f["params"],
            opt_state = f["opt_state"],
            epoch = f["epoch"],
            best_valid_f1 = f["best_valid_f1"],
            args = f["args"],
        )
    end
    return data
end

# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

function train!(cell, head, ps_cell, ps_head, st_cell, st_head, data::OzoneData;
                epochs::Int=200, lr::Float32=0.01f0, batch_size::Int=256,
                start_epoch::Int=0, initial_opt_state=nothing,
                initial_best_valid_f1::Float32=0.0f0,
                save_dir::String="checkpoints", save_every::Int=5,
                warmup_epochs::Int=0, args=nothing)

    params = (cell = ps_cell, head = ps_head)

    if initial_opt_state !== nothing
        opt_state = initial_opt_state
        Optimisers.adjust!(opt_state, lr)
        println("  Resumed optimizer state, adjusted LR to $lr")
    else
        opt_state = Optimisers.setup(Optimisers.Adam(lr), params)
    end

    best_valid_f1 = initial_best_valid_f1
    best_params = deepcopy(params)
    best_epoch = start_epoch
    best_stats = nothing

    n_train = size(data.train_x, 3)

    for epoch in start_epoch:(epochs - 1)
        # ── LR warmup
        if warmup_epochs > 0 && epoch < warmup_epochs
            warmup_frac = (epoch + 1) / warmup_epochs
            current_lr = lr * (0.1f0 + 0.9f0 * Float32(warmup_frac))
            Optimisers.adjust!(opt_state, current_lr)
        elseif warmup_epochs > 0 && epoch == warmup_epochs
            Optimisers.adjust!(opt_state, lr)
        end

        # ── Evaluate
        valid_loss, valid_f1, valid_prec, valid_recall = evaluate(cell, head, params.cell, params.head,
                                                                   st_cell, st_head, data.valid_x, data.valid_y)
        test_loss, test_f1, _, _ = evaluate(cell, head, params.cell, params.head,
                                             st_cell, st_head, data.test_x, data.test_y)

        # ── Model selection (by valid F1)
        if valid_f1 > best_valid_f1 && epoch > start_epoch
            best_valid_f1 = valid_f1
            best_params = deepcopy(params)
            best_epoch = epoch
            best_stats = (0.0f0, 0.0f0, valid_loss, valid_f1, test_loss, test_f1)
            best_path = joinpath(save_dir, "srnn_ozone_best.jld2")
            save_checkpoint(best_path, best_params, opt_state, epoch,
                            best_valid_f1, args)
        end

        # ── Train one epoch
        perm = randperm(n_train)
        n_batches = div(n_train, batch_size)
        epoch_losses = Float32[]
        epoch_preds = Int[]
        epoch_labels = Int[]

        for b in 1:n_batches
            b_start = (b - 1) * batch_size + 1
            b_end = b * batch_size
            batch_idx = perm[b_start:b_end]

            x_batch = data.train_x[:, :, batch_idx]
            y_batch = data.train_y[:, batch_idx]  # (T, B) per-timestep

            loss_val, grads = Zygote.withgradient(params) do p
                batch_loss(cell, head, p.cell, p.head,
                           st_cell, st_head, x_batch, y_batch)
            end

            opt_state, params = Optimisers.update(opt_state, params, grads[1])
            push!(epoch_losses, loss_val)
        end

        train_loss = mean(epoch_losses)

        # ── Periodic checkpoint
        if save_every > 0 && epoch > start_epoch && epoch % save_every == 0
            periodic_path = joinpath(save_dir, @sprintf("srnn_ozone_epoch_%03d.jld2", epoch))
            save_checkpoint(periodic_path, params, opt_state, epoch,
                            best_valid_f1, args)
        end

        # ── Log (report F1 instead of accuracy)
        @printf("Epochs %03d, train loss: %0.2f, valid loss: %0.2f, valid F1: %0.4f (prec: %0.2f, recall: %0.2f), test loss: %0.2f, test F1: %0.4f\n",
            epoch, train_loss,
            valid_loss, valid_f1, valid_prec * 100, valid_recall * 100,
            test_loss, test_f1)

        if !isfinite(train_loss)
            println("NaN detected, stopping training.")
            break
        end
    end

    if best_stats !== nothing
        tl, ta, vl, vf1, tel, tef1 = best_stats
        @printf("Best epoch %03d, valid F1: %0.4f, test F1: %0.4f\n",
            best_epoch, vf1, tef1)
    end

    return best_params
end

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

function main()
    args = parse_args()
    println("Ozone Training — SRNNCell ($(args.solver), batched BPTT)")
    println("  Model size: $(args.model_size), n_E: $(args.n_E), n_I: $(args.model_size - args.n_E)")
    println("  per_neuron: $(args.per_neuron)")
    println("  SFA timescales (n_a_E): $(args.n_a), STD (n_b_E): $(args.n_b)")
    println("  Solver: $(args.solver), h: $(args.h), unfolds: $(args.unfolds)")
    println("  Readout: $(args.readout_mode)")
    println("  LR: $(args.lr), Epochs: $(args.epochs), Batch: $(args.batch_size)")
    println("  Save dir: $(args.save_dir), Save every: $(args.save_every) epochs")
    if !isempty(args.resume_path)
        println("  Resuming from: $(args.resume_path)")
    end

    Random.seed!(args.seed)
    rng = MersenneTwister(args.seed)
    println("  Random seed: $(args.seed)")

    data = load_ozone_data()

    cell, head, ps_cell, st_cell, ps_head, st_head = build_model(args, rng)

    start_epoch = 0
    initial_opt_state = nothing
    initial_best_valid_f1 = 0.0f0

    if !isempty(args.resume_path)
        println("\nLoading checkpoint: $(args.resume_path)")
        ckpt = load_checkpoint(args.resume_path)
        ps_cell = ckpt.params.cell
        ps_head = ckpt.params.head
        initial_opt_state = ckpt.opt_state
        start_epoch = ckpt.epoch + 1
        initial_best_valid_f1 = Float32(ckpt.best_valid_f1)
        println("  Loaded epoch $(ckpt.epoch), best valid F1: $(round(ckpt.best_valid_f1; digits=4))")
        println("  Resuming from epoch $start_epoch with LR $(args.lr)")
    end

    n_cell_params = sum(length(getproperty(ps_cell, k)) for k in keys(ps_cell))
    n_head_params = args.model_size * N_CLASSES + N_CLASSES
    println("  Cell params: $n_cell_params")
    println("  Head params: $n_head_params")
    println("  Total params: $(n_cell_params + n_head_params)")
    println("  State dim: $(cell.state_dim)")

    if isempty(args.resume_path)
        println("\nGradient smoke test (batched)...")
        test_x = data.train_x[:, :, 1:2]
        test_y = data.train_y[:, 1:2]

        test_loss, test_grads = Zygote.withgradient((cell=ps_cell, head=ps_head)) do p
            batch_loss(cell, head, p.cell, p.head, st_cell, st_head, test_x, test_y)
        end
        println("  Initial loss: $(@sprintf("%.4f", test_loss)) (expected ~0.69 = -log(1/2))")

        cell_grad = test_grads[1].cell
        head_grad = test_grads[1].head
        for k in keys(cell_grad)
            g = getproperty(cell_grad, k)
            if g === nothing
                println("  WARNING: SRNN gradient for $k is nothing!")
            end
        end
        println("  All SRNN gradients present ✓")
        println("  Head weight gradient norm: $(sum(abs2, head_grad.weight))")
        println("  Head bias gradient norm: $(sum(abs2, head_grad.bias))")
    end

    println("\nStarting training...\n")
    best_params = train!(cell, head, ps_cell, ps_head, st_cell, st_head, data;
                         epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                         start_epoch=start_epoch,
                         initial_opt_state=initial_opt_state,
                         initial_best_valid_f1=initial_best_valid_f1,
                         save_dir=args.save_dir, save_every=args.save_every,
                         warmup_epochs=args.warmup_epochs, args=args)
end

main()
