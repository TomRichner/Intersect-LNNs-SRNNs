# train_gesture.jl — Gesture Phase Recognition with configurable model (batched BPTT via Zygote)
#
# Adapted from: gesture.py (Hasani et al.)
# Supports: srnn, ltc (via --model flag).
#
# Usage:
#   julia --project=JuliaLang JuliaLang/scripts/train_gesture.jl [--epochs 200] [--size 32] [--lr 0.01] [--bs 256]
#
# Model selection:
#   --model <name>     Model type: srnn, ltc (required)
#
# SRNN-specific flags (ignored for other models):
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
using CSV, DataFrames
using Lux, NNlib, Zygote, Optimisers
using JLD2

# ── Include model registry ────────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "src", "model_registry.jl"))
include(joinpath(@__DIR__, "..", "src", "training_utils.jl"))

# ── Configuration ───────────────────────────────────────────────────────
const SEQ_LEN = 32
const N_FEATURES = 32     # 32 EMG sensor channels
const N_CLASSES = 5      # D, P, S, H, R

const TRAINING_FILES = [
    "a3_va3.csv", "b1_va3.csv", "b3_va3.csv", "c1_va3.csv",
    "c3_va3.csv", "a2_va3.csv", "a1_va3.csv",
]

# Parse simple command-line args
function parse_args()
    model = ""
    epochs = 200
    model_size = 32
    lr = 0.01f0
    batch_size = 256
    n_E = -1  # sentinel: will default to model_size ÷ 2
    n_a_E = 3
    n_a_I = 0
    n_b_E = 1
    n_b_I = 0
    unfolds = 6
    h = Float32(1 / 50)
    readout_mode = :synaptic
    solver = :semi_implicit
    per_neuron = false
    dales = false
    seed = 42
    save_dir = joinpath(@__DIR__, "..", "checkpoints")
    resume_path = ""
    save_every = 5
    warmup_epochs = 0

    for i in eachindex(ARGS)
        if ARGS[i] == "--model" && i < length(ARGS)
            model = ARGS[i+1]
        elseif ARGS[i] == "--epochs" && i < length(ARGS)
            epochs = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--size" && i < length(ARGS)
            model_size = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--lr" && i < length(ARGS)
            lr = parse(Float32, ARGS[i+1])
        elseif ARGS[i] == "--bs" && i < length(ARGS)
            batch_size = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--ne" && i < length(ARGS)
            n_E = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--n_a_E" && i < length(ARGS)
            n_a_E = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--n_a_I" && i < length(ARGS)
            n_a_I = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--n_b_E" && i < length(ARGS)
            n_b_E = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--n_b_I" && i < length(ARGS)
            n_b_I = parse(Int, ARGS[i+1])
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
        elseif ARGS[i] == "--dales"
            dales = true
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

    # Default n_E to half of model_size
    if n_E < 0
        n_E = model_size ÷ 2
    end

    if isempty(model)
        error("--model is required (srnn, ltc)")
    end

    return (; model, epochs, model_size, lr, batch_size, n_E, n_a_E, n_a_I, n_b_E, n_b_I,
        unfolds, h, readout_mode, solver, per_neuron, dales, seed,
        save_dir, resume_path, save_every, warmup_epochs)
end

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

"""
    load_trace(filepath) → (x::Matrix{Float64}, y::Vector{Int})

Load one gesture CSV. Returns features (N×32) and labels (N,) with
Phase mapped: D→1, P→2, S→3, H→4, R→5 (Julia 1-indexed).
"""
function load_trace(filepath::String)
    df = CSV.read(filepath, DataFrame)
    # Features: first 32 columns (numeric sensor channels)
    x = Float64.(Matrix(df[:, 1:N_FEATURES]))
    # Labels: last column "Phase" — map string to integer
    phase_map = Dict("D" => 1, "P" => 2, "S" => 3, "H" => 4, "R" => 5)
    y = [phase_map[String(p)] for p in df.Phase]
    return x, y
end

"""
    cut_in_sequences_gesture(x, y, seq_len; interleaved=false)

Non-overlapping window cut with optional interleaved half-overlap copies.
Returns a vector of (x_seq, y_seq) tuples.
"""
function cut_in_sequences_gesture(x::Matrix{Float64}, y::Vector{Int}, seq_len::Int;
    interleaved::Bool=false)
    n_samples = size(x, 1)
    num_sequences = n_samples ÷ seq_len
    sequences = Tuple{Matrix{Float64},Vector{Int}}[]

    for s in 0:(num_sequences-1)
        start = s * seq_len + 1
        stop = start + seq_len - 1
        push!(sequences, (x[start:stop, :], y[start:stop]))

        if interleaved && s < num_sequences - 1
            # Half-overlap interleaved copy
            i_start = start + seq_len ÷ 2
            i_stop = i_start + seq_len - 1
            push!(sequences, (x[i_start:i_stop, :], y[i_start:i_stop]))
        end
    end
    return sequences
end

struct GestureData
    train_x::Array{Float32,3}    # (features, seq_len, N_train)
    train_y::Matrix{Int}          # (seq_len, N_train)
    valid_x::Array{Float32,3}    # (features, seq_len, N_valid)
    valid_y::Matrix{Int}          # (seq_len, N_valid)
    test_x::Array{Float32,3}     # (features, seq_len, N_test)
    test_y::Matrix{Int}           # (seq_len, N_test)
end

function load_gesture_data(; data_dir=joinpath(@__DIR__, "..", "data", "gesture"))
    println("Loading Gesture data from: $data_dir")

    # Load and cut all training files
    all_sequences = Tuple{Matrix{Float64},Vector{Int}}[]
    for f in TRAINING_FILES
        filepath = joinpath(data_dir, f)
        x, y = load_trace(filepath)
        seqs = cut_in_sequences_gesture(x, y, SEQ_LEN; interleaved=true)
        append!(all_sequences, seqs)
        println("  $f: $(size(x, 1)) samples → $(length(seqs)) sequences")
    end

    # Stack into 3D arrays: (features, seq_len, N_total) — matching our convention
    N_total = length(all_sequences)
    all_x = Array{Float32,3}(undef, N_FEATURES, SEQ_LEN, N_total)
    all_y = Matrix{Int}(undef, SEQ_LEN, N_total)
    for (i, (sx, sy)) in enumerate(all_sequences)
        all_x[:, :, i] .= Float32.(sx')   # transpose: rows→features, cols→time
        all_y[:, i] .= sy
    end

    # Z-score normalize across all data
    flat_x = reshape(all_x, N_FEATURES, :)   # (features, seq_len * N_total)
    mean_x = mean(flat_x, dims=2)            # (features, 1)
    std_x = std(flat_x, dims=2)             # (features, 1)
    all_x = (all_x .- mean_x) ./ std_x

    println("  Total sequences: $N_total")

    # 3-way split (matching Python: seed 23489, 10% valid, 15% test, 75% train)
    perm = randperm(MersenneTwister(23489), N_total)
    n_valid = div(N_total * 10, 100)   # 10%
    n_test = div(N_total * 15, 100)   # 15%

    valid_idx = perm[1:n_valid]
    test_idx = perm[n_valid+1:n_valid+n_test]
    train_idx = perm[n_valid+n_test+1:end]

    println("  Split: $(length(train_idx)) train, $(length(valid_idx)) valid, $(length(test_idx)) test")

    return GestureData(
        all_x[:, :, train_idx], all_y[:, train_idx],
        all_x[:, :, valid_idx], all_y[:, valid_idx],
        all_x[:, :, test_idx], all_y[:, test_idx],
    )
end

# ═══════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════

function build_model(args, rng)
    cell, ps_cell, st_cell = build_cell(args.model, args.model_size, N_FEATURES, args, rng)
    head = Lux.Dense(hidden_size(cell) => N_CLASSES;
        init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32)
    ps_head, st_head = Lux.setup(rng, head)

    return cell, head, ps_cell, st_cell, ps_head, st_head
end

# ── Batched forward pass ────────────────────────────────────────────────
function forward_batch(cell, head, ps_cell, ps_head, st_cell, st_head, x_batch)
    B = size(x_batch, 3)
    S = initial_state(cell, B)

    for t in 1:size(x_batch, 2)
        u_t = @view x_batch[:, t, :]
        st_d = merge(st_cell, (input=u_t,))
        S, _ = cell(S, ps_cell, st_d)
    end

    obs = readout(cell, S, ps_cell)
    logits, _ = head(obs, ps_head, st_head)
    return logits
end

# ── Batched cross-entropy loss ──────────────────────────────────────────
function batch_loss(cell, head, ps_cell, ps_head, st_cell, st_head, x_batch, y_labels)
    logits = forward_batch(cell, head, ps_cell, ps_head, st_cell, st_head, x_batch)
    log_probs = logits .- logsumexp_batch(logits)
    B = length(y_labels)
    loss = zero(eltype(logits))
    for i in 1:B
        loss -= log_probs[y_labels[i], i]
    end
    return loss / B
end

function logsumexp_batch(x::AbstractMatrix)
    m = maximum(x, dims=1)
    return m .+ log.(sum(exp.(x .- m), dims=1))
end

# ═══════════════════════════════════════════════════════════════════════
# EVALUATION (batched)
# ═══════════════════════════════════════════════════════════════════════

function evaluate(cell, head, ps_cell, ps_head, st_cell, st_head,
    data_x::Array{Float32,3}, data_y::Matrix{Int};
    eval_batch_size::Int=128)
    n = size(data_x, 3)
    total_loss = 0.0f0
    correct = 0

    n_batches = cld(n, eval_batch_size)
    for b in 1:n_batches
        b_start = (b - 1) * eval_batch_size + 1
        b_end = min(b * eval_batch_size, n)
        batch_x = @view data_x[:, :, b_start:b_end]
        batch_labels = @view data_y[end, b_start:b_end]
        B = b_end - b_start + 1

        logits = forward_batch(cell, head, ps_cell, ps_head, st_cell, st_head, batch_x)
        log_probs = logits .- logsumexp_batch(logits)
        for i in 1:B
            total_loss -= log_probs[batch_labels[i], i]
        end
        preds = vec(getindex.(argmax(logits, dims=1), 1))
        correct += sum(preds .== batch_labels)
    end

    return total_loss / n, correct / n
end

# ═══════════════════════════════════════════════════════════════════════
# CHECKPOINTING
# ═══════════════════════════════════════════════════════════════════════

function save_checkpoint(path, params, opt_state, epoch, best_valid_acc, args)
    mkpath(dirname(path))
    jldsave(path;
        params=params,
        opt_state=opt_state,
        epoch=epoch,
        best_valid_acc=best_valid_acc,
        args=args,
    )
    println("  💾 Checkpoint saved: $path (epoch $epoch, valid acc $(round(best_valid_acc * 100; digits=2))%)")
end

function load_checkpoint(path)
    data = jldopen(path, "r") do f
        (
            params=f["params"],
            opt_state=f["opt_state"],
            epoch=f["epoch"],
            best_valid_acc=f["best_valid_acc"],
            args=f["args"],
        )
    end
    return data
end

function adjust_lr!(opt_state, new_lr)
    Optimisers.adjust!(opt_state, new_lr)
end

# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

function train!(cell, head, ps_cell, ps_head, st_cell, st_head, data::GestureData;
    epochs::Int=200, lr::Float32=0.01f0, batch_size::Int=256,
    start_epoch::Int=0, initial_opt_state=nothing,
    initial_best_valid_acc::Float32=0.0f0,
    save_dir::String="checkpoints", save_every::Int=5,
    warmup_epochs::Int=0, args=nothing)

    params = (cell=ps_cell, head=ps_head)

    if initial_opt_state !== nothing
        opt_state = initial_opt_state
        Optimisers.adjust!(opt_state, lr)
        println("  Resumed optimizer state, adjusted LR to $lr")
    else
        opt_state = Optimisers.setup(Optimisers.Adam(lr), params)
    end

    best_valid_acc = initial_best_valid_acc
    best_params = deepcopy(params)
    best_epoch = start_epoch
    best_stats = nothing

    n_train = size(data.train_x, 3)

    for epoch in start_epoch:(epochs-1)
        # ── LR schedule (warmup → hold → taper)
        current_lr = lr_schedule(epoch, epochs)
        Optimisers.adjust!(opt_state, current_lr)

        # ── Evaluate
        valid_loss, valid_acc = evaluate(cell, head, params.cell, params.head,
            st_cell, st_head, data.valid_x, data.valid_y)
        test_loss, test_acc = evaluate(cell, head, params.cell, params.head,
            st_cell, st_head, data.test_x, data.test_y)

        # ── Model selection (by valid accuracy)
        if valid_acc > best_valid_acc && epoch > start_epoch
            best_valid_acc = valid_acc
            best_params = deepcopy(params)
            best_epoch = epoch
            best_stats = (0.0f0, 0.0f0, valid_loss, valid_acc, test_loss, test_acc)
            best_path = joinpath(save_dir, "$(args.model)_gesture_best.jld2")
            save_checkpoint(best_path, best_params, opt_state, epoch,
                best_valid_acc, args)
        end

        # ── Train one epoch
        perm = randperm(n_train)
        n_batches = div(n_train, batch_size)
        epoch_losses = Float32[]
        epoch_correct = 0
        epoch_total = 0

        for b in 1:n_batches
            b_start = (b - 1) * batch_size + 1
            b_end = b * batch_size
            batch_idx = perm[b_start:b_end]

            x_batch = data.train_x[:, :, batch_idx]
            y_batch = data.train_y[end, batch_idx]

            loss_val, grads = Zygote.withgradient(params) do p
                batch_loss(cell, head, p.cell, p.head,
                    st_cell, st_head, x_batch, y_batch)
            end

            opt_state, params = Optimisers.update(opt_state, params, grads[1])
            push!(epoch_losses, loss_val)

            logits = forward_batch(cell, head, params.cell, params.head,
                st_cell, st_head, x_batch)
            preds = vec(getindex.(argmax(logits, dims=1), 1))
            epoch_correct += sum(preds .== y_batch)
            epoch_total += batch_size
        end

        train_loss = mean(epoch_losses)
        train_acc = epoch_correct / max(epoch_total, 1)

        # ── Periodic checkpoint
        if save_every > 0 && epoch > start_epoch && epoch % save_every == 0
            periodic_path = joinpath(save_dir, "$(args.model)_gesture_epoch_$(lpad(epoch, 3, '0')).jld2")
            save_checkpoint(periodic_path, params, opt_state, epoch,
                best_valid_acc, args)
        end

        # ── Log
        @printf("Epochs %03d, train loss: %0.2f, train accuracy: %0.2f%%, valid loss: %0.2f, valid accuracy: %0.2f%%, test loss: %0.2f, test accuracy: %0.2f%%\n",
            epoch, train_loss, train_acc * 100,
            valid_loss, valid_acc * 100,
            test_loss, test_acc * 100)

        if !isfinite(train_loss)
            println("NaN detected, stopping training.")
            break
        end
    end

    if best_stats !== nothing
        tl, ta, vl, va, tel, tea = best_stats
        @printf("Best epoch %03d, train loss: %0.2f, train accuracy: %0.2f%%, valid loss: %0.2f, valid accuracy: %0.2f%%, test loss: %0.2f, test accuracy: %0.2f%%\n",
            best_epoch, tl, ta * 100, vl, va * 100, tel, tea * 100)
    end

    return best_params
end

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

function main()
    args = parse_args()
    println("Gesture Training — $(uppercase(args.model)) ($(args.solver), batched BPTT)")
    println("  Model: $(args.model), size: $(args.model_size)")
    println("  per_neuron: $(args.per_neuron)")
    println("  SFA: n_a_E=$(args.n_a_E) n_a_I=$(args.n_a_I), STD: n_b_E=$(args.n_b_E) n_b_I=$(args.n_b_I)")
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

    data = load_gesture_data()

    cell, head, ps_cell, st_cell, ps_head, st_head = build_model(args, rng)

    start_epoch = 0
    initial_opt_state = nothing
    initial_best_valid_acc = 0.0f0

    if !isempty(args.resume_path)
        println("\nLoading checkpoint: $(args.resume_path)")
        ckpt = load_checkpoint(args.resume_path)
        ps_cell = ckpt.params.cell
        ps_head = ckpt.params.head
        initial_opt_state = ckpt.opt_state
        start_epoch = ckpt.epoch + 1
        initial_best_valid_acc = Float32(ckpt.best_valid_acc)
        println("  Loaded epoch $(ckpt.epoch), best valid acc: $(round(ckpt.best_valid_acc * 100; digits=2))%")
        println("  Resuming from epoch $start_epoch with LR $(args.lr)")
    end

    n_cell_params = sum(length(getproperty(ps_cell, k)) for k in propertynames(ps_cell))
    n_head_params = args.model_size * N_CLASSES + N_CLASSES
    println("  Cell params: $n_cell_params")
    println("  Head params: $n_head_params")
    println("  Total params: $(n_cell_params + n_head_params)")
    if hasproperty(cell, :state_dim)
        println("  State dim: $(cell.state_dim)")
    end

    if isempty(args.resume_path)
        println("\nGradient smoke test (batched)...")
        test_x = data.train_x[:, :, 1:2]
        test_y = data.train_y[end, 1:2]

        test_loss, test_grads = Zygote.withgradient((cell=ps_cell, head=ps_head)) do p
            batch_loss(cell, head, p.cell, p.head, st_cell, st_head, test_x, test_y)
        end
        println("  Initial loss: $(@sprintf("%.4f", test_loss)) (expected ~1.61 = -log(1/5))")

        cell_grad = test_grads[1].cell
        head_grad = test_grads[1].head
        for k in keys(cell_grad)
            g = getproperty(cell_grad, k)
            if g === nothing
                println("  WARNING: $(args.model) gradient for $k is nothing!")
            end
        end
        println("  All $(args.model) gradients present ✓")
        println("  Head weight gradient norm: $(sum(abs2, head_grad.weight))")
        println("  Head bias gradient norm: $(sum(abs2, head_grad.bias))")
    end

    println("\nStarting training...\n")
    best_params = train!(cell, head, ps_cell, ps_head, st_cell, st_head, data;
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        start_epoch=start_epoch,
        initial_opt_state=initial_opt_state,
        initial_best_valid_acc=initial_best_valid_acc,
        save_dir=args.save_dir, save_every=args.save_every,
        warmup_epochs=args.warmup_epochs, args=args)
end

main()
