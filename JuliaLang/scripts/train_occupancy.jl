# train_occupancy.jl — Occupancy Detection with configurable model (batched BPTT via Zygote)
#
# Adapted from: train_har_srnn.jl
# Supports: srnn, ltc (via --model flag).
#
# Usage:
#   julia --project=JuliaLang JuliaLang/scripts/train_occupancy.jl [--epochs 200] [--size 32] [--lr 0.01] [--bs 256]
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
const SEQ_LEN    = 16
const N_FEATURES = 5     # Temperature, Humidity, Light, CO2, HumidityRatio
const N_CLASSES  = 2     # Occupied / Not Occupied

# Parse simple command-line args
function parse_args()
    model = ""
    epochs = 200
    model_size = 32
    lr = 0.01f0
    batch_size = 256
    n_E = -1  # sentinel: will default to model_size ÷ 2
    n_a = 3
    n_b = 0
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

    return (; model, epochs, model_size, lr, batch_size, n_E, n_a, n_b,
              unfolds, h, readout_mode, solver, per_neuron, dales, seed,
              save_dir, resume_path, save_every, warmup_epochs)
end

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

function cut_in_sequences(x::Matrix, y::Vector, seq_len::Int; inc::Int=1)
    n_samples = size(x, 1)
    n_seqs = length(0:inc:(n_samples - seq_len - 1))

    # Pre-allocate 3D array: (features, seq_len, n_seqs)
    seqs_x = Array{Float32, 3}(undef, size(x, 2), seq_len, n_seqs)
    seqs_y = Matrix{Int}(undef, seq_len, n_seqs)

    idx = 0
    for s in 0:inc:(n_samples - seq_len - 1)
        idx += 1
        start = s + 1  # Julia 1-indexed
        stop = start + seq_len - 1
        seqs_x[:, :, idx] .= Float32.(x[start:stop, :]')   # transpose: (features, seq_len)
        seqs_y[:, idx] .= y[start:stop]
    end
    return seqs_x, seqs_y
end

struct OccupancyData
    train_x::Array{Float32, 3}    # (features, seq_len, N_train)
    train_y::Matrix{Int}          # (seq_len, N_train)
    valid_x::Array{Float32, 3}    # (features, seq_len, N_valid)
    valid_y::Matrix{Int}          # (seq_len, N_valid)
    test_x::Array{Float32, 3}     # (features, seq_len, N_test)
    test_y::Matrix{Int}           # (seq_len, N_test)
end

function read_occupancy_csv(filepath::String)
    # The CSV has 7 header columns but 8 data columns (extra leading row index).
    # This shifts all column names by one position in CSV.jl's parsing.
    # So we read by column position: cols 3-7 = sensor data, col 8 = occupancy.
    df = CSV.read(filepath, DataFrame; silencewarnings=true)
    ncols = ncol(df)

    # Columns 3-7 are the actual sensor values (Temperature, Humidity, Light, CO2, HumidityRatio)
    x = Float64.(Matrix(df[:, 3:7]))
    # Last column is Occupancy (0 or 1) → add 1 for Julia 1-indexing (1 or 2)
    y = Int.(df[:, ncols]) .+ 1
    return x, y
end

function load_occupancy_data(; data_dir=joinpath(@__DIR__, "..", "data", "occupancy"))
    println("Loading Occupancy data from: $data_dir")

    # Load the 3 CSV files
    train_x_raw, train_y_raw = read_occupancy_csv(joinpath(data_dir, "datatraining.txt"))
    test0_x_raw, test0_y_raw = read_occupancy_csv(joinpath(data_dir, "datatest.txt"))
    test1_x_raw, test1_y_raw = read_occupancy_csv(joinpath(data_dir, "datatest2.txt"))

    println("  Train: $(size(train_x_raw, 1)) samples × $(size(train_x_raw, 2)) features")
    println("  Test0: $(size(test0_x_raw, 1)) samples, Test1: $(size(test1_x_raw, 1)) samples")

    # Z-score normalization (fit on train only)
    mean_x = mean(train_x_raw, dims=1)
    std_x  = std(train_x_raw, dims=1)
    train_x_raw = (train_x_raw .- mean_x) ./ std_x
    test0_x_raw = (test0_x_raw .- mean_x) ./ std_x
    test1_x_raw = (test1_x_raw .- mean_x) ./ std_x

    # Window into sequences
    train_seqs_x, train_seqs_y = cut_in_sequences(train_x_raw, train_y_raw, SEQ_LEN; inc=1)
    test0_seqs_x, test0_seqs_y = cut_in_sequences(test0_x_raw, test0_y_raw, SEQ_LEN; inc=8)
    test1_seqs_x, test1_seqs_y = cut_in_sequences(test1_x_raw, test1_y_raw, SEQ_LEN; inc=8)

    # Concatenate both test sets
    test_seqs_x = cat(test0_seqs_x, test1_seqs_x; dims=3)
    test_seqs_y = cat(test0_seqs_y, test1_seqs_y; dims=2)

    println("  Total training sequences: $(size(train_seqs_x, 3))")
    println("  Total test sequences:     $(size(test_seqs_x, 3))")

    # Validation split (10%, fixed seed matching Python)
    n_total = size(train_seqs_x, 3)
    perm = randperm(MersenneTwister(893429), n_total)
    n_valid = div(n_total, 10)

    valid_idx = perm[1:n_valid]
    train_idx = perm[n_valid+1:end]

    println("  Validation split: $n_valid, training split: $(length(train_idx))")

    return OccupancyData(
        train_seqs_x[:, :, train_idx], train_seqs_y[:, train_idx],
        train_seqs_x[:, :, valid_idx], train_seqs_y[:, valid_idx],
        test_seqs_x, test_seqs_y,
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
# x_batch: (features, seq_len, B)
# Returns: logits (N_CLASSES, B)
function forward_batch(cell, head, ps_cell, ps_head, st_cell, st_head, x_batch)
    B = size(x_batch, 3)
    S = initial_state(cell, B)   # (state_dim, B)

    for t in 1:size(x_batch, 2)
        u_t = @view x_batch[:, t, :]   # (features, B)
        st_d = merge(st_cell, (input = u_t,))
        S, _ = cell(S, ps_cell, st_d)   # batched dispatch → (state_dim, B)
    end

    # Readout: extract (n, B) observation from full state
    obs = readout(cell, S, ps_cell)

    # Dense head: (n, B) → (N_CLASSES, B)
    logits, _ = head(obs, ps_head, st_head)
    return logits
end

# ── Batched cross-entropy loss ──────────────────────────────────────────
function batch_loss(cell, head, ps_cell, ps_head, st_cell, st_head, x_batch, y_labels)
    logits = forward_batch(cell, head, ps_cell, ps_head, st_cell, st_head, x_batch)
    # logits: (N_CLASSES, B), y_labels: (B,) — last time step labels, 1-indexed
    log_probs = logits .- logsumexp_batch(logits)  # (N_CLASSES, B)
    B = length(y_labels)
    loss = zero(eltype(logits))
    for i in 1:B
        loss -= log_probs[y_labels[i], i]
    end
    return loss / B
end

# Stable logsumexp over dim=1 for a matrix
function logsumexp_batch(x::AbstractMatrix)
    m = maximum(x, dims=1)   # (1, B)
    return m .+ log.(sum(exp.(x .- m), dims=1))  # (1, B)
end

# ═══════════════════════════════════════════════════════════════════════
# EVALUATION (batched)
# ═══════════════════════════════════════════════════════════════════════

function evaluate(cell, head, ps_cell, ps_head, st_cell, st_head,
                  data_x::Array{Float32, 3}, data_y::Matrix{Int};
                  eval_batch_size::Int=128)
    n = size(data_x, 3)
    total_loss = 0.0f0
    correct = 0

    n_batches = cld(n, eval_batch_size)  # ceiling division
    for b in 1:n_batches
        b_start = (b - 1) * eval_batch_size + 1
        b_end = min(b * eval_batch_size, n)
        batch_x = @view data_x[:, :, b_start:b_end]
        batch_labels = @view data_y[end, b_start:b_end]   # last time step
        B = b_end - b_start + 1

        logits = forward_batch(cell, head, ps_cell, ps_head, st_cell, st_head, batch_x)
        # Loss
        log_probs = logits .- logsumexp_batch(logits)
        for i in 1:B
            total_loss -= log_probs[batch_labels[i], i]
        end
        # Accuracy
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
        params = params,
        opt_state = opt_state,
        epoch = epoch,
        best_valid_acc = best_valid_acc,
        args = args,
    )
    println("  💾 Checkpoint saved: $path (epoch $epoch, valid acc $(round(best_valid_acc * 100; digits=2))%)")
end

function load_checkpoint(path)
    data = jldopen(path, "r") do f
        (
            params = f["params"],
            opt_state = f["opt_state"],
            epoch = f["epoch"],
            best_valid_acc = f["best_valid_acc"],
            args = f["args"],
        )
    end
    return data
end

"""
    adjust_lr!(opt_state, new_lr)

Walk the optimizer state tree and update Adam's learning rate.
"""
function adjust_lr!(opt_state, new_lr)
    Optimisers.adjust!(opt_state, new_lr)
end

# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

function train!(cell, head, ps_cell, ps_head, st_cell, st_head, data::OccupancyData;
                epochs::Int=200, lr::Float32=0.01f0, batch_size::Int=256,
                start_epoch::Int=0, initial_opt_state=nothing,
                initial_best_valid_acc::Float32=0.0f0,
                save_dir::String="checkpoints", save_every::Int=5,
                warmup_epochs::Int=0, args=nothing)

    # Combine parameters for gradient computation
    params = (cell = ps_cell, head = ps_head)

    # Set up optimizer (or use resumed state)
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

    for epoch in start_epoch:(epochs - 1)
        # ── LR schedule (warmup → hold → taper)
        current_lr = lr_schedule(epoch, epochs)
        Optimisers.adjust!(opt_state, current_lr)

        # ── Evaluate ────────────────────────────────────────────────
        valid_loss, valid_acc = evaluate(cell, head, params.cell, params.head,
                                         st_cell, st_head, data.valid_x, data.valid_y)
        test_loss, test_acc = evaluate(cell, head, params.cell, params.head,
                                       st_cell, st_head, data.test_x, data.test_y)

        # ── Model selection (by valid accuracy) ─────────────────────
        if valid_acc > best_valid_acc && epoch > start_epoch
            best_valid_acc = valid_acc
            best_params = deepcopy(params)
            best_epoch = epoch
            best_stats = (0.0f0, 0.0f0, valid_loss, valid_acc, test_loss, test_acc)
            # Save best checkpoint
            best_path = joinpath(save_dir, "$(args.model)_occupancy_best.jld2")
            save_checkpoint(best_path, best_params, opt_state, epoch,
                            best_valid_acc, args)
        end

        # ── Train one epoch ─────────────────────────────────────────
        perm = randperm(n_train)
        n_batches = div(n_train, batch_size)
        epoch_losses = Float32[]
        epoch_correct = 0
        epoch_total = 0

        for b in 1:n_batches
            b_start = (b - 1) * batch_size + 1
            b_end = b * batch_size
            batch_idx = perm[b_start:b_end]

            # Slice the batch
            x_batch = data.train_x[:, :, batch_idx]     # (features, seq_len, B)
            y_batch = data.train_y[end, batch_idx]       # (B,) last time step labels

            # Single gradient call over the whole batch
            loss_val, grads = Zygote.withgradient(params) do p
                batch_loss(cell, head, p.cell, p.head,
                           st_cell, st_head, x_batch, y_batch)
            end

            # Update parameters
            opt_state, params = Optimisers.update(opt_state, params, grads[1])
            push!(epoch_losses, loss_val)

            # Track accuracy from the same forward pass direction
            logits = forward_batch(cell, head, params.cell, params.head,
                                    st_cell, st_head, x_batch)
            preds = vec(getindex.(argmax(logits, dims=1), 1))
            epoch_correct += sum(preds .== y_batch)
            epoch_total += batch_size
        end

        train_loss = mean(epoch_losses)
        train_acc = epoch_correct / max(epoch_total, 1)

        # ── Periodic checkpoint ──────────────────────────────────────
        if save_every > 0 && epoch > start_epoch && epoch % save_every == 0
            periodic_path = joinpath(save_dir, "$(args.model)_occupancy_epoch_$(lpad(epoch, 3, '0')).jld2")
            save_checkpoint(periodic_path, params, opt_state, epoch,
                            best_valid_acc, args)
        end

        # ── Log ─────────────────────────────────────────────────────
        @printf("Epochs %03d, train loss: %0.2f, train accuracy: %0.2f%%, valid loss: %0.2f, valid accuracy: %0.2f%%, test loss: %0.2f, test accuracy: %0.2f%%\n",
            epoch, train_loss, train_acc * 100,
            valid_loss, valid_acc * 100,
            test_loss, test_acc * 100)

        # Early stopping on NaN
        if !isfinite(train_loss)
            println("NaN detected, stopping training.")
            break
        end
    end

    # Print best epoch
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
    println("Occupancy Training — $(uppercase(args.model)) ($(args.solver), batched BPTT)")
    println("  Model: $(args.model), size: $(args.model_size)")
    println("  per_neuron: $(args.per_neuron)")
    println("  SFA timescales (n_a_E): $(args.n_a), STD (n_b_E): $(args.n_b)")
    println("  Solver: $(args.solver), h: $(args.h), unfolds: $(args.unfolds)")
    println("  Readout: $(args.readout_mode)")
    println("  LR: $(args.lr), Epochs: $(args.epochs), Batch: $(args.batch_size)")
    println("  Save dir: $(args.save_dir), Save every: $(args.save_every) epochs")
    if !isempty(args.resume_path)
        println("  Resuming from: $(args.resume_path)")
    end

    # Seed global RNG (for batch shuffling) and create model init RNG
    Random.seed!(args.seed)
    rng = MersenneTwister(args.seed)
    println("  Random seed: $(args.seed)")

    # Load data
    data = load_occupancy_data()

    # Build model
    cell, head, ps_cell, st_cell, ps_head, st_head = build_model(args, rng)

    # Handle resume
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

    # Count parameters
    n_cell_params = sum(length(getproperty(ps_cell, k)) for k in propertynames(ps_cell))
    n_head_params = args.model_size * N_CLASSES + N_CLASSES
    println("  Cell params: $n_cell_params")
    println("  Head params: $n_head_params")
    println("  Total params: $(n_cell_params + n_head_params)")
    if hasproperty(cell, :state_dim)
        println("  State dim: $(cell.state_dim)")
    end

    # Gradient smoke test (only on fresh start)
    if isempty(args.resume_path)
        println("\nGradient smoke test (batched)...")
        test_x = data.train_x[:, :, 1:2]    # (features, seq_len, 2)
        test_y = data.train_y[end, 1:2]      # (2,)

        test_loss, test_grads = Zygote.withgradient((cell=ps_cell, head=ps_head)) do p
            batch_loss(cell, head, p.cell, p.head, st_cell, st_head, test_x, test_y)
        end
        println("  Initial loss: $(@sprintf("%.4f", test_loss)) (expected ~0.69 = -log(1/2))")

        # Check gradients are non-nothing
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

    # Train
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
