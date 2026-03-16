# train_person.jl — PersonActivity Classification with configurable model (batched BPTT via Zygote)
#
# Adapted from: person.py (Hasani et al.)
# 7-class per-timestep classification from 7 features (4 sensor one-hot + 3 accel).
# Dense(7) head at every timestep, cross-entropy loss, accuracy metric.
#
# Usage:
#   julia --project=JuliaLang JuliaLang/scripts/train_person.jl [--epochs 200] [--size 32] [--lr 0.001] [--bs 64]
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
using Lux, NNlib, Zygote, Optimisers
using JLD2

# ── Include model registry ────────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "src", "model_registry.jl"))
include(joinpath(@__DIR__, "..", "src", "training_utils.jl"))

# ── Configuration ───────────────────────────────────────────────────────
const SEQ_LEN    = 32
const N_FEATURES = 7      # 4 sensor one-hot + 3 accel
const N_CLASSES  = 7      # 7 activity classes

# Activity label mapping (matches Python class_map: 11 labels → 7 classes)
const CLASS_MAP = Dict{String,Int}(
    "lying down" => 0,
    "lying" => 0,
    "sitting down" => 1,
    "sitting" => 1,
    "standing up from lying" => 2,
    "standing up from sitting" => 2,
    "standing up from sitting on the ground" => 2,
    "walking" => 3,
    "falling" => 4,
    "on all fours" => 5,
    "sitting on the ground" => 6,
)

# Sensor ID → one-hot index mapping
const SENSOR_IDS = Dict{String,Int}(
    "010-000-024-033" => 1,
    "010-000-030-096" => 2,
    "020-000-033-111" => 3,
    "020-000-032-221" => 4,
)

# Parse simple command-line args
function parse_args()
    model = ""
    epochs = 200
    model_size = 32
    lr = 0.001f0
    batch_size = 64
    n_E = -1  # sentinel: will default to model_size ÷ 2
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

    if isempty(model)
        error("--model is required (srnn, ltc)")
    end

    return (; model, epochs, model_size, lr, batch_size, n_E, n_a, n_b,
              unfolds, h, readout_mode, solver, per_neuron, seed,
              save_dir, resume_path, save_every, warmup_epochs)
end

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

"""
    load_person_series(filepath) → (all_x, all_y)

Parse the ConfLongDemo_JSI.txt file into per-person time series.
Returns:
  all_x: Vector of matrices, each (T_person, 7) Float32
  all_y: Vector of vectors, each (T_person,) Int (0-indexed class labels)
"""
function load_person_series(filepath::String)
    all_x = Vector{Matrix{Float32}}()
    all_y = Vector{Vector{Int}}()

    series_x = Vector{Vector{Float32}}()
    series_y = Vector{Int}()
    current_person = ""

    for line in eachline(filepath)
        arr = split(line, ",")
        if length(arr) < 7
            break
        end

        person_id = String(arr[1])

        # When person changes, save the current series and start a new one
        if person_id != current_person && !isempty(series_x)
            push!(all_x, reduce(hcat, series_x)')  # (T, 7)
            push!(all_y, copy(series_y))
            empty!(series_x)
            empty!(series_y)
        end
        current_person = person_id

        # Sensor one-hot (4 dims)
        sensor_id = SENSOR_IDS[String(arr[2])]
        feature_sensor = zeros(Float32, 4)
        feature_sensor[sensor_id] = 1.0f0

        # Accelerometer features (3 dims)
        feature_accel = Float32[
            parse(Float32, arr[5]),
            parse(Float32, arr[6]),
            parse(Float32, arr[7]),
        ]

        # Full feature vector: [sensor_one_hot(4); accel(3)]
        feature = vcat(feature_sensor, feature_accel)
        push!(series_x, feature)

        # Label (strip whitespace/newline from last column)
        label_str = strip(String(arr[8]))
        push!(series_y, CLASS_MAP[label_str])
    end

    # Don't forget the last person
    if !isempty(series_x)
        push!(all_x, reduce(hcat, series_x)')  # (T, 7)
        push!(all_y, copy(series_y))
    end

    return all_x, all_y
end

"""
    cut_in_sequences(all_x, all_y, seq_len; inc=1)

Per-person sliding-window segmentation.
  all_x: Vector of (T_i, N_features) matrices
  all_y: Vector of (T_i,) label vectors
Returns:
  seqs_x: (N_features, seq_len, N_total_seqs) Float32
  seqs_y: (seq_len, N_total_seqs) Int32
"""
function cut_in_sequences(all_x::Vector{Matrix{Float32}},
                          all_y::Vector{Vector{Int}},
                          seq_len::Int; inc::Int=1)
    seq_x_list = Vector{Matrix{Float32}}()
    seq_y_list = Vector{Vector{Int32}}()

    for i in eachindex(all_x)
        x = all_x[i]  # (T, features)
        y = all_y[i]  # (T,)
        T = size(x, 1)

        for s in 0:inc:(T - seq_len - 1)
            start = s + 1  # Julia 1-indexed
            stop = start + seq_len - 1
            push!(seq_x_list, x[start:stop, :]')  # (features, seq_len)
            push!(seq_y_list, Int32.(y[start:stop]))   # (seq_len,)
        end
    end

    n_seqs = length(seq_x_list)
    # Stack into 3D/2D arrays
    seqs_x = Array{Float32, 3}(undef, N_FEATURES, seq_len, n_seqs)
    seqs_y = Matrix{Int32}(undef, seq_len, n_seqs)
    for i in 1:n_seqs
        seqs_x[:, :, i] .= seq_x_list[i]
        seqs_y[:, i] .= seq_y_list[i]
    end

    return seqs_x, seqs_y
end

struct PersonData
    train_x::Array{Float32, 3}    # (N_FEATURES, seq_len, N_train)
    train_y::Matrix{Int32}        # (seq_len, N_train)  — 0-indexed class labels
    valid_x::Array{Float32, 3}
    valid_y::Matrix{Int32}
    test_x::Array{Float32, 3}
    test_y::Matrix{Int32}
end

function load_person_data(; data_dir=joinpath(@__DIR__, "..", "data", "person"))
    println("Loading PersonActivity data from: $data_dir")

    filepath = joinpath(data_dir, "ConfLongDemo_JSI.txt")
    all_x, all_y = load_person_series(filepath)
    println("  Persons: $(length(all_x))")
    total_samples = sum(size(x, 1) for x in all_x)
    println("  Total samples across all persons: $total_samples")

    # Sliding window with inc = seq_len ÷ 2 (matching Python)
    inc = SEQ_LEN ÷ 2
    seqs_x, seqs_y = cut_in_sequences(all_x, all_y, SEQ_LEN; inc=inc)
    total_seqs = size(seqs_x, 3)
    println("  Total sequences (inc=$inc): $total_seqs")

    # 75/10/15 split with fixed seed (matching Python: np.random.RandomState(27731))
    perm = randperm(MersenneTwister(27731), total_seqs)
    valid_size = Int(floor(0.1 * total_seqs))
    test_size  = Int(floor(0.15 * total_seqs))
    train_size = total_seqs - valid_size - test_size

    valid_idx = perm[1:valid_size]
    test_idx  = perm[valid_size+1:valid_size+test_size]
    train_idx = perm[valid_size+test_size+1:end]

    println("  Train: $train_size, Valid: $valid_size, Test: $test_size")

    return PersonData(
        seqs_x[:, :, train_idx], seqs_y[:, train_idx],
        seqs_x[:, :, valid_idx], seqs_y[:, valid_idx],
        seqs_x[:, :, test_idx],  seqs_y[:, test_idx],
    )
end

# ═══════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════

function build_model(args, rng)
    cell, ps_cell, st_cell = build_cell(args.model, args.model_size, N_FEATURES, args, rng)
    # Classification head: Dense(n → 7), no activation (logits)
    head = Lux.Dense(hidden_size(cell) => N_CLASSES;
        init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32)
    ps_head, st_head = Lux.setup(rng, head)

    return cell, head, ps_cell, st_cell, ps_head, st_head
end

# ── Batched forward pass (per-timestep) ─────────────────────────────────
# x_batch: (N_FEATURES, seq_len, B)
# Returns: logits (N_CLASSES, seq_len, B) — class logits at each timestep
#
# Uses Zygote.Buffer to accumulate per-timestep outputs without triggering
# Zygote's mutation restriction on regular arrays.
function forward_batch(cell, head, ps_cell, ps_head, st_cell, st_head, x_batch)
    B = size(x_batch, 3)
    T = size(x_batch, 2)
    S = initial_state(cell, B)

    # Zygote.Buffer allows setindex! inside differentiated code
    buf = Zygote.Buffer(x_batch, N_CLASSES, T, B)

    for t in 1:T
        u_t = @view x_batch[:, t, :]
        st_d = merge(st_cell, (input = u_t,))
        S, _ = cell(S, ps_cell, st_d)

        # Readout + Dense at each timestep
        obs = readout(cell, S, ps_cell)      # (model_size, B)
        out, _ = head(obs, ps_head, st_head)  # (N_CLASSES, B)
        buf[:, t, :] = out                    # (N_CLASSES, B) into slice
    end

    return copy(buf)  # (N_CLASSES, seq_len, B)
end

# ── Stable logsumexp over dim=1 for a matrix ───────────────────────────
function logsumexp_batch(x::AbstractMatrix)
    m = maximum(x, dims=1)   # (1, B)
    return m .+ log.(sum(exp.(x .- m), dims=1))  # (1, B)
end

# ── Batched CE loss (per-timestep) ──────────────────────────────────────
# logits: (N_CLASSES, seq_len, B), targets: (seq_len, B) Int32 0-indexed
function batch_ce_loss(cell, head, ps_cell, ps_head, st_cell, st_head, x_batch, y_batch)
    logits = forward_batch(cell, head, ps_cell, ps_head, st_cell, st_head, x_batch)
    C, T, B = size(logits)
    logits_flat = reshape(logits, C, T * B)           # (7, T*B)
    log_probs = logits_flat .- logsumexp_batch(logits_flat)  # (7, T*B)
    targets_flat = reshape(y_batch, T * B)             # (T*B,)
    # Sum negative log-probs at true class (labels are 0-indexed)
    loss = zero(eltype(logits_flat))
    for i in 1:T*B
        loss -= log_probs[targets_flat[i] + 1, i]  # 0-indexed → 1-indexed
    end
    return loss / (T * B)
end

# ═══════════════════════════════════════════════════════════════════════
# EVALUATION (batched)
# ═══════════════════════════════════════════════════════════════════════

function evaluate(cell, head, ps_cell, ps_head, st_cell, st_head,
                  data_x::Array{Float32, 3}, data_y::Matrix{Int32};
                  eval_batch_size::Int=128)
    n = size(data_x, 3)
    total_loss = 0.0f0
    total_correct = 0
    total_count = 0

    n_batches = cld(n, eval_batch_size)
    for b in 1:n_batches
        b_start = (b - 1) * eval_batch_size + 1
        b_end = min(b * eval_batch_size, n)
        batch_x = @view data_x[:, :, b_start:b_end]
        batch_y = @view data_y[:, b_start:b_end]
        B = b_end - b_start + 1

        logits = forward_batch(cell, head, ps_cell, ps_head, st_cell, st_head, batch_x)
        C, T, Bb = size(logits)
        logits_flat = reshape(logits, C, T * Bb)
        log_probs = logits_flat .- logsumexp_batch(logits_flat)
        targets_flat = reshape(batch_y, T * Bb)

        # Loss: sum of negative log-probs
        for i in 1:T*Bb
            total_loss -= log_probs[targets_flat[i] + 1, i]
        end

        # Accuracy: argmax over class dim
        preds = argmax(logits_flat, dims=1)
        for i in 1:T*Bb
            if preds[1, i][1] - 1 == targets_flat[i]
                total_correct += 1
            end
        end
        total_count += T * Bb
    end

    loss = total_loss / total_count
    acc = total_correct / total_count
    return loss, acc
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

# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

function train!(cell, head, ps_cell, ps_head, st_cell, st_head, data::PersonData;
                epochs::Int=200, lr::Float32=0.001f0, batch_size::Int=64,
                start_epoch::Int=0, initial_opt_state=nothing,
                initial_best_valid_acc::Float32=0.0f0,
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

        # ── Model selection (by valid accuracy — higher is better) ────
        if valid_acc > best_valid_acc && epoch > start_epoch
            best_valid_acc = valid_acc
            best_params = deepcopy(params)
            best_epoch = epoch
            best_stats = (0.0f0, 0.0f0, valid_loss, valid_acc, test_loss, test_acc)
            best_path = joinpath(save_dir, "$(args.model)_person_best.jld2")
            save_checkpoint(best_path, best_params, opt_state, epoch,
                            best_valid_acc, args)
        end

        # ── Train one epoch ─────────────────────────────────────────
        perm = randperm(n_train)
        n_batches = div(n_train, batch_size)
        epoch_losses = Float32[]
        epoch_correct = 0
        epoch_count = 0

        for b in 1:n_batches
            b_start = (b - 1) * batch_size + 1
            b_end = b * batch_size
            batch_idx = perm[b_start:b_end]

            x_batch = data.train_x[:, :, batch_idx]     # (features, seq_len, B)
            y_batch = data.train_y[:, batch_idx]         # (seq_len, B)

            loss_val, grads = Zygote.withgradient(params) do p
                batch_ce_loss(cell, head, p.cell, p.head,
                               st_cell, st_head, x_batch, y_batch)
            end

            opt_state, params = Optimisers.update(opt_state, params, grads[1])
            push!(epoch_losses, loss_val)

            # Track accuracy from the same batch
            logits = forward_batch(cell, head, params.cell, params.head,
                                    st_cell, st_head, x_batch)
            C, T, Bb = size(logits)
            preds = argmax(reshape(logits, C, T * Bb), dims=1)
            targets_flat = reshape(y_batch, T * Bb)
            for i in 1:T*Bb
                if preds[1, i][1] - 1 == targets_flat[i]
                    epoch_correct += 1
                end
            end
            epoch_count += T * Bb

            # Batch progress (first batch + every 50)
            if b == 1 || b % 50 == 0
                @printf("  [batch %d/%d] loss: %.4f\n", b, n_batches, loss_val)
                flush(stdout)
            end
        end

        train_loss = mean(epoch_losses)
        train_acc = epoch_correct / max(epoch_count, 1)

        # ── Periodic checkpoint ──────────────────────────────────────
        if save_every > 0 && epoch > start_epoch && epoch % save_every == 0
            periodic_path = joinpath(save_dir, "$(args.model)_person_epoch_$(lpad(epoch, 3, '0')).jld2")
            save_checkpoint(periodic_path, params, opt_state, epoch,
                            best_valid_acc, args)
        end

        # ── Log ──────────────────────────────────────────────────────
        @printf("Epochs %03d, train loss: %0.4f, train acc: %0.2f%%, valid loss: %0.4f, valid acc: %0.2f%%, test loss: %0.4f, test acc: %0.2f%%\n",
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
        @printf("Best epoch %03d, train loss: %0.4f, train acc: %0.2f%%, valid loss: %0.4f, valid acc: %0.2f%%, test loss: %0.4f, test acc: %0.2f%%\n",
            best_epoch, tl, ta * 100, vl, va * 100, tel, tea * 100)
    end

    return best_params
end

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

function main()
    args = parse_args()
    println("Person Training — $(uppercase(args.model)) ($(args.solver), batched BPTT)")
    println("  Model: $(args.model), size: $(args.model_size)")
    println("  per_neuron: $(args.per_neuron)")
    println("  SFA timescales (n_a_E): $(args.n_a), STD (n_b_E): $(args.n_b)")
    println("  Solver: $(args.solver), h: $(args.h), unfolds: $(args.unfolds)")
    println("  Readout: $(args.readout_mode)")
    println("  LR: $(args.lr), Epochs: $(args.epochs), Batch: $(args.batch_size)")
    println("  Warmup: $(args.warmup_epochs) epochs")
    println("  Save dir: $(args.save_dir), Save every: $(args.save_every) epochs")
    if !isempty(args.resume_path)
        println("  Resuming from: $(args.resume_path)")
    end

    Random.seed!(args.seed)
    rng = MersenneTwister(args.seed)
    println("  Random seed: $(args.seed)")

    data = load_person_data()

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
        test_y = data.train_y[:, 1:2]

        test_loss, test_grads = Zygote.withgradient((cell=ps_cell, head=ps_head)) do p
            batch_ce_loss(cell, head, p.cell, p.head, st_cell, st_head, test_x, test_y)
        end
        println("  Initial CE loss: $(@sprintf("%.4f", test_loss)) (expected ~$(round(log(N_CLASSES); digits=2)) for random)")

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
