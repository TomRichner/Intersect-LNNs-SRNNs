# train_cheetah.jl — Half-Cheetah Autoregressive Dynamics with configurable model (batched BPTT via Zygote)
#
# Adapted from: cheetah.py (Hasani et al. 2021, Table 6)
# Vector autoregression: 17-dim observation at time t → predict 17-dim at t+1.
# Data: 25 MuJoCo rollouts stored as .npy files. File-based train/valid/test split.
#
# Usage:
#   julia --project=JuliaLang JuliaLang/scripts/train_cheetah.jl [--epochs 200] [--size 32] [--lr 0.01] [--bs 16]
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
using NPZ
using Lux, NNlib, Zygote, Optimisers
using JLD2

# ── Include model registry ────────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "src", "model_registry.jl"))
include(joinpath(@__DIR__, "..", "src", "training_utils.jl"))

# ── Configuration ───────────────────────────────────────────────────────
const SEQ_LEN    = 32
const OBS_SIZE   = 17     # MuJoCo HalfCheetah-v2 observation dim
const INC        = 10     # stride between sliding windows

# Parse simple command-line args
function parse_args()
    model = ""
    epochs = 200
    model_size = 32
    lr = 0.01f0
    batch_size = 16
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

"""
    cut_in_sequences(arr, seq_len; inc=10)

Create shifted input/target pairs for autoregressive prediction.
  arr: (T_steps, 17) raw rollout — Float32
  Returns: (seqs_x, seqs_y)
    seqs_x: Vector of (17, seq_len) — input  x[t:t+seq_len-1]
    seqs_y: Vector of (17, seq_len) — target x[t+1:t+seq_len]
"""
function cut_in_sequences(arr::Matrix{Float32}, seq_len::Int; inc::Int=10)
    n_steps = size(arr, 1)
    seqs_x = Matrix{Float32}[]
    seqs_y = Matrix{Float32}[]

    for s in 0:inc:(n_steps - seq_len - 1)
        # Julia 1-indexed: start=s+1, input end=s+seq_len, target end=s+seq_len+1
        start_x = s + 1
        end_x = s + seq_len
        start_y = s + 2
        end_y = s + seq_len + 1

        push!(seqs_x, arr[start_x:end_x, :]')   # (17, seq_len)
        push!(seqs_y, arr[start_y:end_y, :]')    # (17, seq_len)
    end

    return seqs_x, seqs_y
end

struct CheetahData
    train_x::Array{Float32, 3}    # (OBS_SIZE, seq_len, N_train)
    train_y::Array{Float32, 3}    # (OBS_SIZE, seq_len, N_train)
    valid_x::Array{Float32, 3}
    valid_y::Array{Float32, 3}
    test_x::Array{Float32, 3}
    test_y::Array{Float32, 3}
end

"""
    load_cheetah_data(; data_dir=...)

Load HalfCheetah rollout data from .npy files.
File split matches cheetah.py:
  - valid:  sorted files[0:4]   (5 files)
  - test:   sorted files[5:14]  (10 files)
  - train:  sorted files[15:24] (10 files)
"""
function load_cheetah_data(; data_dir=joinpath(@__DIR__, "..", "data", "cheetah"))
    println("Loading Cheetah data from: $data_dir")

    # Get sorted .npy files (matching Python: sorted listdir + endswith .npy)
    all_files = sort([joinpath(data_dir, f) for f in readdir(data_dir) if endswith(f, ".npy")])
    println("  Found $(length(all_files)) .npy files")

    # File split (0-indexed in Python, 1-indexed here)
    # Python: valid = files[:5], test = files[5:15], train = files[15:25]
    valid_files = all_files[1:5]
    test_files  = all_files[6:15]
    train_files = all_files[16:25]

    function load_split(files)
        all_x = Matrix{Float32}[]
        all_y = Matrix{Float32}[]
        for f in files
            arr = Float32.(npzread(f))   # (T_steps, 17)
            sx, sy = cut_in_sequences(arr, SEQ_LEN; inc=INC)
            append!(all_x, sx)
            append!(all_y, sy)
        end
        # Stack: each element is (17, seq_len) → cat along dim 3 → (17, seq_len, N)
        x = cat(all_x...; dims=3)
        y = cat(all_y...; dims=3)
        return x, y
    end

    train_x, train_y = load_split(train_files)
    valid_x, valid_y = load_split(valid_files)
    test_x, test_y   = load_split(test_files)

    println("  train_x: $(size(train_x)), train_y: $(size(train_y))")
    println("  valid_x: $(size(valid_x)), valid_y: $(size(valid_y))")
    println("  test_x: $(size(test_x)),  test_y: $(size(test_y))")

    return CheetahData(train_x, train_y, valid_x, valid_y, test_x, test_y)
end

# ═══════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════

function build_model(args, rng)
    cell, ps_cell, st_cell = build_cell(args.model, args.model_size, OBS_SIZE, args, rng)
    head = Lux.Dense(hidden_size(cell) => OBS_SIZE;
        init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32)
    ps_head, st_head = Lux.setup(rng, head)

    return cell, head, ps_cell, st_cell, ps_head, st_head
end

# ── Batched forward pass (per-timestep, 17-dim output) ──────────────────
# x_batch: (OBS_SIZE, seq_len, B)
# Returns: preds (OBS_SIZE, seq_len, B) — predicted observations at each timestep
#
# Uses Zygote.Buffer to accumulate per-timestep outputs without triggering
# Zygote's mutation restriction on regular arrays.
function forward_batch(cell, head, ps_cell, ps_head, st_cell, st_head, x_batch)
    B = size(x_batch, 3)
    T = size(x_batch, 2)
    S = initial_state(cell, B)

    # Zygote.Buffer allows setindex! inside differentiated code
    buf = Zygote.Buffer(x_batch, OBS_SIZE, T, B)

    for t in 1:T
        u_t = @view x_batch[:, t, :]        # (OBS_SIZE, B)
        st_d = merge(st_cell, (input = u_t,))
        S, _ = cell(S, ps_cell, st_d)

        # Readout + Dense at each timestep
        obs = readout(cell, S, ps_cell)       # (model_size, B)
        out, _ = head(obs, ps_head, st_head)  # (OBS_SIZE, B)
        buf[:, t, :] = out                    # (OBS_SIZE,) per sample
    end

    return copy(buf)  # copy() converts Buffer → regular Array for downstream ops
end

# ── Batched MSE loss (per-timestep, per-dimension) ──────────────────────
# preds: (OBS_SIZE, seq_len, B), targets: (OBS_SIZE, seq_len, B)
function batch_mse_loss(cell, head, ps_cell, ps_head, st_cell, st_head, x_batch, y_batch)
    preds = forward_batch(cell, head, ps_cell, ps_head, st_cell, st_head, x_batch)
    # MSE over all dimensions, timesteps, and batch samples
    return mean((preds .- y_batch) .^ 2)
end

# ═══════════════════════════════════════════════════════════════════════
# EVALUATION (batched)
# ═══════════════════════════════════════════════════════════════════════

function evaluate(cell, head, ps_cell, ps_head, st_cell, st_head,
                  data_x::Array{Float32, 3}, data_y::Array{Float32, 3};
                  eval_batch_size::Int=128)
    n = size(data_x, 3)
    total_se = 0.0f0    # sum of squared errors
    total_ae = 0.0f0    # sum of absolute errors
    total_count = 0

    n_batches = cld(n, eval_batch_size)
    for b in 1:n_batches
        b_start = (b - 1) * eval_batch_size + 1
        b_end = min(b * eval_batch_size, n)
        batch_x = @view data_x[:, :, b_start:b_end]
        batch_y = @view data_y[:, :, b_start:b_end]
        B = b_end - b_start + 1

        preds = forward_batch(cell, head, ps_cell, ps_head, st_cell, st_head, batch_x)
        diff = preds .- batch_y
        total_se += sum(diff .^ 2)
        total_ae += sum(abs.(diff))
        total_count += OBS_SIZE * SEQ_LEN * B   # count all elements
    end

    mse = total_se / total_count
    mae = total_ae / total_count
    return mse, mae
end

# ═══════════════════════════════════════════════════════════════════════
# CHECKPOINTING
# ═══════════════════════════════════════════════════════════════════════

function save_checkpoint(path, params, opt_state, epoch, best_valid_mse, args)
    mkpath(dirname(path))
    jldsave(path;
        params = params,
        opt_state = opt_state,
        epoch = epoch,
        best_valid_mse = best_valid_mse,
        args = args,
    )
    println("  💾 Checkpoint saved: $path (epoch $epoch, valid MSE $(round(best_valid_mse; digits=6)))")
end

function load_checkpoint(path)
    data = jldopen(path, "r") do f
        (
            params = f["params"],
            opt_state = f["opt_state"],
            epoch = f["epoch"],
            best_valid_mse = f["best_valid_mse"],
            args = f["args"],
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

function train!(cell, head, ps_cell, ps_head, st_cell, st_head, data::CheetahData;
                epochs::Int=200, lr::Float32=0.01f0, batch_size::Int=16,
                start_epoch::Int=0, initial_opt_state=nothing,
                initial_best_valid_mse::Float32=Inf32,
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

    best_valid_mse = initial_best_valid_mse
    best_params = deepcopy(params)
    best_epoch = start_epoch
    best_stats = nothing

    n_train = size(data.train_x, 3)

    for epoch in start_epoch:(epochs - 1)
        # ── LR schedule (warmup → hold → taper)
        current_lr = lr_schedule(epoch, epochs)
        Optimisers.adjust!(opt_state, current_lr)

        # ── Evaluate ────────────────────────────────────────────────
        valid_mse, valid_mae = evaluate(cell, head, params.cell, params.head,
                                         st_cell, st_head, data.valid_x, data.valid_y)
        test_mse, test_mae = evaluate(cell, head, params.cell, params.head,
                                       st_cell, st_head, data.test_x, data.test_y)

        # ── Model selection (by valid MSE — lower is better) ─────────
        if valid_mse < best_valid_mse && epoch > start_epoch
            best_valid_mse = valid_mse
            best_params = deepcopy(params)
            best_epoch = epoch
            best_stats = (0.0f0, 0.0f0, valid_mse, valid_mae, test_mse, test_mae)
            best_path = joinpath(save_dir, "$(args.model)_cheetah_best.jld2")
            save_checkpoint(best_path, best_params, opt_state, epoch,
                            best_valid_mse, args)
        end

        # ── Train one epoch ─────────────────────────────────────────
        perm = randperm(n_train)
        n_batches = div(n_train, batch_size)
        epoch_losses = Float32[]
        epoch_ae = 0.0f0
        epoch_count = 0

        for b in 1:n_batches
            b_start = (b - 1) * batch_size + 1
            b_end = b * batch_size
            batch_idx = perm[b_start:b_end]

            x_batch = data.train_x[:, :, batch_idx]     # (OBS_SIZE, seq_len, B)
            y_batch = data.train_y[:, :, batch_idx]      # (OBS_SIZE, seq_len, B)

            loss_val, grads = Zygote.withgradient(params) do p
                batch_mse_loss(cell, head, p.cell, p.head,
                               st_cell, st_head, x_batch, y_batch)
            end

            opt_state, params = Optimisers.update(opt_state, params, grads[1])
            push!(epoch_losses, loss_val)

            # Track MAE from the same batch
            preds = forward_batch(cell, head, params.cell, params.head,
                                    st_cell, st_head, x_batch)
            epoch_ae += sum(abs.(preds .- y_batch))
            epoch_count += OBS_SIZE * SEQ_LEN * batch_size

            # Batch progress (first batch + every 50)
            if b == 1 || b % 50 == 0
                @printf("  [batch %d/%d] loss: %.4f\n", b, n_batches, loss_val)
                flush(stdout)
            end
        end

        train_loss = mean(epoch_losses)
        train_mae = epoch_ae / max(epoch_count, 1)

        # ── Periodic checkpoint ──────────────────────────────────────
        if save_every > 0 && epoch > start_epoch && epoch % save_every == 0
            periodic_path = joinpath(save_dir, "$(args.model)_cheetah_epoch_$(lpad(epoch, 3, '0')).jld2")
            save_checkpoint(periodic_path, params, opt_state, epoch,
                            best_valid_mse, args)
        end

        # ── Log (matches Python format) ──────────────────────────────
        @printf("Epochs %03d, train loss: %0.4f, train mae: %0.4f, valid loss: %0.4f, valid mae: %0.4f, test loss: %0.4f, test mae: %0.4f\n",
            epoch, train_loss, train_mae,
            valid_mse, valid_mae,
            test_mse, test_mae)

        if !isfinite(train_loss)
            println("NaN detected, stopping training.")
            break
        end
    end

    if best_stats !== nothing
        tl, ta, vl, va, tel, tea = best_stats
        @printf("Best epoch %03d, train loss: %0.6f, train mae: %0.6f, valid loss: %0.6f, valid mae: %0.6f, test loss: %0.6f, test mae: %0.6f\n",
            best_epoch, tl, ta, vl, va, tel, tea)
    end

    return best_params
end

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

function main()
    args = parse_args()
    println("Cheetah Training — $(uppercase(args.model)) ($(args.solver), batched BPTT)")
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

    data = load_cheetah_data()

    cell, head, ps_cell, st_cell, ps_head, st_head = build_model(args, rng)

    start_epoch = 0
    initial_opt_state = nothing
    initial_best_valid_mse = Inf32

    if !isempty(args.resume_path)
        println("\nLoading checkpoint: $(args.resume_path)")
        ckpt = load_checkpoint(args.resume_path)
        ps_cell = ckpt.params.cell
        ps_head = ckpt.params.head
        initial_opt_state = ckpt.opt_state
        start_epoch = ckpt.epoch + 1
        initial_best_valid_mse = Float32(ckpt.best_valid_mse)
        println("  Loaded epoch $(ckpt.epoch), best valid MSE: $(round(ckpt.best_valid_mse; digits=6))")
        println("  Resuming from epoch $start_epoch with LR $(args.lr)")
    end

    n_cell_params = sum(length(getproperty(ps_cell, k)) for k in propertynames(ps_cell))
    n_head_params = args.model_size * OBS_SIZE + OBS_SIZE
    println("  Cell params: $n_cell_params")
    println("  Head params: $n_head_params")
    println("  Total params: $(n_cell_params + n_head_params)")
    if hasproperty(cell, :state_dim)
        println("  State dim: $(cell.state_dim)")
    end

    if isempty(args.resume_path)
        println("\nGradient smoke test (batched)...")
        test_x = data.train_x[:, :, 1:2]
        test_y = data.train_y[:, :, 1:2]

        test_loss, test_grads = Zygote.withgradient((cell=ps_cell, head=ps_head)) do p
            batch_mse_loss(cell, head, p.cell, p.head, st_cell, st_head, test_x, test_y)
        end
        println("  Initial MSE: $(@sprintf("%.4f", test_loss))")

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
                         initial_best_valid_mse=initial_best_valid_mse,
                         save_dir=args.save_dir, save_every=args.save_every,
                         warmup_epochs=args.warmup_epochs, args=args)
end

main()
