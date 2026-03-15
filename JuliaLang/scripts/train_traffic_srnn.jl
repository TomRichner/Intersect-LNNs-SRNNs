# train_traffic_srnn.jl — Traffic Volume Prediction with SRNNCell (batched BPTT via Zygote)
#
# Adapted from: traffic.py (Hasani et al.)
# Predicts normalized traffic volume from 7 engineered features.
# First *regression* task: MSE loss, MAE metric, Dense(1) head at every timestep.
#
# Usage:
#   julia --project=JuliaLang JuliaLang/scripts/train_traffic_srnn.jl [--epochs 200] [--size 32] [--lr 0.01] [--bs 16]
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
using CSV, DataFrames, Dates
using Lux, NNlib, Zygote, Optimisers
using JLD2

# ── Include SRNNCell ────────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "src", "models", "srnn.jl"))

# ── Configuration ───────────────────────────────────────────────────────
const SEQ_LEN    = 32
const N_FEATURES = 7      # holiday, temp, rain, snow, clouds, weekday, noon
const N_OUT      = 1      # regression: predict traffic volume

# Parse simple command-line args
function parse_args()
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
    load_trace(filepath) → (features, traffic_volume)

Load and engineer features from the Metro Interstate Traffic Volume CSV.
Returns:
  features:       (N_samples, 7) Float32 matrix
  traffic_volume:  (N_samples,) Float32 vector, z-score normalized
"""
function load_trace(filepath::String)
    df = CSV.read(filepath, DataFrame)

    # Holiday: Python checks `holiday == None` which is True for string "None"
    holiday = Float32[row == "None" ? 1.0f0 : 0.0f0 for row in df.holiday]

    # Temperature: subtract mean
    temp = Float32.(df.temp)
    temp .-= mean(temp)

    # Precipitation / weather
    rain  = Float32.(df.rain_1h)
    snow  = Float32.(df.snow_1h)
    clouds = Float32.(df.clouds_all)

    # Date/time features
    date_times = DateTime.(df.date_time, dateformat"y-m-d H:M:S")
    weekday = Float32[Float32(dayofweek(d) - 1) for d in date_times]  # Mon=0..Sun=6
    noon = Float32[sin(Float32(hour(d)) * Float32(π) / 24.0f0) for d in date_times]

    # Stack features: (N, 7)
    features = hcat(holiday, temp, rain, snow, clouds, weekday, noon)

    # Target: z-score normalized
    traffic_volume = Float32.(df.traffic_volume)
    traffic_volume .-= mean(traffic_volume)
    traffic_volume ./= std(traffic_volume)

    return features, traffic_volume
end

"""
    cut_in_sequences(x, y, seq_len; inc=1)

Sliding-window segmentation.
  x: (N_samples, N_features)  → seqs_x: (N_features, seq_len, N_seqs)
  y: (N_samples,)             → seqs_y: (seq_len, N_seqs)
"""
function cut_in_sequences(x::Matrix{Float32}, y::Vector{Float32}, seq_len::Int; inc::Int=1)
    n_samples = size(x, 1)
    starts = 0:inc:(n_samples - seq_len - 1)
    n_seqs = length(starts)

    seqs_x = Array{Float32, 3}(undef, size(x, 2), seq_len, n_seqs)
    seqs_y = Matrix{Float32}(undef, seq_len, n_seqs)

    for (idx, s) in enumerate(starts)
        start = s + 1  # Julia 1-indexed
        stop = start + seq_len - 1
        seqs_x[:, :, idx] .= x[start:stop, :]'   # transpose: (features, seq_len)
        seqs_y[:, idx] .= y[start:stop]
    end
    return seqs_x, seqs_y
end

struct TrafficData
    train_x::Array{Float32, 3}    # (N_FEATURES, seq_len, N_train)
    train_y::Matrix{Float32}      # (seq_len, N_train)
    valid_x::Array{Float32, 3}
    valid_y::Matrix{Float32}
    test_x::Array{Float32, 3}
    test_y::Matrix{Float32}
end

function load_traffic_data(; data_dir=joinpath(@__DIR__, "..", "data", "traffic"))
    println("Loading Traffic data from: $data_dir")

    filepath = joinpath(data_dir, "Metro_Interstate_Traffic_Volume.csv")
    features, traffic_volume = load_trace(filepath)
    println("  Raw samples: $(size(features, 1)), features: $(size(features, 2))")

    # Sliding window with inc=4 (matching Python)
    seqs_x, seqs_y = cut_in_sequences(features, traffic_volume, SEQ_LEN; inc=4)
    total_seqs = size(seqs_x, 3)
    println("  Total sequences (inc=4): $total_seqs")

    # 75/10/15 split with fixed seed (matching Python: np.random.RandomState(23489))
    perm = randperm(MersenneTwister(23489), total_seqs)
    valid_size = Int(floor(0.1 * total_seqs))
    test_size  = Int(floor(0.15 * total_seqs))
    train_size = total_seqs - valid_size - test_size

    valid_idx = perm[1:valid_size]
    test_idx  = perm[valid_size+1:valid_size+test_size]
    train_idx = perm[valid_size+test_size+1:end]

    println("  Train: $train_size, Valid: $valid_size, Test: $test_size")

    return TrafficData(
        seqs_x[:, :, train_idx], seqs_y[:, train_idx],
        seqs_x[:, :, valid_idx], seqs_y[:, valid_idx],
        seqs_x[:, :, test_idx],  seqs_y[:, test_idx],
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
    # Regression head: Dense(n → 1), no activation
    # Python uses TruncatedNormal init — we use Glorot uniform (standard for Lux)
    head = Lux.Dense(args.model_size => N_OUT;
        init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32)

    ps_cell, st_cell = Lux.setup(rng, cell)
    ps_head, st_head = Lux.setup(rng, head)

    return cell, head, ps_cell, st_cell, ps_head, st_head
end

# ── Batched forward pass (per-timestep) ─────────────────────────────────
# x_batch: (N_FEATURES, seq_len, B)
# Returns: preds (seq_len, B) — predicted traffic volume at each timestep
#
# Uses Zygote.Buffer to accumulate per-timestep outputs without triggering
# Zygote's mutation restriction on regular arrays.
function forward_batch(cell, head, ps_cell, ps_head, st_cell, st_head, x_batch)
    B = size(x_batch, 3)
    T = size(x_batch, 2)
    S = srnn_initial_state(cell, B)

    # Zygote.Buffer allows setindex! inside differentiated code
    buf = Zygote.Buffer(x_batch, T, B)

    for t in 1:T
        u_t = @view x_batch[:, t, :]
        st_d = merge(st_cell, (input = u_t,))
        S, _ = cell(S, ps_cell, st_d)

        # Readout + Dense at each timestep
        obs = readout(cell, S, ps_cell)      # (model_size, B)
        out, _ = head(obs, ps_head, st_head)  # (1, B)
        buf[t, :] = out[1, :]                 # scalar output per sample
    end

    return copy(buf)  # copy() converts Buffer → regular Array for downstream ops
end

# ── Batched MSE loss (per-timestep) ─────────────────────────────────────
# preds: (seq_len, B), targets: (seq_len, B)
function batch_mse_loss(cell, head, ps_cell, ps_head, st_cell, st_head, x_batch, y_batch)
    preds = forward_batch(cell, head, ps_cell, ps_head, st_cell, st_head, x_batch)
    # MSE over all timesteps and batch samples
    return mean((preds .- y_batch) .^ 2)
end

# ═══════════════════════════════════════════════════════════════════════
# EVALUATION (batched)
# ═══════════════════════════════════════════════════════════════════════

function evaluate(cell, head, ps_cell, ps_head, st_cell, st_head,
                  data_x::Array{Float32, 3}, data_y::Matrix{Float32};
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
        batch_y = @view data_y[:, b_start:b_end]
        B = b_end - b_start + 1

        preds = forward_batch(cell, head, ps_cell, ps_head, st_cell, st_head, batch_x)
        diff = preds .- batch_y
        total_se += sum(diff .^ 2)
        total_ae += sum(abs.(diff))
        total_count += SEQ_LEN * B
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

function train!(cell, head, ps_cell, ps_head, st_cell, st_head, data::TrafficData;
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
        # ── LR warmup ───────────────────────────────────────────────
        if warmup_epochs > 0 && epoch < warmup_epochs
            warmup_frac = (epoch + 1) / warmup_epochs
            current_lr = lr * (0.1f0 + 0.9f0 * Float32(warmup_frac))
            Optimisers.adjust!(opt_state, current_lr)
        elseif warmup_epochs > 0 && epoch == warmup_epochs
            Optimisers.adjust!(opt_state, lr)
        end

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
            best_path = joinpath(save_dir, "srnn_traffic_best.jld2")
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

            x_batch = data.train_x[:, :, batch_idx]     # (features, seq_len, B)
            y_batch = data.train_y[:, batch_idx]         # (seq_len, B)

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
            epoch_count += SEQ_LEN * batch_size
        end

        train_loss = mean(epoch_losses)
        train_mae = epoch_ae / max(epoch_count, 1)

        # ── Periodic checkpoint ──────────────────────────────────────
        if save_every > 0 && epoch > start_epoch && epoch % save_every == 0
            periodic_path = joinpath(save_dir, @sprintf("srnn_traffic_epoch_%03d.jld2", epoch))
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
    println("Traffic Training — SRNNCell ($(args.solver), batched BPTT)")
    println("  Model size: $(args.model_size), n_E: $(args.n_E), n_I: $(args.model_size - args.n_E)")
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

    data = load_traffic_data()

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

    n_cell_params = sum(length(getproperty(ps_cell, k)) for k in keys(ps_cell))
    n_head_params = args.model_size * N_OUT + N_OUT
    println("  Cell params: $n_cell_params")
    println("  Head params: $n_head_params")
    println("  Total params: $(n_cell_params + n_head_params)")
    println("  State dim: $(cell.state_dim)")

    if isempty(args.resume_path)
        println("\nGradient smoke test (batched)...")
        test_x = data.train_x[:, :, 1:2]
        test_y = data.train_y[:, 1:2]

        test_loss, test_grads = Zygote.withgradient((cell=ps_cell, head=ps_head)) do p
            batch_mse_loss(cell, head, p.cell, p.head, st_cell, st_head, test_x, test_y)
        end
        println("  Initial MSE: $(@sprintf("%.4f", test_loss)) (expected ~1.0 for z-score targets)")

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
                         initial_best_valid_mse=initial_best_valid_mse,
                         save_dir=args.save_dir, save_every=args.save_every,
                         warmup_epochs=args.warmup_epochs, args=args)
end

main()
