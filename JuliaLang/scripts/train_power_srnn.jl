# train_power_srnn.jl — Household Power Consumption with SRNNCell (batched BPTT via Zygote)
#
# Adapted from: power.py (Hasani et al.)
# Regression: predict Global_active_power from 6 electrical measurements.
# MSE loss, MAE metric, Dense(1) head at every timestep.
#
# Usage:
#   julia --project=JuliaLang JuliaLang/scripts/train_power_srnn.jl [--epochs 200] [--size 32] [--lr 0.01] [--bs 256]
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
const N_FEATURES = 6      # 6 electrical measurement columns (after target split)
const N_OUT      = 1      # regression: predict Global_active_power

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
    load_power_trace(filepath) → (x::Matrix{Float32}, y::Vector{Float32})

Load the UCI Household Power Consumption dataset.
Format: semicolon-delimited, header row, 9 columns:
  Date;Time;Global_active_power;Global_reactive_power;Voltage;
  Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3

- Skips Date (col 1) and Time (col 2)
- 7 float columns (cols 3-9), '?' values forward-filled
- Per-feature z-score normalization
- Column 0 (Global_active_power) → target y
- Columns 1-6 → features x
"""
function load_power_trace(filepath::String)
    all_x = Vector{Vector{Float32}}()
    memory = zeros(Float32, 7)  # forward-fill memory for 7 measurement cols

    open(filepath) do f
        lineno = 0
        for line in eachline(f)
            lineno += 1
            lineno == 1 && continue  # skip header

            parts = split(strip(line), ';')
            length(parts) >= 8 || continue

            # Parse 7 measurement columns (indices 3-9 in Python, parts[3:end] here)
            feature_col = parts[3:end]  # 7 values
            length(feature_col) >= 7 || continue

            vals = Vector{Float32}(undef, 7)
            for i in 1:7
                s = strip(feature_col[i])
                if s == "?" || s == ""
                    vals[i] = memory[i]  # forward-fill
                else
                    vals[i] = parse(Float32, s)
                    memory[i] = vals[i]
                end
            end

            push!(all_x, vals)
        end
    end

    println("  Read $(length(all_x)) samples")

    # Stack into matrix (N × 7)
    x = reduce(hcat, all_x)'  # N × 7

    # Per-feature z-score normalization
    col_means = mean(x, dims=1)
    col_stds = std(x, dims=1)
    x = (x .- col_means) ./ col_stds

    # Split: col 1 → target y, cols 2-7 → features x
    y = Float32.(x[:, 1])         # Global_active_power (normalized)
    x_feat = Float32.(x[:, 2:7])  # remaining 6 features

    return x_feat, y
end

"""
    cut_in_sequences_power(x, y, seq_len; inc=seq_len)

Non-overlapping window segmentation (inc=seq_len by default).
  x: (N_samples, N_features)  → seqs_x: (N_features, seq_len, N_seqs)
  y: (N_samples,)             → seqs_y: (seq_len, N_seqs)
"""
function cut_in_sequences_power(x::Matrix{Float32}, y::Vector{Float32}, seq_len::Int; inc::Int=seq_len)
    n_samples = size(x, 1)
    starts = 0:inc:(n_samples - seq_len - 1)
    n_seqs = length(starts)

    seqs_x = Array{Float32, 3}(undef, size(x, 2), seq_len, n_seqs)
    seqs_y = Matrix{Float32}(undef, seq_len, n_seqs)

    for (idx, s) in enumerate(starts)
        start = s + 1  # Julia 1-indexed
        stop = start + seq_len - 1
        seqs_x[:, :, idx] .= x[start:stop, :]'
        seqs_y[:, idx] .= y[start:stop]
    end
    return seqs_x, seqs_y
end

struct PowerData
    train_x::Array{Float32, 3}    # (N_FEATURES, seq_len, N_train)
    train_y::Matrix{Float32}      # (seq_len, N_train)
    valid_x::Array{Float32, 3}
    valid_y::Matrix{Float32}
    test_x::Array{Float32, 3}
    test_y::Matrix{Float32}
end

function load_power_data(; data_dir=joinpath(@__DIR__, "..", "data", "power"))
    println("Loading Power data from: $data_dir")

    filepath = joinpath(data_dir, "household_power_consumption.txt")
    x, y = load_power_trace(filepath)
    println("  Features: $(size(x, 2)), Target: 1 (Global_active_power)")

    # Non-overlapping windows (inc=seq_len, matching Python)
    seqs_x, seqs_y = cut_in_sequences_power(x, y, SEQ_LEN; inc=SEQ_LEN)
    total_seqs = size(seqs_x, 3)
    println("  Total sequences (non-overlapping): $total_seqs")

    # 75/10/15 split with fixed seed (matching Python)
    perm = randperm(MersenneTwister(23489), total_seqs)
    valid_size = Int(floor(0.1 * total_seqs))
    test_size  = Int(floor(0.15 * total_seqs))

    valid_idx = perm[1:valid_size]
    test_idx  = perm[valid_size+1:valid_size+test_size]
    train_idx = perm[valid_size+test_size+1:end]

    println("  Train: $(length(train_idx)), Valid: $(length(valid_idx)), Test: $(length(test_idx))")

    return PowerData(
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
    head = Lux.Dense(args.model_size => N_OUT;
        init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32)

    ps_cell, st_cell = Lux.setup(rng, cell)
    ps_head, st_head = Lux.setup(rng, head)

    return cell, head, ps_cell, st_cell, ps_head, st_head
end

# ── Batched forward pass (per-timestep regression) ─────────────────────
function forward_batch(cell, head, ps_cell, ps_head, st_cell, st_head, x_batch)
    B = size(x_batch, 3)
    T = size(x_batch, 2)
    S = srnn_initial_state(cell, B)

    buf = Zygote.Buffer(x_batch, T, B)

    for t in 1:T
        u_t = @view x_batch[:, t, :]
        st_d = merge(st_cell, (input = u_t,))
        S, _ = cell(S, ps_cell, st_d)

        obs = readout(cell, S, ps_cell)
        out, _ = head(obs, ps_head, st_head)  # (1, B)
        buf[t, :] = out[1, :]
    end

    return copy(buf)
end

# ── Batched MSE loss (per-timestep) ────────────────────────────────────
function batch_mse_loss(cell, head, ps_cell, ps_head, st_cell, st_head, x_batch, y_batch)
    preds = forward_batch(cell, head, ps_cell, ps_head, st_cell, st_head, x_batch)
    return mean((preds .- y_batch) .^ 2)
end

# ═══════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════

function evaluate(cell, head, ps_cell, ps_head, st_cell, st_head,
                  data_x::Array{Float32, 3}, data_y::Matrix{Float32};
                  eval_batch_size::Int=128)
    n = size(data_x, 3)
    total_se = 0.0f0
    total_ae = 0.0f0
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

# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

function train!(cell, head, ps_cell, ps_head, st_cell, st_head, data::PowerData;
                epochs::Int=200, lr::Float32=0.01f0, batch_size::Int=256,
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
        # ── LR warmup
        if warmup_epochs > 0 && epoch < warmup_epochs
            warmup_frac = (epoch + 1) / warmup_epochs
            current_lr = lr * (0.1f0 + 0.9f0 * Float32(warmup_frac))
            Optimisers.adjust!(opt_state, current_lr)
        elseif warmup_epochs > 0 && epoch == warmup_epochs
            Optimisers.adjust!(opt_state, lr)
        end

        # ── Evaluate
        valid_mse, valid_mae = evaluate(cell, head, params.cell, params.head,
                                         st_cell, st_head, data.valid_x, data.valid_y)
        test_mse, test_mae = evaluate(cell, head, params.cell, params.head,
                                       st_cell, st_head, data.test_x, data.test_y)

        # ── Model selection (by valid MSE — lower is better)
        if valid_mse < best_valid_mse && epoch > start_epoch
            best_valid_mse = valid_mse
            best_params = deepcopy(params)
            best_epoch = epoch
            best_stats = (0.0f0, 0.0f0, valid_mse, valid_mae, test_mse, test_mae)
            best_path = joinpath(save_dir, "srnn_power_best.jld2")
            save_checkpoint(best_path, best_params, opt_state, epoch,
                            best_valid_mse, args)
        end

        # ── Train one epoch
        perm = randperm(n_train)
        n_batches = div(n_train, batch_size)
        epoch_losses = Float32[]

        for b in 1:n_batches
            b_start = (b - 1) * batch_size + 1
            b_end = b * batch_size
            batch_idx = perm[b_start:b_end]

            x_batch = data.train_x[:, :, batch_idx]
            y_batch = data.train_y[:, batch_idx]

            loss_val, grads = Zygote.withgradient(params) do p
                batch_mse_loss(cell, head, p.cell, p.head,
                               st_cell, st_head, x_batch, y_batch)
            end

            opt_state, params = Optimisers.update(opt_state, params, grads[1])
            push!(epoch_losses, loss_val)

            if b == 1 || b % 50 == 0
                @printf("  [batch %d/%d] loss: %.4f\n", b, n_batches, loss_val)
                flush(stdout)
            end
        end

        train_loss = mean(epoch_losses)

        # ── Periodic checkpoint
        if save_every > 0 && epoch > start_epoch && epoch % save_every == 0
            periodic_path = joinpath(save_dir, @sprintf("srnn_power_epoch_%03d.jld2", epoch))
            save_checkpoint(periodic_path, params, opt_state, epoch,
                            best_valid_mse, args)
        end

        # ── Log
        @printf("Epochs %03d, train loss: %0.4f, valid loss: %0.4f, valid mae: %0.4f, test loss: %0.4f, test mae: %0.4f\n",
            epoch, train_loss,
            valid_mse, valid_mae,
            test_mse, test_mae)

        if !isfinite(train_loss)
            println("NaN detected, stopping training.")
            break
        end
    end

    if best_stats !== nothing
        tl, ta, vl, va, tel, tea = best_stats
        @printf("Best epoch %03d, valid loss: %0.6f, valid mae: %0.6f, test loss: %0.6f, test mae: %0.6f\n",
            best_epoch, vl, va, tel, tea)
    end

    return best_params
end

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

function main()
    args = parse_args()
    println("Power Training — SRNNCell ($(args.solver), batched BPTT)")
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

    data = load_power_data()

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
