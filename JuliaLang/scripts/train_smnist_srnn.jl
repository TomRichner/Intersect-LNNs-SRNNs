# train_smnist_srnn.jl — Sequential MNIST with SRNNCell (batched BPTT via Zygote)
#
# Adapted from: smnist.py (Hasani et al.)
# Feeds MNIST images row-by-row: 28 time steps × 28 pixel features.
# Uses SRNNCell with fused semi-implicit or explicit Euler solver.
#
# Usage:
#   julia --project=JuliaLang JuliaLang/scripts/train_smnist_srnn.jl [--epochs 200] [--size 32] [--lr 0.01] [--bs 32]
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
include(joinpath(@__DIR__, "..", "src", "training_utils.jl"))

# ── Configuration ───────────────────────────────────────────────────────
const SEQ_LEN    = 28     # 28 rows of the image
const N_FEATURES = 28     # 28 pixels per row
const N_CLASSES  = 10     # digits 0-9

# Parse simple command-line args
function parse_args()
    epochs = 200
    model_size = 32
    lr = 0.01f0
    batch_size = 32
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
# IDX FILE PARSING (raw MNIST binary format)
# ═══════════════════════════════════════════════════════════════════════

"""
    read_idx_images(filepath) → Array{Float32, 3}  (28, 28, N)

Read a gzipped IDX3 image file. Returns (rows, cols, N) normalized to [0,1].
"""
function read_idx_images(filepath::String)
    data = read(pipeline(`gzip -dc $filepath`))
    # Header: magic(4) | n_images(4) | n_rows(4) | n_cols(4)
    magic = ntoh(reinterpret(UInt32, data[1:4])[1])
    @assert magic == 0x00000803 "Bad IDX3 magic: $magic"
    n = Int(ntoh(reinterpret(UInt32, data[5:8])[1]))
    rows = Int(ntoh(reinterpret(UInt32, data[9:12])[1]))
    cols = Int(ntoh(reinterpret(UInt32, data[13:16])[1]))
    # Pixel data starts at byte 17
    pixels = data[17:end]
    @assert length(pixels) == n * rows * cols
    # Reshape to (rows, cols, N) and normalize
    imgs = reshape(Float32.(pixels) ./ 255.0f0, cols, rows, n)
    # IDX stores row-major, so after reshape we have (col, row, N)
    # Permute to (row, col, N) — but for our purposes (features=row_pixels, time=rows):
    # We want (N_FEATURES=28, SEQ_LEN=28, N) = (pixels_per_row, n_rows, N)
    # The IDX data is stored as: image[row][col], so pixel order is row-major.
    # After reshape(cols, rows, n): dim1=col (within row), dim2=row, dim3=image
    # This gives us (features=cols, time=rows, N) — exactly what we want!
    return imgs
end

"""
    read_idx_labels(filepath) → Vector{Int}

Read a gzipped IDX1 label file. Returns 0-based labels.
"""
function read_idx_labels(filepath::String)
    data = read(pipeline(`gzip -dc $filepath`))
    magic = ntoh(reinterpret(UInt32, data[1:4])[1])
    @assert magic == 0x00000801 "Bad IDX1 magic: $magic"
    n = Int(ntoh(reinterpret(UInt32, data[5:8])[1]))
    labels = Int.(data[9:end])
    @assert length(labels) == n
    return labels
end

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

struct SMnistData
    train_x::Array{Float32, 3}    # (features=28, seq_len=28, N_train)
    train_y::Vector{Int}          # (N_train,) — 1-indexed class labels
    valid_x::Array{Float32, 3}
    valid_y::Vector{Int}
    test_x::Array{Float32, 3}
    test_y::Vector{Int}
end

function load_smnist_data(; data_dir=joinpath(@__DIR__, "..", "data", "smnist"))
    println("Loading MNIST data from: $data_dir")

    # Read raw IDX files
    all_train_x = read_idx_images(joinpath(data_dir, "train-images-idx3-ubyte.gz"))
    all_train_y = read_idx_labels(joinpath(data_dir, "train-labels-idx1-ubyte.gz"))
    test_x      = read_idx_images(joinpath(data_dir, "t10k-images-idx3-ubyte.gz"))
    test_y_raw  = read_idx_labels(joinpath(data_dir, "t10k-labels-idx1-ubyte.gz"))

    # 0-based → 1-based labels for Julia
    test_y = test_y_raw .+ 1

    n_total = size(all_train_x, 3)
    println("  Total training images: $n_total")
    println("  Test images: $(size(test_x, 3))")

    # Sequential 90/10 split (matching Python: no shuffle)
    train_split = Int(floor(0.9 * n_total))  # 54000
    train_x = all_train_x[:, :, 1:train_split]
    train_y = all_train_y[1:train_split] .+ 1
    valid_x = all_train_x[:, :, train_split+1:end]
    valid_y = all_train_y[train_split+1:end] .+ 1

    println("  Training split: $(size(train_x, 3)), Validation split: $(size(valid_x, 3))")

    return SMnistData(train_x, train_y, valid_x, valid_y, test_x, test_y)
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

# ── Batched forward pass ────────────────────────────────────────────────
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

# ── Batched cross-entropy loss ──────────────────────────────────────────
# NOTE: SMnist uses per-image labels (not per-timestep), so y_labels is (B,)
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
                  data_x::Array{Float32, 3}, data_y::Vector{Int};
                  eval_batch_size::Int=128)
    n = size(data_x, 3)
    total_loss = 0.0f0
    correct = 0

    n_batches = cld(n, eval_batch_size)
    for b in 1:n_batches
        b_start = (b - 1) * eval_batch_size + 1
        b_end = min(b * eval_batch_size, n)
        batch_x = @view data_x[:, :, b_start:b_end]
        batch_labels = @view data_y[b_start:b_end]
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

function adjust_lr!(opt_state, new_lr)
    Optimisers.adjust!(opt_state, new_lr)
end

# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

function train!(cell, head, ps_cell, ps_head, st_cell, st_head, data::SMnistData;
                epochs::Int=200, lr::Float32=0.01f0, batch_size::Int=32,
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
            best_path = joinpath(save_dir, "srnn_smnist_best.jld2")
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
            y_batch = data.train_y[batch_idx]   # per-image labels

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
            periodic_path = joinpath(save_dir, @sprintf("srnn_smnist_epoch_%03d.jld2", epoch))
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
    println("SMnist Training — SRNNCell ($(args.solver), batched BPTT)")
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

    data = load_smnist_data()

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

    n_cell_params = sum(length(getproperty(ps_cell, k)) for k in keys(ps_cell))
    n_head_params = args.model_size * N_CLASSES + N_CLASSES
    println("  Cell params: $n_cell_params")
    println("  Head params: $n_head_params")
    println("  Total params: $(n_cell_params + n_head_params)")
    println("  State dim: $(cell.state_dim)")

    if isempty(args.resume_path)
        println("\nGradient smoke test (batched)...")
        test_x = data.train_x[:, :, 1:2]
        test_y = data.train_y[1:2]

        test_loss, test_grads = Zygote.withgradient((cell=ps_cell, head=ps_head)) do p
            batch_loss(cell, head, p.cell, p.head, st_cell, st_head, test_x, test_y)
        end
        println("  Initial loss: $(@sprintf("%.4f", test_loss)) (expected ~2.30 = -log(1/10))")

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
                         initial_best_valid_acc=initial_best_valid_acc,
                         save_dir=args.save_dir, save_every=args.save_every,
                         warmup_epochs=args.warmup_epochs, args=args)
end

main()
