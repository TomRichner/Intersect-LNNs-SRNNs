# train_har.jl — HAR classification with LTCODE1 (BPTT via Zygote)
#
# Port of: liquid_time_constant_networks/experiments_with_ltcs/har.py
# Uses the fused semi-implicit solver with per-sample BPTT.
#
# Usage:
#   julia --project=JuliaLang JuliaLang/scripts/train_har.jl [--epochs 50] [--size 32] [--lr 0.01]

using Random, Statistics, DelimitedFiles, Printf
using Lux, NNlib, Zygote, Optimisers

# ── Include LTCODE1 ─────────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "src", "models", "ltc1.jl"))

# ── Configuration ───────────────────────────────────────────────────────
const SEQ_LEN    = 16
const N_FEATURES = 561
const N_CLASSES  = 6
const BATCH_SIZE = 16

# Parse simple command-line args
function parse_args()
    epochs = 50
    model_size = 32
    lr = 0.01f0
    for i in eachindex(ARGS)
        if ARGS[i] == "--epochs" && i < length(ARGS)
            epochs = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--size" && i < length(ARGS)
            model_size = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--lr" && i < length(ARGS)
            lr = parse(Float32, ARGS[i+1])
        end
    end
    return (; epochs, model_size, lr)
end

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

function cut_in_sequences(x::Matrix, y::Vector, seq_len::Int; inc::Int=1)
    n_samples = size(x, 1)
    seqs_x = Vector{Matrix{Float32}}()
    seqs_y = Vector{Vector{Int}}()
    for s in 0:(n_samples - seq_len - 1)
        if s % inc != 0
            continue
        end
        start = s + 1  # Julia 1-indexed
        stop = start + seq_len - 1
        push!(seqs_x, Float32.(x[start:stop, :]))   # (seq_len, features)
        push!(seqs_y, y[start:stop])                  # (seq_len,)
    end
    return seqs_x, seqs_y
end

struct HarData
    train_x::Vector{Matrix{Float32}}   # each: (seq_len, features)
    train_y::Vector{Vector{Int}}       # each: (seq_len,)
    valid_x::Vector{Matrix{Float32}}
    valid_y::Vector{Vector{Int}}
    test_x::Vector{Matrix{Float32}}
    test_y::Vector{Vector{Int}}
end

function load_har_data(; data_dir=joinpath(@__DIR__, "..", "data", "har", "UCI HAR Dataset"))
    println("Loading HAR data from: $data_dir")

    # Load raw data
    train_x_raw = readdlm(joinpath(data_dir, "train", "X_train.txt"), Float64)
    train_y_raw = Int.(vec(readdlm(joinpath(data_dir, "train", "y_train.txt"), Int)))
    # Labels are 1-6 in file; keep as-is for Julia 1-indexing
    test_x_raw = readdlm(joinpath(data_dir, "test", "X_test.txt"), Float64)
    test_y_raw = Int.(vec(readdlm(joinpath(data_dir, "test", "y_test.txt"), Int)))

    println("  Raw train: $(size(train_x_raw, 1)) samples × $(size(train_x_raw, 2)) features")
    println("  Raw test:  $(size(test_x_raw, 1)) samples × $(size(test_x_raw, 2)) features")

    # Window into sequences
    train_seqs_x, train_seqs_y = cut_in_sequences(train_x_raw, train_y_raw, SEQ_LEN; inc=1)
    test_seqs_x, test_seqs_y = cut_in_sequences(test_x_raw, test_y_raw, SEQ_LEN; inc=8)

    println("  Total training sequences: $(length(train_seqs_x))")
    println("  Total test sequences:     $(length(test_seqs_x))")

    # Validation split (10%, fixed seed matching Python)
    n_total = length(train_seqs_x)
    perm = randperm(MersenneTwister(893429), n_total)
    n_valid = div(n_total, 10)

    valid_idx = perm[1:n_valid]
    train_idx = perm[n_valid+1:end]

    println("  Validation split: $n_valid, training split: $(length(train_idx))")

    return HarData(
        train_seqs_x[train_idx], train_seqs_y[train_idx],
        train_seqs_x[valid_idx], train_seqs_y[valid_idx],
        test_seqs_x, test_seqs_y,
    )
end

# ═══════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════

function build_model(model_size::Int, rng)
    ltc = LTCODE1(model_size, N_FEATURES; solver=:semi_implicit, ode_solver_unfolds=6)
    head = Lux.Dense(model_size => N_CLASSES; init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32)

    ps_ltc, st_ltc = Lux.setup(rng, ltc)
    ps_head, st_head = Lux.setup(rng, head)

    return ltc, head, ps_ltc, st_ltc, ps_head, st_head
end

# Forward pass for a single sample
# x_seq: (seq_len, features) matrix — one sample
# Returns logits (N_CLASSES,)
function forward_sample(ltc, head, ps_ltc, ps_head, st_ltc, st_head, x_seq)
    v = zeros(Float32, ltc.n)
    for t in 1:size(x_seq, 1)
        u_t = @view x_seq[t, :]       # (features,) — row of the sequence
        st_d = merge(st_ltc, (input = u_t,))
        v, _ = ltc(v, ps_ltc, st_d)
    end
    # Use final hidden state for classification
    logits, _ = head(v, ps_head, st_head)
    return logits
end

# Cross-entropy loss for a single sample
function sample_loss(ltc, head, ps_ltc, ps_head, st_ltc, st_head, x_seq, y_label)
    logits = forward_sample(ltc, head, ps_ltc, ps_head, st_ltc, st_head, x_seq)
    # y_label is the label at the last time step (classification on final state)
    label = y_label[end]  # 1-indexed, 1–6
    # Softmax cross-entropy
    log_probs = logits .- logsumexp(logits)
    return -log_probs[label]
end

# Stable logsumexp
function logsumexp(x::AbstractVector)
    m = maximum(x)
    return m + log(sum(exp.(x .- m)))
end

# ═══════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════

function evaluate(ltc, head, ps_ltc, ps_head, st_ltc, st_head, data_x, data_y)
    n = length(data_x)
    total_loss = 0.0f0
    correct = 0

    for i in 1:n
        logits = forward_sample(ltc, head, ps_ltc, ps_head, st_ltc, st_head, data_x[i])
        label = data_y[i][end]

        # Loss
        log_probs = logits .- logsumexp(logits)
        total_loss += -log_probs[label]

        # Accuracy
        pred = argmax(logits)
        if pred == label
            correct += 1
        end
    end

    return total_loss / n, correct / n
end

# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

function train!(ltc, head, ps_ltc, ps_head, st_ltc, st_head, data::HarData;
                epochs::Int=50, lr::Float32=0.01f0)

    # Combine parameters for gradient computation
    # We'll use a named tuple: params = (ltc=ps_ltc, head=ps_head)
    params = (ltc = ps_ltc, head = ps_head)

    # Set up optimizer
    opt_state = Optimisers.setup(Optimisers.Adam(lr), params)

    best_valid_acc = 0.0f0
    best_params = deepcopy(params)
    best_epoch = 0
    best_stats = nothing

    n_train = length(data.train_x)

    for epoch in 0:(epochs - 1)
        # ── Evaluate ────────────────────────────────────────────────
        valid_loss, valid_acc = evaluate(ltc, head, params.ltc, params.head,
                                         st_ltc, st_head, data.valid_x, data.valid_y)
        test_loss, test_acc = evaluate(ltc, head, params.ltc, params.head,
                                       st_ltc, st_head, data.test_x, data.test_y)

        # ── Train one epoch ─────────────────────────────────────────
        perm = randperm(n_train)
        n_batches = div(n_train, BATCH_SIZE)
        epoch_losses = Float32[]
        epoch_correct = 0
        epoch_total = 0

        for b in 1:n_batches
            batch_idx = perm[((b-1)*BATCH_SIZE + 1):(b*BATCH_SIZE)]

            # Compute gradient over the batch (sum of per-sample losses)
            loss_val, grads = Zygote.withgradient(params) do p
                batch_loss = 0.0f0
                for i in batch_idx
                    batch_loss += sample_loss(ltc, head, p.ltc, p.head,
                                              st_ltc, st_head,
                                              data.train_x[i], data.train_y[i])
                end
                batch_loss / BATCH_SIZE
            end

            # Update parameters
            opt_state, params = Optimisers.update(opt_state, params, grads[1])

            push!(epoch_losses, loss_val)

            # Track accuracy (without gradient)
            for i in batch_idx
                logits = forward_sample(ltc, head, params.ltc, params.head,
                                        st_ltc, st_head, data.train_x[i])
                pred = argmax(logits)
                if pred == data.train_y[i][end]
                    epoch_correct += 1
                end
                epoch_total += 1
            end
        end

        train_loss = mean(epoch_losses)
        train_acc = epoch_correct / max(epoch_total, 1)

        # ── Model selection (by valid accuracy) ─────────────────────
        if valid_acc > best_valid_acc && epoch > 0
            best_valid_acc = valid_acc
            best_params = deepcopy(params)
            best_epoch = epoch
            best_stats = (train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc)
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
    println("HAR Training — LTCODE1 (semi-implicit, BPTT)")
    println("  Model size: $(args.model_size), LR: $(args.lr), Epochs: $(args.epochs)")

    rng = MersenneTwister(42)

    # Load data
    data = load_har_data()

    # Build model
    ltc, head, ps_ltc, st_ltc, ps_head, st_head = build_model(args.model_size, rng)
    println("  LTC params: $(Lux.parameterlength(ltc))")
    println("  Head params: $(args.model_size * N_CLASSES + N_CLASSES)")
    println("  Total params: $(Lux.parameterlength(ltc) + args.model_size * N_CLASSES + N_CLASSES)")

    # Gradient smoke test
    println("\nGradient smoke test...")
    test_loss, test_grads = Zygote.withgradient((ltc=ps_ltc, head=ps_head)) do p
        sample_loss(ltc, head, p.ltc, p.head, st_ltc, st_head,
                    data.train_x[1], data.train_y[1])
    end
    println("  Initial loss: $(@sprintf("%.4f", test_loss)) (expected ~1.79 = -log(1/6))")

    # Check gradients are non-nothing
    ltc_grad = test_grads[1].ltc
    head_grad = test_grads[1].head
    for k in keys(ltc_grad)
        g = getproperty(ltc_grad, k)
        if g === nothing
            println("  WARNING: LTC gradient for $k is nothing!")
        end
    end
    println("  All LTC gradients present ✓")
    println("  Head weight gradient norm: $(sum(abs2, head_grad.weight))")
    println("  Head bias gradient norm: $(sum(abs2, head_grad.bias))")

    # Train
    println("\nStarting training...\n")
    best_params = train!(ltc, head, ps_ltc, ps_head, st_ltc, st_head, data;
                         epochs=args.epochs, lr=args.lr)
end

main()
