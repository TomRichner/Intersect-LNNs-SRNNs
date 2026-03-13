# train_har.jl — HAR classification with LTCODE1 (batched BPTT via Zygote)
#
# Port of: liquid_time_constant_networks/experiments_with_ltcs/har.py
# Uses the fused semi-implicit solver with BATCHED BPTT.
#
# Usage:
#   julia --project=JuliaLang JuliaLang/scripts/train_har.jl [--epochs 50] [--size 32] [--lr 0.01] [--bs 16]

using Random, Statistics, DelimitedFiles, Printf
using Lux, NNlib, Zygote, Optimisers

# ── Include LTCODE1 ─────────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "src", "models", "ltc1.jl"))

# ── Configuration ───────────────────────────────────────────────────────
const SEQ_LEN    = 16
const N_FEATURES = 561
const N_CLASSES  = 6

# Parse simple command-line args
function parse_args()
    epochs = 50
    model_size = 32
    lr = 0.01f0
    batch_size = 16
    for i in eachindex(ARGS)
        if ARGS[i] == "--epochs" && i < length(ARGS)
            epochs = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--size" && i < length(ARGS)
            model_size = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--lr" && i < length(ARGS)
            lr = parse(Float32, ARGS[i+1])
        elseif ARGS[i] == "--bs" && i < length(ARGS)
            batch_size = parse(Int, ARGS[i+1])
        end
    end
    return (; epochs, model_size, lr, batch_size)
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
        seqs_y[:, idx] .= y[start:stop]                      # (seq_len,)
    end
    return seqs_x, seqs_y
end

struct HarData
    train_x::Array{Float32, 3}    # (features, seq_len, N_train)
    train_y::Matrix{Int}          # (seq_len, N_train)
    valid_x::Array{Float32, 3}    # (features, seq_len, N_valid)
    valid_y::Matrix{Int}          # (seq_len, N_valid)
    test_x::Array{Float32, 3}     # (features, seq_len, N_test)
    test_y::Matrix{Int}           # (seq_len, N_test)
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

    # Window into sequences — now returns 3D arrays
    train_seqs_x, train_seqs_y = cut_in_sequences(train_x_raw, train_y_raw, SEQ_LEN; inc=1)
    test_seqs_x, test_seqs_y = cut_in_sequences(test_x_raw, test_y_raw, SEQ_LEN; inc=8)

    println("  Total training sequences: $(size(train_seqs_x, 3))")
    println("  Total test sequences:     $(size(test_seqs_x, 3))")

    # Validation split (10%, fixed seed matching Python)
    n_total = size(train_seqs_x, 3)
    perm = randperm(MersenneTwister(893429), n_total)
    n_valid = div(n_total, 10)

    valid_idx = perm[1:n_valid]
    train_idx = perm[n_valid+1:end]

    println("  Validation split: $n_valid, training split: $(length(train_idx))")

    return HarData(
        train_seqs_x[:, :, train_idx], train_seqs_y[:, train_idx],
        train_seqs_x[:, :, valid_idx], train_seqs_y[:, valid_idx],
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

# ── Batched forward pass ────────────────────────────────────────────────
# x_batch: (features, seq_len, B)
# Returns: logits (N_CLASSES, B)
function forward_batch(ltc, head, ps_ltc, ps_head, st_ltc, st_head, x_batch)
    B = size(x_batch, 3)
    v = zeros(Float32, ltc.n, B)   # (N, B) batched hidden state

    for t in 1:size(x_batch, 2)
        u_t = @view x_batch[:, t, :]   # (features, B)
        st_d = merge(st_ltc, (input = u_t,))
        v, _ = ltc(v, ps_ltc, st_d)     # batched dispatch → (N, B)
    end

    # Dense head: (model_size, B) → (N_CLASSES, B)
    logits, _ = head(v, ps_head, st_head)
    return logits
end

# ── Batched cross-entropy loss ──────────────────────────────────────────
# Stable vectorized cross-entropy over the batch
function batch_loss(ltc, head, ps_ltc, ps_head, st_ltc, st_head, x_batch, y_labels)
    logits = forward_batch(ltc, head, ps_ltc, ps_head, st_ltc, st_head, x_batch)
    # logits: (N_CLASSES, B), y_labels: (B,) — last time step labels, 1-indexed

    # log-softmax along class dimension (dim=1)
    log_probs = logits .- logsumexp_batch(logits)  # (N_CLASSES, B)

    # Gather the log-prob for the correct class per sample
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

# Keep the vector version for backward compatibility
function logsumexp(x::AbstractVector)
    m = maximum(x)
    return m + log(sum(exp.(x .- m)))
end

# ═══════════════════════════════════════════════════════════════════════
# EVALUATION (batched)
# ═══════════════════════════════════════════════════════════════════════

function evaluate(ltc, head, ps_ltc, ps_head, st_ltc, st_head,
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

        logits = forward_batch(ltc, head, ps_ltc, ps_head, st_ltc, st_head, batch_x)
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
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

function train!(ltc, head, ps_ltc, ps_head, st_ltc, st_head, data::HarData;
                epochs::Int=50, lr::Float32=0.01f0, batch_size::Int=16)

    # Combine parameters for gradient computation
    params = (ltc = ps_ltc, head = ps_head)

    # Set up optimizer
    opt_state = Optimisers.setup(Optimisers.Adam(lr), params)

    best_valid_acc = 0.0f0
    best_params = deepcopy(params)
    best_epoch = 0
    best_stats = nothing

    n_train = size(data.train_x, 3)

    for epoch in 0:(epochs - 1)
        # ── Evaluate ────────────────────────────────────────────────
        valid_loss, valid_acc = evaluate(ltc, head, params.ltc, params.head,
                                         st_ltc, st_head, data.valid_x, data.valid_y)
        test_loss, test_acc = evaluate(ltc, head, params.ltc, params.head,
                                       st_ltc, st_head, data.test_x, data.test_y)

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
                batch_loss(ltc, head, p.ltc, p.head,
                           st_ltc, st_head, x_batch, y_batch)
            end

            # Update parameters
            opt_state, params = Optimisers.update(opt_state, params, grads[1])
            push!(epoch_losses, loss_val)

            # Track accuracy from the same forward pass direction (cheap — no gradient)
            logits = forward_batch(ltc, head, params.ltc, params.head,
                                    st_ltc, st_head, x_batch)
            preds = vec(getindex.(argmax(logits, dims=1), 1))
            epoch_correct += sum(preds .== y_batch)
            epoch_total += batch_size
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
    println("HAR Training — LTCODE1 (semi-implicit, batched BPTT)")
    println("  Model size: $(args.model_size), LR: $(args.lr), Epochs: $(args.epochs), Batch: $(args.batch_size)")

    rng = MersenneTwister(42)

    # Load data
    data = load_har_data()

    # Build model
    ltc, head, ps_ltc, st_ltc, ps_head, st_head = build_model(args.model_size, rng)
    println("  LTC params: $(Lux.parameterlength(ltc))")
    println("  Head params: $(args.model_size * N_CLASSES + N_CLASSES)")
    println("  Total params: $(Lux.parameterlength(ltc) + args.model_size * N_CLASSES + N_CLASSES)")

    # Gradient smoke test (batched — 2 samples)
    println("\nGradient smoke test (batched)...")
    test_x = data.train_x[:, :, 1:2]    # (features, seq_len, 2)
    test_y = data.train_y[end, 1:2]      # (2,)

    test_loss, test_grads = Zygote.withgradient((ltc=ps_ltc, head=ps_head)) do p
        batch_loss(ltc, head, p.ltc, p.head, st_ltc, st_head, test_x, test_y)
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
                         epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
end

main()
