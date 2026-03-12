# cross_validate.jl — Load MATLAB-exported data and compare with Julia ODE integration
#
# Loads .mat files from CrossValidation/data/, reconstructs each model's
# Lux layer with the exact MATLAB parameters, integrates the same ODE,
# and produces overlay comparison plots + L2 error vs time.

using Random, LinearAlgebra, Statistics
using MAT, OrdinaryDiffEq, Plots, NNlib
using Lux

# ── Include source modules (from JuliaLang/) ────────────────────────────
const REPO_DIR = joinpath(@__DIR__, "..", "..")
const JULIA_SRC = joinpath(REPO_DIR, "JuliaLang", "src")

include(joinpath(JULIA_SRC, "connectivity.jl"))
include(joinpath(JULIA_SRC, "activations.jl"))
include(joinpath(JULIA_SRC, "models", "ltc.jl"))
include(joinpath(JULIA_SRC, "models", "ltc1.jl"))
include(joinpath(JULIA_SRC, "models", "ltc2.jl"))
include(joinpath(JULIA_SRC, "models", "srnn.jl"))

const DATA_DIR = joinpath(@__DIR__, "..", "data")

# ── Utility: softplus inverse ──────────────────────────────────────────
"""
    inv_softplus(y) = log(exp(y) - 1)

Inverse of softplus: given τ > 0, returns the unconstrained parameter
such that softplus(log_tau) = τ.
"""
function inv_softplus(y)
    # For large y, log(exp(y) - 1) ≈ y, use that to avoid overflow
    if y > 20.0
        return y
    else
        return log(exp(y) - 1.0)
    end
end

# ════════════════════════════════════════════════════════════════════════
#  LTC CROSS-VALIDATION
# ════════════════════════════════════════════════════════════════════════
function cross_validate_ltc()
    println("\n" * "="^60)
    println("  LTC Cross-Validation")
    println("="^60)

    # Load MATLAB data
    mat_file = joinpath(DATA_DIR, "ltc_cross_val.mat")
    if !isfile(mat_file)
        error("Missing: $mat_file\nRun export_matlab_data.m in MATLAB first.")
    end
    d = matread(mat_file)

    n     = Int(d["n"])
    n_in  = Int(d["n_in"])
    W_mat = Float32.(d["W"])
    W_in  = Float32.(d["W_in"])
    mu    = Float32.(vec(d["mu"]))
    tau   = Float32.(vec(d["tau"]))
    A     = Float32.(vec(d["A"]))
    S0    = Float32.(vec(d["S0"]))
    t_mat = Float64.(vec(d["t_out"]))
    x_mat = Float64.(d["state_out"])   # (nt × n)
    t_ex  = Float64.(vec(d["t_ex"]))
    u_ex  = Float64.(d["u_ex"])        # (n_in × nt_stim)

    println("  Loaded: n=$n, n_in=$n_in, T=[$(t_mat[1]), $(t_mat[end])]")
    println("  MATLAB trajectory: $(size(x_mat))")

    # Build Lux layer
    rng = Random.MersenneTwister(1)
    layer = LTCODE(n, n_in; activation=tanh)
    _, st = Lux.setup(rng, layer)

    # Set parameters from MATLAB (convert τ → log_tau via inv_softplus)
    log_tau = Float32[inv_softplus(Float64(τ_i)) for τ_i in tau]
    ps = (
        W       = W_mat,
        W_in    = W_in,
        mu      = mu,
        log_tau = log_tau,
        A       = A,
    )

    # Verify τ round-trip
    tau_check = NNlib.softplus.(ps.log_tau)
    max_tau_err = maximum(abs.(tau_check .- tau))
    println("  τ round-trip max error: $(max_tau_err)")

    # Build input interpolant from MATLAB stimulus
    dt_stim = t_ex[2] - t_ex[1]
    t0_stim = t_ex[1]
    nt_stim = length(t_ex)
    function u_interp(t)
        idx = clamp(round(Int, (t - t0_stim) / dt_stim) + 1, 1, nt_stim)
        return Float32.(u_ex[:, idx])
    end

    # ODE RHS
    function ltc_rhs!(du, x, p, t)
        st_driven = merge(st, (input = u_interp(t),))
        dxdt, _ = layer(x, p, st_driven)
        du .= dxdt
        return nothing
    end

    # Integrate
    T_span = (Float64(t_mat[1]), Float64(t_mat[end]))
    prob = ODEProblem(ltc_rhs!, S0, T_span, ps)
    sol = solve(prob, Tsit5(); saveat=t_mat, abstol=1e-8, reltol=1e-8)

    x_julia = reduce(hcat, sol.u)'  # (nt × n)
    println("  Julia trajectory: $(size(x_julia))")

    # Compute L2 error vs time
    nt = length(t_mat)
    l2_error = zeros(nt)
    for k in 1:nt
        diff_k = Float64.(x_julia[k, :]) .- x_mat[k, :]
        l2_error[k] = norm(diff_k)
    end

    max_err = maximum(l2_error)
    mean_err = mean(l2_error)
    println("  Max L2 error:  $(max_err)")
    println("  Mean L2 error: $(mean_err)")

    # ── Subplots (returned for combined figure) ────────────────────────
    n_show = min(5, n)
    p1 = plot(title="LTC: MATLAB vs Julia (first $n_show neurons)", ylabel="x(t)")
    for i in 1:n_show
        plot!(p1, t_mat, x_mat[:, i]; label=(i==1 ? "MATLAB" : false),
              color=:steelblue, lw=1.5, alpha=0.7)
        plot!(p1, t_mat, Float64.(x_julia[:, i]); label=(i==1 ? "Julia" : false),
              color=:firebrick, lw=1.0, ls=:dash, alpha=0.8)
    end

    p2 = plot(t_mat, l2_error;
        title="LTC: L2 Error vs Time",
        ylabel="L2 Error", xlabel="Time (s)",
        lw=1.5, color=:black, label=false)
    hline!(p2, [mean_err]; ls=:dash, color=:gray, label="mean=$(round(mean_err, sigdigits=3))")

    return p1, p2, max_err
end

# ════════════════════════════════════════════════════════════════════════
#  SRNN CROSS-VALIDATION
# ════════════════════════════════════════════════════════════════════════
function cross_validate_srnn()
    println("\n" * "="^60)
    println("  SRNN Cross-Validation")
    println("="^60)

    # Load MATLAB data
    mat_file = joinpath(DATA_DIR, "srnn_cross_val.mat")
    if !isfile(mat_file)
        error("Missing: $mat_file\nRun export_matlab_data.m in MATLAB first.")
    end
    d = matread(mat_file)

    n       = Int(d["n"])
    n_E     = Int(d["n_E"])
    n_I     = Int(d["n_I"])
    n_a_E   = Int(d["n_a_E"])
    n_a_I   = Int(d["n_a_I"])
    n_b_E   = Int(d["n_b_E"])
    n_b_I   = Int(d["n_b_I"])
    W_mat   = Float32.(d["W"])
    W_in    = Float32.(d["W_in"])
    tau_d   = Float64(d["tau_d"])
    c_E     = Float64(d["c_E"])
    c_I     = Float64(d["c_I"])
    tau_a_E = Float64.(vec(d["tau_a_E"]))
    tau_b_E_rec = Float64(d["tau_b_E_rec"])
    tau_b_E_rel = Float64(d["tau_b_E_rel"])
    S_a     = Float64(d["S_a"])
    S_c     = Float64(d["S_c"])

    S0      = Float32.(vec(d["S0"]))
    t_mat   = Float64.(vec(d["t_out"]))
    S_mat   = Float64.(d["state_out"])   # (nt × state_dim)
    t_ex    = Float64.(vec(d["t_ex"]))
    u_ex    = Float64.(d["u_ex"])        # (n × nt_stim)

    # Input dimension = number of rows in W_in
    n_in = size(W_in, 2)

    println("  Loaded: n=$n, n_E=$n_E, n_a_E=$n_a_E, n_b_E=$n_b_E")
    println("  MATLAB trajectory: $(size(S_mat))")

    # Build Lux layer with matching activation
    activation = make_piecewise_sigmoid(; S_a=S_a, S_c=S_c)
    layer = SRNN_ODE(n, n_in, n_E; n_a_E=n_a_E, n_a_I=n_a_I,
                     n_b_E=n_b_E, n_b_I=n_b_I, activation=activation)

    rng = Random.MersenneTwister(1)
    _, st = Lux.setup(rng, layer)

    # Build parameter NamedTuple from MATLAB values
    ps_dict = Dict{Symbol, Any}(
        :W         => W_mat,
        :W_in      => W_in,
        :log_tau_d => Float32[inv_softplus(tau_d)],
    )

    if n_a_E > 0
        ps_dict[:log_tau_a_E] = Float32[inv_softplus(t) for t in tau_a_E]
        ps_dict[:log_c_E] = Float32[inv_softplus(c_E)]
    end
    if n_a_I > 0
        ps_dict[:log_c_I] = Float32[inv_softplus(c_I)]
    end
    if n_b_E > 0
        ps_dict[:log_tau_b_E_rec] = Float32[inv_softplus(tau_b_E_rec)]
        ps_dict[:log_tau_b_E_rel] = Float32[inv_softplus(tau_b_E_rel)]
    end

    ps = NamedTuple(ps_dict)

    # Verify round-trips
    tau_d_check = NNlib.softplus(ps.log_tau_d[1])
    println("  τ_d round-trip: MATLAB=$(tau_d), Julia=$(tau_d_check), err=$(abs(tau_d_check - tau_d))")

    # Build input interpolant
    dt_stim = t_ex[2] - t_ex[1]
    t0_stim = t_ex[1]
    nt_stim = length(t_ex)
    function u_interp(t)
        idx = clamp(round(Int, (t - t0_stim) / dt_stim) + 1, 1, nt_stim)
        return Float32.(u_ex[:, idx])
    end

    # ODE RHS
    function srnn_rhs!(dS, S, p, t)
        st_driven = merge(st, (input = u_interp(t),))
        dS_dt, _ = layer(S, p, st_driven)
        dS .= dS_dt
        return nothing
    end

    # Integrate
    T_span = (Float64(t_mat[1]), Float64(t_mat[end]))
    prob = ODEProblem(srnn_rhs!, S0, T_span, ps)
    sol = solve(prob, Tsit5(); saveat=t_mat, abstol=1e-8, reltol=1e-8)

    S_julia = reduce(hcat, sol.u)'  # (nt × state_dim)
    println("  Julia trajectory: $(size(S_julia))")

    # Compute L2 error vs time (on full state vector)
    nt = length(t_mat)
    l2_error = zeros(nt)
    for k in 1:nt
        diff_k = Float64.(S_julia[k, :]) .- S_mat[k, :]
        l2_error[k] = norm(diff_k)
    end

    max_err = maximum(l2_error)
    mean_err = mean(l2_error)
    println("  Max L2 error:  $(max_err)")
    println("  Mean L2 error: $(mean_err)")

    # Extract dendritic states x for overlay plotting
    x_offset = n_E * n_a_E + n_I * n_a_I + n_E * n_b_E + n_I * n_b_I
    x_mat_only = S_mat[:, x_offset+1:x_offset+n]
    x_jul_only = Float64.(S_julia[:, x_offset+1:x_offset+n])

    # ── Subplots (returned for combined figure) ──────────────────────
    n_show = min(5, n)
    p3 = plot(title="SRNN: MATLAB vs Julia — dendritic x (first $n_show neurons)",
              ylabel="x(t)")
    for i in 1:n_show
        plot!(p3, t_mat, x_mat_only[:, i]; label=(i==1 ? "MATLAB" : false),
              color=:steelblue, lw=1.5, alpha=0.7)
        plot!(p3, t_mat, x_jul_only[:, i]; label=(i==1 ? "Julia" : false),
              color=:firebrick, lw=1.0, ls=:dash, alpha=0.8)
    end

    p4 = plot(t_mat, l2_error;
        title="SRNN: L2 Error vs Time",
        ylabel="L2 Error", xlabel="Time (s)",
        lw=1.5, color=:black, label=false)
    hline!(p4, [mean_err]; ls=:dash, color=:gray, label="mean=$(round(mean_err, sigdigits=3))")

    return p3, p4, max_err
end

# ════════════════════════════════════════════════════════════════════════
#  LTC2 CROSS-VALIDATION (Hasani MATLAB-faithful)
# ════════════════════════════════════════════════════════════════════════
function cross_validate_ltc2()
    println("\n" * "="^60)
    println("  LTC2 Cross-Validation (Hasani MATLAB-faithful)")
    println("="^60)

    mat_file = joinpath(DATA_DIR, "ltc2_cross_val.mat")
    if !isfile(mat_file)
        println("  SKIPPED: Missing $mat_file")
        return nothing, nothing, NaN
    end
    d = matread(mat_file)

    n       = Int(d["n"])
    n_in    = Int(d["n_in"])
    k       = Int(d["k"])
    n_layers = Int(d["n_layers"])
    S0      = Float32.(vec(d["S0"]))
    t_mat   = Float64.(vec(d["t_out"]))
    x_mat   = Float64.(d["state_out"])
    t_ex    = Float64.(vec(d["t_ex"]))
    u_ex    = Float64.(d["u_ex"])
    tau     = Float32.(vec(d["tau"]))

    println("  Loaded: n=$n, k=$k, n_layers=$n_layers")

    # Build Lux layer
    rng = Random.MersenneTwister(1)
    layer = LTCODE2(k, n_in; n_layers=n_layers)
    _, st = Lux.setup(rng, layer)

    # Helper: MAT.jl reads MATLAB cell arrays as Vector{Any} (multi-cell) or
    # Matrix{Any} containing the matrix (single-cell). Extract properly.
    function _extract_cell(cell_data, idx)
        if cell_data isa Vector
            return cell_data[idx]
        elseif cell_data isa Matrix && eltype(cell_data) == Any
            # Single-cell: (1×1) Matrix{Any} wrapping the actual matrix
            return cell_data[idx]
        else
            return cell_data  # Already a plain matrix (single layer)
        end
    end

    # Reconstruct parameters from MATLAB cell arrays
    params = Dict{Symbol, Any}()
    params[:tau] = tau
    for j in 1:n_layers
        W_ff_j  = Float32.(_extract_cell(d["W_ff"], j))
        b_ff_j  = Float32.(vec(_extract_cell(d["b_ff"], j)))
        E_ff_j  = Float32.(_extract_cell(d["E_ff"], j))
        W_rec_j = Float32.(_extract_cell(d["W_rec"], j))
        b_rec_j = Float32.(vec(_extract_cell(d["b_rec"], j)))
        E_rec_j = Float32.(_extract_cell(d["E_rec"], j))

        params[Symbol("W_ff_$j")]  = W_ff_j
        params[Symbol("b_ff_$j")]  = b_ff_j
        params[Symbol("E_ff_$j")]  = E_ff_j
        params[Symbol("W_rec_$j")] = W_rec_j
        params[Symbol("b_rec_$j")] = b_rec_j
        params[Symbol("E_rec_$j")] = E_rec_j
    end
    ps = NamedTuple(params)

    # Input interpolant
    dt_stim = t_ex[2] - t_ex[1]
    t0_stim = t_ex[1]
    nt_stim = length(t_ex)
    function u_interp(t)
        idx = clamp(round(Int, (t - t0_stim) / dt_stim) + 1, 1, nt_stim)
        return Float32.(u_ex[:, idx])
    end

    function ltc2_rhs!(du, x, p, t)
        st_d = merge(st, (input = u_interp(t),))
        dxdt, _ = layer(x, p, st_d)
        du .= dxdt
        return nothing
    end

    T_span = (Float64(t_mat[1]), Float64(t_mat[end]))
    prob = ODEProblem(ltc2_rhs!, S0, T_span, ps)
    sol = solve(prob, Tsit5(); saveat=t_mat, abstol=1e-8, reltol=1e-8)

    x_julia = reduce(hcat, sol.u)'
    nt = length(t_mat)
    l2_error = [norm(Float64.(x_julia[i, :]) .- x_mat[i, :]) for i in 1:nt]

    max_err = maximum(l2_error)
    mean_err = mean(l2_error)
    println("  Max L2 error:  $max_err")
    println("  Mean L2 error: $mean_err")

    n_show = min(5, n)
    p_overlay = plot(title="LTC2: MATLAB vs Julia (first $n_show neurons)", ylabel="x(t)")
    for i in 1:n_show
        plot!(p_overlay, t_mat, x_mat[:, i]; label=(i==1 ? "MATLAB" : false),
              color=:steelblue, lw=1.5, alpha=0.7)
        plot!(p_overlay, t_mat, Float64.(x_julia[:, i]); label=(i==1 ? "Julia" : false),
              color=:firebrick, lw=1.0, ls=:dash, alpha=0.8)
    end
    p_err = plot(t_mat, l2_error; title="LTC2: L2 Error", ylabel="L2", xlabel="Time (s)",
        lw=1.5, color=:black, label=false)
    hline!(p_err, [mean_err]; ls=:dash, color=:gray, label="mean=$(round(mean_err, sigdigits=3))")

    return p_overlay, p_err, max_err
end

# ════════════════════════════════════════════════════════════════════════
#  LTC1 CROSS-VALIDATION (Hasani Python-faithful, fused solver)
# ════════════════════════════════════════════════════════════════════════
function cross_validate_ltc1()
    println("\n" * "="^60)
    println("  LTC1 Cross-Validation (Hasani Python-faithful)")
    println("="^60)

    mat_file = joinpath(DATA_DIR, "ltc1_cross_val.mat")
    if !isfile(mat_file)
        println("  SKIPPED: Missing $mat_file")
        return nothing, nothing, NaN
    end
    d = matread(mat_file)

    n     = Int(d["n"])
    n_in  = Int(d["n_in"])
    S0    = Float32.(vec(d["S0"]))
    t_mat = Float64.(vec(d["t_out"]))
    x_mat = Float64.(d["state_out"])
    t_ex  = Float64.(vec(d["t_ex"]))
    u_ex  = Float64.(d["u_ex"])

    println("  Loaded: n=$n, n_in=$n_in")

    # Build Lux layer
    rng = Random.MersenneTwister(1)
    layer = LTCODE1(n, n_in; solver=:semi_implicit, ode_solver_unfolds=6)
    _, st = Lux.setup(rng, layer)

    # Reconstruct parameters from MATLAB
    _inv_sp(y) = y > 20.0 ? Float32(y) : Float32(log(exp(Float64(y)) - 1.0))

    ps = (
        W_raw           = Float32.(_inv_sp.(d["W_syn"])),
        mu              = Float32.(d["mu_syn"]),
        sigma           = Float32.(d["sigma_syn"]),
        erev            = Float32.(d["erev"]),
        sensory_W_raw   = Float32.(_inv_sp.(d["sensory_W"])),
        sensory_mu      = Float32.(d["sensory_mu"]),
        sensory_sigma   = Float32.(d["sensory_sigma"]),
        sensory_erev    = Float32.(d["sensory_erev"]),
        vleak           = Float32.(vec(d["vleak"])),
        gleak_raw       = Float32.(_inv_sp.(vec(d["gleak"]))),
        cm_raw          = Float32.(_inv_sp.(vec(d["cm"]))),
        input_w         = Float32.(vec(d["input_w"])),
        input_b         = Float32.(vec(d["input_b"])),
    )

    # Verify round-trips
    W_check = NNlib.softplus.(ps.W_raw)
    max_W_err = maximum(abs.(W_check .- Float32.(d["W_syn"])))
    println("  W round-trip max error: $max_W_err")

    # Input interpolant
    dt_stim = t_ex[2] - t_ex[1]
    t0_stim = t_ex[1]
    nt_stim = length(t_ex)
    function u_interp(t)
        idx = clamp(round(Int, (t - t0_stim) / dt_stim) + 1, 1, nt_stim)
        return Float32.(u_ex[:, idx])
    end

    # Step through with fused solver (matching MATLAB)
    nt = length(t_mat)
    x_julia = zeros(Float32, nt, n)
    v = copy(S0)
    x_julia[1, :] .= v

    for k_step in 2:nt
        u_k = u_interp(t_mat[k_step - 1])
        st_d = merge(st, (input = u_k,))
        v, _ = layer(v, ps, st_d)
        x_julia[k_step, :] .= v
    end

    l2_error = [norm(Float64.(x_julia[i, :]) .- x_mat[i, :]) for i in 1:nt]
    max_err = maximum(l2_error)
    mean_err = mean(l2_error)
    println("  Max L2 error:  $max_err")
    println("  Mean L2 error: $mean_err")

    n_show = min(5, n)
    p_overlay = plot(title="LTC1: MATLAB vs Julia (first $n_show neurons)", ylabel="v(t)")
    for i in 1:n_show
        plot!(p_overlay, t_mat, x_mat[:, i]; label=(i==1 ? "MATLAB" : false),
              color=:steelblue, lw=1.5, alpha=0.7)
        plot!(p_overlay, t_mat, Float64.(x_julia[:, i]); label=(i==1 ? "Julia" : false),
              color=:firebrick, lw=1.0, ls=:dash, alpha=0.8)
    end
    p_err = plot(t_mat, l2_error; title="LTC1: L2 Error", ylabel="L2", xlabel="Time (s)",
        lw=1.5, color=:black, label=false)
    hline!(p_err, [mean_err]; ls=:dash, color=:gray, label="mean=$(round(mean_err, sigdigits=3))")

    return p_overlay, p_err, max_err
end

# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════
println("Cross-Validation: MATLAB ↔ Julia")
println("Data directory: $DATA_DIR")

p1, p2, ltc_err = cross_validate_ltc()
p3, p4, srnn_err = cross_validate_srnn()
p5, p6, ltc2_err = cross_validate_ltc2()
p7, p8, ltc1_err = cross_validate_ltc1()

println("\n" * "="^60)
println("  Summary")
println("="^60)
println("  LTC  (original) max L2 error: $ltc_err")
println("  SRNN            max L2 error: $srnn_err")
println("  LTC2 (MATLAB)   max L2 error: $ltc2_err")
println("  LTC1 (Python)   max L2 error: $ltc1_err")
println("="^60)

# Build figure: original models on top row, new models on bottom
all_plots = Any[p1, p3, p2, p4]  # original LTC + SRNN
if !isnothing(p5)
    push!(all_plots, p5, p7, p6, p8)
    fig = plot(all_plots...;
        layout=grid(4, 2), size=(1400, 1600),
        plot_title="MATLAB ↔ Julia Cross-Validation (All Models)")
else
    fig = plot(all_plots...;
        layout=grid(2, 2), size=(1400, 800),
        plot_title="MATLAB ↔ Julia Cross-Validation")
end

display(fig)

println("\nDone. Press Enter to close.")
readline()
