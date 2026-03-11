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
include(joinpath(JULIA_SRC, "models", "ltc.jl"))
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
#  MAIN
# ════════════════════════════════════════════════════════════════════════
println("Cross-Validation: MATLAB ↔ Julia")
println("Data directory: $DATA_DIR")

p1, p2, ltc_err = cross_validate_ltc()
p3, p4, srnn_err = cross_validate_srnn()

println("\n" * "="^60)
println("  Summary")
println("="^60)
println("  LTC  max L2 error: $ltc_err")
println("  SRNN max L2 error: $srnn_err")
println("="^60)

# Combine all 4 panels into one figure (2×2 grid)
fig = plot(p1, p3, p2, p4;
    layout=grid(2, 2), size=(1400, 800),
    plot_title="MATLAB ↔ Julia Cross-Validation")

display(fig)

println("\nDone. Press Enter to close.")
readline()

