# lyapunov.jl — Benettin's algorithm for largest Lyapunov exponent (LLE)
#
# Port of MATLAB cRNN.benettin_algorithm_internal.
# Cell-agnostic: works with any stepper closure (SRNNCell, LTCODE1, etc.)
#
# Usage with SRNNCell:
#   # 1. Pre-compute reference trajectory
#   S_traj = collect_trajectory(cell, ps, st, inputs)
#
#   # 2. Build stepper closure that replays the same inputs
#   stepper = (S, step_idx) -> begin
#       u_t = inputs[:, step_idx]
#       st_d = merge(st, (input = u_t,))
#       S_next, _ = cell(S, ps, st_d)
#       return S_next
#   end
#
#   # 3. Compute LLE
#   tau = steps_per_interval * cell.ode_solver_unfolds * cell.h
#   LLE, local_lya, finite_lya = benettin_lle(S_traj, stepper, steps_per_interval; tau_interval=tau)

using LinearAlgebra, Random

"""
    benettin_lle(S_trajectory, stepper, steps_per_interval; kwargs...)

Compute the largest Lyapunov exponent using Benettin's algorithm.

The reference trajectory is pre-computed and stored in `S_trajectory`.
Only the perturbed copy is re-stepped at each renormalization interval.

# Arguments
- `S_trajectory::AbstractMatrix`: `(state_dim, n_total_steps+1)` — reference states
  saved at every macro-step. Column 1 is the initial state, column `k+1` is
  the state after `k` macro-steps.
- `stepper`: Closure `(S, step_idx) → S_next` that advances the state by one
  macro-step. `step_idx` is the 1-based global step index so the closure can
  replay the correct input for that time step.
- `steps_per_interval::Int`: Number of macro-steps per renormalization interval.

# Keyword Arguments
- `tau_interval::Float32 = 1f0`: Real-time duration of one renormalization
  interval (e.g., `steps_per_interval × ode_solver_unfolds × h`). Used to
  convert log-stretching to a proper Lyapunov exponent in units of 1/time.
- `d0::Float32 = 1f-3`: Initial perturbation magnitude.
- `skip_intervals::Int = 0`: Number of initial intervals to skip before
  accumulating `finite_lya` (transient burn-in).
- `rng::AbstractRNG = Random.default_rng()`: RNG for initial perturbation direction.

# Returns
- `LLE::Float32`: Largest Lyapunov exponent (time-averaged, in 1/time).
- `local_lya::Vector{Float32}`: Instantaneous exponent per interval (1/time).
- `finite_lya::Vector{Float32}`: Running time-average exponent (1/time).
  `NaN` for intervals within the burn-in period.
"""
function benettin_lle(
    S_trajectory::AbstractMatrix,
    stepper,
    steps_per_interval::Int;
    tau_interval::Float32 = 1f0,
    d0::Float32 = 1f-3,
    skip_intervals::Int = 0,
    rng::AbstractRNG = Random.default_rng(),
)
    state_dim, n_cols = size(S_trajectory)
    n_total_steps = n_cols - 1  # column 1 = initial state

    n_intervals = n_total_steps ÷ steps_per_interval
    if n_intervals < 1
        error("Trajectory too short: $(n_total_steps) steps, need at least $(steps_per_interval).")
    end

    # Allocate output arrays
    local_lya = zeros(Float32, n_intervals)
    finite_lya = fill(NaN32, n_intervals)
    sum_log_stretching = 0.0f0
    accumulated_time = 0.0f0

    # Random unit perturbation direction, scaled to d0
    rnd_dir = randn(rng, Float32, state_dim)
    pert = (rnd_dir ./ norm(rnd_dir)) .* d0

    for k in 1:n_intervals
        # Reference trajectory indices (1-based)
        # Interval k covers macro-steps [(k-1)*spi + 1, ..., k*spi]
        # Reference start = column (k-1)*spi + 1
        # Reference end   = column k*spi + 1
        ref_start_col = (k - 1) * steps_per_interval + 1
        ref_end_col = k * steps_per_interval + 1

        S_ref_start = S_trajectory[:, ref_start_col]
        S_ref_end = S_trajectory[:, ref_end_col]

        # Perturbed trajectory: start from reference + perturbation
        S_pert = S_ref_start .+ pert

        # Step the perturbed copy forward through the interval
        for step in 1:steps_per_interval
            global_step_idx = (k - 1) * steps_per_interval + step
            S_pert = stepper(S_pert, global_step_idx)
        end

        # Measure divergence
        delta = S_pert .- S_ref_end
        d_k = norm(delta)

        # Local exponent (1/time)
        local_lya[k] = log(d_k / d0) / tau_interval

        # Early termination on divergence
        if !isfinite(local_lya[k])
            @warn "System diverged at interval $k. Truncating."
            local_lya = local_lya[1:k-1]
            finite_lya = finite_lya[1:k-1]
            valid = filter(!isnan, finite_lya)
            LLE = isempty(valid) ? 0f0 : last(valid)
            return LLE, local_lya, finite_lya
        end

        # Renormalize perturbation
        pert = (delta ./ d_k) .* d0

        # Accumulate running average (skip burn-in intervals)
        if k > skip_intervals
            sum_log_stretching += log(d_k / d0)
            accumulated_time += tau_interval
            finite_lya[k] = sum_log_stretching / accumulated_time
        end
    end

    # Final LLE = last valid finite_lya
    valid = filter(!isnan, finite_lya)
    LLE = isempty(valid) ? 0f0 : last(valid)

    return LLE, local_lya, finite_lya
end

"""
    collect_trajectory(cell, ps, st, inputs; rng=Random.default_rng()) → S_trajectory

Run `cell` forward through the input sequence and save the state at every macro-step.

# Arguments
- `cell`: A Lux cell (e.g., `SRNNCell` or `LTCODE1`)
- `ps`: Parameters (NamedTuple)
- `st`: Lux state (NamedTuple with `:input` field)
- `inputs::AbstractMatrix`: `(n_in, T)` — input at each time step

# Returns
- `S_trajectory::Matrix{Float32}`: `(state_dim, T+1)` — column 1 is S0,
  column `t+1` is state after step `t`.
"""
function collect_trajectory(cell, ps, st, inputs, S0::AbstractVector;)
    n_in, T = size(inputs)
    state_dim = length(S0)
    S_traj = zeros(Float32, state_dim, T + 1)
    S_traj[:, 1] .= S0

    S = copy(S0)
    for t in 1:T
        u_t = @view inputs[:, t]
        st_d = merge(st, (input = u_t,))
        S, _ = cell(S, ps, st_d)
        S_traj[:, t + 1] .= S
    end

    return S_traj
end
