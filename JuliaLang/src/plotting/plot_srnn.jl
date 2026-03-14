# plot_srnn.jl — SRNN time series plotting (port of SRNNModel2 plotting methods)
#
# Standalone plotting module for SRNN simulations.
# Depends on Plots.jl — kept separate from srnn.jl to avoid loading Plots during training.
#
# Usage:
#   include("src/plotting/plot_srnn.jl")
#   data = unpack_trajectory(layer, S_all, ps)
#   fig = plot_srnn_tseries(t, layer, data; u=u_matrix, lya_results=lya)
#   display(fig)

using Plots

# Load shared colormaps and palettes
include(joinpath(@__DIR__, "colormaps.jl"))

# ═══════════════════════════════════════════════════════════════════════
# UNPACK TRAJECTORY
# ═══════════════════════════════════════════════════════════════════════

"""
    unpack_trajectory(layer, S_all, ps) → NamedTuple

Unpack a raw state trajectory and compute firing rates, synaptic output, etc.

# Arguments
- `layer::SRNN_ODE`: The ODE layer (needed for dimensions and activation)
- `S_all::AbstractMatrix`: `(state_dim, nt)` — state at each saved time step
- `ps`: Lux parameters (NamedTuple)

# Returns
NamedTuple with fields:
- `x_E`, `x_I`: Dendritic states `(n_E, nt)`, `(n_I, nt)`
- `r_E`, `r_I`: Firing rates `(n_E, nt)`, `(n_I, nt)`
- `br_E`, `br_I`: Synaptic output `(n_E, nt)`, `(n_I, nt)`
- `a_E`: Adaptation `(n_E, n_a_E, nt)` or `nothing`
- `a_I`: Adaptation `(n_I, n_a_I, nt)` or `nothing`
- `b_E`: STD `(n_E, nt)` or `nothing`
- `b_I`: STD `(n_I, nt)` or `nothing`
"""
function unpack_trajectory(layer::SRNN_ODE, S_all::AbstractMatrix, ps)
    n, n_E, n_I = layer.n, layer.n_E, layer.n_I
    n_a_E, n_a_I = layer.n_a_E, layer.n_a_I
    n_b_E, n_b_I = layer.n_b_E, layer.n_b_I
    nt = size(S_all, 2)

    # Unpack using the same layout as _unpack_state, but for whole trajectory
    idx = 0

    # a_E: (n_E * n_a_E) rows → (n_E, n_a_E, nt)
    len_a_E = n_E * n_a_E
    a_E_out = len_a_E > 0 ? reshape(S_all[idx+1:idx+len_a_E, :], n_E, n_a_E, nt) : nothing
    idx += len_a_E

    # a_I
    len_a_I = n_I * n_a_I
    a_I_out = len_a_I > 0 ? reshape(S_all[idx+1:idx+len_a_I, :], n_I, n_a_I, nt) : nothing
    idx += len_a_I

    # b_E: (n_E * n_b_E) rows → (n_E, nt)
    len_b_E = n_E * n_b_E
    b_E_out = len_b_E > 0 ? S_all[idx+1:idx+len_b_E, :] : nothing
    idx += len_b_E

    # b_I
    len_b_I = n_I * n_b_I
    b_I_out = len_b_I > 0 ? S_all[idx+1:idx+len_b_I, :] : nothing
    idx += len_b_I

    # x: (n, nt)
    x = S_all[idx+1:idx+n, :]

    # Compute x_eff (with SFA)
    x_eff = copy(x)
    if !isnothing(a_E_out)
        c_E = NNlib.softplus.(ps.log_c_E)
        sum_a_E = dropdims(sum(a_E_out, dims=2), dims=2)  # (n_E, nt)
        x_eff[1:n_E, :] .-= c_E .* sum_a_E
    end
    if !isnothing(a_I_out)
        c_I = NNlib.softplus.(ps.log_c_I)
        sum_a_I = dropdims(sum(a_I_out, dims=2), dims=2)
        x_eff[n_E+1:n, :] .-= c_I .* sum_a_I
    end

    # Firing rate: r = φ(x_eff - a_0)
    r = layer.activation.(x_eff .- ps.a_0)

    # STD factor b
    b = ones(Float32, n, nt)
    if !isnothing(b_E_out)
        b[1:n_E, :] .= b_E_out
    end
    if !isnothing(b_I_out)
        b[n_E+1:n, :] .= b_I_out
    end

    # Synaptic output
    br = b .* r

    return (
        x_E  = x[1:n_E, :],
        x_I  = x[n_E+1:n, :],
        r_E  = r[1:n_E, :],
        r_I  = r[n_E+1:n, :],
        br_E = br[1:n_E, :],
        br_I = br[n_E+1:n, :],
        a_E  = a_E_out,
        a_I  = a_I_out,
        b_E  = b_E_out,
        b_I  = b_I_out,
    )
end

# Convenience overload for SRNNCell
unpack_trajectory(cell::SRNNCell, S_all::AbstractMatrix, ps) =
    unpack_trajectory(cell.ode, S_all, ps)

# ═══════════════════════════════════════════════════════════════════════
# MULTI-PANEL PLOTTING
# ═══════════════════════════════════════════════════════════════════════

"""
    _plot_ei_lines!(p, t, data_E, data_I; kw...)

Add E and I neuron traces to a subplot using the E/I palettes.
"""
function _plot_ei_lines!(p, t, data_E, data_I; lw=1.0, alpha=0.4)
    n_E = size(data_E, 1)
    n_I = isnothing(data_I) ? 0 : size(data_I, 1)

    for i in 1:n_E
        plot!(p, t, data_E[i, :]; label=false, lw, alpha,
              color=_palette_color(E_PALETTE, i))
    end
    for i in 1:n_I
        plot!(p, t, data_I[i, :]; label=false, lw, alpha,
              color=_palette_color(I_PALETTE, i))
    end
    return p
end

"""
    plot_srnn_tseries(t, layer, data; kwargs...) → Plots.Plot

Create a multi-panel SRNN time series figure.

# Arguments
- `t`: Time vector (length `nt`)
- `layer`: `SRNN_ODE` or `SRNNCell` (for dimension info)
- `data`: NamedTuple from `unpack_trajectory`

# Keyword Arguments
- `u_E=nothing`: External input for E neurons `(n_show, nt)` (for input panel)
- `u_I=nothing`: External input for I neurons `(n_show, nt)`
- `lya_results=nothing`: `(LLE, local_lya, finite_lya)` tuple or NamedTuple
- `lya_t=nothing`: Time vector for Lyapunov data
- `T_plot=nothing`: `(t_start, t_end)` zoom range
- `title_str=""`: Overall figure title
- `figsize=(1200, 0)`: Figure size — height auto-computed from panel count
"""
function plot_srnn_tseries(t, layer, data;
                           u_E=nothing, u_I=nothing,
                           lya_results=nothing, lya_t=nothing,
                           T_plot=nothing, title_str="",
                           figsize=(1200, 0))

    # Get layer from cell if needed
    ode = layer isa SRNNCell ? layer.ode : layer

    # Determine which panels to show
    has_input = !isnothing(u_E) || !isnothing(u_I)
    has_adaptation = ode.n_a_E > 0 || ode.n_a_I > 0
    has_std = ode.n_b_E > 0 || ode.n_b_I > 0
    has_lya = !isnothing(lya_results)

    panels = Plots.Plot[]

    # ── Panel: External input ──────────────────────────────────────────
    if has_input
        p = plot(; ylabel="stim", legend=false)
        if !isnothing(u_E)
            for i in 1:size(u_E, 1)
                plot!(p, t, u_E[i, :]; lw=0.5, alpha=0.6,
                      color=_palette_color(E_PALETTE, i))
            end
        end
        if !isnothing(u_I)
            for i in 1:size(u_I, 1)
                plot!(p, t, u_I[i, :]; lw=0.5, alpha=0.6,
                      color=_palette_color(I_PALETTE, i))
            end
        end
        push!(panels, p)
    end

    # ── Panel: Dendritic states x(t) ──────────────────────────────────
    p_x = plot(; ylabel="dendrite", legend=false)
    _plot_ei_lines!(p_x, t, data.x_E, data.x_I)
    push!(panels, p_x)

    # ── Panel: Firing rates r(t) ──────────────────────────────────────
    p_r = plot(; ylabel="firing rate", legend=false, ylim=(0, 1))
    _plot_ei_lines!(p_r, t, data.r_E, data.r_I)
    push!(panels, p_r)

    # ── Panel: Synaptic output br(t) — only if STD active ────────────
    if has_std
        p_br = plot(; ylabel="synaptic", legend=false, ylim=(0, 1))
        _plot_ei_lines!(p_br, t, data.br_E, data.br_I)
        push!(panels, p_br)
    end

    # ── Panel: Adaptation Σa(t) ──────────────────────────────────────
    if has_adaptation
        p_a = plot(; ylabel="adaptation", legend=false)
        if !isnothing(data.a_E)
            # Sum across timescales: (n_E, n_a_E, nt) → (n_E, nt)
            a_E_sum = dropdims(sum(data.a_E, dims=2), dims=2)
            for i in 1:size(a_E_sum, 1)
                plot!(p_a, t, a_E_sum[i, :]; lw=1.0, alpha=0.5,
                      color=_palette_color(E_PALETTE, i), label=false)
            end
        end
        if !isnothing(data.a_I)
            a_I_sum = dropdims(sum(data.a_I, dims=2), dims=2)
            for i in 1:size(a_I_sum, 1)
                plot!(p_a, t, a_I_sum[i, :]; lw=1.0, alpha=0.5,
                      color=_palette_color(I_PALETTE, i), label=false)
            end
        end
        push!(panels, p_a)
    end

    # ── Panel: STD b(t) ──────────────────────────────────────────────
    if has_std
        p_b = plot(; ylabel="depression", legend=false, ylim=(0, 1))
        if !isnothing(data.b_E)
            for i in 1:size(data.b_E, 1)
                plot!(p_b, t, data.b_E[i, :]; lw=1.0, alpha=0.5,
                      color=_palette_color(E_PALETTE, i), label=false)
            end
        end
        if !isnothing(data.b_I)
            for i in 1:size(data.b_I, 1)
                plot!(p_b, t, data.b_I[i, :]; lw=1.0, alpha=0.5,
                      color=_palette_color(I_PALETTE, i), label=false)
            end
        end
        push!(panels, p_b)
    end

    # ── Panel: Lyapunov ──────────────────────────────────────────────
    if has_lya
        p_l = plot(; ylabel="λ₁", legend=false)

        # Support both NamedTuple and plain tuple
        if lya_results isa NamedTuple
            LLE = lya_results.LLE
            local_lya = lya_results.local_lya
        else
            LLE, local_lya = lya_results[1], lya_results[2]
        end

        t_lya = isnothing(lya_t) ? (1:length(local_lya)) : lya_t

        plot!(p_l, t_lya, local_lya; lw=0.8, alpha=0.7,
              color=:steelblue, label="Local LLE")
        hline!(p_l, [0.0]; ls=:dash, color=:black, lw=1, label=false)
        hline!(p_l, [LLE]; ls=:dash, color=:red, lw=1.5, label=false)

        # Annotate LLE value
        annotate!(p_l, [(t_lya[end], LLE,
                        text("λ₁ = $(round(LLE; digits=3))", 8, :red, :right))])
        push!(panels, p_l)
    end

    # ── Assemble ─────────────────────────────────────────────────────
    n_panels = length(panels)
    height = figsize[2] > 0 ? figsize[2] : n_panels * 180
    fig = plot(panels...; layout=(n_panels, 1),
               size=(figsize[1], height),
               plot_title=title_str,
               link=:x)

    # Apply T_plot zoom
    if !isnothing(T_plot)
        plot!(fig; xlim=T_plot)
    end

    return fig
end
