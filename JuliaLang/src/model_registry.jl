# model_registry.jl — unified model construction for all cell types
#
# Provides:
#   build_cell(model, n, n_in, args, rng) → (cell, ps_cell, st_cell)
#   initial_state(cell, B)                → state matrix
#   readout(cell, S, ps)                  → observation for Dense head (already dispatched)
#
# Supported models: "srnn", "ltc"
# Future: "ctrnn", "node", "ctgru", "lstm"

include(joinpath(@__DIR__, "models", "srnn.jl"))
include(joinpath(@__DIR__, "models", "ltc1.jl"))

"""
    build_cell(model, n, n_in, args, rng) → (cell, ps_cell, st_cell)

Construct a recurrent cell based on `model` name.
Model-specific args (e.g. `n_E`, `n_a`) are read from `args` and ignored if not applicable.
"""
function build_cell(model::String, n::Int, n_in::Int, args, rng)
    if model == "srnn"
        cell = SRNNCell(n, n_in, args.n_E;
            n_a_E=args.n_a, n_b_E=args.n_b,
            ode_solver_unfolds=args.unfolds,
            h=args.h,
            readout=args.readout_mode,
            solver=args.solver,
            per_neuron=args.per_neuron,
            dales=args.dales,
        )
        ps_cell, st_cell = Lux.setup(rng, cell)
        return cell, ps_cell, st_cell
    elseif model == "ltc"
        cell = LTCODE1(n, n_in;
            ode_solver_unfolds=args.unfolds,
            solver=args.solver,
        )
        ps_cell, st_cell = Lux.setup(rng, cell)
        return cell, ps_cell, st_cell
    else
        error("Unknown model: '$model'. Supported: srnn, ltc")
    end
end

"""
    initial_state(cell, B) → state matrix

Return a zero-initialized state for the given cell type and batch size B.
"""
initial_state(cell::SRNNCell, B::Int) = srnn_initial_state(cell, B)
initial_state(cell::LTCODE1, B::Int) = zeros(Float32, cell.n, B)

"""
    hidden_size(cell) → Int

Return the hidden (output) dimension of the cell, for constructing the Dense head.
"""
hidden_size(cell::SRNNCell) = cell.n
hidden_size(cell::LTCODE1) = cell.n

# readout() already dispatches via methods defined in srnn.jl and ltc1.jl:
#   readout(::SRNNCell, S, ps)  → synaptic/rate/dendritic readout
#   readout(::LTCODE1, v, ps)   → identity (v)
