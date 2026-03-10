# cRNN Base Class Refactor

Refactored `SRNNModel2.m` and `LNN.m` to inherit from a shared abstract base class `cRNN.m`, using a strategy pattern for pluggable connectivity, activation, and stimulus components.

**Date:** March 10, 2026  
**Commits:** `f80c8e4`, `f82cf48`, `88c2d4a`

---

## Motivation

`SRNNModel2` and `LNN` shared significant duplicated code:
- ~15 identical properties (n, f, indegree, fs, T_range, ode_solver, etc.)
- Identical RMT connectivity generation via `RMTMatrix`
- Identical lifecycle methods (`build()`, `run()`, `clear_results()`, `reset()`)
- Identical `piecewiseSigmoid` static methods
- Identical dependent property getters (`alpha`, `n_E`, `n_I`)

The refactor extracts shared code into `cRNN`, reducing total lines by ~1100 and enabling a future unified `ESN_reservoir` class.

---

## Architecture

```
                 cRNN < handle (abstract)
                 ├── Shared properties (n, f, indegree, fs, T_range, ...)
                 ├── Shared lifecycle (build → run → plot)
                 ├── Lyapunov computation (Benettin, QR)
                 ├── Strategy objects:
                 │   ├── connectivity  (Connectivity)
                 │   ├── activation    (Activation)
                 │   └── stimulus      (Stimulus)
                 ├── Abstract: get_rhs, get_readout_features,
                 │             get_jacobian, initialize_state,
                 │             decimate_and_unpack
                 │
            ┌────┴────┐
            │         │
       SRNNModel2    LNN
       (SFA/STD,     (LTC ODE,
        Lyapunov)    fused solver)
```

### Strategy Classes

| Interface      | Concrete Classes                                                            | Purpose                   |
| -------------- | --------------------------------------------------------------------------- | ------------------------- |
| `Connectivity` | `RMTConnectivity`                                                           | Weight matrix generation  |
| `Activation`   | `TanhActivation`, `SigmoidActivation`, `PiecewiseSigmoid`, `ReLUActivation` | Nonlinearity + derivative |
| `Stimulus`     | `StepStimulus`, `SinusoidalStimulus`, `ESNStimulus`                         | Input signal generation   |

---

## File Locations

All shared classes live in `Matlab/shared/src/`:

```
Matlab/shared/src/
├── cRNN.m                  # Abstract base class (~730 lines)
├── Connectivity.m          # Abstract connectivity interface
├── RMTConnectivity.m       # RMT weight matrix strategy
├── Activation.m            # Abstract activation interface
├── PiecewiseSigmoid.m      # Piecewise sigmoid (Harris 2023)
├── TanhActivation.m        # tanh activation
├── SigmoidActivation.m     # Logistic sigmoid
├── ReLUActivation.m        # ReLU activation
├── Stimulus.m              # Abstract stimulus interface
├── StepStimulus.m           # Step function stimulus (from SRNNModel2)
├── SinusoidalStimulus.m    # Sinusoidal stimulus (from LNN)
├── ESNStimulus.m           # ESN white/bandlimited noise stimulus
└── RMTMatrix.m             # RMT weight matrix generator (moved from SRNN/src/)
```

---

## SRNNModel2 Refactor

**Before:** 2594 lines, standalone `< handle`  
**After:** ~1630 lines, `< cRNN`  
**Reduction:** ~37%

### Removed from SRNNModel2 (now in cRNN or strategies)

**Properties (~30):**
- Network architecture: `n`, `f`, `indegree`, `mu_E_tilde`, `mu_I_tilde`, `sigma_E_tilde`, `sigma_I_tilde`, `E_W`, `zrs_mode`, `level_of_chaos`
- Simulation: `fs`, `T_range`, `T_plot`, `ode_solver`, `ode_opts`, `rng_seeds`
- Storage: `store_full_state`, `store_decimated_state`, `plot_deci`, `plot_freq`
- Protected: `W`, `W_in`, `t_ex`, `u_ex`, `u_interpolant`, `S0`, `is_built`, `cached_params`, `t_out`, `state_out`, `plot_data`, `lya_results`, `has_run`
- Dependent: `alpha`, `n_E`, `n_I`, `E_indices`, `I_indices`
- Lyapunov: `lya_method`, `lya_T_interval`, `filter_local_lya`, `lya_filter_order`, `lya_filter_cutoff`

**Properties moved to RMTConnectivity:**
- `rescale_by_abscissa`, `default_val`, `mu_se`, `mu_si`, `sigma_se`, `sigma_si`, `R` (dependent)

**Methods removed:**
- Lifecycle: `build()`, `run()`, `compute_lyapunov()`, `filter_lyapunov()`, `clear_results()`, `reset()`
- Build helpers: `build_stimulus()`, `finalize_build()`, `set_defaults()`, `generate_stimulus()`
- Lyapunov statics: `compute_lyapunov_exponents_internal`, `benettin_algorithm_internal`, `lyapunov_spectrum_qr_internal`, `compute_kaplan_yorke_dimension_internal`, `variational_eqs_ode_internal`
- Activation statics: `piecewiseSigmoid`, `piecewiseSigmoidDerivative`
- Stimulus static: `generate_external_input`

### Added to SRNNModel2

**Abstract method implementations:**
- `get_rhs(params)` → returns `@(t, S) SRNNModel2.dynamics_fast(t, S, params)`
- `get_readout_features()` → unpacks state, returns `[r_E; r_I]` (n × T firing rates)
- `get_jacobian(S, params)` → delegates to `compute_Jacobian_fast`
- `initialize_state()` → sets packed `S0` with zeros for adaptation/STD, small random for x
- `get_n_state()` → returns `N_sys_eqs`
- `get_state_bounds()` → returns bounds for STD `b` variables `[0, 1]`

**Overridden methods:**
- `get_params()` → calls `get_params@cRNN()`, adds SRNN-specific fields (SFA, STD, tau_d, activation handles)
- `build_network()` → sets tau_a defaults, calls `build_network@cRNN()`

**Constructor:**
- Calls `obj@cRNN()`
- Sets SRNN defaults: `n=300`, `indegree=100`, `T_range=[0,50]`, `lya_method='benettin'`
- Creates strategy objects: `PiecewiseSigmoid('S_a', 0.9, 'S_c', 0.35)`, `StepStimulus()`, `RMTConnectivity()`

### Preserved intact
- `dynamics_fast` (static ODE RHS)
- `compute_Jacobian_fast` (static Jacobian)
- `compute_Jacobian_at_indices` (static)
- `unpack_and_compute_states` (state unpacking helper)
- `decimate_and_unpack` (model-specific decimation)
- `plot()` and all plotting helpers
- `plot_W_spectrum`, `plot_eigenvalues`
- `validate` (SRNN-specific validation)

---

## LNN Refactor

**Before:** 664 lines, standalone `< handle`  
**After:** ~469 lines, `< cRNN`  
**Reduction:** ~29%

### Removed from LNN (now in cRNN or strategies)

**Properties (~24):**
- Network: `n`, `f`, `indegree`, `mu_E_tilde`, `mu_I_tilde`, `sigma_E_tilde`, `sigma_I_tilde`, `level_of_chaos`, `zrs_mode`
- Simulation: `fs`, `T_range`, `T_plot`, `ode_solver`, `ode_opts`
- Storage: `store_full_state`, `plot_deci`, `plot_freq`
- Protected: `W_r` (→ `W`), `S0`, `u_interpolant`, `is_built`, `cached_params`, `t_ex`, `u_ex`
- Results: `t_out`, `x_out` (→ `state_out`), `plot_data`, `has_run`
- Dependent: `alpha`
- Activation: `S_a`, `S_c` (→ `PiecewiseSigmoid` strategy)
- Input: `input_func` (→ `SinusoidalStimulus` strategy)
- Seed: `rng_seed` (→ `rng_seeds`)

**Methods removed:**
- Lifecycle: `build()`, `clear_results()`, `reset()`
- Build helpers: `build_network()` (RMT part), `build_stimulus()`, `finalize_build()`
- Activation statics: `apply_activation`, `tanhActivation`, `logisticSigmoid`, `reluActivation`, `piecewiseSigmoid` (~80 lines)

### Added to LNN

**Abstract method implementations:**
- `get_rhs(params)` → returns `@(t, x) LNN.dynamics_ltc(t, x, params)`
- `get_readout_features()` → returns `state_out'` (n × T raw states)
- `get_jacobian(S, params)` → **new** LTC Jacobian via chain rule
- `initialize_state()` → `0.01 * randn(n, 1)`

**Overridden methods:**
- `run()` → handles fused solver mode, delegates ODE mode to `run@cRNN`
- `get_params()` → calls `get_params@cRNN()`, adds `n_in`, `mu`, `tau`, `A`
- `build_network()` → calls `build_network@cRNN()`, then inits `W_in`, `mu`, `tau`, `A`

**New:**
- `make_activation(name)` — static factory: maps string name → Activation strategy object
- `activation_name` property — legacy convenience for string-based activation selection

**Constructor:**
- Calls `obj@cRNN()`
- Sets LNN defaults: `n=100`, `T_range=[0,10]`, `store_full_state=true`, `rng_seeds=[1 1]`
- Creates strategy objects: `TanhActivation()`, `SinusoidalStimulus('n_in', 2)`, `RMTConnectivity()`
- Routes legacy name-value pairs (`level_of_chaos`, `mu_E_tilde`, etc.) to connectivity strategy

### Key change: activation dispatch

**Before** (string dispatch):
```matlab
z = params.W_r * x + params.W_in * I_t + params.mu;
f_val = LNN.apply_activation(z, params.activation, params);
```

**After** (strategy object):
```matlab
z = params.W * x + params.W_in * I_t + params.mu;
f_val = params.activation.apply(z);
```

### Preserved intact
- `dynamics_ltc` (static ODE RHS, updated to use `params.activation.apply()`)
- `fused_step` (static fused solver step, updated similarly)
- `run_fused` (protected fused solver loop)
- `decimate_and_unpack` (LNN-specific, computes `f_vals`, `tau_sys`)
- `plot()` (4-panel: input, states, tau_sys, f_vals)
- `default_colormap`

---

## Connectivity Strategy

RMT connectivity parameters were extracted from `cRNN` into a separate `Connectivity` strategy hierarchy:

```
Connectivity < handle (abstract)
├── level_of_chaos
├── build(n, f, indegree, rng_seed)
├── get_params()
│
└── RMTConnectivity < Connectivity
    ├── mu_E_tilde, mu_I_tilde, sigma_E_tilde, sigma_I_tilde
    ├── E_W, zrs_mode, rescale_by_abscissa
    ├── Dependent: default_val, mu_se, mu_si, sigma_se, sigma_si, R
    └── Uses RMTMatrix internally
```

**Access pattern:**
```matlab
model = SRNNModel2();
model.connectivity.level_of_chaos = 2.0;
model.connectivity.E_W = 0.5;
model.connectivity.rescale_by_abscissa = true;
model.build();
model.connectivity.R    % theoretical spectral radius
```

---

## Setup Paths

Both `setup_paths.m` scripts now add `Matlab/shared/src/`:

```matlab
% SRNN/scripts/setup_paths.m adds:
%   SRNN/src/  +  shared/src/

% LNN/scripts/setup_paths.m adds:
%   LNN/src/  +  SRNN/src/  +  shared/src/
```

---

## Remaining Work

- **Phase 5:** Unified `ESN_reservoir` class (composition-based, wraps any cRNN subclass)
- **Phase 6:** Update example scripts, consolidate path management
- **Phase 7:** Comprehensive verification (memory capacity comparison, Lyapunov for both models)
