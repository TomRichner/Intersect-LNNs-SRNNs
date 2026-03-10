# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements and compares two neural network reservoir architectures in MATLAB:
- **SRNN** (Spiking Rate Neural Network): A rate-based E/I network with spike-frequency adaptation (SFA) and short-term synaptic depression (STD)
- **LNN** (Liquid Time-Constant Network): Implements Hasani et al. (2021) LTC dynamics with input-dependent time constants

The primary application is **Echo State Network (ESN) memory capacity measurement**: driving reservoirs with random input and training linear readouts to reconstruct delayed versions of that input.

## Running Scripts

All scripts require running `setup_paths()` first to add `src/` to the MATLAB path.

**SRNN quick test:**
```matlab
cd Matlab/SRNN/scripts
setup_paths()
run test_SRNN2_defaults.m        % build/run/plot with SFA and STD
run example_memory_capacity.m    % MC comparison: baseline vs SFA vs SFA+STD
```

**LNN quick test:**
```matlab
cd Matlab/LNN/scripts
setup_paths()
run test_LNN.m                         % basic LNN build/run/plot
run example_memory_capacity_LNN.m      % MC comparison across spectral radii
```

Note: `Matlab/LNN/scripts/setup_paths.m` adds both `LNN/src/` and `SRNN/src/` (needed for `RMTMatrix.m`).

## Architecture

### Repository structure
```
Matlab/
  SRNN/
    src/
      SRNNModel2.m           % Core SRNN class (base)
      SRNN_ESN_reservoir.m   % Extends SRNNModel2 for ESN/MC measurement
      RMTMatrix.m            % Random Matrix Theory connectivity (shared utility)
    scripts/
      setup_paths.m
      test_SRNN2_defaults.m
      example_memory_capacity.m
  LNN/
    src/
      LNN.m                  % Core LTC class (base)
      LNN_ESN_reservoir.m    % Extends LNN for ESN/MC measurement
    scripts/
      setup_paths.m
      test_LNN.m
      example_memory_capacity_LNN.m
Docs/
  SRNN_docs/   % Mathematical notes, parameter tables, code structure docs
  LNN_docs/    % Papers and mathematical notes for LTC networks
```

### Class hierarchy

**SRNN branch:**
- `SRNNModel2 < handle` — builds RMT connectivity matrix (`W`), generates external input via random step functions, integrates `dx/dt = (-x + W*r + u)/tau_d` with optional SFA (`a`) and STD (`b`) variables, computes Lyapunov exponents (Benettin or QR methods), and plots multi-panel time series. All static helper functions (Jacobian, plotting, activation functions) are internalized as static methods.
- `SRNN_ESN_reservoir < SRNNModel2` — overrides `build_stimulus()` to generate a scalar input `u_scalar` (white/bandlimited/1/f noise), maps it through sparse input weights `W_in`, then runs `run_memory_capacity()` which integrates the reservoir, collects firing rates, trains ridge-regression readouts for delays 1..d_max, and returns R²_d and total MC.

**LNN branch:**
- `LNN < handle` — implements LTC ODE: `dx/dt = -(1/tau + |f|).*x + f.*A` where `f = activation(W_r*x + W_in*I + mu)`. Supports `ode` solver mode (ode45/ode15s) and `fused` semi-implicit Euler mode. `W_r` is constructed via `RMTMatrix`. All activation functions internalized as static methods.
- `LNN_ESN_reservoir < LNN` — mirrors `SRNN_ESN_reservoir` for LTC dynamics, with extra `readout_mode` option ('state' uses `x`, 'nonlinearity' uses `f`).

### Shared utility
- `RMTMatrix.m` (in `SRNN/src/`) — used by both SRNN and LNN to generate random connectivity matrices with E/I structure, spectral radius control, and optional ZRS (Zero-Row-Sum) normalization.

### State vector packing (SRNN)
`S = [a_E(:); a_I(:); b_E(:); b_I(:); x(:)]`
Total size: `n_E*n_a_E + n_I*n_a_I + n_E*n_b_E + n_I*n_b_I + n`

### Key parameters
- `level_of_chaos` — spectral radius scaling; 1.0 = edge of chaos, >1 = chaotic
- `n_a_E` / `n_b_E` — enable SFA (adaptation) / STD (depression) for E neurons
- `tau_d` — dendritic time constant (SRNN); `tau` vector — per-neuron time constants (LNN)
- `T_wash`, `T_train`, `T_test`, `d_max`, `eta` — ESN memory capacity protocol parameters

### Typical workflow
```matlab
esn = SRNN_ESN_reservoir('n', 300, 'n_a_E', 3, 'n_b_E', 1, 'level_of_chaos', 1.0);
esn.build();
[MC, R2_d, results] = esn.run_memory_capacity();
esn.plot_memory_capacity();
esn.plot_esn_timeseries([1, 10, 50]);
```

Use `SRNN_ESN_reservoir.verify_shared_build(esn_array, {'n_a_E','n_b_E'}, {'W','W_in','u_scalar'})` to assert that comparison conditions share identical network structure.

### Data
Saved results go to `Matlab/SRNN/data/memory_capacity/` or `Matlab/LNN/data/memory_capacity/` as `.mat` files.
