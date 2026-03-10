# Intersect-LNNs-SRNNs

**Comparing Stable Recurrent Neural Networks (SRNNs) and Liquid Neural Networks (LNNs) as reservoir computing architectures.**

This repository explores the intersection of two biologically-inspired recurrent network models — SRNNs with spike-frequency adaptation and short-term synaptic depression, and Liquid Time-Constant (LTC) Networks with input-dependent time constants — comparing their performance as echo state network (ESN) reservoirs and, eventually, as trainable sequence models.

---

## Overview

### Stable Recurrent Neural Networks (SRNN)

The SRNN is a rate-based excitatory/inhibitory (E/I) network with two key biological stabilization mechanisms:

- **Spike-Frequency Adaptation (SFA):** Multi-timescale negative feedback that reduces excitatory firing over sustained activity
- **Short-Term Synaptic Depression (STD):** Activity-dependent reduction in effective synaptic strength

The core dynamics follow:

```
dx/dt = (-x + W*r + u) / tau_d
```

with auxiliary adaptation (`a`) and depression (`b`) variables. Connectivity matrices (`W`) are constructed via Random Matrix Theory (RMT) with E/I structure and spectral radius control.

**Implementation:** [`Matlab/SRNN/src/SRNNModel2.m`](Matlab/SRNN/src/SRNNModel2.m)

### Liquid Neural Networks (LNN)

The LNN implements the Liquid Time-Constant (LTC) ODE from [Hasani et al. (2021)](Docs/LNN_docs/Hasani%20et%20al.%20-%202021%20-%20Liquid%20Time-constant%20Networks.pdf):

```
dx/dt = -(1/tau + |f|) .* x + f .* A
```

where `f = activation(W_r * x + W_in * I(t) + mu)` is a nonlinearity that modulates both the effective time constant and the attractor state. This gives LNNs input-dependent dynamics — the network's temporal behavior adapts to the driving signal.

**Implementation:** [`Matlab/LNN/src/LNN.m`](Matlab/LNN/src/LNN.m)

### Shared Infrastructure (`Matlab/shared/src/`)

Both models inherit from a common **`cRNN`** abstract base class that provides:
- **`build()` → `run()` → `plot()`** lifecycle with ODE integration, decimation, and Lyapunov computation
- **Strategy pattern** for swappable components:
  - **`Connectivity`** (base) / **`RMTConnectivity`** — Random Matrix Theory connectivity with E/I structure and spectral radius control, backed by **`RMTMatrix.m`**
  - **`Stimulus`** (base) / **`StepStimulus`**, **`SinusoidalStimulus`**, **`ESNStimulus`** — input generation strategies
  - **`Activation`** (base) / **`TanhActivation`**, **`SigmoidActivation`**, **`PiecewiseSigmoid`** — activation function strategies
- **`parse_name_value_pairs()`** — generic constructor parsing that auto-forwards unknown params to connectivity/stimulus/activation strategies
- **`W_in`** input weight matrix — both models use `W_in` explicitly in their dynamics (`dx/dt` includes `W_in * u_raw`)

**Unified ESN:** **`ESN_reservoir`** wraps any `cRNN` subclass via composition for memory capacity measurement (washout → train → test, ridge regression, Lyapunov). Replaces the deprecated `SRNN_ESN_reservoir` and `LNN_ESN_reservoir`.

---

## Current Results

### Echo State Network Memory Capacity

Both architectures are evaluated as reservoirs using a standard memory capacity (MC) protocol:

1. Drive the reservoir with a scalar random input `u(t)`
2. Train linear readouts via ridge regression to reconstruct delayed versions `u(t-d)`
3. Compute R²_d for each delay and sum to obtain total MC

**SRNN comparison** ([`test_ESN_SRNN.m`](Matlab/SRNN/scripts/test_ESN_SRNN.m)):
Compares three adaptation conditions — Baseline (no adaptation), SFA only, and SFA + STD — demonstrating that biophysical adaptation mechanisms extend memory capacity beyond the edge-of-chaos baseline. Includes Lyapunov exponent computation.

**LNN comparison** ([`test_ESN_LNN.m`](Matlab/LNN/scripts/test_ESN_LNN.m)):
Compares the LTC reservoir across spectral radius settings (R = 0.5, 1.0, 1.5), examining how the input-dependent time constants of the LTC ODE interact with reservoir stability. Includes Lyapunov exponent computation.

---

## Repository Structure

```
Intersect-LNNs-SRNNs/
├── Matlab/
│   ├── shared/
│   │   └── src/
│   │       ├── cRNN.m                  # Abstract base class (build/run/plot lifecycle)
│   │       ├── ESN_reservoir.m         # Unified ESN wrapper (composition-based)
│   │       ├── ESNStimulus.m           # ESN scalar input + sparse W_in generation
│   │       ├── Connectivity.m          # Base connectivity strategy
│   │       ├── RMTConnectivity.m       # RMT connectivity (spectral radius control)
│   │       ├── RMTMatrix.m             # Low-level RMT matrix generator
│   │       ├── Stimulus.m              # Base stimulus strategy
│   │       ├── StepStimulus.m           # Step-function stimulus (for SRNN)
│   │       ├── SinusoidalStimulus.m     # Sinusoidal stimulus (for LNN)
│   │       ├── Activation.m            # Base activation strategy
│   │       ├── TanhActivation.m        # tanh activation
│   │       ├── SigmoidActivation.m     # sigmoid activation
│   │       └── PiecewiseSigmoid.m      # Piecewise sigmoid (for SRNN)
│   │
│   ├── SRNN/
│   │   ├── src/
│   │   │   ├── SRNNModel2.m            # SRNN class (E/I network with SFA + STD)
│   │   │   └── SRNN_ESN_reservoir.m    # (Deprecated) use ESN_reservoir instead
│   │   └── scripts/
│   │       ├── setup_paths.m
│   │       ├── test_SRNN2_defaults.m    # Quick build/run/plot test
│   │       └── test_ESN_SRNN.m          # ESN memory capacity (Baseline vs SFA vs STD)
│   │
│   └── LNN/
│       ├── src/
│       │   ├── LNN.m                   # LTC class (Hasani et al. 2021)
│       │   └── LNN_ESN_reservoir.m     # (Deprecated) use ESN_reservoir instead
│       └── scripts/
│           ├── setup_paths.m
│           ├── test_LNN.m              # Quick build/run/plot test
│           └── test_ESN_LNN.m          # ESN memory capacity (R=0.5, 1.0, 1.5)
│
├── Docs/
│   ├── SRNN_docs/                      # SRNN equations, parameter tables, code structure
│   ├── LNN_docs/                       # LTC papers, mathematical notes, LFM 2.5 notes
│   └── cRNN_base_class_refactor.md     # Refactoring notes for SRNN/LNN unification
│
├── JuliaLang/                          # (Planned) Julia implementations
└── README.md
```

---

## Quick Start

### SRNN

```matlab
cd Matlab/SRNN/scripts
setup_paths()

% Quick test (build → run → plot)
run test_SRNN2_defaults.m

% ESN memory capacity (Baseline vs SFA vs SFA+STD)
run test_ESN_SRNN.m
```

### LNN

```matlab
cd Matlab/LNN/scripts
setup_paths()

% Quick test
run test_LNN.m

% ESN memory capacity (R=0.5, 1.0, 1.5)
run test_ESN_LNN.m
```

### ESN_reservoir (unified API)

```matlab
% Composition-based: wrap any cRNN subclass
model = SRNNModel2('n', 300, 'level_of_chaos', 1.0);
esn = ESN_reservoir(model, 'T_wash', 2000, 'T_train', 5000, 'T_test', 5000, 'd_max', 600);
esn.build();
[MC, R2_d] = esn.run_memory_capacity();
esn.plot_memory_capacity();
esn.plot_esn_timeseries([1, 50, 100, 200]);
esn.model.plot();  % Model-specific internal dynamics
```

> **Note:** `setup_paths.m` adds both `<model>/src/` and `shared/src/` to the path.

---

## Future Directions

### 1. Stability Analysis — Parameter Space Sweeps

Compare the stability of SRNN and LNN networks with random connectivity under step-function stimulus perturbations. This will answer: *which architecture is more robust to parameter variation, and how do their stability boundaries differ?*

**Plan:**

- **Port the FractionalReservoir analysis framework** into this repository. The [`ParamSpaceAnalysis2`](https://github.com/TomRichner/FractionalReservoir/blob/main/src/ParamSpaceAnalysis2.m) class already supports multi-dimensional grid sweeps, batch execution with random ordering, and Lyapunov exponent computation. The [`Fig_2_fraction_excitatory_analysis.m`](https://github.com/TomRichner/FractionalReservoir/blob/main/scripts/Fig_2_fraction_excitatory_analysis.m) script demonstrates a sweep over the fraction of excitatory neurons (`f`) across adaptation conditions, computing Largest Lyapunov Exponent (LLE) and firing rate distributions.

- **~~Refactor for model-agnostic use.~~** ✅ **Done.** Both models now inherit from `cRNN` with a shared strategy pattern and unified `ESN_reservoir`. Next step: create a **common `ParamSpaceAnalysis` class** that works with either model, sweeping over shared parameters (e.g., `f`, `level_of_chaos`, `n`, `indegree`) and model-specific parameters (e.g., `tau_d`/`c_E` for SRNN, `tau`/`A` for LNN)

- **Key metrics to compare:** Largest Lyapunov Exponent (LLE), firing rate distributions, sensitivity to the fraction excitatory (`f`), and response to step-function stimulus changes

### 2. Julia Implementation with Differentiable ODE Solvers

Reimplement both architectures in **Julia** using **DiffEqFlux.jl + Lux.jl** to enable gradient-based training via backpropagation through time (BPTT) or interpolated adjoint methods. This opens up direct comparison of *learning dynamics* across architectures.

**Plan:**

- Implement three model variants as Neural ODE layers:
  1. **Hopfield network** (vanilla RNN, no adaptation) — baseline
  2. **SRNN** (with spike-frequency adaptation) — tests whether biological stabilization mechanisms accelerate learning
  3. **LNN / LTC** (Liquid Time-Constant) — tests whether input-dependent time constants improve gradient flow

- **Compare learning rates and convergence** on sequence tasks (e.g., sequential MNIST, time series forecasting)

- **Leverage Julia's LTC ecosystem.** A [reference Pluto notebook](../LTC_julia_reference_files/LTC_MNIST_strips.jl) (from JuliaHub) demonstrates LTC cells trained on sequential MNIST using Flux.jl with a fused semi-implicit Euler solver. This provides a starting point for the LTC implementation, to be adapted to the DiffEqFlux + Lux stack for adjoint sensitivity support.

- **Framework choice:** DiffEqFlux.jl provides `NeuralODE` wrappers with automatic differentiation through ODE solvers; Lux.jl provides a stateless neural network layer API compatible with explicit parameter handling required by adjoint methods.

---

## Documentation

| Document                                                                                   | Description                                                 |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| [`Docs/SRNN_docs/Code_Structure.md`](Docs/SRNN_docs/Code_Structure.md)                     | Detailed SRNN class hierarchy and code walkthrough          |
| [`Docs/SRNN_docs/parameter_table.md`](Docs/SRNN_docs/parameter_table.md)                   | Complete SRNN parameter reference                           |
| [`Docs/SRNN_docs/Memory_capacity_protocol.md`](Docs/SRNN_docs/Memory_capacity_protocol.md) | ESN memory capacity measurement protocol                    |
| [`Docs/LNN_docs/LNN_Mathematical_Notes.md`](Docs/LNN_docs/LNN_Mathematical_Notes.md)       | LTC ODE derivations and mathematical notes                  |
| [`Docs/LNN_docs/LFM2p5.md`](Docs/LNN_docs/LFM2p5.md)                                       | Notes on the Liquid Foundation Model (LFM) 2.5 architecture |

### Key References

- Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021). *Liquid Time-constant Networks.* AAAI.
- Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2018). *Liquid Time-constant Recurrent Neural Networks as Universal Approximators.*

---

## License

MIT License — see [LICENSE](LICENSE) for details.
