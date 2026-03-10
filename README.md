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

### Shared Infrastructure

Both models share:
- **`RMTMatrix.m`** — Random Matrix Theory connectivity generator with E/I structure, sparsity, and spectral radius control
- **ESN reservoir subclasses** — `SRNN_ESN_reservoir` and `LNN_ESN_reservoir` extend the base models with memory capacity measurement protocols (washout → train → test, ridge regression readouts)

---

## Current Results

### Echo State Network Memory Capacity

Both architectures are evaluated as reservoirs using a standard memory capacity (MC) protocol:

1. Drive the reservoir with a scalar random input `u(t)`
2. Train linear readouts via ridge regression to reconstruct delayed versions `u(t-d)`
3. Compute R²_d for each delay and sum to obtain total MC

**SRNN comparison** ([`example_memory_capacity.m`](Matlab/SRNN/scripts/example_memory_capacity.m)):
Compares three adaptation conditions — Baseline (no adaptation), SFA only, and SFA + STD — demonstrating that biophysical adaptation mechanisms extend memory capacity beyond the edge-of-chaos baseline.

**LNN comparison** ([`example_memory_capacity_LNN.m`](Matlab/LNN/scripts/example_memory_capacity_LNN.m)):
Compares the LTC reservoir across spectral radius settings (R = 0.5, 1.0, 1.5), examining how the input-dependent time constants of the LTC ODE interact with reservoir stability.

---

## Repository Structure

```
Intersect-LNNs-SRNNs/
├── Matlab/
│   ├── SRNN/
│   │   ├── src/
│   │   │   ├── SRNNModel2.m            # Core SRNN class (E/I network with SFA + STD)
│   │   │   ├── SRNN_ESN_reservoir.m    # ESN subclass for memory capacity measurement
│   │   │   └── RMTMatrix.m             # RMT connectivity generator (shared)
│   │   ├── scripts/
│   │   │   ├── setup_paths.m
│   │   │   ├── test_SRNN2_defaults.m   # Quick build/run/plot test
│   │   │   └── example_memory_capacity.m
│   │   ├── data/
│   │   └── figs/
│   │
│   └── LNN/
│       ├── src/
│       │   ├── LNN.m                   # Core LTC class (Hasani et al. 2021)
│       │   └── LNN_ESN_reservoir.m     # ESN subclass for memory capacity measurement
│       ├── scripts/
│       │   ├── setup_paths.m
│       │   ├── test_LNN.m              # Quick build/run/plot test
│       │   └── example_memory_capacity_LNN.m
│       ├── data/
│       └── figs/
│
├── Docs/
│   ├── SRNN_docs/                      # SRNN equations, parameter tables, code structure
│   └── LNN_docs/                       # LTC papers, mathematical notes, LFM 2.5 notes
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

% Memory capacity comparison (Baseline vs SFA vs SFA+STD)
run example_memory_capacity.m
```

### LNN

```matlab
cd Matlab/LNN/scripts
setup_paths()

% Quick test
run test_LNN.m

% Memory capacity comparison across spectral radii
run example_memory_capacity_LNN.m
```

> **Note:** `Matlab/LNN/scripts/setup_paths.m` adds both `LNN/src/` and `SRNN/src/` to the path, since the LNN uses `RMTMatrix.m` from the SRNN source directory.

---

## Future Directions

### 1. Stability Analysis — Parameter Space Sweeps

Compare the stability of SRNN and LNN networks with random connectivity under step-function stimulus perturbations. This will answer: *which architecture is more robust to parameter variation, and how do their stability boundaries differ?*

**Plan:**

- **Port the FractionalReservoir analysis framework** into this repository. The [`ParamSpaceAnalysis2`](https://github.com/TomRichner/FractionalReservoir/blob/main/src/ParamSpaceAnalysis2.m) class already supports multi-dimensional grid sweeps, batch execution with random ordering, and Lyapunov exponent computation. The [`Fig_2_fraction_excitatory_analysis.m`](https://github.com/TomRichner/FractionalReservoir/blob/main/scripts/Fig_2_fraction_excitatory_analysis.m) script demonstrates a sweep over the fraction of excitatory neurons (`f`) across adaptation conditions, computing Largest Lyapunov Exponent (LLE) and firing rate distributions.

- **Refactor for model-agnostic use.** Both `LNN.m` and `SRNNModel2.m` share a common interface (`build()` → `run()` → `plot()`), which makes it feasible to create:
  - A **common ESN class** that both models can plug into
  - A **common `ParamSpaceAnalysis` class** that works with either model, sweeping over shared parameters (e.g., `f`, `level_of_chaos`, `n`, `indegree`) and model-specific parameters (e.g., `tau_d`/`c_E` for SRNN, `tau`/`A` for LNN)

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
