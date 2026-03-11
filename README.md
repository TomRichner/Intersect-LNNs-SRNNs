# Intersect-LNNs-SRNNs

**Comparing Stable Recurrent Neural Networks (SRNNs) and Liquid Neural Networks (LNNs) as reservoir computing architectures and trainable Neural ODE models.**

This repository explores the intersection of two biologically-inspired recurrent network models — SRNNs with spike-frequency adaptation and short-term synaptic depression, and Liquid Time-Constant (LTC) Networks with input-dependent time constants — comparing their performance as echo state network (ESN) reservoirs in MATLAB and as trainable differentiable ODE layers in Julia.

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
  - **`Activation`** (base) / **`TanhActivation`**, **`SigmoidActivation`**, **`PiecewiseSigmoid`**, **`ReLUActivation`** — activation function strategies
- **`parse_name_value_pairs()`** — generic constructor parsing that auto-forwards unknown params to connectivity/stimulus/activation strategies
- **`W_in`** input weight matrix — both models use `W_in` explicitly in their dynamics (`dx/dt` includes `W_in * u_raw`)

**Unified ESN:** **`ESN_reservoir`** wraps any `cRNN` subclass via composition for memory capacity measurement (washout → train → test, ridge regression, Lyapunov). Replaces the deprecated `SRNN_ESN_reservoir` and `LNN_ESN_reservoir`.

### Parameter Space Analysis (`ParamSpace`)

**`ParamSpace`** is a model-agnostic class for multi-dimensional grid sweeps over any `cRNN` subclass. It uses a pluggable architecture:

- **`model_factory`** — function handle that creates any cRNN subclass: `@(args) SRNNModel2(args{:})`
- **`model_args`** — cell array of base name-value pairs passed to the factory
- **`metric_extractor`** — function handle for extracting metrics from a completed run (LLE, firing rate, etc.)
- **Grid parameters** — any model/strategy property can be swept (`f`, `level_of_chaos`, `n_a_E`, `n_b_E`, etc.)

Features include batched `parfor` execution, checkpoint/resume capability, randomized execution order, and vector-valued parameter sweeps.

**Implementation:** [`Matlab/shared/src/ParamSpace.m`](Matlab/shared/src/ParamSpace.m)

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
│   │       ├── ParamSpace.m            # Model-agnostic parameter space analysis
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
│   │       ├── PiecewiseSigmoid.m      # Piecewise sigmoid (for SRNN)
│   │       └── ReLUActivation.m        # ReLU activation
│   │
│   ├── SRNN/
│   │   ├── src/
│   │   │   ├── SRNNModel2.m            # SRNN class (E/I network with SFA + STD)
│   │   │   └── SRNN_ESN_reservoir.m    # (Deprecated) use ESN_reservoir instead
│   │   └── scripts/
│   │       ├── setup_paths.m
│   │       ├── test_SRNN2_defaults.m    # Quick build/run/plot test
│   │       ├── test_ESN_SRNN.m          # ESN memory capacity (Baseline vs SFA vs STD)
│   │       └── Fig_2_fraction_excitatory_analysis.m  # f-sweep with adaptation variants
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
├── JuliaLang/                          # Julia trainable Neural ODE implementations
│   ├── Project.toml                    # Julia 1.12 + DiffEqFlux/Lux/SciMLSensitivity
│   ├── src/
│   │   ├── connectivity.jl             # RMT E/I weight matrix generation
│   │   ├── activations.jl              # Piecewise sigmoid (AD-safe)
│   │   └── models/
│   │       ├── ltc.jl                  # LTC Lux layer (W, W_in, μ, τ, A trainable)
│   │       └── srnn.jl                 # SRNN Lux layer (SFA + STD, augmented state)
│   ├── scripts/                        # Training scripts (planned)
│   └── test/                           # Tests (planned)
│
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

### ParamSpace (parameter sweeps)

```matlab
cd Matlab/SRNN/scripts
setup_paths()

% Create analysis object
ps = ParamSpace('n_levels', 5, 'note', 'my_sweep');
ps.model_factory = @(args) SRNNModel2(args{:});
ps.metric_extractor = @ParamSpace.srnn_metric_extractor;

% Base model configuration
ps.model_args = {'n', 300, 'indegree', 100, 'T_range', [-15, 45], ...
    'lya_method', 'benettin'};

% Define grid (everything is an arg — no "conditions")
ps.add_grid_parameter('f', [0.4, 0.6]);       % fraction excitatory
ps.add_grid_parameter('n_a_E', [0, 3]);        % SFA on/off
ps.add_grid_parameter('n_b_E', [0, 1]);        % STD on/off
ps.add_grid_parameter('reps', 1:3);            % repetitions

ps.run();                                       % batched parfor w/ checkpoints
ps.plot('metric', 'LLE');                       % histogram
ps.plot_sensitivity('metric', 'LLE');           % heatmap vs swept param

% Reload results later
ps2 = ParamSpace();
ps2.load_results(ps.output_dir);
```

> **Note:** `setup_paths.m` adds both `<model>/src/` and `shared/src/` to the path.

---

## Future Directions

### 1. Stability Analysis — Parameter Space Sweeps

~~Compare the stability of SRNN and LNN networks with random connectivity under step-function stimulus perturbations.~~

✅ **Done.** The `ParamSpace` class ([`Matlab/shared/src/ParamSpace.m`](Matlab/shared/src/ParamSpace.m)) provides a model-agnostic framework for multi-dimensional grid sweeps over any `cRNN` subclass. Key capabilities:

- **Pluggable model factory** — works with `SRNNModel2`, `LNN`, or any future `cRNN` subclass
- **Pluggable metric extraction** — built-in extractors for LLE (`default_metric_extractor`) and SRNN-specific metrics like firing rate and synaptic output (`srnn_metric_extractor`)
- **Batched `parfor`** execution with checkpoint/resume, randomized order for representative early-stopping
- **Any model parameter as a grid dimension** — the old "conditions" concept (e.g., no_adapt vs SFA vs STD) is replaced by simply adding those parameters to the grid (e.g., `n_a_E ∈ {0, 3}`, `n_b_E ∈ {0, 1}`)

**Example analysis script:** [`Fig_2_fraction_excitatory_analysis.m`](Matlab/SRNN/scripts/Fig_2_fraction_excitatory_analysis.m) — sweeps `f × n_a_E × n_b_E × reps` for SRNN stability analysis.

**Next steps:**
- Run LNN parameter space sweeps (analogous scripts using `@(args) LNN(args{:})`) to compare stability boundaries
- Build external plotting scripts for cross-model comparison figures

### 2. Julia Implementation with Differentiable ODE Solvers

🚧 **In Progress.** Both architectures are being reimplemented in **Julia 1.12** using **DiffEqFlux.jl + Lux.jl** to enable gradient-based training via adjoint sensitivity methods. This opens direct comparison of *learning dynamics* across architectures.

**Status:**

| Component | Status | Notes |
|---|---|---|
| LTC Lux layer (`ltc.jl`) | ✅ Done | W, W_in, μ, τ, A all trainable; Zygote gradients verified |
| SRNN Lux layer (`srnn.jl`) | ✅ Done | Augmented state (SFA+STD); doubles as Hopfield baseline when n_a=0, n_b=0 |
| Connectivity (`connectivity.jl`) | ✅ Done | RMT E/I matrix generation; provides W initialization |
| Activations (`activations.jl`) | ✅ Done | Branchless piecewise sigmoid (AD-safe) |
| Training pipeline | ⬜ Planned | Optimization.jl + adjoint sensitivity |
| MATLAB cross-check | ⬜ Planned | Compare Julia and MATLAB dynamics on identical W |

**Key design decisions:**
- **Lux.jl (not Flux)** — required for explicit parameter handling in adjoint methods
- **All parameters trainable** — W, time constants, adaptation scaling, etc. (not frozen ESN-style)
- **RMT initialization** — structured E/I connectivity as starting point for training
- **`softplus`-wrapped time constants** — positivity constraint without hard clipping

**Next steps:**
- Training scripts with `NeuralODE` + `InterpolatingAdjoint`/`BacksolveAdjoint`
- Benchmark on sequence tasks (e.g., sequential MNIST)
- Compare learning rates and convergence across SRNN, LTC, and Hopfield baseline

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
