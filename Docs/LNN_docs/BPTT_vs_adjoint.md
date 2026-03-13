---
title: "BPTT vs Adjoint Methods for Training LTC Networks"
author: "Design Notes"
date: "March 2026"
---

# BPTT vs Adjoint Methods for Training LTC Networks

## 1. Overview

When training LTC networks (or any Neural ODE) via gradient descent, there are two
fundamentally different strategies for computing $\partial L / \partial \theta$:

1. **BPTT (Backpropagation Through Time)** — "discretize then optimize"
2. **Continuous Adjoint** — "optimize then discretize"

Both produce gradients of the loss with respect to the model parameters, but they differ
in memory cost, numerical properties, and compatibility with different solver types.

---

## 2. BPTT: Discretize Then Optimize

### How It Works

1. **Forward pass**: Unroll the RNN/ODE step by step over $T$ time steps, storing every
   intermediate state $v_0, v_1, \ldots, v_T$.
2. **Backward pass**: Walk backwards through the stored computation graph using standard
   reverse-mode AD (Zygote in Julia, autograd/tape in TF/PyTorch).

For LTCODE1 with 6 sub-steps per timestep, the full forward graph is:

$$v_0 \xrightarrow{6\text{ fused steps}} v_1 \xrightarrow{6\text{ fused steps}} v_2 \xrightarrow{\cdots} v_T$$

Zygote records all intermediate values and replays them in reverse to compute gradients.

### Memory

$$\text{Memory} = O(N \times T \times K)$$

where $N$ = hidden dimension, $T$ = sequence length, $K$ = ODE sub-steps per timestep
(default $K = 6$).

### What the Original Python Code Does

Hasani's `ltc_model.py` uses BPTT exclusively. TensorFlow's `tf.nn.dynamic_rnn` unrolls
the LTC cell over time steps, and TF's tape-based AD differentiates through the unroll.
The fused semi-implicit solver's 6 sub-steps are part of the computation graph.

This is the "discretize then optimize" approach — the solver is fixed, and we
differentiate through it.

---

## 3. Continuous Adjoint: Optimize Then Discretize

### How It Works (Chen et al., NeurIPS 2018)

Instead of storing the forward trajectory, solve a **reverse-time adjoint ODE**:

1. **Forward pass**: Solve $x' = f(x, \theta, t)$ from $t_0 \to t_T$. Optionally store
   the trajectory, or store only $x(t_T)$.
2. **Backward pass**: Define the **adjoint state** $a(t) = \partial L / \partial x(t)$
   and solve the adjoint ODE backwards:

$$\frac{da}{dt} = -a(t)^\top \frac{\partial f}{\partial x}\bigg|_{x(t), \theta}$$

$$\frac{d\bar{\theta}}{dt} = -a(t)^\top \frac{\partial f}{\partial \theta}\bigg|_{x(t), \theta}$$

The parameter gradients $\bar{\theta}$ are accumulated during the reverse solve.

### Memory Variants in SciML

| Algorithm | Memory | How it gets $x(t)$ during backward | Stability |
|---|---|---|---|
| `BacksolveAdjoint` | $O(1)$ states | Re-solves ODE backward from $x(t_T)$ | **Unstable for dissipative systems** |
| `InterpolatingAdjoint` | $O(T)$ states | Interpolates stored forward trajectory | Stable |
| `QuadratureAdjoint` | $O(N)$ | Numerical quadrature | Stable but slowest |

### Julia / SciML Usage

```julia
using DiffEqFlux, SciMLSensitivity

neural_ode = NeuralODE(model, tspan, Tsit5();
    saveat = dt,
    sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()))

# or for O(1) memory:
neural_ode = NeuralODE(model, tspan, Tsit5();
    saveat = dt,
    sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP()))
```

---

## 4. Key Trade-offs

| Factor | BPTT | Adjoint |
|---|---|---|
| **Memory** | $O(N \cdot T \cdot K)$ — stores all states | $O(1)$ to $O(T)$ depending on variant |
| **Compute** | 2 passes (forward + backward) | 2 ODE solves (forward + adjoint) |
| **Solver coupling** | Differentiates through the fixed solver | Uses its own adaptive solver |
| **Gradient accuracy** | Exact for the chosen discretization | Approximate (depends on solver tolerance) |
| **Input handling** | Natural — feed $u_t$ at each step | Needs interpolant or `DiscreteCallback` |
| **Implementation** | Simple for-loop + AD | Requires NeuralODE / SciML infrastructure |
| **Debugging** | Easy — standard Julia code | Harder — solver internals, sensitivity choices |

---

## 5. Why BPTT Is Right for the Hasani Experiments

### 5.1 The Fused Solver Is Not a Continuous ODE

The semi-implicit solver in `ltc_model.py` (and `LTCODE1` in `ltc1.jl`) is an
algebraic fixed-point iteration, not a standard ODE integration method:

$$v_i \leftarrow \frac{C_{m,i}\,v_i + g_{\text{leak},i}\,v_{\text{leak},i} + \sum g_{ji}\,E_{ji}}{C_{m,i} + g_{\text{leak},i} + \sum g_{ji}}$$

There is no well-defined continuous trajectory to run an adjoint ODE over. To use the
adjoint method, you would need to switch to `ltc1_ode_rhs` with a real ODE solver
(Tsit5, etc.), which produces **different dynamics** than the fused solver.

This means adjoint-trained networks and BPTT-trained networks would learn different
solutions and cannot be directly compared to the Python reference results.

### 5.2 Piecewise-Constant Inputs Break Smooth Adjoints

The adjoint ODE assumes a smooth right-hand side $f(x, t)$. In the Hasani experiments,
the input $u(t)$ is **piecewise-constant** — it jumps at each discrete timestep. The
adjoint solver needs `DiscreteCallback`s at each input transition, and step-size
selection near discontinuities can cause accuracy issues.

BPTT handles this naturally: at each discrete step, the current input is simply passed
to the cell.

### 5.3 BacksolveAdjoint Is Unstable for LTCs

The $O(1)$ `BacksolveAdjoint` recomputes $x(t)$ by solving the ODE **backward in time**
from the final state. For dissipative systems like LTCs — which have leak conductance
terms that make the forward dynamics contracting — the backward-in-time dynamics are
**expanding** (numerically unstable). Small numerical errors grow exponentially.

The remaining options are:

- `InterpolatingAdjoint` — stable but $O(T)$ memory, same order as BPTT
- `QuadratureAdjoint` — stable and $O(N)$ memory, but the slowest option

Neither offers a compelling advantage over BPTT for short sequences.

### 5.4 These Sequences Are Short

All of the Hasani experiments have very short sequence lengths:

| Experiment | Seq Length ($T$) | Hidden ($N$) | Task Type |
|---|---|---|---|
| HAR | 16 | 32 | Classification (6 classes) |
| Gesture | 32 | 32 | Classification |
| Occupancy | 32 | 32 | Classification |
| Traffic | 32 | 32 | Regression |
| Ozone | 32 | 32 | Regression |
| Power | 32 | 32 | Regression |
| Person | 32 | 32 | Classification (7 classes) |
| SMNIST | 28 | 32 | Classification (10 digits) |
| Cheetah | 32 | 32 | Regression (17-dim) |

With $T \leq 32$, the BPTT memory cost is negligible: $32 \times 32 \times 6 \approx
6{,}144$ floats per sample — about 24 KB in Float32. There is no memory pressure to
justify the overhead of adjoint methods.

### 5.5 BPTT Is Faster for Short Sequences

The adjoint method has fixed overhead per solve:

- Forward ODE solve with adaptive stepping (step-size selection, error estimation)
- Reverse adjoint ODE solve (same overhead, plus Jacobian-vector products)
- Solver bookkeeping (interpolation tables, callback handling)

For $T = 16\text{--}32$, this overhead dominates. A simple Zygote-differentiated
for-loop is faster in wall-clock time.

---

## 6. When the Adjoint Method Wins

The adjoint becomes the right choice when:

| Condition | Why adjoint wins |
|---|---|
| $T > \sim\!500$ | BPTT memory becomes prohibitive; $O(1)$ or $O(N)$ matters |
| Continuous-time dynamics are the research focus | You *want* the adaptive solver's accuracy |
| Large $N$ (hidden dim) | $O(N \cdot T)$ vs $O(N)$ matters more |
| Long-horizon control / planning | Gradient quality through smooth integration |
| Irregular time sampling | ODE solvers handle non-uniform $\Delta t$ naturally |
| Solver-independent behavior | Testing whether results depend on discretization choice |

### Potential Future Uses in the Intersect Project

- **Lyapunov exponent analysis**: Using `ltc1_ode_rhs` with adaptive solvers to study
  trajectory stability over long time horizons. This requires continuous-time
  integration, not the fused solver.
- **Long time-series forecasting**: If training on sequences with $T > 500$, BPTT
  memory scales linearly while `QuadratureAdjoint` stays at $O(N)$.
- **Solver comparison**: Studying whether training outcomes change with Tsit5 vs RK4
  vs the fused solver. The adjoint approach makes solver choice explicit.

---

## 7. Practical Implementation Guide (Julia)

### Option A: BPTT with Manual Scan (Recommended for Hasani Experiments)

```julia
function forward_scan(ltc::LTCODE1, head::Dense, ps_ltc, ps_head, st, x_seq)
    # x_seq: (features, T) — one sample
    T = size(x_seq, 2)
    v = zeros(Float32, ltc.n)  # initial state

    for t in 1:T
        st_driven = merge(st, (input = x_seq[:, t],))
        v, st = ltc(v, ps_ltc, st_driven)  # 6 fused sub-steps
    end

    logits = head(v, ps_head, ...)  # Dense(N → n_classes)
    return logits
end

# Zygote differentiates through the for-loop and all 6×T fused steps
grads = Zygote.gradient(ps -> loss(forward_scan(..., ps, ...), y), ps)
```

**Pros**: Matches Python exactly. Simple. Fast for short sequences.

**Cons**: Memory scales with $T$. Not easily swappable between solvers.

### Option B: NeuralODE with Adjoint (For Future Continuous-Time Work)

```julia
function ltc1_driven_rhs(v, p, t; u_interp)
    inputs = u_interp(t)                       # piecewise-constant interpolant
    inputs_mapped = _map_inputs(inputs, p)
    return _f_prime(v, inputs_mapped, p)        # dv/dt
end

prob = ODEProblem(ltc1_driven_rhs, v0, tspan, ps)
sol = solve(prob, Tsit5(); saveat=dt,
    sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
```

**Pros**: Proper continuous-time integration. Choice of solver. $O(1)$ memory possible.

**Cons**: Need input interpolant. Different dynamics from fused solver. More complex
setup. Slower for short sequences.

---

## 8. Summary

| | BPTT (Option A) | Adjoint (Option B) |
|---|---|---|
| **Use for** | Hasani experiment ports, short sequences, faithful comparison to Python | Continuous-time analysis, long sequences, solver-independent research |
| **Memory** | $O(N \cdot T \cdot K)$ | $O(1)$ to $O(T)$ |
| **Speed (short $T$)** | **Faster** | Slower (solver overhead) |
| **Speed (long $T$)** | Slower (graph size) | **Faster** |
| **Matches Python** | ✅ Yes | ❌ Different dynamics |
| **Julia infrastructure** | Zygote only | DiffEqFlux + SciMLSensitivity |
| **Input handling** | Natural (per-step) | Needs interpolant |
| **Stability** | Stable | Backsolve can diverge for LTCs |

**Recommendation**: Start with BPTT (Option A) for porting the Hasani experiments to
Julia. The sequences are short ($T \leq 32$), memory is not a concern, and it
faithfully reproduces the Python training methodology. Reserve the adjoint approach for
future work on continuous-time dynamics, long-horizon tasks, or solver comparison
studies.

---

## References

1. Chen, R.T.Q., Rubanova, Y., Bettencourt, J., Duvenaud, D. "Neural Ordinary
   Differential Equations." *NeurIPS*, 2018.
2. Hasani, R., Lechner, M., Amini, A., Rus, D., Grosu, R. "Liquid Time-constant
   Networks." *AAAI*, 2021.
3. Rackauckas, C., et al. "Universal Differential Equations for Scientific Machine
   Learning." *arXiv:2001.04385*, 2020.
4. Kidger, P. "On Neural Differential Equations." *PhD Thesis, Oxford*, 2022.
   — Excellent treatment of discretize-then-optimize vs optimize-then-discretize.
5. SciML Documentation: Sensitivity Analysis.
   https://docs.sciml.ai/SciMLSensitivity/stable/
