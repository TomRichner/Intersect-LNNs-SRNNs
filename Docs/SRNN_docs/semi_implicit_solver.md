# Semi-Implicit (Fused) ODE Solver for the SRNN

This document derives a semi-implicit ODE solver for the SRNN system, following the approach of Hasani et al. (2021) for Liquid Time-Constant Networks. The key idea: replace the state variables that appear **linearly** in each ODE with their next-time-step values, while keeping nonlinear terms at the current time step. This yields a closed-form update that is unconditionally stable with respect to the linear decay.

---

## 1. Continuous-Time SRNN Equations

The full SRNN system is:

$$
\frac{dx_i}{dt} = \frac{-x_i + u_i + \sum_{j=1}^{N} w_{ij}\, b_j\, r_j}{\tau_d} \tag{1}
$$

$$
r_i = \phi\!\left(x_i - a_{0_i} - c \sum_{k=1}^{K} a_{ik}\right) \tag{2}
$$

$$
\frac{da_{ik}}{dt} = \frac{-a_{ik} + r_i}{\tau_{a_k}} \tag{3}
$$

$$
\frac{db_i}{dt} = \frac{1 - b_i}{\tau_{rec}} - \frac{b_i\, r_i}{\tau_{rel}} \tag{4}
$$

---

## 2. Identifying Linear vs. Nonlinear Dependence

For each ODE, we classify how the state variable appears:

| Equation | Variable | Linear in self? | Nonlinear dependence |
|----------|----------|----------------|---------------------|
| (1) $\dot{x}_i$ | $x_i$ | Yes: $-x_i/\tau_d$ | $r_j$ depends on $x_j$ via $\phi(\cdot)$ |
| (3) $\dot{a}_{ik}$ | $a_{ik}$ | Yes: $-a_{ik}/\tau_{a_k}$ | $r_i$ depends on $x_i$ (separate variable) |
| (4) $\dot{b}_i$ | $b_i$ | Yes: $-(1/\tau_{rec} + r_i/\tau_{rel})\,b_i$ | $r_i$ depends on $x_i$ |

The semi-implicit strategy: for each equation, replace the **linearly-appearing self-variable** with its value at $t + \Delta t$, while evaluating everything else (including nonlinear terms and cross-variable dependencies) at time $t$.

---

## 3. Derivation of the Fused Updates

### 3.1 Dendritic Potential $x_i$

Starting from Eq. (1) in explicit Euler form and making $x_i$ implicit:

$$
x_i^{n+1} = x_i^n + \frac{\Delta t}{\tau_d}\left(-x_i^{n+1} + u_i + \sum_j w_{ij}\, b_j^n\, r_j^n\right)
$$

Collecting $x_i^{n+1}$ on the left:

$$
x_i^{n+1}\left(1 + \frac{\Delta t}{\tau_d}\right) = x_i^n + \frac{\Delta t}{\tau_d}\left(u_i + \sum_j w_{ij}\, b_j^n\, r_j^n\right)
$$

$$
\boxed{x_i^{n+1} = \frac{x_i^n + \frac{\Delta t}{\tau_d}\left(u_i + \sum_j w_{ij}\, b_j^n\, r_j^n\right)}{1 + \frac{\Delta t}{\tau_d}}} \tag{5}
$$

**Interpretation:** This is a weighted average between the current state $x_i^n$ and the "drive equilibrium" $u_i + \sum_j w_{ij} b_j r_j$, with mixing coefficient $\Delta t / \tau_d$. The denominator is always $> 1$, guaranteeing contraction of the leak.

### 3.2 Adaptation Variable $a_{ik}$

From Eq. (3), making $a_{ik}$ implicit:

$$
a_{ik}^{n+1} = a_{ik}^n + \frac{\Delta t}{\tau_{a_k}}\left(-a_{ik}^{n+1} + r_i^n\right)
$$

Solving:

$$
\boxed{a_{ik}^{n+1} = \frac{a_{ik}^n + \frac{\Delta t}{\tau_{a_k}}\, r_i^n}{1 + \frac{\Delta t}{\tau_{a_k}}}} \tag{6}
$$

**Interpretation:** The adaptation variable relaxes toward the current firing rate $r_i^n$ with mixing coefficient $\Delta t / \tau_{a_k}$. Since $\tau_{a_k}$ can be large (up to 10 s), the ratio $\Delta t / \tau_{a_k}$ may be very small, making this update nearly a no-op for slow timescales---which is physically correct.

### 3.3 STD Variable $b_i$

Eq. (4) is slightly more involved because $b_i$ appears in two terms. Rewrite the ODE as:

$$
\frac{db_i}{dt} = \frac{1}{\tau_{rec}} - b_i\left(\frac{1}{\tau_{rec}} + \frac{r_i}{\tau_{rel}}\right)
$$

Making $b_i$ implicit in the linear terms:

$$
b_i^{n+1} = b_i^n + \Delta t\left[\frac{1}{\tau_{rec}} - b_i^{n+1}\left(\frac{1}{\tau_{rec}} + \frac{r_i^n}{\tau_{rel}}\right)\right]
$$

Solving:

$$
\boxed{b_i^{n+1} = \frac{b_i^n + \frac{\Delta t}{\tau_{rec}}}{1 + \Delta t\left(\frac{1}{\tau_{rec}} + \frac{r_i^n}{\tau_{rel}}\right)}} \tag{7}
$$

**Interpretation:** The numerator drives $b_i$ toward 1 (full recovery). The denominator includes an activity-dependent term $r_i^n / \tau_{rel}$ that pulls $b_i$ toward 0 when the neuron is active. Since both terms in the denominator are non-negative, $b_i^{n+1} \in [0, 1]$ is guaranteed whenever $b_i^n \in [0, 1]$---the solver preserves the physical constraint without clamping.

---

## 4. Full Semi-Implicit Algorithm

Given: state $(x^n, a^n, b^n)$, input $u$, parameters $(\tau_d, \tau_{a_k}, \tau_{rec}, \tau_{rel}, c, a_0, W)$


> **For** $l = 1, \ldots, L$ (sub-steps):
>
> 1. Compute effective potential: $\quad x_{eff,i} = x_i^n - a_{0_i} - c\sum_k a_{ik}^n$
>
> 2. Compute firing rates: $\quad r_i = \phi(x_{eff,i})$
>
> 3. Compute synaptic output: $\quad s_j = b_j^n \cdot r_j$
>
> 4. Update $x$: $\quad x_i^{n+1} = \dfrac{x_i^n + \dfrac{\Delta t}{\tau_d}\!\left(u_i + \sum_j w_{ij}\, s_j\right)}{1 + \dfrac{\Delta t}{\tau_d}}$
>
> 5. Update $a$: $\quad a_{ik}^{n+1} = \dfrac{a_{ik}^n + \dfrac{\Delta t}{\tau_{a_k}}\, r_i}{1 + \dfrac{\Delta t}{\tau_{a_k}}}$
>
> 6. Update $b$: $\quad b_i^{n+1} = \dfrac{b_i^n + \dfrac{\Delta t}{\tau_{rec}}}{1 + \Delta t\!\left(\dfrac{1}{\tau_{rec}} + \dfrac{r_i}{\tau_{rel}}\right)}$
>
> 7. Set: $\quad x^n \leftarrow x^{n+1},\; a^n \leftarrow a^{n+1},\; b^n \leftarrow b^{n+1}$
>
> **End for**

In vectorized form, with $D = \text{diag}(1 + \Delta t / \tau_d)$:

$$
\mathbf{x}^{n+1} = D^{-1}\!\left(\mathbf{x}^n + \frac{\Delta t}{\tau_d}\left(\mathbf{u} + W(\mathbf{b}^n \odot \mathbf{r}^n)\right)\right)
$$

where $D^{-1}$ is trivially computed element-wise (no matrix inversion needed).

---

## 5. Stability Analysis

### 5.1 Unconditional Stability of the Leak

For the $x$ update (Eq. 5), the amplification factor of the leak term is:

$$
G_{leak} = \frac{1}{1 + \Delta t / \tau_d}
$$

Since $\Delta t > 0$ and $\tau_d > 0$, we have $0 < G_{leak} < 1$ for **any** step size $\Delta t$. The leak is unconditionally stable---it always contracts, never amplifies. Compare with explicit Euler where the leak amplification factor is $|1 - \Delta t/\tau_d|$, which exceeds 1 when $\Delta t > 2\tau_d$.

### 5.2 Conditional Stability of the Recurrent Drive

The recurrent term $\sum_j w_{ij} b_j r_j$ is evaluated explicitly. Linearizing around a fixed point, the recurrent contribution to the amplification factor involves:

$$
G_{rec} \sim \frac{\Delta t}{\tau_d} \cdot \rho(W \cdot \text{diag}(b \odot \phi'))
$$

where $\rho(\cdot)$ denotes the spectral radius and $\phi'$ is the activation derivative. For stability, we need:

$$
\frac{\Delta t}{\tau_d} \cdot \rho\!\left(W \cdot \text{diag}(b \odot \phi')\right) < 1 + \frac{\Delta t}{\tau_d}
$$

which simplifies to:

$$
\Delta t < \frac{\tau_d\left(1 + \Delta t / \tau_d\right)}{\rho(W \cdot \text{diag}(b \odot \phi'))}
$$

This is a **weaker** constraint than explicit Euler's $\Delta t < 2\tau_d / \rho(\cdot)$ because the denominator on the right includes the stabilizing $1 + \Delta t/\tau_d$ factor---the implicit treatment of the leak "absorbs" some of the recurrent instability.

### 5.3 Preservation of Physical Bounds

- **$b_i \in [0, 1]$**: Eq. (7) preserves this interval exactly. If $b_i^n \in [0, 1]$ and $r_i^n \geq 0$, then $b_i^{n+1} \in [0, 1]$.
- **$a_{ik} \geq 0$**: If $a_{ik}^n \geq 0$ and $r_i^n \geq 0$, then $a_{ik}^{n+1} \geq 0$.
- **$x_i$**: Unbounded in the continuous system, and the solver does not introduce artificial bounds. Stability of $x$ depends on the spectral properties of $W$ and the adaptation mechanisms.

### 5.4 Comparison with Explicit Euler

| Property | Explicit Euler | Semi-Implicit (Fused) |
|----------|---------------|----------------------|
| Leak stability | $\Delta t < 2\tau_d$ | Unconditional |
| Recurrent stability | $\Delta t < 2\tau_d / \rho(\cdot)$ | Relaxed constraint (Sec. 5.2) |
| $b_i \in [0,1]$ preserved | Not guaranteed | Guaranteed |
| Cost per step | 1 RHS evaluation | 1 RHS evaluation + element-wise division |
| Gradient structure | Standard | Bounded denominators may help with exploding gradients |

---

## 6. Practical Considerations for BPTT Training

1. **Larger step sizes.** The relaxed stability constraint allows $\Delta t$ to be 2--10$\times$ larger than explicit Euler for the same network, reducing the number of sub-steps $L$ needed per input time step.

2. **Gradient flow.** The division by $1 + \Delta t/\tau_d$ in the forward pass creates a multiplicative factor in the backward pass that naturally dampens gradient magnitude. This may help prevent exploding gradients during BPTT without requiring gradient clipping.

3. **Computational cost.** Each sub-step requires one extra element-wise division compared to explicit Euler. Since the dominant cost is the matrix-vector product $W \cdot (b \odot r)$, the overhead is negligible.

4. **Differentiability.** All operations (addition, division, $\phi$) are smooth and Zygote-compatible. No special handling is needed for automatic differentiation.
