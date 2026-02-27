# Hasani et al. (2018) --- Liquid Time-Constant Recurrent Neural Networks as Universal Approximators

**Full Citation:** Ramin M. Hasani, Mathias Lechner, Alexander Amini, Daniela Rus, and Radu Grosu. "Liquid Time-constant Recurrent Neural Networks as Universal Approximators." arXiv:1811.00321, November 2018.

**Paper Focus:** This paper formalizes liquid time-constant (LTC) RNNs from biophysical first principles, provides the universal approximation proof, and derives bounds on the time-constant and neuronal states. It is the foundational theoretical paper for LTC networks.

---

## 1. Context and Motivation

Standard continuous-time RNNs (CTRNNs) use a fixed time-constant for each neuron and constant synaptic weights. This paper proposes a new CTRNN model inspired by the nervous system of small species such as *C. elegans*, *Ascaris*, and Leech, where synapses are **nonlinear sigmoidal functions** that model the biophysics of synaptic interactions. As a result, the postsynaptic neuron's state is defined by incoming presynaptic nonlinearities, which gives rise to a **varying time-constant** for each neuron and strengthens individual neurons' expressivity.

---

## 2. LTC RNN Neuron Model

### Equation 1 --- Membrane Integrator ODE

The dynamics of a hidden or output neuron $i$, with membrane potential $V_i(t)$, are modeled as a membrane integrator (Koch and Segev, 1998):

$$C_{m_i} \frac{dV_i}{dt} = G_{Leak_i}\Big(V_{Leak_i} - V_i(t)\Big) + \sum_{j=1}^{n} I_{in}^{(ij)} \tag{1}$$

This is the fundamental single-neuron ODE. The left-hand side represents the capacitive current --- charge accumulation on the neuronal membrane. The right-hand side consists of two terms:

- **Leak current:** $G_{Leak_i}(V_{Leak_i} - V_i(t))$ --- a passive current that drives the membrane potential toward its resting value $V_{Leak_i}$. When $V_i$ is above rest, this current is negative (pulling it down); when below rest, it is positive (pulling it up).
- **Synaptic input current:** $\sum_{j=1}^{n} I_{in}^{(ij)}$ --- the total current arriving from all $n$ presynaptic neurons. This includes both chemical synapses and gap junctions.

**Parameters and variables:**

| Symbol | Description |
|--------|-------------|
| $V_i(t)$ | Membrane potential (hidden state) of neuron $i$ at time $t$ |
| $C_{m_i}$ | Membrane capacitance of neuron $i$. Controls how quickly voltage changes in response to current. Larger values mean slower dynamics. |
| $G_{Leak_i}$ | Leak conductance of neuron $i$. Determines the rate of return to rest when no input is present. |
| $V_{Leak_i}$ | Leak (resting) potential of neuron $i$. The equilibrium voltage the neuron decays toward without input. |
| $I_{in}^{(ij)}$ | Total incoming current to neuron $i$ from neuron $j$. |
| $n$ | Number of presynaptic neurons connected to neuron $i$. |

**Interpretation:** This is a standard leaky integrator from computational neuroscience --- essentially an RC circuit where $C_{m_i}$ is the capacitor and $G_{Leak_i}$ is the resistor. Without input, $V_i(t) \to V_{Leak_i}$ exponentially with intrinsic time-constant $\tau_i = C_{m_i} / G_{Leak_i}$.

**Architecture note:** Hidden nodes are allowed to have recurrent connections, while they synapse into motor neurons (output units) in a feed-forward setting.

---

### Equation 2 --- Chemical Synaptic Transmission

Chemical synaptic transmission from neuron $j$ to neuron $i$ is modeled by a sigmoidal nonlinearity $(\mu_{ij}, \gamma_{ij})$, which is a function of the presynaptic membrane state $V_j(t)$, and has maximum weight $w_{ij}$ (Koch and Segev, 1998):

$$I_{s_{ij}} = \frac{w_{ij}}{1 + e^{-\gamma_{ij}(V_j + \mu_{ij})}} \Big(E_{ij} - V_i(t)\Big) \tag{2}$$

This is the core equation that distinguishes LTC networks from standard RNNs. The synaptic current has two multiplicative components:

1. **Sigmoidal gating:** $\frac{w_{ij}}{1 + e^{-\gamma_{ij}(V_j + \mu_{ij})}}$ --- This determines how "open" the synapse is based on the presynaptic voltage $V_j$. When $V_j$ is high and positive (relative to the threshold $-\mu_{ij}$), the sigmoid approaches 1 and the synapse is fully conducting. When $V_j$ is low, the sigmoid approaches 0 and the synapse is essentially closed.

2. **Driving force:** $(E_{ij} - V_i(t))$ --- The current is proportional to the difference between the synaptic reversal potential $E_{ij}$ and the postsynaptic voltage $V_i$. This means the current is **zero** when $V_i = E_{ij}$ (the neuron has reached the reversal potential), and the sign of the current depends on whether $E_{ij}$ is above or below $V_i$.

The synaptic current $I_{s_{ij}}$ then linearly depends on the state of neuron $i$. The reversal potential $E_{ij}$ sets whether the synapse **excites** or **inhibits** the succeeding neuron's state.

**Parameters and variables:**

| Symbol | Description |
|--------|-------------|
| $w_{ij}$ | Maximum synaptic conductance (weight) from neuron $j$ to neuron $i$. Always positive. Controls the peak magnitude of synaptic current. |
| $\gamma_{ij}$ | Slope (gain) of the sigmoidal activation. Controls the sharpness of the synapse's on/off transition. |
| $\mu_{ij}$ | Horizontal shift (threshold) of the sigmoid. Determines at what presynaptic voltage the synapse is half-activated. |
| $E_{ij}$ | Synaptic reversal potential. If $E_{ij} > V_i$, the synapse is excitatory (depolarizing). If $E_{ij} < V_i$, it is inhibitory (hyperpolarizing). |
| $V_j(t)$ | Presynaptic membrane potential of neuron $j$. |
| $V_i(t)$ | Postsynaptic membrane potential of neuron $i$. |

The sigmoid function that appears throughout the paper is defined as:

$$\sigma_i(V_j(t)) = \frac{1}{1 + e^{-\gamma_{ij}(V_j + \mu_{ij})}} \tag{2a}$$

This is a standard logistic sigmoid, bounded strictly in $(0, 1)$.

---

### Equation 3 --- Electrical Synapse (Gap Junction)

An electrical synapse (gap junction) between node $j$ and $i$ is modeled as a bidirectional junction with weight $\hat{\omega}_{ij}$, based on Ohm's law:

$$\hat{I}_{ij} = \hat{\omega}_{ij}\Big(v_j(t) - v_i(t)\Big) \tag{3}$$

Gap junctions are direct electrical connections between neurons. Unlike chemical synapses, they are:

- **Bidirectional:** Current can flow in either direction depending on the voltage difference.
- **Linear:** The current is proportional to the voltage difference (Ohmic).
- **Symmetric:** The same current flows out of neuron $j$ as flows into neuron $i$ (with opposite sign).

**Parameters:**

| Symbol | Description |
|--------|-------------|
| $\hat{\omega}_{ij}$ | Gap junction conductance between neurons $j$ and $i$. Always non-negative. |
| $v_j(t), v_i(t)$ | Membrane potentials of neurons $j$ and $i$. |

Gap junctions tend to synchronize connected neurons. Their contribution to the network dynamics is by adding a **linear term** to both the time-constant of the system ($\tau_{sys}$) and to the equilibrium state of a neuron ($A$ in Eq. 6 / Eq. 33).

---

### Equation 4 --- Full Single-Neuron LTC Dynamics (One Chemical Synapse from Neuron $j$)

The internal state dynamics of neuron $i$, $V_i(t)$, of an LTC network, receiving one chemical synapse from neuron $j$, is formulated by combining Eq. 1 with Eq. 2:

$$\frac{dV_i}{dt} = \frac{G_{Leak_i}}{C_{m_i}}\Big(V_{Leak_i} - V_i(t)\Big) + \frac{w_{ij}}{C_{m_i}}\sigma_i(V_j(t))\Big(E_{ij} - V_i\Big) \tag{4}$$

This expands the general membrane equation (Eq. 1) by substituting in the explicit chemical synapse model (Eq. 2). The first term is the leak current (normalized by capacitance), and the second term is the sigmoidal synaptic current (also normalized by capacitance). Since everything is divided by $C_{m_i}$, the units of the right-hand side are voltage per unit time (i.e., rate of change of $V_i$).

---

### Equation 5 --- Reformulated LTC Dynamics Revealing the Varying Time-Constant

If we define the intrinsic time-constant of neuron $i$ as $\tau_i = C_{m_i} / G_{Leak_i}$, Eq. 4 can be rewritten as:

$$\frac{dV_i}{dt} = -\Big(\frac{1}{\tau_i} + \frac{w_{ij}}{C_{m_i}}\sigma_i(V_j)\Big)V_i + \Big(\frac{V_{Leak_i}}{\tau_i} + \frac{w_{ij}}{C_{m_i}}\sigma_i(V_j)E_{ij}\Big) \tag{5}$$

This reformulation is crucial because it reveals the key LTC property. The equation has the form $\dot{V}_i = -\alpha(t) V_i + \beta(t)$, where $\alpha(t)$ is time-varying. Specifically, the **effective system time-constant** is:

$$\tau_{system} = \frac{1}{1/\tau_i + (w_{ij}/C_{m_i})\,\sigma_i(V_j)} \tag{5a}$$

This distinguishes LTC cells from CTRNN cells. In a CTRNN, $\tau$ is fixed. In an LTC:

- When the presynaptic neuron $j$ is strongly active ($\sigma_i(V_j) \approx 1$), the additional term $w_{ij}/C_{m_i}$ increases the denominator, making $\tau_{system}$ **smaller** (faster dynamics).
- When neuron $j$ is silent ($\sigma_i(V_j) \approx 0$), the system time-constant returns to the intrinsic value $\tau_i$ (slower dynamics).

This input-dependent modulation of temporal dynamics is the "liquid" property.

---

## 3. Network-Level Dynamics

### Equation 6 --- Matrix-Form Network ODE

The overall network dynamics of the LTC RNNs with $n$ motor neurons (output units) and $N$ interneurons (hidden units), representing the internal states as $u(t) = [u_1(t), \ldots, u_{n+N}(t)]^T$, can be written in matrix format:

$$\dot{u}(t) = -\Big(1/\tau + W\sigma(u(t))\Big)u(t) + A + W\sigma(u(t))B \tag{6}$$

This is the full network dynamics in compact matrix form. Every neuron's state is updated simultaneously. The structure is:

- **Decay term:** $-(1/\tau + W\sigma(u(t)))u(t)$ --- Each neuron decays toward zero, but the rate of decay is modulated by the nonlinear synaptic input $W\sigma(u(t))$.
- **Drive term:** $A + W\sigma(u(t))B$ --- The state is pushed toward equilibrium values determined by the resting potentials ($A$) and the synaptic reversal potentials ($B$), weighted by the synaptic activation.

**Parameters and variables:**

| Symbol | Dimensions | Description |
|--------|-----------|-------------|
| $u(t)$ | $(n+N) \times 1$ | Full state vector: $n$ output neurons + $N$ hidden neurons. |
| $\sigma(\cdot)$ | element-wise | $C^1$-sigmoid function applied element-wise to the state vector. |
| $\tau^{n+N}$ | $(n+N) \times 1$ | Vector of intrinsic neuronal time-constants. All entries positive. |
| $W$ | $(n+N) \times (n+N)$ | Effective weight matrix. This is the product of a raw weight matrix of shape $(n+N) \times (n+N)$ and an $(n+N)$ vector containing the reversed values of all $C_{m_i}$s. |
| $A$ | $(n+N) \times 1$ | Vector of resting state contributions. Contains all $V_{Leak_i}/C_{m_i}$ values. |
| $B$ | $(n+N) \times 1$ | Vector of synaptic reversal potentials. Contains all $E_{ij}$ values. |

**Constraints:** Both $A$ and $B$ have entries bounded to a range $[-\alpha, \beta]$ for $0 < \alpha < +\infty$ and $0 \le \beta < +\infty$.

---

## 4. Universal Approximation Theorem

This is the central theoretical result of the paper. It proves that LTC networks can approximate any continuous dynamical system.

### Theorem 1

Let $S$ be an open subset of $\mathbb{R}^n$ and $F : S \to \mathbb{R}^n$ be an autonomous ordinary differential equation, a $C^1$-mapping, and $\dot{x} = F(x)$ determine a dynamical system on $S$. Let $D$ denote a compact subset of $S$ and we consider a finite trajectory of the system as $I = [0, T]$. Then, for a positive $\epsilon$, there exist an integer $N$ and a liquid time-constant recurrent neural network with $N$ hidden units, $n$ output units, such that for any given trajectory $\{x(t); t \in I\}$ of the system with initial value $x(0) \in D$, and a proper initial condition of the network, the statement below holds:

$$\max_{t \in I} |x(t) - u(t)| < \epsilon \tag{Thm.1}$$

**Interpretation:** Any finite trajectory of any $n$-dimensional continuous dynamical system can be approximated to arbitrary precision $\epsilon$ by an LTC RNN with enough hidden units $N$. This is the LTC analog of the classical universal approximation theorem for feedforward networks, extended to dynamical systems.

The proof builds on the universal approximation theorem for feedforward networks (Hornik, Stinchcombe, and White, 1989), RNNs (Funahashi, 1989), and time-continuous RNNs (Funahashi and Nakamura, 1993).

---

### Supporting Lemma and Proof Equations

#### Equation 7 --- Lemma 1 (Existence and Uniqueness)

For an $F : \mathbb{R}^n \to \mathbb{R}^{+n}$ which is a bounded $C^1$-mapping, the differential equation:

$$\dot{x} = -(1/\tau + F(x))x + A + BF(x) \tag{7}$$

in which $\tau$ is a positive constant, and $A$ and $B$ are constant coefficients bounded to a range $[-\alpha, \beta]$ for $0 < \alpha < +\infty$ and $0 \le \beta < +\infty$, has a **unique solution** on $[0, \infty)$.

This lemma establishes that the LTC ODE always has a unique, well-defined solution for all positive time. This is essential for the network to be well-defined as a computational model.

**Proof sketch:** Take a positive $M$ such that $0 \le F_i(x) \le M$ (since $F$ is bounded), then bound the solution using the comparison ODE:

$$\dot{y} = -(1/\tau + M)y + A + BM \tag{8}$$

#### Equations 9--11 --- State Bounds from Lemma 1

From Eq. 8, the paper shows:

$$\min\left\{|x_i(0)|,\; \frac{\tau(A + BM)}{1 + \tau M}\right\} \le x_i(t) \le \max\left\{|x_i(0)|,\; \frac{\tau(A + BM)}{1 + \tau M}\right\} \tag{9}$$

Setting $C_1 = \min\{C_{min_i}\}$ and $C_2 = \max\{C_{max_i}\}$, the solution $x(t)$ satisfies:

$$\sqrt{n}\,C_1 \le x(t) \le \sqrt{n}\,C_2 \tag{10}$$

This demonstrates that the solution is bounded for all time.

#### Equation 11 --- Approximation Error Bound

Based on the universal approximation theorem, there exist an integer $N$, an $n \times N$ matrix $B$, an $N \times n$ matrix $C$, and an $N$-dimensional vector $\mu$ such that:

$$\max_{x} |F(x) - B\sigma(Cx + \mu)| < \frac{\epsilon_l}{2} \tag{11}$$

This says the target dynamical system $F$ can be approximated by a single hidden-layer sigmoidal network to within $\epsilon_l/2$.

#### Equation 12 --- Compact Subset for the Proof

$D_\eta$ is defined as the $\eta$-neighborhood of the compact set $\tilde{D}$:

$$D_\eta = \{x \in \mathbb{R}^n;\; \exists z \in \tilde{D},\; |x - z| \le \eta\} \tag{12}$$

#### Equation 13 --- Lipschitz Condition

The error tolerance $\epsilon_l$ must satisfy:

$$\epsilon_l < \frac{\eta L_F}{2(\exp(L_F T) - 1)} \tag{13}$$

where $L_F$ is the Lipschitz constant of $F$ on $D_\eta$.

#### Equation 14 --- Approximating Mapping $\tilde{F}$

The $C^1$-mapping $\tilde{F} : \mathbb{R}^n \to \mathbb{R}^n$ is defined as:

$$\tilde{F}(x) = -(1/\tau + W_l\sigma(Cx + \mu))x + A + W_l B\sigma(Cx + \mu) \tag{14}$$

with parameters matching Eq. 6 and $W_l = W$. This is the LTC network that will approximate the target system.

#### Equation 15 --- System Time-Constant for the Proof

$$\tau_{sys} = \frac{1}{1/\tau + W_l\sigma(Cx + \mu)} \tag{15}$$

A large $\tau_{sys}$ is chosen, conditioned on:

$$(a)\quad \forall x \in D_\eta:\; \left|\frac{x}{\tau_{sys}}\right| < \frac{\epsilon_l}{2} \tag{16}$$

$$(b)\quad \left|\frac{\mu}{\tau_{sys}}\right| < \frac{\eta L_{\tilde{G}}}{2(\exp(L_{\tilde{G}} T) - 1)} \quad\text{and}\quad \left|\frac{1}{\tau_{sys}}\right| < \frac{L_{\tilde{G}}}{2} \tag{17}$$

To satisfy conditions (a) and (b), $\tau W_l \ll 1$ must hold.

#### Equations 18--19 --- Approximation Error Propagation

From Eqs. 11 and 14:

$$\max_{x \in D_\eta} |F(x) - \tilde{F}(x)| < \epsilon_l \tag{18}$$

Setting $x(t)$ and $\tilde{x}(t)$ with initial state $x(0) = \tilde{x}(0) = x_0 \in D$ as solutions of:

$$\dot{x} = F(x) \tag{19}$$

$$\dot{\tilde{x}} = \tilde{F}(x) \tag{20}$$

By Lemma 5 in Funahashi and Nakamura (1993), for any $t \in I$:

$$|x(t) - \tilde{x}(t)| \le \frac{\epsilon_l}{L_F}(\exp(L_F t) - 1) \tag{21}$$

$$\le \frac{\epsilon_l}{L_F}(\exp(L_F T) - 1) \tag{22}$$

Thus:

$$\max_{t \in I} |x(t) - \tilde{x}(t)| < \frac{\eta}{2} \tag{23}$$

#### Part 2 of Proof --- Higher-Dimensional Embedding

Considering the dynamical system defined by $\tilde{F}$ in Part 1:

$$\dot{\tilde{x}} = -\frac{1}{\tau_{sys}}\tilde{x} + A_1 + W_l B\sigma(C\tilde{x} + \mu) \tag{24}$$

Setting $\tilde{y} = C\tilde{x} + \mu$, then:

$$\dot{\tilde{y}} = C\dot{\tilde{x}} = -\frac{1}{\tau_{sys}}\tilde{y} + E\sigma(\tilde{y}) + A_2 + \frac{\mu}{\tau_{sys}} \tag{25}$$

where $E = CW_lB$, an $N \times N$ matrix. The augmented state and mapping are:

$$\tilde{z} = [\tilde{x}_1, \ldots, \tilde{x}_n, \tilde{y}_1, \ldots, \tilde{y}_N]^T \tag{26}$$

$$\tilde{G}(\tilde{z}) = -\frac{1}{\tau_{sys}}\tilde{z} + W\sigma(\tilde{z}) + A + \frac{\mu_1}{\tau_{sys}} \tag{27}$$

with block-structured weight matrix, bias, and resting vectors:

$$W^{(n+N)\times(n+N)} = \begin{pmatrix} 0 & B \\ 0 & E \end{pmatrix} \tag{28}$$

$$\mu_1^{n+N} = \begin{pmatrix} 0 \\ \mu \end{pmatrix}, \quad A^{n+N} = \begin{pmatrix} A_1 \\ A_2 \end{pmatrix} \tag{29}$$

#### Part 2 continued --- The realizable LTC system

The dynamical system:

$$G(z) = -\frac{1}{\tau_{sys}}z + W\sigma(z) + A \tag{30}$$

$$\dot{z} = -\frac{1}{\tau_{sys}}z + W\sigma(z) + A \tag{31}$$

can be realized by an LTC RNN, by setting $h(t) = [h_1(t), \ldots, h_N(t)]^T$ as the hidden states and $u(t) = [U_1(t), \ldots, U_n(t)]^T$ as the output states of the system.

From Eq. 27, Eq. 30, and the conditions on $\tau_{sys}$:

$$|\tilde{G}(z) - G(z)| = \left|\frac{\mu}{\tau_{sys}}\right| < \frac{\eta L_{\tilde{G}}}{2(\exp(L_{\tilde{G}} T) - 1)} \tag{32}$$

Setting $\tilde{z}(t)$ and $z(t)$ as solutions of the two dynamical systems with initial conditions:

$$\dot{\tilde{z}} = \tilde{G}(z), \quad \begin{cases} \tilde{x}(0) = x_0 \in D \\ \tilde{y}(0) = Cx_0 + \mu \end{cases} \tag{33}$$

$$\dot{z} = G(z), \quad \begin{cases} u(0) = x_0 \in D \\ \tilde{h}(0) = Cx_0 + \mu \end{cases} \tag{34}$$

By Lemma 5 of Funahashi and Nakamura (1993):

$$\max_{t \in I} |\tilde{z}(t) - z(t)| < \frac{\eta}{2} \tag{35}$$

and therefore:

$$\max_{t \in I} |\tilde{x}(t) - u(t)| < \frac{\eta}{2} \tag{36}$$

#### Part 3 --- Final Result

Combining Eq. 23 and Eq. 36, for a positive $\epsilon$, we can design an LTC network with internal dynamical state $z(t)$, with $\tau_{sys}$ and $W$. For $x(t)$ satisfying $\dot{x} = F(x)$, if we initialize the network by $u(0) = x(0)$ and $h(0) = Cx(0) + \mu$, we obtain:

$$\max_{t \in I} |x(t) - u(t)| < \frac{\eta}{2} + \frac{\eta}{2} = \eta < \epsilon \tag{37}$$

**Remarks from the paper:**

- The LTC architecture allows interneurons (hidden layer) to have recurrent connections to each other, but assumes feed-forward connections from hidden to motor neuron units (output).
- No inputs are assumed (autonomous system). The proof shows the interneurons' network together with motor neurons can approximate any finite trajectory.
- The proof uses only chemical synapses. Extending to gap junctions is straightforward since they add a linear term to $\tau_{sys}$ and to $A$ in Eq. 31.

---

## 5. Bounds on $\tau_{sys}$ and State of an LTC RNN

### Lemma 2 --- Time-Constant Bounds

#### Equation 38 --- Time-Constant Bound Range

Let $v_i$ denote the state of a neuron $i$, receiving $N$ synaptic connections of the form Eq. 2, and $P$ gap junctions of the form Eq. 3 from the other neurons of a LTC network $G$. If dynamics of each neuron's state is determined by Eq. 1, then the time-constant of the activity of the neuron, $\tau_i$, is bound to a range:

$$\frac{C_i}{g_i + \sum_{j=1}^{N} w_{ij} + \sum_{j=1}^{P} \hat{w}_{ij}} \;\le\; \tau_i \;\le\; \frac{C_i}{g_i + \sum_{j=1}^{P} \hat{w}_{ij}} \tag{38}$$

**Proof outline:** The sigmoidal nonlinearity in Eq. 2 is a monotonically increasing function bounded to range 0 and 1:

$$0 < S(Y_j, \sigma_{ij}, \mu_{ij}, E_{ij}) < 1 \tag{39}$$

By replacing the upper bound of $S$ in Eq. 2 and substituting into Eq. 1, the full dynamics expand to:

$$C_i \frac{dv_i}{dt} = g_i(V_{leak} - v_i) + \sum_{j=1}^{N} w_{ij}(E_{ij} - v_i) + \sum_{j=1}^{P} \hat{w}_{ij}(v_j - v_i) \tag{40}$$

Separating into terms that multiply $v_i$ and terms that don't:

$$C_i \frac{dv_i}{dt} = \underbrace{\Big(g_i V_{leak} + \sum_{j=1}^{N} w_{ij} E_{ij}\Big) + \sum_{j=1}^{P} \hat{w}_{ij} v_j}_{A} - \underbrace{\Big(g_i + \sum_{j=1}^{N} w_{ij} + \sum_{j=1}^{P} \hat{w}_{ij}\Big)}_{B} v_i \tag{41}$$

$$C_i \frac{dv_i}{dt} = A - Bv_i \tag{42}$$

Assuming fixed $v_j$, this is a first-order linear ODE with solution:

$$v_i(t) = k_1 e^{-\frac{B}{C_i}t} + \frac{A}{B} \tag{43}$$

From this solution, the **lower bound** (minimum) of the time-constant is:

$$\tau_i^{min} = \frac{C_i}{B} = \frac{C_i}{g_i + \sum_{j=1}^{N} w_{ij} + \sum_{j=1}^{P} \hat{w}_{ij}} \tag{44}$$

By replacing the **lower bound** of $S$ ($\sigma \to 0$), the term $\sum w_{ij}(E_{ij} - v_i)$ becomes zero, leaving:

$$C_i \frac{dv_i}{dt} = \underbrace{\Big(g_i V_{leak} + \sum_{j=1}^{P} \hat{w}_{ij} v_j\Big)}_{A} - \underbrace{\Big(g_i + \sum_{j=1}^{P} \hat{w}_{ij}\Big)}_{B} v_i \tag{45}$$

The **upper bound** (maximum) of the time-constant is:

$$\tau_i^{max} = \frac{C_i}{g_i + \sum_{j=1}^{P} \hat{w}_{ij}} \tag{46}$$

**Interpretation:** The time-constant is always finite and positive. It is smallest (fastest dynamics) when all chemical synapses are fully active, and largest (slowest dynamics) when all chemical synapses are silent. Gap junctions always contribute to both bounds. This proves the dynamics can never become infinitely fast or infinitely slow.

---

### Lemma 3 --- Hidden State Bounds

#### Equation 47 --- State Bound Range

Let $v_i$ denote the state of a neuron $i$, receiving $N$ synaptic connections of form Eq. 2 from the other nodes of a network $G$. If dynamics of each neuron is determined by Eq. 1, then the hidden state of the neurons on a finite trajectory $I = [0, T]$ ($0 < T < +\infty$) is bounded as follows:

$$\min_{t \in I}\big(V_{Leak_i},\; E_{ij}^{min}\big) \;\le\; v_i(t) \;\le\; \max_{t \in I}\big(V_{Leak_i},\; E_{ij}^{max}\big) \tag{47}$$

**Proof outline:** Insert $M = \max\{V_{Leak_i}, E_{ij}^{max}\}$ as the membrane potential into Eq. 40:

$$C_i \frac{dv_i}{dt} = \underbrace{g_i(V_{leak} - M)}_{\le 0} + \underbrace{\sum_{j=1}^{N} w_{ij}\sigma(v_j)(E_{ij} - M)}_{\le 0} \tag{48}$$

The right-hand side is negative (since $M$ is the maximum of all driving potentials), so the derivative is negative. Using an explicit Euler approximation:

$$C_i \frac{dv_i}{dt} \le 0 \implies \frac{dv_i}{dt} \approx \frac{v(t+\delta t) - v(t)}{\delta t} \le 0 \tag{49}$$

Substituting $v(t) = M$:

$$\frac{v(t + \delta t) - M}{\delta t} \le 0 \implies v(t + \delta t) \le M \tag{50}$$

Therefore:

$$v_i(t) \le \max_{t \in I}(V_{Leak_i}, E_{ij}^{max}) \tag{51}$$

Similarly, substituting $m = \min\{V_{Leak_i}, E_{ij}^{min}\}$:

$$v_i(t) \ge \min_{t \in I}(V_{Leak_i}, E_{ij}^{min}) \tag{52}$$

**Interpretation:** The membrane potential can never exceed the most excitatory reversal potential or drop below the most inhibitory reversal potential (or the resting potential, whichever is more extreme). This guarantees that the LTC network's states are always bounded --- they cannot "blow up" regardless of input.

---

## 6. Parameter Summary Table

| Parameter | Symbol | Role | Typical Range | Implementation Notes |
|-----------|--------|------|---------------|---------------------|
| Membrane capacitance | $C_{m_i}$ | Controls speed of voltage change | $[0.1, 10.0]$ | One per neuron. Larger = slower. Public property for biophysical mode. |
| Leak conductance | $G_{Leak_i}$ | Rate of return to rest | Derived: $C_{m_i}/\tau_i$ | Dependent on $C_m$ and $\tau$. Private/derived. |
| Leak potential | $V_{Leak_i}$ | Resting equilibrium voltage | $[-1.0, 0.0]$ | One per neuron. Sets the "default" state. Public for biophysical mode. |
| Intrinsic time-constant | $\tau_i$ | Intrinsic temporal dynamics | $[0.01, 10.0]$ | $= C_{m_i}/G_{Leak_i}$. Public, learnable. |
| Synaptic weight | $w_{ij}$ | Maximum synaptic conductance | $[0, 2.0]$ | Always $\ge 0$ in biophysical model. Private, learnable. |
| Sigmoid slope | $\gamma_{ij}$ | Sharpness of synaptic activation | $[0.5, 5.0]$ | One per synapse. Private, learnable. |
| Sigmoid shift | $\mu_{ij}$ | Threshold for synapse activation | $[-2.0, 2.0]$ | One per synapse. Private, learnable. |
| Reversal potential | $E_{ij}$ | Excitatory/inhibitory character | $[-1.0, 1.0]$ | Determines synapse sign. Private, learnable. |
| Gap junction weight | $\hat{\omega}_{ij}$ | Electrical coupling strength | $[0, 1.0]$ | Optional. Bidirectional. Private, learnable. |
| Resting vector | $A$ | Equilibrium contributions | Bounded $[-\alpha, \beta]$ | Contains $V_{Leak_i}/C_{m_i}$. Private. |
| Reversal vector | $B$ | Synaptic drive targets | Bounded $[-\alpha, \beta]$ | Contains $E_{ij}$ values. Private. |
| Weight matrix | $W$ | Network connectivity | Learned | Product of raw weights and $1/C_m$. Private. |

---

## 7. Variable Summary Table

| Variable | Symbol | Dimensions | Type | Description |
|----------|--------|-----------|------|-------------|
| Membrane potential | $V_i(t)$ | $N \times 1$ | Dependent (ODE solution) | Hidden state of neuron $i$. The primary quantity being solved. |
| Full state vector | $u(t)$ | $(n+N) \times 1$ | Dependent (ODE solution) | Combined hidden + output state. |
| Synaptic current | $I_{s_{ij}}$ | scalar per pair | Dependent | Chemical synapse current from $j$ to $i$. |
| Gap junction current | $\hat{I}_{ij}$ | scalar per pair | Dependent | Electrical synapse current between $j$ and $i$. |
| Total input current | $I_{in}^{(ij)}$ | scalar per pair | Dependent | Sum of all currents from $j$ to $i$. |
| Sigmoid activation | $\sigma_i(V_j)$ | scalar per pair | Dependent | Gating variable for synapse $j \to i$. |
| System time-constant | $\tau_{system}$ | $N \times 1$ | Dependent (varies with state) | Effective time-constant, changes every step. |

---

## 8. Equation Index (Order of Appearance)

| Paper Eq. # | This Doc | Description |
|-------------|----------|-------------|
| Eq. 1 | Eq. 1 | Membrane integrator ODE |
| Eq. 2 | Eq. 2 | Chemical synaptic transmission |
| --- | Eq. 2a | Sigmoid function definition |
| Eq. 3 | Eq. 3 | Gap junction (electrical synapse) |
| Eq. 4 | Eq. 4 | Single-neuron LTC dynamics |
| Eq. 5 | Eq. 5 | Varying time-constant form |
| --- | Eq. 5a | System time-constant definition |
| Eq. 6 | Eq. 6 | Network matrix ODE |
| Eq. 7 | Eq. 7 | Lemma 1 (existence/uniqueness ODE) |
| Eq. 8 | Eq. 8 | Comparison ODE for proof |
| Eq. 9 | --- | Auxiliary equation in Lemma 1 proof |
| Eq. 10 | Eq. 9 | State bounds from Lemma 1 |
| Eq. 11 | Eq. 10 | Norm bound on state |
| Eq. 12 | Eq. 12 | Compact subset definition |
| Eq. 13 | Eq. 13 | Lipschitz error tolerance |
| Eq. 14 | Eq. 11 | Universal approximation of $F$ |
| Eq. 15 | Eq. 14 | Approximating mapping $\tilde{F}$ |
| Eq. 16 | Eq. 15 | System $\tau$ for proof |
| Eq. 17 | Eq. 16 | Condition (a) on $\tau_{sys}$ |
| Eq. 18 | Eq. 17 | Condition (b) on $\tau_{sys}$ |
| Eq. 19 | Eq. 18 | $F$ vs $\tilde{F}$ error bound |
| Eq. 20 | Eq. 19 | Target system ODE |
| Eq. 21 | Eq. 20 | Approximating system ODE |
| Eq. 22--23 | Eq. 21--22 | Error propagation |
| Eq. 24 | Eq. 23 | Part 1 error bound |
| Eq. 25 | Eq. 24 | Higher-dimensional system |
| Eq. 26 | Eq. 25 | Transformed coordinates |
| Eq. 27 | Eq. 26 | Augmented state |
| Eq. 28 | Eq. 27 | Augmented mapping $\tilde{G}$ |
| Eq. 29 | Eq. 28 | Block weight matrix |
| Eq. 30 | Eq. 29 | Block bias and resting vectors |
| Eq. 31 | --- | Equivalent system |
| Eq. 32 | Eq. 30 | Realizable LTC system $G$ |
| Eq. 33 | Eq. 31 | Realizable LTC ODE |
| Eq. 34 | Eq. 32 | $\tilde{G}$ vs $G$ error |
| Eq. 35 | Eq. 33 | Initial conditions for $\tilde{z}$ |
| Eq. 36 | Eq. 34 | Initial conditions for $z$ |
| Eq. 37 | Eq. 35 | $\tilde{z}$ vs $z$ error bound |
| Eq. 38 | Eq. 36 | $\tilde{x}$ vs $u$ error bound |
| Eq. 39 | Eq. 37 | Final universal approximation result |
| Eq. 40 | Eq. 38 | Time-constant bounds (Lemma 2) |
| Eq. 41 | Eq. 39 | Sigmoid bound |
| Eq. 42 | Eq. 40 | Expanded dynamics for bound proof |
| Eq. 43--44 | Eq. 41 | Separated A and B terms |
| Eq. 45 | Eq. 42 | Simplified $A - Bv_i$ form |
| Eq. 46 | Eq. 43 | Solution form for time-constant |
| Eq. 47 | Eq. 44 | $\tau_i^{min}$ |
| Eq. 48 | Eq. 45 | Dynamics with $\sigma = 0$ |
| Eq. 49 | Eq. 46 | $\tau_i^{max}$ |
| Eq. 50 | Eq. 47 | State bounds (Lemma 3) |
| Eq. 51 | Eq. 48 | Upper bound proof (inserting $M$) |
| Eq. 52 | Eq. 49 | Euler approximation for bound |
| Eq. 53 | Eq. 50 | Upper bound derivation |
| Eq. 54 | Eq. 51 | Upper bound result |
| Eq. 55--56 | Eq. 52 | Lower bound result |
