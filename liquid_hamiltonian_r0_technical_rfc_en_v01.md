# RFC: Geometric Liquid–Hamiltonian Hierarchical Prediction

**Technical RFC / Open Research Note — Reference Configuration R0**

- Status: Draft working specification; open for testing, falsification, and extension.
- RFC version: 0.1.
- English revision date: 2026-11-04.
- Source record: `liquid_hamiltonian_research_record_v01.md`, version 1.0, consolidated 2026-11-03.

This document specifies a **modular combination and architectural specification**, designated **Reference Configuration R0**, connecting existing concepts from H-JEPA, Liquid Time-Constant (LTC) networks, Hamiltonian dynamics, Geometric Deep Learning constraints, and continuum field representations. It records interfaces, configuration choices, conditional mathematical properties, and deterministic mechanism experiments. It is a working specification, not an academic journal paper.

**Universal superiority and priority are unverified. Empirical validation on large-scale real-world data remains open.** The reported experiments do not train a world model or measure future-prediction accuracy.

**Collaboration disclosure.** The author initiated the core hypothesis, design choices, and direction as an open **“Ignition” prototype**. Detailed formulations, mathematical derivations, verification code, numerical execution, and document preparation were collaboratively carried out with an LLM (GPT). Recorded checks do not constitute independent verification by the author or by a third party. This RFC provides an open scaffold for the research community to test, falsify, or extend the specified configuration.

The specification distinguishes definitions, configuration choices, derived consequences, numerical observations, and unresolved hypotheses. Sections 3–6 contain the mathematical interfaces and conditions; Section 7 contains the experiment protocol and results; Section 8 maps the configuration to existing work. Supporting derivations and reproducibility artifacts are indexed in Section 12.[^format]

## Contents

1. [Scope and configuration overview](#1-scope-and-configuration-overview)
2. [Design decisions and retained corrections](#2-design-decisions-and-retained-corrections)
3. [R0 interfaces and update specification](#3-r0-interfaces-and-update-specification)
4. [Conditional mathematical properties](#4-conditional-mathematical-properties)
5. [Two-level configuration and coarse-graining compatibility](#5-two-level-configuration-and-coarse-graining-compatibility)
6. [Restricted learning interface](#6-restricted-learning-interface)
7. [Experiments A and B](#7-experiments-a-and-b)
8. [Relationship to existing work](#8-relationship-to-existing-work)
9. [Claim and verification status](#9-claim-and-verification-status)
10. [Open verification and extension tasks](#10-open-verification-and-extension-tasks)
11. [Collaboration provenance and handoff invariants](#11-collaboration-provenance-and-handoff-invariants)
12. [Supporting artifact map](#12-supporting-artifact-map)

## 1. Scope and configuration overview

The configuration investigates the behavior of connecting hierarchical future prediction, input-dependent LTC response, Hamiltonian momentum, geometric compatibility constraints, and continuum field states.

R0 is specified as follows:

> **Jointly update multiple future-state candidates related by coarse-graining, using internal dynamics with momentum and LTC-controlled friction.**

| Component | Current role |
|---|---|
| Hierarchical prediction informed by H-JEPA | Future-prediction tasks and learning interfaces at different resolutions and prediction horizons |
| Continuum | Field-valued candidates in function spaces |
| Hamiltonian | Momentum and inertia in candidate updates |
| LTC | Friction response times and history through auxiliary states |
| Geometric constraints | Conditions relating specified symmetries, metrics, coarse-graining, and updates |

The recorded specification includes:

1. Interfaces from inputs to internal updates, finite-time readout, and prediction loss.
2. Energy balance, conditional convergence, inter-level disagreement, and a friction layout compatible with coarse-graining.
3. Experiment A comparing fixed, direct, and LTC-controlled friction, and Experiment B comparing friction layouts.

R0 connects existing components. Its priority and general performance advantages are not established.[^scope]

## 2. Design decisions and retained corrections

| Design option or question | Current disposition |
|---|---|
| Couple hyperbolic depth directly to a time constant | Retained as an earlier hypothesis; R0 uses a fixed metric |
| Obtain a hierarchy through spontaneous emergence | Not required; a finite hierarchy and coarse-graining maps are specified |
| Use LTC for convergence and Hamiltonian dynamics for inertia | Instantiated as inertial dynamics with friction control |
| Retain information across the entire system | The information quantity and its conservation guarantee remain unspecified; dissipation alone does not establish this property |
| Use a continuum rather than particles | States are fields in Hilbert spaces; general laws of physical continua have not been derived |
| Incorporate GDL | Instantiated through group equivariance, metric compatibility, and coarse-graining compatibility conditions |
| Assign different friction or response to each level | May drive disagreement; a layout sharing coarse-component friction is also specified |
| Defer discretization | Formulation preceded mechanism experiments restricted to an invariant finite-mode subspace |

The following corrections and exclusions apply to the earlier formulation:

- On a suitable domain, the squared-distance gradient is $\operatorname{grad}_z\frac12d(z,z_{\rm eq})^2=-\log_z(z_{\rm eq})$. The attracting negative-gradient term therefore has sign $+\tau^{-1}\log_z(z_{\rm eq})$. The negative sign in the earlier damping proposal had the opposite direction.
- A Riemannian metric alone does not exclude a constant encoder. A distance loss can be zero when both representations equal the same constant.
- A positive time constant or bounded state does not, by itself, establish convergence of an arbitrary LTC network to a single point.
- The friction intertwining condition is not identified with gauge invariance, topological protection, or conservation of all information.
- Inertia and dissipation alone do not establish global optimization, fastest convergence, or deep-reasoning capability.

The evaluation scope is the specified interfaces, assumptions, and consequences of the combination.

## 3. R0 interfaces and update specification

### 3.1 States, inputs, and time variables

| Symbol | Definition |
|---|---|
| $t$ | Observed world time |
| $\Delta>0$ | Future-prediction horizon |
| $s$ | Internal computation time for candidate updates |
| $S$ | Internal readout time |
| $X_\ell$ | Real Hilbert space for level $\ell$ |
| $q_\ell\in X_\ell$ | Candidate future latent state |
| $p_\ell\in X_\ell$ | Momentum of the internal update |
| $h_\ell\in\mathbb R$ | LTC-type controller state that modulates friction |
| $C_\ell:X_\ell\to X_{\ell+1}$ | Fixed bounded linear coarse-graining map |
| $G_\ell:X_\ell\to X_\ell$ | Fixed self-adjoint positive-definite metric operator |

The input $I$ contains available observation history, available action plans, and the prediction horizon $\Delta$, as applicable. Future observations are excluded from inference inputs and used only as training targets.

Using a fixed encoder $E_\psi$, define:

$$
z_1(t)=E_\psi(o_t),\qquad z_{\ell+1}(t)=C_\ell z_\ell(t)
$$

Initial candidates are constructed from available features. Upper-level initial candidates are made consistent by coarse-graining. Momentum is initialized to zero; controller states are initialized within $[0,A_\ell]$.[^jepa]

### 3.2 Coupled potential and Hamiltonian

$$
\begin{aligned}
\mathcal V_\theta(Q;I)
&=\sum_\ell U_{\ell,\theta}(q_\ell;I)
+\sum_\ell\frac{\beta_\ell}{2}
\|C_\ell q_\ell-q_{\ell+1}\|^2,\\
\mathcal H_\theta(Q,P;I)
&=\frac12\sum_\ell\langle p_\ell,G_\ell^{-1}p_\ell\rangle
+\mathcal V_\theta(Q;I).
\end{aligned}
$$

$U_\ell$ evaluates each candidate conditional on the input. The coupling term measures inter-level disagreement at the same future time. With $e_\ell=C_\ell q_\ell-q_{\ell+1}$, the force is:

$$
-\nabla_{q_\ell}\mathcal V
=-\nabla U_\ell-\beta_\ell C_\ell^*e_\ell
+\beta_{\ell-1}e_{\ell-1}.
$$

Terms absent at a hierarchy boundary are zero. Gradients are represented using the background Hilbert inner product.

### 3.3 Liquid–Hamiltonian internal updates

$$
\boxed{\begin{aligned}
\partial_s q_\ell&=G_\ell^{-1}p_\ell,\\
\partial_s p_\ell&=-\nabla_{q_\ell}\mathcal V_\theta(Q;I)
-\gamma_\ell(h_\ell)p_\ell,\\
\epsilon\partial_s h_\ell
&=-\lambda_\ell h_\ell+c_\ell(Q,P,I)(A_\ell-h_\ell),\\
\gamma_\ell(h_\ell)&=\gamma_{\ell,0}+\eta_\ell h_\ell.
\end{aligned}}
$$

The parameter conditions are $\epsilon,\lambda_\ell,A_\ell,\gamma_{\ell,0}>0$, $\eta_\ell\ge0$, and $0\le c_\ell\le c_{\ell,\max}$. One gate choice is:

$$
Y_\ell=\langle p_\ell,G_\ell^{-1}p_\ell\rangle,\qquad
c_\ell=c_{\ell,\max}\sigma(a_\ell Y_\ell+d_\ell(I))
$$

Here $\sigma$ is the logistic sigmoid.

The controller is a special case of the LTC form and can be written as:

$$
\tau_{h,\ell}=\frac{\epsilon}{\lambda_\ell+c_\ell},\qquad
\bar h_\ell=\frac{A_\ell c_\ell}{\lambda_\ell+c_\ell},\qquad
\tau_{h,\ell}\partial_s h_\ell=\bar h_\ell-h_\ell
$$

$\tau_h$ is the controller relaxation time constant with its instantaneous input held fixed. It is distinct from the prediction horizon $\Delta$ and the convergence time of the coupled system.

### 3.4 Readout and learning interface

$$
\widehat z_\ell(t+\Delta)=q_\ell(S;I),\qquad
\mathcal L=\sum_\ell w_\ell
\mathbb E\|q_\ell(S;I)-z_\ell(t+\Delta)\|^2.
$$

Different prediction-horizon distributions may be used at different levels. Inter-level coupling within a query compares the same future time. Differentiability of the finite-time flow with respect to its parameters, together with the applicable regularity conditions, provides an interface from the prediction loss to the potential and gate parameters.

## 4. Conditional mathematical properties

### 4.1 Controller bounds and energy balance

At $h=0$, $h_s\ge0$; at $h=A$, $h_s<0$. Therefore, initialization within the interval preserves $h(s)\in[0,A]$.

For fixed input, fixed parameters, and a fixed metric:

$$
\boxed{\partial_s\mathcal H
=-\sum_\ell\gamma_\ell(h_\ell)
\langle p_\ell,G_\ell^{-1}p_\ell\rangle\le0.}
$$

The potential-force contribution cancels the contribution from the configuration update. Local Lipschitz continuity of the gradient and gate, at-most-linear growth of the potential gradient, and positive bounded friction permit continuation over finite time intervals.

For operator-valued friction $\Gamma$, positive definiteness of $\Gamma$ alone is insufficient. The energy derivative is $-\langle G^{-1}P,\Gamma P\rangle$. Commutation with $G$ is a sufficient condition. Alternatively, a friction force $-BG^{-1}P$, with $B=B^*\succeq0$, directly ensures dissipation.

### 4.2 Convergence in a strongly convex subfamily

Assume that $\nabla\mathcal V$ is globally Lipschitz, $\nabla^2\mathcal V\succeq\mu_0\operatorname{Id}$, the metric has positive uniform lower and upper bounds, and friction is uniformly positive and bounded. A Lyapunov estimate then bounds convergence of the mechanical state to the unique minimizer $Q_*$.

Set $E=\mathcal H-\mathcal V(Q_*)$, $\mu=\mu_0/g_{\max}$, and choose:

$$
0<\delta\le\min\left\{\frac{\sqrt\mu}{2},
\frac{\gamma_{\min}}{2(1+\gamma_{\max}^2/\mu)}\right\},\quad
\kappa_{\rm conv}=\frac23\min(\delta/2,\gamma_{\min})
$$

The resulting bound is:

$$
E(s)\le3E(0)e^{-\kappa_{\rm conv}s}
$$

Using $W=E+\delta\langle Q-Q_*,P\rangle$, the derivation establishes $E/2\le W\le3E/2$ and $W_s\le-\kappa_{\rm conv}W$. The detailed derivation is retained in Section 4.1 of the supporting RFC. This bound does not establish an advantage specific to LTC; it also applies to comparison systems with the same friction bounds.

### 4.3 Equilibria and finite-time outputs

$$
\partial_s Q=\partial_s P=0
\quad\Longrightarrow\quad P_*=0,\quad\nabla\mathcal V(Q_*)=0.
$$

For a fixed potential, friction does not change the set of mechanical equilibria. In the strongly convex configuration, LTC changes the trajectory and arrival time toward the same equilibrium. The finite-time output $Q(S)$ depends on the controller.

Parameters that occur only in friction, without being shared with the potential, do not affect the exact equilibrium output. A loss depending only on that output consequently provides no gradient for those friction parameters. Selection among local minima in nonconvex systems remains untested.

### 4.4 History dependence and the direct-friction approximation

Along a trajectory, set $a=(\lambda+c)/\epsilon$ and $b=Ac/\epsilon$. Then:

$$
h(s)=e^{-\int_0^s a(v)dv}h(0)
+\int_0^s e^{-\int_u^s a(v)dv}b(u)du.
$$

This expression specifies the controller's dependence on past states. If $\bar h(s)$ is differentiable:

$$
|h(s)-\bar h(s)|\le
|h(0)-\bar h(0)|e^{-\lambda s/\epsilon}
+\frac{\epsilon}{\lambda}\sup_{0\le u\le s}|\partial_u\bar h(u)|.
$$

On a common finite time interval, assume uniform bounds, independent of $\epsilon$, on the trajectory and the required derivatives and Lipschitz constants. A Gronwall estimate then gives an $O(\epsilon)$ approximation of the mechanical state by the system with direct friction $\gamma_D=\gamma_0+\eta\bar h$. Uniform approximation of the controller state itself requires treatment of the initial layer.

### 4.5 Symmetry conditions

For unitary group actions $T_\ell(g)$ at each level, impose:

$$
G_\ell T_\ell=T_\ell G_\ell,\quad
C_\ell T_\ell=T_{\ell+1}C_\ell,\quad
U_\ell(T_\ell q;gI)=U_\ell(q;I)
$$

Together with gate invariance, these conditions make the update system equivariant. Corresponding conditions on input processing, the encoder, and initialization extend the property to the full predictor. The claim concerns the specified transformations.[^geometry]

## 5. Two-level configuration and coarse-graining compatibility

### 5.1 Field spaces and decomposition

Use periodic domains of unit volume:

$$
\begin{gathered}
X_1=L^2(\mathbb T^2),\quad X_2=L^2(\mathbb T),\quad
(Cq)(x)=\int_{\mathbb T}q(x,y)dy,\\
(C^*v)(x,y)=v(x),\quad CC^*=\operatorname{Id},\quad
\Pi=C^*C,\quad R=\operatorname{Id}-\Pi,\\
G_1=\operatorname{Id}+\kappa_gR,\quad G_2=\operatorname{Id}.
\end{gathered}
$$

$\Pi$ is the orthogonal projection onto the coarse component; $R$ projects onto the residual. Translation compatibility can also be checked at the full-field level.

Select the potential:

$$
\mathcal V_\beta=
\frac{k}{2}(\|q_1\|^2+\|q_2\|^2)
-\langle b_1,q_1\rangle-\langle b_2,q_2\rangle
+\frac\beta2\|Cq_1-q_2\|^2,\qquad b_2=Cb_1
$$

With $u=Cq_1,v=q_2,r=Rq_1$, the equations are:

$$
\begin{aligned}
u_{ss}+\gamma_1u_s+ku+\beta(u-v)&=Cb_1,\\
v_{ss}+\gamma_2v_s+kv+\beta(v-u)&=b_2,\\
(1+\kappa_g)r_{ss}+(1+\kappa_g)\gamma_1r_s+kr&=Rb_1.
\end{aligned}
$$

### 5.2 Disagreement driven by friction differences

For $e=u-v$, subtracting the first two equations gives:

$$
\boxed{e_{ss}+\gamma_2e_s+(k+2\beta)e
=-(\gamma_1-\gamma_2)u_s.}
$$

The product of the friction difference and coarse velocity drives disagreement; $\beta$ enters the restoring term. For consistent initial data, shared coarse-component friction preserves $e(s)=0$.

Using the energy $E$ relative to equilibrium:

$$
\|e(s)\|\le2\sqrt{\frac{E(0)}{k+2\beta}}.
$$

For the consistent initialization and forces specified above, $E(0)$ is also independent of $\beta$. A decreasing upper bound does not imply monotonic decrease of the observed maximum disagreement.

### 5.3 Friction compatibility with coarse-graining

$$
\boxed{C\Gamma_1=\Gamma_2C}
$$

This condition states that **applying friction and then coarse-graining is equivalent to coarse-graining and then applying friction**. The following layout satisfies it:

$$
\Gamma_1=\gamma_c\Pi+\gamma_rR,\qquad
\Gamma_2=\gamma_c\operatorname{Id}.
$$

The coarse component receives shared friction $\gamma_c$; the residual may receive a separate $\gamma_r$. In this configuration, friction commutes with the metric, giving:

$$
\mathcal H_s=-\gamma_c(\|Cp_1\|^2+\|p_2\|^2)
-\frac{\gamma_r}{1+\kappa_g}\|Rp_1\|^2\le0
$$

Generalizations also require compatibility of projected forces, configurations, momenta, and initial data.

### 5.4 Consistency preservation versus coarse-system closure

Let $F$ update the full state $Z=(Q,P,h)$, and let $\mathcal C$ be its coarse-graining map. Closure in the coarse variables requires:

$$
D\mathcal C_ZF(Z)=\bar F(\mathcal C(Z))
$$

The friction intertwining condition alone is insufficient. Even with globally shared friction, a gate that reads residual velocity allows details to affect coarse dynamics. The coarse-sharing layout in the experiments uses gates based only on coarse velocities for the coarse subsystem, thereby selecting a closed subsystem.

## 6. Restricted learning interface

With a fixed strongly convex $\Omega$, define:

$$
\mathcal V_\theta(Q;I)=\Omega(Q)-\langle b_\theta(I),Q\rangle,
\quad Q_* =\nabla\Omega^*(b_\theta(I))
$$

The learned quantity is the context-dependent $b_\theta(I)$; fixed hierarchical coupling is included in $\Omega$.

Define the Fenchel–Young loss:

$$
\ell_\Omega(b,Z)=\Omega^*(b)+\Omega(Z)-\langle b,Z\rangle
$$

For $M(I)=\mathbb E[Z\mid I]$, conditional excess FY risk satisfies:

$$
\Delta_\Omega(b,I)=D_\Omega(M(I),Q_*(I))
\ge\frac{\mu_0}{2}\|Q_*(I)-M(I)\|^2
$$

Under the required square-integrability assumptions, the finite-time output MSE satisfies:

$$
\boxed{\mathcal R_S-\mathcal R_{\rm Bayes}
\le\frac4{\mu_0}\bigl(\mathbb E E_S+
\Delta\mathcal R_{\rm FY}\bigr)}
$$

This bound separates residual internal-update error from error in the learned equilibrium. It follows from orthogonality of the conditional mean, the squared triangle inequality, and strong convexity.

Approximating $\nabla_b\ell_\Omega=Q_*-Z$ by $Q(S)-Z$ gives an error bounded by $\sqrt{2E_S/\mu_0}$. For a fixed potential, $\mathbb E\mathcal H(Q(S),P(S))$ can also be used as a friction-controller objective.

Weighted bounds for level-dependent query distributions, constants, and conditions for interchanging differentiation and expectation are retained in Section 4.3 of the supporting RFC. These results do not apply to arbitrary learned nonconvex potentials. Experiments A and B do not execute FY training.[^fy]

## 7. Experiments A and B

### 7.1 Verification questions

- **A:** How does the candidate-update process change when friction is determined directly from the state or through an LTC-type controller?
- **B:** How does inter-level consistency change under level-wise friction and shared coarse-component friction?

These are mechanism experiments on the coupled internal dynamics, not prediction-accuracy experiments.

### 7.2 Exact restriction to an invariant finite-mode subspace

$$
q_1=u\sqrt2\cos(2\pi x)+r\sqrt2\cos(2\pi y),\qquad
q_2=v\sqrt2\cos(2\pi x)
$$

Choose $\kappa_g=k=1$ and $b_1=b_2=0$. The field updates preserve this subspace, so the coefficient ODEs are an exact restriction:

$$
\begin{aligned}
u_{ss}+\gamma_u u_s+u+\beta(u-v)&=0,\\
v_{ss}+\gamma_v v_s+v+\beta(v-u)&=0,\\
2r_{ss}+2\gamma_r r_s+r&=0.
\end{aligned}
$$

$$
E=\tfrac12(u_s^2+v_s^2+2r_s^2+u^2+v^2+r^2+\beta(u-v)^2).
$$

Initialize $u(0)=v(0)=1$, all velocities to zero, and $r(0)\in\{0,1,2\}$. Simulate internal time $[0,30]$, recording at intervals of 0.01.

$$
c(Y)=\sigma(4Y-2),\quad
\epsilon h_s=-h+c(Y)(1-h),\quad\gamma=0.2+4h.
$$

Direct friction is $\gamma_D=0.2+4c(Y)/(1+c(Y))$. The primary comparison uses $h(0)=c(0)/(1+c(0))$ to match initial friction to the direct-friction system. The initialization-sensitivity comparison uses $h(0)=0$.

| Layout | Friction $(\gamma_u,\gamma_v,\gamma_r)$ | Gate signals |
|---|---|---|
| Global sharing | $(\gamma_a,\gamma_a,\gamma_a)$ | $(u_s^2+v_s^2+2r_s^2)/2$ |
| Level-wise | $(\gamma_a,\gamma_b,\gamma_a)$ | $u_s^2+2r_s^2$, $v_s^2$ |
| Coarse sharing with separate residual friction | $(\gamma_a,\gamma_a,\gamma_b)$ | $(u_s^2+v_s^2)/2$, $2r_s^2$ |

Experiment A uses $\beta=1$, fixed-friction candidates $\{0.2,0.5,1,1.4,2,2.2\}$, and LTC values $\epsilon\in\{0.05,0.2,1,5\}$. Experiment B uses $\epsilon=1$, $\beta\in\{0,1,4\}$, and the three layouts.

### 7.3 Experiment A results

Arrival time is the earliest internal time at which $E/E(0)\le10^{-3}$. “Shortest” refers only to the scanned candidates.

| Initial residual | Fixed friction: shortest in scan | Direct friction | LTC: matched initial friction | LTC: h(0)=0 |
|---|---:|---:|---:|---:|
| 0 | 5.614 | 9.284 | 8.918 | 8.881 |
| 1 | 6.985 | 9.345 | 8.678 | 8.863 |
| 2 | 7.495 | 9.621 | 6.243 | 8.072 |

LTC did not uniformly outperform fixed friction. For initial residual 2, the ordering of the shortest scanned arrival times reverses when controller initialization changes. Rankings by arrival time and by other metrics, such as integrated energy, need not agree.

### 7.4 Experiment B results

The table reports the maximum $|u-v|$ at recorded times. Each sharing-layout column gives the maximum over all $\beta$ settings for that layout.

| Initial residual | Global sharing | Level-wise β=0 | Level-wise β=1 | Level-wise β=4 | Coarse sharing |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0.083290 | 0.047709 | 0.021232 | 0 |
| 2 | 0 | 0.206430 | 0.119596 | 0.054122 | 0 |

Global sharing and coarse sharing produced zero disagreement in the implementation output, consistent with the analytical invariance condition. Level-wise friction produced disagreement when the residual was nonzero; within the selected coupling settings, maximum disagreement decreased with coupling strength. Global sharing also preserves consistency, so coarse sharing is not the only compatible layout. Its specified property is independent selection of residual friction.

### 7.5 Numerical checks and reproducibility

The primary solver is DOP853 with rtol=10^-9, atol=10^-11, and max_step=0.1. Refined solves use rtol=10^-11 and atol=10^-13.

| Check | Number of conditions | Maximum error |
|---|---:|---:|
| Fixed-friction analytical solutions | 18 | 9.771×10^-11 |
| Refined solves for all conditions | 72 | 1.256×10^-8 |
| Cross-solver checks with Radau | 8 | 1.269×10^-10 |
| Energy plus accumulated dissipated-work balance | All conditions | 2.828×10^-11 |

The 72 entries cover 69 distinct configurations; 3 configurations are intentionally shared between A and B. All checks satisfy the diagnostic threshold 10^-7. State discrepancies use $\max|x-x_{ref}|/(1+|x_{ref}|)$; balance discrepancies use $\max|E+W-E(0)|/E(0)$.

The supporting artifacts contain code, protocols, all conditions, raw trajectories, figures, verification CSV files, environment information, and hashes. This RFC reformats the existing record; it does not report newly executed experiments.[^experiment]

## 8. Relationship to existing work

| Research family | Existing structure | Interface or distinction |
|---|---|---|
| LTC [1] | First-order ODEs with input-dependent time constants | A one-state controller that modulates friction |
| FAD / iKFAD [2,3] | Momentum, kinetic-energy-controlled adaptive friction, cubic damping, and projected friction | Bounded LTC-type control and a friction layout defined by coarse-graining |
| HCFC [4] | Connections between liquid/CfC systems and Hamiltonian evolution | No broad priority claim for the combination; detailed equation-level comparison remains unresolved |
| H-JEPA [5] | Learned hierarchies, prediction, and planning over different time horizons | A reference internal predictor using fixed features and coarse-graining |
| LeHami [6] | Symplectic latent prediction in world time | Dissipative candidate updates in internal time; volume-preservation guarantees are not inherited |
| SPEN [7] | Candidate inference and learning through energy minimization | Explicit field, hierarchy, momentum, and friction specifications |
| Exact lumpability [8] | Compatibility of coarse-graining with vector fields and flows | Friction-operation compatibility is distinguished from closure of the full update |
| Port-Hamiltonian reduction [9] | Reduction preserving conservative and dissipative structure | Friction layouts between levels retained simultaneously |
| G-FNO / HNS [10,11] | Equivariant field operators and learning Hamiltonian functionals | Existing interfaces for fields, symmetry, and Hamiltonian structure |
| Hamiltonian World Models perspective [12] | Structured world models with control and dissipation | Explicit internal time, LTC, and coarse-graining interfaces |
| Predictive coding [13] | A joint objective for inter-level errors and joint state updates | Hierarchical internal inference is existing structure; inertia, friction, and geometric conditions are specified here |
| MDEQ [14] | Joint fixed points across multiple resolutions | The configuration specifies an energy and a finite-time trajectory |
| Mori–Zwanzig [15] | Memory effects of unresolved degrees of freedom after coarse-graining | LTC history dependence is recorded, but the controller is not derived from this formalism |
| CoDA [16] | Connections between LTC-type models and Mori–Zwanzig/Koopman reduction | Internal-update design rather than acceleration of an existing model |

An architectural distinction does not establish priority. The supporting comparison note records equation-level comparisons and the extent of source inspection for each reference.[^literature]

## 9. Claim and verification status

| Status | Recorded scope |
|---|---|
| Defined | A reference configuration connecting five components, finite-time outputs, and a learning loss |
| Derived under conditions | Controller invariant interval, energy balance, convergence in a strongly convex subfamily, two-level disagreement equation, compatible friction layout, and a risk bound for a convex subfamily |
| Numerically observed | Sensitivity to friction response and initialization, ranking reversal, disagreement under level-wise friction, and consistency under sharing layouts |
| Untested | Learning, future-prediction accuracy, planning performance, deep-reasoning effects, general nonconvex systems, spatial PDE effects, and empirical equivariance tests |
| Undetermined | Priority, scientific significance, practical usefulness, and superiority to other methods |
| Excluded claims | An established priority claim for unifying five fields, fastest convergence from LTC, demonstrated hierarchy emergence, conservation of all information, and prevention of representation collapse by a metric alone |

The recorded result consists of **one configuration, its conditions, restricted consequences and experiments, and its relationship to existing work**. Public discussion can evaluate this stated scope without assuming additional guarantees.

## 10. Open verification and extension tasks

### 10.1 Questions for contributors

1. Under what conditions does history-dependent friction help finite-time candidate updates?
2. How can coarse-graining-compatible friction be designed for learned nonlinear potentials?
3. When can friction compatibility and closure of the full flow hold simultaneously?
4. Does the configuration affect prediction accuracy, planning, or computation cost?
5. Which settings make the field formulation operationally relevant, and how should behavior across resolutions be tested?

Contributions may test the stated assumptions, construct counterexamples, reproduce the numerical checks, compare existing controllers, or extend the interfaces. Proposed extensions should identify which R0 definitions or conditions they retain or change and which guarantees require separate verification. A counterexample should state the claim and assumptions it addresses.

### 10.2 Work outside this specification's completion criteria

Establishing additional performance differences, reproving every related result, resolving general nonconvex dynamics, proving hierarchy emergence, running additional large-scale experiments, and preparing a journal submission are not completion criteria for this working specification.

Relevant background for further analysis includes ODE equilibria and stability, differentiation of energy, linear maps, projections, and coarse-graining. These are entry points for contributors rather than mandatory prerequisites for every form of participation.

## 11. Collaboration provenance and handoff invariants

The author initiated the core hypothesis, design choices, direction, and decisions about which configurations to retain, as an open “Ignition” prototype. Detailed formulations, mathematical derivations, literature comparisons, verification code, numerical execution, figures, and document preparation were carried out collaboratively with GPT. Numerical checks were executed; independent verification of all content by the author or a third party is not claimed.

The following invariants define the current handoff:

1. The configuration investigates connections motivated by an open research question; priority and performance competition are not required outcomes.
2. R0 is an internal predictor over hierarchical fields with a fixed encoder, coarse-graining maps, and metrics.
3. World time and internal time are distinct; LTC controls friction in internal updates.
4. Experiments A and B are complete; predictive experiments with a trained world model have not been performed.
5. Literature comparisons are organized within the stated inspection scope. Decisions about publication format and which supporting artifacts to release are outside the R0 mathematical definition.

## 12. Supporting artifact map

The following paths are relative to the root of the source archive. They specify the supporting layout if those artifacts are included in a repository; this Markdown file does not itself bundle them. The archived documents are supporting records, not additional verified claims of this RFC.[^artifacts]

| Artifact | Contents |
|---|---|
| [Source RFC](sources/geometric_liquid_hamiltonian_hjepa_rfc_v01.md) | Types, derivations, convex learning subfamily, horizon weighting, and detailed assumptions |
| [Comparison note v0.2](sources/geometric_liquid_prior_comparison_v01.md) | Mapping to 16 references, equation comparisons, inspection scope, and claim boundaries |
| [Earlier working note v0.7](sources/geometric_liquid_hjepa_ignition_v01.md) | Extended derivations, revision history, and configuration decisions; current status is governed by this RFC and the comparison note |
| [Experiment report HTML](sources/geometric_liquid_experiment_report_v01.html) | Self-contained report with figures for all conditions |
| [Reproduction instructions](experiments/liquid_experiments_v01/README.md) | Execution procedure, complete settings, and interpretation of code and data |
| experiments/liquid_experiments_v01/results/ | CSV files, protocols, all trajectories, numerical checks, and environment information |
| experiments/liquid_experiments_v01/figures/ | PNG and SVG figures |
| provenance.json / SHA256SUMS.txt | Source versions and file-integrity records |

The source archive retains artifacts as retrieved. Earlier documents may describe experiments as pending at the time of writing. Section 7 and the experiment report specify the completed experiment status. Reference numbering is local to each supporting document.

## Notes

[^scope]: A consistent construction means that the equations and interfaces agree under the selected assumptions. It does not establish usefulness of the complete system or validate every possible combination of the components.
[^jepa]: R0 does not reproduce end-to-end training of the referenced H-JEPA. Fixed features do not establish a solution to encoder representation collapse.
[^geometry]: The geometry specified here concerns fixed actions, fixed metrics, and coarse-graining compatibility; it is not a learned-curvature hyperbolic latent space. The experimental cosine-mode subspace is not closed under general translations, so the experiments do not test full-field equivariance.
[^fy]: The FY learning estimates are restricted to a fixed strongly convex generating function. Bibliographic details and source links for this subfamily are retained in Section 5.4, references [9,10,14], of the source RFC. The editorial conversion did not independently reverify those references.
[^experiment]: Diagnostic discrepancies are not rigorous error bounds. Equal internal time does not imply equal computation cost, and a shortest time in a finite scan is not a global optimum. Gate signals also change between layouts. Educational demonstrations preceded the formal sweep; the study was neither externally preregistered nor blinded. The settings are deterministic; the three initial residual values were not treated as IID samples for statistical-significance estimation.
[^literature]: Using a source description for structural comparison is distinct from independently verifying its proofs or experiments. HCFC was inspected through a publisher preview; CoDA through its official abstract; the relevant equations and conditions of iKFAD through arXiv v2. Failure to find an identical configuration does not establish priority.
[^format]: Mathematical expressions and numerical entries are retained from the source record. Math delimiters are adapted for GitHub rendering; table labels and prose are translated. Reference entries use work titles and identifiers without personal author names, contact details, or affiliations. Mathematical eponyms are retained as technical terminology.
[^artifacts]: Linked supporting records have not been rewritten or anonymized as part of this editorial conversion. Inclusion of those records in a public repository requires a separate decision; the links specify their source-archive paths.

## References

References [1]–[16] retain the numbering of the source record and its comparison note. Inspection scope is stated in the table and notes above.

[1] **Liquid Time-constant Networks.** AAAI, 2021. [Source PDF](https://arxiv.org/pdf/2006.04439).

[2] **Friction-adaptive descent: a family of dynamics-based optimization methods.** Journal of Computational Dynamics, 2023. [Source text v3](https://arxiv.org/html/2306.06738v3), DOI:10.3934/jcd.2023007.

[3] **Adaptive Momentum and Nonlinear Damping for Neural Network Training.** ICML, 2026. [Source text v2](https://arxiv.org/html/2602.00334v2), [version history](https://arxiv.org/abs/2602.00334).

[4] **A liquid neural network with physical evolution for variable continuous time series prediction.** Journal of Computational Science 94, 102757, 2026. [Publisher preview](https://www.sciencedirect.com/science/article/pii/S1877750325002340), DOI:10.1016/j.jocs.2025.102757. Inspected through the public preview only.

[5] **H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning.** 2026. [Source text v1](https://arxiv.org/html/2610.06805v1).

[6] **LeHami: Symplectic Latent Dynamics for World Models.** 2026. [Source text v1](https://arxiv.org/html/2610.15267v1).

[7] **End-to-End Learning for Structured Prediction Energy Networks.** ICML, 2017. [Proceedings volume](https://proceedings.mlr.press/v70/).

[8] **Characterization of exact lumpability for vector fields on smooth manifolds.** Differential Geometry and its Applications 48, 46–60, 2016. [Source record](https://arxiv.org/abs/1607.01237), DOI:10.1016/j.difgeo.2016.06.001.

[9] **Structure-preserving tangential interpolation for model reduction of port-Hamiltonian systems.** Automatica 48(9), 1963–1974, 2012. [Source PDF](https://arxiv.org/pdf/1101.3485), DOI:10.1016/j.automatica.2012.05.052.

[10] **Group Equivariant Fourier Neural Operators for Partial Differential Equations.** ICML, 2023. [Proceedings volume](https://proceedings.mlr.press/v202/).

[11] **Hamiltonian Neural PDE Solvers through Functional Approximation.** NeurIPS, 2025. [Source record](https://arxiv.org/abs/2505.13275), [text v2](https://arxiv.org/html/2505.13275v2).

[12] **Physically Native World Models: A Hamiltonian Perspective on Generative World Modeling.** 2026. [Source text v1](https://arxiv.org/html/2605.00412v1).

[13] **An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity.** Neural Computation 29(5), 1229–1262, 2017. [DOI](https://doi.org/10.1162/NECO_a_00949).

[14] **Multiscale Deep Equilibrium Models.** NeurIPS, 2020. [Official abstract](https://papers.neurips.cc/paper_files/paper/2020/hash/3812f9a59b634c2a9c574610eaba5bed-Abstract.html), [paper](https://papers.neurips.cc/paper_files/paper/2020/file/3812f9a59b634c2a9c574610eaba5bed-Paper.pdf).

[15] **Optimal prediction and the Mori–Zwanzig representation of irreversible processes.** PNAS 97(7), 2968–2973, 2000. [DOI](https://doi.org/10.1073/pnas.97.7.2968).

[16] **Efficient Continuous-Depth Modeling with GRU Equivalents.** ICML, PMLR 306, 6194–6209, 2026. [Proceedings volume](https://proceedings.mlr.press/v306/). Only the official abstract was used in the source comparison.
