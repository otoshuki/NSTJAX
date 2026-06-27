<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![JAX](https://img.shields.io/badge/JAX-white?style=for-the-badge&logo=Google%20JAX&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg?style=for-the-badge)](https://opensource.org/licenses/BSD-3-Clause)

</div>

# A JAX based independent reimplementation of the control suite from Nonlinear Systems Toolbox by Professor A.J. Krener

Please note that this is not intended to copy the original MATLAB suite one-to-one. The original toolbox can be obtained from https://www.math.ucdavis.edu/~krener/nst08.zip

I intend to implement the HJB and FBI algorithms and the necessary helper functions for the same. Currently only **FBIJAX** is implemented.

The entire project is divided into two parts:

1. **NSTJAX_suite**: Reimplementation of Al'Brekht's approach based on Krener's designs in JAX.
2. **NSTJAX_bridge**: A bridging component to convert computed matrices between the original NST in MATLAB and NSTJAX.

To use the library please install using

```bash
pip install -e .
```

---

## What this solves
Given a controlled plant, an exosystem that generates the reference signal, and a tracking error, FBIJAX computes a **nominal tracking manifold** and a **feedforward** that renders that manifold invariant, inspired from Krener's work. It returns two polynomial maps in the exosystem state `x_`:

- `theta` -> the steady state the plant should sit on to track the reference,
- `lambda` -> the feedforward that holds the plant on that manifold.

This solution is a **local polynomial approximation** built from a Taylor expansion around the current operating point. It is cheap enough to recompute at every step (about **12 ms** for the full drone system with a six derivative exosystem), so the manifold can be refreshed as the operating point moves and the approximation stays accurate everywhere along the trajectory rather than only near a single linearization.

<p align="center">
  <img src="docs/figures/drone_nominal_trajectory.png" width="840" alt="Nominal drone trajectory and feedforward">
</p>

<p align="center"><em>Nominal flight path produced by rolling the exosystem through the tracking manifold theta, with the feedforward lambda along the trajectory.</em></p>

Both continuous and discrete time plants are supported. For the discrete case (`disc=True`), the manifold is held invariant by a composition $\theta(\bar{f}(\bar{x}))$ instead of a derivative, and the per degree operator spectrum becomes products of the exosystem eigenvalues rather than sums.

---

## Quickstart

The high level `NSTJAX` object wraps map building, solver selection and inference. Define the plant, the exosystem and the error map, warm start once, then solve at any operating point.

```python
import jax.numpy as jnp
from NSTJAX.NSTJAX_suite.nstjax import NSTJAX

#Dimensions: states n, controls m, errors p, exosystem n_, expansion degree d
n, m, p, n_, d = 10, 4, 24, 4, 2
nsum = n + m + n_

#fsys and hsys take z = concat([x, u, x_]); fexo takes the exosystem state w
def fsys(z): ...
def fexo(x_): ...
def hsys(z): ...

#Operating point
z0 = jnp.zeros(nsum)
x_0 = jnp.zeros(n_)

nst = NSTJAX(fsys, fexo, hsys, n, m, p, n_, d, verbose=True)
nst.warm_start(z0, x_0, samples=256)

#Solve at a moving operating point
th, la = nst.compute_fbi(z0, x_0)

#Evaluate the manifold and feedforward at a batch of exosystem samples
W = jnp.zeros((256, n_))
tracking_manifold = nst.compute_theta(W) #(samples x n)
feedforward = nst.compute_lambda(W)      #(samples x m)
```

The auto solver routes small systems to the dense solve and large ones to the decoupled solve. The first call compiles, later calls reuse the cached kernels.

---

## Performance

After warmup the full step, coefficients, solve and inference together, runs at about 12 ms for the drone system with its six derivative exosystem (10 states, 24 exostates, 4 inputs, 4 outputs). On the smaller pendulum system (2 states, 3 exostates, 1 input, 1 output), it takes approx 0.5ms. This is performed on a laptop with Ryzen 7 8845hs CPU, 32GB RAM, and NVIDIA GeForce RTX 4060 Laptop with 8GB VRAM.

<p align="center">
  <img src="docs/figures/timing_breakdown2.png" width="840" alt="Per step timing breakdown and inference throughput for pendulum system">
  <img src="docs/figures/timing_breakdown.png" width="840" alt="Per step timing breakdown and inference throughput for drone system">
</p>

<p align="center">
  <em>Per step cost split into coefficients, FBI solve and inference, with inference throughput against batch size. (a) Pendulum, (b) Drone</em>
</p>

---

## Examples

The repository ships runnable examples for a drone and a controlled pendulum, at the high level `NSTJAX` interface and a lower level direct solve, plus a world model example that learns the plant online and re-solves FBI on it.

<p align="center">
  <img src="docs/figures/pendulum_phase.png" width="840" alt="Pendulum nominal manifold and tracking">
</p>

<p align="center"><em>Pendulum: the nominal manifold phase portrait, and the reference tracked on the manifold.</em></p>

### World model learning with FBI

`example_wm_fbi.py` closes the loop around a learned plant. The true pendulum is unknown; the controller starts from a deliberately wrong nominal prior and corrects it with a neural residual, a neural-ODE world model `f_hat(x, u) = f_nominal(x, u) + MLP(x, u)`. Each round runs the same three steps:

1. **Solve.** Taylor expand the current `f_hat` at the operating point and run the FBI solve to get the manifold `theta` and feedforward `lambda` for the learned model, plus a feedback gain by pole placement on its linearization.
2. **Roll out.** Drive the true system in closed loop with feedforward plus feedback plus a decaying dither, and collect the on-policy transitions.
3. **Learn.** Aggregate the transitions across rounds and fit the residual MLP by matching a one step RK4 prediction to the next state.

As the world model improves, the FBI solution computed on it approaches the one computed on the true plant, so the learned manifold and feedforward converge to the oracle and the closed loop tracking error drops to the oracle floor.

<p align="center">
  <img src="docs/figures/world_model_convergence.png" width="840" alt="World model FBI convergence">
</p>

<p align="center"><em>World model learning: the learned manifold and feedforward errors against the true FBI solution, and the closed loop tracking error converging to the oracle bound.</em></p>

### Manifold degree comparison for swinging payload tracking

`example_swing_drone.py` solves a cable suspended payload problem. We can see the difference in performance between degrees 1, 2, 3, and 4. The higher degree manifolds capture more of the swing nonlinearity, so the payload rides closer to the reference.

<p align="center">
  <img src="docs/figures/payload_degree_compare.gif" width="560" alt="Payload tracking at manifold degree 1, 2 and 3">
</p>

<p align="center"><em>Reference load against the closed loop payload path at degree 1, 2 and 3.</em></p>

<p align="center">
  <img src="docs/figures/payload_degree_error.png" width="560" alt="Payload tracking error by manifold degree">
</p>

<p align="center"><em>Payload tracking error norm over one reference period for each manifold degree.</em></p>

---

## Solution Validation Against Original NST

The solve is cross checked against the original MATLAB NST values when reference arrays are present (`test_fbi_jax.py`), and against the dense Kronecker reference for the decoupled kernels (`test_fbi_fast.py`). The FBI residual measures how well the truncated polynomial solution satisfies the regulator equations at the operating point.

<p align="center">
  <img src="docs/figures/matlab_error.png" width="840" alt="Relative error against the MATLAB reference">
</p>

<p align="center"><em>Per coefficient relative error of the float32 solve against the MATLAB NST reference.</em></p>

---

# Additional Information

## Background

The library targets the **output regulation** problem. A plant, an exosystem driving the reference, and an error output

$$\dot{x} = f(x, u, \bar{x}), \qquad \dot{\bar{x}} = \bar{f}(\bar{x}), \qquad e = h(x, u, \bar{x})$$

are given, and the goal is to drive the error to zero. The Francis Byrnes Isidori (FBI) regulator equations ask for a manifold $x=\theta(\bar{x})$ and a feedforward $u=\lambda(\bar{x})$ satisfying

$$f(\theta(\bar{x}),\lambda(\bar{x})) = \frac{\partial \theta}{\partial \bar{x}}(\bar{x})\bar{f}(\bar{x}) \qquad h(\theta(\bar{x}),\bar{x},\lambda(\bar{x})) = 0.$$

The first equation makes the manifold invariant under the combined flow, the second makes the output vanish on it.

The same machinery solves the **discrete time** problem, a plant $x^+ = f(x,u,\bar{x})$ driven by a discrete exosystem $\bar{x}^+ = \bar{f}(\bar{x})$. Here invariance is a composition rather than a derivative: stepping the manifold forward must land on the manifold at the next exosystem state,

$$f(\theta(\bar{x}),\lambda(\bar{x}),\bar{x}) = \theta(\bar{f}(\bar{x})), \qquad h(\theta(\bar{x}),\bar{x},\lambda(\bar{x})) = 0.$$

The degree by degree solve is unchanged in structure. The left operator now uses the composition

Following NST, the FBIJAX component solves these equations by expanding `theta` and `lambda` as truncated polynomial series in `x_` and matching coefficients degree by degree. Degree one is the linear regulator (a Sylvester type system); each higher degree reuses the same left operator with a right hand side assembled from the lower degree coefficients. Because the expansion is taken around an operating point, the result is a local model, valid in a neighborhood. Recomputing it at each operating point keeps that neighborhood centered on the current state.

<p align="center">
  <img src="docs/figures/residual_vs_distance_truefbi.png" width="48%" alt="True FBI residual versus distance from the operating point">
  <img src="docs/figures/residual_vs_distance.png" width="48%" alt="FBI residual versus distance from the operating point">

</p>
<p align="center">
  <em>(a) FBI solution (b) Fast FBI solution. A stale solve degrades as the state moves away, while re-solving at each point holds the error down.</em>
</p>

---

## The solve pipeline

Each operating point passes through three stages:

1. **Taylor maps** (`taylor.py`) build JIT compiled coefficient maps for `f`, `h` and the exosystem, evaluated by forward mode differentiation. The maps are built once and reused; each call returns the packed graded coefficients of the field at the current operating point.
2. **FBI solve** (`fbi.py` or `fbi_fast.py`) solves the regulator equations degree by degree and returns the packed `theta` and `lambda` coefficients. Degree one is the linear regulator; each higher degree reuses the same left operator with a right hand side assembled from the lower degree coefficients through the polynomial algebra in `polylib.py`.
3. **Inference** (`fbi_eval.py`) evaluates the manifold `theta(x_)` and the feedforward `lambda(x_)` on a batch of exosystem samples, reusing the cached evaluation kernel.

After the first compiling call the warm path is the relevant timing, and the whole pipeline is cheap enough to refresh at every operating point in a moving loop.

<p align="center">
  <img src="docs/figures/pipeline.svg" width="880" alt="FBIJAX solve pipeline">
</p>

<p align="center"><em>The solve pipeline: Taylor maps, the degree by degree FBI solve, and batched inference, refreshed at each operating point.</em></p>

---

## Dense `fbi` versus decoupled `fbi_fast`

At each degree `k` the regulator equation has the operator form

$$M\, W - \mathrm{Sel}\, W\, L^{(k)} = R,$$

where `M` is the combined linear plant and output Jacobian, `Sel` selects the state block, and `L^{(k)}` is the degree `k` Lie operator of the exosystem linear part. The unknown `W` stacks the degree `k` coefficients of `theta` and `lambda`.

- **`fbi`** vectorizes this into the dense Kronecker operator `kron(I, M) - kron(L^T, Sel)` and solves it directly, square systems through a direct solve and non square ones through least squares. The operator has size `(n + p) * crd(n_, d)`, which grows quickly with the system dimension and the degree.

- **`fbi_fast`** decouples the equation through the Schur form of the Lie operator and solves per degree, with two branches:
  - **nilpotent** exosystems (integrator chains, such as polynomial trajectory generator) use a finite Neumann sum that is exact in a small number of terms,
  - **general** exosystems use complex Schur back substitution, one small `(n + m)` solve per column instead of a single large solve.

  With `fixed=True` the exosystem factorization is computed once and reused across operating points, which is the common case in a moving loop. `NSTJAX` routes automatically based on the Kronecker dimension via `auto_threshold`, sending small systems to the dense solve and large ones to the decoupled solve.

<p align="center">
  <img src="docs/figures/solver_crossover.png" width="560" alt="Dense versus decoupled solve time">
</p>

<p align="center"><em>Warm solve time against problem size on a nilpotent family. The dense Kronecker solve overtakes the decoupled solve as the exosystem grows.</em></p>

---

## Solvability screen

The Lie operator eigenvalues at degree `k` are the degree `k` sums of the exosystem eigenvalues (products, in the discrete case). The regulator equations are solvable when none of these coincide with a **transmission zero** of the `(M, Sel)` pencil. `solvability.py` and the `check` option in `fbi_fast` report, per degree, the gap between the operator spectrum and the transmission zeros, flagging resonant degrees and the structural regime (square, over or underdetermined). An overdetermined regime, where the error channel count exceeds the control count, is flagged as structurally infeasible; a resonant degree, where the gap falls below tolerance, signals a coincidence that makes the per degree solve singular.

<p align="center">
  <img src="docs/figures/spectral_screen.png" width="560" alt="Solvability screen, spectrum versus transmission zeros">
</p>

<p align="center"><em>Solvability screen: degree wise operator eigenvalues against the transmission zeros, annotated with the per degree gap.</em></p>

---

## References

- A. Isidori and C. I. Byrnes, output regulation of nonlinear systems.
- B. A. Francis, the linear multivariable regulator problem.
- A. J. Krener, Nonlinear Systems Toolbox.

---

## Citation
Author: otoshuki (gpertin), KAIST.
