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

---

## Quickstart

The high level `NSTJAX` object wraps map building, solver selection and inference. Define the plant, the exosystem and the error map, warm start once, then solve at any operating point.

```python
import jax.numpy as jnp
from NSTJAX.NSTJAX_suite.nstjax import NSTJAX

#Dimensions: states n, controls m, errors p, exosystem n_, expansion degree d
n, m, p, n_, d = 10, 4, 24, 4, 2
nsum = n + m + n_

#fsys and hsys take z = concat([x, u, w]); fexo takes the exosystem state w
def fsys(z): ...
def fexo(w): ...
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
pi = nst.compute_theta(W)
c = nst.compute_lambda(W)
```

The auto solver routes small systems to the dense solve and large ones to the decoupled solve. The first call compiles, later calls reuse the cached kernels.

## Background

The library targets the **output regulation** problem. A plant, an exosystem driving the reference, and an error output

$$\dot{x} = f(x, u, x_-), \qquad \dot{x}_- = s(x_-), \qquad e = h(x, u, x_-)$$

are given, and the goal is to drive the error to zero. The Francis Byrnes Isidori (FBI) regulator equations ask for a manifold `x = theta(x_)` and a feedforward `u = lambda(x_)` satisfying

$$\frac{\partial\, \theta}{\partial x_-}\, s(x_-) = f\big(\theta(x_-), \lambda(x_-), x_-\big), \qquad 0 = h\big(\theta(x_-), \lambda(x_-), x_-\big).$$

The first equation makes the manifold invariant under the combined flow, the second makes the error vanish on it. On this manifold the plant reproduces the reference exactly; the feedforward is what keeps it there.

FBIJAX solves these equations by expanding `theta` and `lambda` as truncated polynomial series in `x_` and matching coefficients degree by degree. Degree one is the linear regulator (a Sylvester type system); each higher degree reuses the same left operator with a right hand side assembled from the lower degree coefficients. Because the expansion is taken around an operating point, the result is a local model, valid in a neighborhood. Recomputing it at each operating point keeps that neighborhood centered on the current state.

## The solve pipeline

Each operating point passes through three stages:

1. **Taylor maps** (`taylor.py`) build JIT compiled coefficient maps for `f`, `h` and the exosystem, evaluated by forward mode differentiation. The maps are built once and reused; each call returns the packed graded coefficients of the field at the current operating point.
2. **FBI solve** (`fbi.py` or `fbi_fast.py`) solves the regulator equations degree by degree and returns the packed `theta` and `lambda` coefficients. Degree one is the linear regulator; each higher degree reuses the same left operator with a right hand side assembled from the lower degree coefficients through the polynomial algebra in `polylib.py`.
3. **Inference** (`fbi_eval.py`) evaluates the manifold `theta(x_)` and the feedforward `lambda(x_)` on a batch of exosystem samples, reusing the cached evaluation kernel.

After the first compiling call the warm path is the relevant timing, and the whole pipeline is cheap enough to refresh at every operating point in a moving loop.

---

## References

- A. Isidori and C. I. Byrnes, output regulation of nonlinear systems.
- B. A. Francis, the linear multivariable regulator problem.
- A. J. Krener, Nonlinear Systems Toolbox.

## Citation
Author: otoshuki (gpertin), KAIST.
