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

## References

- A. Isidori and C. I. Byrnes, output regulation of nonlinear systems.
- B. A. Francis, the linear multivariable regulator problem.
- A. J. Krener, Nonlinear Systems Toolbox.

## Citation
Author: otoshuki (gpertin), KAIST.
