# A JAX based independent reimplementation of the control suite from Nonlinear Systems Toolbox by Professor A.J. Krener

Please note that this is not intended to copy the original MATLAB suite one-to-one. The original toolbox can be obtained from https://www.math.ucdavis.edu/~krener/nst08.zip

I intend to implement the HJB and FBI algorithms and the necessary helper functions for the same. Currently only FBIJAX is implemented.

The entire project is divided into two parts:
1. NSTJAX_suite: Reimplementation of Al' Brekth's approach based on Krener's designs in JAX.
2. NSTJAX_bridge: A bridging component to convert computed matrices between original NST in MATLAB and NSTJAX.

To use the library please install using
```bash
pip install -e .
