"""
Author: gpertin, KAIST
Evaluates tracking manifold and feedforward
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from NSTJAX.NSTJAX_suite.polylib import eval_monomials, crd

def mon(x, d0, d1):
    #All monomials of x of degrees d0..d1 in lex order, single sample
    n = x.shape[0]
    return eval_monomials(x, n, d0, d1)

@partial(jax.jit, static_argnums=(1, 2))
def mon_batched(X, d0, d1):
    #Monomials for a batch X of shape (num_samples, n)
    return jax.vmap(lambda xi: mon(xi, d0, d1))(X)

@partial(jax.jit, static_argnums=(2,))
def compute_manifold(MAT, X, d):
    #Evaluate a packed degrees 1..d field MAT of shape (nout, ncols) over a batch
    #X has shape (num_samples, n_), returns (num_samples, nout)
    Mb = mon_batched(X, 1, d)
    return Mb @ MAT.T

def compute_theta(Th, X, d):
    #Tracking manifold values, shape (num_samples, n)
    return compute_manifold(Th, X, d)

def compute_lambda(La, X, d, uref=0.0):
    #Feedforward control values, shape (num_samples, m)
    return compute_manifold(La, X, d) + uref
