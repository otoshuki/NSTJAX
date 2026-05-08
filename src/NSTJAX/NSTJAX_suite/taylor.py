"""
Author: gpertin, KAIST
JAX version of Krener's tay_poly, but using a single derivative graph for reuse
"""

import numpy as np
import jax
import jax.numpy as jnp
from math import factorial
from .polylib import factor_lists, monomials

def _afac(n, k):
    E = monomials(n, k)
    af = np.array([np.prod([factorial(int(a)) for a in row]) for row in E])
    return af.astype(np.float64)

def _gather_data(n, d):
    idx = []
    af = []
    for k in range(1, d + 1):
        fl = np.asarray(factor_lists(n, k))
        idx.append(tuple(jnp.asarray(fl[:, t]) for t in range(k)))
        af.append(jnp.asarray(_afac(n, k)))
    return idx, af

def _coeffs_at(fn, z0, d, idx, af):
    blocks = []
    cur = fn
    for k in range(1, d + 1):
        cur = jax.jacfwd(cur)
        Tk = cur(z0)
        gather = Tk[(slice(None),) + idx[k - 1]]
        afk = jnp.asarray(af[k - 1], dtype=gather.dtype)
        blocks.append(gather / afk[None, :])
    return jnp.concatenate(blocks, axis=1)

def build_taylor(fn, n, d):
    #Build JIT compiled map
    idx, af = _gather_data(n, d)
    return jax.jit(lambda z0: _coeffs_at(fn, z0, d, idx, af))


def build_taylor_batch(fn, n, d):
    #Batched across operating points
    idx, af = _gather_data(n, d)
    one = lambda z0: _coeffs_at(fn, z0, d, idx, af)
    return jax.jit(jax.vmap(one))


def precompile(taylor_map, z0):
    return taylor_map.lower(jnp.asarray(z0)).compile()
