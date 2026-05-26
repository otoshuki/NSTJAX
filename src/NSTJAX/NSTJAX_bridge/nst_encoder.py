"""
Author: gpertin, KAIST
Encodes new coefficient arrangement to MATLAB format
"""

import numpy as np
import jax
import jax.numpy as jnp
from NSTJAX.NSTJAX_suite.polylib import monomials, crd

def reduced_perm(svlth, k):
    #Permutation taking lex order to the MATLAB reduced order at degree k
    #Follows Krner's redmonomials
    svlth = [int(s) for s in svlth if s > 0]
    ns = sum(svlth)
    E = monomials(ns, k)
    if E.shape[0] <= 1:
        return np.arange(E.shape[0])
    cols = []
    c = 0
    for li in svlth:
        block = E[:, c:c + li]
        cols.append(-block.sum(axis=1, keepdims=True))
        cols.append(-block)
        c += li
    key = np.concatenate(cols, axis=1)
    return np.lexsort(key.T[::-1])

def encode_to_reduced(mat, n_, d, svlth=None):
    #Encode to Krener's reduced format
    if svlth is None:
        svlth = [n_]
    out = []
    c = 0
    for k in range(1, d + 1):
        w = crd(n_, k)
        block = mat[:, c:c + w]
        ordr = reduced_perm(svlth, k)
        out.append(block[:, jnp.asarray(ordr)])
        c += w
    return jnp.concatenate(out, axis=1)
