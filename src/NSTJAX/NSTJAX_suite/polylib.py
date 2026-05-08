"""
Author: gpertin, KAIST
Polynomial vector field algebra library
"""

import numpy as np
import jax
import jax.numpy as jnp
from itertools import combinations_with_replacement
from functools import lru_cache, partial
from math import comb

#Combinatorial functions
def crd(n, k):
    #Follows Krener's design
    #Number of degree k monomials in n variables
    if k == 0:
        return 1
    return comb(n + k - 1, k)

def crdsum(n, d0, d1):
    #Follows Krener's design
    #Number of monomials of degrees d0 through d1 in n variables
    return comb(n + d1, d1) - comb(n + d0 - 1, d0 - 1)

@lru_cache(maxsize=None)
def factor_lists(n, k):
    #Sorted index tuples of length k over 0..n-1 in lex order
    if k == 0:
        return ((),)
    return tuple(combinations_with_replacement(range(n), k))

@lru_cache(maxsize=None)
def monomials(n, k):
    #Exponent table of degree k monomials, shape (crd(n,k), n), lex order
    E = np.zeros((crd(n, k), n), dtype=np.int64)
    for r, fl in enumerate(factor_lists(n, k)):
        for idx in fl:
            E[r, idx] += 1
    return E
