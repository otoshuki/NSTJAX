"""
Author: gpertin, KAIST
Exosystem spectral decoupling for FBI
"""

import numpy as np
import scipy.linalg as sla
from itertools import combinations_with_replacement

#Relative tolerance for nilpotent detection
NIL_TOL = 1e-5

def schur_form(A):
    #Real schur factorization A = Q T Q^T of the exosystem linear part
    A = np.asarray(A, dtype=np.float32)
    T, Q = sla.schur(A, output="real")
    mu = np.linalg.eigvals(A).astype(np.complex64)
    scale = float(np.max(np.abs(A))) + 1e-30
    nil = bool(np.max(np.abs(mu)) < NIL_TOL * scale)
    return Q.astype(np.float32), T.astype(np.float32), mu, nil

def operator_eigs(mu, n_, k):
    #Eigenvalues of the degree k lie operator, sum_i alpha_i mu_i
    out = []
    for combo in combinations_with_replacement(range(n_), k):
        alpha = np.bincount(combo, minlength=n_)
        out.append(complex(np.dot(alpha, mu)))
    return np.asarray(out, dtype=np.complex64)

class ExoDecoupling:
    #Spectral analysis of the exosystem, fixed reuses the factorization
    def __init__(self, fixed=True):
        self.fixed = fixed
        self._cache = None

    def analyze(self, A):
        if self.fixed and self._cache is not None:
            return self._cache
        Q, T, mu, nil = schur_form(A)
        res = {"Q": Q, "T": T, "mu": mu, "nilpotent": nil}
        if self.fixed:
            self._cache = res
        return res
