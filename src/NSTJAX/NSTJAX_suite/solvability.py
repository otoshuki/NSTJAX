"""
Author: gpertin, KAIST
FBI solvability screen via transmission zeros
"""

import numpy as np
import scipy.linalg as sla
from NSTJAX.NSTJAX_suite.decouple import operator_eigs

#Threshold below which a finite pencil eigenvalue counts as a zero coincidence
RES_TOL = 1e-3
#Threshold on beta for an eigenvalue to count as finite
FIN_TOL = 1e-6

def transmission_zeros(M_, Sel):
    #Finite generalized eigenvalues of the pencil (M_, Sel)
    M_ = np.asarray(M_, dtype=np.float64)
    Sel = np.asarray(Sel, dtype=np.float64)
    if M_.shape[0] != M_.shape[1]:
        return np.zeros(0, dtype=np.complex64), False
    alpha, beta = sla.eig(M_, Sel, homogeneous_eigvals=True)[0]
    scale = float(np.max(np.abs(Sel))) + 1e-30
    fin = np.abs(beta) > FIN_TOL * scale
    zeros = (alpha[fin] / beta[fin]).astype(np.complex64)
    return zeros, True

def screen(M_, Sel, mu, n_, d, p, m):
    #Per degree resonance report against the transmission zeros
    report = {}
    #Structural feasibility from the control vs error channel count
    report["regime"] = ("overdetermined" if p > m else
                        "underdetermined" if m > p else "square")
    zeros, square = transmission_zeros(M_, Sel)
    report["zeros"] = zeros
    zscale = float(np.max(np.abs(zeros))) + 1.0 if zeros.size else 1.0
    degrees = {}
    for k in range(1, d + 1):
        lam = operator_eigs(mu, n_, k)
        if zeros.size and square:
            gap = np.min(np.abs(lam[:, None] - zeros[None, :]), axis=1)
        else:
            gap = np.full(lam.shape[0], np.inf, dtype=np.float64)
        mn = float(np.min(gap)) / zscale
        worst = int(np.argmin(gap))
        degrees[k] = {"min_gap": mn,
                      "resonant": bool(mn < RES_TOL),
                      "worst_lambda": complex(lam[worst])}
    report["degrees"] = degrees
    report["resonant"] = any(v["resonant"] for v in degrees.values())
    return report
