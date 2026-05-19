"""
Author: gpertin, KAIST
Test exosystem spectral decoupling
"""

import numpy as np
from NSTJAX.NSTJAX_suite.polylib import lie_operator, crd
from NSTJAX.NSTJAX_suite.decouple import schur_form, operator_eigs, ExoDecoupling
from scipy.optimize import linear_sum_assignment

def multiset_err(pred, ref):
    #Match two complex spectra and report the worst paired distance
    pred = np.asarray(pred, dtype=np.complex64)
    ref = np.asarray(ref, dtype=np.complex64)
    if pred.shape != ref.shape:
        return np.inf
    D = np.abs(pred[:, None] - ref[None, :])
    r, c = linear_sum_assignment(D)
    return float(D[r, c].max())

def check_case(name, A, n_, degs):
    A = A.astype(np.float32)
    Q, T, mu, nil = schur_form(A)
    rec = float(np.max(np.abs(Q @ T @ Q.T - A)))
    orth = float(np.max(np.abs(Q.T @ Q - np.eye(n_, dtype=np.float32))))
    print(f"[{name}] n_={n_}")
    print(f"  schur reconstruction {rec:.2e}   orthogonality {orth:.2e}   nilpotent {nil}")
    for k in degs:
        pred = operator_eigs(mu, n_, k)
        L = np.asarray(lie_operator(A, n_, k))
        ref = np.linalg.eigvals(L)
        err = multiset_err(pred, ref)
        rad = float(np.max(np.abs(ref)))
        print(f"  degree {k}: monomials {crd(n_, k):5d}   ref-radius {rad:.3e}   abs-err {err:.2e}")
    print()

def main():
    #Real diagonal modes
    Ad = np.diag([0.3, -0.7, 1.1, -0.2])
    check_case("diagonal", Ad, 4, [1, 2, 3])

    #Oscillator blocks, sines and cosines, plus a real mode
    w1, w2 = 1.3, 2.1
    Ao = np.zeros((5, 5))
    Ao[0, 1], Ao[1, 0] = w1, -w1
    Ao[2, 3], Ao[3, 2] = w2, -w2
    Ao[4, 4] = -0.5
    check_case("oscillator", Ao, 5, [1, 2, 3])

    #Nilpotent shift chain, like the drone exosystem
    n_ = 24
    As = np.zeros((n_, n_))
    for i in range(n_ - 4):
        As[i, i + 4] = 1.0
    check_case("nilpotent-shift", As, n_, [1, 2])

    #Fixed vs changing option
    dec = ExoDecoupling(fixed=True)
    same = dec.analyze(Ao) is dec.analyze(Ad)
    print(f"[option] fixed=True reuses first factorization: {same}")
    dec2 = ExoDecoupling(fixed=False)
    q1, q2 = dec2.analyze(Ao), dec2.analyze(Ad)
    print(f"[option] fixed=False recomputes: {not np.array_equal(q1['T'], q2['T'])}")

if __name__ == "__main__":
    main()
