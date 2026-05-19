"""
Author: gpertin, KAIST
Test FBI solvability screen
"""

import numpy as np
import jax.numpy as jnp
from NSTJAX.NSTJAX_suite.decouple import schur_form
from NSTJAX.NSTJAX_suite.solvability import transmission_zeros, screen

def build_M_Sel(F1, H1, n, m, p):
    #Same blocks fbi uses for the degree 1 operator
    dt = np.float32
    M_ = np.block([[F1[:, :n], F1[:, n:n + m]],
                   [H1[:, :n], H1[:, n:n + m]]])
    Sel = np.block([[np.eye(n, dtype=dt), np.zeros((n, m), dt)],
                    [np.zeros((p, n + m), dt)]])
    return M_.astype(dt), Sel.astype(dt)

def show(name, rep):
    print(f"[{name}] regime={rep['regime']}  finite-zeros={rep['zeros'].shape[0]}  "
          f"resonant={rep['resonant']}")
    for k, v in rep["degrees"].items():
        print(f"  degree {k}: min-gap {v['min_gap']:.3e}  resonant {v['resonant']}  "
              f"worst-lambda {v['worst_lambda']:.3f}")
    print()

def main():
    rng = np.random.default_rng(0)
    n, m, p = 3, 2, 2
    #Healthy square case, random stable plant, oscillator exosystem
    F1 = rng.standard_normal((n, n + m)).astype(np.float32)
    H1 = rng.standard_normal((p, n + m)).astype(np.float32)
    M_, Sel = build_M_Sel(F1, H1, n, m, p)
    z, _ = transmission_zeros(M_, Sel)
    print(f"random plant: {z.shape[0]} finite transmission zeros\n")

    n_ = 4
    w = 1.7
    Ao = np.zeros((n_, n_), np.float32)
    Ao[0, 1], Ao[1, 0] = w, -w
    Ao[2, 3], Ao[3, 2] = 0.9, -0.9
    _, _, mu, _ = schur_form(Ao)
    show("healthy", screen(M_, Sel, mu, n_, 3, p, m))

    #Forced resonance, place an exosystem mode on a real transmission zero
    z_real = float(np.real(z[np.argmin(np.abs(z.imag))])) if z.size else 1.0
    Ar = np.diag([z_real, -0.4, 0.6, -0.9]).astype(np.float32)
    _, _, mur, _ = schur_form(Ar)
    show("degree1-resonant", screen(M_, Sel, mur, n_, 3, p, m))

    #Higher degree resonance, half the zero so 2*mu lands on it
    Ah = np.diag([z_real / 2.0, -0.31, 0.27, -0.55]).astype(np.float32)
    _, _, muh, _ = schur_form(Ah)
    show("degree2-resonant", screen(M_, Sel, muh, n_, 3, p, m))

    #Overdetermined, more error channels than controls
    p2 = 3
    H1b = rng.standard_normal((p2, n + m)).astype(np.float32)
    M2 = np.block([[F1[:, :n], F1[:, n:n + m]],
                   [H1b[:, :n], H1b[:, n:n + m]]]).astype(np.float32)
    Sel2 = np.block([[np.eye(n, dtype=np.float32), np.zeros((n, m), np.float32)],
                     [np.zeros((p2, n + m), np.float32)]]).astype(np.float32)
    show("overdetermined", screen(M2, Sel2, mu, n_, 3, p2, m))

if __name__ == "__main__":
    main()
