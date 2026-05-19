"""
Author: gpertin, KAIST
Test fast FBI solver
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import scipy.linalg as sla
from NSTJAX.NSTJAX_suite.polylib import lie_operator, crd
from NSTJAX.NSTJAX_suite.taylor import build_taylor
from NSTJAX.NSTJAX_suite.fbi import fbi
from NSTJAX.NSTJAX_suite.fbi_fast import (FBIFast, _solve_nilpotent,
                                          _solve_general, _nil_index)

#Reference solve via the original kron operator
def _vec(X):
    return X.T.reshape(-1)

def _unvec(v, r, c):
    return v.reshape(c, r).T

def kron_solve(M_, Sel, LF, RHS):
    nk = LF.shape[0]
    Op = np.kron(np.eye(nk), M_) - np.kron(LF.T, Sel)
    return _unvec(np.linalg.solve(Op, _vec(RHS)), M_.shape[1], nk)

def part_a():
    print("Part A: solver kernels vs kron reference")
    rng = np.random.default_rng(0)
    n, m, p, n_ = 5, 3, 3, 4
    nm = n + m
    #Diagonally loaded so the random operator is well conditioned
    M_ = (rng.standard_normal((nm, nm)) + 3.0 * np.eye(nm)).astype(np.float32)
    Sel = np.zeros((nm, nm), np.float32)
    Sel[:n, :n] = np.eye(n, dtype=np.float32)

    #Nilpotent exosystem, a shift chain
    An = np.zeros((n_, n_), np.float32)
    for i in range(n_ - 1):
        An[i, i + 1] = 1.0
    #General exosystem, random diagonalizable
    Ag = rng.standard_normal((n_, n_)).astype(np.float32)

    for k in (1, 2, 3):
        nk = crd(n_, k)
        RHS = rng.standard_normal((nm, nk)).astype(np.float32)

        LFn = np.asarray(lie_operator(jnp.asarray(An), n_, k), np.float32)
        ref_n = kron_solve(M_, Sel, LFn, RHS)
        Wn = np.asarray(_solve_nilpotent(jnp.asarray(M_), jnp.asarray(Sel),
                                         jnp.asarray(RHS), jnp.asarray(LFn),
                                         _nil_index(LFn)))
        en = np.max(np.abs(Wn - ref_n)) / (np.max(np.abs(ref_n)) + 1e-30)

        LFg = np.asarray(lie_operator(jnp.asarray(Ag), n_, k), np.float32)
        ref_g = kron_solve(M_, Sel, LFg, RHS)
        R, Z = sla.schur(LFg.astype(np.complex64), output="complex")
        Wg = np.asarray(_solve_general(jnp.asarray(M_), jnp.asarray(Sel),
                                       jnp.asarray(RHS), jnp.asarray(Z),
                                       jnp.asarray(R)))
        eg = np.max(np.abs(Wg - ref_g)) / (np.max(np.abs(ref_g)) + 1e-30)
        print(f"  degree {k}: nilpotent rel-err {en:.2e}   general rel-err {eg:.2e}")
    print()

#Drone system, copied from test_fbi_jax
d = 2
g = 9.81
alp = 0.4
n = 10
m = 4
n_ = 4 * 6
p = 4
nsum = n + m + n_
e3 = jnp.array([0.0, 0.0, 1.0])
x0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
u0 = jnp.zeros(m)
x_0 = jnp.zeros(n_)
z0 = jnp.concatenate([x0, u0, x_0])

def fsys(z):
    x = z[:n]
    u = z[n:n + m]
    vsys = x[3:6]
    psys = x[6:9]
    u0sys = u[0]
    om = u[1:4]
    xsysdot = vsys
    vsysdot = -g * e3 + (u0sys * g + g) * psys
    psysdot = jnp.cross(om, psys) - alp * (psys[0]**2 + psys[1]**2 + psys[2]**2 - 1) * psys
    tsysdot = jnp.sum((psys + e3) * om) / (2 * (1 + psys[2]))
    return jnp.concatenate([xsysdot, vsysdot, psysdot, jnp.array([tsysdot])])

def fexo(xb):
    return jnp.concatenate([xb[4:n_], jnp.zeros(4)])

def hsys(z):
    x = z[:n]
    xb = z[n + m:]
    xsys = x[0:3]
    tsys = x[9]
    traj = xb[0:4]
    return jnp.concatenate([traj[0:3] - xsys, jnp.array([traj[3] - 2 * tsys])])

def build_drone():
    tay_f = build_taylor(fsys, nsum, d)
    tay_h = build_taylor(hsys, nsum, d)
    tay_e = build_taylor(fexo, n_, d)
    f = jax.block_until_ready(tay_f(z0))
    h = jax.block_until_ready(tay_h(z0))
    fe = jax.block_until_ready(tay_e(x_0))
    return f, h, fe

def part_b(f, h, fe):
    print("Part B: drone end to end vs fbi")
    th0, la0 = jax.block_until_ready(fbi(f, h, fe, n, m, p, n_, d))
    th0 = jnp.reshape(th0, (n, -1))
    la0 = jnp.reshape(la0, (m, -1))

    solver = FBIFast(n, m, p, n_, d, fixed=True)
    th1, la1, rep1 = solver.solve(f, h, fe)
    th1, la1 = jax.block_until_ready((th1, la1))

    sth = float(jnp.max(jnp.abs(th0))) + 1e-30
    sla = float(jnp.max(jnp.abs(la0))) + 1e-30
    dth = float(jnp.max(jnp.abs(th1 - th0))) / sth
    dla = float(jnp.max(jnp.abs(la1 - la0))) / sla
    print(f"  branch per degree {rep1['branch']}")
    print(f"  rel-theta {dth:.2e}   rel-lambda {dla:.2e}")

    #Warm timing of the default check off path
    reps = 30
    t = time.time()
    for _ in range(reps):
        out = fbi(f, h, fe, n, m, p, n_, d)
    jax.block_until_ready(out)
    t_old = (time.time() - t) / reps
    t = time.time()
    for _ in range(reps):
        out = solver.solve(f, h, fe)
    jax.block_until_ready(out[:2])
    t_new = (time.time() - t) / reps
    print(f"  fbi warm        {t_old*1e3:8.3f} ms")
    print(f"  fbi_fast warm   {t_new*1e3:8.3f} ms")
    print(f"  speedup         {t_old/t_new:8.2f} x")

    #Changing variant and forced general branch both match
    solver_c = FBIFast(n, m, p, n_, d, fixed=False)
    th2, _, _ = solver_c.solve(f, h, fe)
    d2 = float(jnp.max(jnp.abs(jax.block_until_ready(th2) - th0))) / sth
    solver_g = FBIFast(n, m, p, n_, d, fixed=True, branch="general")
    th3, _, rep3 = solver_g.solve(f, h, fe)
    d3 = float(jnp.max(jnp.abs(jax.block_until_ready(th3) - th0))) / sth
    print(f"  fixed=False rel-theta {d2:.2e}")
    print(f"  branch=general rel-theta {d3:.2e}   branch {rep3['branch']}\n")

def part_c(f, h, fe):
    print("Part C: solvability guard")
    solver = FBIFast(n, m, p, n_, d, fixed=True, check="always")
    _, _, rep = solver.solve(f, h, fe)
    print(f"  regime {rep['regime']}   solvable {rep['solvable']}   "
          f"resonant {rep['resonant']}")
    for k, v in rep["degrees"].items():
        print(f"  degree {k}: min-gap {v['min_gap']:.3e}  resonant {v['resonant']}")

    #Cost of the guard relative to the default path
    reps = 30
    off = FBIFast(n, m, p, n_, d, fixed=True, check="off")
    off.solve(f, h, fe)
    t = time.time()
    for _ in range(reps):
        out = off.solve(f, h, fe)
    jax.block_until_ready(out[:2])
    t_off = (time.time() - t) / reps
    t = time.time()
    for _ in range(reps):
        out = solver.solve(f, h, fe)
    jax.block_until_ready(out[:2])
    t_on = (time.time() - t) / reps
    print(f"  check=off    {t_off*1e3:8.3f} ms")
    print(f"  check=always {t_on*1e3:8.3f} ms\n")

def main():
    part_a()
    f, h, fe = build_drone()
    part_b(f, h, fe)
    part_c(f, h, fe)

if __name__ == "__main__":
    main()
