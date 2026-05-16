"""
Author: gpertin, KAIST
Test JAX version of FBI
"""

import os
import time
from math import comb
from itertools import combinations_with_replacement
import numpy as np
import jax
import jax.numpy as jnp
import NSTJAX.NSTJAX_suite.polylib as P
from NSTJAX.NSTJAX_suite.fbi import fbi
# from NSTJAX.NSTJAX_suite.fbi_fast import FBIFast
from NSTJAX.NSTJAX_suite.taylor import build_taylor, build_taylor_batch, precompile
from NSTJAX.NSTJAX_suite.fbi_eval import compute_theta, compute_lambda

#Test params
SAMPLES = 256         #Inference samples per iteration
ITERS = 10            #Timed simulation steps
BURN = 1              #Steps discarded before averaging (steady state only)
SPREAD = 0.2          #How far the operating point wanders each step
W_SPREAD = 0.2        #Spread of the exosystem inference samples
WARM_REPS = 20        #Amortized warm-call repetitions

#System Design: Drone tracking problem
d = 2
g = 9.81
alp = 0.4
n = 10
m = 4
n_ = 4 * 6
p = 4
nsum = n + m + n_
e3 = jnp.array([0.0, 0.0, 1.0])
#Operating point
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

#FBI residual error
def fbi_residual(th, la, f, h, fe):
    fh = jnp.concatenate([f, h], axis=0)
    fhF = P.unpack(fh, nsum, 1, d)
    thF = P.unpack(th, n_, 1, d)
    laF = P.unpack(la, n_, 1, d)
    feF = P.unpack(fe, n_, 1, d)
    gfield = P.zero_field(nsum, n_, d, dtype=f.dtype)
    for deg in range(1, d + 1):
        gfield[deg] = gfield[deg].at[:n].set(thF[deg])
        gfield[deg] = gfield[deg].at[n:n + m].set(laF[deg])
    gfield[1] = gfield[1].at[n + m:nsum].set(jnp.eye(n_, dtype=f.dtype))
    comp = P.compose(fhF, nsum, gfield, n_, d)
    deriv = P.ddmul(thF, feF, n_, d)
    worst = 0.0
    for k in range(1, d + 1):
        worst = max(worst,
                    float(jnp.max(jnp.abs(comp[k][:n] - deriv[k]))),
                    float(jnp.max(jnp.abs(comp[k][n:]))))
    return worst

#Inference: evaluate the packed FBI fields at a batch of exosystem samples
def _monomial_layout(n_in, d, ncols):
    counts = [comb(n_in + k - 1, k) for k in range(0, d + 1)]
    for lo in range(0, d + 1):
        if sum(counts[lo:d + 1]) == ncols:
            return list(range(lo, d + 1))
    raise ValueError(
        f"cannot match ncols={ncols} to a monomial layout for n_in={n_in}, d={d} "
        f"(degree counts {counts})")

def _index_table(n_in, degs, d):
    rows = []
    for k in degs:
        if k == 0:
            rows.append([-1] * d)
        else:
            for combo in combinations_with_replacement(range(n_in), k):
                rows.append(list(combo) + [-1] * (d - k))
    return np.asarray(rows, dtype=np.int64)

def make_inference(n_in, d, ncols_th, ncols_la):
    idx_t = jnp.asarray(_index_table(n_in, _monomial_layout(n_in, d, ncols_th), d))
    idx_l = jnp.asarray(_index_table(n_in, _monomial_layout(n_in, d, ncols_la), d))
    def monomials(W, idx):
        safe = jnp.clip(idx, 0, None)
        gg = W[:, safe]
        gg = jnp.where(idx[None] >= 0, gg, 1.0)
        return jnp.prod(gg, axis=-1)
    @jax.jit
    def infer(th, la, W):
        pi = monomials(W, idx_t) @ th.T
        c = monomials(W, idx_l) @ la.T
        return pi, c
    return infer

def main():
    print(f"dims: n={n} m={m} p={p} n_={n_} d={d} nsum={nsum}\n")
    #Build reusable taylor maps
    tay_f = build_taylor(fsys, nsum, d)
    tay_h = build_taylor(hsys, nsum, d)
    tay_e = build_taylor(fexo, n_, d)

    #Coefficients at the base point and initial compilation
    t = time.time()
    f = jax.block_until_ready(tay_f(z0))
    h = jax.block_until_ready(tay_h(z0))
    fe = jax.block_until_ready(tay_e(x_0))
    t_map_compile = time.time() - t
    print(f"f shape {f.shape}   h shape {h.shape}   fe shape {fe.shape}")

    #First FBI solve for compilation
    t = time.time()
    th, la = jax.block_until_ready(fbi(f, h, fe, n, m, p, n_, d))
    # solver = FBIFast(n, m, p, n_, d, fixed=True)
    # th, la = jax.block_until_ready(solver.solve(f, h, fe))
    t_compile = time.time() - t
    th = jnp.reshape(th, (n, -1))
    la = jnp.reshape(la, (m, -1))
    print(f"theta shape {th.shape}   lambda shape {la.shape}")
    print(f"regulator residual ({th.dtype}) = {fbi_residual(th, la, f, h, fe):.3e}\n")

    #MATLAB comparison
    if (os.path.exists("x_test_batch.npy") and os.path.exists("th_val_mat.npy")
            and os.path.exists("la_val_mat.npy")):
        Xm = jnp.asarray(np.load("x_test_batch.npy"))
        th_ref = jnp.asarray(np.load("th_val_mat.npy"))
        la_ref = jnp.asarray(np.load("la_val_mat.npy"))
        sth = float(jnp.max(jnp.abs(th_ref))) + 1e-30
        sla = float(jnp.max(jnp.abs(la_ref))) + 1e-30
        def mat_rel(thc, lac):
            thc = jnp.reshape(thc, (n, -1)); lac = jnp.reshape(lac, (m, -1))
            tv = compute_theta(thc, Xm, d); lv = compute_lambda(lac, Xm, d)
            return (float(jnp.max(jnp.abs(tv - th_ref))) / sth,
                    float(jnp.max(jnp.abs(lv - la_ref))) / sla)
        print(f"MATLAB comparison, {Xm.shape[0]} samples (reference = MATLAB evaluated values)")
        dth, dla = mat_rel(th, la)
        print(f"  f32            rel-theta {dth:.2e}   rel-lambda {dla:.2e}")
        print()
    else:
        print("MATLAB reference values not found, skipping comparison\n")

    #Amortized timing: first call compiles, later calls reuse
    t = time.time()
    for _ in range(WARM_REPS):
        out = fbi(f, h, fe, n, m, p, n_, d)
        # out = solver.solve(f, h, fe)
    jax.block_until_ready(out)
    t_warm = (time.time() - t) / WARM_REPS
    print("timing")
    print(f"  maps compile (1st call) {t_map_compile*1e3:9.2f} ms")
    print(f"  fbi compile  (1st call) {t_compile*1e3:9.2f} ms")
    print(f"  fbi warm (cached)       {t_warm*1e3:9.2f} ms\n")

    #Batched evaluation
    X = jax.random.normal(jax.random.PRNGKey(0), (256, n_))
    compute_theta(th, X, d); compute_lambda(la, X, d)          # warm jit
    t = time.time()
    theta_feat = compute_theta(th, X, d)
    lam_feat = compute_lambda(la, X, d)
    jax.block_until_ready((theta_feat, lam_feat))
    t_infer = time.time() - t
    print(f"feature batch: theta {theta_feat.shape}  lambda {lam_feat.shape}  "
          f"in {t_infer*1e3:.2f} ms\n")

    #Re-approximation stress test
    infer = make_inference(n_, d, th.shape[1], la.shape[1])
    W0 = W_SPREAD * jax.random.normal(jax.random.PRNGKey(1), (SAMPLES, n_))
    pi, c = jax.block_until_ready(infer(th, la, W0))   #Compile inference kernel
    print(f"infer out  pi {pi.shape}   c {c.shape}\n")

    #Pre-generate moving centres and sample batches
    key = jax.random.PRNGKey(0)
    key, kz, kx, kw = jax.random.split(key, 4)
    z_pts = z0 + SPREAD * jax.random.normal(kz, (ITERS, nsum))
    x_pts = x_0 + SPREAD * jax.random.normal(kx, (ITERS, n_))
    W_all = W_SPREAD * jax.random.normal(kw, (ITERS, SAMPLES, n_))
    z_pts, x_pts, W_all = jax.block_until_ready((z_pts, x_pts, W_all))

    t_coeff = np.zeros(ITERS); t_solve = np.zeros(ITERS)
    t_inf = np.zeros(ITERS); t_total = np.zeros(ITERS)

    #Stress-test
    for i in range(ITERS):
        zp, xp, W = z_pts[i], x_pts[i], W_all[i]
        t0 = time.perf_counter()

        f = tay_f(zp); h = tay_h(zp); fe = tay_e(xp)
        f, h, fe = jax.block_until_ready((f, h, fe))
        t1 = time.perf_counter()

        th, la = fbi(f, h, fe, n, m, p, n_, d)
        # th, la = solver.solve(f, h, fe)
        th, la = jax.block_until_ready((th, la))
        th = jnp.reshape(th, (n, -1)); la = jnp.reshape(la, (m, -1))
        t2 = time.perf_counter()

        pi, c = jax.block_until_ready(infer(th, la, W))
        t3 = time.perf_counter()

        t_coeff[i] = t1 - t0; t_solve[i] = t2 - t1
        t_inf[i] = t3 - t2; t_total[i] = t3 - t0

    s = slice(BURN, ITERS)
    nstep = ITERS - BURN
    print(f"stress test, {nstep} steps averaged ({BURN} burn in), {SAMPLES} samples/step")
    print(f"  1 coeffs  {t_coeff[s].mean()*1e3:8.3f} ms")
    print(f"  2 fbi     {t_solve[s].mean()*1e3:8.3f} ms")
    print(f"  3 infer   {t_inf[s].mean()*1e3:8.3f} ms   ({SAMPLES} samples)")
    print(f"  -------------------------")
    print(f"  total     {t_total[s].mean()*1e3:8.3f} ms/step")
    print(f"  rate      {1.0/t_total[s].mean():8.1f} steps/s")
    print(f"  infer     {SAMPLES/t_inf[s].mean():8.0f} samples/s\n")

if __name__ == "__main__":
    main()
