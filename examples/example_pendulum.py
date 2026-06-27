"""
Author: gpertin, KAIST
NSTJAX example: pendulum tracking
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from NSTJAX import NSTJAX

#Loop params
SAMPLES = 256         #Inference samples per step
ITERS = 10            #Timed steps
SPREAD = 0.2          #How far the operating point wanders each step
W_SPREAD = 0.2        #Spread of the exosystem inference samples

#System Design: pendulum tracking problem
d = 3
g = 9.81
l = 1.0
b = 0.1
mass = 1.0
om = 1.0
n = 2
m = 1
n_ = 3
p = 1
nsum = n + m + n_
#Operating point
x0 = jnp.array([0.0, 0.0])
u0 = jnp.zeros(m)
x_0 = jnp.zeros(n_)
z0 = jnp.concatenate([x0, u0, x_0])

def fsys(z):
    x = z[:n]
    u = z[n:n + m]
    th = x[0]
    thd = x[1]
    usys = u[0]
    thdot = thd
    thddot = -g / l * jnp.sin(th) - b / (mass * l**2) * thd + usys
    return jnp.array([thdot, thddot])

def fexo(xb):
    return jnp.concatenate([jnp.array([-(om + xb[2]) * xb[1],
                                       (om + xb[2]) * xb[0]]),
                            jnp.zeros(1)])

def hsys(z):
    x = z[:n]
    xb = z[n + m:]
    return jnp.array([x[0] - xb[0]])

def main():
    print(f"dims: n={n} m={m} p={p} n_={n_} d={d} nsum={nsum}\n")
    #Small system, auto routing falls back to the dense fbi solve
    nst = NSTJAX(fsys, fexo, hsys, n, m, p, n_, d, verbose=True)
    nst.warm_start(z0, x_0, samples=SAMPLES)
    print(f"theta shape {nst.th.shape}   lambda shape {nst.la.shape}\n")

    #Pre-generate moving centres and sample batches
    key = jax.random.PRNGKey(0)
    key, kz, kx, kw = jax.random.split(key, 4)
    z_pts = z0 + SPREAD * jax.random.normal(kz, (ITERS, nsum))
    x_pts = x_0 + SPREAD * jax.random.normal(kx, (ITERS, n_))
    W_all = W_SPREAD * jax.random.normal(kw, (ITERS, SAMPLES, n_))
    z_pts, x_pts, W_all = jax.block_until_ready((z_pts, x_pts, W_all))

    #Recompute fbi + inference at each moving operating point
    t_step = np.zeros(ITERS)
    for i in range(ITERS):
        zp, xp, W = z_pts[i], x_pts[i], W_all[i]
        t0 = time.perf_counter()
        th, la = nst.compute_fbi(zp, xp)
        pi = nst.compute_theta(W)
        c = nst.compute_lambda(W)
        jax.block_until_ready((pi, c))
        t_step[i] = time.perf_counter() - t0

    dt = t_step.mean()
    print(f"fbi + inference, {ITERS} steps, {SAMPLES} samples/step")
    print(f"  solver      {nst.solver_name}")
    print(f"  per step    {dt*1e3:8.3f} ms")
    print(f"  frequency   {1.0/dt:8.1f} Hz")

if __name__ == "__main__":
    main()
