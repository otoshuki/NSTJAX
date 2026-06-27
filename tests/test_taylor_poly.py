"""
Author: gpertin, KAIST
Test JAX version of Krener's tay_poly
"""

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from NSTJAX.NSTJAX_suite import build_taylor, build_taylor_batch, precompile

#System Design: Drone tracking problem
d = 3 #This example works for higher degrees
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

def main():
    #Build reusable taylor polynomial graph
    t = time.time()
    tay_f = build_taylor(fsys, nsum, d)
    tay_h = build_taylor(hsys, nsum, d)
    tay_e = build_taylor(fexo, n_, d)
    t_build = time.time() - t

    #Warm up
    t = time.time()
    f = jax.block_until_ready(tay_f(z0))
    h = jax.block_until_ready(tay_h(z0))
    fe = jax.block_until_ready(tay_e(x_0))
    t_taylor = time.time() - t
    print(f"f shape {f.shape}   h shape {h.shape}   fe shape {fe.shape}")

    #Generate random batch
    key = jax.random.PRNGKey(0)
    key, ks = jax.random.split(key)
    pts = z0 + 0.2 * jax.random.normal(ks, (200, nsum))

    #Batch case
    tay_f_b = build_taylor_batch(fsys, nsum, d)
    jax.block_until_ready(tay_f_b(pts))                   #compile
    t = time.time()
    fb = jax.block_until_ready(tay_f_b(pts))
    t_batch = time.time() - t
    print(f"batch shape {fb.shape}   {t_batch*1e3:.3f} ms for {pts.shape[0]} points")
    assert float(jnp.max(jnp.abs(fb[0] - tay_f(pts[0])))) < 1e-9

    #timing report
    print("timing")
    print(f"  build maps           {t_build:8.3f} s")
    print(f"  taylor first call    {t_taylor:8.3f} s")
    print(f"  taylor batch         {t_batch * 1e3:8.3f} ms  for {pts.shape[0]} points")

if __name__ == "__main__":
    main()
