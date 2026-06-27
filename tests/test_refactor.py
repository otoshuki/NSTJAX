"""
Author: gpertin, KAIST
Check the relocated wrapper, suite re-exports and bridge encoding
"""

import jax.numpy as jnp
from NSTJAX import nstjax
from NSTJAX import NSTJAX
from NSTJAX.NSTJAX_suite import crd, fbi, FBIFast, build_report
from NSTJAX.NSTJAX_bridge import encode_to_reduced

#Drone tracking, degree 2
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

def main():
    #Wrapper reachable as both module and class
    assert nstjax.NSTJAX is NSTJAX
    print("imports ok: wrapper, suite re-exports, bridge")

    nst = NSTJAX(fsys, fexo, hsys, n, m, p, n_, d, verbose=True)
    nst.warm_start(z0, x_0)
    print(f"theta {nst.th.shape}   lambda {nst.la.shape}")

    th_r, la_r = nst.encode_matlab()
    assert th_r.shape == nst.th.shape
    assert la_r.shape == nst.la.shape
    print(f"encoded theta {th_r.shape}   lambda {la_r.shape}")
    print("wrapper encode_matlab ok")

if __name__ == "__main__":
    main()
