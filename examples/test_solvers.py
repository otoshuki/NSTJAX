"""
Author: gpertin, KAIST
Check the unified FBI report matches across the dense and fast solvers
"""

import jax.numpy as jnp
from NSTJAX.NSTJAX_suite.nstjax import NSTJAX

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

def _fmt(r):
    degs = {k: (round(v["min_gap"], 4), v["resonant"]) for k, v in r["degrees"].items()}
    return (f"regime={r['regime']} resonant={r['resonant']} "
            f"solvable={r['solvable']} degrees={degs}")

def main():
    print(f"dims: n={n} m={m} p={p} n_={n_} d={d}\n")

    nst_d = NSTJAX(fsys, fexo, hsys, n, m, p, n_, d, solver="fbi", check="always")
    nst_d.warm_start(z0, x_0)
    nst_f = NSTJAX(fsys, fexo, hsys, n, m, p, n_, d, solver="fbi_fast", check="always")
    nst_f.warm_start(z0, x_0)

    rd, rf = nst_d.report, nst_f.report
    print("dense :", _fmt(rd))
    print("fast  :", _fmt(rf))

    assert rd["regime"] == rf["regime"]
    assert rd["resonant"] == rf["resonant"]
    assert rd["solvable"] == rf["solvable"]
    for k in rd["degrees"]:
        assert rd["degrees"][k]["resonant"] == rf["degrees"][k]["resonant"]
        assert abs(rd["degrees"][k]["min_gap"] - rf["degrees"][k]["min_gap"]) < 1e-6
    print("\nreports match across solvers")

    #Solutions should also agree
    sth = float(jnp.max(jnp.abs(nst_d.th))) + 1e-30
    sla = float(jnp.max(jnp.abs(nst_d.la))) + 1e-30
    dth = float(jnp.max(jnp.abs(nst_f.th - nst_d.th))) / sth
    dla = float(jnp.max(jnp.abs(nst_f.la - nst_d.la))) / sla
    print(f"rel-theta dense vs fast {dth:.2e}   rel-lambda {dla:.2e}")

    #check=off path stays cheap and reports nothing screened
    nst_off = NSTJAX(fsys, fexo, hsys, n, m, p, n_, d, solver="fbi_fast", check="off")
    nst_off.warm_start(z0, x_0)
    assert nst_off.report["checked"] is False
    print(f"check=off report: {nst_off.report}")

if __name__ == "__main__":
    main()
