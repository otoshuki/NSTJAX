"""
Author: gpertin, KAIST
Discrete time FBI test: invariance residual order and dense vs decoupled cross check
"""

import numpy as np
import jax
import jax.numpy as jnp
from NSTJAX.NSTJAX_suite import build_taylor, fbi, compute_theta, compute_lambda
from NSTJAX.NSTJAX_suite.fbi_fast import FBIFast

jax.config.update("jax_enable_x64", True)

#Dimensions, discrete damped pendulum tracking a reference
d = 3
n = 2
m = 1
p = 1
DT = 0.1

z0_osc = jnp.zeros(n + m + 2)
x0_osc = jnp.zeros(2)
z0_nil = jnp.zeros(n + m + 2)
x0_nil = jnp.zeros(2)

#Discrete plant, one forward step, fixed point at the origin
def fsys(z, n_):
    nsum = n + m + n_
    x = z[:n]
    u = z[n:n + m]
    x1n = x[0] + DT * x[1]
    x2n = x[1] + DT * (-jnp.sin(x[0]) - 0.2 * x[1] + u[0])
    return jnp.array([x1n, x2n])

def hsys(z, n_):
    x = z[:n]
    xb = z[n + m:]
    return jnp.array([x[0] - xb[0]])

#General branch exosystem, discrete oscillator with a small nonlinear twist
OM = 0.7
def fexo_osc(xb):
    c, s = jnp.cos(OM), jnp.sin(OM)
    return jnp.array([c * xb[0] - s * xb[1] + 0.05 * xb[0] * xb[1],
                      s * xb[0] + c * xb[1]])

#Nilpotent branch exosystem, A_ has zero eigenvalues
def fexo_nil(xb):
    return jnp.array([xb[1] + 0.1 * xb[1] ** 2, 0.0])

#Discrete invariance residual, f(th, la, w) - th(f_(w)), and output h
def residuals(th, la, fsys_n, fexo, hsys_n, n_, scale):
    key = jax.random.PRNGKey(1)
    W = scale * jax.random.normal(key, (64, n_))
    pi = compute_theta(th, W, d)
    c = compute_lambda(la, W, d)
    fw = jax.vmap(fexo)(W)
    pi_next = compute_theta(th, fw, d)

    def one(w, p_, c_, pn_):
        z = jnp.concatenate([p_, c_, w])
        inv = fsys_n(z) - pn_
        out = hsys_n(z)
        return jnp.max(jnp.abs(inv)), jnp.max(jnp.abs(out))

    inv, out = jax.vmap(one)(W, pi, c, pi_next)
    return float(jnp.max(inv)), float(jnp.max(out))

def run_case(name, fexo, z0, x_0, n_):
    fsys_n = lambda z: fsys(z, n_)
    hsys_n = lambda z: hsys(z, n_)
    nsum = n + m + n_
    tay_f = build_taylor(fsys_n, nsum, d)
    tay_h = build_taylor(hsys_n, nsum, d)
    tay_e = build_taylor(fexo, n_, d)
    f = tay_f(z0); h = tay_h(z0); fe = tay_e(x_0)

    #Dense discrete solve
    th, la = fbi(f, h, fe, n, m, p, n_, d, disc=True)
    th = jnp.reshape(th, (n, -1)); la = jnp.reshape(la, (m, -1))

    #Decoupled discrete solve and cross check
    solver = FBIFast(n, m, p, n_, d, fixed=True, disc=True, check="on")
    thf, laf, rep = solver.solve(f, h, fe)
    thf = jnp.reshape(thf, (n, -1)); laf = jnp.reshape(laf, (m, -1))
    dense_vs_fast = float(jnp.max(jnp.abs(th - thf)) + jnp.max(jnp.abs(la - laf)))

    print(f"\n[{name}]  n_={n_}  branch={rep.get('branch')}  "
          f"solvable={rep.get('solvable')}")
    print(f"  dense vs decoupled (f64 vs f32)   {dense_vs_fast:.2e}")
    for sc in (0.10, 0.05):
        inv, out = residuals(th, la, fsys_n, fexo, hsys_n, n_, sc)
        print(f"  |w|~{sc:.2f}   invariance-res {inv:.2e}   output-res {out:.2e}")
    print("  (residuals should drop ~2^(d+1)=16x when |w| halves)")

def main():
    print(f"dims: n={n} m={m} p={p} d={d}  discrete time FBI")
    run_case("oscillator / general", fexo_osc, z0_osc, x0_osc, 2)
    run_case("nilpotent / neumann", fexo_nil, z0_nil, x0_nil, 2)

if __name__ == "__main__":
    main()
