"""
Author: gpertin, KAIST
NSTJAX example: 3D thruster tracking a moving path in periodic wind
"""

import time
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import scipy.signal as sig
from NSTJAX import NSTJAX
from NSTJAX.NSTJAX_suite import compute_theta, compute_lambda

#Closed loop rollout params
DT = 0.02
STEPS = 900
POLES = [-2.5, -3.5, -2.6, -3.6, -2.7, -3.7]

#Loop params
SAMPLES = 256
ITERS = 10
SPREAD = 0.1

#System Design: 3D thruster following a moving reference under a periodic wind
#The exosystem carries a reference oscillator and a wind oscillator at distinct
#frequencies, the error is the position relative to the moving reference
d = 3
mass = 1.0
c_d = 1.0
k_f = 1.0
om_r = 0.8
om_w = 1.7
n = 6
m = 3
p = 3
n_ = 4
nsum = n + m + n_
REF_AMP = 1.0
WIND_AMP = 1.0
#Reference shape and wind shape, each maps a 2D oscillator into 3D
R = jnp.array([[1.0, 0.0],
               [0.0, 1.0],
               [0.5, 0.5]])
G = jnp.array([[0.7, 0.0],
               [0.0, 0.7],
               [0.4, 0.4]])
#Operating point, equilibrium at the exosystem origin
z0 = jnp.zeros(nsum)
x_0 = jnp.zeros(n_)
w0 = jnp.array([REF_AMP, 0.0, WIND_AMP, 0.0])

def fsys(z):
    x = z[:n]
    u = z[n:n + m]
    xb = z[n + m:]
    v = x[3:6]
    v_air = G @ xb[2:4]
    rel = v - v_air
    drag = (c_d / mass) * jnp.dot(rel, rel) * rel
    acc = u + k_f * v_air - drag
    return jnp.concatenate([v, acc])

def fexo(xb):
    return jnp.array([om_r * xb[1], -om_r * xb[0],
                      om_w * xb[3], -om_w * xb[2]])

def hsys(z):
    x = z[:n]
    xb = z[n + m:]
    return x[0:3] - R @ xb[0:2]

#Analytic manifold and feedforward, the plant is cubic so degree 3 is exact
def theta_star(W):
    rp = W[:, 0:2] @ R.T
    rdot = jnp.stack([om_r * W[:, 1], -om_r * W[:, 0]], axis=1)
    rv = rdot @ R.T
    return jnp.concatenate([rp, rv], axis=1)

def lambda_star(W):
    rp = W[:, 0:2] @ R.T
    rdot = jnp.stack([om_r * W[:, 1], -om_r * W[:, 0]], axis=1)
    rv = rdot @ R.T
    va = W[:, 2:4] @ G.T
    rel = rv - va
    sp = jnp.sum(rel**2, axis=1, keepdims=True)
    return -om_r**2 * rp - k_f * va + (c_d / mass) * sp * rel

#Feedback gain by pole placement on the double integrator linearization
def feedback_gain():
    A = np.zeros((n, n)); A[0, 3] = A[1, 4] = A[2, 5] = 1.0
    B = np.zeros((n, m)); B[3, 0] = B[4, 1] = B[5, 2] = 1.0
    K = -sig.place_poles(A, B, np.asarray(POLES)).gain_matrix
    return jnp.asarray(K, dtype=z0.dtype)

#FBI residuals, how well the truncated solution satisfies the regulator equations
def fbi_residuals(Th, La, W):
    def th_fn(w):
        return compute_theta(Th, w[None], d)[0]
    def la_fn(w):
        return compute_lambda(La, w[None], d)[0]
    def one(w):
        x = th_fn(w)
        u = la_fn(w)
        z = jnp.concatenate([x, u, w])
        inv = fsys(z) - jax.jacfwd(th_fn)(w) @ fexo(w)
        out = hsys(z)
        return inv, out
    inv, out = jax.vmap(one)(W)
    return float(jnp.max(jnp.abs(inv))), float(jnp.max(jnp.abs(out)))

#Closed loop rollout on the true plant, feedforward plus feedback
def _rk4(field, x, dt):
    k1 = field(x)
    k2 = field(x + 0.5 * dt * k1)
    k3 = field(x + 0.5 * dt * k2)
    k4 = field(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

def _plant_step(x, u, w, dt):
    def fld(xx):
        v = xx[3:6]
        v_air = G @ w[2:4]
        rel = v - v_air
        drag = (c_d / mass) * jnp.dot(rel, rel) * rel
        return jnp.concatenate([v, u + k_f * v_air - drag])
    return _rk4(fld, x, dt)

@partial(jax.jit, static_argnums=(5,))
def rollout(Th, La, K, xi, wi, steps):
    def step(carry, _):
        x, w = carry
        pi = compute_theta(Th, w[None], d)[0]
        c = compute_lambda(La, w[None], d)[0]
        u = c + K @ (x - pi)
        x2 = _plant_step(x, u, w, DT)
        w2 = _rk4(fexo, w, DT)
        return (x2, w2), (x, c, w)
    _, (xs, cs, ws) = jax.lax.scan(step, (xi, wi), None, length=steps)
    return xs, cs, ws

def make_outputs(xf, cf, wf, xb, nst):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception:
        print("matplotlib unavailable, skipping animation")
        return
    Rn = np.asarray(R)
    Gn = np.asarray(G)
    t = np.arange(xf.shape[0]) * DT
    ref = np.asarray(nst.compute_theta(jnp.asarray(wf)))[:, :3]
    en_f = np.linalg.norm(xf[:, :3] - ref, axis=1)
    en_b = np.linalg.norm(xb[:, :3] - ref, axis=1)
    wind = wf[:, 2:4] @ Gn.T
    sp = np.sum(wind**2, axis=1, keepdims=True)
    rv = np.stack([om_r * wf[:, 1], -om_r * wf[:, 0]], axis=1) @ Rn.T
    dforce = k_f * wind + (c_d / mass) * np.sum((rv - wind)**2, axis=1, keepdims=True) * (rv - wind)

    #Reference loop for context
    tt = np.linspace(0, 2 * np.pi / om_r, 300)
    loop = (np.stack([REF_AMP * np.cos(om_r * tt), REF_AMP * np.sin(om_r * tt)], axis=1)) @ Rn.T

    lim = float(np.abs(loop).max()) * 1.4
    sc = 0.5 * lim / (np.abs(dforce).max() + 1e-9)
    cmax = float(np.abs(cf).max()) * 1.2
    trail = 80

    #Single animation, three panels with clear spacing
    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0], hspace=0.55, wspace=0.30)
    ax1 = fig.add_subplot(gs[:, 0], projection="3d")
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    fig.subplots_adjust(left=0.01, right=0.95, top=0.93, bottom=0.10)
    frames = range(0, xf.shape[0], 6)
    def draw(i):
        ax1.cla(); ax2.cla(); ax3.cla()
        ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim); ax1.set_zlim(-lim, lim)
        ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
        ax1.set_title(f"tracking a path in wind   t = {i * DT:4.2f} s")
        ax1.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="0.6", ls="--", lw=0.8,
                 label="reference path")
        s0 = max(0, i - trail)
        ax1.plot(xb[s0:i + 1, 0], xb[s0:i + 1, 1], xb[s0:i + 1, 2],
                 color="tab:red", lw=0.8, alpha=0.6)
        ax1.plot(xf[s0:i + 1, 0], xf[s0:i + 1, 1], xf[s0:i + 1, 2],
                 color="tab:green", lw=1.2, alpha=0.8)
        ax1.scatter(xb[i, 0], xb[i, 1], xb[i, 2], color="tab:red", s=40, label="feedback only")
        ax1.scatter(xf[i, 0], xf[i, 1], xf[i, 2], color="tab:green", s=55, label="FBI")
        ax1.quiver(xf[i, 0], xf[i, 1], xf[i, 2],
                   sc * dforce[i, 0], sc * dforce[i, 1], sc * dforce[i, 2],
                   color="tab:blue", label="wind force")
        ax1.legend(loc="upper left", fontsize=8)
        ax2.plot(t[:i + 1], en_b[:i + 1], color="tab:red", label="feedback only")
        ax2.plot(t[:i + 1], en_f[:i + 1], color="tab:green", label="FBI")
        ax2.set_xlim(0, t[-1]); ax2.set_ylim(0, max(en_b.max() * 1.15, 1e-3))
        ax2.set_xlabel("time [s]"); ax2.set_ylabel("tracking error [m]")
        ax2.set_title("tracking error"); ax2.legend(loc="upper right", fontsize=8)
        for j, lab in enumerate(["x", "y", "z"]):
            ax3.plot(t[:i + 1], cf[:i + 1, j], label=f"lambda_{lab}")
        ax3.set_xlim(0, t[-1]); ax3.set_ylim(-cmax, cmax)
        ax3.set_xlabel("time [s]"); ax3.set_ylabel("feedforward")
        ax3.set_title("FBI feedforward"); ax3.legend(loc="upper right", ncol=3, fontsize=8)
    anim = FuncAnimation(fig, draw, frames=frames, interval=50)
    anim.save("tracking_anim.gif", writer=PillowWriter(fps=20))
    print("saved tracking_anim.gif")

def main():
    print(f"dims: n={n} m={m} p={p} n_={n_} d={d} nsum={nsum}\n")
    #Two frequency exosystem, fbi_fast exercises the general complex schur branch
    nst = NSTJAX(fsys, fexo, hsys, n, m, p, n_, d, check="setup", verbose=True)
    nst.warm_start(z0, x_0, samples=SAMPLES)
    print(f"theta shape {nst.th.shape}   lambda shape {nst.la.shape}\n")

    #Solvability screen
    rep = nst.report
    print("solvability")
    print(f"  regime    {rep.get('regime')}")
    if rep.get("checked"):
        z = rep.get("zeros")
        print(f"  zeros     {0 if z is None else len(z)}")
        print(f"  solvable  {rep.get('solvable')}")
        for k, dv in rep.get("degrees", {}).items():
            print(f"  degree {k}  min-gap {dv['min_gap']:.2e}  resonant {dv['resonant']}")
    print()

    #Pre-generate moving operating points and inference batches
    key = jax.random.PRNGKey(0)
    key, kz, kx, kw = jax.random.split(key, 4)
    z_pts = z0 + SPREAD * jax.random.normal(kz, (ITERS, nsum))
    x_pts = x_0 + SPREAD * jax.random.normal(kx, (ITERS, n_))
    W_all = REF_AMP * jax.random.normal(kw, (ITERS, SAMPLES, n_))
    z_pts, x_pts, W_all = jax.block_until_ready((z_pts, x_pts, W_all))

    #Timed fbi + inference at each moving operating point
    t_step = np.zeros(ITERS)
    for i in range(ITERS):
        zp, xp, W = z_pts[i], x_pts[i], W_all[i]
        t0 = time.perf_counter()
        th, la = nst.compute_fbi(zp, xp)
        pi = nst.compute_theta(W)
        c = nst.compute_lambda(W)
        jax.block_until_ready((pi, c))
        t_step[i] = time.perf_counter() - t0
    dt_step = t_step.mean()
    print(f"fbi + inference, {ITERS} steps, {SAMPLES} samples/step")
    print(f"  solver      {nst.solver_name}")
    print(f"  kron-dim    {nst.kron_dim}")
    print(f"  per step    {dt_step * 1e3:8.3f} ms")
    print(f"  frequency   {1.0 / dt_step:8.1f} Hz\n")

    #Re-solve at the base operating point for validation and rollout
    nst.compute_fbi(z0, x_0)

    #Validation against the analytic manifold and feedforward and the residuals
    key, ks = jax.random.split(key)
    Wv = REF_AMP * jax.random.normal(ks, (2048, n_))
    th_fbi = nst.compute_theta(Wv)
    la_fbi = nst.compute_lambda(Wv)
    rel_th = float(jnp.max(jnp.abs(th_fbi - theta_star(Wv))) / (jnp.max(jnp.abs(theta_star(Wv))) + 1e-12))
    rel_la = float(jnp.max(jnp.abs(la_fbi - lambda_star(Wv))) / (jnp.max(jnp.abs(lambda_star(Wv))) + 1e-12))
    inv_res, out_res = fbi_residuals(nst.th, nst.la, Wv)
    print("validation")
    print(f"  rel-theta vs analytic  {rel_th:.2e}")
    print(f"  rel-lambda vs analytic {rel_la:.2e}")
    print(f"  invariance residual    {inv_res:.2e}")
    print(f"  output residual        {out_res:.2e}\n")

    #Closed loop rollout, started on the manifold, FBI versus feedback only
    K = feedback_gain()
    x_init = nst.compute_theta(w0[None])[0]
    xf, cf, wf = rollout(nst.th, nst.la, K, x_init, w0, STEPS)
    xb, cb, wb = rollout(nst.th, jnp.zeros_like(nst.la), K, x_init, w0, STEPS)
    jax.block_until_ready((xf, xb))
    ref = nst.compute_theta(wf)[:, :3]
    rms_f = float(jnp.sqrt(jnp.mean(jnp.sum((xf[:, :3] - ref)**2, axis=1))))
    rms_b = float(jnp.sqrt(jnp.mean(jnp.sum((xb[:, :3] - ref)**2, axis=1))))
    print("closed loop tracking on the true plant")
    print(f"  feedback only   rms {rms_b:.3e}")
    print(f"  FBI feedforward rms {rms_f:.3e}")
    print(f"  improvement     {rms_b / max(rms_f, 1e-12):.1f}x\n")

    make_outputs(np.asarray(xf), np.asarray(cf), np.asarray(wf), np.asarray(xb), nst)

if __name__ == "__main__":
    main()
