"""
Author: gpertin, KAIST
High level NSTJAX example: 3D thruster disturbance rejection in periodic wind
"""

import time
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import scipy.signal as sig
from NSTJAX import NSTJAX
from NSTJAX.NSTJAX_suite import compute_theta, compute_lambda

#Loop params
SAMPLES = 256         #Inference samples per step
ITERS = 10            #Timed steps
SPREAD = 0.1          #How far the operating point wanders each step
W_SPREAD = 0.4        #Spread of the exosystem inference samples

#Closed loop rollout params
DT = 0.02
STEPS = 620
POLES = [-2.0, -3.0, -2.2, -3.2, -2.4, -3.4]

#System Design: 3D thruster holding station against a periodic wind
#The exosystem generates a 3D periodic air velocity, the wind couples in as a
#body force plus a nonlinear relative velocity drag, the error is the position
d = 3
mass = 1.0
c_d = 1.0
k_f = 1.0
om = 1.0
n = 6
m = 3
p = 3
n_ = 2
nsum = n + m + n_
WIND_AMP = 0.6
#Wind shape, maps the 2D oscillator onto a 3D elliptical air velocity
G = jnp.array([[0.8, 0.0],
               [0.0, 0.8],
               [0.4, 0.4]])
#Operating point, equilibrium at rest with no wind
x0 = jnp.zeros(n)
u0 = jnp.zeros(m)
x_0 = jnp.zeros(n_)
z0 = jnp.concatenate([x0, u0, x_0])
w0 = jnp.array([WIND_AMP, 0.0])

def fsys(z):
    x = z[:n]
    u = z[n:n + m]
    xb = z[n + m:]
    v = x[3:6]
    v_air = G @ xb
    rel = v - v_air
    drag = (c_d / mass) * jnp.dot(rel, rel) * rel
    acc = u + k_f * v_air - drag
    return jnp.concatenate([v, acc])

def fexo(xb):
    return jnp.array([om * xb[1], -om * xb[0]])

def hsys(z):
    x = z[:n]
    return x[0:3]

#Analytic feedforward, the manifold collapses to the setpoint so this is exact
def lambda_star(W):
    va = W @ G.T
    sp = jnp.sum(va**2, axis=1, keepdims=True)
    return -(k_f * va) - (c_d / mass) * sp * va

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
        v_air = G @ w
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
        return (x2, w2), (x, u, w)
    _, (xs, us, ws) = jax.lax.scan(step, (xi, wi), None, length=steps)
    return xs, us, ws

def make_outputs(xf, uf, wf, xb, nst):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception:
        print("matplotlib unavailable, skipping figures")
        return
    Gn = np.asarray(G)
    t = np.arange(xf.shape[0]) * DT

    #Error norm, FBI versus feedback only
    plt.figure(figsize=(7, 3.2))
    plt.semilogy(t, np.linalg.norm(xb[:, :3], axis=1) + 1e-14, label="feedback only")
    plt.semilogy(t, np.linalg.norm(xf[:, :3], axis=1) + 1e-14, label="FBI + feedback")
    plt.xlabel("time [s]"); plt.ylabel("position error norm [m]")
    plt.legend(); plt.tight_layout(); plt.savefig("disturbance_error.png", dpi=130)

    #Feedforward over one wind period against the analytic cancellation law
    tp = np.linspace(0, 2 * np.pi / om, 200)
    Wp = WIND_AMP * np.stack([np.cos(om * tp), -np.sin(om * tp)], axis=1)
    la_fbi = np.asarray(nst.compute_lambda(jnp.asarray(Wp)))
    la_ana = np.asarray(lambda_star(jnp.asarray(Wp)))
    plt.figure(figsize=(7, 3.2))
    for j, lab in enumerate(["x", "y", "z"]):
        line, = plt.plot(tp, la_fbi[:, j], label=f"FBI lambda_{lab}")
        plt.plot(tp, la_ana[:, j], ls="none", marker="o", ms=2, color=line.get_color())
    plt.xlabel("time over one wind period [s]"); plt.ylabel("feedforward thrust")
    plt.title("markers: analytic wind cancellation law")
    plt.legend(ncol=3); plt.tight_layout(); plt.savefig("disturbance_feedforward.png", dpi=130)

    #Net disturbance force felt at rest, exactly what the feedforward cancels
    wind = wf @ Gn.T
    sp = np.sum(wind**2, axis=1, keepdims=True)
    dforce = k_f * wind + (c_d / mass) * sp * wind
    en_f = np.linalg.norm(xf[:, :3], axis=1)
    en_b = np.linalg.norm(xb[:, :3], axis=1)
    lim = max(float(np.abs(xb[:, :3]).max()), 0.15) * 1.25

    #3D scene, FBI holds the target while feedback only is blown off by the wind
    fig = plt.figure(figsize=(6.4, 5)); ax = fig.add_subplot(111, projection="3d")
    ax.plot(xb[:, 0], xb[:, 1], xb[:, 2], color="tab:red", lw=1.2,
            label="feedback only (drifts)")
    ax.scatter(xf[-1, 0], xf[-1, 1], xf[-1, 2], color="tab:green", s=70,
               label="FBI (holds target)")
    ax.scatter([0], [0], [0], color="k", marker="*", s=140, label="target")
    ns = 12
    idx = np.linspace(0, xf.shape[0] - 1, ns).astype(int)
    sc = 0.55 * lim / (np.abs(dforce).max() + 1e-9)
    ax.quiver(np.zeros(ns), np.zeros(ns), np.zeros(ns),
              sc * dforce[idx, 0], sc * dforce[idx, 1], sc * dforce[idx, 2],
              color="tab:blue", alpha=0.55, linewidth=0.9, label="wind force")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Station-keeping in periodic wind")
    ax.legend(loc="upper left", fontsize=8); plt.tight_layout()
    plt.savefig("disturbance_3d.png", dpi=130)

    #Animation, the wind force against the counter thrust with a live error trace
    frames = range(0, xf.shape[0], 5)
    sc = 0.6 * lim / (np.abs(dforce).max() + 1e-9)
    figA = plt.figure(figsize=(10, 4.4))
    ax1 = figA.add_subplot(121, projection="3d")
    ax2 = figA.add_subplot(122)
    def draw(i):
        ax1.cla(); ax2.cla()
        ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim); ax1.set_zlim(-lim, lim)
        ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
        ax1.set_title(f"station-keeping in wind   t = {i * DT:4.2f} s")
        ax1.scatter([0], [0], [0], color="k", marker="*", s=140)
        ax1.plot(xb[:i + 1, 0], xb[:i + 1, 1], xb[:i + 1, 2],
                 color="tab:red", lw=0.8, alpha=0.5)
        ax1.scatter(xb[i, 0], xb[i, 1], xb[i, 2], color="tab:red", s=45, label="feedback only")
        ax1.scatter(xf[i, 0], xf[i, 1], xf[i, 2], color="tab:green", s=60, label="FBI")
        ax1.quiver(0, 0, 0, sc * dforce[i, 0], sc * dforce[i, 1], sc * dforce[i, 2],
                   color="tab:blue", label="wind force")
        ax1.quiver(0, 0, 0, sc * uf[i, 0], sc * uf[i, 1], sc * uf[i, 2],
                   color="tab:orange", label="FBI thrust")
        ax1.legend(loc="upper left", fontsize=8)
        ax2.plot(t[:i + 1], en_b[:i + 1], color="tab:red", label="feedback only")
        ax2.plot(t[:i + 1], en_f[:i + 1], color="tab:green", label="FBI")
        ax2.set_xlim(0, t[-1]); ax2.set_ylim(0, max(en_b.max() * 1.15, 1e-3))
        ax2.set_xlabel("time [s]"); ax2.set_ylabel("position error norm [m]")
        ax2.set_title("FBI holds the error at zero")
        ax2.legend(loc="upper right", fontsize=8)
    anim = FuncAnimation(figA, draw, frames=frames, interval=50)
    anim.save("disturbance_anim.gif", writer=PillowWriter(fps=20))
    print("saved disturbance_error.png, disturbance_feedforward.png, "
          "disturbance_3d.png, disturbance_anim.gif")

def main():
    print(f"dims: n={n} m={m} p={p} n_={n_} d={d} nsum={nsum}\n")
    #Harmonic exosystem, fbi_fast here exercises the general complex schur branch
    #auto routing would send a system this small to the dense solve instead
    nst = NSTJAX(fsys, fexo, hsys, n, m, p, n_, d,
                 solver="fbi_fast", check="setup", verbose=True)
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
    W_all = W_SPREAD * jax.random.normal(kw, (ITERS, SAMPLES, n_))
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

    #Validation against the analytic wind cancellation law and the residuals
    key, ks = jax.random.split(key)
    Wv = WIND_AMP * jax.random.normal(ks, (2048, n_))
    la_fbi = nst.compute_lambda(Wv)
    la_ana = lambda_star(Wv)
    rel_la = float(jnp.max(jnp.abs(la_fbi - la_ana)) / (jnp.max(jnp.abs(la_ana)) + 1e-12))
    max_theta = float(jnp.max(jnp.abs(nst.compute_theta(Wv))))
    inv_res, out_res = fbi_residuals(nst.th, nst.la, Wv)
    print("validation")
    print(f"  max |theta|            {max_theta:.2e}")
    print(f"  rel-lambda vs analytic {rel_la:.2e}")
    print(f"  invariance residual    {inv_res:.2e}")
    print(f"  output residual        {out_res:.2e}\n")

    #Closed loop rollout, FBI versus feedback only
    K = feedback_gain()
    xf, uf, wf = rollout(nst.th, nst.la, K, x0, w0, STEPS)
    xb, ub, wb = rollout(jnp.zeros_like(nst.th), jnp.zeros_like(nst.la), K, x0, w0, STEPS)
    jax.block_until_ready((xf, xb))
    rms_f = float(jnp.sqrt(jnp.mean(jnp.sum(xf[:, :3]**2, axis=1))))
    rms_b = float(jnp.sqrt(jnp.mean(jnp.sum(xb[:, :3]**2, axis=1))))
    print("closed loop tracking on the true plant")
    print(f"  feedback only   rms {rms_b:.3e}")
    print(f"  FBI feedforward rms {rms_f:.3e}")
    print(f"  improvement     {rms_b / max(rms_f, 1e-12):.1f}x\n")

    make_outputs(np.asarray(xf), np.asarray(uf), np.asarray(wf), np.asarray(xb), nst)

if __name__ == "__main__":
    main()
