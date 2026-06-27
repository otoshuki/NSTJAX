"""
Author: gpertin, KAIST
Generate documentation figures for the FBIJAX README
"""

import os
import time
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from NSTJAX import NSTJAX
import NSTJAX.NSTJAX_suite.polylib as P
from NSTJAX.NSTJAX_suite import fbi, crd, build_taylor, FBIFast, compute_theta, compute_lambda, schur_form, operator_eigs, transmission_zeros

OUTDIR = "docs/figures"
DPI = 150

#Drone tracking system, copied from the examples
def build_drone():
    d = 2
    g = 9.81
    alp = 0.4
    n, m, p, n_ = 10, 4, 4, 4 * 6
    nsum = n + m + n_
    e3 = jnp.array([0.0, 0.0, 1.0])
    x0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    u0 = jnp.zeros(m)
    x_0 = jnp.zeros(n_)
    z0 = jnp.concatenate([x0, u0, x_0])

    def fsys(z):
        x = z[:n]; u = z[n:n + m]
        vsys = x[3:6]; psys = x[6:9]; u0sys = u[0]; om = u[1:4]
        xsysdot = vsys
        vsysdot = -g * e3 + (u0sys * g + g) * psys
        psysdot = jnp.cross(om, psys) - alp * (psys[0]**2 + psys[1]**2 + psys[2]**2 - 1) * psys
        tsysdot = jnp.sum((psys + e3) * om) / (2 * (1 + psys[2]))
        return jnp.concatenate([xsysdot, vsysdot, psysdot, jnp.array([tsysdot])])

    def fexo(xb):
        return jnp.concatenate([xb[4:n_], jnp.zeros(4)])

    def hsys(z):
        x = z[:n]; xb = z[n + m:]
        xsys = x[0:3]; tsys = x[9]; traj = xb[0:4]
        return jnp.concatenate([traj[0:3] - xsys, jnp.array([traj[3] - 2 * tsys])])

    return SimpleNamespace(fsys=fsys, fexo=fexo, hsys=hsys, n=n, m=m, p=p,
                           n_=n_, d=d, nsum=nsum, z0=z0, x_0=x_0)

#Double pendulum tracking system, copied from the examples
def build_pendulum():
    d = 3
    g = 9.81; l = 1.0; b = 0.1; mass = 1.0; om = 1.0
    n, m, p, n_ = 2, 1, 1, 3
    nsum = n + m + n_
    x0 = jnp.array([0.0, 0.0])
    u0 = jnp.zeros(m)
    x_0 = jnp.zeros(n_)
    z0 = jnp.concatenate([x0, u0, x_0])

    def fsys(z):
        x = z[:n]; u = z[n:n + m]
        th = x[0]; thd = x[1]; usys = u[0]
        thddot = -g / l * jnp.sin(th) - b / (mass * l**2) * thd + usys
        return jnp.array([thd, thddot])

    def fexo(xb):
        return jnp.concatenate([jnp.array([-(om + xb[2]) * xb[1],
                                           (om + xb[2]) * xb[0]]), jnp.zeros(1)])

    def hsys(z):
        x = z[:n]; xb = z[n + m:]
        return jnp.array([x[0] - xb[0]])

    return SimpleNamespace(fsys=fsys, fexo=fexo, hsys=hsys, n=n, m=m, p=p,
                           n_=n_, d=d, nsum=nsum, z0=z0, x_0=x_0)

#Scalable nilpotent system, fixed plant with a shift chain exosystem
def build_synth_chain(n_, d):
    n, m, p = 8, 4, 4
    nsum = n + m + n_
    A = 2.0 * jnp.eye(n)
    B = jnp.concatenate([jnp.eye(m), jnp.zeros((n - m, m))], axis=0)
    C = jnp.concatenate([jnp.eye(p), jnp.zeros((p, n - p))], axis=1)
    E = jnp.zeros((n, n_)).at[0, 0].set(1.0)

    def fsys(z):
        x = z[:n]; u = z[n:n + m]; w = z[n + m:]
        return A @ x + B @ u + E @ w

    def fexo(w):
        return jnp.concatenate([w[1:], jnp.zeros(1)])

    def hsys(z):
        x = z[:n]; w = z[n + m:]
        return C @ x - jnp.concatenate([w[:1], jnp.zeros(p - 1)])

    return SimpleNamespace(fsys=fsys, fexo=fexo, hsys=hsys, n=n, m=m, p=p,
                           n_=n_, d=d, nsum=nsum, z0=jnp.zeros(nsum), x_0=jnp.zeros(n_))

#Demo system with an oscillator exosystem and one finite transmission zero
def build_screen_demo():
    d = 3
    om = 1.0
    zloc = -0.7
    n, m, p, n_ = 2, 1, 1, 2
    nsum = n + m + n_
    A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    B = jnp.array([[0.0], [1.0]])
    C = jnp.array([[-zloc, 1.0]])

    def fsys(z):
        x = z[:n]; u = z[n:n + m]
        return A @ x + B @ u

    def fexo(w):
        return jnp.array([-om * w[1], om * w[0]])

    def hsys(z):
        x = z[:n]; w = z[n + m:]
        return C @ x - jnp.array([w[0]])

    return SimpleNamespace(fsys=fsys, fexo=fexo, hsys=hsys, n=n, m=m, p=p,
                           n_=n_, d=d, nsum=nsum, z0=jnp.zeros(nsum),
                           x_0=jnp.zeros(n_), zloc=zloc)

#FBI residual at one operating point, copied from test_fbi_jax and parametrized
def fbi_residual(th, la, f, h, fe, n, m, n_, d, nsum):
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

#Runge Kutta rollout of an exosystem field
def rollout(fexo, w0, steps, dt):
    w = np.asarray(w0, dtype=np.float64)
    out = [w.copy()]
    for _ in range(steps):
        k1 = np.asarray(fexo(jnp.asarray(w)))
        k2 = np.asarray(fexo(jnp.asarray(w + 0.5 * dt * k1)))
        k3 = np.asarray(fexo(jnp.asarray(w + 0.5 * dt * k2)))
        k4 = np.asarray(fexo(jnp.asarray(w + dt * k3)))
        w = w + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(w.copy())
    return np.stack(out)

def _time(fn, reps=20):
    fn()
    t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    jax.block_until_ready(out)
    return (time.perf_counter() - t) / reps

def save(name):
    plt.tight_layout()
    path = os.path.join(OUTDIR, name)
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path}")

#Figure 1, nominal drone trajectory and feedforward
def fig_drone_trajectory():
    s = build_drone()
    nst = NSTJAX(s.fsys, s.fexo, s.hsys, s.n, s.m, s.p, s.n_, s.d, store=True)
    nst.warm_start(s.z0, s.x_0, samples=1)
    th, la = nst.compute_fbi(s.z0, s.x_0)
    #Small initial derivatives keep \bar{x} inside the local region, yaw rate adds body rates
    w0 = np.zeros(s.n_)
    w0[4:8] = [0.35, 0.25, 0.15, 0.15]
    w0[8:12] = [-0.2, 0.25, 0.0, 0.0]
    W = rollout(s.fexo, w0, steps=50, dt=0.02)
    t = np.arange(W.shape[0]) * 0.02
    Wj = jnp.asarray(W)
    pi = np.asarray(compute_theta(th, Wj, s.d))
    c = np.asarray(compute_lambda(la, Wj, s.d))

    fig = plt.figure(figsize=(11, 4.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(pi[:, 0], pi[:, 1], pi[:, 2], color="#2563eb", lw=2)
    ax.scatter(pi[0, 0], pi[0, 1], pi[0, 2], color="#16a34a", s=30)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(r"Nominal flight path  $\theta(\bar{x})$")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t, c[:, 0], color="#2563eb", lw=2, label="thrust")
    ax2.set_xlabel("time"); ax2.set_ylabel("thrust", color="#2563eb")
    ax2.tick_params(axis="y", labelcolor="#2563eb")
    ax2b = ax2.twinx()
    rates = ["omega_1", "omega_2", "omega_3"]
    cols = ["#ea580c", "#16a34a", "#9333ea"]
    for j, (nm, col) in enumerate(zip(rates, cols), start=1):
        ax2b.plot(t, c[:, j], color=col, label=nm)
    ax2b.set_ylabel("body rates")
    ax2.set_title(r"Feedforward  $\lambda(\bar{x})$  along the trajectory")
    ax2.grid(alpha=0.3)
    lines = ax2.get_lines() + ax2b.get_lines()
    ax2.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="upper right")
    save("drone_nominal_trajectory.png")

#Figure 2, residual versus distance from the operating point
def fig_residual_vs_distance():
    s = build_drone()
    degrees = [1, 2]
    r_grid = np.linspace(0.0, 0.6, 9)
    n_dir = 4
    rng = np.random.default_rng(0)
    plt.figure(figsize=(7, 4.5))
    fresh_ref = None
    for d in degrees:
        nst = NSTJAX(s.fsys, s.fexo, s.hsys, s.n, s.m, s.p, s.n_, d, store=False)
        nst.warm_start(s.z0, s.x_0, samples=1)
        th0, la0 = nst.compute_fbi(s.z0, s.x_0)
        stale = np.zeros(len(r_grid)); fresh = np.zeros(len(r_grid))
        for i, r in enumerate(r_grid):
            sv = []; fv = []
            for _ in range(n_dir):
                dz = rng.standard_normal(s.nsum); dz /= np.linalg.norm(dz)
                dx = rng.standard_normal(s.n_); dx /= np.linalg.norm(dx)
                z = jnp.asarray(np.asarray(s.z0) + r * dz)
                x_ = jnp.asarray(np.asarray(s.x_0) + r * dx)
                f = nst._tay_f(z); h = nst._tay_h(z); fe = nst._tay_e(x_)
                sv.append(fbi_residual(th0, la0, f, h, fe, s.n, s.m, s.n_, d, s.nsum))
                thf, laf = nst.compute_fbi(z, x_)
                fv.append(fbi_residual(thf, laf, f, h, fe, s.n, s.m, s.n_, d, s.nsum))
            stale[i] = np.mean(sv); fresh[i] = np.mean(fv)
        plt.semilogy(r_grid, np.maximum(stale, 1e-16), "-o", label=f"stale solve, d={d}")
        if d == degrees[-1]:
            fresh_ref = fresh
    plt.semilogy(r_grid, np.maximum(fresh_ref, 1e-16), "--k", label="re-solved each point")
    plt.xlabel("distance from operating point")
    plt.ylabel("FBI residual")
    plt.title("Local manifold degrades, re-solving restores accuracy")
    plt.grid(alpha=0.3, which="both"); plt.legend()
    save("residual_vs_distance.png")

#Figure 4, solver time versus problem size on one nilpotent family
def fig_solver_crossover():
    n_list = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    d = 2
    kd, t_fbi, t_fast = [], [], []
    for n_ in n_list:
        s = build_synth_chain(n_, d)
        tay_f = build_taylor(s.fsys, s.nsum, d)
        tay_h = build_taylor(s.hsys, s.nsum, d)
        tay_e = build_taylor(s.fexo, s.n_, d)
        f = jax.block_until_ready(tay_f(s.z0))
        h = jax.block_until_ready(tay_h(s.z0))
        fe = jax.block_until_ready(tay_e(s.x_0))
        solver = FBIFast(s.n, s.m, s.p, s.n_, d, fixed=True)
        reps = 20 if n_ <= 12 else 8
        try:
            tf = _time(lambda: fbi(f, h, fe, s.n, s.m, s.p, s.n_, d), reps)
            tn = _time(lambda: solver.solve(f, h, fe)[:2], reps)
        except Exception as exc:
            print(f"  skip n_={n_}: {exc}")
            continue
        kd.append((s.n + s.p) * crd(s.n_, d))
        t_fbi.append(tf * 1e3); t_fast.append(tn * 1e3)

    plt.figure(figsize=(7, 4.5))
    plt.loglog(kd, t_fbi, "-o", label="fbi (dense kron)")
    plt.loglog(kd, t_fast, "-s", label="fbi_fast (decoupled)")
    plt.axvline(20e2, color="#9ca3af", ls="--", label="auto threshold")
    plt.xlabel("kron dimension  (n + p) * crd(n_, d)")
    plt.ylabel("warm solve time  (ms)")
    plt.title("Dense versus decoupled solve, exosystem dimension sweep")
    plt.grid(alpha=0.3, which="both"); plt.legend()
    save("solver_crossover.png")

#Figure 5, exosystem spectrum against transmission zeros
def fig_spectral_screen():
    s = build_screen_demo()
    tay_f = build_taylor(s.fsys, s.nsum, s.d)
    tay_h = build_taylor(s.hsys, s.nsum, s.d)
    tay_e = build_taylor(s.fexo, s.n_, s.d)
    f = np.asarray(tay_f(s.z0)); h = np.asarray(tay_h(s.z0)); fe = np.asarray(tay_e(s.x_0))
    nsum, n, m, p, n_ = s.nsum, s.n, s.m, s.p, s.n_
    F1 = f[:, :nsum]; H1 = h[:, :nsum]
    M_ = np.block([[F1[:, :n], F1[:, n:n + m]], [H1[:, :n], H1[:, n:n + m]]])
    Sel = np.zeros((n + p, n + m)); Sel[:n, :n] = np.eye(n)
    A_ = fe[:, :n_]
    _, _, mu, _ = schur_form(A_)
    zeros, _ = transmission_zeros(M_, Sel)

    plt.figure(figsize=(7, 5.5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, s.d))
    txt = []
    for k in range(1, s.d + 1):
        lam = operator_eigs(mu, n_, k)
        off = 0.015 * (k - (s.d + 1) / 2.0)
        plt.scatter(lam.real + off, lam.imag, color=cmap[k - 1], s=55,
                    edgecolors="white", linewidths=0.5, zorder=3,
                    label=f"operator eigs, degree {k}")
        if zeros.size:
            gap = float(np.min(np.abs(lam[:, None] - zeros[None, :])))
            txt.append(f"degree {k}: min gap {gap:.3f}")
    if zeros.size:
        plt.scatter(zeros.real, zeros.imag, marker="x", color="#dc2626", s=140,
                    linewidths=2.5, zorder=4, label="transmission zeros")
    plt.axhline(0, color="#d0d7de", lw=0.8); plt.axvline(0, color="#d0d7de", lw=0.8)
    plt.xlabel("real"); plt.ylabel("imag")
    plt.title("Resonance screen: exosystem spectrum vs transmission zeros")
    if txt:
        plt.gca().text(0.03, 0.03, "\n".join(txt), transform=plt.gca().transAxes,
                       fontsize=9, va="bottom",
                       bbox=dict(boxstyle="round", fc="white", ec="#d0d7de"))
    plt.grid(alpha=0.3); plt.legend(fontsize=8, loc="upper right")
    save("spectral_screen.png")

#Figure 6, per step timing breakdown and inference throughput
def fig_timing_breakdown():
    s = build_drone()
    tay_f = build_taylor(s.fsys, s.nsum, s.d)
    tay_h = build_taylor(s.hsys, s.nsum, s.d)
    tay_e = build_taylor(s.fexo, s.n_, s.d)
    solver = FBIFast(s.n, s.m, s.p, s.n_, s.d, fixed=True)

    f = tay_f(s.z0); h = tay_h(s.z0); fe = tay_e(s.x_0)
    th, la, _ = solver.solve(f, h, fe)
    th = jnp.reshape(th, (s.n, -1)); la = jnp.reshape(la, (s.m, -1))
    SAMPLES = 256
    W = 0.2 * jax.random.normal(jax.random.PRNGKey(0), (SAMPLES, s.n_))
    jax.block_until_ready((compute_theta(th, W, s.d), compute_lambda(la, W, s.d)))

    iters = 8
    tc = np.zeros(iters); ts = np.zeros(iters); ti = np.zeros(iters)
    for i in range(iters):
        t0 = time.perf_counter()
        f = tay_f(s.z0); h = tay_h(s.z0); fe = tay_e(s.x_0)
        jax.block_until_ready((f, h, fe))
        t1 = time.perf_counter()
        th, la, _ = solver.solve(f, h, fe)
        th = jnp.reshape(th, (s.n, -1)); la = jnp.reshape(la, (s.m, -1))
        jax.block_until_ready((th, la))
        t2 = time.perf_counter()
        pi = compute_theta(th, W, s.d); c = compute_lambda(la, W, s.d)
        jax.block_until_ready((pi, c))
        t3 = time.perf_counter()
        tc[i] = t1 - t0; ts[i] = t2 - t1; ti[i] = t3 - t2
    sl = slice(1, iters)
    seg = np.array([tc[sl].mean(), ts[sl].mean(), ti[sl].mean()]) * 1e3

    batches = [64, 128, 256, 512, 1024]
    thr = []
    for B in batches:
        Wb = 0.2 * jax.random.normal(jax.random.PRNGKey(B), (B, s.n_))
        jax.block_until_ready(compute_theta(th, Wb, s.d))
        t = time.perf_counter()
        for _ in range(20):
            out = compute_theta(th, Wb, s.d)
        jax.block_until_ready(out)
        dt = (time.perf_counter() - t) / 20
        thr.append(B / dt)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    names = ["coeffs", "fbi", "infer"]
    colors = ["#60a5fa", "#2563eb", "#1e3a8a"]
    bottom = 0.0
    for v, nm, col in zip(seg, names, colors):
        a1.bar(0, v, width=0.6, bottom=bottom, color=col, label=f"{nm}  {v:.2f} ms")
        bottom += v
    a1.set_xlim(-1, 1)
    a1.set_xticks([]); a1.set_ylabel("ms per step")
    a1.set_title(f"Per step cost, total {bottom:.2f} ms")
    a1.legend()
    a2.plot(batches, np.asarray(thr) / 1e6, "-o", color="#2563eb")
    a2.set_xlabel("batch size"); a2.set_ylabel("million samples / s")
    a2.set_title("Inference throughput")
    a2.grid(alpha=0.3)
    save("timing_breakdown.png")

#Figure 7, pendulum nominal phase portrait
def fig_pendulum_phase():
    s = build_pendulum()
    nst = NSTJAX(s.fsys, s.fexo, s.hsys, s.n, s.m, s.p, s.n_, s.d, store=True)
    nst.warm_start(s.z0, s.x_0, samples=1)
    th, la = nst.compute_fbi(s.z0, s.x_0)
    w0 = np.array([0.5, 0.0, 0.0])
    W = rollout(s.fexo, w0, steps=400, dt=0.02)
    Wj = jnp.asarray(W)
    pi = np.asarray(compute_theta(th, Wj, s.d))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(pi[:, 0], pi[:, 1], color="#2563eb", lw=1.6)
    a1.scatter(pi[0, 0], pi[0, 1], color="#16a34a", s=30)
    a1.set_xlabel("theta"); a1.set_ylabel("theta dot")
    a1.set_title("Nominal manifold phase portrait")
    a1.grid(alpha=0.3)
    t = np.arange(W.shape[0]) * 0.02
    a2.plot(t, pi[:, 0], color="#2563eb", lw=2.4, label=r"nominal  $\theta(\bar{x})_0$")
    a2.plot(t, W[:, 0], "--", color="#111827", lw=1.2, label=r"reference  $\bar{x}_0$")
    a2.set_xlabel("time"); a2.set_ylabel("angle")
    a2.set_title("Reference tracking on the manifold")
    a2.grid(alpha=0.3); a2.legend(fontsize=8)
    save("pendulum_phase.png")

#Figure 8, error against the MATLAB reference, optional
def fig_matlab_error():
    need = ["tests/x_test_batch.npy", "tests/th_val_mat.npy", "tests/la_val_mat.npy"]
    if not all(os.path.exists(f) for f in need):
        print("  skip matlab_error: reference arrays not found")
        return
    s = build_drone()
    nst = NSTJAX(s.fsys, s.fexo, s.hsys, s.n, s.m, s.p, s.n_, s.d, store=True)
    nst.warm_start(s.z0, s.x_0, samples=1)
    th, la = nst.compute_fbi(s.z0, s.x_0)
    Xm = jnp.asarray(np.load("tests/x_test_batch.npy"))
    th_ref = np.load("tests/th_val_mat.npy"); la_ref = np.load("tests/la_val_mat.npy")
    tv = np.asarray(compute_theta(th, Xm, s.d))
    lv = np.asarray(compute_lambda(la, Xm, s.d))
    eth = np.abs(tv - th_ref) / (np.max(np.abs(th_ref)) + 1e-30)
    ela = np.abs(lv - la_ref) / (np.max(np.abs(la_ref)) + 1e-30)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    im1 = a1.imshow(eth, aspect="auto", cmap="magma")
    a1.set_title("theta relative error"); a1.set_xlabel("state"); a1.set_ylabel("sample")
    fig.colorbar(im1, ax=a1)
    im2 = a2.imshow(ela, aspect="auto", cmap="magma")
    a2.set_title("lambda relative error"); a2.set_xlabel("control"); a2.set_ylabel("sample")
    fig.colorbar(im2, ax=a2)
    save("matlab_error.png")

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    figs = [fig_drone_trajectory, fig_residual_vs_distance, fig_solver_crossover,
            fig_spectral_screen, fig_timing_breakdown, fig_pendulum_phase,
            fig_matlab_error]
    for fn in figs:
        print(fn.__name__)
        try:
            fn()
        except Exception as exc:
            print(f"  failed: {exc}")

if __name__ == "__main__":
    main()
