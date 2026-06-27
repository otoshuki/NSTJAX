"""
Author: gpertin, KAIST
High level NSTJAX example: manifold degree comparison on the swing drone payload tracker
"""

import os
import time
from functools import partial
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import scipy.linalg as sla
from NSTJAX import NSTJAX
from NSTJAX.NSTJAX_suite import build_taylor

#Loop params
SAMPLES = 256
DT = 0.01
T_END = 12.566        #One closed 2:3 Lissajous period
STEPS = int(T_END / DT)
DEGREES = [1, 2, 3, 4]   #Manifold degrees to compare
WARM = 3              #Untimed reps to settle before timing
REPS = 20             #Timed inference reps
FIGDIR = "docs/figures"

#System Design: planar drone with a rigid cable suspended payload
#The regulated output is the payload position, not the drone position
G = 9.81
MQ = 1.0             #Drone mass
ML = 0.2             #Payload mass
L = 1.0              #Cable length
M = MQ + ML
n = 7                 #States px, pz, vx, vz, phi, alpha, alphadot
m = 2                 #Inputs thrust, pitch rate
p = 2                 #Outputs payload x and z error
n_ = 4                #Exosystem, two oscillator pairs
nsum = n + m + n_

#Reference exosystem, load Lissajous, frequencies kept clear of the swing zero
OM1 = 2.0
OM2 = 0.2
AX = 1.0
AZ = 0.8

#Operating point, hover with the cable taut and nonzero thrust f0 = M g
x0 = jnp.zeros(n)
u0 = jnp.array([M * G, 0.0])
x_0 = jnp.zeros(n_)
z0 = jnp.concatenate([x0, u0, x_0])
w0 = jnp.array([AX, 0.0, AZ, 0.0])

#Coupled drone load accelerations, the swing reacts back on the drone
def _accel(x, u):
    phi, al, ald = x[4], x[5], x[6]
    f = u[0]
    sa, ca = jnp.sin(al), jnp.cos(al)
    a = M - ML * ca**2
    b = -ML * sa * ca
    c = M - ML * sa**2
    det = M * MQ
    r1 = -f * jnp.sin(phi) + ML * G * sa * ca + ML * L * ald**2 * sa
    r2 = f * jnp.cos(phi) - M * G + ML * G * sa**2 - ML * L * ald**2 * ca
    ax = (c * r1 - b * r2) / det
    az = (a * r2 - b * r1) / det
    ala = (-G * sa - ca * ax - sa * az) / L
    return ax, az, ala

#Plant rate, fsys ignores the exosystem block, the load couples only through the swing
def _rate(x, u):
    ax, az, ala = _accel(x, u)
    return jnp.array([x[2], x[3], ax, az, u[1], x[6], ala])

def fsys(z):
    return _rate(z[:n], z[n:n + m])

#Two harmonic oscillators, references are the sin components
def fexo(xb):
    return jnp.array([-OM1 * xb[1], OM1 * xb[0], -OM2 * xb[3], OM2 * xb[2]])

#Payload position error against the reference deviation around the hung load
def hsys(z):
    x = z[:n]
    xb = z[n + m:]
    al = x[5]
    return jnp.array([x[0] + L * jnp.sin(al) - xb[1],
                      x[1] - L * jnp.cos(al) + L - xb[3]])

#Stabilizing feedback by LQR on the hover linearization
def _gain():
    A = np.asarray(jax.jacfwd(_rate, 0)(x0, u0), dtype=np.float64)
    B = np.asarray(jax.jacfwd(_rate, 1)(x0, u0), dtype=np.float64)
    Q = np.diag([20.0, 20.0, 2.0, 2.0, 1.0, 10.0, 2.0])
    R = np.diag([0.1, 0.1])
    P = sla.solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return jnp.asarray(-K)

def _rk4(x, u, dt):
    k1 = _rate(x, u)
    k2 = _rate(x + 0.5 * dt * k1, u)
    k3 = _rate(x + 0.5 * dt * k2, u)
    k4 = _rate(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

#Autonomous exosystem path, the reference is generated ahead of the loop
@partial(jax.jit, static_argnums=(1,))
def _exo_path(w_init, steps, dt):
    def step(w, _):
        k1 = fexo(w)
        k2 = fexo(w + 0.5 * dt * k1)
        k3 = fexo(w + 0.5 * dt * k2)
        k4 = fexo(w + dt * k3)
        return w + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), w
    _, ws = jax.lax.scan(step, w_init, None, length=steps)
    return ws

#Closed loop on the true plant, feedforward plus feedback onto the manifold
@jax.jit
def _rollout(x_init, Theta, Lam, W, Kfb):
    def step(x, inp):
        th_t, la_t, w_t = inp
        u = u0 + la_t + Kfb @ (x - th_t)
        e = hsys(jnp.concatenate([x, u, w_t]))
        return _rk4(x, u, DT), (x, u, e)
    _, out = jax.lax.scan(step, x_init, (Theta, Lam, W))
    return out

#Solve, time the three phases, roll out one closed loop at a single degree
def run_case(d, W, Kfb):
    #Setup, structural build of the Taylor maps and the solver
    t0 = time.perf_counter()
    nst = NSTJAX(fsys, fexo, hsys, n, m, p, n_, d, check="setup")
    build_taylor(fsys, nsum, d)
    build_taylor(hsys, nsum, d)
    build_taylor(fexo, n_, d)
    t_setup = time.perf_counter() - t0

    #Warmup, warm_start rebuilds from cache and JIT compiles solve and inference
    t0 = time.perf_counter()
    nst.warm_start(z0, x_0, samples=STEPS)
    jax.block_until_ready((nst.th, nst.la))
    t_warm = time.perf_counter() - t0

    #Settle before timing
    for _ in range(WARM):
        nst.compute_fbi(z0, x_0)
        nst.compute_theta(W)
        nst.compute_lambda(W)
    jax.block_until_ready((nst.th, nst.la))

    #Inference, steady state solve and batched evaluate at the operating point
    t_steps = np.zeros(REPS)
    for i in range(REPS):
        t0 = time.perf_counter()
        nst.compute_fbi(z0, x_0)
        Theta = nst.compute_theta(W)
        Lam = nst.compute_lambda(W)
        jax.block_until_ready((Theta, Lam))
        t_steps[i] = time.perf_counter() - t0
    t_infer = float(t_steps.mean())

    #Closed loop rollout on the true plant for this degree
    nst.compute_fbi(z0, x_0)
    Theta = nst.compute_theta(W)
    Lam = nst.compute_lambda(W)
    x_init = x0 + Theta[0]
    xs, us, es = _rollout(x_init, Theta, Lam, W, Kfb)
    xs, es = jax.block_until_ready((xs, es))
    xs, es = np.asarray(xs), np.asarray(es)
    rms = float(np.sqrt(np.mean(np.sum(es**2, axis=1))))
    return dict(d=d, t_setup=t_setup, t_warm=t_warm, t_infer=t_infer,
                xs=xs, es=es, rms=rms, report=nst.report)

def _print_table(cases):
    print("\n  degree |  setup ms |  warmup ms | infer ms |   rms m")
    print("  -------+-----------+------------+----------+----------")
    for c in cases:
        print(f"    {c['d']:>4} | {c['t_setup']*1e3:9.2f} | "
              f"{c['t_warm']*1e3:10.2f} | {c['t_infer']*1e3:8.3f} | "
              f"{c['rms']:.2e}")

#Overlaid tracking error norm for the three degrees
def _error_fig(t, cases, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {1: "tab:red", 2: "tab:orange", 3: "tab:green"}
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in cases:
        enorm = np.linalg.norm(c["es"], axis=1)
        ax.plot(t, enorm, color=colors.get(c["d"], None), lw=1.4,
                label=f"degree {c['d']}")
    ax.set_yscale("log")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("|e| [m]")
    ax.set_ylim(10e-5, 0)
    ax.set_title("Payload tracking error by manifold degree")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"saved {fname}")

#Single gif overlaying the reference load and the three degree assemblies
def _animate_compare(cases, W, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    rx, rz = W[:, 1], -L + W[:, 3]
    colors = {1: "tab:red", 2: "tab:orange", 3: "tab:green"}
    nc = len(cases)
    geo = []
    for i, c in enumerate(cases):
        xs = c["xs"]
        px, pz, phi = xs[:, 0], xs[:, 1], xs[:, 4]
        lx = px + L * np.sin(xs[:, 5])
        lz = pz - L * np.cos(xs[:, 5])
        al = 0.3 + 0.7 * (i / max(1, nc - 1))
        geo.append(dict(d=c["d"], px=px, pz=pz, phi=phi, lx=lx, lz=lz, al=al))
    N = len(rx)
    step = max(1, N // 220)
    idx = np.arange(0, N, step)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    allx = np.concatenate([rx] + [g["lx"] for g in geo] + [g["px"] for g in geo])
    allz = np.concatenate([rz] + [g["lz"] for g in geo] + [g["pz"] for g in geo])
    pad = 0.4
    ax.set_xlim(allx.min() - pad, allx.max() + pad)
    ax.set_ylim(allz.min() - pad, allz.max() + pad)
    ax.plot(rx, rz, "--", color="0.7", lw=1, label="reference")
    refdot, = ax.plot([], [], "o", color="k", ms=7, label="reference load")
    arts_by_case = []
    for g in geo:
        col = colors.get(g["d"], None)
        body, = ax.plot([], [], "-", color=col, lw=3, alpha=g["al"])
        cable, = ax.plot([], [], "-", color=col, lw=1, alpha=g["al"])
        load, = ax.plot([], [], "o", color=col, ms=8, alpha=g["al"],
                        label=f"degree {g['d']}")
        trail, = ax.plot([], [], "-", color=col, lw=1.2, alpha=0.7 * g["al"])
        arts_by_case.append((body, cable, load, trail))
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Payload tracking: manifold degree 1 vs 2 vs 3")
    bl = 0.18

    def upd(i):
        k = idx[i]
        refdot.set_data([rx[k]], [rz[k]])
        arts = [refdot]
        for g, (body, cable, load, trail) in zip(geo, arts_by_case):
            px, pz, phi = g["px"][k], g["pz"][k], g["phi"][k]
            body.set_data([px - bl * np.cos(phi), px + bl * np.cos(phi)],
                          [pz - bl * np.sin(phi), pz + bl * np.sin(phi)])
            cable.set_data([px, g["lx"][k]], [pz, g["lz"][k]])
            load.set_data([g["lx"][k]], [g["lz"][k]])
            trail.set_data(g["lx"][:k + 1], g["lz"][:k + 1])
            arts += [body, cable, load, trail]
        return arts

    anim = FuncAnimation(fig, upd, frames=len(idx), interval=40, blit=True)
    anim.save(fname, writer=PillowWriter(fps=25))
    plt.close(fig)
    print(f"saved {fname}")

def main():
    print(f"dims: n={n} m={m} p={p} n_={n_} nsum={nsum}  degrees {DEGREES}\n")
    os.makedirs(FIGDIR, exist_ok=True)
    Kfb = _gain()
    W = jax.block_until_ready(_exo_path(w0, STEPS, DT))

    cases = []
    for d in DEGREES:
        c = run_case(d, W, Kfb)
        rep = c["report"]
        print(f"  degree {d}: regime {rep['regime']}  "
              f"solvable {rep.get('solvable')}  resonant {rep.get('resonant')}")
        cases.append(c)
    _print_table(cases)

    t = np.arange(STEPS) * DT
    Wn = np.asarray(W)
    try:
        _error_fig(t, cases, f"{FIGDIR}/payload_degree_error.png")
        _animate_compare(cases, Wn, f"{FIGDIR}/payload_degree_compare.gif")
    except Exception as exc:
        print(f"plotting skipped: {exc}")

if __name__ == "__main__":
    main()
