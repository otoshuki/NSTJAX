"""
Author: gpertin, KAIST
High level NSTJAX example: planar drone with a cable suspended payload, load trajectory tracking
"""

import time
from functools import partial
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import scipy.linalg as sla
from NSTJAX import NSTJAX

#Loop params
SAMPLES = 256         #Inference samples for the warm start
DT = 0.01             #Rollout step
T_END = 12.566        #One closed 2:3 Lissajous period
STEPS = int(T_END / DT)

#System Design: planar drone with a rigid cable suspended payload
#The regulated output is the payload position, not the drone position
d = 3
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

def _print_report(rep):
    print(f"solvability: regime {rep['regime']}  "
          f"solvable {rep.get('solvable')}  resonant {rep.get('resonant')}")
    if rep.get("checked"):
        for k, dd in rep["degrees"].items():
            print(f"  degree {k}  min_gap {dd['min_gap']:.3f}  "
                  f"resonant {dd['resonant']}")

def _results_fig(t, xs, us, es, W, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lx = xs[:, 0] + L * np.sin(xs[:, 5])
    lz = xs[:, 1] - L * np.cos(xs[:, 5])
    rx, rz = W[:, 1], -L + W[:, 3]
    enorm = np.linalg.norm(es, axis=1)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].plot(rx, rz, "--", color="0.6", label="reference")
    ax[0, 0].plot(lx, lz, "-", color="tab:blue", lw=1.4, label="load")
    ax[0, 0].set_aspect("equal")
    ax[0, 0].legend()
    ax[0, 0].set_xlabel("x [m]")
    ax[0, 0].set_ylabel("z [m]")
    ax[0, 0].set_title("payload path")
    ax[0, 1].plot(t, enorm, color="tab:red")
    ax[0, 1].set_xlabel("t [s]")
    ax[0, 1].set_ylabel("|e| [m]")
    ax[0, 1].set_title("payload tracking error")
    ax[1, 0].plot(t, np.degrees(xs[:, 5]), color="tab:green")
    ax[1, 0].set_xlabel("t [s]")
    ax[1, 0].set_ylabel("swing [deg]")
    ax[1, 0].set_title("cable angle")
    axb = ax[1, 1]
    axc = axb.twinx()
    lb, = axb.plot(t, us[:, 0], color="tab:purple", label="thrust")
    lc, = axc.plot(t, us[:, 1], color="tab:orange", label="pitch rate")
    axb.set_xlabel("t [s]")
    axb.set_ylabel("thrust [N]")
    axc.set_ylabel("pitch rate [rad/s]")
    axb.set_title("control")
    axb.legend(handles=[lb, lc], loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"saved {fname}")

def _animate(xs, W, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    lx = xs[:, 0] + L * np.sin(xs[:, 5])
    lz = xs[:, 1] - L * np.cos(xs[:, 5])
    rx, rz = W[:, 1], -L + W[:, 3]
    step = max(1, len(xs) // 220)
    idx = np.arange(0, len(xs), step)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    pad = 0.4
    ax.set_xlim(min(lx.min(), rx.min()) - pad, max(lx.max(), rx.max()) + pad)
    ax.set_ylim(min(lz.min(), rz.min()) - pad, max(xs[:, 1].max(), 0.0) + pad)
    ax.plot(rx, rz, "--", color="0.75", lw=1, label="reference")
    refdot, = ax.plot([], [], "o", color="tab:orange", ms=7, label="reference load")
    trail, = ax.plot([], [], "-", color="tab:blue", lw=1.2, alpha=0.7)
    body, = ax.plot([], [], "-", color="k", lw=3)
    cable, = ax.plot([], [], "-", color="0.4", lw=1)
    load, = ax.plot([], [], "o", color="tab:blue", ms=9)
    thrust, = ax.plot([], [], "-", color="tab:red", lw=1.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Planar drone tracking a suspended payload")
    bl = 0.18
    def upd(i):
        k = idx[i]
        px, pz, phi = xs[k, 0], xs[k, 1], xs[k, 4]
        body.set_data([px - bl * np.cos(phi), px + bl * np.cos(phi)],
                      [pz - bl * np.sin(phi), pz + bl * np.sin(phi)])
        cable.set_data([px, lx[k]], [pz, lz[k]])
        load.set_data([lx[k]], [lz[k]])
        thrust.set_data([px, px - 0.3 * np.sin(phi)], [pz, pz + 0.3 * np.cos(phi)])
        refdot.set_data([rx[k]], [rz[k]])
        trail.set_data(lx[:k + 1], lz[:k + 1])
        return body, cable, load, thrust, refdot, trail
    anim = FuncAnimation(fig, upd, frames=len(idx), interval=40, blit=True)
    anim.save(fname, writer=PillowWriter(fps=25))
    plt.close(fig)
    print(f"saved {fname}")

def main():
    print(f"dims: n={n} m={m} p={p} n_={n_} d={d} nsum={nsum}\n")

    #High level solve, the harmonic exosystem keeps this on the dense path
    #Output regulation onto the payload makes the swing a transmission zero,
    #so the solvability screen is turned on
    nst = NSTJAX(fsys, fexo, hsys, n, m, p, n_, d, check="setup", verbose=True)
    nst.warm_start(z0, x_0, samples=SAMPLES)
    print(f"theta shape {nst.th.shape}   lambda shape {nst.la.shape}")
    _print_report(nst.report)

    t0 = time.perf_counter()
    nst.compute_fbi(z0, x_0)
    jax.block_until_ready((nst.th, nst.la))
    print(f"warm solve {(time.perf_counter() - t0) * 1e3:.3f} ms\n")

    #Generate the reference and evaluate the manifold and feedforward along it
    W = _exo_path(w0, STEPS, DT)
    Theta = nst.compute_theta(W)
    Lam = nst.compute_lambda(W)
    Kfb = _gain()

    #Roll out the closed loop on the true plant, starting on the manifold
    x_init = x0 + Theta[0]
    xs, us, es = _rollout(x_init, Theta, Lam, W, Kfb)
    xs, us, es = jax.block_until_ready((xs, us, es))
    xs, us, es, Wn = np.asarray(xs), np.asarray(us), np.asarray(es), np.asarray(W)

    rms = float(np.sqrt(np.mean(np.sum(es**2, axis=1))))
    print(f"closed loop payload tracking rms {rms:.3e} m")
    print(f"peak swing {np.degrees(np.max(np.abs(xs[:, 5]))):.2f} deg\n")

    #Figures and animation
    t = np.arange(STEPS) * DT
    try:
        _results_fig(t, xs, us, es, Wn, "payload_results.png")
        _animate(xs, Wn, "payload_tracking.gif")
    except Exception as exc:
        print(f"plotting skipped: {exc}")

if __name__ == "__main__":
    main()
