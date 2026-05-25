"""
Author: gpertin, KAIST
World model learning with FBI: neural-ODE plant, decoupled tracking solve, on-policy data
"""

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import scipy.signal as sig
from NSTJAX.NSTJAX_suite.taylor import build_taylor
from NSTJAX.NSTJAX_suite.fbi import fbi
from NSTJAX.NSTJAX_suite.fbi_eval import compute_theta, compute_lambda

#System dimensions, pendulum tracking, degree 2
d = 2
n = 2
m = 1
p = 1
n_ = 3
nsum = n + m + n_

#True plant parameters, the unknown ground truth
G_TRUE, L_TRUE, B_TRUE, MASS, Q_TRUE = 9.81, 1.0, 0.1, 1.0, 0.5
#Nominal prior, deliberately wrong, the neural residual must correct it
G_NOM, L_NOM, B_NOM = 6.0, 1.0, 0.0

#Exosystem and reference
OM = 1.0
AMP = 0.3

#Loop and learning parameters
ROUNDS = 6
STEPS = 400
DT = 0.02
EVAL_STEPS = 200
EPOCHS = 400
LR = 1e-3
HIDDEN = 32
DITHER0 = 0.4
DECAY = 0.6
DITHER_FLOOR = 0.03
NOISE = 0.003
POLES = [-2.0, -3.0]

#Base operating point and reference start
z0 = jnp.zeros(nsum)
x_0 = jnp.zeros(n_)
x0 = jnp.array([0.0, 0.0])
w0 = jnp.array([AMP, 0.0, 0.0])

#Plant and exosystem fields
def fsys_true(x, u):
    th, thd = x[0], x[1]
    acc = (-G_TRUE / L_TRUE * jnp.sin(th)
           - B_TRUE / (MASS * L_TRUE**2) * thd
           + Q_TRUE * th * thd + u[0])
    return jnp.array([thd, acc])

def f_nominal(x, u):
    th, thd = x[0], x[1]
    acc = -G_NOM / L_NOM * jnp.sin(th) - B_NOM / (MASS * L_NOM**2) * thd + u[0]
    return jnp.array([thd, acc])

def fsys_true_z(z):
    return fsys_true(z[:n], z[n:n + m])

def fexo(xb):
    return jnp.array([-(OM + xb[2]) * xb[1], (OM + xb[2]) * xb[0], 0.0])

def hsys(z):
    x = z[:n]
    xb = z[n + m:]
    return jnp.array([x[0] - xb[0]])

#Fixed taylor maps for the structural parts
tay_h = build_taylor(hsys, nsum, d)
tay_e = build_taylor(fexo, n_, d)

#Neural network world model, residual on top of the nominal prior
def init_params(key, sizes):
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for i, (din, dout) in enumerate(zip(sizes[:-1], sizes[1:])):
        scale = 0.1 if i < len(sizes) - 2 else 0.01
        W = scale * jax.random.normal(keys[i], (dout, din))
        b = jnp.zeros((dout,))
        params.append((W, b))
    return params

def mlp_apply(params, x):
    h = x
    for W, b in params[:-1]:
        h = jnp.tanh(W @ h + b)
    W, b = params[-1]
    return W @ h + b

def plant_field(params, x, u):
    return f_nominal(x, u) + mlp_apply(params, jnp.concatenate([x, u]))

def make_fhat(params):
    def fhat(z):
        return plant_field(params, z[:n], z[n:n + m])
    return fhat

#Integrators
def rk4(field, x, dt):
    k1 = field(x)
    k2 = field(x + 0.5 * dt * k1)
    k3 = field(x + 0.5 * dt * k2)
    k4 = field(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

def step_true(x, u, dt):
    return rk4(lambda xx: fsys_true(xx, u), x, dt)

def step_model(params, x, u, dt):
    return rk4(lambda xx: plant_field(params, xx, u), x, dt)

def step_exo(w, dt):
    return rk4(fexo, w, dt)

#FBI solves
def solve_fbi(params):
    tay_f = build_taylor(make_fhat(params), nsum, d)
    f = tay_f(z0)
    h = tay_h(z0)
    fe = tay_e(x_0)
    th, la = fbi(f, h, fe, n, m, p, n_, d)
    return jnp.reshape(th, (n, -1)), jnp.reshape(la, (m, -1)), f

def solve_true():
    tay_f = build_taylor(fsys_true_z, nsum, d)
    f = tay_f(z0)
    h = tay_h(z0)
    fe = tay_e(x_0)
    th, la = fbi(f, h, fe, n, m, p, n_, d)
    return jnp.reshape(th, (n, -1)), jnp.reshape(la, (m, -1)), f

#Feedback gain by pole placement on the learned linearization
def gain(f):
    A = np.asarray(f[:, :n], dtype=np.float64)
    B = np.asarray(f[:, n:n + m], dtype=np.float64)
    Kpp = sig.place_poles(A, B, np.asarray(POLES, dtype=np.float64)).gain_matrix
    return jnp.asarray(-Kpp, dtype=f.dtype)

#Closed loop rollout on the true system, feedforward plus feedback plus dither
@partial(jax.jit, static_argnums=(7,))
def rollout(th, la, K, x_init, w_init, key, dither, steps, dt):
    def step(carry, _):
        x, w, key = carry
        key, sk = jax.random.split(key)
        pi = compute_theta(th, w[None], d)[0]
        c = compute_lambda(la, w[None], d)[0]
        u = c + K @ (x - pi) + dither * jax.random.normal(sk, (m,))
        x_next = step_true(x, u, dt)
        w_next = step_exo(w, dt)
        e = x[0] - w[0]
        return (x_next, w_next, key), (x, u, e)
    (xf, _, _), (xs, us, es) = jax.lax.scan(
        step, (x_init, w_init, key), None, length=steps)
    return xs, us, xf, es

#Exosystem path for the evaluation grid
@partial(jax.jit, static_argnums=(1,))
def exo_path(w_init, steps, dt):
    def step(w, _):
        return step_exo(w, dt), w
    _, ws = jax.lax.scan(step, w_init, None, length=steps)
    return ws

#Why useful, error of learned manifold and feedforward against the true solution
def fbi_errors(th, la, th_star, la_star, W):
    pi = compute_theta(th, W, d)
    pis = compute_theta(th_star, W, d)
    c = compute_lambda(la, W, d)
    cs = compute_lambda(la_star, W, d)
    em = float(jnp.max(jnp.abs(pi - pis)) / (jnp.max(jnp.abs(pis)) + 1e-12))
    ec = float(jnp.max(jnp.abs(c - cs)) / (jnp.max(jnp.abs(cs)) + 1e-12))
    return em, ec

#Neural-ODE training, one step rk4 prediction matched to the next state
def loss_fn(params, X, U, Y, dt):
    pred = jax.vmap(lambda x, u: step_model(params, x, u, dt))(X, U)
    return jnp.mean(jnp.sum((pred - Y)**2, axis=1))

def _adam_init(params):
    z = jax.tree_util.tree_map(jnp.zeros_like, params)
    return z, z, 0

def _adam_step(params, grads, state, lr):
    m_, v_, t = state
    t = t + 1
    m_ = jax.tree_util.tree_map(lambda a, g: 0.9 * a + 0.1 * g, m_, grads)
    v_ = jax.tree_util.tree_map(lambda a, g: 0.999 * a + 0.001 * g * g, v_, grads)
    mh = jax.tree_util.tree_map(lambda a: a / (1 - 0.9**t), m_)
    vh = jax.tree_util.tree_map(lambda a: a / (1 - 0.999**t), v_)
    params = jax.tree_util.tree_map(
        lambda p, a, b: p - lr * a / (jnp.sqrt(b) + 1e-8), params, mh, vh)
    return params, (m_, v_, t)

def train(params, X, U, Y, dt, epochs, lr):
    state = _adam_init(params)
    grad_fn = jax.jit(jax.grad(loss_fn))
    for _ in range(epochs):
        g = grad_fn(params, X, U, Y, dt)
        params, state = _adam_step(params, g, state, lr)
    return params

def _rms(es):
    return float(jnp.sqrt(jnp.mean(es**2)))

def main():
    print(f"dims: n={n} m={m} p={p} n_={n_} d={d} nsum={nsum}\n")

    #Ground truth FBI oracle, manifold and feedforward we want to recover
    th_star, la_star, f_true = solve_true()
    K_star = gain(f_true)
    W_eval = exo_path(w0, EVAL_STEPS, DT)
    es_oracle = rollout(th_star, la_star, K_star, x0, w0,
                        jax.random.PRNGKey(7), 0.0, STEPS, DT)[3]
    rms_oracle = _rms(es_oracle)

    #Nominal baseline, FBI on the wrong prior with no learning
    key = jax.random.PRNGKey(0)
    key, ki = jax.random.split(key)
    params = init_params(ki, [n + m, HIDDEN, n])
    th_nom, la_nom, _ = solve_fbi(params)
    em_nom, ec_nom = fbi_errors(th_nom, la_nom, th_star, la_star, W_eval)

    print("columns: rel-theta and rel-lambda are the learned manifold and")
    print("feedforward errors against the true FBI solution, track-rms is the")
    print("closed loop tracking error rolled out on the true system\n")
    print(f"  oracle  (true model)   track-rms {rms_oracle:.3e}")
    print(f"  nominal (no learning)  rel-theta {em_nom:.2e}   "
          f"rel-lambda {ec_nom:.2e}\n")

    X_all = U_all = Y_all = None
    hist = []
    for r in range(ROUNDS):
        th, la, f = solve_fbi(params)
        K = gain(f)
        em, ec = fbi_errors(th, la, th_star, la_star, W_eval)

        dither = max(DITHER_FLOOR, DITHER0 * DECAY**r)
        key, kr, kn = jax.random.split(key, 3)
        xs, us, xf, es = rollout(th, la, K, x0, w0, kr, dither, STEPS, DT)
        rms = _rms(es)
        hist.append((em, ec, rms))
        print(f"  round {r}  dither {dither:.3f}  rel-theta {em:.2e}   "
              f"rel-lambda {ec:.2e}   track-rms {rms:.3e}")

        #On policy data, slightly noisy, aggregated across rounds
        states = jnp.concatenate([xs, xf[None]], axis=0)
        states = states + NOISE * jax.random.normal(kn, states.shape)
        X, Y = states[:-1], states[1:]
        if X_all is None:
            X_all, U_all, Y_all = X, us, Y
        else:
            X_all = jnp.concatenate([X_all, X], axis=0)
            U_all = jnp.concatenate([U_all, us], axis=0)
            Y_all = jnp.concatenate([Y_all, Y], axis=0)
        params = train(params, X_all, U_all, Y_all, DT, EPOCHS, LR)

    #Final model after the last update
    th, la, f = solve_fbi(params)
    K = gain(f)
    em, ec = fbi_errors(th, la, th_star, la_star, W_eval)
    es = rollout(th, la, K, x0, w0, jax.random.PRNGKey(9), 0.0, STEPS, DT)[3]
    rms = _rms(es)
    print(f"\n  final   learned model   rel-theta {em:.2e}   "
          f"rel-lambda {ec:.2e}   track-rms {rms:.3e}")
    print(f"  oracle  lower bound     track-rms {rms_oracle:.3e}\n")

    #Optional convergence figure
    try:
        import matplotlib.pyplot as plt
        h = np.array(hist)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        ax[0].semilogy(range(ROUNDS), h[:, 0], "o-", label="rel-theta")
        ax[0].semilogy(range(ROUNDS), h[:, 1], "s-", label="rel-lambda")
        ax[0].set_xlabel("round"); ax[0].set_ylabel("error vs true FBI")
        ax[0].legend()
        ax[1].semilogy(range(ROUNDS), h[:, 2], "o-", label="track-rms")
        ax[1].axhline(rms_oracle, ls="--", color="k", label="oracle")
        ax[1].set_xlabel("round"); ax[1].set_ylabel("true tracking rms")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig("world_model_convergence.png", dpi=130)
        print("saved world_model_convergence.png")
    except Exception:
        pass

if __name__ == "__main__":
    main()
