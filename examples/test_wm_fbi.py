"""
Author: gpertin, KAIST
Sanity tests for the world model FBI example
"""

import numpy as np
import jax
import jax.numpy as jnp
import example_wm_fbi as ex

def test_model_init():
    params = ex.init_params(jax.random.PRNGKey(0), [ex.n + ex.m, 32, ex.n])
    out = ex.mlp_apply(params, jnp.zeros(ex.n + ex.m))
    assert out.shape == (ex.n,)
    assert float(jnp.max(jnp.abs(out))) < 1e-1

def test_fbi_shapes():
    params = ex.init_params(jax.random.PRNGKey(0), [ex.n + ex.m, 32, ex.n])
    th, la, f = ex.solve_fbi(params)
    assert th.shape[0] == ex.n and la.shape[0] == ex.m
    assert np.all(np.isfinite(np.asarray(th)))
    assert np.all(np.isfinite(np.asarray(la)))

def test_true_self_error():
    th_s, la_s, _ = ex.solve_true()
    W = ex.exo_path(ex.w0, 50, ex.DT)
    em, ec = ex.fbi_errors(th_s, la_s, th_s, la_s, W)
    assert em < 1e-5 and ec < 1e-5

def test_gain_places_poles():
    _, _, f_s = ex.solve_true()
    K = np.asarray(ex.gain(f_s))
    A = np.asarray(f_s[:, :ex.n], dtype=np.float64)
    B = np.asarray(f_s[:, ex.n:ex.n + ex.m], dtype=np.float64)
    ev = np.sort(np.linalg.eigvals(A + B @ K).real)
    assert np.allclose(ev, np.sort(ex.POLES), atol=1e-3)

def test_rollout_shapes():
    th_s, la_s, f_s = ex.solve_true()
    K = ex.gain(f_s)
    xs, us, xf, es = ex.rollout(th_s, la_s, K, ex.x0, ex.w0,
                                jax.random.PRNGKey(1), 0.0, 100, ex.DT)
    assert xs.shape == (100, ex.n) and us.shape == (100, ex.m)
    assert xf.shape == (ex.n,) and es.shape == (100,)
    assert np.all(np.isfinite(np.asarray(es)))

def test_one_round_learns():
    params = ex.init_params(jax.random.PRNGKey(0), [ex.n + ex.m, 32, ex.n])
    th, la, f = ex.solve_fbi(params)
    K = ex.gain(f)
    xs, us, xf, es = ex.rollout(th, la, K, ex.x0, ex.w0,
                                jax.random.PRNGKey(2), 0.3, 100, ex.DT)
    states = jnp.concatenate([xs, xf[None]], axis=0)
    X, Y = states[:-1], states[1:]
    l0 = float(ex.loss_fn(params, X, us, Y, ex.DT))
    params2 = ex.train(params, X, us, Y, ex.DT, 30, 1e-3)
    l1 = float(ex.loss_fn(params2, X, us, Y, ex.DT))
    assert l1 <= l0 + 1e-6

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} ok")
    print("all tests passed")
