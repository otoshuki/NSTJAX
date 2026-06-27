"""
Author: gpertin, KAIST
Solves the Francis-Byrnes-Isidori equations in JAX
Based on Krener's Nonlinear Systems Toolbox
"""

from functools import partial
import jax
import jax.numpy as jnp
from NSTJAX.NSTJAX_suite.polylib import crd, unpack, zero_field, compose, ddmul, lie_operator, comp_operator

def _vec(X):
    #Column major vectorization
    return X.T.reshape(-1)

def _unvec(v, r, c):
    #Inverse of _vec for an r by c matrix
    return v.reshape(c, r).T

def _solve(Op, rhs):
    #Square -> direct solve, otherwise least squares.
    if Op.shape[0] == Op.shape[1]:
        W = jnp.linalg.solve(Op, rhs)
    else:
        W = jnp.linalg.lstsq(Op, rhs, rcond=None)[0]
    return W

def _operator(M_, Sel, LF):
    #Degree k regulator operator for the equation M_ W - Sel W LF = RHS
    nk = LF.shape[0]
    eye = jnp.eye(nk, dtype=M_.dtype)
    return jnp.kron(eye, M_) - jnp.kron(LF.T, Sel)

#Partial compiled to improve speed
@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def fbi(f, h, fexo, n, m, p, n_, d, disc=False):
    nsum = n + m + n_
    dt = f.dtype
    #Linear parts
    F1 = f[:, :nsum]
    H1 = h[:, :nsum]
    Fxb = F1[:, n + m:nsum]
    Hxb = H1[:, n + m:nsum]
    A_ = fexo[:, :n_]
    #Combined linear plant output map and the selector onto the x block
    M_ = jnp.block([[F1[:, :n], F1[:, n:n + m]],
                    [H1[:, :n], H1[:, n:n + m]]])
    Sel = jnp.block([[jnp.eye(n, dtype=dt), jnp.zeros((n, m), dt)],
                     [jnp.zeros((p, n + m), dt)]])
    #Degree 1, the linear regulator equations
    Op = _operator(M_, Sel, A_)
    rhs = -_vec(jnp.concatenate([Fxb, Hxb], axis=0))
    W = _unvec(_solve(Op, rhs), n + m, n_)
    th = W[:n, :]
    la = W[n:, :]
    #Degrees 2..d, same operator with the lower degree right hand side
    fh = jnp.concatenate([f, h], axis=0)
    fh_field = unpack(fh, nsum, 1, d)
    fexo_field = unpack(fexo, n_, 1, d)
    for k in range(2, d + 1):
        n_k = crd(n_, k)
        LF = comp_operator(A_, n_, k) if disc else lie_operator(A_, n_, k)
        Op = _operator(M_, Sel, LF)
        #Term 1, composition cross term (discrete) or derivative cross term (continuous)
        th_field = unpack(th, n_, 1, k - 1)
        la_field = unpack(la, n_, 1, k - 1)
        deriv = (compose(th_field, n_, fexo_field, n_, k)[k] if disc
                 else ddmul(th_field, fexo_field, n_, k)[k])
        #Term 2, the nonlinear part of [f; h] composed with (th, la, x_)
        data_hi = zero_field(n + p, nsum, k, dtype=dt)
        for kk in range(2, k + 1):
            data_hi[kk] = fh_field[kk]
        g_sub = zero_field(nsum, n_, k, dtype=dt)
        for deg in range(1, k):
            g_sub[deg] = g_sub[deg].at[:n].set(th_field[deg])
            g_sub[deg] = g_sub[deg].at[n:n + m].set(la_field[deg])
        g_sub[1] = g_sub[1].at[n + m:nsum].set(jnp.eye(n_, dtype=dt))
        comp = compose(data_hi, nsum, g_sub, n_, k)[k]
        #Assemble and solve
        rhs_F = deriv - comp[:n]
        rhs_H = -comp[n:]
        rhs = _vec(jnp.concatenate([rhs_F, rhs_H], axis=0))
        W = _unvec(_solve(Op, rhs), n + m, n_k)
        th = jnp.concatenate([th, W[:n, :]], axis=1)
        la = jnp.concatenate([la, W[n:, :]], axis=1)

    return th, la
