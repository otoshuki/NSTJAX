"""
Author: gpertin, KAIST
Polynomial vector field algebra library
"""

import numpy as np
import jax
import jax.numpy as jnp
from itertools import combinations_with_replacement
from functools import lru_cache, partial
from math import comb

#Combinatorial functions
def crd(n, k):
    #Follows Krener's design
    #Number of degree k monomials in n variables
    if k == 0:
        return 1
    return comb(n + k - 1, k)

def crdsum(n, d0, d1):
    #Follows Krener's design
    #Number of monomials of degrees d0 through d1 in n variables
    return comb(n + d1, d1) - comb(n + d0 - 1, d0 - 1)

@lru_cache(maxsize=None)
def factor_lists(n, k):
    #Sorted index tuples of length k over 0..n-1 in lex order
    if k == 0:
        return ((),)
    return tuple(combinations_with_replacement(range(n), k))

@lru_cache(maxsize=None)
def monomials(n, k):
    #Exponent table of degree k monomials, shape (crd(n,k), n), lex order
    E = np.zeros((crd(n, k), n), dtype=np.int64)
    for r, fl in enumerate(factor_lists(n, k)):
        for idx in fl:
            E[r, idx] += 1
    return E

@lru_cache(maxsize=None)
def _pos_map(n, k):
    #Map from exponent tuple to its column index in monomials(n,k)
    return {tuple(row): r for r, row in enumerate(monomials(n, k))}

@lru_cache(maxsize=None)
def polymul_map(n, i, j):
    #For degree i times degree j, position of each product monomial in degree i+j
    Ei = monomials(n, i)
    Ej = monomials(n, j)
    pos = _pos_map(n, i + j)
    out = np.zeros((Ei.shape[0], Ej.shape[0]), dtype=np.int64)
    for a in range(Ei.shape[0]):
        for b in range(Ej.shape[0]):
            out[a, b] = pos[tuple(Ei[a] + Ej[b])]
    return out

@lru_cache(maxsize=None)
def deriv_map(n, k, j):
    #Differentiate degree k monomials by variable j
    E = monomials(n, k)
    aj = E[:, j]
    keep = np.nonzero(aj > 0)[0]
    if keep.size == 0:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64))
    pos = _pos_map(n, k - 1)
    dst = np.empty(keep.size, np.int64)
    for t, r in enumerate(keep):
        beta = E[r].copy()
        beta[j] -= 1
        dst[t] = pos[tuple(beta)]
    return (keep, dst, aj[keep])

#Flat layout helpers
@lru_cache(maxsize=None)
def _offsets(n, dmax):
    #Flat start offset of each degree, offs[k] begins degree k
    offs = [0]
    for k in range(dmax + 1):
        offs.append(offs[-1] + crd(n, k))
    return tuple(offs)

@lru_cache(maxsize=None)
def _flatlen(n, dmax):
    #Total number of monomials over degrees 0..dmax
    return int(sum(crd(n, k) for k in range(dmax + 1)))

def _pack_full(F, n, dmax):
    #Pack a possibly short field into (nout, flatlen) over degrees 0..dmax
    blocks = [F[k] for k in range(min(len(F), dmax + 1))]
    flat = jnp.concatenate(blocks, axis=1)
    have = flat.shape[1]
    want = _flatlen(n, dmax)
    if have < want:
        flat = jnp.concatenate(
            [flat, jnp.zeros((flat.shape[0], want - have), flat.dtype)], axis=1)
    return flat

def _unpack_full(mat, n, dmax):
    #Split a packed matrix into graded blocks over degrees 0..dmax
    off = _offsets(n, dmax)
    return [mat[:, off[k]:off[k + 1]] for k in range(dmax + 1)]

def _active_degrees(F, dmax):
    #Degrees present in F up to dmax
    return tuple(k for k in range(min(len(F), dmax + 1)) if F[k].size)

#Packed matrix and graded block conversions
def zero_field(nout, n, dmax, dtype=None):
    #Graded field of zeros with degrees 0..dmax
    kw = {} if dtype is None else {"dtype": dtype}
    return [jnp.zeros((nout, crd(n, k)), **kw) for k in range(dmax + 1)]

def zero_scalar(n, dmax, dtype=None):
    #Graded scalar field of zeros with degrees 0..dmax
    kw = {} if dtype is None else {"dtype": dtype}
    return [jnp.zeros(crd(n, k), **kw) for k in range(dmax + 1)]

def const_scalar(n, dmax, dtype=None):
    #Scalar field equal to the constant one
    f = zero_scalar(n, dmax, dtype)
    f[0] = f[0].at[0].set(1.0)
    return f

def pack(blocks, d0, d1):
    #Concatenate degree blocks d0..d1 into a packed coefficient matrix
    return jnp.concatenate([blocks[k] for k in range(d0, d1 + 1)], axis=1)

def unpack(mat, n, d0, d1):
    #Split a packed matrix into graded blocks, zero filled below d0
    nout = mat.shape[0]
    blocks = [jnp.zeros((nout, crd(n, k)), mat.dtype) for k in range(d1 + 1)]
    c = 0
    for k in range(d0, d1 + 1):
        w = crd(n, k)
        blocks[k] = mat[:, c:c + w]
        c += w
    return blocks

#Monomial evaluation, the general degree mon basis
def eval_monomials(x, n, d0, d1):
    #values of all monomials of x of degrees d0..d1 in lex order
    parts = []
    for k in range(d0, d1 + 1):
        E = jnp.asarray(monomials(n, k))
        parts.append(jnp.prod(x[None, :] ** E, axis=1))
    return jnp.concatenate(parts)

#Polynomial product functions
def polymul_field(P, Q, n, dmax):
    #Truncated product of two scalar fields over the same n variables
    out = zero_scalar(n, dmax)
    for i in range(len(P)):
        for j in range(len(Q)):
            k = i + j
            if k > dmax:
                continue
            pos = jnp.asarray(polymul_map(n, i, j).ravel())
            contrib = (P[i][:, None] * Q[j][None, :]).ravel()
            out[k] = out[k].at[pos].add(contrib)
    return out


def polymulvec(P, Q, n, dmax):
    #Product of a vector field P and a scalar field Q over the same n variables
    nout = P[0].shape[0]
    out = zero_field(nout, n, dmax)
    for i in range(len(P)):
        for j in range(len(Q)):
            k = i + j
            if k > dmax:
                continue
            pos = jnp.asarray(polymul_map(n, i, j).ravel())
            contrib = (P[i][:, :, None] * Q[j][None, None, :]).reshape(nout, -1)
            out[k] = out[k].at[:, pos].add(contrib)
    return out


def partialfield(F, n, j):
    #Partial derivative of a field by variable j, degrees shift down by one
    D = len(F) - 1
    nout = F[0].shape[0]
    res = [jnp.zeros((nout, crd(n, k))) for k in range(D + 1)]
    for k in range(1, D + 1):
        src, dst, scale = deriv_map(n, k, j)
        if src.size == 0:
            continue
        block = F[k][:, jnp.asarray(src)] * jnp.asarray(scale, dtype=F[k].dtype)[None, :]
        res[k - 1] = res[k - 1].at[:, jnp.asarray(dst)].add(block)
    return res


@lru_cache(maxsize=None)
def _g_table(n_new, dmax):
    #Per degree exponent table and flat base column of the substitution G
    off = _offsets(n_new, dmax)
    degs, exps, bases = [], [], []
    for kg in range(dmax + 1):
        E = monomials(n_new, kg)
        for b in range(E.shape[0]):
            degs.append(kg); exps.append(E[b]); bases.append(off[kg] + b)
    exp = (np.stack(exps).astype(np.int64)
           if exps else np.zeros((0, n_new), np.int64))
    return np.array(degs, np.int64), exp, np.array(bases, np.int64)


@lru_cache(maxsize=None)
def _compose_plan(n_inter, n_new, dmax, fdegs):
    #Enumerate which G factors land in which output slot for each F column
    off_in = _offsets(n_inter, dmax)
    off_new = _offsets(n_new, dmax)
    fl_new = _flatlen(n_new, dmax)
    #Sentinel gathers the appended 1.0
    sentinel = n_inter * fl_new
    gm_deg, gm_exp, gm_base = _g_table(n_new, dmax)
    posmaps = [_pos_map(n_new, k) for k in range(dmax + 1)]

    fcol_all, gidx_all, opos_all = [], [], []
    for kf in fdegs:
        for r, fl in enumerate(factor_lists(n_inter, kf)):
            fcol = off_in[kf] + r
            exp = np.zeros((1, n_new), np.int64)
            deg = np.zeros((1,), np.int64)
            gid = np.zeros((1, 0), np.int64)
            #vectorized cartesian product over the factors
            for var in fl:
                S, Gn = exp.shape[0], gm_deg.shape[0]
                nexp = (exp[:, None, :] + gm_exp[None, :, :]).reshape(S * Gn, n_new)
                ndeg = (deg[:, None] + gm_deg[None, :]).reshape(S * Gn)
                gflat = var * fl_new + gm_base
                ngid = np.concatenate(
                    [np.repeat(gid, Gn, axis=0), np.tile(gflat, S)[:, None]], axis=1)
                keep = ndeg <= dmax
                exp, deg, gid = nexp[keep], ndeg[keep], ngid[keep]
            for t in range(exp.shape[0]):
                dd = int(deg[t])
                opos_all.append(off_new[dd] + posmaps[dd][tuple(exp[t])])
                fcol_all.append(fcol)
                gidx_all.append(list(gid[t]) + [sentinel] * (dmax - gid.shape[1]))

    fcol = np.array(fcol_all, np.int64)
    gidx = (np.array(gidx_all, np.int64).reshape(-1, dmax)
            if fcol_all else np.zeros((0, dmax), np.int64))
    opos = np.array(opos_all, np.int64)
    return fcol, gidx, opos


@lru_cache(maxsize=None)
def _ddmul_plan(n, dmax, fdegs):
    #Enumerate the gather multiply scatter for h = sum_j (dF/dx_j) * G_j
    off = _offsets(n, dmax)
    fl = _flatlen(n, dmax)
    posmaps = [_pos_map(n, k) for k in range(dmax + 1)]
    fpos, gpos, opos, scale = [], [], [], []
    for kf in fdegs:
        #d/dx of a constant is 0
        if kf == 0:
            continue
        E = monomials(n, kf)
        for r in range(E.shape[0]):
            alpha = E[r]; fcol = off[kf] + r
            for j in range(n):
                aj = int(alpha[j])
                if aj == 0:
                    continue
                beta = alpha.copy(); beta[j] -= 1; dkf = kf - 1
                for kg in range(dmax + 1):
                    od = dkf + kg
                    if od > dmax:
                        continue
                    Eg = monomials(n, kg)
                    for bg in range(Eg.shape[0]):
                        out_exp = beta + Eg[bg]
                        fpos.append(fcol)
                        gpos.append(j * fl + off[kg] + bg)
                        opos.append(off[od] + posmaps[od][tuple(out_exp)])
                        scale.append(aj)
    return (np.array(fpos, np.int64), np.array(gpos, np.int64),
            np.array(opos, np.int64), np.array(scale, np.float32))

@lru_cache(maxsize=None)
def _compose_plan_j(n_inter, n_new, dmax, fdegs):
    return _compose_plan(n_inter, n_new, dmax, fdegs)

@lru_cache(maxsize=None)
def _ddmul_plan_j(n, dmax, fdegs):
    return _ddmul_plan(n, dmax, fdegs)

#JIT numeric kernels
@partial(jax.jit, static_argnums=(5,))
def _compose_exec(F_flat, G1d, fcol, gidx, opos, fl_new):
    gprod = jnp.prod(G1d[gidx], axis=1)
    contrib = F_flat[:, fcol] * gprod[None, :]
    H = jnp.zeros((F_flat.shape[0], fl_new), F_flat.dtype)
    return H.at[:, opos].add(contrib)

@partial(jax.jit, static_argnums=(6,))
def _ddmul_exec(F_flat, G1d, fpos, gpos, opos, scale, fl):
    scale = scale.astype(F_flat.dtype)
    contrib = F_flat[:, fpos] * (scale * G1d[gpos])[None, :]
    H = jnp.zeros((F_flat.shape[0], fl), F_flat.dtype)
    return H.at[:, opos].add(contrib)

#Composition h(y) = f(g(y))
def compose(F, n_inter, G, n_new, dmax):
    #Substitute g into f truncated to degree dmax
    nout = F[0].shape[0]
    dt = F[0].dtype
    fdegs = _active_degrees(F, dmax)
    if not fdegs:
        return [jnp.zeros((nout, crd(n_new, k)), dt) for k in range(dmax + 1)]
    fcol, gidx, opos = _compose_plan_j(n_inter, n_new, dmax, fdegs)
    F_flat = _pack_full(F, n_inter, dmax)
    G_flat = _pack_full(G, n_new, dmax)
    fl_new = G_flat.shape[1]
    G1d = jnp.concatenate([G_flat.reshape(-1), jnp.ones((1,), G_flat.dtype)])
    H_flat = _compose_exec(F_flat, G1d, fcol, gidx, opos, fl_new)
    return _unpack_full(H_flat, n_new, dmax)

#Directional derivative h = (dF/dx) g
def ddmul(F, G, n, dmax):
    #h = sum_j (dF/dx_j) * G_j with F and G over the same n variables
    nout = F[0].shape[0]
    dt = F[0].dtype
    fdegs = _active_degrees(F, dmax)
    if not fdegs:
        return [jnp.zeros((nout, crd(n, k)), dt) for k in range(dmax + 1)]
    fpos, gpos, opos, scale = _ddmul_plan_j(n, dmax, fdegs)
    F_flat = _pack_full(F, n, dmax)
    G_flat = _pack_full(G, n, dmax)
    fl = G_flat.shape[1]
    G1d = G_flat.reshape(-1)
    H_flat = _ddmul_exec(F_flat, G1d, fpos, gpos, opos, scale, fl)
    return _unpack_full(H_flat, n, dmax)

def lie_operator(A, n, k):
    #Matrix of p -> (dp/dx)(A x) on the degree k monomial basis
    A = jnp.asarray(A)
    dt = A.dtype
    Mk = crd(n, k)
    #F is degree k identity, G is linear A
    F = [jnp.zeros((Mk, crd(n, kk)), dt) for kk in range(k + 1)]
    F[k] = jnp.eye(Mk, dtype=dt)
    G = [jnp.zeros((n, crd(n, kk)), dt) for kk in range(k + 1)]
    G[1] = A
    fpos, gpos, opos, scale = _ddmul_plan_j(n, k, (k,))
    F_flat = _pack_full(F, n, k)
    G1d = _pack_full(G, n, k).reshape(-1)
    fl = _flatlen(n, k)
    H_flat = _ddmul_exec(F_flat, G1d, fpos, gpos, opos, scale, fl)
    return _unpack_full(H_flat, n, k)[k]
