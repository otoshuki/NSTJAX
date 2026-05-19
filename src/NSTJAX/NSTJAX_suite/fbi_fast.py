"""
Author: gpertin, KAIST
Fast FBI solve via exosystem spectral decoupling, with solvability guard
"""

from functools import partial
import numpy as np
import scipy.linalg as sla
import jax
import jax.numpy as jnp
from NSTJAX.NSTJAX_suite.polylib import unpack, zero_field, compose, ddmul, lie_operator
from NSTJAX.NSTJAX_suite.decouple import schur_form, operator_eigs

#Relative tolerance for a vanished lie power
NIL_TOL = 1e-5
#Threshold below which a mode counts as resonant with a transmission zero
RES_TOL = 1e-3
#Threshold on beta for a finite pencil eigenvalue
FIN_TOL = 1e-6

def _nil_index(LF):
    #Smallest power at which the nilpotent lie operator vanishes
    nrm = np.max(np.abs(LF)) + 1e-30
    P = np.eye(LF.shape[0], dtype=LF.dtype)
    for j in range(1, LF.shape[0] + 2):
        P = P @ LF
        if np.max(np.abs(P)) < NIL_TOL * nrm:
            return j
    return LF.shape[0]

@partial(jax.jit, static_argnums=(4,))
def _solve_nilpotent(M_, Sel, RHS, LF, N):
    #Finite neumann sum, exact for a nilpotent exosystem
    Minv = jnp.linalg.pinv(M_)
    MinvSel = Minv @ Sel
    Term = Minv @ RHS
    W = Term
    for _ in range(1, N):
        Term = MinvSel @ Term @ LF
        W = W + Term
    return W

@jax.jit
def _solve_general(M_, Sel, RHS, Z, R):
    #Back substitution in the complex schur basis of the lie operator
    Mc = M_.astype(Z.dtype)
    Sc = Sel.astype(Z.dtype)
    RHS_t = RHS.astype(Z.dtype) @ Z
    n_k = R.shape[0]
    def body(j, W_t):
        col = W_t @ R[:, j]
        rhs_j = RHS_t[:, j] + Sc @ col
        Op = Mc - R[j, j] * Sc
        return W_t.at[:, j].set(jnp.linalg.solve(Op, rhs_j))
    W_t = jax.lax.fori_loop(0, n_k, body,
                            jnp.zeros((M_.shape[1], n_k), Z.dtype))
    return jnp.real(W_t @ jnp.conj(Z).T).astype(M_.dtype)

def _transmission_zeros(M_, Sel):
    #Finite generalized eigenvalues of the pencil (M_, Sel)
    M_ = np.asarray(M_, dtype=np.float64)
    Sel = np.asarray(Sel, dtype=np.float64)
    if M_.shape[0] != M_.shape[1]:
        return np.zeros(0, dtype=np.complex64)
    alpha, beta = sla.eig(M_, Sel, homogeneous_eigvals=True)[0]
    scale = float(np.max(np.abs(Sel))) + 1e-30
    fin = np.abs(beta) > FIN_TOL * scale
    return (alpha[fin] / beta[fin]).astype(np.complex64)

def _screen(M_, Sel, mu, n_, d, p, m):
    #Per degree resonance report against the transmission zeros
    rep = {"regime": ("overdetermined" if p > m else
                      "underdetermined" if m > p else "square")}
    zeros = _transmission_zeros(M_, Sel)
    zscale = float(np.max(np.abs(zeros))) + 1.0 if zeros.size else 1.0
    degs = {}
    for k in range(1, d + 1):
        lam = operator_eigs(mu, n_, k)
        if zeros.size:
            gap = float(np.min(np.abs(lam[:, None] - zeros[None, :]))) / zscale
        else:
            gap = np.inf
        degs[k] = {"min_gap": gap, "resonant": bool(gap < RES_TOL)}
    rep["degrees"] = degs
    rep["resonant"] = any(v["resonant"] for v in degs.values())
    rep["solvable"] = (rep["regime"] != "overdetermined") and not rep["resonant"]
    return rep

class FBIFast:
    #Decoupled FBI solver, fixed reuses the factorization, check guards solvability
    def __init__(self, n, m, p, n_, d, fixed=True, check="off", branch="auto"):
        self.n, self.m, self.p, self.n_, self.d = n, m, p, n_, d
        self.nsum = n + m + n_
        self.fixed = fixed
        self.check = check
        self.branch = branch
        self._dec = None
        self._mu = None
        self._rep = None

    def _decouple(self, fexo):
        #Per degree spectral data from the exosystem linear part
        if self.fixed and self._dec is not None:
            return self._dec, self._mu
        n_, d = self.n_, self.d
        A_ = np.asarray(fexo[:, :n_], dtype=np.float32)
        _, _, mu, nil = schur_form(A_)
        use_nil = nil if self.branch == "auto" else (self.branch == "nilpotent")
        Aj = jnp.asarray(A_)
        dec = []
        for k in range(1, d + 1):
            LF = np.asarray(lie_operator(Aj, n_, k), dtype=np.float32)
            if use_nil:
                dec.append((True, jnp.asarray(LF), _nil_index(LF), None))
            else:
                R, Z = sla.schur(LF.astype(np.complex64), output="complex")
                dec.append((False, jnp.asarray(Z), None, jnp.asarray(R)))
        if self.fixed:
            self._dec, self._mu = dec, mu
        return dec, mu

    def _guard(self, M_, Sel, mu, dec):
        #Branch report always, spectral screen only when asked
        branch = {k + 1: ("nilpotent" if dec[k][0] else "general")
                  for k in range(self.d)}
        if self.check == "off":
            return {"checked": False, "branch": branch}
        if self.check == "setup" and self._rep is not None:
            rep = dict(self._rep)
            rep["branch"] = branch
            return rep
        rep = _screen(np.asarray(M_), np.asarray(Sel), mu,
                      self.n_, self.d, self.p, self.m)
        rep["checked"] = True
        rep["branch"] = branch
        if self.check == "setup":
            self._rep = rep
        return rep

    def _solve_deg(self, M_, Sel, RHS, dec_k):
        nil, A_or_Z, N, R = dec_k
        if nil:
            return _solve_nilpotent(M_, Sel, RHS, A_or_Z, N)
        return _solve_general(M_, Sel, RHS, A_or_Z, R)

    def solve(self, f, h, fexo):
        n, m, p, n_, d, nsum = self.n, self.m, self.p, self.n_, self.d, self.nsum
        dt = f.dtype
        dec, mu = self._decouple(fexo)
        F1 = f[:, :nsum]
        H1 = h[:, :nsum]
        M_ = jnp.block([[F1[:, :n], F1[:, n:n + m]],
                        [H1[:, :n], H1[:, n:n + m]]])
        Sel = jnp.block([[jnp.eye(n, dtype=dt), jnp.zeros((n, m), dt)],
                         [jnp.zeros((p, n + m), dt)]])
        report = self._guard(M_, Sel, mu, dec)
        Fxb = F1[:, n + m:nsum]
        Hxb = H1[:, n + m:nsum]
        #Degree 1
        RHS = -jnp.concatenate([Fxb, Hxb], axis=0)
        W = self._solve_deg(M_, Sel, RHS, dec[0])
        th = W[:n, :]
        la = W[n:, :]
        #Degrees 2..d, same operator with the lower degree right hand side
        fh = jnp.concatenate([f, h], axis=0)
        fh_field = unpack(fh, nsum, 1, d)
        fexo_field = unpack(fexo, n_, 1, d)
        for k in range(2, d + 1):
            th_field = unpack(th, n_, 1, k - 1)
            la_field = unpack(la, n_, 1, k - 1)
            deriv = ddmul(th_field, fexo_field, n_, k)[k]
            data_hi = zero_field(n + p, nsum, k, dtype=dt)
            for kk in range(2, k + 1):
                data_hi[kk] = fh_field[kk]
            g_sub = zero_field(nsum, n_, k, dtype=dt)
            for deg in range(1, k):
                g_sub[deg] = g_sub[deg].at[:n].set(th_field[deg])
                g_sub[deg] = g_sub[deg].at[n:n + m].set(la_field[deg])
            g_sub[1] = g_sub[1].at[n + m:nsum].set(jnp.eye(n_, dtype=dt))
            comp = compose(data_hi, nsum, g_sub, n_, k)[k]
            rhs_F = deriv - comp[:n]
            rhs_H = -comp[n:]
            RHS = jnp.concatenate([rhs_F, rhs_H], axis=0)
            W = self._solve_deg(M_, Sel, RHS, dec[k - 1])
            th = jnp.concatenate([th, W[:n, :]], axis=1)
            la = jnp.concatenate([la, W[n:, :]], axis=1)
        return th, la, report
