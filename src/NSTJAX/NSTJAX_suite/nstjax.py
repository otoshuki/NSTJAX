"""
Author: gpertin, KAIST
High level NSTJAX driver, taylor maps, FBI solve and inference in one object
"""

import jax
import jax.numpy as jnp
from NSTJAX.NSTJAX_suite.polylib import crd
from NSTJAX.NSTJAX_suite.taylor import build_taylor
from NSTJAX.NSTJAX_suite.fbi import fbi
from NSTJAX.NSTJAX_suite.fbi_fast import FBIFast
from NSTJAX.NSTJAX_suite.reporter import build_report
from NSTJAX.NSTJAX_suite.fbi_eval import (compute_theta as _eval_theta,
                                          compute_lambda as _eval_lambda)

class NSTJAX:
    #User provides the system, then warm starts, then solves at any operating point
    def __init__(self, fsys, fexo, hsys, n, m, p, n_, d,
                 solver="auto", fixed=True, branch="auto", check="off",
                 auto_threshold=20e2, reshape=True, store=True, verbose=False):
        self.fsys, self.fexo, self.hsys = fsys, fexo, hsys
        self.n, self.m, self.p, self.n_, self.d = n, m, p, n_, d
        self.nsum = n + m + n_
        #Solver selection, auto switches on the kron dimension below
        self.solver = solver
        self.auto_threshold = auto_threshold
        #fbi_fast options, fixed assumes a constant exosystem linearization
        self.fixed = fixed
        self.branch = branch
        self.check = check
        #Output options
        self.reshape = reshape
        self.store = store
        self.verbose = verbose
        #Size of the original dense regulator operator
        self.kron_dim = (n + p) * crd(n_, d)
        self.solver_name = None
        self.report = None
        self._rep_cache = None
        self.th = None
        self.la = None
        self._tay_f = self._tay_h = self._tay_e = None
        self._fast = None

    def _pick_solver(self):
        #Auto routes small systems to the dense solve, large ones to decoupling
        if self.solver == "auto":
            use_fast = self.kron_dim >= self.auto_threshold
        else:
            use_fast = (self.solver == "fbi_fast")
        if use_fast:
            self._fast = FBIFast(self.n, self.m, self.p, self.n_, self.d,
                                 fixed=self.fixed, check="off", branch=self.branch)
            self.solver_name = "fbi_fast"
        else:
            self._fast = None
            self.solver_name = "fbi"

    def _solve(self, f, h, fe):
        if self._fast is not None:
            th, la, _ = self._fast.solve(f, h, fe)
            return th, la
        th, la = fbi(f, h, fe, self.n, self.m, self.p, self.n_, self.d)
        return th, la

    def warm_start(self, z0=None, x_0=None, warm_reps=2, samples=1):
        #Build the maps, pick the solver, compile fbi and inference at real shapes
        nsum, n_ = self.nsum, self.n_
        z0 = jnp.zeros(nsum) if z0 is None else jnp.asarray(z0)
        x_0 = jnp.zeros(n_) if x_0 is None else jnp.asarray(x_0)
        self._z0, self._x_0 = z0, x_0
        self._tay_f = build_taylor(self.fsys, nsum, self.d)
        self._tay_h = build_taylor(self.hsys, nsum, self.d)
        self._tay_e = build_taylor(self.fexo, n_, self.d)
        self._pick_solver()
        for _ in range(max(1, warm_reps)):
            th, la = self.compute_fbi(z0, x_0)
        #Warm inference at the batch size used in the loop
        W = jnp.zeros((samples, n_))
        pi = self.compute_theta(W, th)
        c = self.compute_lambda(W, la)
        jax.block_until_ready((th, la, pi, c))
        if self.verbose:
            print(f"NSTJAX warm: solver {self.solver_name}  kron-dim {self.kron_dim}")
        return self

    def compute_fbi(self, z, x_=None):
        #Solve the regulator at one operating point, reusing the chosen solver
        if x_ is None:
            x_ = z[self.n + self.m:]
        f = self._tay_f(z)
        h = self._tay_h(z)
        fe = self._tay_e(x_)
        th, la = self._solve(f, h, fe)
        if self.reshape:
            th = jnp.reshape(th, (self.n, -1))
            la = jnp.reshape(la, (self.m, -1))
        #Solver agnostic report, setup caches the first screen
        if self.check == "setup" and self._rep_cache is not None:
            self.report = self._rep_cache
        else:
            self.report = build_report(f, h, fe, self.n, self.m,
                                       self.p, self.n_, self.d, self.check)
            if self.check == "setup":
                self._rep_cache = self.report
        if self.store:
            self.th, self.la = th, la
        return th, la

    def compute_theta(self, W, th=None):
        th = self.th if th is None else th
        return _eval_theta(th, W, self.d)

    def compute_lambda(self, W, la=None):
        la = self.la if la is None else la
        return _eval_lambda(la, W, self.d)
