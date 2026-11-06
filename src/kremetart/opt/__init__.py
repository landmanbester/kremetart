"""Generic solvers, ``xp``-injectable: preconditioned conjugate gradients and (reweighted-L1) FISTA."""

from kremetart.opt.cg import cg
from kremetart.opt.fista import fista, fista_quadratic

__all__ = ["cg", "fista", "fista_quadratic"]
