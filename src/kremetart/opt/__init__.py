"""Optimisation solvers for kremetart (xp-injectable; CPU via numpy, GPU via cupy)."""

from kremetart.opt.cg import cg
from kremetart.opt.fista import fista, fista_quadratic

__all__ = ["cg", "fista", "fista_quadratic"]
