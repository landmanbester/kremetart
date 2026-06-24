"""Optimisation solvers for kremetart (xp-injectable; CPU via numpy, GPU via cupy)."""

from kremetart.opt.cg import cg
from kremetart.opt.fista import fista

__all__ = ["cg", "fista"]
