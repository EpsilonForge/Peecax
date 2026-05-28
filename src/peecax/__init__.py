"""Peecax – differentiable PEEC-style EM solver for RFIC/MMIC.

64-bit floating-point is enabled globally for JAX on import so that
EM matrix entries have sufficient precision.

Public API
----------
Geometry
    :class:`~peecax.geometry.Segment`
    :data:`~peecax.geometry.MU0`
    :data:`~peecax.geometry.EPS0`
    :data:`~peecax.geometry.RHO_COPPER`

Matrix builders
    :func:`~peecax.matrices.build_R`
    :func:`~peecax.matrices.build_L`
    :func:`~peecax.matrices.build_P`
    :func:`~peecax.matrices.build_CG`

Solver
    :class:`~peecax.solver.FreqSolver`
    :class:`~peecax.solver.SolverParams`
    :class:`~peecax.solver.FreqResult`

GDS integration
    :func:`~peecax.gds.segments_from_path`
    :func:`~peecax.gds.segments_from_component`
    :func:`~peecax.gds.spiral_inductor_segments`
"""

import jax
# Enable 64-bit precision globally; must be called before any JAX computation.
jax.config.update("jax_enable_x64", True)

from .geometry import EPS0, MU0, RHO_COPPER, Segment
from .matrices import build_CG, build_L, build_P, build_R
from .solver import FreqResult, FreqSolver, SolverParams
from .gds import (
    segments_from_path,
    segments_from_component,
    spiral_inductor_segments,
)

__all__ = [
    # geometry
    "Segment",
    "MU0",
    "EPS0",
    "RHO_COPPER",
    # matrices
    "build_R",
    "build_L",
    "build_P",
    "build_CG",
    # solver
    "FreqSolver",
    "SolverParams",
    "FreqResult",
    # gds
    "segments_from_path",
    "segments_from_component",
    "spiral_inductor_segments",
]
