"""Segment geometry definitions and physical constants for Peecax."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
MU0: float = 4e-7 * np.pi   # Permeability of free space  [H/m]
EPS0: float = 8.854187817e-12  # Permittivity of free space [F/m]

# Resistivity of copper at 20 °C  [Ω·m]
RHO_COPPER: float = 1.72e-8

# Minimum length / direction-norm below which a segment is considered degenerate
_MIN_NORM: float = 1e-15

# ── Segment dataclass ─────────────────────────────────────────────────────────


@dataclass
class Segment:
    """One straight PEEC segment.

    Parameters
    ----------
    midpoint:
        Centre of the segment in 3-D space, in metres  ``(x, y, z)``.
    length:
        Segment length *l_i* [m].
    width:
        Cross-section width *w_i* [m] (in-plane, perpendicular to current).
    thickness:
        Cross-section thickness *t_i* [m] (out-of-plane / metal layer height).
    direction:
        Unit vector along the current direction  ``(dx, dy, dz)``.
    resistivity:
        DC resistivity ρ_i [Ω·m].  Defaults to copper at 20 °C.
    permeability:
        Magnetic permeability μ_i [H/m].  Defaults to μ₀.
    """

    midpoint: np.ndarray
    length: float
    width: float
    thickness: float
    direction: np.ndarray
    resistivity: float = field(default=RHO_COPPER)
    permeability: float = field(default=MU0)

    # ── post-init normalisation ───────────────────────────────────────────────

    def __post_init__(self) -> None:
        self.midpoint = np.asarray(self.midpoint, dtype=float)
        self.direction = np.asarray(self.direction, dtype=float)
        norm = np.linalg.norm(self.direction)
        if norm < _MIN_NORM:
            raise ValueError("Segment direction vector must be non-zero.")
        self.direction = self.direction / norm

    # ── derived quantities ────────────────────────────────────────────────────

    @property
    def equiv_radius(self) -> float:
        """Equivalent radius *a_i = 0.2235 (w_i + t_i)* [m]."""
        return 0.2235 * (self.width + self.thickness)

    # ── convenience constructors ──────────────────────────────────────────────

    @classmethod
    def from_endpoints(
        cls,
        p0: Sequence[float],
        p1: Sequence[float],
        width: float,
        thickness: float,
        resistivity: float = RHO_COPPER,
        permeability: float = MU0,
    ) -> "Segment":
        """Construct a segment from its two 3-D end-points.

        Parameters
        ----------
        p0, p1:
            End-points ``[x, y, z]`` in metres.
        width, thickness:
            Cross-section dimensions in metres.
        """
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length < _MIN_NORM:
            raise ValueError("Segment end-points must be distinct.")
        direction = vec / length
        midpoint = 0.5 * (p0 + p1)
        return cls(
            midpoint=midpoint,
            length=length,
            width=width,
            thickness=thickness,
            direction=direction,
            resistivity=resistivity,
            permeability=permeability,
        )
