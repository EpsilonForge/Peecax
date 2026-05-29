"""Frequency-domain PEEC solver.

Assembles the full admittance matrix Y(ω) and computes the
equivalent impedance Z(ω) = Y(ω)⁻¹ for a list of segments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import jax.numpy as jnp

from .geometry import Segment
from .matrices import build_R, build_L, build_CG


@dataclass
class SolverParams:
    """Material and substrate parameters for the PEEC solve.

    Parameters
    ----------
    eps_r:
        Real part of substrate relative permittivity ε'.  Default: 11.7 (Si).
    sigma_d:
        Substrate conductivity σ_d [S/m].  Default: 10 (lossy Si).
    """

    eps_r: float = 11.7
    sigma_d: float = 10.0


@dataclass
class FreqResult:
    """Result at a single angular frequency.

    Attributes
    ----------
    omega:
        Angular frequency [rad/s].
    Z:
        Equivalent impedance matrix (N × N) [Ω].
    Y:
        Admittance matrix (N × N) [S].
    R:
        Resistance matrix (N × N) [Ω].
    L:
        Inductance matrix (N × N) [H].
    C:
        Real capacitance matrix (N × N) [F].
    Gd:
        Dielectric conductance matrix (N × N) [S].
    """

    omega: float
    Z: jnp.ndarray
    Y: jnp.ndarray
    R: jnp.ndarray
    L: jnp.ndarray
    C: jnp.ndarray
    Gd: jnp.ndarray


class FreqSolver:
    """PEEC-style frequency-domain solver.

    Parameters
    ----------
    segments:
        List of :class:`~peecax.geometry.Segment` objects describing the
        conductor discretisation.
    params:
        :class:`SolverParams` holding substrate / material parameters.

    Examples
    --------
    >>> from peecax import Segment, FreqSolver, SolverParams
    >>> import numpy as np
    >>> seg = Segment.from_endpoints([0,0,0],[100e-6,0,0],
    ...                               width=10e-6, thickness=1e-6)
    >>> solver = FreqSolver([seg], SolverParams())
    >>> result = solver.solve(omega=2*np.pi*1e9)
    """

    def __init__(
        self,
        segments: Sequence[Segment],
        params: SolverParams | None = None,
    ) -> None:
        if len(segments) == 0:
            raise ValueError("At least one segment is required.")
        self.segments = list(segments)
        self.params = params if params is not None else SolverParams()
        # Pre-compute frequency-independent inductance matrix
        self._L = build_L(self.segments)

    # ── single-frequency solve ────────────────────────────────────────────────

    def solve(self, omega: float) -> FreqResult:
        """Solve at one angular frequency.

        Parameters
        ----------
        omega:
            Angular frequency ω [rad/s].

        Returns
        -------
        :class:`FreqResult`
        """
        if omega <= 0:
            raise ValueError(f"omega must be positive, got {omega}.")

        p = self.params

        R = build_R(self.segments, omega)
        L = self._L
        C, Gd = build_CG(self.segments, omega, p.eps_r, p.sigma_d)

        # Branch impedance / admittance
        Z_RL = R + 1j * omega * L
        Y_RL = jnp.linalg.inv(Z_RL)

        # Full admittance
        Y = Y_RL + 1j * omega * C + Gd

        # Equivalent impedance
        Z = jnp.linalg.inv(Y)

        return FreqResult(omega=omega, Z=Z, Y=Y, R=R, L=L, C=C, Gd=Gd)

    # ── frequency sweep ───────────────────────────────────────────────────────

    def sweep(self, freqs_hz: Sequence[float]) -> list[FreqResult]:
        """Solve over a list of frequencies.

        Parameters
        ----------
        freqs_hz:
            Frequencies in Hz.

        Returns
        -------
        list of :class:`FreqResult`, one per frequency.
        """
        return [self.solve(2.0 * np.pi * f) for f in freqs_hz]

    # ── current-excitation solve ──────────────────────────────────────────────

    def solve_current(
        self, omega: float, i_exc: Sequence[complex]
    ) -> tuple[jnp.ndarray, FreqResult]:
        """Solve for node voltages given current excitation.

        .. math::

            \\bm{v}(\\omega) = \\bm{Y}(\\omega)^{-1} \\bm{i}(\\omega)
                              = \\bm{Z}(\\omega)\\,\\bm{i}(\\omega).

        Parameters
        ----------
        omega:
            Angular frequency [rad/s].
        i_exc:
            Current excitation vector of length N [A].

        Returns
        -------
        v : (N,) complex JAX array of node voltages [V].
        result : :class:`FreqResult`
        """
        result = self.solve(omega)
        i_vec = jnp.array(i_exc, dtype=complex)
        v = result.Z @ i_vec
        return v, result
