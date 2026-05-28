"""RLGC matrix assembly for the crude CES PEEC model.

All heavy lifting uses JAX so that every matrix is differentiable
with respect to geometry parameters.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import jax.numpy as jnp

from .geometry import Segment, MU0, EPS0


# ── helpers ───────────────────────────────────────────────────────────────────


def _build_geometry_arrays(
    segments: Sequence[Segment],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """Extract flat geometry arrays from a list of segments.

    Returns
    -------
    midpoints  : (N, 3)
    lengths    : (N,)
    widths     : (N,)
    thicknesses: (N,)
    directions : (N, 3)
    equiv_radii: (N,)
    resistivities: (N,)
    permeabilities: (N,)
    """
    N = len(segments)
    midpoints = np.array([s.midpoint for s in segments])      # (N, 3)
    lengths = np.array([s.length for s in segments])          # (N,)
    widths = np.array([s.width for s in segments])            # (N,)
    thicknesses = np.array([s.thickness for s in segments])   # (N,)
    directions = np.array([s.direction for s in segments])    # (N, 3)
    equiv_radii = np.array([s.equiv_radius for s in segments])  # (N,)
    resistivities = np.array([s.resistivity for s in segments])  # (N,)
    permeabilities = np.array([s.permeability for s in segments])  # (N,)
    return (midpoints, lengths, widths, thicknesses, directions,
            equiv_radii, resistivities, permeabilities)


# ── Resistance matrix R(ω) ────────────────────────────────────────────────────


def build_R(segments: Sequence[Segment], omega: float) -> jnp.ndarray:
    """Resistance matrix **R(ω)** – diagonal, skin-effect corrected.

    .. math::

        R_{ii}(\\omega) = \\frac{\\rho_i l_i}{w_i t_i}
            \\Re\\!\\left[\\kappa_i \\coth(\\kappa_i)\\right],
        \\quad \\kappa_i = (1+j)\\frac{t_i}{\\delta_i(\\omega)},
        \\quad \\delta_i = \\sqrt{\\frac{2\\rho_i}{\\omega\\mu_i}}.

    Parameters
    ----------
    segments:
        List of :class:`~peecax.geometry.Segment` objects.
    omega:
        Angular frequency [rad/s].  Must be > 0.

    Returns
    -------
    R : (N, N) complex JAX array  (imaginary part is zero by construction).
    """
    if omega <= 0:
        raise ValueError("omega must be positive.")
    _, lengths, widths, thicknesses, _, _, resistivities, permeabilities = (
        _build_geometry_arrays(segments)
    )
    N = len(segments)

    # Skin depth  δ_i = sqrt(2 ρ_i / (ω μ_i))
    delta = np.sqrt(2.0 * resistivities / (omega * permeabilities))  # (N,)

    # κ_i = (1+j) t_i / δ_i
    kappa = (1.0 + 1j) * thicknesses / delta  # (N,) complex

    # kappa * coth(kappa) – safe for small |kappa| (DC limit → 1)
    with np.errstate(over="ignore", invalid="ignore"):
        factor = np.where(
            np.abs(kappa) < 1e-8,
            1.0 + 0j,
            kappa / np.tanh(kappa),
        )
    Rii = resistivities * lengths / (widths * thicknesses) * np.real(factor)

    R = jnp.diag(jnp.array(Rii.astype(complex)))
    return R


# ── Inductance matrix L ───────────────────────────────────────────────────────


def build_L(segments: Sequence[Segment]) -> jnp.ndarray:
    """Partial inductance matrix **L** (frequency-independent).

    Self terms:

    .. math::

        L_{ii} = \\frac{\\mu_0}{2\\pi} l_i \\left[
            \\ln\\!\\left(\\frac{2 l_i}{a_i}\\right) - 1 \\right].

    Mutual terms:

    .. math::

        L_{ij} = \\frac{\\mu_0}{4\\pi}
            (\\hat{\\bm{l}}_i \\cdot \\hat{\\bm{l}}_j)
            \\frac{l_i l_j}{\\sqrt{d_{ij}^2 + \\eta^2}},
        \\quad \\eta = \\tfrac{1}{2}(a_i + a_j).

    Returns
    -------
    L : (N, N) real JAX array, symmetrised.
    """
    (midpoints, lengths, _, _, directions,
     equiv_radii, _, _) = _build_geometry_arrays(segments)

    N = len(segments)

    # Self terms
    self_terms = (MU0 / (2.0 * np.pi)) * lengths * (
        np.log(2.0 * lengths / equiv_radii) - 1.0
    )

    # Mutual terms – vectorised over (i, j)
    # d_ij : (N, N)
    diff = midpoints[:, None, :] - midpoints[None, :, :]  # (N, N, 3)
    dij2 = np.sum(diff ** 2, axis=-1)                     # (N, N)

    ai_plus_aj = equiv_radii[:, None] + equiv_radii[None, :]  # (N, N)
    eta2 = (0.5 * ai_plus_aj) ** 2                            # (N, N)

    dot_dirs = directions @ directions.T  # (N, N)

    li_lj = lengths[:, None] * lengths[None, :]  # (N, N)

    mutual = (MU0 / (4.0 * np.pi)) * dot_dirs * li_lj / np.sqrt(dij2 + eta2)

    # Combine: self on diagonal, mutual off-diagonal
    L = np.where(np.eye(N, dtype=bool), self_terms[:, None], mutual)
    # Zero the diagonal of the mutual matrix then add self
    np.fill_diagonal(L, self_terms)

    # Symmetrize
    L = 0.5 * (L + L.T)

    return jnp.array(L)


# ── Potential matrix P(ω) and capacitance / conductance ──────────────────────


def build_P(
    segments: Sequence[Segment],
    omega: float,
    eps_r: float,
    sigma_d: float,
) -> jnp.ndarray:
    """Potential coefficient matrix **P(ω)** with complex effective permittivity.

    .. math::

        \\epsilon^*(\\omega) = \\epsilon_r - j \\frac{\\sigma_d}{\\omega},
        \\quad
        \\epsilon_{\\mathrm{eff}}^* = \\frac{1 + \\epsilon^*(\\omega)}{2}.

    Self terms:

    .. math::

        P_{ii} = \\frac{\\ln(2 l_i / a_i)}{2 \\pi \\epsilon_0
            \\epsilon_{\\mathrm{eff}}^* l_i}.

    Mutual terms:

    .. math::

        P_{ij} = \\frac{1}{4 \\pi \\epsilon_0 \\epsilon_{\\mathrm{eff}}^*
            \\sqrt{d_{ij}^2 + \\eta^2}}.

    Parameters
    ----------
    segments:
        List of :class:`~peecax.geometry.Segment` objects.
    omega:
        Angular frequency [rad/s].
    eps_r:
        Real part of substrate relative permittivity ε'.
    sigma_d:
        Substrate conductivity σ_d [S/m].

    Returns
    -------
    P : (N, N) complex JAX array.
    """
    if omega <= 0:
        raise ValueError("omega must be positive.")

    (midpoints, lengths, _, _, _,
     equiv_radii, _, _) = _build_geometry_arrays(segments)

    N = len(segments)

    # Complex effective permittivity
    eps_star = complex(eps_r) - 1j * sigma_d / omega
    eps_eff = (1.0 + eps_star) / 2.0

    # Self terms
    self_terms = (
        np.log(2.0 * lengths / equiv_radii)
        / (2.0 * np.pi * EPS0 * eps_eff * lengths)
    )  # (N,) complex

    # Mutual terms
    diff = midpoints[:, None, :] - midpoints[None, :, :]  # (N, N, 3)
    dij2 = np.sum(diff ** 2, axis=-1)                     # (N, N)

    ai_plus_aj = equiv_radii[:, None] + equiv_radii[None, :]  # (N, N)
    eta2 = (0.5 * ai_plus_aj) ** 2                            # (N, N)

    mutual = 1.0 / (
        4.0 * np.pi * EPS0 * eps_eff * np.sqrt(dij2 + eta2)
    )  # (N, N) complex

    P = np.where(np.eye(N, dtype=bool), self_terms[:, None], mutual)
    np.fill_diagonal(P, self_terms)

    return jnp.array(P)


def build_CG(
    segments: Sequence[Segment],
    omega: float,
    eps_r: float,
    sigma_d: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Capacitance **C(ω)** and dielectric conductance **G_d(ω)** matrices.

    Computed via complex matrix inversion of P:

    .. math::

        \\bm{C}^*(\\omega) = \\bm{P}(\\omega)^{-1},
        \\quad
        \\bm{C} = \\Re(\\bm{C}^*),
        \\quad
        \\bm{G}_d = -\\omega\\,\\Im(\\bm{C}^*).

    Returns
    -------
    C  : (N, N) real JAX array  [F].
    Gd : (N, N) real JAX array  [S].
    """
    P = build_P(segments, omega, eps_r, sigma_d)
    C_star = jnp.linalg.inv(P)
    C = jnp.real(C_star)
    Gd = -omega * jnp.imag(C_star)
    return C, Gd
