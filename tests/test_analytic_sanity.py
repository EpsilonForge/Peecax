"""Simple analytic sanity checks for Peecax.

These tests intentionally use geometries where closed-form formulas are
expected to be accurate, so they provide quick physical validation.
"""

from __future__ import annotations

import math

import numpy as np

from peecax import EPS0, MU0, Segment, build_CG, build_L, build_R


OMEGA_1MHZ = 2.0 * math.pi * 1e6


def _parallel_plate_segments(
    length_m: float,
    width_m: float,
    gap_m: float,
    strips_per_plate: int,
    thickness_m: float,
) -> tuple[list[Segment], int]:
    """Create two discretized plates using parallel metal strips."""
    segs: list[Segment] = []
    dy = width_m / strips_per_plate

    for z in (-0.5 * gap_m, 0.5 * gap_m):
        for i in range(strips_per_plate):
            y = -0.5 * width_m + (i + 0.5) * dy
            p0 = [-0.5 * length_m, y, z]
            p1 = [0.5 * length_m, y, z]
            segs.append(
                Segment.from_endpoints(
                    p0,
                    p1,
                    width=dy,
                    thickness=thickness_m,
                )
            )
    return segs, strips_per_plate


def _extract_two_conductor_capacitance(C: np.ndarray, n_plate: int) -> float:
    """Return equivalent C from +/-0.5 V differential plate excitation."""
    V = np.zeros(2 * n_plate)
    V[:n_plate] = 0.5
    V[n_plate:] = -0.5

    q = C @ V
    q_plate_a = float(np.sum(q[:n_plate]))
    return abs(q_plate_a)


def test_straight_wire_dc_resistance_matches_analytic():
    """R = rho * l / (w * t) should hold near DC."""
    seg = Segment.from_endpoints(
        [0.0, 0.0, 0.0],
        [200e-6, 0.0, 0.0],
        width=10e-6,
        thickness=2e-6,
    )

    omega_dc = 2.0 * math.pi * 1.0
    R = np.array(build_R([seg], omega_dc))

    expected = seg.resistivity * seg.length / (seg.width * seg.thickness)
    got = float(R[0, 0].real)
    rel_err = abs(got - expected) / expected

    assert rel_err < 0.01


def test_parallel_wires_mutual_inductance_matches_analytic():
    """Two long, co-directional wires should follow the implemented L12 formula."""
    length_m = 400e-6
    sep_m = 120e-6

    s1 = Segment.from_endpoints(
        [0.0, 0.0, 0.0],
        [length_m, 0.0, 0.0],
        width=8e-6,
        thickness=2e-6,
    )
    s2 = Segment.from_endpoints(
        [0.0, sep_m, 0.0],
        [length_m, sep_m, 0.0],
        width=8e-6,
        thickness=2e-6,
    )

    L = np.array(build_L([s1, s2]))
    got = float(L[0, 1])

    eta = 0.5 * (s1.equiv_radius + s2.equiv_radius)
    expected = (MU0 / (4.0 * math.pi)) * (s1.length * s2.length) / math.sqrt(
        sep_m ** 2 + eta ** 2
    )

    rel_err = abs(got - expected) / expected
    assert rel_err < 1e-12


def test_parallel_plate_capacitance_close_to_area_over_gap():
    """Discretized parallel-plate capacitor should be near C = eps0 * A / d."""
    length_m = 2.0e-3
    width_m = 2.0e-3
    gap_m = 0.3e-3
    strips = 8

    segs, n_plate = _parallel_plate_segments(
        length_m=length_m,
        width_m=width_m,
        gap_m=gap_m,
        strips_per_plate=strips,
        thickness_m=5e-6,
    )

    C, _ = build_CG(segs, OMEGA_1MHZ, eps_r=1.0, sigma_d=0.0)
    C_eq = _extract_two_conductor_capacitance(np.array(C), n_plate)

    C_analytic = EPS0 * (length_m * width_m) / gap_m
    rel_err = abs(C_eq - C_analytic) / C_analytic

    # Includes fringe fields and strip discretization error.
    assert rel_err < 0.25
