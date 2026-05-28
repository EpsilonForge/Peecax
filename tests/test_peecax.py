"""Unit tests for Peecax PEEC solver.

Tests cover:
- Segment geometry and derived quantities
- Individual matrix builders (R, L, P, C/G)
- FreqSolver integration (single-frequency and sweep)
- GDS spiral helper
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import peecax
from peecax import (
    EPS0,
    MU0,
    RHO_COPPER,
    Segment,
    FreqSolver,
    SolverParams,
    build_CG,
    build_L,
    build_P,
    build_R,
    spiral_inductor_segments,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def single_segment() -> Segment:
    """A 100 µm copper wire, 10 µm wide, 1 µm thick."""
    return Segment.from_endpoints(
        [0, 0, 0], [100e-6, 0, 0],
        width=10e-6,
        thickness=1e-6,
    )


@pytest.fixture
def two_segments() -> list[Segment]:
    """Two parallel copper wires 100 µm long, separated by 20 µm."""
    s1 = Segment.from_endpoints(
        [0, 0, 0], [100e-6, 0, 0],
        width=5e-6, thickness=1e-6,
    )
    s2 = Segment.from_endpoints(
        [0, 20e-6, 0], [100e-6, 20e-6, 0],
        width=5e-6, thickness=1e-6,
    )
    return [s1, s2]


OMEGA_1GHZ = 2 * math.pi * 1e9


# ── Segment geometry ──────────────────────────────────────────────────────────

class TestSegment:
    def test_from_endpoints_length(self, single_segment):
        assert abs(single_segment.length - 100e-6) < 1e-15

    def test_direction_unit_vector(self, single_segment):
        norm = np.linalg.norm(single_segment.direction)
        assert abs(norm - 1.0) < 1e-12

    def test_midpoint(self, single_segment):
        expected = np.array([50e-6, 0.0, 0.0])
        np.testing.assert_allclose(single_segment.midpoint, expected)

    def test_equiv_radius(self, single_segment):
        expected = 0.2235 * (10e-6 + 1e-6)
        assert abs(single_segment.equiv_radius - expected) < 1e-20

    def test_zero_length_raises(self):
        with pytest.raises(ValueError, match="distinct"):
            Segment.from_endpoints([0, 0, 0], [0, 0, 0], width=1e-6, thickness=1e-6)

    def test_zero_direction_raises(self):
        with pytest.raises(ValueError):
            Segment(
                midpoint=[0, 0, 0],
                length=1e-4,
                width=1e-5,
                thickness=1e-6,
                direction=[0, 0, 0],
            )


# ── Resistance matrix ─────────────────────────────────────────────────────────

class TestBuildR:
    def test_shape(self, single_segment):
        R = build_R([single_segment], OMEGA_1GHZ)
        assert R.shape == (1, 1)

    def test_real_positive(self, single_segment):
        R = build_R([single_segment], OMEGA_1GHZ)
        assert float(R[0, 0].real) > 0

    def test_off_diagonal_zero(self, two_segments):
        R = build_R(two_segments, OMEGA_1GHZ)
        assert float(abs(R[0, 1])) == 0.0
        assert float(abs(R[1, 0])) == 0.0

    def test_dc_limit(self, single_segment):
        """At very low ω the skin-depth correction → 1, so R = ρl/(wt)."""
        omega_dc = 2 * math.pi * 1.0  # 1 Hz – near-DC
        R_dc = build_R([single_segment], omega_dc)
        seg = single_segment
        expected = seg.resistivity * seg.length / (seg.width * seg.thickness)
        assert abs(float(R_dc[0, 0].real) - expected) / expected < 0.01

    def test_negative_omega_raises(self, single_segment):
        with pytest.raises(ValueError):
            build_R([single_segment], -1.0)


# ── Inductance matrix ─────────────────────────────────────────────────────────

class TestBuildL:
    def test_shape(self, single_segment):
        L = build_L([single_segment])
        assert L.shape == (1, 1)

    def test_self_inductance_positive(self, single_segment):
        L = build_L([single_segment])
        assert float(L[0, 0]) > 0

    def test_symmetric(self, two_segments):
        L = build_L(two_segments)
        np.testing.assert_allclose(np.array(L), np.array(L).T, atol=1e-30)

    def test_mutual_same_direction_positive(self, two_segments):
        """Parallel co-directional wires → positive mutual inductance."""
        L = build_L(two_segments)
        assert float(L[0, 1]) > 0

    def test_mutual_opposite_direction_negative(self):
        """Anti-parallel wires → negative mutual inductance."""
        s1 = Segment.from_endpoints([0, 0, 0], [100e-6, 0, 0],
                                     width=5e-6, thickness=1e-6)
        s2 = Segment.from_endpoints([100e-6, 20e-6, 0], [0, 20e-6, 0],
                                     width=5e-6, thickness=1e-6)
        L = build_L([s1, s2])
        assert float(L[0, 1]) < 0

    def test_self_inductance_formula(self, single_segment):
        """Check self-inductance against the closed-form formula."""
        seg = single_segment
        a = seg.equiv_radius
        expected = MU0 / (2 * math.pi) * seg.length * (
            math.log(2 * seg.length / a) - 1.0
        )
        L = build_L([seg])
        assert abs(float(L[0, 0]) - expected) / expected < 1e-10


# ── Potential matrix ──────────────────────────────────────────────────────────

class TestBuildP:
    def test_shape(self, single_segment):
        P = build_P([single_segment], OMEGA_1GHZ, eps_r=11.7, sigma_d=0.0)
        assert P.shape == (1, 1)

    def test_self_potential_positive_real(self, single_segment):
        P = build_P([single_segment], OMEGA_1GHZ, eps_r=11.7, sigma_d=0.0)
        assert float(P[0, 0].real) > 0

    def test_with_loss_has_imaginary(self, single_segment):
        P_lossless = build_P([single_segment], OMEGA_1GHZ, eps_r=11.7, sigma_d=0.0)
        P_lossy = build_P([single_segment], OMEGA_1GHZ, eps_r=11.7, sigma_d=10.0)
        # Lossy substrate should produce imaginary part in P
        assert abs(float(P_lossy[0, 0].imag)) > abs(float(P_lossless[0, 0].imag))

    def test_negative_omega_raises(self, single_segment):
        with pytest.raises(ValueError):
            build_P([single_segment], -1.0, eps_r=4.0, sigma_d=0.0)


# ── Capacitance and conductance ───────────────────────────────────────────────

class TestBuildCG:
    def test_shapes(self, two_segments):
        C, Gd = build_CG(two_segments, OMEGA_1GHZ, eps_r=11.7, sigma_d=0.0)
        assert C.shape == (2, 2)
        assert Gd.shape == (2, 2)

    def test_diagonal_C_positive(self, two_segments):
        C, _ = build_CG(two_segments, OMEGA_1GHZ, eps_r=11.7, sigma_d=0.0)
        for i in range(2):
            assert float(C[i, i]) > 0

    def test_Gd_zero_lossless(self, two_segments):
        _, Gd = build_CG(two_segments, OMEGA_1GHZ, eps_r=4.0, sigma_d=0.0)
        np.testing.assert_allclose(np.array(Gd), 0.0, atol=1e-6)


# ── FreqSolver ────────────────────────────────────────────────────────────────

class TestFreqSolver:
    def test_single_freq_returns_result(self, single_segment):
        solver = FreqSolver([single_segment])
        result = solver.solve(OMEGA_1GHZ)
        assert result.omega == OMEGA_1GHZ
        assert result.Z.shape == (1, 1)
        assert result.Y.shape == (1, 1)

    def test_Z_Y_are_inverses(self, single_segment):
        solver = FreqSolver([single_segment])
        result = solver.solve(OMEGA_1GHZ)
        ZY = np.array(result.Z) @ np.array(result.Y)
        np.testing.assert_allclose(ZY, np.eye(1), atol=1e-6)

    def test_Z_has_positive_real_part(self, single_segment):
        solver = FreqSolver([single_segment])
        result = solver.solve(OMEGA_1GHZ)
        assert float(result.Z[0, 0].real) > 0

    def test_Z_has_positive_imaginary_part(self, single_segment):
        """Inductive segment: Im(Z) > 0."""
        solver = FreqSolver([single_segment])
        result = solver.solve(OMEGA_1GHZ)
        assert float(result.Z[0, 0].imag) > 0

    def test_sweep_length(self, single_segment):
        solver = FreqSolver([single_segment])
        freqs = [1e8, 1e9, 1e10]
        results = solver.sweep(freqs)
        assert len(results) == 3

    def test_solve_current(self, single_segment):
        solver = FreqSolver([single_segment])
        v, result = solver.solve_current(OMEGA_1GHZ, [1.0 + 0j])
        assert v.shape == (1,)

    def test_empty_segments_raises(self):
        with pytest.raises(ValueError):
            FreqSolver([])

    def test_negative_omega_raises(self, single_segment):
        solver = FreqSolver([single_segment])
        with pytest.raises(ValueError):
            solver.solve(-1.0)

    def test_two_segment_solver(self, two_segments):
        solver = FreqSolver(two_segments)
        result = solver.solve(OMEGA_1GHZ)
        assert result.Z.shape == (2, 2)
        # Off-diagonal coupling should produce non-zero Z12
        assert float(abs(complex(result.Z[0, 1]))) > 0

    def test_inductance_increases_with_n_turns(self):
        """More turns → larger total inductance."""
        segs1 = spiral_inductor_segments(n_turns=1, inner_radius=30e-6,
                                          pitch=10e-6, width=5e-6,
                                          thickness=1e-6)
        segs2 = spiral_inductor_segments(n_turns=3, inner_radius=30e-6,
                                          pitch=10e-6, width=5e-6,
                                          thickness=1e-6)
        L1 = float(np.sum(np.array(build_L(segs1))))
        L2 = float(np.sum(np.array(build_L(segs2))))
        assert L2 > L1


# ── GDS spiral helper ─────────────────────────────────────────────────────────

class TestSpiralInductorSegments:
    def test_returns_segments(self):
        segs = spiral_inductor_segments(n_turns=2, inner_radius=50e-6,
                                         pitch=20e-6, width=5e-6,
                                         thickness=1e-6)
        assert len(segs) > 0
        assert all(isinstance(s, Segment) for s in segs)

    def test_all_segments_at_z_zero(self):
        segs = spiral_inductor_segments()
        for s in segs:
            assert abs(s.midpoint[2]) < 1e-12

    def test_correct_material(self):
        segs = spiral_inductor_segments(resistivity=2e-8)
        for s in segs:
            assert s.resistivity == pytest.approx(2e-8)

    def test_direction_is_unit_vector(self):
        segs = spiral_inductor_segments()
        for s in segs:
            assert abs(np.linalg.norm(s.direction) - 1.0) < 1e-10
