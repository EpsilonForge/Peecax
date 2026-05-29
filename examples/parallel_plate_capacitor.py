"""Simple parallel-plate capacitor check using Peecax.

Workflow
--------
1. Build two rectangular plates as sets of parallel PEEC segments.
2. Compute C matrix in vacuum (eps_r = 1, sigma_d = 0).
3. Extract equivalent two-conductor capacitance from a differential excitation.
4. Compare against C = eps0 * A / d.

Run with:

    python examples/parallel_plate_capacitor.py
"""

from __future__ import annotations

import math

import numpy as np

from peecax import EPS0, Segment, build_CG


# 1) Geometry parameters
L_M = 2.0e-3          # plate length [m]
W_M = 2.0e-3          # plate width  [m]
GAP_M = 0.3e-3        # plate separation [m]
STRIPS_PER_PLATE = 8
METAL_THICKNESS_M = 5e-6

# 2) Frequency / material parameters
OMEGA = 2.0 * math.pi * 1e6  # 1 MHz
EPS_R = 1.0
SIGMA_D = 0.0


def build_plate_segments(
    length_m: float,
    width_m: float,
    gap_m: float,
    strips_per_plate: int,
    thickness_m: float,
) -> tuple[list[Segment], int]:
    """Discretize two facing plates into parallel strips."""
    segs: list[Segment] = []
    dy = width_m / strips_per_plate

    for z in (-0.5 * gap_m, 0.5 * gap_m):
        for i in range(strips_per_plate):
            y = -0.5 * width_m + (i + 0.5) * dy
            segs.append(
                Segment.from_endpoints(
                    [-0.5 * length_m, y, z],
                    [0.5 * length_m, y, z],
                    width=dy,
                    thickness=thickness_m,
                )
            )
    return segs, strips_per_plate


def equivalent_two_plate_capacitance(C: np.ndarray, n_plate: int) -> float:
    """Use +/-0.5 V plate excitation and summed charge to get C_eq."""
    v = np.zeros(2 * n_plate)
    v[:n_plate] = 0.5
    v[n_plate:] = -0.5

    q = C @ v
    return abs(float(np.sum(q[:n_plate])))


print("Building discretized parallel-plate geometry ...")
segments, n_plate = build_plate_segments(
    length_m=L_M,
    width_m=W_M,
    gap_m=GAP_M,
    strips_per_plate=STRIPS_PER_PLATE,
    thickness_m=METAL_THICKNESS_M,
)
print(f"  Segments: {len(segments)}")

print("Solving for capacitance matrix ...")
C, _ = build_CG(segments, OMEGA, eps_r=EPS_R, sigma_d=SIGMA_D)
C_eq = equivalent_two_plate_capacitance(np.array(C), n_plate)

C_analytic = EPS0 * (L_M * W_M) / GAP_M
rel_err = abs(C_eq - C_analytic) / C_analytic

print("Results:")
print(f"  C_eq (PEEC)        : {C_eq:.4e} F")
print(f"  C_analytic (A/d)   : {C_analytic:.4e} F")
print(f"  Relative error     : {100.0 * rel_err:.2f} %")
