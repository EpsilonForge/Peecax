"""Simple straight-wire R and L check using Peecax.

Workflow
--------
1. Build one straight copper segment.
2. Compute near-DC resistance and self-inductance.
3. Compare against closed-form formulas:
   R = rho * l / (w * t)
   L = mu0/(2*pi) * l * (ln(2l/a) - 1)

Run with:

    python examples/straight_wire_rl.py
"""

from __future__ import annotations

import math

import numpy as np

from peecax import MU0, Segment, build_L, build_R


# 1) Segment geometry
L_M = 200e-6
W_M = 10e-6
T_M = 2e-6

segment = Segment.from_endpoints(
    [0.0, 0.0, 0.0],
    [L_M, 0.0, 0.0],
    width=W_M,
    thickness=T_M,
)

# 2) Near-DC for resistance check
omega_dc = 2.0 * math.pi * 1.0
R = np.array(build_R([segment], omega_dc))
L = np.array(build_L([segment]))

R_num = float(R[0, 0].real)
L_num = float(L[0, 0])

# 3) Closed forms
R_analytic = segment.resistivity * segment.length / (segment.width * segment.thickness)
L_analytic = (MU0 / (2.0 * math.pi)) * segment.length * (
    math.log(2.0 * segment.length / segment.equiv_radius) - 1.0
)

err_R = abs(R_num - R_analytic) / R_analytic
err_L = abs(L_num - L_analytic) / L_analytic

print("Straight-wire sanity check")
print(f"  R_num      : {R_num:.4e} ohm")
print(f"  R_analytic : {R_analytic:.4e} ohm")
print(f"  R error    : {100.0 * err_R:.4f} %")
print()
print(f"  L_num      : {L_num:.4e} H")
print(f"  L_analytic : {L_analytic:.4e} H")
print(f"  L error    : {100.0 * err_L:.4f} %")
