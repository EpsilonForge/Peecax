"""Simple two-wire mutual-inductance check using Peecax.

Workflow
--------
1. Build two equal, parallel, co-directional segments.
2. Compute L matrix with Peecax.
3. Compare L12 with the mutual-inductance expression used by the solver.

Run with:

    python examples/parallel_wires_mutual_l.py
"""

from __future__ import annotations

import math

import numpy as np

from peecax import MU0, Segment, build_L


# 1) Geometry
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

# 2) PEEC mutual term
L = np.array(build_L([s1, s2]))
L12_num = float(L[0, 1])

# 3) Same closed form used in the matrix builder
eta = 0.5 * (s1.equiv_radius + s2.equiv_radius)
L12_analytic = (MU0 / (4.0 * math.pi)) * (s1.length * s2.length) / math.sqrt(
    sep_m ** 2 + eta ** 2
)

rel_err = abs(L12_num - L12_analytic) / L12_analytic

print("Parallel-wire mutual inductance check")
print(f"  L12_num      : {L12_num:.4e} H")
print(f"  L12_analytic : {L12_analytic:.4e} H")
print(f"  Relative err : {100.0 * rel_err:.6f} %")
