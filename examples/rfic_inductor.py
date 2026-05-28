"""RFIC spiral-inductor PEEC analysis demo.

This script demonstrates how to use Peecax together with GDSFactory to:

1. Build a square spiral inductor component in GDSFactory.
2. Extract its metal path as PEEC segments.
3. Run a frequency sweep with the PEEC solver.
4. Extract and print the equivalent impedance Z(f).
5. Estimate the effective inductance L_eff(f) and quality factor Q(f).

Run with::

    python examples/rfic_inductor.py
"""

from __future__ import annotations

import numpy as np
import gdsfactory as gf
from gdsfactory.gpdk import get_generic_pdk

# Activate the generic PDK so cross-section extrusion works without a custom PDK.
get_generic_pdk().activate()

import peecax
from peecax import FreqSolver, SolverParams, spiral_inductor_segments
from peecax.gds import segments_from_path


# ── Spiral geometry parameters ────────────────────────────────────────────────

N_TURNS = 2
INNER_RADIUS_UM = 25.0          # half-side of innermost square  [µm]
PITCH_UM = 10.0                  # centre-to-centre turn spacing  [µm]
WIDTH_UM = 5.0                   # metal trace width              [µm]
THICKNESS_UM = 1.0               # metal thickness                [µm]

# Convert to SI [m]
INNER_RADIUS_M = INNER_RADIUS_UM * 1e-6
PITCH_M = PITCH_UM * 1e-6
WIDTH_M = WIDTH_UM * 1e-6
THICKNESS_M = THICKNESS_UM * 1e-6


# ── 1. Build the GDSFactory component ────────────────────────────────────────

def _spiral_waypoints(n_turns, inner_radius_um, pitch_um):
    """Return (M, 2) waypoint array in µm for a square outward spiral."""
    waypoints = []
    x, y = 0.0, 0.0
    r = inner_radius_um
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    side = 2.0 * r
    waypoints.append([x, y])
    for k in range(n_turns * 4):
        dx, dy = dirs[k % 4]
        if k % 2 == 0 and k > 0:
            side += pitch_um
        x += dx * side
        y += dy * side
        waypoints.append([x, y])
        if k % 2 == 1:
            side += pitch_um
    return np.array(waypoints, dtype=float)


print("Building GDSFactory spiral inductor component …")

waypoints_um = _spiral_waypoints(N_TURNS, INNER_RADIUS_UM, PITCH_UM)

# Create a GDSFactory Path from the waypoints and extrude it
gds_path = gf.path.Path(waypoints_um)
cross_section = gf.cross_section.strip(width=WIDTH_UM)
component = gf.path.extrude(gds_path, cross_section=cross_section)

print(f"  Number of GDS waypoints : {len(waypoints_um)}")
print(f"  GDS layers              : {component.layers}")


# ── 2. Extract PEEC segments from the GDSFactory Path ────────────────────────

# segments_from_path converts the GDSFactory Path waypoints → Segment list.
segments = segments_from_path(
    gds_path,
    width=WIDTH_M,
    thickness=THICKNESS_M,
    z=0.0,
    scale=1e-6,                  # GDSFactory uses µm; convert to m
)

print(f"  PEEC segments extracted : {len(segments)}")
total_um = sum(s.length for s in segments) * 1e6
print(f"  Total conductor length  : {total_um:.1f} µm")


# ── 3. Set up the PEEC solver ─────────────────────────────────────────────────

# Silicon substrate  (ε_r ≈ 11.7, σ_d ≈ 10 S/m for lightly-doped Si)
params = SolverParams(eps_r=11.7, sigma_d=10.0)
solver = FreqSolver(segments, params)


# ── 4. Frequency sweep  100 MHz – 10 GHz ─────────────────────────────────────

freqs_hz = np.logspace(8, 10, 20)   # 20 log-spaced points
print("\nRunning PEEC frequency sweep (100 MHz – 10 GHz) …")
results = solver.sweep(freqs_hz)


# ── 5. Report Z_11, L_eff, Q ─────────────────────────────────────────────────

print(f"\n{'Freq (GHz)':>12} {'|Z_11| (Ω)':>12} {'L_eff (nH)':>12} {'Q':>8}")
print("-" * 48)

for f, res in zip(freqs_hz, results):
    omega = 2.0 * np.pi * f
    Z11 = complex(res.Z[0, 0])
    Leff_nH = np.imag(Z11) / omega * 1e9
    Q = np.imag(Z11) / np.real(Z11) if np.real(Z11) > 0 else float("nan")
    print(f"{f/1e9:>12.3f} {abs(Z11):>12.4f} {Leff_nH:>12.4f} {Q:>8.2f}")


# ── 6. Summary ────────────────────────────────────────────────────────────────

print("\nInductance matrix L (nH) – frequency-independent partial inductances:")
L_nH = np.array(results[0].L) * 1e9
with np.printoptions(precision=3, suppress=True, linewidth=120):
    print(L_nH)

print("\nDemo complete.")
