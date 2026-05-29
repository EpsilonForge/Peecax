"""IHP SG13G2 spiral-inductor PEEC analysis.

Workflow
--------
1. Activate the IHP PDK and build the TopMetal2 spiral inductor with a
   Metal1 guard ring using GDSFactory (exactly as specified in the design).
2. Extract the spiral centreline from the actual TopMetal2 GDS polygon by
   averaging the outer and inner hull-edge sequences.
3. Add the terminal lead-out wire segments from the component geometry.
4. Run a PEEC frequency sweep with the Peecax FreqSolver.
5. Report Z(f), L_eff(f), and Q(f) over 100 MHz – 10 GHz.

PDK note
--------
``gf.components.inductor`` always produces a single-turn spiral regardless of
the ``turns`` parameter (known IHP PDK limitation).

Run with::

    python examples/rfic_inductor.py
"""

from __future__ import annotations

import numpy as np
import gdsfactory as gf
from ihp import PDK

from peecax import (
    FreqSolver, SolverParams, Segment,
    fit_rlc, recover_LC, circulax_refine, z_rlc,
)
from peecax.geometry import RHO_COPPER, MU0


# ── IHP PDK activation ────────────────────────────────────────────────────────
PDK.activate()


# ── Inductor design parameters (matching the specified component) ─────────────
INDUCTOR_WIDTH  = 2      # trace width            [µm]
INDUCTOR_SPACE  = 2.1    # terminal gap           [µm]
INDUCTOR_DIAM   = 50     # nominal diameter       [µm]
INDUCTOR_TURNS  = 1      # always 1 due to PDK limitation

# IHP SG13G2 TopMetal2 layer-stack (AlCu):
#   thickness = 3.0 µm,  z_bottom = 11.03 µm
TM2_THICKNESS_M = 3.0e-6
TM2_Z_M         = (11.03 + 3.0 / 2.0) * 1e-6  # segment centroid z
TM2_LAYER       = (134, 0)                      # GDS layer tuple

# AlCu resistivity (~0.5 % Cu alloy, 20 °C)  [Ω·m]
RHO_ALCU = 3.0e-8

# Silicon substrate parameters
EPS_R_SI  = 11.7    # relative permittivity
SIGMA_D   = 10.0    # conductivity  [S/m]


# ── 1. Build the GDSFactory component (inductor + guard ring) ─────────────────

print("Building IHP spiral inductor component …")

c = gf.components.inductor(
    width=INDUCTOR_WIDTH,
    space=INDUCTOR_SPACE,
    diameter=INDUCTOR_DIAM,
    turns=INDUCTOR_TURNS,
    layer_metal="TopMetal2drawing",
    layer_inductor="INDdrawing",
    layer_metal_pin="TopMetal2drawing",
    layers_no_fill=("NoMetFillerdrawing",),
).copy()

# ── Guard ring dimensions based on the inductor bounding box ─────────────────
bbox = c.bbox()
xmin, ymin = bbox.left, bbox.bottom
xmax, ymax = bbox.right, bbox.top

margin_outer = 0.0
margin_inner = -15.0

ol,   oright = xmin - margin_outer, xmax + margin_outer
ob,   ot     = ymin - margin_outer, ymax + margin_outer
il,   ir     = xmin - margin_inner, xmax + margin_inner
ib,   it_    = ymin - margin_inner, ymax + margin_inner

w_v  = il - ol       # width of vertical guard-ring walls
h_h  = ot - it_      # height of horizontal guard-ring walls
over = 0.5           # overlap to fuse pieces in Gmsh

# Left wall
c.add_ref(
    gf.components.rectangle(
        size=(w_v + over, ot - ob), layer="Metal1drawing", centered=True
    )
).move((ol + w_v / 2 + over / 2, (ot + ob) / 2))
# Right wall
c.add_ref(
    gf.components.rectangle(
        size=(w_v + over, ot - ob), layer="Metal1drawing", centered=True
    )
).move((oright - w_v / 2 - over / 2, (ot + ob) / 2))
# Top wall
c.add_ref(
    gf.components.rectangle(
        size=(oright - ol, h_h + over), layer="Metal1drawing", centered=True
    )
).move(((oright + ol) / 2, ot - h_h / 2 - over / 2))
# Bottom wall
c.add_ref(
    gf.components.rectangle(
        size=(oright - ol, h_h + over), layer="Metal1drawing", centered=True
    )
).move(((oright + ol) / 2, ob + h_h / 2 + over / 2))

cc = c.copy()
c.draw_ports()

print(f"  Component bbox          : {bbox}")
print(f"  GDS layers present      : {sorted(c.layers)}")


# ── 2. Extract spiral centreline from the TopMetal2 GDS polygon ───────────────

def _ring_centreline_um(component, layer=TM2_LAYER, trace_width_um=INDUCTOR_WIDTH):
    """Return (N+2, 2) centreline waypoints [µm] for the spiral ring.

    The main TopMetal2 polygon is a closed ring (horseshoe) whose hull
    consists of an *outer* boundary (8 pts for a single-turn square spiral)
    followed by an *inner* boundary (12 pts).  The centreline of the metal
    trace is the midpoint between corresponding outer and inner edge vertices.

    Two extra terminal connection points are prepended/appended so that the
    path runs from the bottom of the right lead-out wire to the bottom of the
    left lead-out wire, making it suitable for a 2-port PEEC model.
    """
    polys = component.get_polygons(by="tuple")
    ring_polys = polys.get(layer, [])
    if not ring_polys:
        raise RuntimeError(f"No polygons found on layer {layer}")

    # Main ring = polygon with the most hull vertices
    ring_poly = max(ring_polys, key=lambda p: p.num_points_hull())

    # Extract hull as (M, 2) array in µm (GDS units are nm → /1000)
    hull = np.array(
        [(pt.x / 1000.0, pt.y / 1000.0) for pt in ring_poly.each_point_hull()],
        dtype=float,
    )

    # Find the 4 hull points at or near the bottom (the terminal-gap region).
    # For a single-turn square spiral these are always 4 consecutive points.
    y_bottom = hull[:, 1].min()
    gap_idx = np.where(hull[:, 1] <= y_bottom + 1.0)[0]  # within 1 µm of bottom
    # gap_idx[0,1] are end of outer boundary; gap_idx[2,3] are start of inner.
    n_outer_end   = gap_idx[1]   # inclusive
    n_inner_start = gap_idx[2]   # inclusive

    outer     = hull[: n_outer_end + 1]           # 8 pts: outer edge CCW
    inner_all = hull[n_inner_start:]               # 12 pts: inner edge + gap
    # First 2 and last 2 of inner_all are the gap side-edges; middle 8 = ring.
    inner_ring = inner_all[2:-2]                   # 8 pts: inner ring edge

    # Reverse inner_ring so its direction matches outer (both go CCW)
    inner_ring_rev = inner_ring[::-1]

    assert len(outer) == len(inner_ring_rev), (
        f"Outer/inner count mismatch: {len(outer)} vs {len(inner_ring_rev)}"
    )
    ring_cl = (outer + inner_ring_rev) / 2.0  # (8, 2) centreline pts [µm]

    # Terminal lead-out rectangles: the two 4-pt polygons with the largest y-span
    term_polys = sorted(
        [p for p in ring_polys if p.num_points_hull() == 4],
        key=lambda p: (p.bbox().top - p.bbox().bottom),
        reverse=True,
    )[:2]

    term_centres = []
    for tp in sorted(term_polys, key=lambda p: p.bbox().left):
        bb = tp.bbox()
        xc  = (bb.left  + bb.right) / 2.0 / 1000.0   # µm
        ybot = bb.bottom / 1000.0                       # µm
        ytop = bb.top    / 1000.0                       # µm
        term_centres.append((xc, ybot, ytop))

    # term_centres[0] = left terminal, term_centres[1] = right terminal
    (xl, yl_bot, yl_top), (xr, yr_bot, yr_top) = term_centres

    # Full path: bottom of right terminal → ring centreline → bottom of left terminal
    # ring_cl[7] = bottom-right ring entry, ring_cl[0] = bottom-left ring exit
    path_um = np.vstack([
        [xr, yr_bot],        # port A (right terminal bottom)
        [xr, yr_top],        # right terminal top / ring entry
        ring_cl[7],          # bottom-right ring foot
        ring_cl[6],          # lower-right miter corner
        ring_cl[5],          # upper-right miter corner
        ring_cl[4],          # top-right miter corner
        ring_cl[3],          # top-left  miter corner
        ring_cl[2],          # upper-left miter corner
        ring_cl[1],          # lower-left miter corner
        ring_cl[0],          # bottom-left ring foot
        [xl, yl_top],        # left terminal top / ring exit
        [xl, yl_bot],        # port B (left terminal bottom)
    ])
    return path_um


print("Extracting spiral centreline from GDS polygon …")
centreline_um = _ring_centreline_um(c)

print(f"  Centreline waypoints    : {len(centreline_um)}")
total_length_um = float(
    np.sum(np.linalg.norm(np.diff(centreline_um, axis=0), axis=1))
)
print(f"  Total conductor length  : {total_length_um:.1f} µm")


# ── 3. Build PEEC segment list from the centreline ───────────────────────────

def _segments_from_waypoints(
    waypoints_um: np.ndarray,
    width_m: float,
    thickness_m: float,
    z_m: float,
    resistivity: float = RHO_ALCU,
    permeability: float = MU0,
) -> list[Segment]:
    """Create Segment objects from 2-D (x, y) waypoints in µm."""
    segs: list[Segment] = []
    for k in range(len(waypoints_um) - 1):
        p0 = np.array([waypoints_um[k,   0] * 1e-6, waypoints_um[k,   1] * 1e-6, z_m])
        p1 = np.array([waypoints_um[k+1, 0] * 1e-6, waypoints_um[k+1, 1] * 1e-6, z_m])
        try:
            seg = Segment.from_endpoints(
                p0, p1,
                width=width_m,
                thickness=thickness_m,
                resistivity=resistivity,
                permeability=permeability,
            )
            segs.append(seg)
        except ValueError:
            pass  # skip zero-length duplicates
    return segs


WIDTH_M = INDUCTOR_WIDTH * 1e-6   # 2 µm → m

segments = _segments_from_waypoints(
    centreline_um,
    width_m=WIDTH_M,
    thickness_m=TM2_THICKNESS_M,
    z_m=TM2_Z_M,
    resistivity=RHO_ALCU,
)

print(f"  PEEC segments           : {len(segments)}")


# ── 4. PEEC frequency sweep  100 MHz – 10 GHz ────────────────────────────────

params = SolverParams(eps_r=EPS_R_SI, sigma_d=SIGMA_D)
solver  = FreqSolver(segments, params)

freqs_hz = np.logspace(9, np.log10(20e9), 40)   # 100 MHz … 20 GHz (40 points)

print("\nRunning PEEC frequency sweep (100 MHz – 200 GHz) …")
results = solver.sweep(freqs_hz)


# ── 5. Report Z_11, L_eff, Q ─────────────────────────────────────────────────

# Collect Z11 across the sweep for downstream RLC fitting
Z11_sim = np.array([complex(res.Z[0, 0]) for res in results])

print(f"\n{'Freq (GHz)':>12} {'|Z_11| (Ω)':>12} {'L_eff (nH)':>12} {'Q':>8}")
print("-" * 52)

for f, Z11 in zip(freqs_hz, Z11_sim):
    omega = 2.0 * np.pi * f
    Leff  = np.imag(Z11) / omega * 1e9
    Q     = np.imag(Z11) / np.real(Z11) if np.real(Z11) > 0 else float("nan")
    print(f"{f/1e9:>12.3f} {abs(Z11):>12.4f} {Leff:>12.4f} {Q:>8.2f}")


# ── 6. Partial inductance matrix ──────────────────────────────────────────────

print("\nPartial inductance matrix L [nH] (frequency-independent):")
L_nH = np.array(results[0].L) * 1e9
with np.printoptions(precision=3, suppress=True, linewidth=120):
    print(L_nH)

print("\nDone.")


# ── 7. RLC compact-model extraction ──────────────────────────────────────────

print("\n─── Stage 1: JAX/Optax RLC fit ──────────────────────────────────────")
f0_fit, Q_fit, R_fit = fit_rlc(freqs_hz, Z11_sim, steps=2000, lr=0.01)
L_fit, C_fit = recover_LC(f0_fit, Q_fit, R_fit)

print(f"\n  R  = {R_fit:.6f} Ω")
print(f"  L  = {L_fit * 1e12:.4f} pH")
print(f"  C  = {C_fit * 1e15:.4f} fF")
print(f"  f0 = {f0_fit / 1e9:.4f} GHz   Q = {Q_fit:.3f}")

print("\n─── Stage 2: Circulax inverse design ────────────────────────────────")
R_cx, L_cx, C_cx = circulax_refine(
    freqs_hz, Z11_sim,
    R_ini=R_fit, L_ini=L_fit, C_ini=C_fit,
    steps=200, lr=1e-2,
)
f0_cx = 1.0 / (2 * np.pi * np.sqrt(L_cx * C_cx))
Q_cx  = 2 * np.pi * f0_cx * L_cx / R_cx

print(f"\n  R  = {R_cx:.6f} Ω")
print(f"  L  = {L_cx * 1e12:.4f} pH")
print(f"  C  = {C_cx * 1e15:.4f} fF")
print(f"  f0 = {f0_cx / 1e9:.4f} GHz   Q = {Q_cx:.3f}")


# ── 8. Comparison plot: |Z| and phase ─────────────────────────────────────────

import matplotlib.pyplot as plt

# Evaluate both models over the simulation frequencies
Z_analytic = np.array([R_fit  * z_rlc(f / f0_fit, Q_fit) for f in freqs_hz])

# Circulax model evaluated via z_rlc with refined parameters
Z_cx_eval  = np.array([R_cx   * z_rlc(f / f0_cx,  Q_cx)  for f in freqs_hz])

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Panel 1 – magnitude
axes[0].loglog(freqs_hz / 1e9, np.abs(Z11_sim),   ".",  ms=4,
               label="PEEC simulation")
axes[0].loglog(freqs_hz / 1e9, np.abs(Z_analytic), "-",  lw=1.5,
               label=f"JAX/Optax fit   L={L_fit*1e12:.1f} pH  "
                     f"C={C_fit*1e15:.1f} fF  R={R_fit:.3f} Ω")
axes[0].loglog(freqs_hz / 1e9, np.abs(Z_cx_eval),  "--", lw=1.5,
               label=f"Circulax fit    L={L_cx*1e12:.1f} pH  "
                     f"C={C_cx*1e15:.1f} fF  R={R_cx:.3f} Ω")
axes[0].set_ylabel(r"$|Z_{11}|$ (Ω)")
axes[0].set_title("IHP SG13G2 Spiral Inductor – RLC Compact Model Extraction")
axes[0].legend(fontsize=8)
axes[0].grid(True, which="both", linestyle="--", linewidth=0.5)

# Panel 2 – phase
axes[1].semilogx(freqs_hz / 1e9, np.angle(Z11_sim,   deg=True), ".",  ms=4,
                 label="PEEC simulation")
axes[1].semilogx(freqs_hz / 1e9, np.angle(Z_analytic, deg=True), "-",  lw=1.5,
                 label="JAX/Optax fit")
axes[1].semilogx(freqs_hz / 1e9, np.angle(Z_cx_eval,  deg=True), "--", lw=1.5,
                 label="Circulax fit")
axes[1].set_xlabel("Frequency (GHz)")
axes[1].set_ylabel(r"$\angle Z_{11}$ (°)")
axes[1].legend(fontsize=8)
axes[1].grid(True, which="both", linestyle="--", linewidth=0.5)

fig.tight_layout()
plt.savefig("examples/inductor_impedance.png", dpi=150)
print("\nPlot saved to examples/inductor_impedance.png")
plt.show()


# ── 9. Imaginary impedance plot: Im(Z11) ─────────────────────────────────────

fig_im, ax_im = plt.subplots(1, 1, figsize=(9, 4.5))

# Log y-axis requires positive values, so plot |Im(Z11)|.
ax_im.loglog(freqs_hz / 1e9, np.abs(np.imag(Z11_sim)), ".", ms=4,
               label="PEEC simulation")
ax_im.loglog(freqs_hz / 1e9, np.abs(np.imag(Z_analytic)), "-", lw=1.5,
               label="JAX/Optax fit")
ax_im.loglog(freqs_hz / 1e9, np.abs(np.imag(Z_cx_eval)), "--", lw=1.5,
               label="Circulax fit")

ax_im.set_xlabel("Frequency (GHz)")
ax_im.set_ylabel(r"$|\mathrm{Im}(Z_{11})|$ (Ω)")
ax_im.set_title("IHP SG13G2 Spiral Inductor – |Im(Z11)| (log-log)")
ax_im.legend(fontsize=8)
ax_im.grid(True, which="both", linestyle="--", linewidth=0.5)

fig_im.tight_layout()
plt.savefig("examples/inductor_imag_z.png", dpi=150)
print("Plot saved to examples/inductor_imag_z.png")
plt.show()


# ── 10. Slope analysis of imaginary impedance ────────────────────────────────

def _slope_loglog(x: np.ndarray, y: np.ndarray) -> float:
    """Return slope m from log10(y) = m*log10(x) + b."""
    m, _ = np.polyfit(np.log10(x), np.log10(y), 1)
    return float(m)


def _power_law_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y = k*x^m in log-space and return (m, k)."""
    m, b = np.polyfit(np.log10(x), np.log10(y), 1)
    k = 10.0 ** b
    return float(m), float(k)


def _slope_linear(x: np.ndarray, y: np.ndarray) -> float:
    """Return slope m from y = m*x + b."""
    m, _ = np.polyfit(x, y, 1)
    return float(m)


f_ghz = freqs_hz / 1e9
f_hz = freqs_hz
im_sim_abs = np.abs(np.imag(Z11_sim))
im_fit_abs = np.abs(np.imag(Z_analytic))
im_cx_abs = np.abs(np.imag(Z_cx_eval))

mask = (
    np.isfinite(f_ghz)
    & np.isfinite(im_sim_abs)
    & np.isfinite(im_fit_abs)
    & np.isfinite(im_cx_abs)
    & (f_ghz > 0.0)
    & (im_sim_abs > 0.0)
    & (im_fit_abs > 0.0)
    & (im_cx_abs > 0.0)
)

f_fit = f_ghz[mask]
im_sim_fit = im_sim_abs[mask]
im_fit_fit = im_fit_abs[mask]
im_cx_fit = im_cx_abs[mask]

slope_loglog_sim = _slope_loglog(f_fit, im_sim_fit)
slope_loglog_fit = _slope_loglog(f_fit, im_fit_fit)
slope_loglog_cx = _slope_loglog(f_fit, im_cx_fit)

slope_linear_sim = _slope_linear(f_fit, im_sim_fit)
slope_linear_fit = _slope_linear(f_fit, im_fit_fit)
slope_linear_cx = _slope_linear(f_fit, im_cx_fit)

# Power-law fit on SI frequency so the coefficient has physical units.
m_sim, k_sim = _power_law_fit(f_hz[mask], im_sim_fit)
m_fit, k_fit = _power_law_fit(f_hz[mask], im_fit_fit)
m_cx, k_cx = _power_law_fit(f_hz[mask], im_cx_fit)

# If |Im(Z)| = k*f^m and m ≈ 1, an inductor has k ≈ 2*pi*L.
L_from_powerlaw_sim = k_sim / (2.0 * np.pi)
L_from_powerlaw_fit = k_fit / (2.0 * np.pi)
L_from_powerlaw_cx = k_cx / (2.0 * np.pi)

print("\nSlope analysis for |Im(Z11)| vs frequency:")
print("  Log-log slope  (PEEC)     : "
    f"{slope_loglog_sim:.6f}  (|Im(Z)| ~ f^{slope_loglog_sim:.3f})")
print("  Log-log slope  (JAX fit)  : "
    f"{slope_loglog_fit:.6f}  (|Im(Z)| ~ f^{slope_loglog_fit:.3f})")
print("  Log-log slope  (Circulax) : "
    f"{slope_loglog_cx:.6f}  (|Im(Z)| ~ f^{slope_loglog_cx:.3f})")

print("  Linear slope   (PEEC)     : "
    f"{slope_linear_sim:.6e} ohm/GHz")
print("  Linear slope   (JAX fit)  : "
    f"{slope_linear_fit:.6e} ohm/GHz")
print("  Linear slope   (Circulax) : "
    f"{slope_linear_cx:.6e} ohm/GHz")

print("\nInductance from power-law coefficient (k/(2*pi)):")
print("  L_power (PEEC)     : "
    f"{L_from_powerlaw_sim * 1e12:.4f} pH   (m = {m_sim:.6f})")
print("  L_power (JAX fit)  : "
    f"{L_from_powerlaw_fit * 1e12:.4f} pH   (m = {m_fit:.6f})")
print("  L_power (Circulax) : "
    f"{L_from_powerlaw_cx * 1e12:.4f} pH   (m = {m_cx:.6f})")
