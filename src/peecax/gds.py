"""GDSFactory integration – extract PEEC segments from GDS components.

This module converts GDSFactory metal paths/routes into lists of
:class:`~peecax.geometry.Segment` objects suitable for the PEEC solver.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .geometry import Segment, RHO_COPPER, MU0


def _polyline_to_segments(
    points: np.ndarray,
    width: float,
    thickness: float,
    z: float = 0.0,
    resistivity: float = RHO_COPPER,
    permeability: float = MU0,
) -> list[Segment]:
    """Convert an ordered list of 2-D waypoints to PEEC segments.

    Parameters
    ----------
    points:
        (M, 2) array of ``[x, y]`` waypoints in **metres**.
    width:
        Metal width [m].
    thickness:
        Metal thickness [m].
    z:
        Vertical position of the metal layer [m].
    resistivity:
        Metal resistivity [Ω·m].
    permeability:
        Metal permeability [H/m].

    Returns
    -------
    List of :class:`~peecax.geometry.Segment` objects.
    """
    segments: list[Segment] = []
    pts = np.asarray(points, dtype=float)
    for k in range(len(pts) - 1):
        p0 = np.append(pts[k], z)
        p1 = np.append(pts[k + 1], z)
        try:
            seg = Segment.from_endpoints(
                p0, p1,
                width=width,
                thickness=thickness,
                resistivity=resistivity,
                permeability=permeability,
            )
            segments.append(seg)
        except ValueError:
            # Skip zero-length segments (duplicate waypoints)
            pass
    return segments


def segments_from_path(
    path,
    width: float,
    thickness: float,
    z: float = 0.0,
    scale: float = 1e-6,
    resistivity: float = RHO_COPPER,
    permeability: float = MU0,
) -> list[Segment]:
    """Extract PEEC segments from a GDSFactory ``Path`` object.

    Parameters
    ----------
    path:
        A ``gdsfactory.path.Path`` instance.
    width:
        Metal width [m].
    thickness:
        Metal thickness [m].
    z:
        Vertical position of the metal layer [m].
    scale:
        Unit conversion factor from GDS database units to metres.
        GDSFactory uses µm by default, so ``scale=1e-6``.
    resistivity, permeability:
        Metal material properties.

    Returns
    -------
    list of :class:`~peecax.geometry.Segment`
    """
    pts = np.asarray(path.points, dtype=float) * scale
    return _polyline_to_segments(pts, width, thickness, z,
                                 resistivity, permeability)


def segments_from_component(
    component,
    layer: tuple[int, int],
    thickness: float,
    z: float = 0.0,
    scale: float = 1e-6,
    resistivity: float = RHO_COPPER,
    permeability: float = MU0,
) -> list[Segment]:
    """Extract PEEC segments from the skeleton of a GDSFactory component.

    The function traces the **centreline** of each polygon on ``layer`` by
    walking the polygon vertices.  This is a reasonable approximation for
    rectilinear metal traces.

    Parameters
    ----------
    component:
        A ``gdsfactory.Component`` instance.
    layer:
        GDS layer tuple ``(layer_number, datatype)`` identifying the metal.
    thickness:
        Metal thickness [m].
    z:
        Vertical position of the metal layer [m].
    scale:
        Unit conversion from GDS units (µm) to metres.
    resistivity, permeability:
        Metal material properties.

    Returns
    -------
    list of :class:`~peecax.geometry.Segment`
    """
    polygons = component.get_polygons(by="tuple").get(layer, [])
    segments: list[Segment] = []
    for poly in polygons:
        pts = np.asarray(poly, dtype=float) * scale  # (V, 2)
        # Close the polygon
        pts_closed = np.vstack([pts, pts[0]])
        segs = _polyline_to_segments(pts_closed, width=0.0,
                                     thickness=thickness, z=z,
                                     resistivity=resistivity,
                                     permeability=permeability)
        # Infer approximate width from adjacent segment lengths as a
        # placeholder; users should pass an explicit width for accuracy.
        for s in segs:
            s.width = s.length * 0.1  # rough 10 % aspect ratio default
        segments.extend(segs)
    return segments


def spiral_inductor_segments(
    n_turns: int = 2,
    inner_radius: float = 50e-6,
    pitch: float = 20e-6,
    width: float = 5e-6,
    thickness: float = 1e-6,
    z: float = 0.0,
    resistivity: float = RHO_COPPER,
) -> list[Segment]:
    """Generate PEEC segments for a square spiral inductor.

    This helper builds the inductor geometry analytically without
    requiring a full GDS file, making it useful for quick parameter
    sweeps and unit tests.

    Parameters
    ----------
    n_turns:
        Number of full turns.
    inner_radius:
        Half-side of the innermost square turn [m].
    pitch:
        Centre-to-centre spacing between turns [m].
    width:
        Metal width [m].
    thickness:
        Metal thickness [m].
    z:
        Vertical position of the metal layer [m].
    resistivity:
        Metal resistivity [Ω·m].

    Returns
    -------
    list of :class:`~peecax.geometry.Segment`
    """
    waypoints: list[np.ndarray] = []
    x, y = 0.0, 0.0
    r = inner_radius

    # Each full turn is 4 straight segments.
    # We walk outward with each half-turn.
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    side_len = 2 * r
    waypoints.append(np.array([x, y]))

    for turn_idx in range(n_turns * 4):
        dx, dy = directions[turn_idx % 4]
        # Grow the side length every two half-sides (complete half-turn)
        if turn_idx % 2 == 0 and turn_idx > 0:
            side_len += pitch
        x += dx * side_len
        y += dy * side_len
        waypoints.append(np.array([x, y]))
        if turn_idx % 2 == 1:
            side_len += pitch

    pts = np.array(waypoints)
    return _polyline_to_segments(pts, width, thickness, z, resistivity)
