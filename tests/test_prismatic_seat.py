"""
Tests for prismatic seat-offset recovery (open-state scans).

q=0 in the generated URDF is the as-scanned pose and prismatic limits are
[0, travel], so a drawer scanned open could never close. The seat offset
measures how far the child can slide toward closed before its motion-facing
surfaces contact the parent's opposing surfaces; make_prismatic_joint uses it
to re-anchor q=0 at the closed pose.

These tests exercise `compute_prismatic_seat_offset`, which is pure
numpy/trimesh: no PyBullet simulation and no VLM call is required.

Run with pytest, or standalone:

    python tests/test_prismatic_seat.py
"""

import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from articulate_anything.api.odio_urdf import (  # noqa: E402
    compute_prismatic_seat_offset,
    engagement_capped_travel,
    sample_mesh_surface_with_normals,
)

AXIS = np.array([1.0, 0.0, 0.0])  # +x = opening direction
SPACING = 0.01


def _box(extents, center):
    b = trimesh.creation.box(extents=extents)
    b.apply_translation(center)
    return b


def _carcass():
    """Open-front shell: interior cavity 0.4 (x) x 0.3 (y) x 0.2 (z), opening at +x."""
    t = 0.02  # wall thickness
    parts = [
        _box((t, 0.34, 0.24), (-0.2 - t / 2, 0, 0)),        # back wall
        _box((0.4 + t, 0.34, t), (-t / 2, 0, 0.1 + t / 2)),  # top
        _box((0.4 + t, 0.34, t), (-t / 2, 0, -0.1 - t / 2)),  # bottom
        _box((0.4 + t, t, 0.2), (-t / 2, 0.15 + t / 2, 0)),  # left
        _box((0.4 + t, t, 0.2), (-t / 2, -0.15 - t / 2, 0)),  # right
    ]
    return trimesh.util.concatenate(parts)


def _drawer(open_by=0.0):
    """Tray with a proud front panel; closed = panel back flush with the shell's front rims.

    The tray (x in [-0.16, 0.20]) is shallower than the cavity (back at -0.20), so the
    governing closed contact is the oversized front panel against the wall front rims
    at x = 0.20 — the drawer-front-lip case, which the seat search must find.
    """
    t = 0.015
    body = [
        _box((0.36, 0.28, t), (0.02, 0, -0.09)),             # tray bottom
        _box((t, 0.28, 0.16), (-0.16 + t / 2, 0, 0)),        # tray back
        _box((0.36, t, 0.16), (0.02, 0.14, 0)),              # tray left
        _box((0.36, t, 0.16), (0.02, -0.14, 0)),             # tray right
        _box((t, 0.32, 0.22), (0.20 + t / 2, 0, 0)),         # proud front panel
    ]
    drawer = trimesh.util.concatenate(body)
    drawer.apply_translation([open_by, 0.0, 0.0])
    return drawer


def _points(mesh):
    return sample_mesh_surface_with_normals(mesh, SPACING)


def test_closed_scan_is_noop():
    cp, cn = _points(_drawer(open_by=0.0))
    pp, pn = _points(_carcass())
    offset, info = compute_prismatic_seat_offset(cp, cn, pp, pn, AXIS, max_travel=0.36)
    assert offset < 0.02, f"closed drawer should not be re-anchored, got {offset:.4f} ({info})"


def test_open_scan_recovers_offset():
    for d in (0.08, 0.15, 0.25):
        cp, cn = _points(_drawer(open_by=d))
        pp, pn = _points(_carcass())
        offset, info = compute_prismatic_seat_offset(cp, cn, pp, pn, AXIS, max_travel=0.36)
        assert info["status"] == "seated", info
        assert abs(offset - d) < 0.02, f"open_by={d}: recovered {offset:.4f} ({info})"


def test_no_blocking_geometry_is_skipped():
    # Two plates parallel to the axis: no surface faces the travel direction.
    plate = _box((0.4, 0.01, 0.2), (0, 0.05, 0))
    other = _box((0.4, 0.01, 0.2), (0, -0.05, 0))
    cp, cn = _points(plate)
    pp, pn = _points(other)
    offset, info = compute_prismatic_seat_offset(cp, cn, pp, pn, AXIS, max_travel=0.4)
    assert offset == 0.0
    assert info["status"] == "no_blocking_geometry"


def test_implausible_offset_gated_by_travel():
    cp, cn = _points(_drawer(open_by=0.25))
    pp, pn = _points(_carcass())
    offset, info = compute_prismatic_seat_offset(cp, cn, pp, pn, AXIS, max_travel=0.05)
    assert offset == 0.0
    assert info["status"] == "exceeds_travel"


def test_engagement_cap_limits_travel():
    # Real measurements from small_cabinet_open drawer_1: scanned child span
    # [-0.0886, 0.1711], parent front 0.0668, seat 0.0899, VLM travel = bbox 0.2597.
    # Engagement at closed = 0.0668 - (-0.0886 - 0.0899) = 0.2453; keep 10% of 0.2597.
    capped = engagement_capped_travel(
        (-0.0886, 0.1711), (-0.2508, 0.0668), seat_offset=0.0899, travel=0.2597)
    assert abs(capped - (0.2453 - 0.02597)) < 1e-6
    assert capped < 0.2597


def test_engagement_cap_keeps_scanned_pose_reachable():
    # A cap smaller than the seat would make the scanned pose unreachable; floor at seat.
    capped = engagement_capped_travel(
        (-0.01, 0.30), (-0.5, 0.0), seat_offset=0.25, travel=0.4)
    assert capped >= 0.25


def test_engagement_cap_noop_when_barely_engaged():
    # Child already almost entirely outside the parent at closed: cap not meaningful.
    capped = engagement_capped_travel(
        (0.0, 0.3), (-0.5, 0.02), seat_offset=0.0, travel=0.3)
    assert capped == 0.3


def test_engagement_cap_noop_when_travel_within_cap():
    capped = engagement_capped_travel(
        (-0.2, 0.0), (-0.25, 0.05), seat_offset=0.0, travel=0.1)
    assert capped == 0.1


def test_seat_info_reports_axial_spans():
    cp, cn = _points(_drawer(open_by=0.1))
    pp, pn = _points(_carcass())
    _, info = compute_prismatic_seat_offset(cp, cn, pp, pn, AXIS, max_travel=0.36)
    assert abs(info["parent_axial_span"][1] - 0.2) < 0.02  # carcass front rim ~0.20+t
    assert info["child_axial_span"][1] > info["child_axial_span"][0]


if __name__ == "__main__":
    test_closed_scan_is_noop()
    test_open_scan_recovers_offset()
    test_no_blocking_geometry_is_skipped()
    test_implausible_offset_gated_by_travel()
    test_engagement_cap_limits_travel()
    test_engagement_cap_keeps_scanned_pose_reachable()
    test_engagement_cap_noop_when_barely_engaged()
    test_engagement_cap_noop_when_travel_within_cap()
    test_seat_info_reports_axial_spans()
    print("all prismatic seat tests passed")
