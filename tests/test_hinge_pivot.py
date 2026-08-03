"""
Regression tests for the hinge-pivot geometry (see docs/hinge-pivot-geometry.md).

The defect these guard against survived because **no test ever exercised a
non-canonical pose**. Indexing a corner of a part's axis-aligned bounding box
gives the right answer when the part is fully closed (theta = 0) and when it is
fully upright (theta = 90 deg), and is maximally wrong in between - which is
exactly the regime a corpus of canonically-authored assets never visits.
The sweep test below therefore walks theta across the whole range.

These tests exercise `compute_hinge_pivot_from_points`, which is pure numpy /
scipy: no PyBullet simulation and no VLM call is required.

Run with pytest, or standalone:

    python tests/test_hinge_pivot.py
"""

import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from articulate_anything.api.odio_urdf import (  # noqa: E402
    PIVOT_SNAP_TOLERANCE_FRACTION,
    PIVOT_SNAP_TOLERANCE_MIN,
    compute_aabb_vertices,
    compute_hinge_pivot_from_points,
    perpendicular_basis,
    sample_mesh_surface,
)

# Surface-sampling resolution used by the fixtures. Coarser than the production
# default (PIVOT_SAMPLE_SPACING_FRACTION) purely to keep the suite quick; the
# recovered pivot is not sensitive to it at this scale.
TEST_SPACING = 0.0025

# The synthetic laptop: a thin lid hinged along the back-bottom edge of a base.
HINGE_AXIS = np.array([1.0, 0.0, 0.0])       # the hinge runs along x
HINGE_Y, HINGE_Z = -0.15, 0.0                # the hinge line, in the plane perp to x
TRUE_PIVOT = np.array([0.0, HINGE_Y, HINGE_Z])

LID_LENGTH = 0.30
LID_THICKNESS = 0.004

# theta is the lid's opening angle: 0 = shut flat on the base, 90 = upright,
# >90 = leaning back past vertical. The angles above 90 are the important ones -
# see `test_aabb_corner_error_peaks_past_vertical`. The real reconstructed laptop
# that motivated this work sits at about 116 degrees.
SWEEP_ANGLES_DEG = [0, 15, 30, 45, 60, 75, 90, 105, 116, 120, 135]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def perp_distance(a, b, axis):
    """
    Distance between two points measured in the plane perpendicular to `axis`.

    A revolute joint is a *line*, so displacing a pivot along the axis produces
    an identical joint (docs section 6). Euclidean distance would flag such a
    harmless pivot and mis-rank a genuinely broken one; this is the metric that
    actually corresponds to joint behaviour.
    """
    _, e1, e2 = perpendicular_basis(axis)
    basis = np.stack([e1, e2])
    return float(np.linalg.norm(np.asarray(a, float) @ basis.T
                                - np.asarray(b, float) @ basis.T))


def snap_gate_would_fire(pivot, child_points, axis):
    """
    Mirrors the decision rule inside `Robot._snap_pivot_to_hinge`.

    The real method is decorated with `@pybullet_session` and needs a loaded
    simulator, so the arithmetic is reproduced here against the same module
    constants. It is the *condition* that matters: a pivot is repaired only when
    it lies further than the tolerance from the child, measured perpendicular to
    the axis.
    """
    child = np.asarray(child_points, float)
    scale = float(np.linalg.norm(child.max(axis=0) - child.min(axis=0)))
    _, e1, e2 = perpendicular_basis(axis)
    basis = np.stack([e1, e2])
    offset = float(np.min(np.linalg.norm(
        child @ basis.T - np.asarray(pivot, float) @ basis.T, axis=1)))
    tolerance = max(PIVOT_SNAP_TOLERANCE_FRACTION * scale, PIVOT_SNAP_TOLERANCE_MIN)
    return offset > tolerance, offset, tolerance


def make_laptop(theta_deg, spacing=TEST_SPACING):
    """
    Builds a lid/base pair with the lid opened `theta_deg` about the hinge line.

    The hinge is the lid's back-bottom edge, at (y, z) = (HINGE_Y, HINGE_Z),
    running along x. At theta = 0 the lid lies flat on the base; at theta = 90 it
    stands upright; beyond that it leans back past vertical, which is the pose a
    filmed laptop is usually in. The base overhangs the hinge in -y so that lid
    and base remain in contact across the whole sweep, as a real hinge would.

    Returns (child_points, parent_points, aabb_corner_hint).
    """
    base = trimesh.creation.box(extents=[0.40, 0.35, 0.10])
    base.apply_translation([0.0, -0.025, -0.05])          # y in [-0.20, 0.15], z in [-0.10, 0]

    lid = trimesh.creation.box(extents=[0.40, LID_LENGTH, LID_THICKNESS])
    lid.apply_translation([0.0, 0.0, LID_THICKNESS / 2])  # y in [-0.15, 0.15], z in [0, t]

    rotation = trimesh.transformations.rotation_matrix(
        np.radians(theta_deg), HINGE_AXIS, TRUE_PIVOT)
    lid.apply_transform(rotation)

    child_points = sample_mesh_surface(lid, spacing)
    parent_points = sample_mesh_surface(base, spacing)

    # The idiom under test: the Back-Left-Bottom corner of the lid's *axis-aligned*
    # bounding box. Correct at theta = 0, drifts off the mesh as the lid tilts.
    aabb_corner = compute_aabb_vertices(*lid.bounds)[0]
    return child_points, parent_points, aabb_corner


# --------------------------------------------------------------------------- #
# 1. the theta sweep - the test whose absence let the defect ship
# --------------------------------------------------------------------------- #
def test_pivot_recovered_at_every_opening_angle():
    """
    The recovered pivot must be within a few mm of the true hinge at EVERY
    opening angle - including the past-vertical poses where the AABB corner is
    badly wrong, and the canonical poses where it happens to be right.

    A solver validated only at theta = 0 (the pose every PartNet-Mobility asset
    is authored in) looks perfect and is unusable on scanned input.
    """
    tolerance_m = 0.005  # 5 mm
    failures = []
    for theta in SWEEP_ANGLES_DEG:
        child, parent, hint = make_laptop(theta)
        pivot = compute_hinge_pivot_from_points(
            child, parent, HINGE_AXIS, hint_point=hint)
        error = perp_distance(pivot, TRUE_PIVOT, HINGE_AXIS)
        if error > tolerance_m:
            failures.append((theta, round(error, 5),
                             round(perp_distance(hint, TRUE_PIVOT, HINGE_AXIS), 5)))
    assert not failures, (
        "pivot recovery failed at these angles "
        "(theta_deg, solver_error_m, aabb_corner_error_m): " + repr(failures))


def test_solver_never_does_worse_than_the_aabb_corner():
    """The whole point of the change: the solver must beat, or at minimum match,
    the idiom it replaces - at every pose, not on average."""
    regressions = []
    for theta in SWEEP_ANGLES_DEG:
        child, parent, hint = make_laptop(theta)
        pivot = compute_hinge_pivot_from_points(
            child, parent, HINGE_AXIS, hint_point=hint)
        solver_err = perp_distance(pivot, TRUE_PIVOT, HINGE_AXIS)
        corner_err = perp_distance(hint, TRUE_PIVOT, HINGE_AXIS)
        if solver_err > corner_err + 1e-4:
            regressions.append((theta, round(solver_err, 5), round(corner_err, 5)))
    assert not regressions, (
        "solver was worse than the raw AABB corner at "
        "(theta_deg, solver_error_m, corner_error_m): " + repr(regressions))


def test_aabb_corner_error_peaks_past_vertical():
    """
    Guards the *premise* of the fix: if the fixture stops reproducing the real
    failure mode, the sweep test above would pass vacuously.

    NOTE - this contradicts the error model in docs section 4, and the measured
    behaviour is what is encoded here. For a lid hinged at its own back-bottom
    edge and labelled with the Back-Left-Bottom corner, the corner tracks the
    hinge almost exactly from 0 to 90 degrees (the residual is just the lid's
    thickness). It only goes wrong once the lid passes vertical, because only
    then are min(y) and min(z) attained at OPPOSITE ends of the part - which is
    the actual mechanism described in docs section 3. The error then grows like
    L*|cos(theta)|, reaching ~135 mm at the 116 degrees of the real laptop.
    """
    errors = {theta: perp_distance(make_laptop(theta)[2], TRUE_PIVOT, HINGE_AXIS)
              for theta in SWEEP_ANGLES_DEG}

    for theta in (0, 15, 30, 45, 60, 75, 90):
        assert errors[theta] <= 2 * LID_THICKNESS, (
            f"corner should still track the hinge at {theta} deg, "
            f"got {errors[theta] * 1000:.1f} mm")

    for theta in (105, 116, 120, 135):
        assert errors[theta] > 0.05, (
            f"corner should be badly wrong at {theta} deg, "
            f"got {errors[theta] * 1000:.1f} mm")

    assert errors[135] > errors[116] > errors[105], (
        f"error should grow as the lid leans further back, got {errors}")


# --------------------------------------------------------------------------- #
# 2. axis-aligned no-op - the 90% case must not be perturbed
# --------------------------------------------------------------------------- #
def test_axis_aligned_pivot_does_not_trip_the_snap_gate():
    """
    On a closed, world-axis-aligned lid the AABB corner really does sit on the
    mesh. The gate must not fire, and `make_revolute_joint` must therefore
    return the caller's pivot bit-for-bit. This is what the entire existing
    example corpus looks like, so a "fix" that perturbs it is a regression.
    """
    child, _, corner = make_laptop(0)
    fired, offset, tolerance = snap_gate_would_fire(corner, child, HINGE_AXIS)
    assert not fired, (
        f"snap gate fired on an axis-aligned part: offset {offset * 1000:.3f} mm "
        f"exceeded tolerance {tolerance * 1000:.3f} mm")
    assert offset < 1e-6, f"corner should lie on the mesh, is {offset * 1000:.4f} mm off"


def test_snap_tolerance_floor_clears_pybullets_aabb_margin():
    """
    PyBullet reports AABBs inflated by roughly 3 mm. The absolute floor on the
    snap tolerance has to exceed that, or the gate fires on correct pivots.
    """
    assert PIVOT_SNAP_TOLERANCE_MIN > 0.003


# --------------------------------------------------------------------------- #
# 3. along-axis invariance - a revolute pivot is a line, not a point
# --------------------------------------------------------------------------- #
def test_displacing_a_pivot_along_the_axis_does_not_trip_the_gate():
    """
    Translating a pivot along the hinge axis yields an identical joint
    (docs section 6), so no amount of such displacement may trigger a repair.
    """
    child, _, corner = make_laptop(0)
    axis_hat = HINGE_AXIS / np.linalg.norm(HINGE_AXIS)
    _, baseline_offset, _ = snap_gate_would_fire(corner, child, HINGE_AXIS)

    for shift in (0.001, 0.05, 0.4, 5.0, -3.0):
        moved = corner + axis_hat * shift
        fired, offset, tolerance = snap_gate_would_fire(moved, child, HINGE_AXIS)
        assert not fired, (
            f"gate fired for a {shift} m displacement purely along the axis "
            f"(offset {offset * 1000:.3f} mm, tolerance {tolerance * 1000:.3f} mm)")
        assert abs(offset - baseline_offset) < 1e-9, (
            "perpendicular offset must be invariant to motion along the axis")


def test_solver_result_is_invariant_to_hint_displacement_along_the_axis():
    """
    Same property, one level deeper: moving the *hint* along the axis must not
    change the geometrically meaningful part of the solver's answer.
    """
    child, parent, hint = make_laptop(30)
    axis_hat = HINGE_AXIS / np.linalg.norm(HINGE_AXIS)

    reference = compute_hinge_pivot_from_points(
        child, parent, HINGE_AXIS, hint_point=hint)
    for shift in (0.05, 0.4, -0.4):
        moved = compute_hinge_pivot_from_points(
            child, parent, HINGE_AXIS, hint_point=hint + axis_hat * shift)
        assert perp_distance(moved, reference, HINGE_AXIS) < 1e-9, (
            f"perpendicular pivot changed when the hint moved {shift} m along the axis")


def test_perpendicular_basis_is_orthonormal():
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, -3], [-0.4, 0.0, 0.9]):
        axis_hat, e1, e2 = perpendicular_basis(axis)
        frame = np.stack([axis_hat, e1, e2])
        assert np.allclose(frame @ frame.T, np.eye(3), atol=1e-12)
        assert np.allclose(np.cross(axis_hat, e1), e2, atol=1e-12)


# --------------------------------------------------------------------------- #
# 4. degenerate inputs - fail loudly, or return something demonstrably on the part
# --------------------------------------------------------------------------- #
def _assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as exc:  # noqa: BLE001 - we are asserting on the type
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(exc).__name__}: {exc}")
    raise AssertionError(f"expected {exc_type.__name__}, but the call succeeded")


def test_zero_extent_child_raises():
    """A child with no spatial extent has no scale, so every tolerance is
    meaningless. It must fail loudly rather than divide by zero."""
    degenerate = np.zeros((50, 3))
    parent = np.random.default_rng(0).normal(size=(50, 3))
    _assert_raises(ValueError, compute_hinge_pivot_from_points,
                   degenerate, parent, HINGE_AXIS)


def test_zero_length_axis_raises():
    child, parent, _ = make_laptop(0)
    _assert_raises(ValueError, compute_hinge_pivot_from_points,
                   child, parent, [0.0, 0.0, 0.0])


def test_disjoint_parts_still_return_a_point_on_the_child():
    """
    Two parts that never touch have no hinge. The solver currently degrades to
    "the child's closest approach to the parent" rather than raising; what must
    never happen is a NaN, an infinity, or a point floating off the child.

    NOTE: docs section 9.4 item 4 asks for a *clean error* here. The solver does
    not raise today, so this test pins the weaker guarantee it does provide.
    """
    child, parent, hint = make_laptop(0)
    parent = parent + np.array([0.0, 0.0, 10.0])   # move the base 10 m away

    pivot = compute_hinge_pivot_from_points(
        child, parent, HINGE_AXIS, hint_point=hint)

    assert np.all(np.isfinite(pivot)), f"non-finite pivot {pivot}"
    _, offset, _ = snap_gate_would_fire(pivot, child, HINGE_AXIS)
    assert offset < 0.005, (
        f"pivot floated {offset * 1000:.1f} mm off the child instead of staying on it")


def test_sparse_contact_still_returns_a_point_on_the_child():
    """
    Fewer than 20 points in the contact band triggers the solver's quantile
    fallback. It must still land on the child rather than on noise.

    NOTE: as above, docs section 9.4 item 4 asks for an error; the solver
    degrades instead, so this pins the guarantee it actually offers.
    """
    rng = np.random.default_rng(7)
    # a thin child slab, and a parent that touches it at a single small patch
    child = np.column_stack([
        rng.uniform(-0.20, 0.20, 4000),
        rng.uniform(-0.15, 0.15, 4000),
        rng.uniform(0.0, 0.02, 4000),
    ])
    parent = np.column_stack([
        rng.uniform(-0.01, 0.01, 12),
        rng.uniform(-0.151, -0.149, 12),
        rng.uniform(-0.002, 0.0, 12),
    ])

    pivot = compute_hinge_pivot_from_points(child, parent, HINGE_AXIS)
    assert np.all(np.isfinite(pivot)), f"non-finite pivot {pivot}"
    _, offset, _ = snap_gate_would_fire(pivot, child, HINGE_AXIS)
    assert offset < 0.02, f"pivot landed {offset * 1000:.1f} mm off the child"


def test_empty_parent_is_a_known_gap():
    """
    An empty parent cloud means "no parent surface at all", for which there is no
    meaningful hinge. `cKDTree.query` returns infinite distances, the whole child
    ends up classified as the contact region, and an extreme point of the child
    comes back with no warning.

    This is unreachable through `Robot.get_hinge_pivot`, which raises when
    `_get_link_surface_points` returns None, so it is recorded rather than fixed.
    The test pins today's behaviour so a future change to it is a visible,
    deliberate one.
    """
    child, _, _ = make_laptop(0)
    pivot = compute_hinge_pivot_from_points(child, np.empty((0, 3)), HINGE_AXIS)
    assert np.all(np.isfinite(pivot)), (
        f"an empty parent produced a non-finite pivot {pivot}; if this now raises "
        "instead, update this test - raising would be an improvement")


# --------------------------------------------------------------------------- #
# standalone runner (the repo does not ship pytest as a dependency)
# --------------------------------------------------------------------------- #
def _main():
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}\n      {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
