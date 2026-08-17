"""Tests for the interactive articulation refinement package.

Covers the non-UI edit model (parse / FK / edits / validation / save
round-trip) against a synthetic results directory shaped exactly like the
publisher's output (empty base link, fixed base joint with rpy=(pi/2,0,pi/2),
movable joint whose origin xyz is the negative of the child's visual origins),
plus the physics-overrides merge helpers and a figure/app construction smoke
test (skipped when dash/plotly are absent).

Run with pytest:  pytest tests/test_refine_articulation.py
"""

import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simfoundry"))

from refine_articulation.physics_overrides import (  # noqa: E402
    load_physics_overrides,
    merge_parts_properties,
)
from refine_articulation.urdf_model import (  # noqa: E402
    ArticulationModel,
    make_transform,
)

BASE_RPY = (math.pi / 2, 0.0, math.pi / 2)
DOOR_VISUAL_XYZ = np.array([0.19338, 0.09793, -0.06656])

URDF_TEMPLATE = """<?xml version='1.0' encoding='utf-8'?>
<robot name="toaster_oven">
 <link name="base" />
 <link name="toaster_oven_base_link">
  <visual>
   <geometry>
    <mesh filename="meshes/toaster_oven_base.glb" />
   </geometry>
   <origin rpy="0 0 0" xyz="0 0 0" />
  </visual>
  <collision>
   <geometry>
    <mesh filename="meshes/toaster_oven_base.glb" />
   </geometry>
   <origin rpy="0 0 0" xyz="0 0 0" />
  </collision>
 </link>
 <joint type="fixed" name="base_to_toaster_oven_base_link">
  <parent link="base" />
  <child link="toaster_oven_base_link" />
  <origin rpy="{base_rpy}" xyz="0 0 0" />
 </joint>
 <link name="door_link">
  <visual>
   <geometry>
    <mesh filename="meshes/door.glb" />
   </geometry>
   <origin rpy="0 0 0" xyz="{door_xyz}" />
  </visual>
  <collision>
   <geometry>
    <mesh filename="meshes/door.glb" />
   </geometry>
   <origin rpy="0 0 0" xyz="{door_xyz}" />
  </collision>
 </link>
 <joint type="revolute" name="toaster_oven_base_link_to_door_link">
  <parent link="toaster_oven_base_link" />
  <child link="door_link" />
  <axis xyz="1.0 0.0 0.0" />
  <limit upper="1.5707963267948966" effort="5" lower="0.0" velocity="5" />
  <origin xyz="{joint_xyz}" />
 </joint>
</robot>"""

DOOR_JOINT = "toaster_oven_base_link_to_door_link"
BASE_JOINT = "base_to_toaster_oven_base_link"


@pytest.fixture
def results_dir(tmp_path):
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    trimesh.creation.box(extents=(0.4, 0.2, 0.3)).export(meshes / "toaster_oven_base.glb")
    trimesh.creation.box(extents=(0.35, 0.02, 0.25)).export(meshes / "door.glb")
    urdf = URDF_TEMPLATE.format(
        base_rpy=" ".join(str(v) for v in BASE_RPY),
        door_xyz=" ".join(str(v) for v in DOOR_VISUAL_XYZ),
        joint_xyz=" ".join(str(v) for v in -DOOR_VISUAL_XYZ),
    )
    (tmp_path / "mobility.urdf").write_text(urdf)
    return str(tmp_path)


def door_visual_world(model, q=0.0):
    link = model.links["door_link"]
    tf = model.link_world_transform("door_link", {DOOR_JOINT: q})
    return tf @ make_transform(link.geoms[0].xyz, link.geoms[0].rpy)


# ---------------------------------------------------------------------------
# Parsing / kinematics
# ---------------------------------------------------------------------------

def test_parse_structure(results_dir):
    model = ArticulationModel(results_dir)
    assert set(model.joints) == {BASE_JOINT, DOOR_JOINT}
    assert model.movable_joints() == [DOOR_JOINT]
    # The virtual base joint is not user-editable; the movable joint is.
    assert model.editable_joints() == [DOOR_JOINT]
    assert model.root_link == "base"
    door = model.joints[DOOR_JOINT]
    assert door.limit == {"lower": 0.0, "upper": pytest.approx(math.pi / 2),
                          "effort": 5.0, "velocity": 5.0}
    assert door.dynamics is None
    np.testing.assert_allclose(door.origin_xyz, -DOOR_VISUAL_XYZ)
    assert model.geometry_links() == ["toaster_oven_base_link", "door_link"]


def test_rest_pose_places_door_at_base_rotation(results_dir):
    """Joint origin and child visual origin cancel: at q=0 the door's visual
    frame is exactly the base rotation."""
    model = ArticulationModel(results_dir)
    expected = make_transform((0, 0, 0), BASE_RPY)
    np.testing.assert_allclose(door_visual_world(model), expected, atol=1e-12)


def test_joint_world_frame_uses_base_rotation(results_dir):
    model = ArticulationModel(results_dir)
    pivot, axis_world, _ = model.joint_world_frame(DOOR_JOINT)
    base_rot = make_transform((0, 0, 0), BASE_RPY)[:3, :3]
    np.testing.assert_allclose(axis_world, base_rot @ [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(pivot, base_rot @ -DOOR_VISUAL_XYZ, atol=1e-12)


def test_world_dir_round_trip(results_dir):
    model = ArticulationModel(results_dir)
    world_z = np.array([0.0, 0.0, 1.0])
    local = model.world_dir_to_parent_frame(DOOR_JOINT, world_z)
    parent_rot = model.link_world_transform("toaster_oven_base_link")[:3, :3]
    np.testing.assert_allclose(parent_rot @ local, world_z, atol=1e-12)


def test_world_point_round_trip(results_dir):
    model = ArticulationModel(results_dir)
    point_world = np.array([0.05, -0.02, 0.11])
    local = model.world_point_to_parent_frame(DOOR_JOINT, point_world)
    parent_tf = model.link_world_transform("toaster_oven_base_link")
    np.testing.assert_allclose(
        parent_tf[:3, :3] @ local + parent_tf[:3, 3], point_world, atol=1e-12)


# ---------------------------------------------------------------------------
# Edits
# ---------------------------------------------------------------------------

def test_set_origin_preserves_rest_pose(results_dir):
    model = ArticulationModel(results_dir)
    before = door_visual_world(model)
    model.set_origin(DOOR_JOINT, [0.02, -0.15, 0.08])
    np.testing.assert_allclose(door_visual_world(model), before, atol=1e-12)
    # Collision origin must be compensated identically to the visual origin.
    link = model.links["door_link"]
    np.testing.assert_allclose(link.geoms[0].xyz, link.geoms[1].xyz)
    # But the pose away from rest changes (the pivot actually moved).
    moved = model.link_world_transform("door_link", {DOOR_JOINT: 0.5})
    model2 = ArticulationModel(results_dir)
    orig = model2.link_world_transform("door_link", {DOOR_JOINT: 0.5})
    assert not np.allclose(moved, orig)


def test_set_origin_compensates_grandchild_joints(results_dir, tmp_path_factory):
    """Moving a pivot must not translate the grandchild subtree: joints
    parented to the edited joint's child are defined in the child frame and
    need the same compensation as the child's geometry."""
    chain_dir = tmp_path_factory.mktemp("chain")
    meshes = chain_dir / "meshes"
    meshes.mkdir()
    for part in ("body", "door", "handle"):
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(meshes / f"{part}.glb")

    def link(name, xyz):
        return f"""
 <link name="{name}_link">
  <visual>
   <geometry><mesh filename="meshes/{name}.glb" /></geometry>
   <origin rpy="0 0 0" xyz="{xyz[0]} {xyz[1]} {xyz[2]}" />
  </visual>
 </link>"""

    door_v, handle_v = np.array([0.2, 0.1, 0.0]), np.array([0.05, -0.3, 0.1])
    urdf = f"""<?xml version='1.0' encoding='utf-8'?>
<robot name="cabinet">
 <link name="base" />{link('body', (0, 0, 0))}{link('door', door_v)}{link('handle', handle_v)}
 <joint type="fixed" name="base_to_body_link">
  <parent link="base" /><child link="body_link" />
  <origin rpy="{BASE_RPY[0]} {BASE_RPY[1]} {BASE_RPY[2]}" xyz="0 0 0" />
 </joint>
 <joint type="revolute" name="body_link_to_door_link">
  <parent link="body_link" /><child link="door_link" />
  <axis xyz="0.0 1.0 0.0" />
  <limit lower="0.0" upper="1.2" effort="5" velocity="5" />
  <origin xyz="{-door_v[0]} {-door_v[1]} {-door_v[2]}" />
 </joint>
 <joint type="revolute" name="door_link_to_handle_link">
  <parent link="door_link" /><child link="handle_link" />
  <axis xyz="1.0 0.0 0.0" />
  <limit lower="0.0" upper="0.5" effort="5" velocity="5" />
  <origin xyz="{-handle_v[0]} {-handle_v[1]} {-handle_v[2]}" />
 </joint>
</robot>"""
    (chain_dir / "mobility.urdf").write_text(urdf)

    model = ArticulationModel(str(chain_dir))

    def visual_world(link_name):
        link_state = model.links[link_name]
        tf = model.link_world_transform(link_name)
        return tf @ make_transform(link_state.geoms[0].xyz, link_state.geoms[0].rpy)

    door_before = visual_world("door_link")
    handle_before = visual_world("handle_link")
    model.set_origin("body_link_to_door_link", [0.3, 0.0, 0.0])
    np.testing.assert_allclose(visual_world("door_link"), door_before, atol=1e-12)
    np.testing.assert_allclose(visual_world("handle_link"), handle_before, atol=1e-12)


def test_world_point_to_pivot_maps_back_through_joint_motion(results_dir):
    """Picking a point on the displaced child (q != 0) must resolve to that
    material feature's rest position, so the pivot lands where the user aimed."""
    model = ArticulationModel(results_dir)
    q = 1.0
    # A material feature: expressed in the child frame, then placed in world at q.
    feature_child = np.array([0.05, -0.02, 0.11, 1.0])
    child_tf = model.link_world_transform("door_link", {DOOR_JOINT: q})
    feature_world_at_q = (child_tf @ feature_child)[:3]
    picked = model.world_point_to_pivot(
        DOOR_JOINT, feature_world_at_q, {DOOR_JOINT: q}, clicked_link="door_link")
    # Expected: the feature's parent-frame position at rest (q=0).
    rest_tf = model.link_world_transform("door_link", {DOOR_JOINT: 0.0})
    parent_tf = model.link_world_transform("toaster_oven_base_link")
    expected = (np.linalg.inv(parent_tf) @ rest_tf @ feature_child)[:3]
    np.testing.assert_allclose(picked, expected, atol=1e-12)
    # Clicks outside the joint's subtree stay plain spatial points.
    plain = model.world_point_to_pivot(
        DOOR_JOINT, feature_world_at_q, {DOOR_JOINT: q}, clicked_link="toaster_oven_base_link")
    np.testing.assert_allclose(
        plain, model.world_point_to_parent_frame(DOOR_JOINT, feature_world_at_q), atol=1e-12)


def test_set_origin_without_compensation_moves_part(results_dir):
    model = ArticulationModel(results_dir)
    before = door_visual_world(model)
    model.set_origin(DOOR_JOINT, [0.0, 0.0, 0.0], compensate=False)
    assert not np.allclose(door_visual_world(model), before)


def test_set_axis_normalizes_and_rejects_zero(results_dir):
    model = ArticulationModel(results_dir)
    axis = model.set_axis(DOOR_JOINT, [0.0, 2.0, 0.0])
    np.testing.assert_allclose(axis, [0.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        model.set_axis(DOOR_JOINT, [0.0, 0.0, 0.0])


def test_set_limits_validation(results_dir):
    model = ArticulationModel(results_dir)
    model.set_limits(DOOR_JOINT, -0.3, 1.2, effort=10.0, velocity=2.0)
    assert model.joints[DOOR_JOINT].limit == {
        "lower": -0.3, "upper": 1.2, "effort": 10.0, "velocity": 2.0}
    with pytest.raises(ValueError):
        model.set_limits(DOOR_JOINT, 1.0, 1.0)
    with pytest.raises(ValueError):
        model.set_limits(DOOR_JOINT, 0.0, 1.0, effort=0.0)


def test_joint_type_transitions(results_dir):
    model = ArticulationModel(results_dir)
    model.set_joint_type(DOOR_JOINT, "continuous")
    # Continuous is limitless by URDF semantics: the stale revolute limit is
    # dropped immediately (not only at save), so the UI never renders it.
    assert model.joints[DOOR_JOINT].limit is None
    with pytest.raises(ValueError):
        model.set_limits(DOOR_JOINT, 0.0, 1.0)
    model.set_joint_type(DOOR_JOINT, "prismatic")
    assert model.joints[DOOR_JOINT].limit is not None  # fresh defaults
    with pytest.raises(ValueError):
        model.set_joint_type(BASE_JOINT, "revolute")


def test_parse_drops_limit_on_continuous(results_dir):
    urdf_path = os.path.join(results_dir, "mobility.urdf")
    text = open(urdf_path).read().replace('type="revolute"', 'type="continuous"')
    open(urdf_path, "w").write(text)
    model = ArticulationModel(results_dir)
    assert model.joints[DOOR_JOINT].joint_type == "continuous"
    assert model.joints[DOOR_JOINT].limit is None


def test_validate_catches_broken_joints(results_dir):
    model = ArticulationModel(results_dir)
    model.joints[DOOR_JOINT].limit = None
    errors = model.validate()
    assert any("requires a <limit>" in e for e in errors)
    with pytest.raises(ValueError):
        model.save()


# ---------------------------------------------------------------------------
# Save round-trip / persistence
# ---------------------------------------------------------------------------

def test_save_roundtrip_and_versioning(results_dir):
    model = ArticulationModel(results_dir)
    model.set_limits(DOOR_JOINT, 0.1, 1.1)
    model.set_axis(DOOR_JOINT, [0.0, 0.0, -1.0])
    model.set_origin(DOOR_JOINT, [0.01, 0.02, 0.03])
    model.set_joint_dynamics(DOOR_JOINT, damping=0.25, friction=0.01)
    summary = model.save()

    assert summary["changed_joints"] == [DOOR_JOINT]
    assert summary["version"] == 1
    assert os.path.exists(os.path.join(results_dir, "mobility_original.urdf"))
    assert os.path.exists(os.path.join(results_dir, "mobility_refined_1.urdf"))
    assert not model.dirty

    reloaded = ArticulationModel(results_dir)
    door = reloaded.joints[DOOR_JOINT]
    assert door.limit["lower"] == pytest.approx(0.1)
    assert door.limit["upper"] == pytest.approx(1.1)
    np.testing.assert_allclose(door.axis, [0.0, 0.0, -1.0])
    np.testing.assert_allclose(door.origin_xyz, [0.01, 0.02, 0.03])
    assert door.dynamics == {"damping": 0.25, "friction": 0.01}
    # Rest pose still intact after the origin edit round-trips through XML.
    expected = make_transform((0, 0, 0), BASE_RPY)
    np.testing.assert_allclose(door_visual_world(reloaded), expected, atol=1e-9)

    # Second save gets the next version and does not clobber the backup.
    reloaded.set_limits(DOOR_JOINT, 0.0, 0.9)
    summary2 = reloaded.save()
    assert summary2["version"] == 2
    backup = ET.parse(os.path.join(results_dir, "mobility_original.urdf"))
    limit = backup.getroot().find(f"joint[@name='{DOOR_JOINT}']/limit")
    assert float(limit.attrib["upper"]) == pytest.approx(math.pi / 2)

    log = json.load(open(os.path.join(results_dir, "refinement_log.json")))
    assert [entry["version"] for entry in log] == [1, 2]


def test_continuous_joint_serializes_without_limit(results_dir):
    model = ArticulationModel(results_dir)
    model.set_joint_type(DOOR_JOINT, "continuous")
    model.save()
    root = ET.parse(os.path.join(results_dir, "mobility.urdf")).getroot()
    joint_el = root.find(f"joint[@name='{DOOR_JOINT}']")
    assert joint_el.attrib["type"] == "continuous"
    assert joint_el.find("limit") is None


def test_overrides_persistence(results_dir):
    model = ArticulationModel(results_dir)
    model.set_joint_dynamics(DOOR_JOINT, damping=0.4, friction=None)
    model.set_part_properties("door_link", mass_kg=1.7, friction=0.6, joint_damping=0.4)
    model.save()

    overrides = load_physics_overrides(results_dir)
    assert overrides["joints"][DOOR_JOINT] == {"damping": 0.4}
    assert overrides["parts"]["door_link"] == {
        "mass_kg": 1.7, "friction": 0.6, "joint_damping": 0.4}

    # A fresh model picks the overrides back up.
    reloaded = ArticulationModel(results_dir)
    assert reloaded.overrides["parts"]["door_link"]["mass_kg"] == 1.7


def test_load_physics_overrides_missing_and_corrupt(results_dir):
    assert load_physics_overrides(results_dir) == {"version": 1, "parts": {}, "joints": {}}
    path = os.path.join(results_dir, "physics_overrides.json")
    with open(path, "w") as f:
        f.write("{not json")
    assert load_physics_overrides(results_dir)["parts"] == {}
    # Well-formed JSON with unusable values must not raise either (the model
    # constructor calls this; a bad file must not brick the whole UI).
    with open(path, "w") as f:
        json.dump({"parts": {"door_link": {"mass_kg": "1.5kg"},
                             "shelf_link": {"friction": 0.9}},
                   "joints": "nope"}, f)
    overrides = load_physics_overrides(results_dir)
    assert overrides["parts"] == {"shelf_link": {"friction": 0.9}}
    assert overrides["joints"] == {}
    with open(path, "w") as f:
        json.dump(["not", "a", "dict"], f)
    assert load_physics_overrides(results_dir) == {"version": 1, "parts": {}, "joints": {}}


def test_set_part_properties_validates_before_writing(results_dir):
    model = ArticulationModel(results_dir)
    with pytest.raises(ValueError):
        model.set_part_properties("door_link", mass_kg=2.0, friction=-1.0)
    # The rejected edit must not have written the valid half.
    assert "door_link" not in model.overrides["parts"]


def test_merge_parts_properties():
    vlm_parts = [
        {"name": "door_link", "mass_kg": 1.0, "friction": 0.5, "joint_damping": 0.5},
        {"name": "base_link", "mass_kg": 4.0, "friction": 0.5, "joint_damping": 0.5},
    ]
    overrides = {"parts": {
        "door_link": {"mass_kg": 2.5},
        "shelf_link": {"friction": 0.9},
    }}
    merged = merge_parts_properties(vlm_parts, overrides)
    by_name = {p["name"]: p for p in merged}
    assert by_name["door_link"]["mass_kg"] == 2.5
    assert by_name["door_link"]["friction"] == 0.5  # untouched VLM value
    assert by_name["base_link"] == vlm_parts[1]
    assert by_name["shelf_link"] == {"name": "shelf_link", "friction": 0.9}
    # Inputs are not mutated.
    assert vlm_parts[0]["mass_kg"] == 1.0


# ---------------------------------------------------------------------------
# Undo / reset
# ---------------------------------------------------------------------------

def test_snapshot_restore_and_reset(results_dir):
    model = ArticulationModel(results_dir)
    snap = model.snapshot()
    model.set_axis(DOOR_JOINT, [0.0, 1.0, 0.0])
    model.set_origin(DOOR_JOINT, [0.0, 0.0, 0.0])
    assert model.dirty
    model.restore(snap)
    np.testing.assert_allclose(model.joints[DOOR_JOINT].axis, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(model.joints[DOOR_JOINT].origin_xyz, -DOOR_VISUAL_XYZ)
    # Undoing back to the pristine loaded state clears the dirty flag.
    assert not model.dirty

    model.set_limits(DOOR_JOINT, -1.0, 1.0)
    model.reset_joint(DOOR_JOINT)
    assert model.joints[DOOR_JOINT].limit["lower"] == 0.0

    model.set_axis(DOOR_JOINT, [0.0, 1.0, 0.0])
    model.reset_all()
    np.testing.assert_allclose(model.joints[DOOR_JOINT].axis, [1.0, 0.0, 0.0])
    assert model.changed_joints() == []


# ---------------------------------------------------------------------------
# UI construction smoke tests (no server)
# ---------------------------------------------------------------------------

def test_figure_builder_smoke(results_dir):
    pytest.importorskip("plotly")
    from refine_articulation.visualization import ArticulationFigureBuilder

    model = ArticulationModel(results_dir)
    builder = ArticulationFigureBuilder(model)
    fig = builder.build({DOOR_JOINT: 0.3}, selected_joint=DOOR_JOINT,
                        show_ghosts=True, color_mode="segmented")
    names = [t.name for t in fig.data if t.name]
    assert any(n.startswith("door_link") for n in names)
    assert "axis" in names and "pivot" in names and "range" in names
    # Ghost meshes at both limits.
    assert sum(1 for n in names if n.startswith("door_link@")) == 2
    assert fig.layout.uirevision == "constant"


def test_app_constructs(results_dir):
    pytest.importorskip("dash")
    pytest.importorskip("plotly")
    from refine_articulation.interactive_ui import ArticulationRefinementApp

    model = ArticulationModel(results_dir)
    app = ArticulationRefinementApp({"toaster_oven": model})
    assert app.result_holder == {'result': None, 'done': False}
    assert app._default_joint("toaster_oven") == DOOR_JOINT


def test_app_part_edits(results_dir):
    pytest.importorskip("dash")
    pytest.importorskip("plotly")
    from refine_articulation.interactive_ui import ArticulationRefinementApp

    model = ArticulationModel(results_dir)
    app = ArticulationRefinementApp({"toaster_oven": model})
    links = model.geometry_links()
    app._apply_part_edits("toaster_oven", links,
                          masses=[3.0, None], frictions=[None, 0.8], dampings=[None, None])
    assert model.overrides["parts"]["toaster_oven_base_link"] == {"mass_kg": 3.0}
    assert model.overrides["parts"]["door_link"] == {"friction": 0.8}
    # Clearing an input removes the override entirely.
    app._apply_part_edits("toaster_oven", links,
                          masses=[None, None], frictions=[None, None], dampings=[None, None])
    assert model.overrides["parts"] == {}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
