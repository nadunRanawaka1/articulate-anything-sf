"""Tests for the physics-estimation step (simfoundry/estimate_physics.py).

All VLM calls are faked via the query_fn hook; fixtures mirror the publisher's
results layout (mobility.urdf + meshes/*.glb with the base-joint/pivot
conventions).

Run with pytest:  pytest tests/test_estimate_physics.py
"""

import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simfoundry"))

from estimate_physics import (  # noqa: E402
    DEFAULT_JOINT_DAMPING,
    DEFAULT_MASS_KG,
    DEFAULT_PRISMATIC_JOINT_FRICTION,
    DEFAULT_REVOLUTE_JOINT_FRICTION,
    DEFAULT_SURFACE_FRICTION,
    apply_user_values,
    collect_urdf_physics_targets,
    ensure_physics_current,
    estimate_physics_properties,
    load_user_values,
    sanitize_physics_result,
)

BASE_RPY = (math.pi / 2, 0.0, math.pi / 2)
DOOR_VISUAL_XYZ = np.array([0.19338, 0.09793, -0.06656])
DOOR_JOINT = "toaster_oven_base_link_to_door_link"

URDF_TEMPLATE = """<?xml version='1.0' encoding='utf-8'?>
<robot name="toaster_oven">
 <link name="base" />
 <link name="toaster_oven_base_link">
  <visual>
   <geometry><mesh filename="meshes/toaster_oven_base.glb" /></geometry>
   <origin rpy="0 0 0" xyz="0 0 0" />
  </visual>
 </link>
 <joint type="fixed" name="base_to_toaster_oven_base_link">
  <parent link="base" /><child link="toaster_oven_base_link" />
  <origin rpy="{base_rpy}" xyz="0 0 0" />
 </joint>
 <link name="door_link">
  <visual>
   <geometry><mesh filename="meshes/door.glb" /></geometry>
   <origin rpy="0 0 0" xyz="{door_xyz}" />
  </visual>
 </link>
 <joint type="{joint_type}" name="{door_joint}">
  <parent link="toaster_oven_base_link" /><child link="door_link" />
  <axis xyz="1.0 0.0 0.0" />
  {limit}
  <origin xyz="{joint_xyz}" />
 </joint>
</robot>"""


def make_results_dir(tmp_path, joint_type="revolute"):
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    trimesh.creation.box(extents=(0.4, 0.2, 0.3)).export(meshes / "toaster_oven_base.glb")
    trimesh.creation.box(extents=(0.35, 0.02, 0.25)).export(meshes / "door.glb")
    limit = '<limit lower="0.0" upper="1.5707963267948966" effort="5" velocity="5" />'
    if joint_type in ("continuous", "fixed"):
        limit = ""
    urdf = URDF_TEMPLATE.format(
        base_rpy=" ".join(str(v) for v in BASE_RPY),
        door_xyz=" ".join(str(v) for v in DOOR_VISUAL_XYZ),
        joint_xyz=" ".join(str(v) for v in -DOOR_VISUAL_XYZ),
        joint_type=joint_type,
        door_joint=DOOR_JOINT,
        limit=limit,
    )
    (tmp_path / "mobility.urdf").write_text(urdf)
    return str(tmp_path)


VLM_RESPONSE = """Here is my reasoning: the body is steel, the door is glass+steel.
```json
{
  "parts": [
    {"name": "toaster_oven_base_link", "mass_kg": 4.5, "friction": 0.6},
    {"name": "door_link", "mass_kg": 1.2, "friction": 0.4},
    {"name": "hallucinated_link", "mass_kg": 99.0, "friction": 9.0}
  ],
  "joints": [
    {"name": "toaster_oven_base_link_to_door_link", "damping": 0.15, "friction": 0.04},
    {"name": "made_up_joint", "damping": 5.0, "friction": 5.0}
  ],
  "reasoning": "steel body, hinged glass door"
}
```"""

CFG = OmegaConf.create({"model_name": "fake-model"})


def read_urdf_dynamics(results_dir):
    root = ET.parse(os.path.join(results_dir, "mobility.urdf")).getroot()
    dyn = root.find(f"joint[@name='{DOOR_JOINT}']/dynamics")
    return None if dyn is None else {k: float(v) for k, v in dyn.attrib.items()}


def test_collect_targets(tmp_path):
    results_dir = make_results_dir(tmp_path)
    parts, joints = collect_urdf_physics_targets(results_dir, scale=2.0)
    assert [p["name"] for p in parts] == ["toaster_oven_base_link", "door_link"]
    # Bounding boxes are in cm at the requested scale (0.4 m * 2.0 -> 80 cm).
    assert max(parts[0]["bounding_box_cm"]) == pytest.approx(80.0, rel=0.05)
    assert joints == {DOOR_JOINT: {"type": "revolute", "child": "door_link",
                                   "range": pytest.approx(math.pi / 2)}}


def test_estimate_full_flow(tmp_path):
    results_dir = make_results_dir(tmp_path)
    calls = []

    def fake_query(user_prompt, system_prompt, image_paths):
        calls.append(user_prompt)
        return VLM_RESPONSE

    payload = estimate_physics_properties(
        CFG, results_dir, "toaster oven", image_path=None,
        scale=1.5, query_fn=fake_query)

    # Prompt includes parts, the joint, and no approximate-scale caveat.
    assert "toaster_oven_base_link" in calls[0]
    assert DOOR_JOINT in calls[0]
    assert "approximate" not in calls[0]

    by_name = {p["name"]: p for p in payload["parts"]}
    assert set(by_name) == {"toaster_oven_base_link", "door_link"}  # hallucination dropped
    assert by_name["door_link"]["mass_kg"] == 1.2
    assert by_name["door_link"]["joint_damping"] == 0.15  # synced from the joint
    assert "joint_damping" not in by_name["toaster_oven_base_link"]
    assert payload["joints"] == {DOOR_JOINT: {"damping": 0.15, "friction": 0.04}}
    assert payload["source"] == "vlm"
    assert payload["scale"] == 1.5

    # Persisted artifacts: json, raw response, and URDF <dynamics>.
    on_disk = json.load(open(os.path.join(results_dir, "physics_properties.json")))
    assert on_disk["joints"][DOOR_JOINT]["damping"] == 0.15
    assert os.path.exists(os.path.join(results_dir, "physics_vlm_response.txt"))
    assert read_urdf_dynamics(results_dir) == {"damping": 0.15, "friction": 0.04}


def test_unparseable_response_falls_back_to_defaults(tmp_path):
    results_dir = make_results_dir(tmp_path)
    payload = estimate_physics_properties(
        CFG, results_dir, "toaster oven",
        query_fn=lambda *a: "sorry, I cannot help with that")
    by_name = {p["name"]: p for p in payload["parts"]}
    assert by_name["door_link"]["mass_kg"] == DEFAULT_MASS_KG
    assert by_name["door_link"]["friction"] == DEFAULT_SURFACE_FRICTION
    assert payload["joints"][DOOR_JOINT] == {
        "damping": DEFAULT_JOINT_DAMPING, "friction": DEFAULT_REVOLUTE_JOINT_FRICTION}
    assert payload["scale"] is None
    assert read_urdf_dynamics(results_dir) is not None


def test_sanitize_rejects_bad_values():
    parts_info = [{"name": "a", "bounding_box_cm": None, "volume_cm": None}]
    joints_info = {"j": {"type": "prismatic", "child": "a", "range": 0.1}}
    result = sanitize_physics_result(
        {"parts": [{"name": "a", "mass_kg": -5, "friction": "sticky"}],
         "joints": [{"name": "j", "damping": float("nan"), "friction": None}]},
        parts_info, joints_info)
    assert result["parts"][0]["mass_kg"] == DEFAULT_MASS_KG
    assert result["parts"][0]["friction"] == DEFAULT_SURFACE_FRICTION
    assert result["joints"]["j"] == {
        "damping": DEFAULT_JOINT_DAMPING, "friction": DEFAULT_PRISMATIC_JOINT_FRICTION}
    assert result["parts"][0]["joint_damping"] == DEFAULT_JOINT_DAMPING


def test_offline_user_values(tmp_path):
    """use_vlm=False: no query, defaults + user values only; user scale wins."""
    results_dir = make_results_dir(tmp_path)

    def must_not_call(*a):
        raise AssertionError("VLM must not be queried with use_vlm=False")

    user_entry = {
        "scale": 2.0,
        "parts": {"door_link": {"mass_kg": 3.3}},
        "joints": {DOOR_JOINT: {"damping": 0.9, "friction": 0.2}},
    }
    payload = estimate_physics_properties(
        CFG, results_dir, "toaster oven", scale=1.0,
        query_fn=must_not_call, use_vlm=False, user_values=user_entry)

    assert payload["source"] == "defaults+user"
    assert payload["scale"] == 2.0  # user scale overrides the passed one
    assert payload["model_name"] is None
    by_name = {p["name"]: p for p in payload["parts"]}
    assert by_name["door_link"]["mass_kg"] == 3.3
    assert by_name["door_link"]["friction"] == DEFAULT_SURFACE_FRICTION
    assert by_name["door_link"]["joint_damping"] == 0.9  # synced from user joint value
    assert payload["joints"][DOOR_JOINT] == {"damping": 0.9, "friction": 0.2}
    assert read_urdf_dynamics(results_dir) == {"damping": 0.9, "friction": 0.2}
    assert not os.path.exists(os.path.join(results_dir, "physics_vlm_response.txt"))


def test_user_values_overlay_vlm(tmp_path):
    results_dir = make_results_dir(tmp_path)
    payload = estimate_physics_properties(
        CFG, results_dir, "toaster oven",
        query_fn=lambda *a: VLM_RESPONSE,
        user_values={"parts": {"door_link": {"friction": 0.75}}})
    assert payload["source"] == "vlm+user"
    by_name = {p["name"]: p for p in payload["parts"]}
    assert by_name["door_link"]["friction"] == 0.75  # user wins
    assert by_name["door_link"]["mass_kg"] == 1.2    # VLM value kept


def test_rigid_object_without_joints(tmp_path):
    results_dir = make_results_dir(tmp_path, joint_type="fixed")
    prompts = []
    payload = estimate_physics_properties(
        CFG, results_dir, "toaster oven",
        query_fn=lambda up, sp, ip: (prompts.append(up), VLM_RESPONSE)[1])
    assert payload["joints"] == {}
    assert "movable joints" not in prompts[0]
    assert all("joint_damping" not in p for p in payload["parts"])


def test_prismatic_travel_scaled_in_prompt(tmp_path):
    """Prismatic travel must be reported at the same real-world scale as the
    bounding boxes, or the prompt is dimensionally inconsistent."""
    results_dir = make_results_dir(tmp_path, joint_type="prismatic")
    _, joints = collect_urdf_physics_targets(results_dir, scale=2.0)
    assert joints[DOOR_JOINT]["range"] == pytest.approx(math.pi / 2 * 2.0)


def test_revolute_range_not_scaled(tmp_path):
    results_dir = make_results_dir(tmp_path)
    _, joints = collect_urdf_physics_targets(results_dir, scale=2.0)
    assert joints[DOOR_JOINT]["range"] == pytest.approx(math.pi / 2)


def test_ensure_physics_current(tmp_path):
    results_dir = make_results_dir(tmp_path)
    # No physics file yet.
    assert not ensure_physics_current(results_dir)
    estimate_physics_properties(CFG, results_dir, "toaster oven",
                                query_fn=lambda *a: VLM_RESPONSE)
    assert ensure_physics_current(results_dir)

    # Simulate a step-5 republish that strips <dynamics>: they are re-applied
    # from the existing json without a VLM call.
    urdf_path = os.path.join(results_dir, "mobility.urdf")
    tree = ET.parse(urdf_path)
    joint = tree.getroot().find(f"joint[@name='{DOOR_JOINT}']")
    joint.remove(joint.find("dynamics"))
    tree.write(urdf_path, xml_declaration=True, encoding="utf-8")
    assert read_urdf_dynamics(results_dir) is None
    assert ensure_physics_current(results_dir)
    assert read_urdf_dynamics(results_dir) == {"damping": 0.15, "friction": 0.04}

    # A joint-inventory mismatch (re-articulated object) invalidates the file.
    text = open(urdf_path).read().replace(DOOR_JOINT, "renamed_joint")
    open(urdf_path, "w").write(text)
    assert not ensure_physics_current(results_dir)


def test_ensure_physics_current_rejects_corrupt_json(tmp_path):
    results_dir = make_results_dir(tmp_path)
    with open(os.path.join(results_dir, "physics_properties.json"), "w") as f:
        json.dump({"joints": {DOOR_JOINT: {"damping": "soft"}}}, f)
    assert not ensure_physics_current(results_dir)


def test_load_user_values(tmp_path):
    assert load_user_values(None) == {}
    assert load_user_values(str(tmp_path / "missing.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    assert load_user_values(str(bad)) == {}
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"obj": {"scale": 1.5}}))
    assert load_user_values(str(good)) == {"obj": {"scale": 1.5}}


def test_apply_user_values_ignores_bad_entries():
    joints_info = {"j": {"type": "revolute", "child": "a", "range": 1.0}}
    physics = {"parts": [{"name": "a", "mass_kg": 1.0, "friction": 0.5}],
               "joints": {"j": {"damping": 0.5, "friction": 0.01}}}
    out = apply_user_values(
        physics,
        {"parts": {"a": {"mass_kg": -3}}, "joints": {"j": "not-a-dict"}},
        joints_info)
    assert out["parts"][0]["mass_kg"] == 1.0  # negative rejected, original kept
    assert out["joints"]["j"]["damping"] == 0.5


def test_refinement_model_sees_estimates(tmp_path):
    """The refinement UI shows pipeline estimates as baselines and parses the
    step's URDF dynamics."""
    pytest.importorskip("plotly")
    results_dir = make_results_dir(tmp_path)
    estimate_physics_properties(CFG, results_dir, "toaster oven",
                                query_fn=lambda *a: VLM_RESPONSE)

    from refine_articulation.urdf_model import ArticulationModel

    model = ArticulationModel(results_dir)
    assert model.joints[DOOR_JOINT].dynamics == {"damping": 0.15, "friction": 0.04}
    assert model.estimates["parts"]["door_link"]["mass_kg"] == 1.2
    # Estimates are baselines, not user overrides.
    assert model.overrides == {"version": 1, "parts": {}, "joints": {}}
    assert not model.dirty


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
