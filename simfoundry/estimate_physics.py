"""
Step 5b: estimate physical simulation properties for a published articulation
result with a single VLM call per object.

The articulation pipeline is the sole source of dynamics for downstream
consumers: this step estimates per-link mass (kg) and surface friction, and
per-movable-joint damping and friction, then persists them twice:

  - <dynamics damping friction> elements written into results/mobility.urdf
    (the values simulators read directly);
  - results/physics_properties.json, carrying everything a sim-ready importer
    needs — including values that cannot live in the URDF because importers
    rebuild those elements (mass -> <inertial> is recomputed from scaled mesh
    geometry downstream; surface friction has no URDF home at all). Its
    "parts" list uses the same schema the SimFoundry sim-ready stage used for
    its own estimates ({name, mass_kg, friction, joint_damping}), so it can be
    consumed as parts_properties verbatim.

User edits from the refinement UI live in results/physics_overrides.json and
take precedence over this file downstream.

Standalone use (no SimFoundry pipeline) is fully supported: real-world scale
is optional (without it the prompt tells the VLM to trust the picture over the
listed dimensions), and known values can be supplied directly via
``s5b_estimate_physics.user_values_path`` — a JSON file keyed by object name:

    {
        "my_cabinet": {
            "scale": 1.2,
            "parts":  {"door_link":  {"mass_kg": 2.0, "friction": 0.6}},
            "joints": {"body_link_to_door_link": {"damping": 0.3, "friction": 0.05}}
        }
    }

User values overlay the VLM estimates; with ``use_vlm: false`` the step runs
offline from defaults + user values only.
"""

import json
import logging
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

PHYSICS_FILENAME = "physics_properties.json"
RAW_RESPONSE_FILENAME = "physics_vlm_response.txt"

DEFAULT_MASS_KG = 1.0
DEFAULT_SURFACE_FRICTION = 0.5
DEFAULT_JOINT_DAMPING = 0.5
# Fallbacks matching the constants the SimFoundry sim-ready stage used to
# hard-code (asset_conversion_utils REVOLUTE_JOINT_FRIC / PRISMATIC_JOINT_FRIC).
DEFAULT_REVOLUTE_JOINT_FRICTION = 0.01
DEFAULT_PRISMATIC_JOINT_FRICTION = 0.4

MOVABLE_TYPES = ("revolute", "prismatic", "continuous")


def _pos_float(value, default: float) -> float:
    """Coerce a VLM-provided number; non-numeric/negative/non-finite -> default."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out) or out < 0:
        return default
    return out


def collect_urdf_physics_targets(results_dir: str, scale: float = 1.0):
    """Parse results/mobility.urdf into the link/joint inventory to estimate.

    Returns (parts_info, joints_info):
      parts_info: [{"name", "bounding_box_cm", "volume_cm"}] for every mesh-
        bearing link (the virtual "base" link has no geometry and is skipped);
      joints_info: {joint_name: {"type", "child", "range"}} for movable joints
        (range is upper-lower in rad or m, None for continuous).
    """
    urdf_path = os.path.join(results_dir, "mobility.urdf")
    root = ET.parse(urdf_path).getroot()

    parts_info = []
    for link in root.findall("link"):
        name = link.attrib["name"]
        mesh_el = link.find("visual/geometry/mesh")
        if mesh_el is None or not mesh_el.attrib.get("filename"):
            continue
        filename = mesh_el.attrib["filename"]
        mesh_path = os.path.join(results_dir, filename)
        if not os.path.exists(mesh_path):
            mesh_path = os.path.join(results_dir, "meshes", os.path.basename(filename))
        entry = {"name": name, "bounding_box_cm": None, "volume_cm": None}
        try:
            tm = trimesh.load(mesh_path)
            if isinstance(tm, trimesh.Scene):
                tm = tm.to_geometry()
            tm.apply_scale(scale)
            _, obb_extent = trimesh.bounds.oriented_bounds(tm)
            entry["bounding_box_cm"] = [float(v) * 100 for v in obb_extent]
            volume = float(tm.volume) if tm.is_volume else float(np.prod(obb_extent)) * 0.65
            entry["volume_cm"] = volume * (100 ** 3)
        except Exception as exc:
            logger.warning("Could not measure mesh for link %s (%s): %s", name, mesh_path, exc)
        parts_info.append(entry)

    joints_info = {}
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type", "fixed")
        if joint_type not in MOVABLE_TYPES:
            continue
        name = joint.attrib["name"]
        child = joint.find("child").attrib["link"]
        limit_el = joint.find("limit")
        joint_range = None
        if limit_el is not None and joint_type != "continuous":
            try:
                joint_range = float(limit_el.attrib.get("upper", 0.0)) - float(limit_el.attrib.get("lower", 0.0))
            except (TypeError, ValueError):
                joint_range = None
        # Prismatic limits are in mesh units; report real-world travel so the
        # prompt is dimensionally consistent with the scaled bounding boxes.
        # Revolute/continuous ranges are radians and scale-free.
        if joint_range is not None and joint_type == "prismatic":
            joint_range *= scale
        joints_info[name] = {"type": joint_type, "child": child, "range": joint_range}
    return parts_info, joints_info


def physics_prompt(object_name: str, parts_info: list, joints_info: dict,
                   scale_is_real: bool) -> tuple:
    """(user_prompt, system_prompt) asking for per-part mass/surface friction
    and per-joint damping/friction, mirroring the estimates the SimFoundry
    sim-ready stage used to make itself."""
    part_lines = []
    for p in parts_info:
        if p["bounding_box_cm"] is not None:
            bb = p["bounding_box_cm"]
            part_lines.append(
                f"  - {p['name']}: bounding box {bb[0]:.2f}cm x {bb[1]:.2f}cm x {bb[2]:.2f}cm, "
                f"volume {p['volume_cm']:.2f}cm³")
        else:
            part_lines.append(f"  - {p['name']}")
    joint_lines = []
    for name, info in joints_info.items():
        unit = "m" if info["type"] == "prismatic" else "rad"
        range_desc = f", travel {info['range']:.3f} {unit}" if info["range"] is not None else ""
        joint_lines.append(f"  - {name}: {info['type']} joint moving part '{info['child']}'{range_desc}")

    scale_note = "" if scale_is_real else (
        "\nNote: the listed dimensions are approximate and may not be at real-world "
        "scale; rely primarily on the object type and the picture for size intuition.")
    joints_section = ""
    if joint_lines:
        joints_section = f"""
It has the following movable joints:
{chr(10).join(joint_lines)}

For each joint, estimate:
3. damping - viscous joint damping (N·m·s/rad for revolute/continuous, N·s/m for prismatic)
4. friction - dry joint friction (N·m for revolute/continuous, N for prismatic); e.g. a
   free-swinging hinge ~0.01-0.1, a sliding drawer ~0.2-1.0
"""

    user_prompt = f"""Shown is a picture of a {object_name}, an articulated object with the following parts:
{chr(10).join(part_lines)}
{scale_note}
For each part, estimate:
1. mass_kg - mass in kilograms, based on the typical material for that part type
2. friction - surface friction coefficient (unitless)
{joints_section}
Consider that different parts likely have different materials:
- A cabinet body is typically wood/particleboard
- Doors/drawers share the body material but may have different handles (metal)
- Handles and knobs are often metal or plastic

Use chain-of-thought reasoning, then return ONLY a JSON object of this form:
{{
  "parts": [
    {{"name": "part_name", "mass_kg": 0.0, "friction": 0.0}},
    ...
  ],
  "joints": [
    {{"name": "joint_name", "damping": 0.0, "friction": 0.0}},
    ...
  ],
  "reasoning": "your reasoning here"
}}
Use the exact part and joint names listed above.
"""
    system_prompt = ("You estimate physical simulation parameters (mass, friction, "
                     "joint damping) for reconstructed articulated objects.")
    return user_prompt, system_prompt


def sanitize_physics_result(result_json: dict, parts_info: list, joints_info: dict) -> dict:
    """Normalize the VLM response onto the URDF's actual link/joint inventory.

    Every listed link/joint gets an entry (defaults fill gaps); names the VLM
    invented are dropped. Parts additionally carry joint_damping (the damping
    of the joint whose child they are) for sim-ready parts_properties schema
    compatibility.
    """
    if not isinstance(result_json, dict):
        result_json = {}
    parts_by_name = {p.get("name"): p for p in result_json.get("parts") or [] if isinstance(p, dict)}
    joints_by_name = {j.get("name"): j for j in result_json.get("joints") or [] if isinstance(j, dict)}

    joints_out = {}
    for name, info in joints_info.items():
        entry = joints_by_name.get(name, {})
        default_friction = (DEFAULT_PRISMATIC_JOINT_FRICTION if info["type"] == "prismatic"
                            else DEFAULT_REVOLUTE_JOINT_FRICTION)
        joints_out[name] = {
            "damping": _pos_float(entry.get("damping"), DEFAULT_JOINT_DAMPING),
            "friction": _pos_float(entry.get("friction"), default_friction),
        }

    parts_out = []
    for part in parts_info:
        name = part["name"]
        entry = parts_by_name.get(name, {})
        parts_out.append({
            "name": name,
            "mass_kg": _pos_float(entry.get("mass_kg"), DEFAULT_MASS_KG),
            "friction": _pos_float(entry.get("friction"), DEFAULT_SURFACE_FRICTION),
        })

    physics = {"parts": parts_out, "joints": joints_out,
               "reasoning": str(result_json.get("reasoning", ""))}
    _sync_part_joint_damping(physics, joints_info)
    return physics


def _sync_part_joint_damping(physics: dict, joints_info: dict):
    """Copy each joint's damping onto its child part's joint_damping (the
    sim-ready parts_properties schema keys damping by child link)."""
    child_damping = {info["child"]: physics["joints"][name]["damping"]
                     for name, info in joints_info.items() if name in physics["joints"]}
    for part in physics["parts"]:
        if part["name"] in child_damping:
            part["joint_damping"] = child_damping[part["name"]]


def load_user_values(path: str | None) -> dict:
    """Load the optional user-supplied physics JSON ({object_name: entry});
    returns {} when absent/unreadable so standalone runs degrade gracefully."""
    if not path:
        return {}
    if not os.path.exists(path):
        logger.warning("user_values_path %s does not exist; ignoring", path)
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Ignoring unreadable user physics file %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def apply_user_values(physics: dict, user_entry: dict, joints_info: dict) -> dict:
    """Overlay user-known values (authoritative) onto the estimates."""
    if not isinstance(user_entry, dict):
        return physics
    user_parts = user_entry.get("parts") or {}
    for part in physics["parts"]:
        override = user_parts.get(part["name"])
        if isinstance(override, dict):
            for key in ("mass_kg", "friction"):
                if override.get(key) is not None:
                    part[key] = _pos_float(override[key], part[key])
    user_joints = user_entry.get("joints") or {}
    for name, values in physics["joints"].items():
        override = user_joints.get(name)
        if isinstance(override, dict):
            for key in ("damping", "friction"):
                if override.get(key) is not None:
                    values[key] = _pos_float(override[key], values[key])
    _sync_part_joint_damping(physics, joints_info)
    return physics


def write_dynamics_to_urdf(results_dir: str, joint_dynamics: dict):
    """Write <dynamics damping friction> onto each movable joint in place."""
    urdf_path = os.path.join(results_dir, "mobility.urdf")
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if name not in joint_dynamics:
            continue
        for old in joint.findall("dynamics"):
            joint.remove(old)
        dyn = ET.SubElement(joint, "dynamics")
        dyn.attrib = {
            "damping": repr(float(joint_dynamics[name]["damping"])),
            "friction": repr(float(joint_dynamics[name]["friction"])),
        }
    try:
        ET.indent(tree, space=" ")
    except AttributeError:  # Python < 3.9
        pass
    tree.write(urdf_path, xml_declaration=True, encoding="utf-8")


def ensure_physics_current(results_dir: str) -> bool:
    """Check an existing physics_properties.json against the published URDF.

    Returns True (and the caller may skip re-estimation) only when the file's
    joint inventory matches the URDF's movable joints; if the URDF lost its
    <dynamics> elements (step 5 republishes mobility.urdf without them on
    reruns), they are re-applied from the file — no VLM call needed. Returns
    False when the inventories diverge (e.g. the object was re-articulated),
    so the caller runs a fresh estimation.
    """
    physics_path = os.path.join(results_dir, PHYSICS_FILENAME)
    if not os.path.exists(physics_path):
        return False
    try:
        with open(physics_path) as f:
            physics = json.load(f)
        joints = physics.get("joints")
        if not isinstance(joints, dict):
            return False
        for entry in joints.values():
            float(entry["damping"])
            float(entry["friction"])
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        return False

    urdf_path = os.path.join(results_dir, "mobility.urdf")
    root = ET.parse(urdf_path).getroot()
    movable = {
        j.attrib["name"]: j for j in root.findall("joint")
        if j.attrib.get("type", "fixed") in MOVABLE_TYPES
    }
    if set(movable) != set(joints):
        logger.info("physics_properties.json joints %s do not match URDF %s; re-estimating",
                    sorted(joints), sorted(movable))
        return False
    missing_dynamics = [name for name, el in movable.items() if el.find("dynamics") is None]
    if missing_dynamics:
        logger.info("Re-applying <dynamics> from physics_properties.json to %s", missing_dynamics)
        write_dynamics_to_urdf(results_dir, joints)
    return True


def estimate_physics_properties(cfg, results_dir: str, object_name: str,
                                image_path: str | None = None,
                                scale: float | None = None,
                                query_fn=None, use_vlm: bool = True,
                                user_values: dict | None = None,
                                verbose: bool = False) -> dict:
    """Run the physics estimation for one published object.

    Args:
        cfg: step config (model_name, gcloud_project, gcloud_location, ...) —
            only used to build the default VLM client.
        results_dir: the object's results/ directory (mobility.urdf + meshes/).
        object_name: human-readable object name for the prompt.
        image_path: the object's source image (optional; without it the VLM
            estimates from text alone).
        scale: canonical-mesh -> real-world scale when known (the SimFoundry
            pipeline passes it per object; standalone users can set it in
            their objects config entry or the user-values JSON); None means
            unknown, in which case the prompt tells the VLM to trust the
            picture over the dimensions.
        query_fn: optional (user_prompt, system_prompt, image_paths) -> str
            override, used by tests to avoid a live VLM call.
        use_vlm: set False to skip the VLM entirely (offline standalone use:
            defaults filled, then user_values applied).
        user_values: this object's entry from the user-values JSON (see module
            docstring); known values here are authoritative over estimates.

    Returns the dict written to physics_properties.json.
    """
    user_values = user_values or {}
    if user_values.get("scale") is not None:
        scale = float(user_values["scale"])
    scale_is_real = scale is not None
    parts_info, joints_info = collect_urdf_physics_targets(
        results_dir, scale=scale if scale_is_real else 1.0)
    if not parts_info:
        raise ValueError(f"No mesh-bearing links found in {results_dir}/mobility.urdf")

    result_json = {}
    if use_vlm:
        user_prompt, system_prompt = physics_prompt(object_name, parts_info, joints_info, scale_is_real)

        if query_fn is None:
            from query_vlm import make_vlm

            def query_fn(user_prompt, system_prompt, image_paths):
                model = make_vlm(cfg, verbose=verbose)
                result = model(user_prompt, system_prompt, image_paths=image_paths)
                return model.get_result_text(result)

        image_paths = [image_path] if image_path and os.path.exists(image_path) else []
        if not image_paths:
            logger.warning("Object image %s missing; estimating physics from text only", image_path)
        result_text = query_fn(user_prompt, system_prompt, image_paths)

        with open(os.path.join(results_dir, RAW_RESPONSE_FILENAME), "w") as f:
            f.write(result_text)

        from articulate_anything.utils.prompt_utils import extract_json_from_response
        try:
            result_json = extract_json_from_response(result_text)
        except Exception as exc:
            logger.warning("Physics VLM response was not parseable JSON (%s); using defaults", exc)
            result_json = {}

    physics = sanitize_physics_result(result_json, parts_info, joints_info)
    physics = apply_user_values(physics, user_values, joints_info)
    write_dynamics_to_urdf(results_dir, physics["joints"])

    source = "vlm" if use_vlm else "defaults"
    if user_values.get("parts") or user_values.get("joints"):
        source += "+user"
    payload = {
        "version": 1,
        "source": source,
        "model_name": (cfg.get("model_name", None) if hasattr(cfg, "get") else None) if use_vlm else None,
        "scale": float(scale) if scale_is_real else None,
        **physics,
    }
    with open(os.path.join(results_dir, PHYSICS_FILENAME), "w") as f:
        json.dump(payload, f, indent=4)

    if verbose:
        print(f"  Physics ({source}) for {object_name}: "
              f"{len(physics['parts'])} part(s), {len(physics['joints'])} joint(s)")
    return payload
