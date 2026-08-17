"""Edit model for a published articulation result (results/mobility.urdf).

Parses the URDF with stdlib ElementTree (the workflow's own idiom — the
articulate conda envs have no lxml), keeps joint/link state in dataclasses,
and writes edits back to the same file with a pristine backup and a versioned
copy per save.

Frame conventions (from the publisher in articulate_simfoundry.py):
    - Part meshes are stored in the whole-object canonical frame (y-up).
    - A fixed joint ``base -> <root>_link`` with origin rpy ~ (pi/2, 0, pi/2)
      rotates everything into the z-up simulator frame; the empty ``base``
      link is the world frame here.
    - Movable joints carry <origin xyz> (no rpy) and <axis xyz> in the parent
      link frame, and their origin xyz is exactly compensated by the child
      link's visual/collision origins so the part stays put at q=0 (the
      translate_link pivot relocation in odio_urdf). Every origin edit here
      preserves that invariant.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .physics_overrides import (
    OVERRIDES_FILENAME,
    load_physics_estimates,
    load_physics_overrides,
)

logger = logging.getLogger(__name__)

MOVABLE_TYPES = ("revolute", "prismatic", "continuous")
# Placeholder effort/velocity written by the articulation workflow; kept as
# defaults when a limit element has to be created from scratch.
DEFAULT_EFFORT = 5.0
DEFAULT_VELOCITY = 5.0
DEFAULT_REVOLUTE_RANGE = (0.0, math.pi / 2)
DEFAULT_PRISMATIC_RANGE = (0.0, 0.1)

BACKUP_FILENAME = "mobility_original.urdf"
REFINEMENT_LOG_FILENAME = "refinement_log.json"


def rpy_to_matrix(rpy) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw to a 3x3 rotation (R = Rz @ Ry @ Rx)."""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def make_transform(xyz, rpy=None) -> np.ndarray:
    tf = np.eye(4)
    if rpy is not None:
        tf[:3, :3] = rpy_to_matrix(rpy)
    tf[:3, 3] = np.asarray(xyz, dtype=float)
    return tf


def rotation_about_axis(axis, angle: float) -> np.ndarray:
    """4x4 rotation of `angle` radians about `axis` through the origin."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(4)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    rot = np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])
    tf = np.eye(4)
    tf[:3, :3] = rot
    return tf


def _parse_vec(text: str | None, default: str) -> np.ndarray:
    return np.array([float(v) for v in (text or default).split()], dtype=float)


def _fmt_vec(vec) -> str:
    return " ".join(repr(float(v)) for v in vec)


@dataclass
class GeomOrigin:
    """One <visual> or <collision> origin of a link, tied to its element."""

    element: ET.Element  # the <visual> / <collision> element
    xyz: np.ndarray
    rpy: np.ndarray


@dataclass
class LinkState:
    name: str
    element: ET.Element
    geoms: list[GeomOrigin] = field(default_factory=list)
    mesh_files: list[str] = field(default_factory=list)


@dataclass
class JointState:
    name: str
    joint_type: str
    parent: str
    child: str
    element: ET.Element
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray | None
    # limit/dynamics are plain dicts so snapshots stay trivially serializable
    limit: dict | None  # {"lower","upper","effort","velocity"}
    dynamics: dict | None  # {"damping","friction"}

    @property
    def is_movable(self) -> bool:
        return self.joint_type in MOVABLE_TYPES


class ArticulationModel:
    """Load, edit, and save one articulated object's mobility.urdf."""

    def __init__(self, results_dir: str, max_faces_per_link: int = 40000):
        self.results_dir = os.path.abspath(results_dir)
        self.urdf_path = os.path.join(self.results_dir, "mobility.urdf")
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"No mobility.urdf in {self.results_dir}")
        self.max_faces_per_link = max_faces_per_link

        self.tree = ET.parse(self.urdf_path)
        self.root = self.tree.getroot()
        self.robot_name = self.root.attrib.get(
            "name", os.path.basename(os.path.dirname(self.results_dir)))

        self.links: dict[str, LinkState] = {}
        self.joints: dict[str, JointState] = {}
        self._child_to_joint: dict[str, str] = {}
        self._parse()

        self.overrides = load_physics_overrides(self.results_dir)
        # Pipeline-estimated physics (physics_properties.json), if the
        # estimation step ran: displayed as baseline values in the UI.
        # URDF <dynamics> already carry the per-joint values, so nothing is
        # seeded into overrides — those hold user edits only.
        self.estimates = load_physics_estimates(self.results_dir)

        self._original = self._snapshot()
        self._mesh_cache: dict[str, dict | None] = {}
        self.dirty = False

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self):
        for link_el in self.root.findall("link"):
            name = link_el.attrib["name"]
            state = LinkState(name=name, element=link_el)
            for geom_el in list(link_el.findall("visual")) + list(link_el.findall("collision")):
                origin_el = geom_el.find("origin")
                xyz = _parse_vec(origin_el.attrib.get("xyz") if origin_el is not None else None, "0 0 0")
                rpy = _parse_vec(origin_el.attrib.get("rpy") if origin_el is not None else None, "0 0 0")
                state.geoms.append(GeomOrigin(element=geom_el, xyz=xyz, rpy=rpy))
                mesh_el = geom_el.find("geometry/mesh")
                if mesh_el is not None and mesh_el.attrib.get("filename"):
                    state.mesh_files.append(mesh_el.attrib["filename"])
            self.links[name] = state

        for joint_el in self.root.findall("joint"):
            name = joint_el.attrib["name"]
            joint_type = joint_el.attrib.get("type", "fixed")
            parent = joint_el.find("parent").attrib["link"]
            child = joint_el.find("child").attrib["link"]
            origin_el = joint_el.find("origin")
            origin_xyz = _parse_vec(origin_el.attrib.get("xyz") if origin_el is not None else None, "0 0 0")
            origin_rpy = _parse_vec(origin_el.attrib.get("rpy") if origin_el is not None else None, "0 0 0")
            axis_el = joint_el.find("axis")
            axis = _parse_vec(axis_el.attrib.get("xyz"), "1 0 0") if axis_el is not None else None
            limit_el = joint_el.find("limit")
            limit = None
            # Continuous joints have no limits by URDF semantics; drop any
            # stray element a hand-edited file might carry.
            if limit_el is not None and joint_type != "continuous":
                limit = {
                    "lower": float(limit_el.attrib.get("lower", 0.0)),
                    "upper": float(limit_el.attrib.get("upper", 0.0)),
                    "effort": float(limit_el.attrib.get("effort", DEFAULT_EFFORT)),
                    "velocity": float(limit_el.attrib.get("velocity", DEFAULT_VELOCITY)),
                }
            dyn_el = joint_el.find("dynamics")
            dynamics = None
            if dyn_el is not None:
                dynamics = {}
                if "damping" in dyn_el.attrib:
                    dynamics["damping"] = float(dyn_el.attrib["damping"])
                if "friction" in dyn_el.attrib:
                    dynamics["friction"] = float(dyn_el.attrib["friction"])
            self.joints[name] = JointState(
                name=name,
                joint_type=joint_type,
                parent=parent,
                child=child,
                element=joint_el,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                axis=axis,
                limit=limit,
                dynamics=dynamics,
            )
            self._child_to_joint[child] = name

    @property
    def root_link(self) -> str:
        for name in self.links:
            if name not in self._child_to_joint:
                return name
        raise ValueError(f"URDF {self.urdf_path} has no root link (joint cycle?)")

    def movable_joints(self) -> list[str]:
        return [j.name for j in self.joints.values() if j.is_movable]

    def editable_joints(self) -> list[str]:
        """Movable joints first, then fixed joints except the virtual-base one
        (whose rpy is the object's global orientation, baked into all meshes,
        joint origins and axes by the sim-ready importer — not a per-joint
        refinement knob)."""
        movable = [j.name for j in self.joints.values() if j.is_movable]
        fixed = [
            j.name for j in self.joints.values()
            if not j.is_movable and j.parent != "base"
        ]
        return movable + fixed

    def geometry_links(self) -> list[str]:
        return [name for name, link in self.links.items() if link.mesh_files]

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    def joint_motion(self, joint: JointState, q: float) -> np.ndarray:
        if joint.joint_type in ("revolute", "continuous"):
            return rotation_about_axis(joint.axis if joint.axis is not None else (1, 0, 0), q)
        if joint.joint_type == "prismatic":
            axis = np.asarray(joint.axis if joint.axis is not None else (1, 0, 0), dtype=float)
            norm = np.linalg.norm(axis)
            axis = axis / norm if norm > 1e-12 else axis
            return make_transform(axis * q)
        return np.eye(4)

    def link_world_transform(self, link_name: str, q_values: dict[str, float] | None = None) -> np.ndarray:
        """World (z-up sim frame) pose of a link frame at joint config q_values."""
        q_values = q_values or {}
        joint_name = self._child_to_joint.get(link_name)
        if joint_name is None:
            return np.eye(4)
        joint = self.joints[joint_name]
        parent_tf = self.link_world_transform(joint.parent, q_values)
        origin_tf = make_transform(joint.origin_xyz, joint.origin_rpy)
        motion_tf = self.joint_motion(joint, float(q_values.get(joint_name, 0.0)))
        return parent_tf @ origin_tf @ motion_tf

    def joint_world_frame(self, joint_name: str, q_values: dict[str, float] | None = None):
        """(pivot_world, axis_world, R_parent_world) of a joint at q_values."""
        joint = self.joints[joint_name]
        parent_tf = self.link_world_transform(joint.parent, q_values)
        origin_tf = parent_tf @ make_transform(joint.origin_xyz, joint.origin_rpy)
        pivot = origin_tf[:3, 3]
        axis = joint.axis if joint.axis is not None else np.array([1.0, 0.0, 0.0])
        axis_world = origin_tf[:3, :3] @ np.asarray(axis, dtype=float)
        norm = np.linalg.norm(axis_world)
        if norm > 1e-12:
            axis_world = axis_world / norm
        return pivot, axis_world, parent_tf[:3, :3]

    def descendant_links(self, joint_name: str) -> list[str]:
        """The joint's child link and everything hanging below it."""
        result = []
        stack = [self.joints[joint_name].child]
        children_of: dict[str, list[str]] = {}
        for j in self.joints.values():
            children_of.setdefault(j.parent, []).append(j.child)
        while stack:
            link = stack.pop()
            result.append(link)
            stack.extend(children_of.get(link, []))
        return result

    def world_point_to_parent_frame(self, joint_name: str, point_world, q_values=None) -> np.ndarray:
        joint = self.joints[joint_name]
        parent_tf = self.link_world_transform(joint.parent, q_values)
        inv = np.linalg.inv(parent_tf)
        p = np.asarray(point_world, dtype=float)
        return (inv[:3, :3] @ p) + inv[:3, 3]

    def world_point_to_pivot(self, joint_name: str, point_world, q_values=None,
                             clicked_link: str | None = None) -> np.ndarray:
        """Convert a clicked world point into a joint-origin candidate (parent
        frame).

        When the click landed on the joint's own subtree while the joint is
        displaced (q != 0), the picked feature is at its *moved* position; the
        user means the feature itself, so map the point back through the
        inverse joint motion to where that material point sits at rest. Clicks
        on links outside the subtree (or with unknown provenance) are treated
        as plain spatial points in the parent frame.
        """
        joint = self.joints[joint_name]
        p_parent = self.world_point_to_parent_frame(joint_name, point_world, q_values)
        q = float((q_values or {}).get(joint_name, 0.0))
        on_subtree = clicked_link is not None and clicked_link in self.descendant_links(joint_name)
        if not on_subtree or not joint.is_movable or abs(q) < 1e-12:
            return p_parent
        origin_tf = make_transform(joint.origin_xyz, joint.origin_rpy)
        # rest position of the picked material point: T_origin @ M(q)^-1 @ T_origin^-1 @ p
        undo = origin_tf @ np.linalg.inv(self.joint_motion(joint, q)) @ np.linalg.inv(origin_tf)
        return undo[:3, :3] @ p_parent + undo[:3, 3]

    def world_dir_to_parent_frame(self, joint_name: str, dir_world, q_values=None) -> np.ndarray:
        """World direction -> the frame <axis> lives in (the joint frame:
        parent rotation composed with the joint origin rpy, which is zero for
        all workflow-generated movable joints)."""
        joint = self.joints[joint_name]
        parent_tf = self.link_world_transform(joint.parent, q_values)
        joint_rot = parent_tf[:3, :3] @ rpy_to_matrix(joint.origin_rpy)
        return joint_rot.T @ np.asarray(dir_world, dtype=float)

    # ------------------------------------------------------------------
    # Mesh access (display only)
    # ------------------------------------------------------------------

    def _resolve_mesh_path(self, filename: str) -> str | None:
        candidates = [
            os.path.join(self.results_dir, filename),
            os.path.join(self.results_dir, "meshes", os.path.basename(filename)),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def link_mesh(self, link_name: str) -> dict | None:
        """Cached {vertices, faces, vertex_colors, centroid} for a link's first
        visual mesh.

        Vertices are raw mesh coordinates: the caller applies the visual origin
        and link world transform. Decimated for display when above the face cap.
        """
        if link_name in self._mesh_cache:
            return self._mesh_cache[link_name]
        link = self.links.get(link_name)
        if link is None or not link.mesh_files:
            self._mesh_cache[link_name] = None
            return None
        mesh_path = self._resolve_mesh_path(link.mesh_files[0])
        if mesh_path is None:
            logger.warning("Mesh for link %s not found (%s)", link_name, link.mesh_files[0])
            self._mesh_cache[link_name] = None
            return None

        import trimesh

        tm = trimesh.load(mesh_path)
        if isinstance(tm, trimesh.Scene):
            tm = tm.to_geometry()

        vertex_colors = None
        try:
            color_visual = tm.visual.to_color()
            colors = np.asarray(color_visual.vertex_colors)
            if colors.ndim == 2 and len(colors) == len(tm.vertices):
                vertex_colors = colors[:, :3].astype(np.uint8)
        except Exception:  # texture conversion is best-effort, palette fallback
            vertex_colors = None

        if len(tm.faces) > self.max_faces_per_link:
            try:
                simplified = tm.simplify_quadric_decimation(face_count=self.max_faces_per_link)
                if len(simplified.faces) > 0:
                    tm = simplified
                    vertex_colors = None  # decimation invalidates per-vertex colors
            except BaseException:
                logger.info("Decimation unavailable; rendering %s at full resolution", link_name)

        vertices = np.asarray(tm.vertices, dtype=float)
        data = {
            "vertices": vertices,
            "faces": np.asarray(tm.faces, dtype=np.int64),
            "vertex_colors": vertex_colors,
            "centroid": vertices.mean(axis=0) if len(vertices) else np.zeros(3),
        }
        self._mesh_cache[link_name] = data
        return data

    def bounding_radius(self) -> float:
        """Rough world-space scene radius, used to size gizmos."""
        radius = 0.1
        for link_name in self.geometry_links():
            mesh = self.link_mesh(link_name)
            if mesh is None or len(mesh["vertices"]) == 0:
                continue
            link = self.links[link_name]
            tf = self.link_world_transform(link_name)
            if link.geoms:
                tf = tf @ make_transform(link.geoms[0].xyz, link.geoms[0].rpy)
            verts = mesh["vertices"] @ tf[:3, :3].T + tf[:3, 3]
            radius = max(radius, float(np.abs(verts).max()))
        return radius

    # ------------------------------------------------------------------
    # Edits
    # ------------------------------------------------------------------

    def set_axis(self, joint_name: str, xyz) -> np.ndarray:
        joint = self.joints[joint_name]
        axis = np.asarray([float(v) for v in xyz], dtype=float)
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            raise ValueError("Joint axis must be non-zero")
        axis = axis / norm
        joint.axis = axis
        self.dirty = True
        return axis

    def set_origin(self, joint_name: str, xyz, compensate: bool = True):
        """Move a joint's pivot; by default shift everything defined in the
        child link's frame — its visual/collision origins AND the origins of
        joints parented to it — by the opposite amount so the whole subtree's
        rest pose (q=0) is unchanged, the invariant the workflow's own pivot
        relocation (translate_link) maintains."""
        joint = self.joints[joint_name]
        new_xyz = np.asarray([float(v) for v in xyz], dtype=float)
        if compensate:
            rot = rpy_to_matrix(joint.origin_rpy)
            delta_child = rot.T @ (joint.origin_xyz - new_xyz)
            child = self.links[joint.child]
            for geom in child.geoms:
                geom.xyz = geom.xyz + delta_child
            for other in self.joints.values():
                if other.parent == joint.child:
                    other.origin_xyz = other.origin_xyz + delta_child
        joint.origin_xyz = new_xyz
        self.dirty = True

    def set_limits(self, joint_name: str, lower: float, upper: float,
                   effort: float | None = None, velocity: float | None = None):
        joint = self.joints[joint_name]
        if joint.joint_type == "continuous":
            raise ValueError("Continuous joints have no limits; change the joint type first")
        lower, upper = float(lower), float(upper)
        if not lower < upper:
            raise ValueError(f"Limit lower ({lower}) must be < upper ({upper})")
        current = joint.limit or {"effort": DEFAULT_EFFORT, "velocity": DEFAULT_VELOCITY}
        effort = float(effort) if effort is not None else current.get("effort", DEFAULT_EFFORT)
        velocity = float(velocity) if velocity is not None else current.get("velocity", DEFAULT_VELOCITY)
        if effort <= 0 or velocity <= 0:
            raise ValueError("Limit effort and velocity must be > 0")
        joint.limit = {"lower": lower, "upper": upper, "effort": effort, "velocity": velocity}
        self.dirty = True

    def set_joint_type(self, joint_name: str, new_type: str):
        if new_type not in MOVABLE_TYPES + ("fixed",):
            raise ValueError(f"Unsupported joint type: {new_type}")
        joint = self.joints[joint_name]
        if joint.parent == "base":
            raise ValueError("The virtual base joint cannot change type")
        old_type = joint.joint_type
        joint.joint_type = new_type
        if new_type == "continuous":
            joint.limit = None  # limitless by URDF semantics
        elif new_type in ("revolute", "prismatic") and joint.limit is None:
            lower, upper = DEFAULT_REVOLUTE_RANGE if new_type == "revolute" else DEFAULT_PRISMATIC_RANGE
            joint.limit = {"lower": lower, "upper": upper,
                           "effort": DEFAULT_EFFORT, "velocity": DEFAULT_VELOCITY}
        if new_type in MOVABLE_TYPES and joint.axis is None:
            joint.axis = np.array([1.0, 0.0, 0.0])
        if old_type != new_type:
            self.dirty = True

    def set_joint_dynamics(self, joint_name: str, damping: float | None, friction: float | None):
        """Per-joint damping/friction. Written to the URDF for provenance and
        direct consumers, and mirrored to physics_overrides.json for pipeline
        stages that re-author <dynamics> from other sources."""
        joint = self.joints[joint_name]
        dynamics = {}
        if damping is not None:
            if float(damping) < 0:
                raise ValueError("Damping must be >= 0")
            dynamics["damping"] = float(damping)
        if friction is not None:
            if float(friction) < 0:
                raise ValueError("Joint friction must be >= 0")
            dynamics["friction"] = float(friction)
        joint.dynamics = dynamics or None
        if dynamics:
            self.overrides["joints"][joint_name] = dict(dynamics)
        else:
            self.overrides["joints"].pop(joint_name, None)
        self.dirty = True

    def set_part_properties(self, link_name: str, mass_kg: float | None = None,
                            friction: float | None = None, joint_damping: float | None = None):
        """Per-link physical properties for the sim-ready importer (mass ->
        inertial, surface friction, joint_damping -> child joint's damping)."""
        if link_name not in self.links:
            raise ValueError(f"Unknown link: {link_name}")
        values = (("mass_kg", mass_kg), ("friction", friction), ("joint_damping", joint_damping))
        # Validate everything before writing anything, so a rejected edit
        # never leaves a partially-updated entry behind.
        for key, value in values:
            if value is not None and float(value) < 0:
                raise ValueError(f"{key} must be >= 0")
        entry = self.overrides["parts"].setdefault(link_name, {})
        for key, value in values:
            if value is None:
                entry.pop(key, None)
            else:
                entry[key] = float(value)
        if not entry:
            self.overrides["parts"].pop(link_name, None)
        self.dirty = True

    # ------------------------------------------------------------------
    # Snapshots / undo / reset
    # ------------------------------------------------------------------

    def _snapshot(self) -> dict:
        return {
            "joints": {
                name: {
                    "joint_type": j.joint_type,
                    "origin_xyz": j.origin_xyz.tolist(),
                    "origin_rpy": j.origin_rpy.tolist(),
                    "axis": None if j.axis is None else np.asarray(j.axis, dtype=float).tolist(),
                    "limit": copy.deepcopy(j.limit),
                    "dynamics": copy.deepcopy(j.dynamics),
                }
                for name, j in self.joints.items()
            },
            "links": {
                name: [g.xyz.tolist() for g in link.geoms]
                for name, link in self.links.items()
            },
            "overrides": copy.deepcopy(self.overrides),
        }

    def snapshot(self) -> dict:
        return self._snapshot()

    def restore(self, state: dict):
        for name, saved in state["joints"].items():
            joint = self.joints[name]
            joint.joint_type = saved["joint_type"]
            joint.origin_xyz = np.array(saved["origin_xyz"], dtype=float)
            joint.origin_rpy = np.array(saved["origin_rpy"], dtype=float)
            joint.axis = None if saved["axis"] is None else np.array(saved["axis"], dtype=float)
            joint.limit = copy.deepcopy(saved["limit"])
            joint.dynamics = copy.deepcopy(saved["dynamics"])
        for name, geom_xyzs in state["links"].items():
            link = self.links[name]
            for geom, xyz in zip(link.geoms, geom_xyzs):
                geom.xyz = np.array(xyz, dtype=float)
        self.overrides = copy.deepcopy(state["overrides"])
        # Restoring may land exactly back on the saved baseline (e.g. undoing
        # the only edit) — recompute instead of assuming dirty.
        self.dirty = self._snapshot() != self._original

    def reset_joint(self, joint_name: str):
        """Restore just this joint and its child geometry from the original."""
        original = self._original
        saved = original["joints"][joint_name]
        joint = self.joints[joint_name]
        joint.joint_type = saved["joint_type"]
        joint.origin_xyz = np.array(saved["origin_xyz"], dtype=float)
        joint.origin_rpy = np.array(saved["origin_rpy"], dtype=float)
        joint.axis = None if saved["axis"] is None else np.array(saved["axis"], dtype=float)
        joint.limit = copy.deepcopy(saved["limit"])
        joint.dynamics = copy.deepcopy(saved["dynamics"])
        link = self.links[joint.child]
        for geom, xyz in zip(link.geoms, original["links"][joint.child]):
            geom.xyz = np.array(xyz, dtype=float)
        if joint_name in original["overrides"]["joints"]:
            self.overrides["joints"][joint_name] = copy.deepcopy(original["overrides"]["joints"][joint_name])
        else:
            self.overrides["joints"].pop(joint_name, None)
        self.dirty = self._snapshot() != self._original

    def reset_all(self):
        self.restore(self._original)

    # ------------------------------------------------------------------
    # Validation / save
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Hard errors that would break downstream consumers."""
        errors = []
        for joint in self.joints.values():
            if joint.joint_type in ("revolute", "prismatic"):
                if joint.limit is None:
                    # urdfpy-based consumers hard-fail on movable joints
                    # without limits.
                    errors.append(f"{joint.name}: {joint.joint_type} joint requires a <limit>")
                elif not joint.limit["lower"] < joint.limit["upper"]:
                    errors.append(f"{joint.name}: limit lower must be < upper")
            if joint.is_movable:
                axis = joint.axis
                if axis is None or np.linalg.norm(axis) < 1e-9:
                    errors.append(f"{joint.name}: movable joint requires a non-zero <axis>")
        return errors

    def warnings(self) -> list[str]:
        warns = []
        for joint in self.joints.values():
            if joint.joint_type in ("revolute", "prismatic") and joint.limit is not None:
                if joint.limit["lower"] > 0.0 or joint.limit["upper"] < 0.0:
                    warns.append(
                        f"{joint.name}: limits exclude q=0 (the as-scanned rest pose); "
                        "downstream metadata is still computed at q=0"
                    )
        return warns

    def _sync_tree(self):
        """Write dataclass state back into the ElementTree."""
        for joint in self.joints.values():
            el = joint.element
            el.attrib["type"] = joint.joint_type

            origin_el = el.find("origin")
            if origin_el is None:
                origin_el = ET.SubElement(el, "origin")
            origin_el.attrib["xyz"] = _fmt_vec(joint.origin_xyz)
            if np.any(np.abs(joint.origin_rpy) > 1e-15) or "rpy" in origin_el.attrib:
                origin_el.attrib["rpy"] = _fmt_vec(joint.origin_rpy)

            axis_el = el.find("axis")
            if joint.axis is not None:
                if axis_el is None:
                    axis_el = ET.SubElement(el, "axis")
                axis = np.asarray(joint.axis, dtype=float)
                norm = np.linalg.norm(axis)
                if norm > 1e-12:
                    axis = axis / norm
                axis_el.attrib["xyz"] = _fmt_vec(axis)
            # A fixed joint keeping a scaffold axis element is harmless; leave it.

            limit_el = el.find("limit")
            if joint.joint_type in ("revolute", "prismatic") and joint.limit is not None:
                if limit_el is None:
                    limit_el = ET.SubElement(el, "limit")
                limit_el.attrib["lower"] = repr(float(joint.limit["lower"]))
                limit_el.attrib["upper"] = repr(float(joint.limit["upper"]))
                limit_el.attrib["effort"] = repr(float(joint.limit["effort"]))
                limit_el.attrib["velocity"] = repr(float(joint.limit["velocity"]))
            elif limit_el is not None:
                # Continuous/fixed joints must not carry a limit element.
                el.remove(limit_el)

            dyn_el = el.find("dynamics")
            if joint.is_movable and joint.dynamics:
                if dyn_el is None:
                    dyn_el = ET.SubElement(el, "dynamics")
                dyn_el.attrib.clear()
                for key in ("damping", "friction"):
                    if key in joint.dynamics:
                        dyn_el.attrib[key] = repr(float(joint.dynamics[key]))
            elif dyn_el is not None:
                el.remove(dyn_el)

        for link in self.links.values():
            for geom in link.geoms:
                origin_el = geom.element.find("origin")
                if origin_el is None:
                    origin_el = ET.SubElement(geom.element, "origin")
                    origin_el.attrib["rpy"] = _fmt_vec(geom.rpy)
                origin_el.attrib["xyz"] = _fmt_vec(geom.xyz)

    def _next_version(self) -> int:
        version = 1
        while os.path.exists(os.path.join(self.results_dir, f"mobility_refined_{version}.urdf")):
            version += 1
        return version

    def changed_joints(self) -> list[str]:
        changed = []
        current = self._snapshot()
        for name, saved in self._original["joints"].items():
            if current["joints"][name] != saved:
                changed.append(name)
        return changed

    def save(self) -> dict:
        """Persist edits.

        Writes, in order: a one-time pristine backup (mobility_original.urdf),
        a versioned copy (mobility_refined_<N>.urdf), the in-place
        results/mobility.urdf that downstream stages consume,
        physics_overrides.json, and an entry in refinement_log.json.
        """
        errors = self.validate()
        if errors:
            raise ValueError("Cannot save invalid articulation:\n  " + "\n  ".join(errors))

        backup_path = os.path.join(self.results_dir, BACKUP_FILENAME)
        if not os.path.exists(backup_path):
            shutil.copy(self.urdf_path, backup_path)

        self._sync_tree()
        try:
            ET.indent(self.tree, space=" ")
        except AttributeError:  # Python < 3.9
            pass
        version = self._next_version()
        versioned_path = os.path.join(self.results_dir, f"mobility_refined_{version}.urdf")
        self.tree.write(versioned_path, xml_declaration=True, encoding="utf-8")
        self.tree.write(self.urdf_path, xml_declaration=True, encoding="utf-8")

        overrides_path = os.path.join(self.results_dir, OVERRIDES_FILENAME)
        with open(overrides_path, "w") as f:
            json.dump(self.overrides, f, indent=4)

        changed = self.changed_joints()
        log_path = os.path.join(self.results_dir, REFINEMENT_LOG_FILENAME)
        log = []
        if os.path.exists(log_path):
            try:
                with open(log_path) as f:
                    log = json.load(f)
            except (json.JSONDecodeError, OSError):
                log = []
        log.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "version": version,
            "changed_joints": changed,
            "warnings": self.warnings(),
        })
        with open(log_path, "w") as f:
            json.dump(log, f, indent=4)

        # The refined URDF is now the baseline for subsequent edit sessions.
        self._original = self._snapshot()
        self.dirty = False
        summary = {
            "urdf_path": self.urdf_path,
            "versioned_path": versioned_path,
            "overrides_path": overrides_path,
            "backup_path": backup_path,
            "changed_joints": changed,
            "version": version,
        }
        logger.info("Saved refined articulation: %s", summary)
        return summary
