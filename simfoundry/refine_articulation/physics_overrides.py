"""User physics overrides for articulated objects.

The SimFoundry pipeline's sim-ready stage (stage 10) deletes every <dynamics>
and <inertial> element of results/mobility.urdf and re-authors them from a VLM
per-part-properties query, so dynamics edited into the URDF alone would be
silently lost downstream. The refinement UI therefore also persists them to
``results/physics_overrides.json``:

    {
        "version": 1,
        "parts":  {"<link_name>":  {"mass_kg": 1.5, "friction": 0.5, "joint_damping": 0.2}},
        "joints": {"<joint_name>": {"damping": 0.2, "friction": 0.02}}
    }

``parts`` entries mirror the schema of stage 10's VLM parts_properties query
(mass -> <inertial>, friction -> surface friction, joint_damping -> damping of
the joint whose child is that link) so a consumer can merge them over the VLM
values with ``merge_parts_properties``. ``joints`` entries carry the final
per-joint <dynamics> values, keyed by joint name.

This module is stdlib-only so any pipeline stage can import it without UI
dependencies.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

OVERRIDES_FILENAME = "physics_overrides.json"
# Pipeline-estimated physics, written by the estimation step (estimate_physics.py).
ESTIMATES_FILENAME = "physics_properties.json"

PART_KEYS = ("mass_kg", "friction", "joint_damping")
JOINT_KEYS = ("damping", "friction")


def empty_overrides() -> dict:
    return {"version": 1, "parts": {}, "joints": {}}


def load_physics_overrides(results_dir: str) -> dict:
    """Load results/physics_overrides.json, tolerating absence and bad files."""
    path = os.path.join(results_dir, OVERRIDES_FILENAME)
    if not os.path.exists(path):
        return empty_overrides()
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Ignoring unreadable %s: %s", path, exc)
        return empty_overrides()
    overrides = empty_overrides()
    if not isinstance(data, dict):
        logger.warning("Ignoring %s: expected a JSON object, got %s", path, type(data).__name__)
        return overrides
    for section, allowed in (("parts", PART_KEYS), ("joints", JOINT_KEYS)):
        entries = data.get(section)
        if not isinstance(entries, dict):
            continue
        for name, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            try:
                clean = {k: float(v) for k, v in entry.items() if k in allowed and v is not None}
            except (TypeError, ValueError) as exc:
                logger.warning("Ignoring %s entry %r in %s: %s", section, name, path, exc)
                continue
            if clean:
                overrides[section][name] = clean
    return overrides


def load_physics_estimates(results_dir: str) -> dict:
    """Load the pipeline's physics_properties.json as display baselines.

    Returns {"parts": {link: {mass_kg, friction, ...}}, "joints": {joint: {...}}}
    (empty sections when the estimation step has not run or the file is bad).
    """
    path = os.path.join(results_dir, ESTIMATES_FILENAME)
    estimates = {"parts": {}, "joints": {}}
    if not os.path.exists(path):
        return estimates
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Ignoring unreadable %s: %s", path, exc)
        return estimates
    if not isinstance(data, dict):
        return estimates
    for entry in data.get("parts") or []:
        if isinstance(entry, dict) and entry.get("name"):
            estimates["parts"][entry["name"]] = {k: v for k, v in entry.items() if k != "name"}
    joints = data.get("joints")
    if isinstance(joints, dict):
        for name, entry in joints.items():
            if isinstance(entry, dict):
                estimates["joints"][name] = dict(entry)
    return estimates


def merge_parts_properties(parts_properties: list, overrides: dict) -> list:
    """Apply per-link overrides to a VLM parts_properties list (in a copy).

    parts_properties is a list of ``{"name", "mass_kg", "friction",
    "joint_damping"}`` dicts; entries are matched by link name. Overrides for
    links the VLM did not report are appended so a user-added property still
    reaches the consumer.
    """
    part_overrides = overrides.get("parts") or {}
    merged = [dict(entry) for entry in parts_properties]
    seen = set()
    for entry in merged:
        name = entry.get("name")
        seen.add(name)
        if name in part_overrides:
            entry.update(part_overrides[name])
    for name, entry in part_overrides.items():
        if name not in seen:
            merged.append({"name": name, **entry})
    if part_overrides:
        logger.info("Applied user physics overrides for parts: %s", sorted(part_overrides))
    return merged
