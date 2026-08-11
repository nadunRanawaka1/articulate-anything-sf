"""Mesh part merging and export utilities."""

import logging
import os
import re
import shutil
from pathlib import Path

import numpy as np
import trimesh
from omegaconf import DictConfig


def load_segmentation_data(image_dir: str) -> tuple:
    """
    Load segmentation data from files.
    
    Args:
        image_dir: Directory containing segmentation outputs
        
    Returns:
        Tuple of (face2label, label2face_mask, mesh)
    """
    face2label = np.load(f"{image_dir}/face_ids.npy", allow_pickle=True)
    label2face_mask = np.load(f"{image_dir}/label_to_faces_mask.npz")['arr_0']
    mesh = trimesh.load(f"{image_dir}/original_mesh.glb").to_mesh()
    return face2label, label2face_mask, mesh


def get_base_segments(
    face2label: np.ndarray,
    part_segment_dict: dict,
    fixed_part_name: str
) -> list:
    """
    Get segments that should be assigned to the base/fixed part.
    
    Includes all unassigned segments plus any already assigned to fixed part.
    
    Args:
        face2label: Per-face segment labels
        part_segment_dict: Current part->segments mapping
        fixed_part_name: Name of the fixed/base part
        
    Returns:
        List of segment IDs for the base part
    """
    all_segments = [int(s) for s in np.unique(face2label) if s >= 0]
    detected_segments = [
        segment for segment_list in part_segment_dict.values() 
        for segment in segment_list
    ]
    base_segments = [
        segment for segment in all_segments if segment not in detected_segments
    ] + part_segment_dict.get(fixed_part_name, [])
    return base_segments


def _canonical_part_key(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def reconcile_part_segment_dict(
    part_segment_dict: dict,
    parts_list: list,
    fixed_part_name: str,
    num_segments: int,
) -> dict:
    """
    Validate and normalize the VLM's part->segments mapping against the
    stage-2 part inventory.

    The raw mapping is free-text VLM output: keys can drift from the supplied
    part names ('Door' / 'door 1'), segment ids can be hallucinated out of
    range, and one segment can be claimed by several parts. Every later stage
    keys meshes by exact part name, so drift here used to surface far
    downstream as missing meshes and name-contract failures.
    """
    allowed = list(dict.fromkeys(list(parts_list) + [fixed_part_name]))
    by_canonical = {}
    for name in allowed:
        by_canonical.setdefault(_canonical_part_key(name), name)

    resolved = {}
    unknown = []
    for raw_name, segments in part_segment_dict.items():
        if raw_name in allowed:
            target = raw_name
        else:
            target = by_canonical.get(_canonical_part_key(raw_name))
        if target is None:
            unknown.append(str(raw_name))
            continue
        if target != raw_name:
            logging.warning(
                f"part merge: normalized part name {raw_name!r} -> {target!r}"
            )
        resolved.setdefault(target, []).extend(int(s) for s in segments)

    if unknown:
        raise ValueError(
            f"part merge returned unknown part name(s) {sorted(unknown)}; "
            f"expected a subset of {allowed}. Refusing to guess: an unmatched "
            "name silently loses its mesh faces downstream."
        )

    claimed = {}
    cleaned = {}
    for name in allowed:
        if name not in resolved:
            continue
        kept = []
        for segment in dict.fromkeys(resolved[name]):
            if not 0 <= segment < num_segments:
                logging.warning(
                    f"part merge: dropping out-of-range segment {segment} "
                    f"claimed by {name!r} (segmentation has {num_segments})"
                )
                continue
            owner = claimed.get(segment)
            if owner is not None:
                logging.warning(
                    f"part merge: segment {segment} claimed by both {owner!r} "
                    f"and {name!r}; keeping {owner!r}"
                )
                continue
            claimed[segment] = name
            kept.append(segment)
        cleaned[name] = kept
    return cleaned


def compute_partname2face(part_segment_dict: dict, label2face_mask: np.ndarray) -> dict:
    """Expand part -> segment ids into part -> sorted face indices."""
    partname2face = {}
    for part_name, segment_ids in part_segment_dict.items():
        part_faces = []
        for seg_id in segment_ids:
            seg_id = int(seg_id)
            if seg_id < label2face_mask.shape[0]:
                part_faces.extend(label2face_mask[seg_id].nonzero()[0].tolist())
        partname2face[part_name] = sorted(set(part_faces))
    return partname2face


def write_part_id_map(
    mesh_parts_dir: str,
    part_segment_dict: dict,
    face2label: np.ndarray,
    partname2face: dict,
    verbose: bool = False,
):
    """
    Persist the explicit part-identity map next to the exported meshes, for
    every execution path. Downstream consumers (URDF scaffold, semantics,
    scoring) can key parts by these durable ids instead of re-deriving
    identity from mesh filenames. These files used to be written only after
    interactive correction, so autonomous runs had no durable part ids.
    """
    import json

    face2label_path = os.path.join(mesh_parts_dir, "face2label.json")
    with open(face2label_path, "w") as f:
        json.dump(np.asarray(face2label).tolist(), f)

    partname2face_path = os.path.join(mesh_parts_dir, "partname2face.json")
    with open(partname2face_path, "w") as f:
        json.dump(partname2face, f, indent=2)

    part_segment_path = os.path.join(mesh_parts_dir, "part_segment_dict.json")
    with open(part_segment_path, "w") as f:
        json.dump(
            {k: [int(s) for s in v] for k, v in part_segment_dict.items()},
            f, indent=2,
        )

    if verbose:
        print(f"Saved face2label to: {face2label_path}")
        print(f"Saved partname2face to: {partname2face_path}")
        print(f"Saved part_segment_dict to: {part_segment_path}")


def required_movable_parts(articulation_tree_dict: dict) -> list:
    """Part names whose meshes must exist to realize the tree's movable joints."""
    required = []
    for joint in articulation_tree_dict.get("joints", []):
        if joint.get("joint_type") == "fixed":
            continue
        child = joint["child_link"]
        part = child[: -len("_link")] if child.endswith("_link") else child
        required.append(part)
    return list(dict.fromkeys(required))


def _part_has_faces(part: str, partname2face: dict) -> bool:
    stem = re.sub(r"_\d+$", "", part)
    for name, faces in partname2face.items():
        if not faces:
            continue
        if name == part or re.sub(r"_\d+$", "", name) == stem:
            return True
    return False


def export_mesh_parts(
    mesh: trimesh.Trimesh,
    part_segment_dict: dict,
    label2face_mask: np.ndarray,
    output_dir: str,
    verbose: bool = False
) -> list:
    """
    Export mesh parts based on segment assignments.
    
    Args:
        mesh: The original mesh
        part_segment_dict: Mapping of part_name -> segment IDs
        label2face_mask: Binary mask [num_segments, num_faces]
        output_dir: Directory to save mesh parts
        verbose: Print debug info
        
    Returns:
        List of exported part meshes
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_part_meshes = []
    for part, segments in part_segment_dict.items():
        part_faces = []
        segments = set(segments)
        
        for segment in segments:
            segment = int(segment)
            faces = label2face_mask[segment].nonzero()[0]
            part_faces.extend(faces)
            if verbose:
                print(f"Part: {part}, Segment: {segment}, Faces: {len(faces)}")
        
        if not part_faces:
            continue
        
        part_mesh = mesh.copy()
        part_mesh.update_faces(part_faces)
        part_mesh.remove_unreferenced_vertices()
        
        # Fix face normals to ensure consistent outward-facing orientation
        # This prevents "invisible from one side" issues in renderers that use backface culling
        part_mesh.fix_normals()
        
        # Export as both formats:
        # - OBJ for reliable pipeline execution (PyBullet compatible)
        # - GLB for final visualization (preserves any textures/materials)
        part_mesh.export(f"{output_dir}/{part}.obj")
        part_mesh.export(f"{output_dir}/{part}.glb")
        all_part_meshes.append(part_mesh)
    
    return all_part_meshes


def create_unified_mesh(part_meshes: list, output_path: str):
    """
    Create a unified scene from individual part meshes.
    
    Args:
        part_meshes: List of trimesh objects
        output_path: Path to save the unified mesh
    """
    unified_mesh = trimesh.Scene()
    for part_mesh in part_meshes:
        unified_mesh.add_geometry(part_mesh)
    unified_mesh.export(output_path)


def merge_and_center_segmented_mesh(
    cfg: DictConfig,
    articulation_tree_dict: dict,
    verbose: bool = False,
    interactive: bool = False
):
    """
    Merge segmented mesh parts with optional interactive correction.
    
    Args:
        cfg: Configuration dict with image_dir, out_dir, object_path, rerun
        articulation_tree_dict: Articulation tree from VLM with parts list
        verbose: Print debug info
        interactive: If True, show interactive correction UI before merging
    """
    from query_vlm import merge_mesh_parts
    
    image_dir = cfg.image_dir
    out_dir = cfg.out_dir
    
    # Check if already done
    if Path(f"{out_dir}/unified_mesh.glb").exists():
        rerun = cfg.rerun or (interactive and getattr(cfg, 'rerun_correction', False))
        if not rerun:
            return
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Get parts list from articulation tree
    parts_list = [part['part_name'] for part in articulation_tree_dict['parts']]
    
    # Load segmentation data
    face2label, label2face_mask, mesh = load_segmentation_data(image_dir)
    
    # Setup output directories
    mesh_parts_dir = os.path.join(out_dir, "meshes")
    os.makedirs(mesh_parts_dir, exist_ok=True)
    
    # Copy original mesh
    original_mesh_file = f"{image_dir}/original_mesh.glb"
    shutil.copy(original_mesh_file, out_dir)
    
    # Check if we can resume from a previous correction session
    import json as json_module
    saved_correction_path = os.path.join(mesh_parts_dir, "part_segment_dict.json")
    saved_face2label_path = os.path.join(mesh_parts_dir, "face2label.json")
    resume_correction = getattr(cfg, 'resume_correction', False)
    
    if resume_correction and os.path.exists(saved_correction_path) and os.path.exists(saved_face2label_path):
        # Load from previous correction session
        if verbose:
            print(f"Resuming from previous correction session...")
            print(f"  Loading: {saved_correction_path}")
            print(f"  Loading: {saved_face2label_path}")
        
        with open(saved_correction_path, 'r') as f:
            part_segment_dict = json_module.load(f)
        
        with open(saved_face2label_path, 'r') as f:
            face2label = np.array(json_module.load(f))
        
        # Rebuild label2face_mask from face2label
        max_segment = int(np.max(face2label)) + 1
        label2face_mask = np.zeros((max_segment, len(face2label)), dtype=bool)
        for face_idx, seg_id in enumerate(face2label):
            if seg_id >= 0:
                label2face_mask[int(seg_id), face_idx] = True
        
        if verbose:
            print(f"  Loaded {len(part_segment_dict)} parts, {max_segment} segments")
    else:
        # Get VLM's segment assignments (fresh start)
        merge_json = merge_mesh_parts(cfg, parts_list, verbose)
        part_segment_dict = reconcile_part_segment_dict(
            merge_json['part_segment_dict'],
            parts_list,
            articulation_tree_dict['fixed_part_name'],
            num_segments=label2face_mask.shape[0],
        )

        # Add undetected segments to base/fixed part
        fixed_part_name = articulation_tree_dict['fixed_part_name']
        base_segments = get_base_segments(face2label, part_segment_dict, fixed_part_name)
        part_segment_dict[fixed_part_name] = base_segments
    
    # Interactive correction if enabled
    if interactive:
        from .interactive_ui import interactive_segment_correction
        
        result = interactive_segment_correction(
            mesh=mesh,
            face2label=face2label,
            label2face_mask=label2face_mask,
            part_segment_dict=part_segment_dict,
            parts_list=parts_list,
            explosion_factor=getattr(cfg, 'explosion_factor', 0.3),
            verbose=verbose,
            port=getattr(cfg, 'port', 8050)
        )
        # Update with results from interactive session (including any new segments from splits)
        part_segment_dict = result['assignment']
        face2label = result['face2label']
        label2face_mask = result['label2face_mask']
        
    # Persist the explicit part-identity map on every path (fresh, resumed,
    # interactive), then gate on it BEFORE any completion sentinel is written:
    # a movable part fused into the base cannot be recovered downstream, so
    # the object must fail here, loudly and diagnosably, not in stage 5.
    partname2face = compute_partname2face(part_segment_dict, label2face_mask)
    write_part_id_map(
        mesh_parts_dir, part_segment_dict, face2label, partname2face,
        verbose=verbose,
    )

    missing = [
        part for part in required_movable_parts(articulation_tree_dict)
        if not _part_has_faces(part, partname2face)
    ]
    if missing:
        raise ValueError(
            f"part merge left required movable part(s) {missing} without any "
            f"mesh faces (parts with faces: "
            f"{[p for p, f in partname2face.items() if f]}); refusing to "
            "export a scaffold whose movable parts have no geometry"
        )

    if verbose:
        print(f"Total faces: {np.sum(label2face_mask)}")

    # Export parts
    all_part_meshes = export_mesh_parts(
        mesh=mesh,
        part_segment_dict=part_segment_dict,
        label2face_mask=label2face_mask,
        output_dir=mesh_parts_dir,
        verbose=verbose
    )
    
    # Create unified mesh
    create_unified_mesh(all_part_meshes, f"{out_dir}/unified_mesh.glb")
    
    print(f"Saved segmentation to: {out_dir}")
