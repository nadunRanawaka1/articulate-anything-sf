"""Mesh part merging and export utilities."""

import os
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
        part_segment_dict = merge_json['part_segment_dict']
        
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
        
        # Save updated mappings to meshes directory
        import json
        
        # Save face2label (per-face segment IDs)
        face2label_path = os.path.join(mesh_parts_dir, "face2label.json")
        with open(face2label_path, 'w') as f:
            json.dump(face2label.tolist(), f)
        
        # Save partname2face (part name -> list of face indices)
        partname2face = {}
        for part_name, segment_ids in part_segment_dict.items():
            part_faces = []
            for seg_id in segment_ids:
                seg_id = int(seg_id)
                if seg_id < label2face_mask.shape[0]:
                    faces = label2face_mask[seg_id].nonzero()[0].tolist()
                    part_faces.extend(faces)
            partname2face[part_name] = sorted(set(part_faces))
        
        partname2face_path = os.path.join(mesh_parts_dir, "partname2face.json")
        with open(partname2face_path, 'w') as f:
            json.dump(partname2face, f, indent=2)
        
        # Also save the part_segment_dict (part name -> segment IDs)
        part_segment_path = os.path.join(mesh_parts_dir, "part_segment_dict.json")
        with open(part_segment_path, 'w') as f:
            json.dump(part_segment_dict, f, indent=2)
        
        if verbose:
            print(f"Saved face2label to: {face2label_path}")
            print(f"Saved partname2face to: {partname2face_path}")
            print(f"Saved part_segment_dict to: {part_segment_path}")
    
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
