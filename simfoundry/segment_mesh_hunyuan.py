import os
import json
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import trimesh
import torch
import random
from omegaconf import DictConfig
from tqdm import tqdm

from p3sam.demo import AutoMask
from render import (
    render_with_blender, render_with_pyrender_offscreen, generate_camera_matrices, 
    get_scale_multiplier, add_part_labels, load_partitioned_mesh, 
    create_scene_from_parts, render_scene_with_labels, explode_mesh
)

import matplotlib.pyplot as plt
from matplotlib import colormaps
from PIL import Image

# Import shared post-processing functions from samesh module
from segment_mesh_samesh import (
    hierarchical_merge_segments,
)

# Import segment boundary smoothing
from smooth_segments import smooth_segment_boundaries, remove_small_disconnected_regions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def remove_textures_and_colorize(mesh, color=(128, 128, 128), verbose=False):
    """
    Remove textures from a mesh and replace with a solid color.
    
    This can improve segmentation quality by removing texture-based noise
    and forcing the model to rely purely on geometry.
    
    Args:
        mesh: trimesh.Trimesh or trimesh.Scene
        color: RGB tuple (0-255) for the solid color, default is neutral gray
        verbose: Print debug info
        
    Returns:
        trimesh.Trimesh with solid color visual
    """
    # Handle Scene objects
    if isinstance(mesh, trimesh.Scene):
        # Concatenate all geometries into a single mesh
        meshes = []
        for geom_name, geom in mesh.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                # Apply scene graph transform if present
                try:
                    geometry_nodes = mesh.graph.geometry_nodes
                    node_names = geometry_nodes.get(geom_name, [])
                    if node_names:
                        transform, _ = mesh.graph[node_names[0]]
                        if transform is not None and not np.allclose(transform, np.eye(4)):
                            geom = geom.copy()
                            geom.apply_transform(transform)
                except (KeyError, TypeError, ValueError):
                    pass
                meshes.append(geom)
        
        if len(meshes) == 1:
            mesh = meshes[0].copy()
        elif len(meshes) > 1:
            mesh = trimesh.util.concatenate(meshes)
        else:
            raise ValueError("No valid geometries found in scene")
    else:
        mesh = mesh.copy()
    
    # Create solid color visual (per-face coloring)
    color_rgba = np.array([color[0], color[1], color[2], 255], dtype=np.uint8)
    face_colors = np.tile(color_rgba, (len(mesh.faces), 1))
    
    # Replace visual with solid color
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=face_colors)
    
    if verbose:
        print(f"  Removed textures, applied solid color RGB{color}")
    
    return mesh


def segment_mesh(cfg: DictConfig, verbose: bool = False):
    segment_model_path = cfg.segment_model_path
    mesh_path = cfg.object_path
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # Check if already done (look for our standardized output)
    standardized_output = Path(out_dir) / f"{cfg.object_name}_segmented.glb"
    if standardized_output.exists() and not cfg.rerun:
        if verbose:
            print(f"Hunyuan segmentation already complete: {standardized_output}")
        return
    
    # Load model
    auto_mask = AutoMask(ckpt_path=segment_model_path)
    # Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    set_seed(42)
    
    # Option to remove textures and use solid color for segmentation
    # This can improve segmentation by forcing the model to rely on geometry only
    use_solid_color = cfg.get('use_solid_color', False)
    if use_solid_color:
        # Parse color - can be a list [R,G,B] or use default gray
        solid_color = cfg.get('solid_color', [128, 128, 128])
        if isinstance(solid_color, (list, tuple)) and len(solid_color) >= 3:
            solid_color = tuple(solid_color[:3])
        else:
            solid_color = (128, 128, 128)
        
        if verbose:
            print(f"Removing textures and applying solid color RGB{solid_color}")
        mesh = remove_textures_and_colorize(mesh, color=solid_color, verbose=verbose)
    
    if verbose:
        print(f"Segmenting mesh with Hunyuan/P3-SAM")
        print(f"Object num faces: {len(mesh.faces)}")
        print(f"Point sampling: {cfg.get('point_num', 100000)} points, {cfg.get('prompt_num', 400)} prompts")
    
    # Run segmentation
    aabb, face_ids, mesh = auto_mask.predict_aabb(
        mesh,
        save_path=out_dir,
        point_num=cfg.get('point_num', 100000),
        prompt_num=cfg.get('prompt_num', 400),
        prompt_bs=cfg.get('prompt_bs', 8),
        threshold=cfg.get('threshold', 0.95),
        post_process=cfg.get('post_process', True),
        seed=42,
        save_mid_res=True,
        show_info=True,
        clean_mesh_flag=cfg.get('clean_mesh', False),
    )

    print(f"Found {len(aabb)} parts")
    auto_mask.release()
    
    # Optional: Remove small disconnected regions (cleans up isolated face clusters)
    remove_disconnected = cfg.get('remove_disconnected_regions', False)
    if remove_disconnected:
        min_region_ratio = cfg.get('min_region_ratio', 0.1)  # Regions < 10% of largest get reassigned
        min_region_faces = cfg.get('min_region_faces', 50)   # But keep if >= 50 faces
        print(f"\nRemoving small disconnected regions...")
        face_ids = remove_small_disconnected_regions(
            mesh, face_ids,
            min_region_ratio=min_region_ratio,
            min_region_faces=min_region_faces,
            verbose=verbose
        )
    
    # Optional: Smooth segment boundaries to reduce jagged edges
    smooth_iterations = cfg.get('smooth_iterations', 0)
    if smooth_iterations > 0:
        print(f"\nSmoothing segment boundaries ({smooth_iterations} iterations)...")
        face_ids = smooth_segment_boundaries(
            mesh, face_ids, 
            iterations=smooth_iterations,
            boundary_only=True,
            verbose=verbose
        )
        print(f"Smoothing complete")
    
    # IMPORTANT: P3-SAM's clean_mesh removes some faces, so the returned mesh
    # may have fewer faces than the original. Save the cleaned mesh with original
    # textures BEFORE applying segment colors for use in original_colors rendering.
    cleaned_mesh_path = Path(out_dir) / f"{cfg.object_name}_cleaned.glb"
    
    # Reload the original mesh to get textures, then keep only the faces that P3-SAM kept
    try:
        # Load without force='mesh' to preserve textures
        original_with_textures = trimesh.load(mesh_path)
        
        if verbose:
            print(f"Original mesh type: {type(original_with_textures)}")
            if hasattr(original_with_textures, 'visual'):
                print(f"Original mesh visual type: {type(original_with_textures.visual)}")
        
        if isinstance(original_with_textures, trimesh.Scene):
            # For scenes, just save the original directly (preserves textures better)
            original_mesh_for_colors = original_with_textures
        else:
            original_mesh_for_colors = original_with_textures
        
        # Get face count for comparison
        if isinstance(original_mesh_for_colors, trimesh.Scene):
            original_face_count = sum(len(g.faces) for g in original_mesh_for_colors.geometry.values())
        else:
            original_face_count = len(original_mesh_for_colors.faces)
        
        # Check if face counts match (they should if clean_mesh=False)
        if original_face_count == len(mesh.faces):
            # Same face count - save the textured version directly
            original_mesh_for_colors.export(str(cleaned_mesh_path))
            if verbose:
                print(f"Saved cleaned mesh with textures (same faces): {cleaned_mesh_path}")
        elif original_face_count > len(mesh.faces):
            # P3-SAM removed some faces - textures won't match
            # Save the original anyway, and we'll handle the mismatch in render_and_label
            original_mesh_for_colors.export(str(cleaned_mesh_path))
            if verbose:
                print(f"Warning: P3-SAM removed {original_face_count - len(mesh.faces)} faces")
                print(f"Saved original mesh with textures (may have extra faces): {cleaned_mesh_path}")
        else:
            # Unexpected - original has fewer faces?
            mesh_copy = mesh.copy()
            mesh_copy.export(str(cleaned_mesh_path))
            if verbose:
                print(f"Warning: Unexpected face count mismatch, using P3-SAM mesh")
    except Exception as e:
        # Fallback: just save the mesh from P3-SAM
        mesh_copy = mesh.copy()
        mesh_copy.export(str(cleaned_mesh_path))
        if verbose:
            print(f"Warning: Could not load textured mesh: {e}")
    
    # Convert to standardized format (same as SAMesh) for post-processing compatibility
    # Create face2label.json from face_ids array
    face2label = {int(i): int(face_ids[i]) for i in range(len(face_ids))}
    face2label_path = os.path.join(out_dir, "face2label.json")
    with open(face2label_path, 'w') as f:
        json.dump(face2label, f)
    if verbose:
        print(f"Created face2label.json: {face2label_path}")
    
    # Save segmented mesh with standardized name
    # Apply colors based on face_ids for visualization
    num_segments = len(np.unique(face_ids))
    cmap = colormaps['tab20'].resampled(max(num_segments, 20))
    face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    for face_idx, seg_id in enumerate(face_ids):
        if seg_id >= 0:
            color = np.array(cmap(seg_id % 20)[:3]) * 255
            face_colors[face_idx] = [*color.astype(np.uint8), 255]
    mesh.visual.face_colors = face_colors
    mesh.export(str(standardized_output))
    if verbose:
        print(f"Saved standardized segmented mesh: {standardized_output}")
    
    # Apply hierarchical merge to reduce over-segmentation
    max_segments = cfg.get('max_segments', None)
    if max_segments is not None and max_segments > 0:
        if verbose:
            print(f"\nApplying hierarchical merge (max_segments={max_segments})...")
        hierarchical_merge_segments(
            out_dir=out_dir,
            object_name=cfg.object_name,
            max_segments=max_segments,
            verbose=verbose
        )
    




def renumbered_face2label(face2label: dict):
    """Renumber labels to be consecutive starting from 1."""
    face2label = {int(k): int(v) for k, v in face2label.items()}
    label2face = defaultdict(list)
    for face, label in face2label.items():
        label2face[label].append(face)
    labels = sorted(list(label2face.keys()))
    renumbered_labels = {j: i for i, j in enumerate(labels, start=1)}
    renumbered_face2label = {k: renumbered_labels[v] for k, v in face2label.items()}
    return renumbered_face2label


def render_and_label(cfg: DictConfig, camera_angles, verbose: bool = False):
    if Path(cfg.out_dir + "/rendered_parts").exists() and not cfg.rerun:
        return
    if verbose:
        print(f"Rendering and labeling parts")
    
    out_dir = cfg.out_dir + "/rendered_parts"
    os.makedirs(out_dir, exist_ok=True)
    
    # Use standardized format (face2label.json) if available
    face2label_path = os.path.join(cfg.out_dir, "face2label.json")
    standardized_mesh_path = os.path.join(cfg.out_dir, f"{cfg.object_name}_segmented.glb")
    
    if os.path.exists(face2label_path) and os.path.exists(standardized_mesh_path):
        # Use standardized format (same as SAMesh)
        if verbose:
            print(f"  Using standardized format: {face2label_path}")
        
        with open(face2label_path, 'r') as f:
            face2label = json.load(f)
        face2label = renumbered_face2label(face2label)
        face_ids = np.array([face2label[i] for i in range(len(face2label))])
        face_ids = face_ids.astype(np.int32)
        
        # Save as face_ids.npy for compatibility
        face_ids_path = os.path.join(cfg.out_dir, "face_ids.npy")
        np.save(face_ids_path, face_ids)
        
        mesh = trimesh.load(standardized_mesh_path, force='mesh')
    else:
        # Fallback to legacy format
        if verbose:
            print(f"  Using legacy Hunyuan format")
        if 'auto_mask_mesh_final_post.glb' in os.listdir(cfg.out_dir):
            mesh_path = cfg.out_dir + "/auto_mask_mesh_final_post.glb"
            face_ids_path = cfg.out_dir + "/auto_mask_mesh_final_post_face_ids.npy"
        elif 'auto_mask_mesh_final.glb' in os.listdir(cfg.out_dir):
            mesh_path = cfg.out_dir + "/auto_mask_mesh_final.glb"
            face_ids_path = cfg.out_dir + "/auto_mask_mesh_final_face_ids.npy"
        else:
            raise FileNotFoundError(f"No segmented mesh found in {cfg.out_dir}. Please run segmentation first.")
        
        mesh, face_ids = load_partitioned_mesh(mesh_path, face_ids_path, verbose)
    
    # Path to cleaned mesh for rendering with original colors/textures
    # Use the cleaned mesh (same face count as segmented) to avoid index mismatches
    cleaned_mesh_path = os.path.join(cfg.out_dir, f"{cfg.object_name}_cleaned.glb")
    if os.path.exists(cleaned_mesh_path):
        original_mesh_path = cleaned_mesh_path
        if verbose:
            print(f"  Using cleaned mesh for original colors: {cleaned_mesh_path}")
    else:
        # Fallback to original (may cause issues if face counts differ)
        original_mesh_path = cfg.input_object_path
        if verbose:
            print(f"  Warning: Cleaned mesh not found, using original: {original_mesh_path}")
    
    # Create scene with separate parts (this overwrites colors with distinct part colors)
    scene, part_info, label_mapping = create_scene_from_parts(mesh, face_ids, verbose)

    mapping_path = os.path.join(out_dir, 'label_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)

    label_colors_path = os.path.join(out_dir, 'label_colors.json')
    label_colors = {str(k): [int(c) for c in v['color']] for k, v in part_info.items()}
    with open(label_colors_path, 'w') as f:
        json.dump(label_colors, f, indent=2)

    label_to_faces = np.zeros((len(label_mapping), len(face_ids)), dtype=np.int32)
    for old_id, new_id in label_mapping.items():
        label_to_faces[new_id, face_ids == old_id] = 1
        face_ids[face_ids == old_id] = new_id
    
    label_to_faces_path = os.path.join(out_dir, 'label_to_faces_mask.npz')
    np.savez_compressed(label_to_faces_path, label_to_faces)
    print(f"Label to faces mapping saved to: {label_to_faces_path}")
    
    np.save(os.path.join(out_dir, 'face_ids.npy'), face_ids)
    print(f"New face_ids saved to: {os.path.join(out_dir, 'face_ids.npy')}")

    # Export individual segments as separate mesh files
    segments_dir = os.path.join(out_dir, 'segments')
    os.makedirs(segments_dir, exist_ok=True)
    print(f"\nExporting individual segments to: {segments_dir}")
    
    unique_segments = sorted(np.unique(face_ids))
    for seg_id in unique_segments:
        if seg_id < 0:  # Skip invalid/unassigned segments
            continue
        
        # Get faces belonging to this segment
        seg_face_mask = face_ids == seg_id
        seg_face_indices = np.where(seg_face_mask)[0]
        
        if len(seg_face_indices) == 0:
            continue
        
        # Create submesh for this segment
        seg_mesh = mesh.submesh([seg_face_indices], append=True)
        
        # Apply segment color for visualization
        if seg_id in part_info:
            seg_color = part_info[seg_id]['color']
            seg_mesh.visual.face_colors = np.tile(
                np.array([*seg_color, 255], dtype=np.uint8),
                (len(seg_mesh.faces), 1)
            )
        
        # Export as GLB
        seg_path = os.path.join(segments_dir, f'segment_{seg_id}.glb')
        seg_mesh.export(seg_path)
        
        if verbose:
            print(f"  Segment {seg_id}: {len(seg_face_indices)} faces -> {seg_path}")
    
    print(f"Exported {len(unique_segments)} segments to: {segments_dir}")

    # Save a copy of the original input mesh (for reference, not for rendering)
    input_mesh_path = cfg.input_object_path
    shutil.copy(input_mesh_path, os.path.join(out_dir, 'original_mesh.glb'))
    print(f"Original mesh saved to: {input_mesh_path}")

    segmented_original_mesh_path = os.path.join(out_dir, 'original_segmented_mesh.glb')
    mesh.export(segmented_original_mesh_path)
    print(f"Segmented original mesh saved to: {segmented_original_mesh_path}")

    use_explode = cfg.get('explode', False)
    explosion_factor = cfg.get('explosion_factor', 0.3)
    
    if use_explode and explosion_factor > 0:
        print(f"\nCreating exploded mesh (factor={explosion_factor})...")
        exploded_mesh, exploded_face_ids = explode_mesh(mesh, face_ids, explosion_factor)
        
        exploded_mesh_path = os.path.join(out_dir, 'exploded_mesh.glb')
        exploded_mesh.export(exploded_mesh_path)
        print(f"Exploded mesh saved to: {exploded_mesh_path}")
        
        render_scene, render_part_info, _ = create_scene_from_parts(exploded_mesh, exploded_face_ids, verbose)
        
        print(f"\nRendering exploded views...")
        render_scene_with_labels(
            render_scene, 
            render_part_info, 
            camera_angles,
            out_dir,
            resolution=tuple(cfg.resolution),
            renderer=cfg.renderer,
            camera_mode=cfg.camera_mode,
            num_views=cfg.num_views,
            verbose=verbose,
            flat_shading=cfg.get('flat_shading', True),
            original_colors_source=original_mesh_path,
            original_colors_explosion_factor=explosion_factor,
            original_colors_face_ids=face_ids
        )
    else:
        print(f"\nRendering views...")
        render_scene_with_labels(
            scene, 
            part_info, 
            camera_angles,
            out_dir,
            resolution=tuple(cfg.resolution),
            renderer=cfg.renderer,
            camera_mode=cfg.camera_mode,
            num_views=cfg.num_views,
            verbose=verbose,
            flat_shading=cfg.get('flat_shading', True),
            original_colors_source=original_mesh_path
        )
    
    print(f"Rendered and labeled scene saved to: {out_dir}")

    return scene, part_info, label_mapping