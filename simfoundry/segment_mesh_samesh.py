import os
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import trimesh
import torch
import pymeshlab
from scipy.spatial import cKDTree

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from render import (
    render_with_blender, render_with_pyrender_offscreen, generate_camera_matrices, 
    get_scale_multiplier, add_part_labels, load_partitioned_mesh, 
    create_scene_from_parts, render_scene_with_labels, explode_mesh
)

import matplotlib.pyplot as plt
from matplotlib import colormaps
from PIL import Image
from collections import defaultdict


def decimate_mesh(mesh: trimesh.Trimesh, target_faces: int = 20000, aggressive: bool = True, verbose: bool = False):
    """
    Decimate mesh to target face count while preserving structure.
    
    Args:
        mesh: High-resolution input mesh
        target_faces: Target number of faces
        aggressive: If True, disable topology preservation to reach target (default: True)
        verbose: Print progress
    
    Returns:
        decimated_mesh: Lower-resolution mesh
        original_centroids: Centroids of original faces (for label transfer)
        was_decimated: Whether decimation was actually performed
    """
    original_centroids = mesh.triangles_center
    
    if len(mesh.faces) <= target_faces:
        if verbose:
            print(f"Mesh already has {len(mesh.faces)} faces (<= {target_faces}), skipping decimation")
        return mesh, original_centroids, False
    
    if verbose:
        print(f"Decimating mesh: {len(mesh.faces)} → {target_faces} faces")
    
    # Use pymeshlab for quality decimation
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    
    # First try: Conservative decimation (preserves topology)
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        qualitythr=0.5,
        preserveboundary=True,
        preservenormal=True,
        preservetopology=True,
        planarquadric=True
    )
    
    current_faces = ms.current_mesh().face_number()
    
    # If we didn't reach target and aggressive mode is enabled, try again without topology preservation
    if current_faces > target_faces * 1.2 and aggressive:
        if verbose:
            print(f"Conservative decimation reached {current_faces} faces, retrying with aggressive settings...")
        
        # Reset and try again with aggressive settings
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
        
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            qualitythr=0.3,              # Lower quality threshold
            preserveboundary=False,       # Don't preserve boundaries
            preservenormal=True,          # Still preserve normals for visual quality
            preservetopology=False,       # Allow topology changes
            planarquadric=True
        )
    
    decimated = ms.current_mesh()
    decimated_mesh = trimesh.Trimesh(
        vertices=decimated.vertex_matrix(),
        faces=decimated.face_matrix()
    )
    
    if verbose:
        print(f"Decimated: {len(mesh.faces)} → {len(decimated_mesh.faces)} faces "
              f"({100 * len(decimated_mesh.faces) / len(mesh.faces):.1f}%)")
    
    return decimated_mesh, original_centroids, True


def transfer_labels(
    original_mesh: trimesh.Trimesh,
    decimated_mesh: trimesh.Trimesh, 
    decimated_face2label: dict,
    original_centroids: np.ndarray,
    verbose: bool = False
) -> dict:
    """
    Transfer labels from decimated mesh back to original high-res mesh.
    
    Uses nearest-centroid lookup: each original face gets the label of the 
    nearest decimated face (by centroid distance).
    """
    decimated_centroids = decimated_mesh.triangles_center
    
    # Build KD-tree for fast nearest-neighbor lookup
    tree = cKDTree(decimated_centroids)
    
    # Find nearest decimated face for each original face
    _, nearest_decimated_faces = tree.query(original_centroids, k=1)
    
    # Transfer labels
    original_face2label = {}
    for orig_face_idx, dec_face_idx in enumerate(nearest_decimated_faces):
        if dec_face_idx in decimated_face2label:
            original_face2label[orig_face_idx] = decimated_face2label[dec_face_idx]
        else:
            original_face2label[orig_face_idx] = 0  # Background/unlabeled
    
    if verbose:
        unique_labels = set(original_face2label.values())
        print(f"Transferred {len(unique_labels)} unique labels to {len(original_face2label)} faces")
    
    return original_face2label


def hierarchical_merge_segments(
    out_dir: str,
    object_name: str,
    max_segments: int = 15,
    verbose: bool = False
) -> dict:
    """
    Hierarchically merge segments until reaching max_segments.
    
    Uses adjacency-based agglomerative clustering, prioritizing:
    1. Smallest segments (by face count)
    2. Most similar colors (if available)
    3. Adjacent pairs only
    
    Args:
        out_dir: Directory containing face2label.json and segmented mesh
        object_name: Name of the object (for finding the segmented mesh)
        max_segments: Maximum number of segments to keep
        verbose: Print debug info
        
    Returns:
        Updated face2label dict
    """
    import heapq
    
    # Load data
    face2label_path = os.path.join(out_dir, "face2label.json")
    mesh_path = os.path.join(out_dir, f"{object_name}_segmented.glb")
    
    with open(face2label_path, 'r') as f:
        face2label = json.load(f)
    face2label = {int(k): int(v) for k, v in face2label.items()}
    
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    
    # Merge vertices to enable proper face adjacency computation
    # GLB files often have per-face vertices which breaks adjacency detection
    mesh.merge_vertices()
    
    if verbose:
        print(f"  Mesh: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices (after merge)")
        print(f"  Face adjacency pairs: {len(mesh.face_adjacency)}")
    
    # Count current segments
    unique_segments = set(face2label.values())
    current_count = len(unique_segments)
    
    if verbose:
        print(f"Hierarchical merge: {current_count} segments -> target max {max_segments}")
    
    if current_count <= max_segments:
        if verbose:
            print(f"  Already at or below target, no merging needed")
        return face2label
    
    # Build segment info: face count, centroid, avg color
    segment_faces = defaultdict(list)
    for face_idx, seg_id in face2label.items():
        segment_faces[seg_id].append(face_idx)
    
    segment_info = {}
    face_centroids = mesh.triangles_center
    
    # Get face colors if available
    has_colors = hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None
    face_colors = mesh.visual.face_colors[:, :3] if has_colors else None
    
    for seg_id, faces in segment_faces.items():
        faces_arr = np.array(faces)
        segment_info[seg_id] = {
            'face_count': len(faces),
            'centroid': face_centroids[faces_arr].mean(axis=0),
            'color': face_colors[faces_arr].mean(axis=0) if has_colors else np.array([128, 128, 128]),
            'faces': set(faces)
        }
    
    # Build adjacency from mesh face adjacency
    adjacency = defaultdict(set)
    for edge in mesh.face_adjacency:
        seg1 = face2label.get(edge[0])
        seg2 = face2label.get(edge[1])
        if seg1 is not None and seg2 is not None and seg1 != seg2:
            adjacency[seg1].add(seg2)
            adjacency[seg2].add(seg1)
    
    if verbose:
        num_adj_pairs = sum(len(v) for v in adjacency.values()) // 2
        print(f"  Found {num_adj_pairs} adjacent segment pairs across {len(adjacency)} segments")
    
    if len(adjacency) == 0:
        print("  WARNING: No adjacent segments found! Merging will not work.")
        print("  This may happen if segments are topologically disconnected.")
        return face2label
    
    # Union-Find for tracking merges
    parent = {seg: seg for seg in unique_segments}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            # Merge smaller into larger
            if segment_info[px]['face_count'] < segment_info[py]['face_count']:
                px, py = py, px
            parent[py] = px
            # Update segment info
            segment_info[px]['face_count'] += segment_info[py]['face_count']
            segment_info[px]['faces'].update(segment_info[py]['faces'])
            # Update adjacency
            for neighbor in adjacency[py]:
                if find(neighbor) != px:
                    adjacency[px].add(neighbor)
            adjacency.pop(py, None)
            return px
        return px
    
    def merge_cost(seg1, seg2):
        """Lower cost = better merge candidate. Prefer merging two small segments together."""
        info1, info2 = segment_info[seg1], segment_info[seg2]
        size1, size2 = info1['face_count'], info2['face_count']
        
        # Compute total faces in mesh for normalization
        total_faces = len(mesh.faces)
        
        # Prefer merging small segments together, penalize merging into large segments
        # Use the SUM of sizes (not min) - this prefers two 100-face segments (cost=200)
        # over a 100-face + 50000-face pair (cost=50100)
        combined_size = size1 + size2
        
        # Also penalize if one segment is already too large (>20% of mesh)
        max_segment_ratio = max(size1, size2) / total_faces
        large_penalty = 1000000 if max_segment_ratio > 0.3 else 0  # Heavy penalty if >30% of mesh
        
        # Color similarity (lower distance = lower cost)
        color_dist = np.linalg.norm(info1['color'] - info2['color'])
        
        # Combined cost: prefer small+small merges, penalize large segments
        return (large_penalty, combined_size, color_dist)
    
    # Priority queue: (cost, seg1, seg2)
    heap = []
    processed_pairs = set()
    
    def add_pairs_for_segment(seg_id):
        for neighbor in adjacency.get(seg_id, []):
            pair = tuple(sorted([seg_id, neighbor]))
            if pair not in processed_pairs:
                cost = merge_cost(seg_id, neighbor)
                heapq.heappush(heap, (cost, seg_id, neighbor))
    
    # Initialize heap with all adjacent pairs
    for seg_id in unique_segments:
        add_pairs_for_segment(seg_id)
    
    # Merge until we reach target
    merges_done = 0
    while current_count > max_segments and heap:
        cost, seg1, seg2 = heapq.heappop(heap)
        
        # Check if segments still exist (not already merged)
        root1, root2 = find(seg1), find(seg2)
        if root1 == root2:
            continue  # Already merged
        
        # Check if still adjacent
        if root2 not in adjacency.get(root1, set()) and root1 not in adjacency.get(root2, set()):
            continue  # No longer adjacent
        
        # Merge
        new_root = union(root1, root2)
        current_count -= 1
        merges_done += 1
        
        if verbose and merges_done % 10 == 0:
            print(f"  Merged {merges_done} pairs, {current_count} segments remaining")
        
        # Add new pairs for merged segment
        add_pairs_for_segment(new_root)
    
    if verbose:
        print(f"  Completed {merges_done} merges, final segment count: {current_count}")
    
    # Build new face2label with renumbered segments
    root_to_new_id = {}
    new_id = 1
    new_face2label = {}
    
    for face_idx, old_seg in face2label.items():
        root = find(old_seg)
        if root not in root_to_new_id:
            root_to_new_id[root] = new_id
            new_id += 1
        new_face2label[face_idx] = root_to_new_id[root]
    
    # Save updated face2label
    with open(face2label_path, 'w') as f:
        json.dump(new_face2label, f)
    
    if verbose:
        print(f"  Saved merged face2label to: {face2label_path}")
    
    # Re-color the segmented mesh with new segment IDs
    num_segments = len(root_to_new_id)
    cmap = colormaps['tab20'].resampled(max(num_segments, 20))
    
    new_face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    for face_idx, seg_id in new_face2label.items():
        color = np.array(cmap(seg_id % 20)[:3]) * 255
        new_face_colors[face_idx] = [*color.astype(np.uint8), 255]
    
    mesh.visual.face_colors = new_face_colors
    mesh.export(mesh_path)
    
    if verbose:
        print(f"  Re-colored and saved mesh to: {mesh_path}")
    
    return new_face2label


def segment_mesh(cfg: DictConfig, verbose: bool = False):
    """
    Segment mesh using SAMesh.
    
    Runs segmentation in a subprocess to avoid Hydra conflicts.
    Optionally decimates mesh first for speed, then transfers labels back.
    """
    if Path(f"{cfg.out_dir}/{cfg.object_name}_segmented.glb").exists() and not cfg.rerun:
        return
    
    # Resolve samesh config path
    samesh_config_path = cfg.samesh_config_path
    if not os.path.isabs(samesh_config_path):
        script_dir = Path(__file__).parent
        samesh_config_path = script_dir / "cfg" / samesh_config_path
    samesh_config_path = str(Path(samesh_config_path).resolve())
    
    mesh_path = str(Path(cfg.object_path).resolve())
    original_mesh = trimesh.load(mesh_path)
    if type(original_mesh) == trimesh.Scene:
        original_mesh = original_mesh.to_geometry()
    out_dir = str(Path(cfg.out_dir).resolve())
    os.makedirs(out_dir, exist_ok=True)
    
    if verbose:
        print(f"Segmenting mesh with SAMesh: {mesh_path}")
        print(f"Object num faces: {len(original_mesh.faces)}")

    

    segment_mesh_path = mesh_path
   
    
    # Get object name for output naming
    object_name = cfg.object_name
    
    subprocess_script = textwrap.dedent(f'''
        import sys
        from pathlib import Path
        from omegaconf import OmegaConf
        
        # Load config and set output directory
        config = OmegaConf.load("{samesh_config_path}") 
        config.output = "{out_dir}"
        
        # Import and run segmentation
        # SAM2's __init__.py will initialize Hydra in this fresh interpreter
        from samesh.models.sam_mesh import segment_mesh as segment_mesh_samesh
        
        mesh = segment_mesh_samesh(
            "{segment_mesh_path}", 
            config, 
            visualize=True, 
            target_labels=config.target_labels,
            texture=True,  # Keep textures for 'rgb' mode in use_modes
            output_name="{object_name}"  # Use object name instead of mesh filename
        )
    ''')
    
    result = subprocess.run(
        [sys.executable, "-c", subprocess_script],
        capture_output=not verbose,
        text=True
    )
    
    if result.returncode != 0:
        error_msg = result.stderr if result.stderr else "Unknown error"
        raise RuntimeError(f"SAMesh segmentation failed:\n{error_msg}")
    
    # Move contents from subfolder to out_dir (samesh creates a subfolder with the mesh name)
    out_dir_path = Path(out_dir)
    subfolders = [f for f in out_dir_path.iterdir() if f.is_dir()]
    if len(subfolders) == 1:
        subfolder = subfolders[0]
        # Move all contents from subfolder to out_dir
        for item in subfolder.iterdir():
            dest = out_dir_path / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        # Remove the now-empty subfolder
        subfolder.rmdir()
        if verbose:
            print(f"Moved segmentation results from {subfolder.name}/ to {out_dir}")
    
    # Hierarchical merge to reduce over-segmentation
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
    face2label = {int(k): int(v) for k, v in face2label.items()}
    label2face = defaultdict(list)
    for face, label in face2label.items():
        label2face[label].append(face)
    labels = sorted(list(label2face.keys()))
    renumbered_labels = {j: i for i, j in enumerate(labels, start=1)}  # Start from 1 to avoid label 0
    renumbered_face2label = {k: renumbered_labels[v] for k, v in face2label.items()}
    return renumbered_face2label


def render_and_label(cfg: DictConfig, camera_angles, verbose: bool = True):
    if Path(cfg.out_dir + "/rendered_parts").exists() and not cfg.rerun:
        return 
    if verbose:
        print(f"Rendering and labeling parts")

    # Convert face2label to face_ids
    face2label_path = cfg.out_dir + "/face2label.json"
    face2label = json.load(open(face2label_path))
    face2label = renumbered_face2label(face2label)
    face_ids = np.array([face2label[i] for i in range(len(face2label))])
    face_ids = face_ids.astype(np.int32)
    face_ids_path = cfg.out_dir + "/face_ids.npy"
    np.save(face_ids_path, face_ids)


    out_dir = cfg.out_dir + "/rendered_parts"
    os.makedirs(out_dir, exist_ok=True)

    # Use ORIGINAL mesh (not SAMesh-normalized) for rendering
    # This ensures segmented and original_colors renders have identical bounds
    # SAMesh normalizes internally, but we use the original mesh + face2label for rendering
    original_mesh_path = cfg.input_object_path
    
    if verbose:
        print(f"  Loading original mesh for rendering: {original_mesh_path}")
    mesh = trimesh.load(original_mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    
    if verbose:
        print(f"  Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"  Bounds: {mesh.bounds}")
    
    # Verify face count matches face2label
    if len(mesh.faces) != len(face_ids):
        # Face count mismatch - fall back to SAMesh segmented mesh
        if verbose:
            print(f"  Warning: Face count mismatch! Original: {len(mesh.faces)}, face2label: {len(face_ids)}")
            print(f"  Falling back to SAMesh segmented mesh")
        mesh_path = cfg.out_dir + f"/{cfg.object_name}_segmented.glb"
        mesh, face_ids = load_partitioned_mesh(mesh_path, face_ids_path, verbose)
        # In this case, we can't use original_colors since bounds won't match
        original_mesh_path = None
    
    # Create scene with separate parts (this applies distinct part colors based on face_ids)
    scene, part_info, label_mapping = create_scene_from_parts(mesh, face_ids, verbose)

    mapping_path = os.path.join(out_dir, 'label_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)

    label_colors_path = os.path.join(out_dir, 'label_colors.json')
    label_colors = {str(k): [int(c) for c in v['color']] for k, v in part_info.items()}
    with open(label_colors_path, 'w') as f:
        json.dump(label_colors, f, indent=2)

    # Create and save new face_ids array, as well as mask of labels to faces
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

    # Copy original mesh (only if we successfully used the original mesh for rendering)
    if original_mesh_path is not None:
        shutil.copy(original_mesh_path, os.path.join(out_dir, 'original_mesh.glb'))
        print(f"Original mesh saved to: {os.path.join(out_dir, 'original_mesh.glb')}")


    segmented_original_mesh_path = os.path.join(out_dir, 'original_segmented_mesh.glb')
    mesh.export(segmented_original_mesh_path)
    print(f"Segmented original mesh saved to: {segmented_original_mesh_path}")

    use_explode = cfg.get('explode', False)
    explosion_factor = cfg.get('explosion_factor', 0.3)
    
    # Load textured mesh for original_colors rendering
    # We load it here so we can apply the SAME explosion as the segmented mesh
    # This ensures perfect alignment between segmented and original_colors renders
    textured_mesh = None
    if original_mesh_path is not None:
        if verbose:
            print(f"\nLoading textured mesh for original_colors rendering...")
        loaded = trimesh.load(str(original_mesh_path))
        
        # Handle Scene vs Trimesh - apply scene graph transforms
        if isinstance(loaded, trimesh.Scene):
            geom_names = list(loaded.geometry.keys())
            if len(geom_names) == 1:
                geom_name = geom_names[0]
                textured_mesh = loaded.geometry[geom_name].copy()
                # Apply scene graph transform if present
                try:
                    transform, _ = loaded.graph[geom_name]
                    if transform is not None and not np.allclose(transform, np.eye(4)):
                        if verbose:
                            print(f"  Applying scene graph transform to textured mesh")
                        textured_mesh.apply_transform(transform)
                except (KeyError, TypeError, ValueError):
                    # ValueError can occur when trimesh hits iteration limit in complex graphs
                    pass
            else:
                # Multiple geometries - concatenate with transforms
                transformed_geoms = []
                for geom_name in geom_names:
                    geom = loaded.geometry[geom_name].copy()
                    try:
                        transform, _ = loaded.graph[geom_name]
                        if transform is not None and not np.allclose(transform, np.eye(4)):
                            geom.apply_transform(transform)
                    except (KeyError, TypeError, ValueError):
                        # ValueError can occur when trimesh hits iteration limit in complex graphs
                        pass
                    transformed_geoms.append(geom)
                textured_mesh = trimesh.util.concatenate(transformed_geoms)
        else:
            textured_mesh = loaded
        
        if verbose:
            visual_type = type(textured_mesh.visual).__name__
            has_uv = hasattr(textured_mesh.visual, 'uv') and textured_mesh.visual.uv is not None
            print(f"  Textured mesh: {len(textured_mesh.vertices)} vertices, visual: {visual_type}, has_uv: {has_uv}")
            print(f"  Bounds: {textured_mesh.bounds}")
    
    if use_explode and explosion_factor > 0:
        print(f"\nCreating exploded mesh (factor={explosion_factor})...")
        exploded_mesh, exploded_face_ids = explode_mesh(mesh, face_ids, explosion_factor)
        
        exploded_mesh_path = os.path.join(out_dir, 'exploded_mesh.glb')
        exploded_mesh.export(exploded_mesh_path)
        print(f"Exploded mesh saved to: {exploded_mesh_path}")
        
        render_scene, render_part_info, _ = create_scene_from_parts(exploded_mesh, exploded_face_ids, verbose)
        
        # Explode the textured mesh with the SAME parameters
        # This ensures identical vertex positions for both renders
        exploded_textured_mesh = None
        if textured_mesh is not None:
            if verbose:
                print(f"  Exploding textured mesh with same parameters...")
            exploded_textured_mesh, _ = explode_mesh(textured_mesh, face_ids, explosion_factor)
            if verbose:
                print(f"  Exploded textured mesh bounds: {exploded_textured_mesh.bounds}")
        
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
            original_colors_source=exploded_textured_mesh  # Pass the already-exploded mesh
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
            original_colors_source=textured_mesh  # Pass the mesh directly (same coords as segmented)
        )
    
    print(f"Rendered and labeled scene saved to: {out_dir}")

    return scene, part_info, label_mapping