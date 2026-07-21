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

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from render import (
    render_with_blender, render_with_pyrender_offscreen, generate_camera_matrices, 
    get_scale_multiplier, add_part_labels, load_partitioned_mesh, 
    create_scene_from_parts, render_scene_with_labels, explode_mesh
)

import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from collections import defaultdict


def segment_mesh(cfg: DictConfig, verbose: bool = False):
    """
    Segment mesh using PartField.
    
    Steps:
    1. Run PartField to extract per-face features
    2. Cluster features using agglomerative clustering
    3. Save face2label.json and segmented mesh
    
    Runs in a subprocess to avoid Hydra/Lightning conflicts.
    """
    segmented_mesh_path = Path(f"{cfg.out_dir}/{cfg.object_name}_segmented.glb")
    if segmented_mesh_path.exists() and not cfg.rerun:
        return

    partfield_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'deps', 'PartField'))
    config_file = os.path.join(partfield_path, 'configs', 'final', 'demo.yaml')
    # Use the directory from step 1 render as the input directory
    # PartField needs the converted mesh from step 1, not the original
    input_dir = os.path.abspath(os.path.join(cfg.out_dir, '..', 's1_render'))
    # Use absolute path for output since subprocess runs from PartField directory
    out_dir = os.path.abspath(cfg.out_dir)
    
    # Derive object path from step 1 output (GLB format)
    object_path = os.path.join(input_dir, f"{cfg.object_name}.glb")
    mesh = trimesh.load(object_path)
    if type(mesh) == trimesh.Scene:
        mesh = mesh.to_geometry()
    object_uid = cfg.object_name
    
    os.makedirs(out_dir, exist_ok=True)
    
    if verbose:
        print(f"Running PartField segmentation:")
        print(f"  Input dir: {input_dir}")
        print(f"  Object path: {object_path}")
        print(f"Object num faces: {len(mesh.faces)}")
        print(f"  Output: {out_dir}")
    
    # Number of clusters (can be configured)
    num_clusters = cfg.get('num_clusters', 10)
    
    # PartField saves features to exp_results/{result_name}
    result_name = f"articulate_{cfg.object_name}"
    feature_dir = os.path.join(partfield_path, 'exp_results', result_name)
    checkpoint_path = os.path.join(partfield_path, 'model', 'model.ckpt')
    
    # Run PartField feature extraction + clustering in a subprocess
    subprocess_script = textwrap.dedent(f'''
import sys
import os
import numpy as np
import trimesh
import json

sys.path.insert(0, "{partfield_path}")
os.chdir("{partfield_path}")  # PartField uses relative paths

from partfield_inference import predict
from partfield.config import default_argument_parser, setup
from run_part_clustering_with_glb import (
    construct_face_adjacency_matrix_facemst,
    export_colored_mesh_glb
)
from sklearn.cluster import AgglomerativeClustering

# Create argument parser with --opts for config overrides
parser = default_argument_parser()
args = parser.parse_args([
    '-c', '{config_file}',
    '--opts',
    'continue_ckpt', '{checkpoint_path}',
    'result_name', '{result_name}',
    'dataset.data_path', '{input_dir}',
    'feature_output_dir', '{out_dir}',  # Save features directly to our output dir
])

# Setup the full config (merges YAML with command line args)
partfield_cfg = setup(args, freeze=False)

# Run feature extraction
print("Running PartField feature extraction...")
print(f"  Input data path: {{partfield_cfg.dataset.data_path}}")
print(f"  Output dir: {{partfield_cfg.output_dir}}")
predict(partfield_cfg)

# Load extracted features - PartField now saves to our output dir
feature_dir = '{out_dir}'
print(f"Using feature directory: {{feature_dir}}")

uid = '{object_uid}'
feature_path = os.path.join(feature_dir, f'part_feat_{{uid}}_0_batch.npy')
if not os.path.exists(feature_path):
    feature_path = os.path.join(feature_dir, f'part_feat_{{uid}}_0.npy')

if not os.path.exists(feature_path):
    available = os.listdir(feature_dir)
    raise FileNotFoundError(f"PartField features not found. Available files: {{available}}")

print(f"Loading features from {{feature_path}}")
features = np.load(feature_path)
features = features / np.linalg.norm(features, axis=-1, keepdims=True)

mesh = trimesh.load('{object_path}')
if hasattr(mesh, 'geometry'):
    mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

V = np.array(mesh.vertices)
F = np.array(mesh.faces)
print(f"Mesh has {{len(F)}} faces, features shape: {{features.shape}}")


print("Building face adjacency matrix (with MST for disconnected components)...")
adj_matrix = construct_face_adjacency_matrix_facemst(F, V, with_knn=True)

# Run agglomerative clustering
num_clusters = {num_clusters}
print(f"Running agglomerative clustering with {{num_clusters}} clusters...")
clustering = AgglomerativeClustering(
    n_clusters=num_clusters,
    connectivity=adj_matrix
).fit(features)

labels = clustering.labels_

# Create face2label mapping
face2label = {{str(i): int(labels[i]) for i in range(len(labels))}}

# Ensure output directory exists
os.makedirs('{out_dir}', exist_ok=True)

# Save face2label.json
face2label_path = '{out_dir}/face2label.json'
with open(face2label_path, 'w') as f:
    json.dump(face2label, f)
print(f"Saved face2label to {{face2label_path}}")

# Export colored mesh using PartField's utility
segmented_path = '{out_dir}/{cfg.object_name}_segmented.glb'
export_colored_mesh_glb(V, F, labels, filename=segmented_path)
print(f"Saved segmented mesh to {{segmented_path}}")
    ''')
    
    result = subprocess.run(
        [sys.executable, "-c", subprocess_script],
        capture_output=not verbose,
        text=True,
        cwd=partfield_path  # Run from PartField directory for relative paths
    )
    
    if result.returncode != 0:
        error_msg = result.stderr if result.stderr else "Unknown error"
        raise RuntimeError(f"PartField segmentation failed:\n{error_msg}")
    
    if verbose:
        print(f"PartField segmentation complete")


def renumbered_face2label(face2label: dict):
    """Renumber labels to be consecutive starting from 1."""
    face2label = {int(k): int(v) for k, v in face2label.items()}
    label2face = defaultdict(list)
    for face, label in face2label.items():
        label2face[label].append(face)
    labels = sorted(list(label2face.keys()))
    renumbered_labels = {j: i for i, j in enumerate(labels, start=1)}  # Start from 1 to avoid label 0
    renumbered_face2label = {k: renumbered_labels[v] for k, v in face2label.items()}
    return renumbered_face2label


def render_and_label(cfg: DictConfig, camera_angles, verbose: bool = False):
    """
    Render the segmented mesh from multiple angles with part labels.
    """
    if Path(cfg.out_dir + "/rendered_parts").exists() and not cfg.rerun:
        return
    if verbose:
        print(f"Rendering and labeling parts")
    
    mesh_path = cfg.out_dir + f"/{cfg.object_name}_segmented.glb"

    face2label_path = cfg.out_dir + "/face2label.json"
    face2label = json.load(open(face2label_path))
    face2label = renumbered_face2label(face2label)
    face_ids = np.array([face2label[i] for i in range(len(face2label))])
    face_ids = face_ids.astype(np.int32)
    face_ids_path = cfg.out_dir + "/face_ids.npy"
    np.save(face_ids_path, face_ids)

    out_dir = cfg.out_dir + "/rendered_parts"
    os.makedirs(out_dir, exist_ok=True)

    mesh, face_ids = load_partitioned_mesh(mesh_path, face_ids_path, verbose)
    
    # Path to original mesh for rendering with original colors/textures
    original_mesh_path = cfg.input_object_path
    
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

    original_mesh_path = cfg.input_object_path
    shutil.copy(original_mesh_path, os.path.join(out_dir, 'original_mesh.glb'))
    print(f"Original mesh saved to: {original_mesh_path}")

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
        print(f"\nRendering normal views...")
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
