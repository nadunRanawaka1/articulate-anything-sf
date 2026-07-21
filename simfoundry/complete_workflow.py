"""
Complete workflow to articulate objects.
Step 1: render the object
Step 2: recognize parts and generate articulation tree
Step 3: segment the object with Hunyuan 3D-part
Step 4: merge segmented parts and generate dummy URDF
Step 5: articulate the object

# TODO: we don't need fixed part, it is always the base.
# TODO: change filepaths in final generated urdf to be relative to the object directory.
# TODO: massive cleanup needed in the codebase.
# TODO: add co-tracker to all the joints, maybe pass in multiple videos to critic vlm
# TODO: add top split to all segmentation methods
"""
import hydra
import json
from omegaconf import DictConfig, OmegaConf
from render import render_object
from query_vlm import recognize_parts, generate_articulation_tree
from pathlib import Path
import os

# from segment_mesh import segment_mesh, render_and_label
from postprocess_segmentation import merge_and_center_segmented_mesh
from generate_urdf import generate_base_urdf
from articulate_simfoundry import articulate_simfoundry
import traceback
import time
from datetime import timedelta


def repair_mesh_for_sdf(mesh_path: str, output_path: str = None, verbose: bool = False) -> str:
    """
    Repair a mesh to make it watertight and suitable for SDF computation.
    
    This is particularly important for meshes from TRELLIS which may have:
    - Inconsistent normals
    - Non-watertight geometry
    - Double-sided faces
    
    Args:
        mesh_path: Path to input mesh file
        output_path: Path to save repaired mesh (if None, overwrites input)
        verbose: Print debug info
        
    Returns:
        Path to the repaired mesh
    """
    import trimesh
    import numpy as np
    
    if output_path is None:
        # Create a repaired version alongside the original
        base, ext = os.path.splitext(mesh_path)
        output_path = f"{base}_repaired{ext}"
    
    if verbose:
        print(f"  Repairing mesh: {mesh_path}")
    
    # Load the mesh
    mesh = trimesh.load(mesh_path, force='mesh', validate=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    
    if verbose:
        print(f"    Original: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"    Is watertight: {mesh.is_watertight}")
    
    # Step 1: Remove duplicate vertices and degenerate faces
    # mesh.remove_duplicate_faces()
    # mesh.remove_degenerate_faces()
    mesh.merge_vertices()
    
    # Step 2: Fix normals to point outward consistently
    trimesh.repair.fix_normals(mesh, multibody=True)
    
    # Step 3: Fill holes to make watertight (if possible)
    trimesh.repair.fill_holes(mesh)
    
    # Step 4: Fix winding order
    trimesh.repair.fix_winding(mesh)
    
    if verbose:
        print(f"    Repaired: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"    Is watertight: {mesh.is_watertight}")
    
    # Export the repaired mesh
    mesh.export(output_path)
    
    if verbose:
        print(f"    Saved repaired mesh to: {output_path}")
    
    return output_path


def make_watertight_marching_cubes(mesh_path: str, output_path: str = None, 
                                    grid_res: int = 256, epsilon: float = 0.008,
                                    target_faces: int = 100000,
                                    verbose: bool = False) -> str:
    """
    Convert a thin-shell mesh to a watertight solid using Marching Cubes.
    
    This technique is from Hunyuan3D and works by:
    1. Computing SDF on a 3D grid using libigl's signed_distance with pseudo-normals
    2. Running Marching Cubes to extract an epsilon-thick watertight surface
    
    This is essential for TRELLIS meshes which are thin shells - it adds
    "thickness" so that Shape Diameter Function rays have geometry to hit.
    
    Args:
        mesh_path: Path to input mesh file
        output_path: Path to save watertight mesh (if None, creates alongside original)
        grid_res: Resolution of the SDF grid (higher = more detail, slower)
        epsilon: Thickness of the shell (fraction of bounding box)
        verbose: Print debug info
        
    Returns:
        Path to the watertight mesh
    """
    import trimesh
    import numpy as np
    
    try:
        import igl
    except ImportError:
        raise ImportError(
            "libigl is required for watertight conversion. "
            "Install with: conda install -c conda-forge igl"
        )
    
    if output_path is None:
        base, ext = os.path.splitext(mesh_path)
        output_path = f"{base}_watertight{ext}"
    
    if verbose:
        print(f"  Making mesh watertight via Marching Cubes: {mesh_path}")
    
    # Load the mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    
    # Convert to proper numpy arrays with correct dtypes for igl
    # igl.signed_distance requires: float64 for vertices/points, int64 for faces
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    
    if verbose:
        print(f"    Original: {len(V)} vertices, {len(F)} faces")
        print(f"    Is watertight: {mesh.is_watertight}")
    
    # Normalize to unit box for consistent epsilon
    V_min = V.min(axis=0)
    V_max = V.max(axis=0)
    original_scale = (V_max - V_min).max()
    original_center = (V_min + V_max) / 2
    V_normalized = np.asarray((V - original_center) / original_scale, dtype=np.float64)
    
    # Compute bounding box with padding
    min_corner = V_normalized.min(axis=0)
    max_corner = V_normalized.max(axis=0)
    padding = 0.05 * (max_corner - min_corner)
    min_corner -= padding
    max_corner += padding
    
    # Create a uniform grid
    x = np.linspace(min_corner[0], max_corner[0], grid_res)
    y = np.linspace(min_corner[1], max_corner[1], grid_res)
    z = np.linspace(min_corner[2], max_corner[2], grid_res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.asarray(np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T, dtype=np.float64)
    
    if verbose:
        print(f"    Computing SDF on {grid_res}^3 grid ({len(grid_points)} points)...")
    
    # Compute SDF at grid points using pseudo-normals (handles non-watertight meshes)
    # igl.signed_distance returns: (distances, face_indices, closest_points, normals)
    try:
        sdf, _, _, _ = igl.signed_distance(
            grid_points, 
            V_normalized, 
            F,
            sign_type=igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
        )
    except Exception as e:
        if verbose:
            print(f"    Warning: Pseudo-normal SDF failed ({e}), trying fast winding number...")
        sdf, _, _, _ = igl.signed_distance(
            grid_points, 
            V_normalized, 
            F,
            sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
        )
    
    if verbose:
        print(f"    Running Marching Cubes with epsilon={epsilon}...")
    
    # Run Marching Cubes on the epsilon-thickened surface
    # epsilon - |sdf| gives a shell of thickness 2*epsilon
    # igl.marching_cubes returns (vertices, faces, edge_to_vertex_map)
    mc_verts, mc_faces, _ = igl.marching_cubes(
        epsilon - np.abs(sdf), 
        grid_points, 
        grid_res, grid_res, grid_res, 
        0.0
    )
    
    # Transform back to original scale
    mc_verts = mc_verts * original_scale + original_center
    
    # Create new mesh
    watertight_mesh = trimesh.Trimesh(vertices=mc_verts, faces=mc_faces, process=False)
    watertight_mesh = watertight_mesh.simplify_quadric_decimation(face_count=target_faces)
    
    if verbose:
        print(f"    Watertight: {len(mc_verts)} vertices, {len(mc_faces)} faces")
        print(f"    Is watertight: {watertight_mesh.is_watertight}")
    
    # Export
    watertight_mesh.export(output_path)
    
    if verbose:
        print(f"    Saved watertight mesh to: {output_path}")
    
    return output_path


CFG_DIR = "cfg"


def copy_articulation_tree(reference_name: str, target_name: str, reference_dir: str, target_dir: str, verbose: bool = False):
    """
    Copy articulation tree files from a reference object to a target object,
    replacing the object name in the JSON content.
    
    This is useful when processing multiple similar objects (e.g., same type of trash can)
    to avoid redundant VLM queries.
    
    Args:
        reference_name: Name of the reference object (e.g., 'trash_can_cousin_003_v3')
        target_name: Name of the target object (e.g., 'trash_can_cousin_005_v5')
        reference_dir: Path to reference object's s2 output directory
        target_dir: Path to target object's s2 output directory
        verbose: Print debug info
        
    Returns:
        Tuple of (parts_dict, articulation_tree_dict) loaded from copied files
    """
    os.makedirs(target_dir, exist_ok=True)
    
    files_to_copy = [
        'result_recognize_parts.json',
        'result_generate_articulation_tree.json'
    ]
    
    if verbose:
        print(f"Copying articulation tree from '{reference_name}' to '{target_name}'")
    
    for filename in files_to_copy:
        src_path = os.path.join(reference_dir, filename)
        dst_path = os.path.join(target_dir, filename)
        
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Reference file not found: {src_path}")
        
        # Read the JSON content
        with open(src_path, 'r') as f:
            content = f.read()
        
        # Replace the reference object name with the target object name
        modified_content = content.replace(reference_name, target_name)
        
        # Write to target
        with open(dst_path, 'w') as f:
            f.write(modified_content)
        
        if verbose:
            print(f"  Copied and modified: {filename}")
    
    # Load and return the copied files
    with open(os.path.join(target_dir, 'result_recognize_parts.json'), 'r') as f:
        parts_dict = json.load(f)
    
    with open(os.path.join(target_dir, 'result_generate_articulation_tree.json'), 'r') as f:
        articulation_tree_dict = json.load(f)
    
    return parts_dict, articulation_tree_dict


def process_single_object(cfg, object_config, reference_object_name: str = None):
    """
    Process a single object through the complete pipeline.
    
    Args:
        cfg: Main configuration
        object_config: Dict with 'name', 'mesh_path', 'image_path'
        reference_object_name: If provided, copy articulation tree from this object
                              instead of querying VLM (for similar objects)
        
    Returns:
        Dict with timing information for each step
    """
    object_name = object_config['name']
    object_start_time = time.time()
    step_timings = {}
    
    if cfg.verbose:
        print(f"\n{'='*80}")
        print(f"Processing object: {object_name}")
        print(f"{'='*80}")
    
    # Update paths to include object subdirectory
    # Structure: root_dir/scene_name/object_name/step_name/
    object_root = f"{cfg.root_dir}/{cfg.scene_name}/{object_name}"
    
    # =========== Step 1: Render the object ===========
    step_start = time.time()
    s1_cfg = OmegaConf.create(cfg.s1_render)
    s1_cfg.out_dir = f"{object_root}/{s1_cfg.out_dirname}"
    s1_cfg.input_object_path = object_config['mesh_path'] # This is the path to the original object mesh
    s1_cfg.object_name = object_name
    s1_cfg.object_path = f"{s1_cfg.out_dir}/{object_name}.glb" # GLB format preserves PBR materials for rendering
    s1_cfg.object_image_path = object_config['image_path']
    if Path(f"{s1_cfg.out_dir}/render_view00.png").exists() and not s1_cfg.rerun:
        step_timings['Step 1: Render'] = 0.0  # Skipped
    else:
        render_object(s1_cfg, verbose=cfg.verbose)
        step_timings['Step 1: Render'] = time.time() - step_start

    # =========== Step 2: Recognize the parts and generate the articulation tree ===========
    step_start = time.time()
    s2_cfg = OmegaConf.create(cfg.s2_generate_articulation_tree)
    s2_cfg.out_dir = f"{object_root}/{s2_cfg.out_dirname}"
    s2_cfg.image_dir = s1_cfg.out_dir
    s2_cfg.object_name = object_name
    s2_cfg.gcloud_project = cfg.gcloud_project
    s2_cfg.gcloud_location = cfg.gcloud_location
    if Path(f"{s2_cfg.out_dir}/result_recognize_parts.json").exists() and Path(f"{s2_cfg.out_dir}/result_generate_articulation_tree.json").exists() and not s2_cfg.rerun:
        with open(f"{s2_cfg.out_dir}/result_recognize_parts.json", 'r') as f:
            parts_dict = json.load(f)
        with open(f"{s2_cfg.out_dir}/result_generate_articulation_tree.json", 'r') as f:
            articulation_tree_dict = json.load(f)
        step_timings['Step 2: Recognize & Tree'] = 0.0  # Skipped
    else:
        if reference_object_name and reference_object_name != object_name:
            # Copy articulation tree from reference object instead of querying VLM
            reference_s2_dir = f"{cfg.root_dir}/{cfg.scene_name}/{reference_object_name}/{s2_cfg.out_dirname}"
            print(f"Running Step 2: Copy articulation tree from '{reference_object_name}'")
            parts_dict, articulation_tree_dict = copy_articulation_tree(
                reference_name=reference_object_name,
                target_name=object_name,
                reference_dir=reference_s2_dir,
                target_dir=s2_cfg.out_dir,
                verbose=cfg.verbose
            )
        else:
            # Normal flow: query VLM
            print(f"Running Step 2: Recognize parts and generate articulation tree")
            parts_dict = recognize_parts(s2_cfg, verbose=cfg.verbose)  
            articulation_tree_dict = generate_articulation_tree(s2_cfg, parts_dict, verbose=cfg.verbose)
        step_timings['Step 2: Recognize & Tree'] = time.time() - step_start

    # =========== Step 3: Segment the mesh ===========
    step_start = time.time()
    s3_cfg = OmegaConf.create(cfg.s3_segment_mesh)
    s3_cfg.out_dir = f"{object_root}/{s3_cfg.out_dirname}"
    if Path(f"{s3_cfg.out_dir}/rendered_parts/label_mapping.json").exists() and not s3_cfg.rerun:
        step_timings['Step 3: Segment'] = 0.0  # Skipped
    else:
        # Determine the mesh path to use for segmentation
        mesh_path_for_segmentation = object_config['mesh_path']
        
        # Optional mesh preprocessing for TRELLIS meshes which may have SDF issues
        # Options:
        #   repair_mesh: true  - Basic trimesh repair (fix normals, fill holes)
        #   make_watertight: true - Full watertight conversion via Marching Cubes (from Hunyuan3D)
        #                           This adds thickness to thin-shell meshes so SDF rays hit geometry
        os.makedirs(s3_cfg.out_dir, exist_ok=True)
        
        if s3_cfg.get('make_watertight', False):
            print(f"Running Step 3a: Make mesh watertight via Marching Cubes")
            watertight_mesh_path = os.path.join(s3_cfg.out_dir, f"{object_name}_watertight.glb")
            mesh_path_for_segmentation = make_watertight_marching_cubes(
                object_config['mesh_path'], 
                watertight_mesh_path,
                grid_res=s3_cfg.get('watertight_grid_res', 256),
                epsilon=s3_cfg.get('watertight_epsilon', 0.008),
                target_faces=s3_cfg.get('target_faces', 100000),
                verbose=cfg.verbose
            )
        elif s3_cfg.get('repair_mesh', False):
            print(f"Running Step 3a: Repair mesh for SDF computation")
            repaired_mesh_path = os.path.join(s3_cfg.out_dir, f"{object_name}_repaired.glb")
            mesh_path_for_segmentation = repair_mesh_for_sdf(
                object_config['mesh_path'], 
                repaired_mesh_path, 
                verbose=cfg.verbose
            )
        
        s3_cfg.object_path = mesh_path_for_segmentation
        s3_cfg.object_name = object_name
        s3_cfg.input_object_path = mesh_path_for_segmentation

        if s3_cfg.camera_mode == 'angles':
            camera_angles = [[az, el] for az in s3_cfg.camera_angles for el in s3_cfg.camera_angles]
        else:
            camera_angles = None

        # Hunyuan3D-Part is the default segmentation backend; samesh and partfield are optional.
        segment_method = s3_cfg.get('segment_method', 'hunyuan')
        if segment_method == 'hunyuan':
            from segment_mesh_hunyuan import segment_mesh, render_and_label
        elif segment_method == 'samesh':
            from segment_mesh_samesh import segment_mesh, render_and_label
        elif segment_method == 'partfield':
            from segment_mesh_partfield import segment_mesh, render_and_label
        else:
            raise ValueError(f"Invalid segment method: {segment_method}")

        print(f"Running Step 3: Segment the mesh")
        segment_mesh(s3_cfg, verbose=cfg.verbose)
        render_and_label(s3_cfg, camera_angles, verbose=cfg.verbose)

        step_timings['Step 3: Segment'] = time.time() - step_start

    # =========== Step 4: Merge the segmented parts and generate dummy URDF ===========
    step_start = time.time()
    s4_cfg = OmegaConf.create(cfg.s4_merge_mesh_parts)
    s4_cfg.out_dir = f"{object_root}/{s4_cfg.out_dirname}"
    s4_cfg.image_dir = f"{s3_cfg.out_dir}/rendered_parts"
    s4_cfg.object_name = object_name
    s4_cfg.object_path = s1_cfg.object_path
    s4_cfg.gcloud_project = cfg.gcloud_project
    s4_cfg.gcloud_location = cfg.gcloud_location
    s4_cfg.mesh_parts_dir = f"{s4_cfg.out_dir}/meshes"
    if Path(f"{s4_cfg.out_dir}/dummy.urdf").exists() and not s4_cfg.rerun:
        urdf_path = f"{s4_cfg.out_dir}/dummy.urdf"
        step_timings['Step 4: Merge & URDF'] = 0.0  # Skipped
    else:
        print(f"Running Step 4: Merge the segmented parts")
        merge_and_center_segmented_mesh(
            s4_cfg, 
            articulation_tree_dict, 
            verbose=cfg.verbose,
            interactive=s4_cfg.interactive_correction
        )
        urdf_path = generate_base_urdf(s4_cfg, articulation_tree_dict, verbose=cfg.verbose)
        step_timings['Step 4: Merge & URDF'] = time.time() - step_start


    # =========== Step 5: Articulate the object ===========
    step_start = time.time()
    s5_cfg = OmegaConf.create(cfg.s5_articulate)
    s5_cfg.out_dir = f"{object_root}/{s5_cfg.out_dirname}"
    s5_cfg.object_name = object_name
    s5_cfg.dummy_urdf_path = urdf_path
    s5_cfg.object_image_path = object_config['image_path']
    s5_cfg.mesh_parts_dir = f"{s4_cfg.out_dir}/meshes"
    
    if Path(f"{s5_cfg.out_dir}/{s5_cfg.object_name}/mobility_final.urdf").exists() and not s5_cfg.rerun:
        step_timings['Step 5: Articulate'] = 0.0  # Skipped
    else:
        print(f"Running Step 5: Articulate the object")
        articulate_simfoundry(s5_cfg, articulation_tree_dict, verbose=cfg.verbose)
        step_timings['Step 5: Articulate'] = time.time() - step_start
    
    object_total_time = time.time() - object_start_time
    
    # Print timing summary for this object
    print(f"\n{'='*80}")
    print(f"Completed: {object_name}")
    print(f"{'='*80}")
    print(f"Timing Breakdown:")
    for step_name, duration in step_timings.items():
        print(f"  {step_name:<25} {str(timedelta(seconds=int(duration)))}")
    print(f"  {'─'*50}")
    print(f"  {'Total':<25} {str(timedelta(seconds=int(object_total_time)))}")
    print(f"{'='*80}\n")
    
    return {
        'object_name': object_name,
        'steps': step_timings,
        'total_time': object_total_time
    }


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    """
    Main entry point - processes single or multiple objects.
    """
    pipeline_start_time = time.time()
    all_results = []
    
    # Check if processing multiple objects or single object
    if 'objects' in cfg and cfg.objects:
        # Multi-object mode
        if cfg.verbose:
            print(f"\n{'='*80}")
            print(f"Multi-object mode: Processing {len(cfg.objects)} objects in scene '{cfg.scene_name}'")
            print(f"{'='*80}\n")
        
        # Check if we should copy articulation tree from the first object
        copy_articulation_from_first = cfg.get('copy_articulation_from_first', False)
        first_object_name = cfg.objects[0]['name'] if copy_articulation_from_first else None
        
        if copy_articulation_from_first and cfg.verbose:
            print(f"Note: Will copy articulation tree from first object '{first_object_name}' to all others\n")
        
        for idx, obj_config in enumerate(cfg.objects, 1):
            if cfg.verbose:
                print(f"[{idx}/{len(cfg.objects)}] Starting: {obj_config['name']}")
            
            # For first object, don't pass reference (run VLM normally)
            # For subsequent objects, pass first object as reference if enabled
            reference_object = None
            if copy_articulation_from_first and idx > 1:
                reference_object = first_object_name
            
            try:
                result = process_single_object(cfg, obj_config, reference_object_name=reference_object)
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {obj_config['name']}: {e}")
                if cfg.verbose:
                    print(traceback.format_exc())
                all_results.append({
                    'object_name': obj_config['name'],
                    'error': str(e),
                    'total_time': 0
                })
        
        # Print final summary
        pipeline_total_time = time.time() - pipeline_start_time
        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"Scene: {cfg.scene_name}")
        print(f"Objects processed: {len(all_results)}")
        print(f"\nPer-Object Summary:")
        for result in all_results:
            status = "Success" if 'error' not in result else "Failure"
            duration = str(timedelta(seconds=int(result['total_time'])))
            print(f"  {status} {result['object_name']:<20} {duration}")
        print(f"\n{'─'*80}")
        print(f"Total Pipeline Time: {str(timedelta(seconds=int(pipeline_total_time)))}")
        print(f"{'='*80}\n")
    else:
        # Single object mode
        if cfg.verbose:
            print("Single object mode")
        
        # Create object config from existing settings
        object_config = {
            'name': cfg.s1_render.get('object_name', 'object'),
            'mesh_path': cfg.s1_render.input_object_path,
            'image_path': cfg.s1_render.object_image_path
        }
        try:
            result = process_single_object(cfg, object_config)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing single object: {e}")
            print(f"Traceback: {traceback.format_exc()}")
        
        # Print final summary for single object
        pipeline_total_time = time.time() - pipeline_start_time
        print(f"\n{'='*80}")
        print(f"Pipeline Complete: {str(timedelta(seconds=int(pipeline_total_time)))}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

