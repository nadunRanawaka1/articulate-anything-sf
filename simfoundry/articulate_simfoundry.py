from articulate_anything.utils.utils import join_path, Steps
from omegaconf import DictConfig, OmegaConf
import hydra
import os
import shutil
import sys
from pathlib import Path

import datetime
import logging
from typing import Dict, Optional, Callable, Any
import time
import trimesh
from scipy.spatial.transform import Rotation as R
import numpy as np



project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from articulate import ArticulationPipeline
from articulate_joint import articulate_joint
from articulate_anything.preprocess.preprocess_partnet import render_partnet_obj


urdf_to_semantic = {
    "prismatic": "translation",
    "revolute": "rotation",
    "continuous": "continuous", 
    "fixed": "free"  # for base
}


class DummyObjectSelector():
    def __init__(self, cfg: DictConfig, obj_id: str):
        self.cfg = cfg
        self.obj_id = obj_id
    
    def load_prediction(self):
        return {"obj_id": self.obj_id}


class DummyLinkActor():
    """
    Dummy link actor that provides the interface expected by articulate_joint.
    Points to a generated link_placement.py file.
    """
    OUT_RESULT_PATH = "link_placement.py"
    
    def __init__(self, out_dir: str):
        # Create a simple namespace object for cfg.out_dir
        self.cfg = OmegaConf.create({"out_dir": out_dir})


def generate_link_placement_from_urdf(urdf_path: str, object_name: str, output_path: str = None) -> str:
    """
    Generate a dummy link placement Python file from an existing URDF.
    
    Since the mesh parts are already correctly positioned (from segmentation),
    we just need to create a Python file that:
    1. Creates a Robot with all links
    2. Adds fixed joints between base and other links
    
    This allows the joint actor to work with the existing structure.
    
    Args:
        urdf_path: Path to the existing URDF file
        object_name: Name of the object (used for function naming)
        output_path: Where to save the Python file (defaults to same dir as URDF)
    
    Returns:
        Path to the generated Python file
    """
    import xml.etree.ElementTree as ET
    
    if output_path is None:
        output_path = join_path(os.path.dirname(urdf_path), "link_placement.py")
    
    # Parse the URDF
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # Get all links (excluding 'base' which is special)
    links = []
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        links.append(link_name)
    
    # Get all joints to understand the structure
    joints = []
    for joint in root.findall("joint"):
        joint_info = {
            'name': joint.attrib["name"],
            'type': joint.attrib["type"],
            'parent': joint.find("parent").attrib["link"],
            'child': joint.find("child").attrib["link"],
        }
        joints.append(joint_info)
    
    # Find the root link (child of base)
    root_link = None
    for joint in joints:
        if joint['parent'] == 'base':
            root_link = joint['child']
            break
    
    # Generate Python code
    # Clean object name for function name (replace hyphens, spaces with underscores)
    func_name = object_name.replace('-', '_').replace(' ', '_')
    
    code_lines = [
        "from articulate_anything.api.odio_urdf import *",
        "",
        "",
        f"def partnet_{func_name}(input_dir, links):",
        '    """',
        f"    No. masked_links: {len(links)}",
        "    Robot Link Summary:",
    ]
    
    # Add link summary
    for link_name in links:
        code_lines.append(f"    - {link_name}")
    
    code_lines.extend([
        "",
        f"    Object: {object_name}",
        '    """',
        f'    pred_robot = Robot(input_dir=input_dir, name="{func_name}")',
    ])
    
    # Add base link first
    if 'base' in links:
        code_lines.append("    pred_robot.add_link(links['base'])")
    
    # Add the root link (child of base) with fixed joint
    if root_link and root_link in links:
        code_lines.append(f"    pred_robot.add_link(links['{root_link}'])")
        code_lines.append(f'    pred_robot.add_joint(Joint("base_to_{root_link}",')
        code_lines.append('                         Parent("base"),')
        code_lines.append(f'                         Child("{root_link}"),')
        code_lines.append('                         type="fixed"),')
        code_lines.append('                         )')
    
    # Add remaining links with fixed joints to root_link
    for link_name in links:
        if link_name in ['base', root_link]:
            continue
        code_lines.append(f"    pred_robot.add_link(links['{link_name}'])")
        # Add fixed joint to root_link (or base if no root_link)
        parent = root_link if root_link else 'base'
        code_lines.append(f'    pred_robot.add_joint(Joint("{parent}_to_{link_name}",')
        code_lines.append(f'                         Parent("{parent}"),')
        code_lines.append(f'                         Child("{link_name}"),')
        code_lines.append('                         type="fixed"),')
        code_lines.append('                         )')
    
    code_lines.extend([
        "",
        "    return pred_robot",
        "",
    ])
    
    # Write the Python file
    code = "\n".join(code_lines)
    with open(output_path, 'w') as f:
        f.write(code)
    
    return output_path

def create_semantics_file(out_dir: str, articulation_tree_dict: dict):
    semantics_file = join_path(out_dir, "semantics.txt")

    with open(semantics_file, "w") as f:
        for joint in articulation_tree_dict['joints']:
            f.write(f"{joint['child_link']} {urdf_to_semantic[joint['joint_type']]} {joint['child_link']}\n")

        f.write(f"{articulation_tree_dict['fixed_part_name']}_link free {articulation_tree_dict['fixed_part_name']}_link")

def edit_urdf_to_use_mesh_type(urdf_path: str, mesh_type: str = '.obj'):
    f"""
    Edit the URDF to use <mesh_type> meshes instead of .glb, .stl, .dae meshes.
    """
    with open(urdf_path, 'r') as file:
        urdf_content = file.read()
    urdf_content = urdf_content.replace('.glb', mesh_type)
    urdf_content = urdf_content.replace('.stl', mesh_type)
    urdf_content = urdf_content.replace('.dae', mesh_type)
    urdf_content = urdf_content.replace('.obj', mesh_type)
    with open(urdf_path, 'w') as file:
        file.write(urdf_content)


def edit_urdf_to_use_relative_paths(urdf_path: str):
    """
    Edit URDF mesh filenames to use relative paths.
    
    Converts absolute paths like '/path/to/meshes/part_0.obj' 
    to relative paths like 'meshes/part_0.obj'.
    
    Args:
        urdf_path: Path to the URDF file to edit in-place.
    """
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # Find all mesh elements and update their filenames
    for mesh in root.iter('mesh'):
        filename = mesh.get('filename')
        if filename:
            # Extract just the meshes/filename.ext portion
            # Handle both forward and backward slashes
            filename = filename.replace('\\', '/')
            
            # Find 'meshes/' in the path and keep everything from there
            if 'meshes/' in filename:
                relative_path = 'meshes/' + filename.split('meshes/')[-1]
            else:
                # If 'meshes/' not in path, just use the basename with meshes/ prefix
                basename = os.path.basename(filename)
                relative_path = f'meshes/{basename}'
            
            mesh.set('filename', relative_path)
    
    # Write back the modified URDF
    tree.write(urdf_path, encoding='unicode', xml_declaration=True)

def convert_meshes_to_obj(mesh_dir: str, verbose: bool = False):
    """
    Convert all .glb, .stl, .dae mesh files in mesh_dir to .obj format (if not already .obj).
    This is needed because pybullet does not support .glb meshes.
    
    IMPORTANT: OBJ format doesn't preserve coordinate system metadata, so we ensure
    consistent handling by always using to_geometry() which applies any scene transforms.
    """
    for file in os.listdir(mesh_dir):
        if file.endswith(('.glb', '.stl', '.dae')):
            mesh_path = os.path.join(mesh_dir, file)
            obj_path = os.path.splitext(mesh_path)[0] + ".obj"
            try:
                loaded = trimesh.load(mesh_path)
                
                # Handle Scene objects consistently - apply embedded transforms
                if isinstance(loaded, trimesh.Scene):
                    if hasattr(loaded, 'to_geometry'):
                        mesh = loaded.to_geometry()  # New API (applies transforms)
                    else:
                        mesh = loaded.dump(concatenate=True)  # Fallback
                else:
                    mesh = loaded
                
                if verbose:
                    print(f"  Converting {file}:")
                    print(f"    Type: {type(loaded).__name__}")
                    print(f"    Bounds before export: {mesh.bounds[0]} to {mesh.bounds[1]}")
                
                # Export with explicit vertex order preservation
                mesh.export(obj_path, file_type='obj')
                
                if verbose:
                    # Verify by reloading
                    reloaded = trimesh.load(obj_path)
                    print(f"    Bounds after reload: {reloaded.bounds[0]} to {reloaded.bounds[1]}")
                    if not np.allclose(mesh.bounds, reloaded.bounds, atol=0.001):
                        print(f"   WARNING: Bounds changed after export!")
                    
            except Exception as e:
                print(f"Failed to convert {mesh_path} to OBJ: {e}")


def compute_auto_raise_offset(mesh_dir: str, rotation_pose: dict = None, verbose: bool = False) -> float:
    """
    Automatically compute raise_distance_offset based on mesh bounds.
    
    Args:
        mesh_dir: Directory containing mesh files
        rotation_pose: Dict with 'rx', 'ry', 'rz' rotation angles in radians
        verbose: Print debug output
    """
    import trimesh
    
    # Create rotation matrix if rotation_pose is provided
    rotation_matrix = np.eye(4)
    if rotation_pose:
        rotation = R.from_euler('xyz', [
            rotation_pose.get('rx', 0),
            rotation_pose.get('ry', 0),
            rotation_pose.get('rz', 0)
        ])
        rotation_matrix[:3, :3] = rotation.as_matrix()
        if verbose:
            print(f"  Computing raise offset with rotation: "
                  f"rx={rotation_pose.get('rx', 0):.2f}, "
                  f"ry={rotation_pose.get('ry', 0):.2f}, "
                  f"rz={rotation_pose.get('rz', 0):.2f}")
    
    min_z = 0.0
    
    for file in os.listdir(mesh_dir):
        if file.endswith(('.obj', '.glb', '.stl')):
            mesh_path = os.path.join(mesh_dir, file)
            try:
                loaded = trimesh.load(mesh_path)
                if isinstance(loaded, trimesh.Scene):
                    if hasattr(loaded, 'to_geometry'):
                        mesh = loaded.to_geometry()
                    else:
                        mesh = loaded.dump(concatenate=True)
                else:
                    mesh = loaded
                
                # Apply rotation transform BEFORE computing bounds
                if rotation_pose:
                    mesh = mesh.copy()
                    mesh.apply_transform(rotation_matrix)
                
                # Get the minimum Z coordinate after rotation
                bounds_min = mesh.bounds[0]
                if bounds_min[2] < min_z:
                    min_z = bounds_min[2]
                    if verbose:
                        print(f"  {file}: min_z = {min_z:.4f} (after rotation)")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not load {file}: {e}")
    
    # The offset should lift the lowest point to ground level (0)
    offset = -min_z + 0.01
    
    if verbose:
        print(f"\nAuto-computed raise_distance_offset: {offset:.4f}")
        print(f"  (Lowest point after rotation: {min_z:.4f}, clearance: 0.01)")
    
    return float(offset)


def compute_auto_camera_params(
    mesh_dir: str, 
    rotation_pose: dict = None, 
    padding: float = 2.0,
    raise_offset: float = 0.0,
    min_distance: float = 0.5,
    verbose: bool = False
) -> dict:
    """
    Compute camera positions that automatically frame the object based on its bounds.
    
    Args:
        mesh_dir: Directory containing mesh files
        rotation_pose: Dict with 'rx', 'ry', 'rz' rotation angles in radians
                      (camera positions are computed AFTER this rotation is applied)
        padding: Multiplier for camera distance (larger = further away)
        raise_offset: Z offset applied to the object (from raise_distances.json)
        min_distance: Minimum camera distance to ensure object is visible
        verbose: Print debug output
        
    Returns:
        Dict with camera view parameters that can be merged into simulator_cfg.camera_params.views
    """
    import trimesh
    
    # Create rotation matrix if rotation_pose is provided
    rotation_matrix = np.eye(4)
    if rotation_pose:
        rotation = R.from_euler('xyz', [
            rotation_pose.get('rx', 0),
            rotation_pose.get('ry', 0),
            rotation_pose.get('rz', 0)
        ])
        rotation_matrix[:3, :3] = rotation.as_matrix()
    
    # Compute combined bounds of all meshes
    all_vertices = []
    
    for file in os.listdir(mesh_dir):
        if file.endswith(('.obj', '.glb', '.stl')):
            mesh_path = os.path.join(mesh_dir, file)
            try:
                loaded = trimesh.load(mesh_path)
                if isinstance(loaded, trimesh.Scene):
                    if hasattr(loaded, 'to_geometry'):
                        mesh = loaded.to_geometry()
                    else:
                        mesh = loaded.dump(concatenate=True)
                else:
                    mesh = loaded
                
                # Apply rotation transform
                if rotation_pose:
                    mesh = mesh.copy()
                    mesh.apply_transform(rotation_matrix)
                
                all_vertices.append(mesh.vertices)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not load {file}: {e}")
    
    if not all_vertices:
        if verbose:
            print("  No meshes found, using default camera params")
        return {}
    
    # Combine all vertices and compute bounds
    combined_vertices = np.vstack(all_vertices)
    bounds_min = combined_vertices.min(axis=0)
    bounds_max = combined_vertices.max(axis=0)
    
    # Compute center and size
    center = (bounds_min + bounds_max) / 2
    size = bounds_max - bounds_min
    max_dimension = float(max(size))  # Convert to native Python float
    
    # Camera distance based on object size, with minimum distance
    distance = max(max_dimension * padding, min_distance)
    
    # Look-at point is the center of the object, adjusted for raise offset
    # The raise_offset moves the object up in Z in world space
    cx = float(center[0])
    cy = float(center[1])
    cz = float(center[2]) + raise_offset  # Add raise offset to Z
    look_at = [cx, cy, cz]
    
    # Compute camera positions for each view
    # These are positioned relative to the object center
    # All values are native Python floats for OmegaConf compatibility
    views = {
        'frontview': {
            'cam_pos': [cx + distance, cy + distance * 0.5, cz + distance * 0.5],
            'look_at': look_at
        },
        'leftview': {
            'cam_pos': [cx, cy - distance, cz + distance * 0.5],
            'look_at': look_at
        },
        'right_45': {
            'cam_pos': [cx, cy + distance, cz + distance * 0.5],
            'look_at': look_at
        },
        'left_45': {
            'cam_pos': [cx, cy - distance, cz + distance * 0.5],
            'look_at': look_at
        },
        'rightview': {
            'cam_pos': [cx, cy + distance, cz + distance * 0.5],
            'look_at': look_at
        },
        'backview': {
            'cam_pos': [cx - distance, cy - distance * 0.5, cz + distance * 0.5],
            'look_at': look_at
        },
        'topview': {
            'cam_pos': [cx, cy, cz + distance * 1.5],
            'look_at': look_at
        },
        'angle45': {
            'cam_pos': [cx + distance * 0.7, cy + distance * 0.7, cz + distance * 0.7],
            'look_at': look_at
        }
    }
    
    if verbose:
        print(f"\n=== Auto-computed camera parameters ===")
        print(f"  Object bounds: {bounds_min} to {bounds_max}")
        print(f"  Object size: {size} (max dim: {max_dimension:.3f})")
        print(f"  Object center (local): {center}")
        print(f"  Raise offset: {raise_offset:.3f}")
        print(f"  Look-at point (world): {look_at}")
        print(f"  Camera distance: {distance:.3f} (padding={padding}, min={min_distance})")
        for view_name, params in views.items():
            print(f"  {view_name}: pos={[f'{x:.2f}' for x in params['cam_pos']]}, look_at={[f'{x:.2f}' for x in params['look_at']]}")
    
    return views


def setup_dir(cfg: DictConfig, articulation_tree_dict: dict, verbose: bool = False):

        out_dir = join_path(cfg.out_dir, f"{cfg.object_name}")
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(join_path(out_dir, "meshes")):
            shutil.rmtree(join_path(out_dir, "meshes"))
        
        shutil.copytree(cfg.mesh_parts_dir, join_path(out_dir, "meshes"), dirs_exist_ok=True)
        shutil.copy(cfg.dummy_urdf_path, join_path(out_dir, "mobility.urdf"))
        shutil.copy(cfg.object_image_path, join_path(out_dir, "robot_frontview.png"))
        create_semantics_file(out_dir, articulation_tree_dict)
        
        # Convert to OBJ if needed (Step 5 already exports as OBJ now)
        meshes_dir = join_path(out_dir, "meshes")
        needs_conversion = any(f.endswith(('.glb', '.stl', '.dae')) for f in os.listdir(meshes_dir))
        
        if needs_conversion:
            if verbose:
                print("  Converting meshes to OBJ for PyBullet compatibility")
            convert_meshes_to_obj(meshes_dir, verbose=verbose)
            edit_urdf_to_use_mesh_type(join_path(out_dir, "mobility.urdf"), '.obj')
        elif verbose:
            print("  Meshes already in OBJ format")


def create_articulation_cfg(cfg: DictConfig, verbose: bool = False):

    # This avoids conflicts when already inside a Hydra context
    articulation_cfg_path = cfg.articulation_cfg_path
    if not os.path.isabs(articulation_cfg_path):
        articulation_cfg_path = os.path.join(project_root, articulation_cfg_path)
    
    if verbose:
        print(f"Loading articulation config from: {articulation_cfg_path}")
    
    articulation_cfg = OmegaConf.load(articulation_cfg_path)
    
    if verbose:
        print(f"Loaded base config with keys: {list(articulation_cfg.keys())}")
    
    # Load and merge default configs from conf/ directory
    conf_dir = os.path.join(project_root, "conf")
    
    # Handle defaults if they exist in the config
    if 'defaults' in articulation_cfg:
        defaults_list = articulation_cfg.defaults
        if verbose:
            print(f"Processing {len(defaults_list)} defaults...")
        
        del articulation_cfg['defaults']
        
        for default_item in defaults_list:
            if verbose:
                print(f"  Processing item: {default_item}, type: {type(default_item)}")
            
            if isinstance(default_item, str):
                if verbose:
                    print(f"  Skipping string default: {default_item}")
                continue
            elif isinstance(default_item, (dict, DictConfig)):
                # Dict format like {simulator: default}
                
                for key, value in default_item.items():
                    default_config_path = os.path.join(conf_dir, key, f"{value}.yaml")
                    if verbose:
                        print(f"  Loading {key}: {default_config_path}")
                    if os.path.exists(default_config_path):
                        default_cfg = OmegaConf.load(default_config_path)
                        if key in articulation_cfg:
                            if verbose:
                                print(f"    Merging with existing config for {key}")
                            articulation_cfg[key] = OmegaConf.merge(default_cfg, articulation_cfg[key])
                        else:
                            if verbose:
                                print(f"    Adding new config for {key}")
                            articulation_cfg[key] = default_cfg
                    else:
                        if verbose:
                            print(f"    Warning: Default config not found: {default_config_path}")
    else:
        if verbose:
            print("No defaults found in config")
    
    articulation_cfg.project_root = project_root
    if verbose:
        print(f"\nFinal config keys: {list(articulation_cfg.keys())}")

    
    
    return articulation_cfg

def articulate_simfoundry(
    cfg: DictConfig,
    articulation_tree_dict: dict,
    gpu_id: int = 0,
    api_key: str = None,
    verbose: bool = False,
):
    """
    Refine joint parameters using an image, preserving existing link placement.
    """
    
    if verbose:
        print("\n=== Creating Articulation Config ===")
    
    articulation_cfg = create_articulation_cfg(cfg, verbose=verbose)
    
    if not os.path.exists(join_path(cfg.out_dir, cfg.object_name, "meshes")):
        setup_dir(cfg, articulation_tree_dict, verbose=verbose)
    
    
    # Auto-compute raise offset
    auto_offset = None
    if cfg.get('auto_raise_offset', True):
        mesh_dir = join_path(cfg.out_dir, cfg.object_name, "meshes")
        # Pass rotation_pose from articulation_cfg to account for rotated bounds
        rotation_pose = {
            'rx': articulation_cfg.simulator.urdf.rotation_pose.rx,
            'ry': articulation_cfg.simulator.urdf.rotation_pose.ry,
            'rz': articulation_cfg.simulator.urdf.rotation_pose.rz,
        }
        auto_offset = compute_auto_raise_offset(mesh_dir, rotation_pose=rotation_pose, verbose=verbose)
        auto_offset += cfg.auto_raise_offset_clearance
        
        import xml.etree.ElementTree as ET
        urdf_path = join_path(cfg.out_dir, cfg.object_name, "mobility.urdf")
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        num_joints = len(root.findall("joint"))
        
        raise_dist_file = join_path(cfg.out_dir, cfg.object_name, "raise_distances.json")
        from articulate_anything.utils.utils import save_json
        save_json([auto_offset] * num_joints, raise_dist_file)

        # TODO: copy raise_distances.json to the link actor directory
        
        if verbose:
            print(f"\nCreated raise_distances.json BEFORE link articulation:")
            print(f"  File: {raise_dist_file}")
            print(f"  Values: [{auto_offset:.4f}] × {num_joints}")
            print(f"  (setup_pybullet will now skip creating it since it exists)")
    
    # Configure for joint refinement
    articulation_cfg.modality = "image"
    articulation_cfg.prompt = cfg.object_image_path
    articulation_cfg.out_dir = f"{cfg.out_dir}/articulation_results"
    articulation_cfg.dataset_dir = cfg.out_dir
    articulation_cfg.gpu_id = gpu_id
    
    if api_key:
        articulation_cfg.api_key = api_key

    
    # Force pybullet to use the raise_distances.json file just created
    if cfg.get('auto_raise_offset', True):
        articulation_cfg.simulator.urdf.raise_distance_offset = 0.0
        articulation_cfg.simulator.urdf.raise_distance_file = join_path(
            cfg.out_dir, cfg.object_name, "raise_distances.json"
        )
        if verbose:
            print(f"Using raise_distances.json (offset=0.0): {articulation_cfg.simulator.urdf.raise_distance_file}")
    
    articulation_cfg.joint_actor.mode = "image"
    articulation_cfg.joint_actor.targetted_affordance = False  # Articulate ALL joints
    articulation_cfg.simulator.script_path = os.path.join(project_root, articulation_cfg.simulator.script_path)
    articulation_cfg.simulator.conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
    
    pipeline = ArticulationPipeline(articulation_cfg)
    
    # Skip mesh retrieval - use dummy object selector
    obj_selector = DummyObjectSelector(articulation_cfg, cfg.object_name)
    
    main_steps = Steps()
    mesh_retrieval_steps = Steps()
    mesh_retrieval_steps.add_step("Object Selection", obj_selector)
    main_steps.add_step("Mesh Retrieval", mesh_retrieval_steps)
    
    # Skip link articulation - parts already have correct positions from mesh segmentation
    # Generate a dummy link placement Python file from the URDF
    # This is needed because the joint actor expects link placement code as part of its prompt
    object_dir = join_path(cfg.out_dir, cfg.object_name)
    urdf_file = join_path(object_dir, "mobility.urdf")
    
    # Auto-compute camera positions based on object size
    if cfg.get('auto_camera', True):
        mesh_dir = join_path(object_dir, "meshes")
        rotation_pose = {
            'rx': articulation_cfg.simulator.urdf.rotation_pose.rx,
            'ry': articulation_cfg.simulator.urdf.rotation_pose.ry,
            'rz': articulation_cfg.simulator.urdf.rotation_pose.rz,
        }
        # Use the auto_offset computed earlier (if available) to position camera correctly
        camera_raise_offset = auto_offset if auto_offset is not None else 0.0
        auto_views = compute_auto_camera_params(
            mesh_dir, 
            rotation_pose=rotation_pose,
            padding=cfg.get('auto_camera_padding', 2.5),
            raise_offset=camera_raise_offset,
            min_distance=cfg.get('auto_camera_min_distance', 0.5),
            verbose=verbose
        )
        if auto_views:
            # Update simulator camera params with auto-computed views
            for view_name, view_params in auto_views.items():
                if view_name in articulation_cfg.simulator.camera_params.views:
                    articulation_cfg.simulator.camera_params.views[view_name].cam_pos = view_params['cam_pos']
                    articulation_cfg.simulator.camera_params.views[view_name].look_at = view_params['look_at']
            
            # Set axes_origin to match the look_at point (object center)
            # This positions the coordinate axes at the object's location
            first_view = list(auto_views.values())[0]
            articulation_cfg.simulator.axes_origin = first_view['look_at']
            
            if verbose:
                print(f"  Camera params updated with auto-computed values")
                print(f"  Axes origin set to: {first_view['look_at']}")
    
    # Apply URDF rotation and render
    # This rotates the base joint by 180° around the y-axis for correct camera orientation
    render_partnet_obj(cfg.object_name, gpu_id, articulation_cfg, "stationary", urdf_file)
    
    # Generate the link placement Python file from the URDF
    link_placement_path = generate_link_placement_from_urdf(
        urdf_file, 
        cfg.object_name,
        output_path=join_path(object_dir, "link_placement.py")
    )
    if verbose:
        print(f"\n=== Generated link placement code from URDF ===")
        print(f"  Output: {link_placement_path}")
    
    # Create dummy link actor that points to the generated Python file
    dummy_link_actor = DummyLinkActor(out_dir=object_dir)
    
    link_articulation_steps = Steps()
    link_articulation_steps.add_step("Link actor", [dummy_link_actor])  # List with one dummy actor
    link_articulation_steps.add_step("Link critic", [])  # Empty critic list
    main_steps.add_step("Link Articulation", link_articulation_steps)

    
    
    # Set global simulator rotation context for coordinate transformation
    # This allows Robot.make_prismatic_joint and make_revolute_joint to automatically
    # transform from simulator/view frame to mesh frame
    from articulate_anything.api.odio_urdf import set_simulator_rotation_context, clear_simulator_rotation_context
    
    set_simulator_rotation_context(
        rx=articulation_cfg.simulator.urdf.rotation_pose.rx,
        ry=articulation_cfg.simulator.urdf.rotation_pose.ry,
        rz=articulation_cfg.simulator.urdf.rotation_pose.rz
    )
    
    if verbose:
        print(f"\n=== Set simulator rotation context for coordinate transforms ===")
        print(f"  rx={articulation_cfg.simulator.urdf.rotation_pose.rx:.4f}, "
              f"ry={articulation_cfg.simulator.urdf.rotation_pose.ry:.4f}, "
              f"rz={articulation_cfg.simulator.urdf.rotation_pose.rz:.4f}")
    
    try:
        if verbose:
            print("\n=== Running Joint Articulation ===")
        
        # dataset_dir should include object_name since that's where mobility.urdf and semantics.txt are
        articulation_cfg.dataset_dir = join_path(cfg.out_dir, cfg.object_name)
        result = articulate_joint(articulation_cfg.prompt, main_steps, str(articulation_cfg.gpu_id), articulation_cfg)
    
    finally:
        # Clear context after joint articulation
        clear_simulator_rotation_context()
        if verbose:
            print("Cleared simulator rotation context")
    
    print(f"Joint refinement complete!")
    print(f"Refined joints saved to: {articulation_cfg.out_dir}")

    # Copy the final urdf to the out_dir and edit to use .glb meshes
    joint_actor_result = result["Joint actor"][-1]
    final_urdf_path = join_path(joint_actor_result.cfg.out_dir, 'mobility.urdf')
    shutil.copy(final_urdf_path, join_path(cfg.out_dir, cfg.object_name, "mobility_final.urdf"))
    edit_urdf_to_use_mesh_type(join_path(cfg.out_dir, cfg.object_name, "mobility_final.urdf"), '.glb')

    # Now copy this urdf and the meshes to a results directory for the object
    object_results_dir = join_path(cfg.out_dir, cfg.object_name, "../../results")
    os.makedirs(object_results_dir, exist_ok=True)
    shutil.copy(join_path(cfg.out_dir, cfg.object_name, "mobility_final.urdf"), join_path(object_results_dir, "mobility.urdf"))
    shutil.copytree(join_path(cfg.out_dir, cfg.object_name, "meshes"), join_path(object_results_dir, "meshes"))

    # edit mobility.urdf to use relative paths for the meshes
    edit_urdf_to_use_relative_paths(join_path(object_results_dir, "mobility.urdf"))

    return pipeline.steps
