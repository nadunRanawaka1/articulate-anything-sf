"""
Generate base URDF from articulation tree and segmented mesh parts.
"""
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from omegaconf import DictConfig
from pathlib import Path
import trimesh


def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def generate_base_urdf(cfg: DictConfig, articulation_tree_dict: dict, verbose: bool = False):
    """
    Generate a base URDF file from the articulation tree and mesh parts.
    
    Args:
        cfg: Configuration with paths
        articulation_tree_dict: Dict with 'parts', 'links', and 'joints' from VLM
        parts_dict: Optional dict with 'fixed_part_name' for base identification
        verbose: Print verbose output
    
    The URDF will have:
    - Proper link/joint structure from articulation tree
    - Real mesh filenames from segmented parts
    - Dummy values for origins, axes, and limits (to be refined later)
    """
    
    
    mesh_parts_dir = cfg.mesh_parts_dir
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)
    if Path(out_dir + "/dummy.urdf").exists() and not cfg.rerun:
        return str(Path(out_dir + "/dummy.urdf"))


    if verbose:
        print("Generating base URDF...")
    
    robot_name = cfg.get('object_name', 'ROBOT')
    
    root = ET.Element('robot', name=robot_name)
    
    # Map link names to their corresponding mesh files
    link_to_meshes = {}
    
    mesh_files = {}
    if os.path.exists(mesh_parts_dir):
        for file in os.listdir(mesh_parts_dir):
            if file.endswith(('.obj', '.glb', '.stl', '.dae')):
                part_name = os.path.splitext(file)[0]
                # Prefer .obj files if both .obj and .glb exist
                if part_name not in mesh_files or file.endswith('.obj'):
                    mesh_files[part_name] = file
                if verbose:
                    print(f"  Found mesh: {part_name} -> {file}")
    
    
    links = articulation_tree_dict.get('links', [])
    parts = articulation_tree_dict.get('parts', [])
    joints = articulation_tree_dict.get('joints', [])
    
    # Create mapping: for simplicity, assume part names match link names
    # In a more sophisticated version, this could be derived from the VLM or user input
    for link in links:
        link_name = link['link_name']
        
        if link_name in mesh_files:
            link_to_meshes[link_name] = [mesh_files[link_name]]
        else:
            link_to_meshes[link_name] = []
            for part in parts:
                part_name = part['part_name']
                if part_name.lower() in link_name.lower() or link_name.lower() in part_name.lower():
                    if part_name in mesh_files:
                        link_to_meshes[link_name].append(mesh_files[part_name])
    
    fixed_part_name = articulation_tree_dict.get('fixed_part_name', 'base') if articulation_tree_dict else 'base'
    if verbose:
        print(f"  Fixed part name: {fixed_part_name}")
    
    # Find the root link (the one that's not a child in any joint)
    child_links = set(joint['child_link'] for joint in joints)
    all_links = set(link['link_name'] for link in links)
    root_link_candidates = all_links - child_links
    
    root_link_name = None
    if len(root_link_candidates) == 1:
        root_link_name = root_link_candidates.pop()
        if verbose:
            print(f"  Root link (top of kinematic tree): {root_link_name}")
    elif len(root_link_candidates) > 1:
        # Multiple root candidates - use fixed_part_name 
        if fixed_part_name in root_link_candidates:
            root_link_name = fixed_part_name
            if verbose:
                print(f"  Multiple root candidates, using fixed_part_name as root: {root_link_name}")
    elif len(root_link_candidates) == 0:
        if verbose:
            print("  Warning: No root link found (all links are children), kinematic tree may be cyclic")
        # Pick the first parent link as fallback
        if joints:
            root_link_name = joints[0]['parent_link']
    
    # Create empty base link (required for rotate_urdf to work)
    base_link = ET.SubElement(root, 'link', name='base')
    
    # Create links with visual and collision elements
    for link in links:
        link_name = link['link_name']
        link_elem = ET.SubElement(root, 'link', name=link_name)
        
        meshes = link_to_meshes.get(link_name, [])
        
        # If no meshes found, check if this is the fixed part
        if not meshes and link_name.lower() == fixed_part_name.lower():
            if fixed_part_name in mesh_files:
                meshes = [mesh_files[fixed_part_name]]
        
        # If still no meshes, try to find any mesh with similar name
        if not meshes:
            for mesh_name, mesh_file in mesh_files.items():
                if mesh_name.lower() in link_name.lower() or link_name.lower() in mesh_name.lower():
                    meshes.append(mesh_file)
        
        if verbose:
            print(f"  Link '{link_name}': {len(meshes)} mesh(es)")
        
        # Add visual and collision elements for each mesh
        for mesh_file in meshes:
            mesh_path = f"meshes/{mesh_file}"
            
            # Compute mesh centroid to set as visual origin
            # This makes PyBullet bbox calculations match VLM expectations
            full_mesh_path = os.path.join(mesh_parts_dir, mesh_file)
            mesh_origin = "0.0 0.0 0.0"
            
            try:
                loaded_mesh = trimesh.load(full_mesh_path)
                if isinstance(loaded_mesh, trimesh.Scene):
                    if hasattr(loaded_mesh, 'to_geometry'):
                        loaded_mesh = loaded_mesh.to_geometry()
                    else:
                        loaded_mesh = loaded_mesh.dump(concatenate=True)
            
                
                if verbose:
                    print(f"    Mesh {mesh_file} center offset: {mesh_origin}")
            except Exception as e:
                if verbose:
                    print(f"    Warning: Could not compute origin for {mesh_file}: {e}")
            
            # Visual element with computed origin
            visual = ET.SubElement(link_elem, 'visual')
            geometry = ET.SubElement(visual, 'geometry')
            ET.SubElement(geometry, 'mesh', filename=mesh_path)
            ET.SubElement(visual, 'origin', rpy="0.0 0.0 0.0", xyz=mesh_origin)
            
            # Collision element (same as visual)
            collision = ET.SubElement(link_elem, 'collision')
            geometry = ET.SubElement(collision, 'geometry')
            ET.SubElement(geometry, 'mesh', filename=mesh_path)
            ET.SubElement(collision, 'origin', rpy="0.0 0.0 0.0", xyz=mesh_origin)
    
    # Create joints
    for joint in joints:
        joint_name = joint['joint_name']
        joint_type = joint['joint_type']
        parent_link = joint['parent_link']
        child_link = joint['child_link']
        
        if verbose:
            print(f"  Joint '{joint_name}': {joint_type} -> emitted as fixed "
                  f"({parent_link} -> {child_link})")

        # Emit every scaffold joint as FIXED, whatever type was detected.
        #
        # This file builds the *input scaffold* for stage 5; the real joint - its
        # type, axis, pivot and range - is what the joint actor is asked to predict.
        # Emitting `revolute` here asserted three things this file cannot know, and
        # supplied placeholders for all of them: axis (0,0,1) and limits +/-pi about
        # a pivot at the origin.
        #
        # That placeholder was not inert. The renderer poses every movable joint at
        # a limit before each stationary photo (`set_joint_to_target_limit`), so the
        # scaffold was rendered at q = +pi: a point reflection that flipped the
        # moving part upside down and dropped it by twice its height above the
        # raise line - 26 mm for a laptop screen, 210 mm for a mailbox lid.
        #
        # A fixed joint has no movable DOF, so `get_manipulatable_joints` returns
        # nothing, no qpos is ever written, and the scaffold renders in its true
        # rest pose regardless of renderer config. This also matches the generated
        # `link_placement.py`, which already declares every joint fixed.
        #
        # See docs/scaffold-render-pose.md.
        joint_elem = ET.SubElement(root, 'joint', type='fixed', name=joint_name)
        ET.SubElement(joint_elem, 'parent', link=parent_link)
        ET.SubElement(joint_elem, 'child', link=child_link)

        # Dummy origin
        ET.SubElement(joint_elem, 'origin', rpy="0.0 0.0 0.0", xyz="0.0 0.0 0.0")

        # Fixed joints don't strictly need an axis, but include for completeness.
        ET.SubElement(joint_elem, 'axis', xyz="1 0 0")

        # No <limit>: a fixed joint has no range, and any value emitted here would
        # be a guess that the renderer would then pose the part at.
    
    # Add fixed joint from base to root link
    if root_link_name:
        if verbose:
            print(f"  Creating base joint: base -> {root_link_name}")
        
        base_joint = ET.SubElement(root, 'joint', type='fixed', name='base_joint')
        ET.SubElement(base_joint, 'parent', link='base')
        ET.SubElement(base_joint, 'child', link=root_link_name)
        # PartNet standard orientation: rx=90°, rz=-90°
        # This makes objects stand up and face the correct direction
        ET.SubElement(base_joint, 'origin', rpy="1.570796326794897 0 -1.570796326794897", xyz="0 0 0")
        ET.SubElement(base_joint, 'axis', xyz="1 0 0")
        
        if verbose:
            print(f"    Base joint created with PartNet standard orientation: rpy='1.57 0 -1.57'")
    
    if verbose:
        print(f"  URDF kinematic tree: base -> {root_link_name} -> children")
        print(f"  Total links: {len(links) + 1}, Total joints: {len(joints) + 1}")  # +1 for base
    

    output_path = os.path.join(out_dir, 'dummy.urdf')
    xml_string = prettify_xml(root)
    
   
    xml_string = '\n'.join([line for line in xml_string.split('\n') if line.strip()])
    
    with open(output_path, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        # Skip the first line (XML declaration from prettify_xml)
        f.write('\n'.join(xml_string.split('\n')[1:]))
    
    if verbose:
        print(f"  Dummy URDF written to: {output_path}")
    
    return output_path

