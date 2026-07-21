"""
Transform joint axes from simulator/view coordinate frame back to mesh coordinate frame.

When simulator applies rotation (rx, ry, rz), the VLM sees the rotated object and predicts
joint axes in the rotated frame. But URDF joints must be specified in the mesh frame.

This script transforms the axes back.
"""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path


def transform_urdf_joints_to_mesh_frame(urdf_path: str, rx: float, ry: float, rz: float, 
                                        output_path: str = None, verbose: bool = False):
    """
    Transform all joint axes in a URDF from simulator/view frame to mesh frame.
    
    Args:
        urdf_path: Path to URDF with axes in rotated frame
        rx, ry, rz: Rotation angles in radians (same as simulator.urdf.rotation_pose)
        output_path: Where to save transformed URDF (default: overwrite input)
        verbose: Print transformations
    
    Returns:
        Path to transformed URDF
    """
    if output_path is None:
        output_path = urdf_path
    
    # Create inverse rotation matrix
    # Simulator does: R = Rz(rz) @ Ry(ry) @ Rx(rx)
    # We need: R_inv to go from rotated frame → mesh frame
    rotation = R.from_euler('xyz', [rx, ry, rz])
    rotation_inv = rotation.inv()
    rotation_matrix_inv = rotation_inv.as_matrix()
    
    if verbose:
        print(f"Transforming URDF joints from rotated frame to mesh frame")
        print(f"  Simulator rotation: rx={rx:.4f}, ry={ry:.4f}, rz={rz:.4f}")
        print(f"  Using inverse rotation matrix")
    
    # Parse URDF
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # Transform each joint's axis
    joints_transformed = 0
    for joint in root.findall("joint"):
        joint_name = joint.attrib.get("name")
        joint_type = joint.attrib.get("type")
        
        # Only transform movable joints
        if joint_type not in ["revolute", "prismatic", "continuous"]:
            continue
        
        axis_elem = joint.find("axis")
        if axis_elem is not None:
            # Get current axis (in rotated frame)
            axis_str = axis_elem.attrib.get("xyz", "1 0 0")
            axis_rotated = np.array([float(x) for x in axis_str.split()])
            
            # Transform to mesh frame
            axis_mesh = rotation_matrix_inv @ axis_rotated
            
            # Normalize
            axis_mesh = axis_mesh / np.linalg.norm(axis_mesh)
            
            # Update XML
            axis_elem.attrib["xyz"] = f"{axis_mesh[0]} {axis_mesh[1]} {axis_mesh[2]}"
            
            if verbose:
                print(f"  {joint_name} ({joint_type}):")
                print(f"    Rotated frame: [{axis_rotated[0]:.4f}, {axis_rotated[1]:.4f}, {axis_rotated[2]:.4f}]")
                print(f"    Mesh frame:    [{axis_mesh[0]:.4f}, {axis_mesh[1]:.4f}, {axis_mesh[2]:.4f}]")
            
            joints_transformed += 1
    
    # Save transformed URDF
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    if verbose:
        print(f"\n✅ Transformed {joints_transformed} joint axes")
        print(f"  Output: {output_path}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Transform URDF joint axes to mesh frame')
    parser.add_argument('--urdf', type=str, required=True, help='Path to URDF file')
    parser.add_argument('--rx', type=float, required=True, help='X rotation in radians')
    parser.add_argument('--ry', type=float, default=0.0, help='Y rotation in radians')
    parser.add_argument('--rz', type=float, required=True, help='Z rotation in radians')
    parser.add_argument('--output', type=str, help='Output path (default: overwrite input)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    transform_urdf_joints_to_mesh_frame(
        args.urdf, args.rx, args.ry, args.rz,
        output_path=args.output,
        verbose=args.verbose
    )

