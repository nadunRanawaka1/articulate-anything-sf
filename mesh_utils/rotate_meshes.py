#!/usr/bin/env python3
"""
Fix mesh orientations by rotating them.
"""

import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import argparse
import shutil


def rotate_mesh_file(input_path, output_path, rotation_matrix):
    """Load, rotate, and save a mesh."""
    print(f"Processing: {input_path.name}")
    
    mesh = trimesh.load(str(input_path))
    
    # Handle Scene objects (GLB files)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    
    # Apply rotation
    mesh.apply_transform(rotation_matrix)
    
    # Save as OBJ
    mesh.export(str(output_path))
    print(f"  → Saved: {output_path.name}")
    
    return mesh


def fix_all_meshes(mesh_dir, axis, angle_degrees, backup=True):
    """Fix orientation of all meshes in a directory."""
    mesh_dir = Path(mesh_dir)
    
    # Create rotation matrix
    rotation = R.from_euler(axis, angle_degrees, degrees=True)
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = rotation.as_matrix()
    
    print(f"\n{'='*80}")
    print(f"Rotating all meshes by {angle_degrees}° around {axis.upper()}-axis")
    print('='*80)
    print(f"This will transform:")
    if axis == 'y' and angle_degrees == -90:
        print(f"  X (width, {1.99:.2f}) → Z (height)")
        print(f"  Y (depth, {1.20:.2f}) → Y (depth)")  
        print(f"  Z (height, {1.08:.2f}) → -X (width)")
    print()
    
    # Find all mesh files
    mesh_files = list(mesh_dir.glob("*.obj")) + list(mesh_dir.glob("*.glb"))
    
    if not mesh_files:
        print(f"No mesh files found in {mesh_dir}")
        return
    
    # Backup if requested
    if backup:
        backup_dir = mesh_dir.parent / "meshes_backup"
        if not backup_dir.exists():
            print(f"📦 Creating backup: {backup_dir}")
            shutil.copytree(mesh_dir, backup_dir)
            print()
    
    # Process each mesh
    for mesh_file in sorted(mesh_files):
        output_path = mesh_dir / (mesh_file.stem + "_rotated.obj")
        rotate_mesh_file(mesh_file, output_path, rotation_matrix)
    
    print(f"\n✅ Done! Processed {len(mesh_files)} meshes")
    print(f"\nNEXT STEPS:")
    print(f"1. Rename the rotated files (remove '_rotated' suffix)")
    print(f"2. Update your URDF to point to the rotated .obj files")
    print(f"3. Test rendering again")


def main():
    parser = argparse.ArgumentParser(description="Fix mesh orientations")
    parser.add_argument(
        "--mesh-dir",
        type=str,
        required=True,
        help="Directory containing mesh files"
    )
    parser.add_argument(
        "--axis",
        type=str,
        choices=['x', 'y', 'z'],
        default='y',
        help="Rotation axis (default: y)"
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=-90,
        help="Rotation angle in degrees (default: -90)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup of original meshes"
    )
    
    args = parser.parse_args()
    
    fix_all_meshes(
        args.mesh_dir,
        args.axis,
        args.angle,
        backup=not args.no_backup
    )
    
    print("="*80)


if __name__ == "__main__":
    main()