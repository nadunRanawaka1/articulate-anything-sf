#!/usr/bin/env python3
"""
Fix GLB files by baking scene graph transforms into vertex positions.

This script takes meshes that have non-identity transforms in their scene graph
and "bakes" those transforms into the vertex positions, resulting in a clean
mesh with identity transforms that's easier to work with.

Usage:
    python fix_glb_transforms.py <directory_or_file> [--output-dir <output_dir>] [--suffix <suffix>]
    
Examples:
    # Fix all GLB files in a directory (saves to same directory with _fixed suffix)
    python fix_glb_transforms.py /path/to/meshes/
    
    # Fix all GLB files and save to a different directory
    python fix_glb_transforms.py /path/to/meshes/ --output-dir /path/to/output/
    
    # Fix a single file
    python fix_glb_transforms.py /path/to/mesh.glb
    
    # Fix with custom suffix
    python fix_glb_transforms.py /path/to/meshes/ --suffix _baked
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh


def fix_glb_transforms(
    input_path: str,
    output_path: Optional[str] = None,
    verbose: bool = True
) -> bool:
    """
    Fix a GLB file by baking scene graph transforms into vertex positions.
    
    Args:
        input_path: Path to the input GLB file
        output_path: Path for the output GLB file (if None, overwrites input)
        verbose: Print detailed information
        
    Returns:
        True if the file was modified, False if no changes needed
    """
    if verbose:
        print(f"\nProcessing: {input_path}")
    
    # Load the mesh
    loaded = trimesh.load(input_path)
    
    if not isinstance(loaded, trimesh.Scene):
        if verbose:
            print(f"  Skipping: Not a Scene (type: {type(loaded).__name__})")
        # If it's already a simple Trimesh, just copy it
        if output_path and output_path != input_path:
            loaded.export(output_path)
            print(f"  Copied to: {output_path}")
        return False
    
    # Get geometry to node mapping
    geometry_nodes = loaded.graph.geometry_nodes
    
    # Check if any transforms are non-identity
    has_transforms = False
    transforms_info = []
    
    for geom_name, geom in loaded.geometry.items():
        node_names = geometry_nodes.get(geom_name, [])
        for node_name in node_names:
            try:
                transform, _ = loaded.graph[node_name]
                if transform is not None and not np.allclose(transform, np.eye(4)):
                    has_transforms = True
                    transforms_info.append({
                        'geometry': geom_name,
                        'node': node_name,
                        'transform': transform
                    })
            except (KeyError, TypeError, ValueError):
                pass
    
    if not has_transforms:
        if verbose:
            print(f"  Skipping: All transforms are already identity")
        if output_path and output_path != input_path:
            loaded.export(output_path)
            print(f"  Copied to: {output_path}")
        return False
    
    if verbose:
        print(f"  Found {len(transforms_info)} non-identity transforms:")
        for info in transforms_info:
            print(f"    - Geometry '{info['geometry']}' via node '{info['node']}'")
    
    # Create a new mesh with transforms baked in
    # We'll concatenate all geometries with their transforms applied
    transformed_geoms = []
    
    for geom_name, geom in loaded.geometry.items():
        geom_copy = geom.copy()
        
        # Find the transform for this geometry
        node_names = geometry_nodes.get(geom_name, [])
        if node_names:
            try:
                node_name = node_names[0]
                transform, _ = loaded.graph[node_name]
                if transform is not None and not np.allclose(transform, np.eye(4)):
                    if verbose:
                        print(f"  Baking transform for '{geom_name}'...")
                    geom_copy.apply_transform(transform)
            except (KeyError, TypeError, ValueError) as e:
                if verbose:
                    print(f"  Warning: Could not get transform for '{geom_name}': {e}")
        
        transformed_geoms.append(geom_copy)
    
    # Concatenate all geometries into a single mesh
    if len(transformed_geoms) == 1:
        final_mesh = transformed_geoms[0]
    else:
        final_mesh = trimesh.util.concatenate(transformed_geoms)
    
    # Export the fixed mesh
    if output_path is None:
        output_path = input_path
    
    final_mesh.export(output_path)
    
    if verbose:
        print(f"  Saved fixed mesh to: {output_path}")
        print(f"    Vertices: {len(final_mesh.vertices)}")
        print(f"    Faces: {len(final_mesh.faces)}")
        print(f"    Bounds: {final_mesh.bounds}")
    
    return True


def fix_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    suffix: str = "_fixed",
    verbose: bool = True
) -> dict:
    """
    Fix all GLB files in a directory.
    
    Args:
        input_dir: Path to the input directory
        output_dir: Path for the output directory (if None, saves in same directory)
        suffix: Suffix to add to output filenames (e.g., "_fixed")
        verbose: Print detailed information
        
    Returns:
        Dict with counts of processed, fixed, and skipped files
    """
    input_path = Path(input_dir)
    
    if not input_path.is_dir():
        raise ValueError(f"Not a directory: {input_dir}")
    
    # Find all GLB files
    glb_files = list(input_path.glob("*.glb")) + list(input_path.glob("*.GLB"))
    gltf_files = list(input_path.glob("*.gltf")) + list(input_path.glob("*.GLTF"))
    all_files = glb_files + gltf_files
    
    if not all_files:
        print(f"No GLB/GLTF files found in: {input_dir}")
        return {'processed': 0, 'fixed': 0, 'skipped': 0}
    
    print(f"\nFound {len(all_files)} GLB/GLTF files in: {input_dir}")
    
    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None
    
    stats = {'processed': 0, 'fixed': 0, 'skipped': 0, 'errors': 0}
    
    for input_file in sorted(all_files):
        stats['processed'] += 1
        
        # Determine output filename
        if output_path:
            # Save to output directory with same name (no suffix needed)
            output_file = output_path / input_file.name
        else:
            # Save in same directory with suffix
            stem = input_file.stem
            ext = input_file.suffix
            output_file = input_file.parent / f"{stem}{suffix}{ext}"
        
        try:
            was_fixed = fix_glb_transforms(
                str(input_file),
                str(output_file),
                verbose=verbose
            )
            
            if was_fixed:
                stats['fixed'] += 1
            else:
                stats['skipped'] += 1
                
        except Exception as e:
            print(f"  ERROR processing {input_file.name}: {e}")
            stats['errors'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Fix GLB files by baking scene graph transforms into vertex positions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "path",
        help="Path to a GLB file or directory containing GLB files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory (default: same directory with suffix)"
    )
    parser.add_argument(
        "--suffix", "-s",
        default="_fixed",
        help="Suffix to add to output filenames (default: _fixed)"
    )
    parser.add_argument(
        "--in-place", "-i",
        action="store_true",
        help="Overwrite input files (use with caution!)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Less verbose output"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.path)
    
    if not input_path.exists():
        print(f"Error: Path does not exist: {args.path}")
        sys.exit(1)
    
    if input_path.is_file():
        # Single file mode
        if args.in_place:
            output_path = str(input_path)
        elif args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / input_path.name)
        else:
            stem = input_path.stem
            ext = input_path.suffix
            output_path = str(input_path.parent / f"{stem}{args.suffix}{ext}")
        
        was_fixed = fix_glb_transforms(
            str(input_path),
            output_path,
            verbose=not args.quiet
        )
        
        if was_fixed:
            print(f"\nFixed mesh saved to: {output_path}")
        else:
            print(f"\nNo fixes needed (transforms were already identity)")
            
    else:
        # Directory mode
        if args.in_place:
            # For in-place, we use empty suffix and no output dir
            stats = fix_directory(
                str(input_path),
                output_dir=None,
                suffix="",  # This will overwrite files
                verbose=not args.quiet
            )
        else:
            stats = fix_directory(
                str(input_path),
                output_dir=args.output_dir,
                suffix=args.suffix,
                verbose=not args.quiet
            )
        
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Processed: {stats['processed']} files")
        print(f"  Fixed: {stats['fixed']} files")
        print(f"  Skipped (no changes needed): {stats['skipped']} files")
        if stats.get('errors', 0) > 0:
            print(f"  Errors: {stats['errors']} files")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
