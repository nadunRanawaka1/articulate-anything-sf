#!/usr/bin/env python3
"""
Fix GLB files by making geometry names match their node names.

This script fixes the naming mismatch between geometry names and node names
in GLTF/GLB files, which can cause issues with scene graph lookups.
The transforms are preserved (not baked into vertices).

Usage:
    python fix_glb_names.py <directory_or_file> [--output-dir <output_dir>] [--suffix <suffix>]
    
Examples:
    # Fix all GLB files in a directory (saves to same directory with _fixed suffix)
    python fix_glb_names.py /path/to/meshes/
    
    # Fix all GLB files and save to a different directory
    python fix_glb_names.py /path/to/meshes/ --output-dir /path/to/output/
    
    # Fix a single file
    python fix_glb_names.py /path/to/mesh.glb
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import OrderedDict

import numpy as np
import trimesh


def fix_glb_names(
    input_path: str,
    output_path: Optional[str] = None,
    verbose: bool = True
) -> bool:
    """
    Fix a GLB file by making geometry names match their node names.
    
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
        if output_path and output_path != input_path:
            loaded.export(output_path)
            print(f"  Copied to: {output_path}")
        return False
    
    # Get geometry to node mapping
    geometry_nodes = loaded.graph.geometry_nodes  # {geometry_name: [node_names]}
    
    # Check which geometries need renaming
    renames_needed: List[Tuple[str, str]] = []  # [(old_name, new_name), ...]
    
    for geom_name in loaded.geometry.keys():
        node_names = geometry_nodes.get(geom_name, [])
        if node_names:
            node_name = node_names[0]
            if geom_name != node_name:
                renames_needed.append((geom_name, node_name))
    
    if not renames_needed:
        if verbose:
            print(f"  Skipping: All geometry names already match node names")
        if output_path and output_path != input_path:
            loaded.export(output_path)
            print(f"  Copied to: {output_path}")
        return False
    
    if verbose:
        print(f"  Found {len(renames_needed)} geometries to rename:")
        for old_name, new_name in renames_needed:
            print(f"    '{old_name}' -> '{new_name}'")
    
    # Create a new scene with renamed geometries
    # We need to rebuild the scene because geometry names are dict keys
    
    new_geometry = OrderedDict()
    name_mapping = {}  # old_name -> new_name
    
    for geom_name, geom in loaded.geometry.items():
        # Check if this geometry needs renaming
        node_names = geometry_nodes.get(geom_name, [])
        if node_names and geom_name != node_names[0]:
            new_name = node_names[0]
            name_mapping[geom_name] = new_name
        else:
            new_name = geom_name
            name_mapping[geom_name] = geom_name
        
        new_geometry[new_name] = geom
    
    # Create a new scene with the renamed geometries
    new_scene = trimesh.Scene()
    new_scene.graph.base_frame = loaded.graph.base_frame
    
    # Add geometries and rebuild the graph
    for old_name, new_name in name_mapping.items():
        geom = new_geometry[new_name]
        node_names = geometry_nodes.get(old_name, [])
        
        if node_names:
            node_name = node_names[0]
            # Get the transform for this node
            try:
                transform, _ = loaded.graph[node_name]
            except (KeyError, TypeError, ValueError):
                transform = np.eye(4)
            
            # Add to new scene with the node name as both geometry and node name
            new_scene.add_geometry(
                geom,
                node_name=node_name,
                geom_name=new_name,  # This should now match node_name
                transform=transform
            )
        else:
            # No node mapping, just add with geometry name
            new_scene.add_geometry(geom, geom_name=new_name)
    
    # Export the fixed mesh
    if output_path is None:
        output_path = input_path
    
    new_scene.export(output_path)
    
    if verbose:
        print(f"  Saved fixed mesh to: {output_path}")
        
        # Verify the fix
        verify = trimesh.load(output_path)
        if isinstance(verify, trimesh.Scene):
            print(f"  Verification:")
            print(f"    Geometries: {list(verify.geometry.keys())}")
            print(f"    Nodes: {list(verify.graph.nodes)}")
            verify_geom_nodes = verify.graph.geometry_nodes
            print(f"    Geometry->Node mapping: {dict(verify_geom_nodes)}")
            
            # Check if direct lookup works now
            for geom_name in verify.geometry.keys():
                try:
                    transform, _ = verify.graph[geom_name]
                    print(f"    Direct lookup '{geom_name}': SUCCESS")
                except Exception as e:
                    print(f"    Direct lookup '{geom_name}': FAILED - {e}")
    
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
            output_file = output_path / input_file.name
        else:
            stem = input_file.stem
            ext = input_file.suffix
            output_file = input_file.parent / f"{stem}{suffix}{ext}"
        
        try:
            was_fixed = fix_glb_names(
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
            import traceback
            traceback.print_exc()
            stats['errors'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Fix GLB files by making geometry names match node names",
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
        
        was_fixed = fix_glb_names(
            str(input_path),
            output_path,
            verbose=not args.quiet
        )
        
        if was_fixed:
            print(f"\nFixed mesh saved to: {output_path}")
        else:
            print(f"\nNo fixes needed (names already match)")
            
    else:
        # Directory mode
        if args.in_place:
            stats = fix_directory(
                str(input_path),
                output_dir=None,
                suffix="",
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
