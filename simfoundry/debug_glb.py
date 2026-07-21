#!/usr/bin/env python3
"""
Debug script to inspect GLB/GLTF file structure.

Usage:
    python debug_glb.py <path_to_glb_file>
    
Example:
    python debug_glb.py ../simfoundry_results/trash_can_hunyuan/black_trash_can_cousin_003_v3/s3_segment_mesh/black_trash_can_cousin_003_v3_cleaned.glb
"""

import argparse
import sys
import numpy as np
import trimesh


def debug_glb(glb_path: str, verbose: bool = True):
    """
    Debug a GLB/GLTF file and print its structure.
    
    Args:
        glb_path: Path to the GLB/GLTF file
        verbose: If True, print detailed information
    """
    print(f"\n{'='*80}")
    print(f"Debugging GLB: {glb_path}")
    print(f"{'='*80}\n")
    
    # Load the mesh
    loaded = trimesh.load(glb_path)
    
    print(f"Loaded type: {type(loaded).__name__}")
    
    if isinstance(loaded, trimesh.Scene):
        print("\n--- SCENE INFORMATION ---")
        print(f"Base frame: {loaded.graph.base_frame}")
        
        # Geometry information
        print(f"\n--- GEOMETRIES ({len(loaded.geometry)}) ---")
        for i, (geom_name, geom) in enumerate(loaded.geometry.items()):
            print(f"  [{i}] Geometry name: '{geom_name}'")
            print(f"      Type: {type(geom).__name__}")
            if hasattr(geom, 'vertices'):
                print(f"      Vertices: {len(geom.vertices)}")
                print(f"      Faces: {len(geom.faces)}")
                print(f"      Bounds: {geom.bounds}")
                print(f"      Centroid: {geom.centroid}")
            if hasattr(geom, 'visual'):
                visual_type = type(geom.visual).__name__
                has_uv = hasattr(geom.visual, 'uv') and geom.visual.uv is not None
                print(f"      Visual type: {visual_type}, has_uv: {has_uv}")
            print()
        
        # Scene graph information
        print(f"\n--- SCENE GRAPH ---")
        graph = loaded.graph
        print(f"Nodes ({len(graph.nodes)}): {list(graph.nodes)}")
        
        # Node data
        print(f"\n--- NODE DATA ---")
        for node_name, node_data in graph.transforms.node_data.items():
            print(f"  Node '{node_name}': {node_data}")
        
        # Parent relationships
        print(f"\n--- PARENT RELATIONSHIPS ---")
        parents = graph.transforms.parents
        print(f"Parents dict ({len(parents)} entries): {parents}")
        
        # Geometry to node mapping
        print(f"\n--- GEOMETRY TO NODE MAPPING ---")
        geometry_nodes = graph.geometry_nodes
        print(f"geometry_nodes: {dict(geometry_nodes)}")
        
        # Nodes with geometry
        print(f"\n--- NODES WITH GEOMETRY ---")
        nodes_geometry = graph.nodes_geometry
        print(f"nodes_geometry: {nodes_geometry}")
        
        # Edge data (transforms)
        print(f"\n--- EDGE DATA (TRANSFORMS) ---")
        for edge, data in graph.transforms.edge_data.items():
            print(f"  Edge {edge}:")
            if 'matrix' in data:
                matrix = data['matrix']
                is_identity = np.allclose(matrix, np.eye(4))
                print(f"    Matrix: {'Identity' if is_identity else 'Non-identity'}")
                if not is_identity and verbose:
                    print(f"    {matrix}")
            if 'geometry' in data:
                print(f"    Geometry: {data['geometry']}")
        
        # Try to get transforms for each geometry
        print(f"\n--- TRANSFORM LOOKUP TEST ---")
        for geom_name in loaded.geometry.keys():
            print(f"  Testing geometry '{geom_name}':")
            
            # Method 1: Direct lookup (might fail)
            try:
                transform, geom_ref = loaded.graph[geom_name]
                is_identity = np.allclose(transform, np.eye(4))
                print(f"    Direct lookup: Success (identity={is_identity})")
            except Exception as e:
                print(f"    Direct lookup: FAILED - {type(e).__name__}: {e}")
            
            # Method 2: Via geometry_nodes mapping (correct way)
            node_names = geometry_nodes.get(geom_name, [])
            if node_names:
                for node_name in node_names:
                    try:
                        transform, geom_ref = loaded.graph[node_name]
                        is_identity = np.allclose(transform, np.eye(4))
                        print(f"    Via node '{node_name}': Success (identity={is_identity})")
                    except Exception as e:
                        print(f"    Via node '{node_name}': FAILED - {type(e).__name__}: {e}")
            else:
                print(f"    No node mapping found for this geometry")
            print()
        
        # Scene bounds
        print(f"\n--- SCENE BOUNDS ---")
        print(f"Scene bounds: {loaded.bounds}")
        print(f"Scene centroid: {loaded.centroid}")
        print(f"Scene extents: {loaded.extents}")
        
    elif isinstance(loaded, trimesh.Trimesh):
        print("\n--- SINGLE MESH (not a Scene) ---")
        print(f"Vertices: {len(loaded.vertices)}")
        print(f"Faces: {len(loaded.faces)}")
        print(f"Bounds: {loaded.bounds}")
        print(f"Centroid: {loaded.centroid}")
        visual_type = type(loaded.visual).__name__
        has_uv = hasattr(loaded.visual, 'uv') and loaded.visual.uv is not None
        print(f"Visual type: {visual_type}, has_uv: {has_uv}")
    
    else:
        print(f"Unknown type: {type(loaded)}")
    
    print(f"\n{'='*80}")
    print("Debug complete")
    print(f"{'='*80}\n")
    
    return loaded


def main():
    parser = argparse.ArgumentParser(
        description="Debug GLB/GLTF file structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("glb_path", help="Path to the GLB/GLTF file to debug")
    parser.add_argument("-q", "--quiet", action="store_true", 
                        help="Less verbose output")
    
    args = parser.parse_args()
    
    debug_glb(args.glb_path, verbose=not args.quiet)


if __name__ == "__main__":
    main()
