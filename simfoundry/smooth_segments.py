#!/usr/bin/env python3
"""
Smooth segment boundaries on segmented meshes.

This module provides functions to smooth jagged boundaries between mesh segments
using adjacency-based regularization techniques.

Usage:
    # As a module
    from smooth_segments import smooth_segment_boundaries
    smoothed_face_ids = smooth_segment_boundaries(mesh, face_ids, iterations=3)
    
    # As a script
    python smooth_segments.py mesh.glb face_ids.npy --output smoothed_face_ids.npy --iterations 3
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, Optional
from collections import Counter

import numpy as np
import trimesh
from tqdm import tqdm


def build_face_adjacency(mesh: trimesh.Trimesh) -> dict:
    """
    Build a face adjacency dictionary from mesh.
    
    Args:
        mesh: Trimesh object
        
    Returns:
        Dict mapping face_idx -> list of adjacent face indices
    """
    # Use trimesh's face_adjacency which gives pairs of adjacent faces
    adjacency = mesh.face_adjacency
    
    # Build adjacency dict
    adj_dict = {i: [] for i in range(len(mesh.faces))}
    for f1, f2 in adjacency:
        adj_dict[f1].append(f2)
        adj_dict[f2].append(f1)
    
    return adj_dict


def smooth_segment_boundaries(
    mesh: trimesh.Trimesh,
    face_ids: np.ndarray,
    iterations: int = 3,
    boundary_only: bool = True,
    min_segment_size: int = 10,
    verbose: bool = False
) -> np.ndarray:
    """
    Smooth segment boundaries using majority voting among neighbors.
    
    This function iteratively smooths segment boundaries by having each 
    boundary face adopt the label of the majority of its neighbors.
    
    Args:
        mesh: Trimesh object
        face_ids: Array of segment IDs for each face
        iterations: Number of smoothing iterations
        boundary_only: If True, only update faces on segment boundaries
        min_segment_size: Don't allow segments to shrink below this size
        verbose: Print progress information
        
    Returns:
        Smoothed face_ids array
    """
    if verbose:
        print(f"Smoothing segment boundaries ({iterations} iterations)...")
    
    # Build adjacency
    adj_dict = build_face_adjacency(mesh)
    
    # Work with a copy
    smoothed = face_ids.copy()
    
    for iteration in range(iterations):
        new_smoothed = smoothed.copy()
        changes = 0
        
        # Find boundary faces (faces with neighbors of different segments)
        boundary_faces = []
        for face_idx in range(len(mesh.faces)):
            neighbors = adj_dict[face_idx]
            if not neighbors:
                continue
            
            current_label = smoothed[face_idx]
            neighbor_labels = [smoothed[n] for n in neighbors]
            
            # Is this a boundary face?
            if boundary_only:
                if all(nl == current_label for nl in neighbor_labels):
                    continue  # Not a boundary face
            
            boundary_faces.append(face_idx)
        
        if verbose:
            print(f"  Iteration {iteration + 1}: {len(boundary_faces)} boundary faces")
        
        # Process boundary faces
        for face_idx in boundary_faces:
            neighbors = adj_dict[face_idx]
            current_label = smoothed[face_idx]
            
            # Count labels in neighborhood (including self)
            all_labels = [current_label] + [smoothed[n] for n in neighbors]
            label_counts = Counter(all_labels)
            
            # Get majority label (excluding self to avoid bias toward current label)
            neighbor_labels_only = [smoothed[n] for n in neighbors]
            if not neighbor_labels_only:
                continue
                
            neighbor_counts = Counter(neighbor_labels_only)
            majority_label, majority_count = neighbor_counts.most_common(1)[0]
            
            # Change if majority of neighbors have a different label
            # Use >= instead of > for ties, and only require > half of neighbors
            if majority_label != current_label and majority_count >= len(neighbor_labels_only) / 2:
                new_smoothed[face_idx] = majority_label
                changes += 1
        
        smoothed = new_smoothed
        
        if verbose:
            print(f"    Changed {changes} faces")
        
        if changes == 0:
            if verbose:
                print(f"  Converged after {iteration + 1} iterations")
            break
    
    # Ensure we don't eliminate small segments completely
    original_segments = set(np.unique(face_ids))
    final_segments = set(np.unique(smoothed))
    lost_segments = original_segments - final_segments
    
    if lost_segments and verbose:
        print(f"  Warning: {len(lost_segments)} segments were eliminated during smoothing")
    
    return smoothed


def remove_small_disconnected_regions(
    mesh: trimesh.Trimesh,
    face_ids: np.ndarray,
    min_region_ratio: float = 0.1,
    min_region_faces: int = 50,
    verbose: bool = False
) -> np.ndarray:
    """
    Remove small disconnected regions within each segment by reassigning them
    to neighboring segments.
    
    This is particularly useful for cleaning up jagged boundaries where
    isolated face clusters from one segment are embedded in another.
    
    Args:
        mesh: Trimesh object
        face_ids: Array of segment IDs for each face
        min_region_ratio: Minimum ratio of region size to largest region in segment.
                         Regions smaller than this ratio get reassigned.
        min_region_faces: Minimum number of faces for a region to be kept regardless of ratio.
        verbose: Print progress information
        
    Returns:
        Cleaned face_ids array with small disconnected regions reassigned
    """
    if verbose:
        print(f"Removing small disconnected regions (min_ratio={min_region_ratio}, min_faces={min_region_faces})...")
    
    # Build adjacency
    adj_dict = build_face_adjacency(mesh)
    
    # Work with a copy
    cleaned = face_ids.copy()
    
    # Get unique segment labels (excluding invalid ones)
    unique_labels = [l for l in np.unique(face_ids) if l >= 0]
    
    total_reassigned = 0
    
    for label in unique_labels:
        # Find all faces with this label
        label_faces = np.where(cleaned == label)[0]
        
        if len(label_faces) == 0:
            continue
        
        # Find connected components within this segment
        components = []
        visited = set()
        
        for start_face in label_faces:
            if start_face in visited:
                continue
            
            # BFS to find connected component
            component = []
            queue = [start_face]
            
            while queue:
                face_idx = queue.pop(0)
                if face_idx in visited:
                    continue
                if cleaned[face_idx] != label:
                    continue
                    
                visited.add(face_idx)
                component.append(face_idx)
                
                # Add neighbors with same label
                for neighbor in adj_dict[face_idx]:
                    if neighbor not in visited and cleaned[neighbor] == label:
                        queue.append(neighbor)
            
            if component:
                components.append(component)
        
        if len(components) <= 1:
            continue  # Only one component, nothing to do
        
        # Sort components by size (largest first)
        components.sort(key=len, reverse=True)
        largest_size = len(components[0])
        
        if verbose:
            print(f"  Segment {label}: {len(components)} connected regions "
                  f"(largest: {largest_size} faces)")
        
        # Process smaller components
        for component in components[1:]:
            # Check if this component should be reassigned
            if len(component) >= min_region_faces:
                continue  # Large enough to keep
            
            ratio = len(component) / largest_size
            if ratio >= min_region_ratio:
                continue  # Large enough relative to main region
            
            # Find the best neighboring segment to reassign to
            neighbor_labels = Counter()
            for face_idx in component:
                for neighbor in adj_dict[face_idx]:
                    neighbor_label = cleaned[neighbor]
                    if neighbor_label != label and neighbor_label >= 0:
                        neighbor_labels[neighbor_label] += 1
            
            if not neighbor_labels:
                # No valid neighbors, keep original label
                continue
            
            # Assign to the most common neighboring segment
            new_label = neighbor_labels.most_common(1)[0][0]
            
            # Reassign all faces in this component
            for face_idx in component:
                cleaned[face_idx] = new_label
            
            total_reassigned += len(component)
            
            if verbose:
                print(f"    Reassigned {len(component)} faces from segment {label} to {new_label}")
    
    if verbose:
        print(f"  Total: {total_reassigned} faces reassigned")
    
    return cleaned


def smooth_with_erosion_dilation(
    mesh: trimesh.Trimesh,
    face_ids: np.ndarray,
    erosion_iterations: int = 1,
    dilation_iterations: int = 1,
    verbose: bool = False
) -> np.ndarray:
    """
    Smooth segment boundaries using morphological erosion followed by dilation.
    
    This is similar to morphological opening which removes small protrusions.
    
    Args:
        mesh: Trimesh object
        face_ids: Array of segment IDs for each face
        erosion_iterations: Number of erosion iterations
        dilation_iterations: Number of dilation iterations
        verbose: Print progress information
        
    Returns:
        Smoothed face_ids array
    """
    if verbose:
        print(f"Smoothing with erosion ({erosion_iterations}) + dilation ({dilation_iterations})...")
    
    adj_dict = build_face_adjacency(mesh)
    smoothed = face_ids.copy()
    unique_labels = np.unique(face_ids)
    
    # Erosion: shrink each segment by removing boundary faces
    for iteration in range(erosion_iterations):
        new_smoothed = smoothed.copy()
        
        for label in unique_labels:
            if label < 0:  # Skip invalid labels
                continue
                
            # Find faces of this label
            label_faces = np.where(smoothed == label)[0]
            
            for face_idx in label_faces:
                neighbors = adj_dict[face_idx]
                neighbor_labels = [smoothed[n] for n in neighbors]
                
                # If any neighbor has a different label, this is a boundary face
                if any(nl != label for nl in neighbor_labels):
                    # Get the most common non-self label
                    other_labels = [nl for nl in neighbor_labels if nl != label]
                    if other_labels:
                        most_common = Counter(other_labels).most_common(1)[0][0]
                        new_smoothed[face_idx] = most_common
        
        smoothed = new_smoothed
    
    # Dilation: expand each segment into neighboring unlabeled/eroded areas
    for iteration in range(dilation_iterations):
        new_smoothed = smoothed.copy()
        
        for label in unique_labels:
            if label < 0:
                continue
                
            # Find faces of this label
            label_faces = np.where(smoothed == label)[0]
            
            for face_idx in label_faces:
                neighbors = adj_dict[face_idx]
                
                # Expand into neighbors
                for neighbor in neighbors:
                    neighbor_labels = [smoothed[n] for n in adj_dict[neighbor]]
                    label_count = sum(1 for nl in neighbor_labels if nl == label)
                    
                    if label_count > len(neighbor_labels) // 2:
                        new_smoothed[neighbor] = label
        
        smoothed = new_smoothed
    
    return smoothed


def gaussian_smooth_boundaries(
    mesh: trimesh.Trimesh,
    face_ids: np.ndarray,
    sigma: float = 1.0,
    iterations: int = 2,
    verbose: bool = False
) -> np.ndarray:
    """
    Smooth segment boundaries using a Gaussian-weighted voting scheme.
    
    Faces vote for labels based on distance-weighted contributions from neighbors.
    
    Args:
        mesh: Trimesh object
        face_ids: Array of segment IDs for each face
        sigma: Standard deviation for Gaussian weighting (in mesh units)
        iterations: Number of smoothing iterations
        verbose: Print progress information
        
    Returns:
        Smoothed face_ids array
    """
    if verbose:
        print(f"Gaussian smoothing (sigma={sigma}, {iterations} iterations)...")
    
    # Compute face centers
    face_centers = mesh.triangles_center
    
    # Build adjacency with distances
    adj_dict = build_face_adjacency(mesh)
    
    smoothed = face_ids.copy()
    unique_labels = np.unique(face_ids)
    
    for iteration in range(iterations):
        new_smoothed = smoothed.copy()
        changes = 0
        
        for face_idx in range(len(mesh.faces)):
            neighbors = adj_dict[face_idx]
            if not neighbors:
                continue
            
            current_label = smoothed[face_idx]
            current_center = face_centers[face_idx]
            
            # Compute weighted votes
            label_weights = {}
            for neighbor in neighbors:
                neighbor_center = face_centers[neighbor]
                distance = np.linalg.norm(current_center - neighbor_center)
                weight = np.exp(-distance**2 / (2 * sigma**2))
                
                neighbor_label = smoothed[neighbor]
                label_weights[neighbor_label] = label_weights.get(neighbor_label, 0) + weight
            
            # Add self vote
            label_weights[current_label] = label_weights.get(current_label, 0) + 1.0
            
            # Get highest weighted label
            best_label = max(label_weights, key=label_weights.get)
            
            if best_label != current_label:
                new_smoothed[face_idx] = best_label
                changes += 1
        
        smoothed = new_smoothed
        
        if verbose:
            print(f"  Iteration {iteration + 1}: {changes} changes")
        
        if changes == 0:
            break
    
    return smoothed


def apply_smoothing_to_mesh(
    mesh_path: str,
    face_ids_path: str,
    output_face_ids_path: str,
    output_mesh_path: Optional[str] = None,
    method: str = 'majority',
    iterations: int = 3,
    verbose: bool = True
) -> Tuple[np.ndarray, trimesh.Trimesh]:
    """
    Load mesh and face_ids, apply smoothing, and save results.
    
    Args:
        mesh_path: Path to mesh file (GLB, OBJ, etc.)
        face_ids_path: Path to face_ids.npy file
        output_face_ids_path: Path to save smoothed face_ids
        output_mesh_path: Optional path to save colored mesh with smoothed segments
        method: Smoothing method ('majority', 'erosion_dilation', 'gaussian')
        iterations: Number of smoothing iterations
        verbose: Print progress information
        
    Returns:
        Tuple of (smoothed_face_ids, mesh)
    """
    if verbose:
        print(f"Loading mesh from: {mesh_path}")
        print(f"Loading face_ids from: {face_ids_path}")
    
    # Load mesh
    loaded = trimesh.load(mesh_path)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    
    # Load face_ids
    face_ids = np.load(face_ids_path)
    
    if verbose:
        print(f"Mesh: {len(mesh.faces)} faces, {len(np.unique(face_ids))} segments")
    
    # Apply smoothing
    if method == 'majority':
        smoothed = smooth_segment_boundaries(mesh, face_ids, iterations=iterations, verbose=verbose)
    elif method == 'erosion_dilation':
        smoothed = smooth_with_erosion_dilation(mesh, face_ids, 
                                                 erosion_iterations=iterations, 
                                                 dilation_iterations=iterations,
                                                 verbose=verbose)
    elif method == 'gaussian':
        # Estimate sigma based on mesh scale
        sigma = np.max(mesh.extents) / 50
        smoothed = gaussian_smooth_boundaries(mesh, face_ids, sigma=sigma, 
                                               iterations=iterations, verbose=verbose)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Save smoothed face_ids
    np.save(output_face_ids_path, smoothed)
    if verbose:
        print(f"Saved smoothed face_ids to: {output_face_ids_path}")
    
    # Optionally save colored mesh
    if output_mesh_path:
        from matplotlib import colormaps
        num_segments = len(np.unique(smoothed))
        cmap = colormaps['tab20'].resampled(max(num_segments, 20))
        
        face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
        for face_idx, seg_id in enumerate(smoothed):
            if seg_id >= 0:
                color = np.array(cmap(seg_id % 20)[:3]) * 255
                face_colors[face_idx] = [int(color[0]), int(color[1]), int(color[2]), 255]
        
        mesh.visual.face_colors = face_colors
        mesh.export(output_mesh_path)
        if verbose:
            print(f"Saved smoothed mesh to: {output_mesh_path}")
    
    return smoothed, mesh


def main():
    parser = argparse.ArgumentParser(
        description="Smooth segment boundaries on segmented meshes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("mesh_path", help="Path to mesh file (GLB, OBJ, etc.)")
    parser.add_argument("face_ids_path", help="Path to face_ids.npy file")
    parser.add_argument("-o", "--output", required=True, help="Output path for smoothed face_ids.npy")
    parser.add_argument("-m", "--mesh-output", help="Optional output path for colored mesh")
    parser.add_argument("--method", choices=['majority', 'erosion_dilation', 'gaussian'],
                        default='majority', help="Smoothing method (default: majority)")
    parser.add_argument("-i", "--iterations", type=int, default=3,
                        help="Number of smoothing iterations (default: 3)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    
    apply_smoothing_to_mesh(
        args.mesh_path,
        args.face_ids_path,
        args.output,
        output_mesh_path=args.mesh_output,
        method=args.method,
        iterations=args.iterations,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
