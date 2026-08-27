"""Segment geometry preprocessing utilities."""

from dataclasses import dataclass
import numpy as np
import trimesh


@dataclass
class SegmentGeometry:
    """Precomputed geometry data for a single segment."""
    vertices_base: np.ndarray  # Base vertices without explosion offset
    direction: np.ndarray      # Direction vector for explosion
    faces: np.ndarray          # Remapped face indices
    face_count: int            # Number of faces in this segment


def precompute_segment_geometry(
    mesh: trimesh.Trimesh,
    face2label: np.ndarray,
    all_segments: list
) -> tuple[dict, float, np.ndarray]:
    """
    Precompute geometry data for all segments.
    
    Args:
        mesh: The mesh to process
        face2label: Per-face segment labels
        all_segments: List of valid segment IDs
        
    Returns:
        Tuple of (segment_data dict, scene_scale, mesh_centroid)
    """
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    mesh_centroid = vertices.mean(axis=0)
    face_centroids = vertices[faces].mean(axis=1)
    scene_scale = float(np.max(vertices.max(axis=0) - vertices.min(axis=0)))
    
    segment_data = {}
    for seg_id in all_segments:
        seg_mask = face2label == seg_id
        seg_face_indices = np.where(seg_mask)[0]
        if len(seg_face_indices) == 0:
            continue
        
        seg_faces = faces[seg_face_indices]
        vertices_used = np.unique(seg_faces.flatten())
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(vertices_used)}
        
        # Compute explosion direction
        seg_centroid = face_centroids[seg_face_indices].mean(axis=0)
        direction = seg_centroid - mesh_centroid
        distance = np.linalg.norm(direction)
        if distance > 1e-6:
            direction = direction / distance
        else:
            direction = np.array([0, 0, 0])
        
        # Store base vertices and remapped faces
        seg_vertices_base = vertices[vertices_used].copy()
        seg_faces_remapped = np.array([[vertex_map[v] for v in face] for face in seg_faces])
        
        segment_data[seg_id] = SegmentGeometry(
            vertices_base=seg_vertices_base,
            direction=direction,
            faces=seg_faces_remapped,
            face_count=len(seg_face_indices)
        )
    
    return segment_data, scene_scale, mesh_centroid


def compute_exploded_vertices(
    segment: SegmentGeometry,
    explosion_factor: float,
    scene_scale: float
) -> np.ndarray:
    """
    Compute exploded vertex positions for a segment.
    
    Args:
        segment: The segment geometry
        explosion_factor: How much to explode (0 = no explosion)
        scene_scale: Scale factor for consistent explosion across meshes
        
    Returns:
        Exploded vertex positions
    """
    from .styles import EXPLOSION_SCALE
    offset = segment.direction * explosion_factor * scene_scale * EXPLOSION_SCALE
    return segment.vertices_base + offset


def find_coplanar_faces(
    mesh: trimesh.Trimesh,
    face2label: np.ndarray,
    segment_id: int,
    clicked_face_idx: int,
    angle_threshold_deg: float = 15.0,
    distance_threshold_ratio: float = 0.02,
    cross_segment: bool = False
) -> np.ndarray:
    """
    Find all faces in a segment that are coplanar with a clicked face.
    
    Uses flood-fill from the clicked face to find connected coplanar regions.
    
    Args:
        mesh: The mesh
        face2label: Per-face segment labels
        segment_id: The segment to search within
        clicked_face_idx: Index of the face the user clicked
        angle_threshold_deg: Max angle difference from clicked face normal (degrees)
        distance_threshold_ratio: Max distance from plane as ratio of mesh size
        cross_segment: If True, include faces from ALL segments (not just segment_id)
        
    Returns:
        Array of face indices that are coplanar with the clicked face
    """
    # IMPORTANT: Merge duplicate vertices for proper edge-based adjacency
    # Many meshes (especially from GLB/GLTF) have per-face vertices which breaks edge sharing
    mesh_copy = mesh.copy()
    original_vertex_count = len(mesh_copy.vertices)
    mesh_copy.merge_vertices(merge_tex=True, merge_norm=True)  # position-only: keep adjacency across texture/normal seams
    merged_vertex_count = len(mesh_copy.vertices)
    
    # Face count should remain the same after merge
    assert len(mesh_copy.faces) == len(mesh.faces), "Face count changed after merge_vertices!"
    
    vertices = np.array(mesh_copy.vertices)
    faces = np.array(mesh_copy.faces)
    
    # Get faces to consider
    if cross_segment:
        # Include ALL faces in the mesh
        segment_face_indices = np.arange(len(faces))
        print(f"[find_coplanar_faces] Cross-segment mode: considering ALL {len(segment_face_indices)} faces")
    else:
        # Only faces in this segment
        segment_mask = face2label == segment_id
        segment_face_indices = np.where(segment_mask)[0]
    
    if clicked_face_idx not in segment_face_indices:
        print(f"[find_coplanar_faces] ERROR: clicked_face_idx {clicked_face_idx} not in segment {segment_id}")
        return np.array([], dtype=int)
    
    # Compute face normals and centroids
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_norms[face_norms < 1e-10] = 1  # Avoid division by zero
    face_normals = face_normals / face_norms
    
    face_centroids = (v0 + v1 + v2) / 3
    
    # Get clicked face properties
    clicked_normal = face_normals[clicked_face_idx]
    clicked_centroid = face_centroids[clicked_face_idx]
    
    # Compute mesh scale for distance threshold
    mesh_scale = np.max(vertices.max(axis=0) - vertices.min(axis=0))
    distance_threshold = mesh_scale * distance_threshold_ratio
    
    # Convert angle threshold to cosine
    angle_threshold_rad = np.radians(angle_threshold_deg)
    cos_threshold = np.cos(angle_threshold_rad)
    
    # Build face adjacency for flood fill (within segment only)
    # Two faces are adjacent if they share an edge
    from collections import defaultdict
    edge_to_faces = defaultdict(list)
    
    for face_idx in segment_face_indices:
        face = faces[face_idx]
        # Create edge keys (sorted vertex pairs)
        edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[2], face[0]]))
        ]
        for edge in edges:
            edge_to_faces[edge].append(face_idx)
    
    # Build adjacency dict
    face_neighbors = defaultdict(set)
    for edge, face_list in edge_to_faces.items():
        for i, f1 in enumerate(face_list):
            for f2 in face_list[i+1:]:
                face_neighbors[f1].add(f2)
                face_neighbors[f2].add(f1)
    
    # Debug: Check adjacency for clicked face
    clicked_neighbors = face_neighbors.get(clicked_face_idx, set())
    print(f"[find_coplanar_faces] Clicked face {clicked_face_idx} has {len(clicked_neighbors)} neighbors in segment")
    print(f"[find_coplanar_faces] Vertices merged: {original_vertex_count} -> {merged_vertex_count}")
    
    # Flood fill from clicked face
    coplanar_faces = set()
    queue = [clicked_face_idx]
    visited = set()
    
    while queue:
        face_idx = queue.pop(0)
        if face_idx in visited:
            continue
        visited.add(face_idx)
        
        # Check if this face is coplanar with clicked face
        normal = face_normals[face_idx]
        centroid = face_centroids[face_idx]
        
        # Check normal angle (dot product)
        cos_angle = abs(np.dot(normal, clicked_normal))
        if cos_angle < cos_threshold:
            continue
        
        # Check distance from plane
        # Distance = |dot(centroid - clicked_centroid, clicked_normal)|
        distance = abs(np.dot(centroid - clicked_centroid, clicked_normal))
        if distance > distance_threshold:
            continue
        
        # This face is coplanar - add it and explore neighbors
        coplanar_faces.add(face_idx)
        for neighbor in face_neighbors[face_idx]:
            if neighbor not in visited:
                queue.append(neighbor)
    
    print(f"[find_coplanar_faces] Found {len(coplanar_faces)} coplanar faces (angle<={angle_threshold_deg}°, dist<={distance_threshold_ratio*100:.1f}% mesh)")
    return np.array(sorted(coplanar_faces), dtype=int)


def split_segment_by_faces(
    face2label: np.ndarray,
    label2face_mask: np.ndarray,
    segment_id: int,
    faces_to_split: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Split faces from a segment into a new segment.
    
    Args:
        face2label: Per-face segment labels (will be modified in-place)
        label2face_mask: Binary mask [num_segments, num_faces] (will be modified)
        segment_id: The segment to split from
        faces_to_split: Face indices to move to new segment
        
    Returns:
        Tuple of (updated face2label, updated label2face_mask, new_segment_id)
    """
    if len(faces_to_split) == 0:
        return face2label, label2face_mask, -1

    # New segment ID: must equal its row index in label2face_mask. Taking the
    # max of (max label + 1) and the current row count guarantees that even
    # after a merge vacated a high segment id (merge keeps the dead row, so
    # max(face2label)+1 alone could reuse an id BELOW the appended row index,
    # writing the faces into an orphaned trailing row and leaving
    # label2face_mask[new_id] empty).
    new_segment_id = int(max(np.max(face2label) + 1, label2face_mask.shape[0]))

    # Update face2label
    face2label[faces_to_split] = new_segment_id

    # Update label2face_mask
    # Remove faces from old segment
    label2face_mask[segment_id, faces_to_split] = 0

    # Grow the mask so row new_segment_id exists, then set it
    n_new_rows = new_segment_id + 1 - label2face_mask.shape[0]
    pad = np.zeros((n_new_rows, label2face_mask.shape[1]), dtype=label2face_mask.dtype)
    label2face_mask = np.vstack([label2face_mask, pad])
    label2face_mask[new_segment_id, faces_to_split] = 1

    return face2label, label2face_mask, new_segment_id


def _find_connected_regions(face_list, face_neighbors, min_faces):
    """Helper to find connected components of faces using BFS."""
    face_set = set(face_list)
    visited = set()
    components = []
    
    for start_face in face_list:
        if start_face in visited:
            continue
        
        # BFS to find connected component
        component = []
        queue = [start_face]
        
        while queue:
            face_idx = queue.pop(0)
            if face_idx in visited:
                continue
            visited.add(face_idx)
            component.append(face_idx)
            
            for neighbor in face_neighbors[face_idx]:
                if neighbor in face_set and neighbor not in visited:
                    queue.append(neighbor)
        
        if len(component) >= min_faces:
            components.append(np.array(component, dtype=int))
    
    return components


def find_top_plane_faces(
    mesh: trimesh.Trimesh,
    face2label: np.ndarray,
    segment_id: int,
    up_vector: np.ndarray = None,
    angle_threshold_deg: float = 30.0,
    min_faces: int = 10,
    upward_faces_lb: float = 0.05,
    upward_faces_ub: float = 0.80,
    require_connectivity: bool = True,
    top_height_ratio: float = 1.0
) -> list[np.ndarray]:
    """
    Find all upward-facing planar regions in a segment.
    
    Returns a list of face index arrays, one per connected top-plane region.
    
    Args:
        mesh: The mesh
        face2label: Per-face segment labels
        segment_id: The segment to search within
        up_vector: Direction considered "up" (default: +Z)
        angle_threshold_deg: Max angle from up to be considered "top-facing"
        min_faces: Minimum faces for a region to be returned
        upward_faces_lb: Lower bound for upward faces percentage (0-1). Skip if below.
        upward_faces_ub: Upper bound for upward faces percentage (0-1). Skip if above.
        require_connectivity: If True, only return connected components. If False,
                              return ALL upward faces as a single group (ignores mesh topology).
        top_height_ratio: Only consider faces in the top X% of the segment's height
                          in the up direction (0-1). E.g., 0.2 = top 20%. Default 1.0 = all.
        
    Returns:
        List of numpy arrays, each containing face indices for one top-plane region
    """
    from collections import defaultdict
    
    if up_vector is None:
        up_vector = np.array([0, 0, 1])
    up_vector = np.array(up_vector) / np.linalg.norm(up_vector)
    
    # Merge duplicate vertices to get proper face adjacency
    # Many meshes (especially from GLB/GLTF) have per-face vertices
    mesh_copy = mesh.copy()
    mesh_copy.merge_vertices(merge_tex=True, merge_norm=True)  # position-only: keep adjacency across texture/normal seams
    
    vertices = np.array(mesh_copy.vertices)
    faces = np.array(mesh_copy.faces)
    
    # Build face neighbor dict from trimesh adjacency
    face_neighbors = defaultdict(set)
    for f1, f2 in mesh_copy.face_adjacency:
        face_neighbors[f1].add(f2)
        face_neighbors[f2].add(f1)
    
    # Get faces in this segment
    segment_mask = face2label == segment_id
    segment_face_indices = np.where(segment_mask)[0]
    
    if len(segment_face_indices) == 0:
        return []
    
    # Compute face centroids for height filtering
    face_centroids = mesh_copy.triangles_center
    
    # Calculate height bounds for this segment in the up direction
    # Use percentiles instead of min/max to be robust to outliers (like handles)
    segment_centroids = face_centroids[segment_face_indices]
    segment_heights = np.dot(segment_centroids, up_vector)
    
    # Use 5th and 95th percentiles to avoid outliers (handles, decorations)
    height_min = segment_heights.min()
    height_max = segment_heights.max()
    height_p5 = np.percentile(segment_heights, 5)
    height_p95 = np.percentile(segment_heights, 95)
    height_range = height_p95 - height_p5
    
    # Height threshold: only consider faces above this height
    if height_range > 1e-6 and top_height_ratio < 1.0:
        height_threshold = height_p95 - (height_range * top_height_ratio)
    else:
        height_threshold = height_p5  # No filtering if ratio is 1.0 or segment is flat
    
    print(f"\n  DEBUG Segment {segment_id}:")
    print(f"    Height min={height_min:.4f}, max={height_max:.4f}")
    print(f"    Height P5={height_p5:.4f}, P95={height_p95:.4f}, range={height_range:.4f}")
    print(f"    Height threshold={height_threshold:.4f} (top {top_height_ratio*100:.0f}%)")
    
    # Compute face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_norms[face_norms < 1e-10] = 1
    face_normals = face_normals / face_norms
    
    # Debug: Find faces with normals pointing STRAIGHT UP (dot > 0.9) and check their heights
    highly_upward_faces = []
    highly_upward_heights = []
    
    for face_idx in segment_face_indices:
        dot = np.dot(face_normals[face_idx], up_vector)
        if dot > 0.9:  # Nearly straight up
            face_height = np.dot(face_centroids[face_idx], up_vector)
            highly_upward_faces.append(face_idx)
            highly_upward_heights.append(face_height)
    
    print(f"    Faces with normals nearly straight up (dot > 0.9): {len(highly_upward_faces)}")
    if len(highly_upward_heights) > 0:
        heights_arr = np.array(highly_upward_heights)
        print(f"    Their heights: min={heights_arr.min():.4f}, max={heights_arr.max():.4f}, mean={heights_arr.mean():.4f}")
        print(f"    Height threshold is: {height_threshold:.4f}")
        above_threshold = np.sum(heights_arr >= height_threshold)
        print(f"    How many are above threshold: {above_threshold} ({100*above_threshold/len(heights_arr):.1f}%)")
    
    # Also check: what's at the MIDDLE height range?
    mid_height = (height_p5 + height_p95) / 2
    faces_at_mid = []
    for face_idx in segment_face_indices:
        face_height = np.dot(face_centroids[face_idx], up_vector)
        if abs(face_height - mid_height) < height_range * 0.1:  # Within 10% of middle
            dot = np.dot(face_normals[face_idx], up_vector)
            if dot > 0.9:
                faces_at_mid.append(face_idx)
    
    print(f"    Faces with upward normals near MIDDLE height: {len(faces_at_mid)}")
    
    # NEW DEBUG: Check normals of faces at the GEOMETRIC TOP of the segment
    top_height_cutoff = height_p95 - (height_range * 0.1)  # Top 10% by height
    faces_at_geometric_top = []
    for face_idx in segment_face_indices:
        face_height = np.dot(face_centroids[face_idx], up_vector)
        if face_height >= top_height_cutoff:
            faces_at_geometric_top.append(face_idx)
    
    if len(faces_at_geometric_top) > 0:
        top_normals = face_normals[faces_at_geometric_top]
        top_dots = np.dot(top_normals, up_vector)
        print(f"    --- Faces at GEOMETRIC TOP (height >= {top_height_cutoff:.4f}): {len(faces_at_geometric_top)} ---")
        print(f"    Their dot products with UP: min={top_dots.min():.4f}, max={top_dots.max():.4f}, mean={top_dots.mean():.4f}")
        print(f"    Pointing UP (dot > 0.5): {np.sum(top_dots > 0.5)}")
        print(f"    Pointing DOWN (dot < -0.5): {np.sum(top_dots < -0.5)}")
        print(f"    Pointing SIDEWAYS (-0.5 <= dot <= 0.5): {np.sum((top_dots >= -0.5) & (top_dots <= 0.5))}")
        # Sample a few
        sample_indices = faces_at_geometric_top[:5]
        for i, fidx in enumerate(sample_indices):
            normal = face_normals[fidx]
            dot = np.dot(normal, up_vector)
            height = np.dot(face_centroids[fidx], up_vector)
            print(f"      Sample face {fidx}: height={height:.4f}, normal={normal}, dot={dot:.4f}")
    
    # First, find ALL horizontal faces (facing up OR down) to check percentage bounds
    # This handles hollow shells where the lid has faces on BOTH sides
    cos_threshold = np.cos(np.radians(angle_threshold_deg))
    all_horizontal_faces = []
    all_upward_faces = []
    
    for face_idx in segment_face_indices:
        dot = np.dot(face_normals[face_idx], up_vector)
        if abs(dot) >= cos_threshold:  # Horizontal = facing up OR down
            all_horizontal_faces.append(face_idx)
        if dot >= cos_threshold:  # Upward only (for backwards compatibility)
            all_upward_faces.append(face_idx)
    
    if len(all_horizontal_faces) == 0:
        print(f"    No horizontal faces found (threshold={cos_threshold:.3f}), skipping")
        return []
    
    # Check if horizontal faces percentage is within bounds (BEFORE height filter)
    # Using horizontal (up + down) count handles hollow shells correctly
    horizontal_pct = len(all_horizontal_faces) / len(segment_face_indices)
    print(f"    Horizontal faces: {len(all_horizontal_faces)}/{len(segment_face_indices)} ({horizontal_pct*100:.1f}%)")
    print(f"    Bounds: {upward_faces_lb*100:.0f}% - {upward_faces_ub*100:.0f}%")
    
    if horizontal_pct < upward_faces_lb or horizontal_pct > upward_faces_ub:
        # Skip this segment - either too few or too many horizontal faces
        print(f"    SKIPPED: horizontal percentage {horizontal_pct*100:.1f}% outside bounds")
        return []
    
    # Now apply height filter to get faces in the top portion
    # Key insight: for HOLLOW objects (like drawers), the "lid" at the top is a shell
    # with faces pointing BOTH up AND down (inner and outer surfaces).
    # So we find faces at the geometric top that are HORIZONTAL (aligned with up vector,
    # either positive or negative direction) and split them all together.
    
    # Find faces at the geometric top
    top_height_cutoff = height_p95 - (height_range * top_height_ratio)
    faces_at_top = [f for f in segment_face_indices if np.dot(face_centroids[f], up_vector) >= top_height_cutoff]
    
    if len(faces_at_top) > 0:
        # Find HORIZONTAL faces at the top (pointing either UP or DOWN along the up vector)
        # These are faces whose normal is aligned with the up axis (dot product close to +1 or -1)
        horizontal_faces_at_top = []
        for face_idx in faces_at_top:
            dot = np.dot(face_normals[face_idx], up_vector)
            # Face is horizontal if |dot| >= cos_threshold (pointing up OR down)
            if abs(dot) >= cos_threshold:
                horizontal_faces_at_top.append(face_idx)
        
        top_dots = np.array([np.dot(face_normals[f], up_vector) for f in faces_at_top])
        upward_at_top = np.sum(top_dots >= cos_threshold)
        downward_at_top = np.sum(top_dots <= -cos_threshold)
        sideways_at_top = len(faces_at_top) - upward_at_top - downward_at_top
        
        print(f"    --- At geometric top (height >= {top_height_cutoff:.4f}): {len(faces_at_top)} faces ---")
        print(f"    Pointing UP: {upward_at_top}, DOWN: {downward_at_top}, SIDEWAYS: {sideways_at_top}")
        print(f"    HORIZONTAL (up OR down): {len(horizontal_faces_at_top)}")
        
        if len(horizontal_faces_at_top) >= min_faces:
            print(f"    Splitting {len(horizontal_faces_at_top)} HORIZONTAL faces at top (hollow shell)")
            if require_connectivity:
                regions = _find_connected_regions(horizontal_faces_at_top, face_neighbors, min_faces)
                return regions
            else:
                return [np.array(horizontal_faces_at_top)]
    
    # Fallback to original upward-only logic if no horizontal faces at top
    upward_face_heights = []
    for face_idx in all_upward_faces:
        face_height = np.dot(face_centroids[face_idx], up_vector)
        upward_face_heights.append((face_idx, face_height))
    
    # Sort by height and take the top portion
    upward_face_heights.sort(key=lambda x: x[1], reverse=True)
    
    # For hollow objects, use the height of upward-facing faces themselves
    # Take faces in the top X% of UPWARD-FACING faces (not all segment faces)
    all_heights = [h for _, h in upward_face_heights]
    if len(all_heights) > 0:
        upward_height_max = max(all_heights)
        upward_height_min = min(all_heights)
        upward_height_range = upward_height_max - upward_height_min
        
        if upward_height_range > 1e-6 and top_height_ratio < 1.0:
            # Threshold based on upward-facing faces' heights
            upward_height_threshold = upward_height_max - (upward_height_range * top_height_ratio)
        else:
            upward_height_threshold = upward_height_min
        
        print(f"    Upward faces height range: {upward_height_min:.4f} to {upward_height_max:.4f}")
        print(f"    Upward faces height threshold: {upward_height_threshold:.4f} (top {top_height_ratio*100:.0f}%)")
    else:
        upward_height_threshold = height_threshold  # Fallback
    
    upward_faces = []
    upward_heights_passed = []
    upward_heights_failed = []
    
    for face_idx, face_height in upward_face_heights:
        if face_height >= upward_height_threshold:
            upward_faces.append(face_idx)
            upward_heights_passed.append(face_height)
        else:
            upward_heights_failed.append(face_height)
    
    upward_faces = np.array(upward_faces)

    print(f"    All upward faces: {len(all_upward_faces)}")
    print(f"    After height filter: {len(upward_faces)} passed, {len(upward_heights_failed)} filtered out")
    if len(upward_heights_passed) > 0:
        print(f"    Passed heights: min={min(upward_heights_passed):.4f}, max={max(upward_heights_passed):.4f}")
    if len(upward_heights_failed) > 0:
        print(f"    Filtered heights: min={min(upward_heights_failed):.4f}, max={max(upward_heights_failed):.4f}")
    
    if len(upward_faces) < min_faces:
        print(f"    SKIPPED: only {len(upward_faces)} faces < min_faces={min_faces}")
        return []
    
    # If not requiring connectivity, return all upward faces as a single group
    if not require_connectivity:
        if len(upward_faces) >= min_faces:
            return [upward_faces]
        else:
            return []
    
    upward_set = set(upward_faces)
    
    # Find connected components using trimesh adjacency
    visited = set()
    components = []
    
    for start_face in upward_faces:
        if start_face in visited:
            continue
        
        # BFS to find connected component
        component = []
        queue = [start_face]
        
        while queue:
            face_idx = queue.pop(0)
            if face_idx in visited:
                continue
            visited.add(face_idx)
            component.append(face_idx)
            
            # Only follow neighbors that are also upward-facing
            for neighbor in face_neighbors[face_idx]:
                if neighbor in upward_set and neighbor not in visited:
                    queue.append(neighbor)
        
        if len(component) >= min_faces:
            components.append(np.array(component, dtype=int))
    
    return components


def split_all_top_planes(
    mesh: trimesh.Trimesh,
    face2label: np.ndarray,
    label2face_mask: np.ndarray,
    up_vector: np.ndarray = None,
    angle_threshold_deg: float = 30.0,
    min_faces: int = 10,
    upward_faces_lb: float = 0.05,
    upward_faces_ub: float = 0.80,
    require_connectivity: bool = True,
    top_height_ratio: float = 1.0,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Split top-facing planes from all segments.
    
    For each segment, finds upward-facing planar regions and splits them
    into new segments.
    
    Args:
        mesh: The mesh
        face2label: Per-face segment labels (will be copied, not modified in-place)
        label2face_mask: Binary mask [num_segments, num_faces]
        up_vector: Direction considered "up" (default: +Z)
        angle_threshold_deg: Max angle from up to be considered "top-facing"
        min_faces: Minimum faces for a region to be split
        upward_faces_lb: Lower bound for upward faces percentage (0-1). Skip if below.
        upward_faces_ub: Upper bound for upward faces percentage (0-1). Skip if above.
        require_connectivity: If True, only split connected components. If False,
                              split ALL upward faces as a single group.
        top_height_ratio: Only split faces in the top X% of each segment's height (0-1).
                          E.g., 0.2 = top 20%. Default 1.0 = all heights.
        verbose: Print progress
        
    Returns:
        Tuple of (new_face2label, new_label2face_mask, list of new segment IDs)
    """
    face2label = np.array(face2label)  # Copy
    label2face_mask = np.array(label2face_mask)  # Copy
    
    all_segments = sorted(set(face2label[face2label >= 0]))
    new_segment_ids = []
    
    if verbose:
        print(f"Splitting top planes from {len(all_segments)} segments...")
    
    for seg_id in all_segments:
        top_regions = find_top_plane_faces(
            mesh, face2label, seg_id,
            up_vector=up_vector,
            angle_threshold_deg=angle_threshold_deg,
            min_faces=min_faces,
            upward_faces_lb=upward_faces_lb,
            upward_faces_ub=upward_faces_ub,
            require_connectivity=require_connectivity,
            top_height_ratio=top_height_ratio
        )
        
        for region in top_regions:
            face2label, label2face_mask, new_id = split_segment_by_faces(
                face2label, label2face_mask, seg_id, region
            )
            if new_id >= 0:
                new_segment_ids.append(new_id)
                if verbose:
                    print(f"  Split {len(region)} faces from segment {seg_id} -> new segment {new_id}")
    
    if verbose:
        print(f"Created {len(new_segment_ids)} new top-plane segments")
    
    return face2label, label2face_mask, new_segment_ids


def split_by_connected_components(
    mesh: trimesh.Trimesh,
    face2label: np.ndarray,
    label2face_mask: np.ndarray,
    segment_id: int,
    min_component_size: int = 1,
    spatial_threshold: float = 0.0
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Split a segment into its topologically disconnected components.
    
    Args:
        mesh: The mesh
        face2label: Per-face segment labels
        label2face_mask: Binary mask [num_segments, num_faces]
        segment_id: The segment to split
        min_component_size: Minimum number of faces for a component to be split off.
                           Components smaller than this are kept with the main segment.
        spatial_threshold: If > 0, also connect faces whose centroids are within this
                          distance (as a fraction of mesh bounding box diagonal).
                          This prevents over-splitting due to small mesh gaps.
                          Recommended: 0.01-0.05 (1-5% of mesh size)
        
    Returns:
        Tuple of (updated face2label, updated label2face_mask, list of new segment IDs)
    """
    # Merge vertices for proper adjacency
    mesh_copy = mesh.copy()
    mesh_copy.merge_vertices(merge_tex=True, merge_norm=True)  # position-only: keep adjacency across texture/normal seams
    
    vertices = np.array(mesh_copy.vertices)
    faces = np.array(mesh_copy.faces)
    
    # Get faces in this segment
    segment_face_indices = np.where(face2label == segment_id)[0]
    
    if len(segment_face_indices) <= 1:
        print(f"[split_by_connected_components] Segment {segment_id} has <= 1 faces, nothing to split")
        return face2label, label2face_mask, []
    
    # Build face adjacency within segment
    from collections import defaultdict
    edge_to_faces = defaultdict(list)
    
    for face_idx in segment_face_indices:
        face = faces[face_idx]
        edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[2], face[0]]))
        ]
        for edge in edges:
            edge_to_faces[edge].append(face_idx)
    
    face_neighbors = defaultdict(set)
    for edge, face_list in edge_to_faces.items():
        for i, f1 in enumerate(face_list):
            for f2 in face_list[i+1:]:
                face_neighbors[f1].add(f2)
                face_neighbors[f2].add(f1)
    
    # Add spatial proximity connections if enabled
    if spatial_threshold > 0:
        # Compute mesh bounding box diagonal
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)
        distance_threshold = spatial_threshold * bbox_diagonal
        
        # Compute face centroids for this segment
        face_centroids = {}
        for face_idx in segment_face_indices:
            face = faces[face_idx]
            centroid = vertices[face].mean(axis=0)
            face_centroids[face_idx] = centroid
        
        # Build KD-tree for fast spatial queries
        from scipy.spatial import cKDTree
        centroid_array = np.array([face_centroids[fi] for fi in segment_face_indices])
        tree = cKDTree(centroid_array)
        
        # Find pairs within distance threshold
        pairs = tree.query_pairs(r=distance_threshold)
        spatial_connections = 0
        for i, j in pairs:
            f1 = segment_face_indices[i]
            f2 = segment_face_indices[j]
            if f2 not in face_neighbors[f1]:  # Only count new connections
                spatial_connections += 1
            face_neighbors[f1].add(f2)
            face_neighbors[f2].add(f1)
        
        if spatial_connections > 0:
            print(f"  Added {spatial_connections} spatial connections (threshold={spatial_threshold:.2%} of bbox)")
    
    # Find connected components using BFS
    visited = set()
    components = []
    
    for start_face in segment_face_indices:
        if start_face in visited:
            continue
        
        component = []
        queue = [start_face]
        
        while queue:
            face_idx = queue.pop(0)
            if face_idx in visited:
                continue
            visited.add(face_idx)
            component.append(face_idx)
            
            for neighbor in face_neighbors[face_idx]:
                if neighbor not in visited:
                    queue.append(neighbor)
        
        components.append(np.array(component, dtype=int))
    
    print(f"[split_by_connected_components] Segment {segment_id} has {len(components)} connected components")
    
    if len(components) <= 1:
        print(f"[split_by_connected_components] Only 1 component, nothing to split")
        return face2label, label2face_mask, []
    
    # Keep the largest component as the original segment, split others
    components.sort(key=len, reverse=True)
    new_segment_ids = []
    skipped_small = 0
    
    for i, component in enumerate(components[1:], start=1):  # Skip the largest
        if len(component) < min_component_size:
            skipped_small += 1
            continue
            
        face2label, label2face_mask, new_id = split_segment_by_faces(
            face2label, label2face_mask, segment_id, component
        )
        if new_id >= 0:
            new_segment_ids.append(new_id)
            print(f"  Component {i}: {len(component)} faces -> new segment {new_id}")
    
    if skipped_small > 0:
        print(f"  Skipped {skipped_small} components with < {min_component_size} faces")
    
    return face2label, label2face_mask, new_segment_ids


def _build_face_neighbors_for_faces(mesh: trimesh.Trimesh, face_indices: np.ndarray) -> dict:
    """Edge-sharing adjacency restricted to the given faces.

    Vertices are merged first (GLB/GLTF meshes often have per-face vertices,
    which would otherwise report every face as an island).
    """
    from collections import defaultdict

    mesh_copy = mesh.copy()
    mesh_copy.merge_vertices(merge_tex=True, merge_norm=True)  # position-only: keep adjacency across texture/normal seams
    faces = np.array(mesh_copy.faces)

    edge_to_faces = defaultdict(list)
    for face_idx in face_indices:
        face = faces[face_idx]
        for edge in (
            tuple(sorted((face[0], face[1]))),
            tuple(sorted((face[1], face[2]))),
            tuple(sorted((face[2], face[0]))),
        ):
            edge_to_faces[edge].append(face_idx)

    face_neighbors = defaultdict(set)
    for face_list in edge_to_faces.values():
        for i, f1 in enumerate(face_list):
            for f2 in face_list[i + 1:]:
                face_neighbors[f1].add(f2)
                face_neighbors[f2].add(f1)
    return face_neighbors


def segment_islands(
    mesh: trimesh.Trimesh,
    face2label: np.ndarray,
    segment_id: int,
) -> list[np.ndarray]:
    """Topology-only connected components ("islands") of one segment.

    Purely edge-adjacency based — no spatial linking — so it stays fast even
    on large segments, unlike split_by_connected_components with a spatial
    threshold. Returned largest-first.
    """
    from collections import deque

    segment_face_indices = np.where(face2label == segment_id)[0]
    if len(segment_face_indices) == 0:
        return []

    face_neighbors = _build_face_neighbors_for_faces(mesh, segment_face_indices)
    face_set = set(int(f) for f in segment_face_indices)
    visited = set()
    islands = []
    for start_face in segment_face_indices:
        start_face = int(start_face)
        if start_face in visited:
            continue
        island = []
        queue = deque([start_face])
        while queue:
            face_idx = queue.popleft()
            if face_idx in visited:
                continue
            visited.add(face_idx)
            island.append(face_idx)
            for neighbor in face_neighbors[face_idx]:
                if neighbor in face_set and neighbor not in visited:
                    queue.append(neighbor)
        islands.append(np.array(sorted(island), dtype=int))

    islands.sort(key=len, reverse=True)
    return islands


def island_containing_face(
    mesh: trimesh.Trimesh,
    face2label: np.ndarray,
    segment_id: int,
    seed_face: int,
) -> np.ndarray:
    """The connected island of `segment_id` that contains `seed_face`.

    One-click flood fill for grabbing a stray piece of a segment (e.g. debris
    floating away from the main body).
    """
    for island in segment_islands(mesh, face2label, segment_id):
        if int(seed_face) in island:
            return island
    return np.array([], dtype=int)


def stray_islands(
    mesh: trimesh.Trimesh,
    face2label: np.ndarray,
    segment_id: int,
    max_faces: int | None = None,
) -> np.ndarray:
    """All faces of a segment outside its largest island.

    The one-shot "grab all the debris" selection: everything not part of the
    segment's main body. With max_faces set, only islands at or below that
    size are included (larger secondary islands are treated as real geometry).
    """
    islands = segment_islands(mesh, face2label, segment_id)
    if len(islands) <= 1:
        return np.array([], dtype=int)
    selected = []
    for island in islands[1:]:
        if max_faces is not None and len(island) > max_faces:
            continue
        selected.append(island)
    if not selected:
        return np.array([], dtype=int)
    return np.concatenate(selected)


def find_faces_by_normal(
    mesh: trimesh.Trimesh,
    face2label: np.ndarray,
    segment_id: int,
    reference_face_idx: int,
    angle_threshold_deg: float = 15.0,
    cross_segment: bool = False
) -> np.ndarray:
    """
    Find all faces with similar normal to a reference face (not requiring connectivity).
    
    Unlike find_coplanar_faces, this doesn't require faces to be connected -
    it finds ALL faces in the segment with similar orientation.
    
    Args:
        mesh: The mesh
        face2label: Per-face segment labels
        segment_id: The segment to search within
        reference_face_idx: Index of the reference face
        angle_threshold_deg: Max angle difference from reference face normal (degrees)
        cross_segment: If True, search ALL segments
        
    Returns:
        Array of face indices with similar normals
    """
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    # Compute face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_norms[face_norms < 1e-10] = 1
    face_normals = face_normals / face_norms
    
    # Get reference normal
    reference_normal = face_normals[reference_face_idx]
    
    # Convert threshold to cosine
    cos_threshold = np.cos(np.radians(angle_threshold_deg))
    
    # Get faces to consider
    if cross_segment:
        candidate_faces = np.arange(len(faces))
    else:
        candidate_faces = np.where(face2label == segment_id)[0]
    
    # Find faces with similar normal
    matching_faces = []
    for face_idx in candidate_faces:
        normal = face_normals[face_idx]
        cos_angle = abs(np.dot(normal, reference_normal))
        if cos_angle >= cos_threshold:
            matching_faces.append(face_idx)
    
    print(f"[find_faces_by_normal] Found {len(matching_faces)} faces with similar normal (angle<={angle_threshold_deg}°)")
    return np.array(matching_faces, dtype=int)


def merge_segments(
    face2label: np.ndarray,
    label2face_mask: np.ndarray,
    segment_ids: list
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Merge multiple segments into a single segment.
    
    Args:
        face2label: Per-face segment labels
        label2face_mask: Binary mask [num_segments, num_faces]
        segment_ids: List of segment IDs to merge
        
    Returns:
        Tuple of (updated face2label, updated label2face_mask, merged segment ID)
    """
    if len(segment_ids) < 2:
        print(f"[merge_segments] Need at least 2 segments to merge")
        return face2label, label2face_mask, segment_ids[0] if segment_ids else -1
    
    # Use the first segment ID as the target
    target_id = segment_ids[0]
    source_ids = segment_ids[1:]
    
    total_faces_merged = 0
    for source_id in source_ids:
        # Find faces in source segment
        source_faces = np.where(face2label == source_id)[0]
        total_faces_merged += len(source_faces)
        
        # Update face2label
        face2label[source_faces] = target_id
        
        # Update label2face_mask
        label2face_mask[target_id, source_faces] = True
        label2face_mask[source_id, source_faces] = False
    
    print(f"[merge_segments] Merged {len(source_ids)} segments into segment {target_id} ({total_faces_merged} faces)")
    return face2label, label2face_mask, target_id
