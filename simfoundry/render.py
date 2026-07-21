import os
import numpy as np
import colorsys
from PIL import Image, ImageDraw, ImageFont
import trimesh
import torch
from tqdm import tqdm
from camera_utils import view_matrix, sample_view_matrices, sample_view_matrices_polyhedra
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import shutil
from omegaconf import DictConfig


def add_part_labels(img, scene, part_info, camera_transform, resolution, render_buffers=None, verbose=False):
    """
    Add numerical labels to each part in the rendered image.
    Labels are placed along the image edges (top, bottom, left, right) based on which edge
    provides the straightest line to the part.
    
    Args:
        img: PIL Image
        scene: trimesh.Scene
        part_info: dict with part information (must have 'center', 'color' keys)
        camera_transform: 4x4 camera transformation matrix (camera to world)
        resolution: (width, height) tuple
        render_buffers: dict with 'depth' (optional, for future use)
    
    Returns:
        img: PIL Image with labels added
    """
    from scipy import ndimage
    
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Try to load a larger, bolder font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception as e:
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except Exception as e:
            font = ImageFont.load_default()
    
    # Get image as numpy array for color matching
    img_array = np.array(img)
    color_tolerance = 20  # Max distance to be considered a match at all
    
    # Get image dimensions for coverage calculation
    total_pixels = resolution[0] * resolution[1]
    MIN_COVERAGE_PERCENT = 0.1  # Minimum percentage of image a part must cover to be labeled
    min_pixels = int(total_pixels * MIN_COVERAGE_PERCENT / 100)
    
    # Padding and margins
    padding = 8
    label_spacing = 30  # Spacing between labels along an edge
    edge_margin = 20    # Margin from edge of image
    
    img_center_x = resolution[0] // 2
    img_center_y = resolution[1] // 2
    
    # Pre-compute color distances for all parts to enable "closest color wins" matching
    # This prevents two similar colors from matching the same pixels
    all_part_colors = []
    all_part_ids = []
    for part_id, info in part_info.items():
        all_part_colors.append(info['color'][:3])
        all_part_ids.append(part_id)
    
    if len(all_part_colors) > 0:
        all_part_colors = np.array(all_part_colors)  # Shape: (n_parts, 3)
        
        # Compute distance from each pixel to each part's color
        # img_array shape: (H, W, 3+), all_part_colors shape: (n_parts, 3)
        img_rgb = img_array[:, :, :3].astype(float)  # (H, W, 3)
        # Reshape for broadcasting: (H, W, 1, 3) - (1, 1, n_parts, 3) -> (H, W, n_parts, 3)
        color_diffs = np.abs(img_rgb[:, :, np.newaxis, :] - all_part_colors[np.newaxis, np.newaxis, :, :])
        color_distances = np.mean(color_diffs, axis=3)  # (H, W, n_parts)
        
        # For each pixel, find the closest part
        min_distances = np.min(color_distances, axis=2)  # (H, W)
        closest_part_idx = np.argmin(color_distances, axis=2)  # (H, W) - index into all_part_ids
    else:
        min_distances = np.full((resolution[1], resolution[0]), np.inf)
        closest_part_idx = np.zeros((resolution[1], resolution[0]), dtype=int)
    
    def find_part_pixels_by_color(part_color, part_id):
        """
        Find pixels belonging to this part using 'closest color wins' approach.
        A pixel belongs to this part only if:
        1. This part's color is the closest match among all parts
        2. The distance is within the tolerance
        """
        if part_id not in all_part_ids:
            return np.zeros((resolution[1], resolution[0]), dtype=bool)
        
        part_idx = all_part_ids.index(part_id)
        
        # Pixel belongs to this part if it's the closest match AND within tolerance
        is_closest = closest_part_idx == part_idx
        is_within_tolerance = min_distances <= color_tolerance
        
        return is_closest & is_within_tolerance
    
    def get_label_size(text):
        """Get the size of a label with padding."""
        temp_bbox = draw.textbbox((0, 0), text, font=font)
        width = temp_bbox[2] - temp_bbox[0] + 2 * padding
        height = temp_bbox[3] - temp_bbox[1] + 2 * padding
        return width, height
    
    def find_anchor_on_boundary(matching_mask, label_x, label_y):
        """Find the best anchor point on the part boundary for a connecting line."""
        # Get boundary pixels
        eroded = ndimage.binary_erosion(matching_mask, iterations=3)
        boundary = matching_mask & ~eroded
        boundary_coords = np.argwhere(boundary)
        
        if len(boundary_coords) == 0:
            boundary_coords = np.argwhere(matching_mask)
        if len(boundary_coords) == 0:
            return None
        
        # Find boundary point closest to label
        distances = np.sqrt((boundary_coords[:, 0] - label_y)**2 + 
                           (boundary_coords[:, 1] - label_x)**2)
        closest_idx = np.argmin(distances)
        anchor_y, anchor_x = boundary_coords[closest_idx]
        
        return int(anchor_x), int(anchor_y)
    
    def determine_best_edge(center_x, center_y):
        """
        Determine which edge (top, bottom, left, right) provides the straightest line
        from a label to the part centroid.
        """
        # Calculate relative position from image center
        dx = center_x - img_center_x
        dy = center_y - img_center_y
        
        # Normalize by image dimensions to account for aspect ratio
        norm_dx = dx / (resolution[0] / 2)
        norm_dy = dy / (resolution[1] / 2)
        
        # Determine dominant direction
        if abs(norm_dx) > abs(norm_dy):
            # Horizontal dominance
            return 'right' if norm_dx > 0 else 'left'
        else:
            # Vertical dominance
            return 'bottom' if norm_dy > 0 else 'top'
    
    # Collect visible parts
    labels_to_draw = []
    
    for part_id, info in part_info.items():
        part_color = info['color']
        
        # Find pixels for this part using "closest color wins" matching
        matching_mask = find_part_pixels_by_color(part_color, part_id)
        
        if not np.any(matching_mask):
            if verbose:
                print(f"    Part {part_id}: not visible")
            continue
        
        matching_coords = np.argwhere(matching_mask)
        pixel_count = len(matching_coords)
        
        # Skip parts that cover less than the minimum percentage of the image
        if pixel_count < min_pixels:
            coverage_percent = (pixel_count / total_pixels) * 100
            if verbose:
                print(f"    Skipping part {part_id}: only {coverage_percent:.1f}% coverage (< {MIN_COVERAGE_PERCENT}%)")
            continue
        
        if verbose:
            coverage_percent = (pixel_count / total_pixels) * 100
            print(f"    Part {part_id}: {coverage_percent:.1f}% coverage")
        
        # Calculate part centroid
        center_y = int(matching_coords[:, 0].mean())
        center_x = int(matching_coords[:, 1].mean())
        
        # Determine best edge for this part
        best_edge = determine_best_edge(center_x, center_y)
        
        labels_to_draw.append({
            'part_id': part_id,
            'color': part_color,
            'matching_mask': matching_mask,
            'pixel_count': pixel_count,
            'center_x': center_x,
            'center_y': center_y,
            'edge': best_edge,
            'vertex_count': info.get('vertex_count', 0)
        })
    
    if verbose:
        print(f"  Found {len(labels_to_draw)} visible parts before filtering")
        for label_info in labels_to_draw:
            print(f"    Part {label_info['part_id']}: color=RGB{tuple(int(c) for c in label_info['color'][:3])}, "
                  f"pixels={label_info['pixel_count']}, center=({label_info['center_x']}, {label_info['center_y']})")
    
    if len(labels_to_draw) == 0:
        return img, []
    
    # Filter out labels with similar colors pointing to nearby/overlapping regions
    # This handles over-segmentation where one visual part is split into multiple segments
    BBOX_MARGIN = 50  # Margin to expand bounding boxes for overlap check
    COLOR_SIMILARITY_THRESHOLD = 30  # Max color difference to be considered "similar" (strict to avoid merging distinct colors)
    
    # Sort by pixel count descending - larger parts get priority
    labels_to_draw.sort(key=lambda l: -l['pixel_count'])
    
    # Pre-compute bounding boxes for filtering
    for label_info in labels_to_draw:
        matching_coords = np.argwhere(label_info['matching_mask'])
        if len(matching_coords) > 0:
            # Bounding box: (y_min, y_max, x_min, x_max)
            label_info['bbox'] = (
                int(matching_coords[:, 0].min()),
                int(matching_coords[:, 0].max()),
                int(matching_coords[:, 1].min()),
                int(matching_coords[:, 1].max())
            )
            label_info['anchor_y'] = int(matching_coords[:, 0].mean())
            label_info['anchor_x'] = int(matching_coords[:, 1].mean())
        else:
            label_info['bbox'] = (0, 0, 0, 0)
            label_info['anchor_x'] = label_info['center_x']
            label_info['anchor_y'] = label_info['center_y']
    
    def colors_are_similar(color1, color2):
        """Check if two colors are similar enough to be considered the same visual part."""
        diff = np.abs(np.array(color1[:3]) - np.array(color2[:3]))
        return np.mean(diff) < COLOR_SIMILARITY_THRESHOLD
    
    def bboxes_overlap_or_adjacent(label1, label2):
        """Check if two bounding boxes overlap or are adjacent (within margin)."""
        y1_min, y1_max, x1_min, x1_max = label1['bbox']
        y2_min, y2_max, x2_min, x2_max = label2['bbox']
        
        # Expand bboxes by margin
        y1_min -= BBOX_MARGIN
        y1_max += BBOX_MARGIN
        x1_min -= BBOX_MARGIN
        x1_max += BBOX_MARGIN
        
        # Check overlap
        x_overlap = not (x1_max < x2_min or x2_max < x1_min)
        y_overlap = not (y1_max < y2_min or y2_max < y1_min)
        
        return x_overlap and y_overlap
    
    # Filter: keep only labels that don't have a larger, similar-colored part nearby
    filtered_labels = []
    for label_info in labels_to_draw:
        should_keep = True
        for kept_label in filtered_labels:
            # If this label has similar color AND overlapping/adjacent bbox, skip it
            if colors_are_similar(label_info['color'], kept_label['color']) and \
               bboxes_overlap_or_adjacent(label_info, kept_label):
                if verbose:
                    print(f"    Filtering out part {label_info['part_id']}: similar to part {kept_label['part_id']}")
                should_keep = False
                break
        if should_keep:
            filtered_labels.append(label_info)
    
    labels_to_draw = filtered_labels
    
    if verbose:
        print(f"  Drawing {len(labels_to_draw)} labels after filtering:")
        for label_info in labels_to_draw:
            print(f"    Part {label_info['part_id']}: color=RGB{tuple(int(c) for c in label_info['color'][:3])}")
    
    # Calculate label dimensions
    max_label_width = 0
    max_label_height = 0
    for label_info in labels_to_draw:
        label_text = str(label_info['part_id'])
        w, h = get_label_size(label_text)
        max_label_width = max(max_label_width, w)
        max_label_height = max(max_label_height, h)
    
    # Group labels by edge
    edge_labels = {'top': [], 'bottom': [], 'left': [], 'right': []}
    for label_info in labels_to_draw:
        edge_labels[label_info['edge']].append(label_info)
    
    # Sort labels on each edge by their position along that edge
    # Top/bottom: sort by center_x; Left/right: sort by center_y
    edge_labels['top'].sort(key=lambda l: l['center_x'])
    edge_labels['bottom'].sort(key=lambda l: l['center_x'])
    edge_labels['left'].sort(key=lambda l: l['center_y'])
    edge_labels['right'].sort(key=lambda l: l['center_y'])
    
    def calculate_edge_positions(labels, edge):
        """Calculate positions for labels along an edge."""
        if not labels:
            return []
        
        positions = []
        n = len(labels)
        
        if edge in ('top', 'bottom'):
            # Horizontal edge - distribute labels horizontally
            total_width = n * max_label_width + (n - 1) * label_spacing
            start_x = (resolution[0] - total_width) // 2
            start_x = max(edge_margin + max_label_width // 2, start_x)
            
            # Fixed y position based on edge
            if edge == 'top':
                label_y = edge_margin + max_label_height // 2
            else:  # bottom
                label_y = resolution[1] - edge_margin - max_label_height // 2
            
            for i, label_info in enumerate(labels):
                label_x = start_x + i * (max_label_width + label_spacing) + max_label_width // 2
                # Clamp to image bounds
                label_x = min(label_x, resolution[0] - edge_margin - max_label_width // 2)
                positions.append((label_x, label_y, label_info))
        else:
            # Vertical edge - distribute labels vertically
            total_height = n * max_label_height + (n - 1) * label_spacing
            start_y = (resolution[1] - total_height) // 2
            start_y = max(edge_margin + max_label_height // 2, start_y)
            
            # Fixed x position based on edge
            if edge == 'left':
                label_x = edge_margin + max_label_width // 2
            else:  # right
                label_x = resolution[0] - edge_margin - max_label_width // 2
            
            for i, label_info in enumerate(labels):
                label_y = start_y + i * (max_label_height + label_spacing) + max_label_height // 2
                # Clamp to image bounds
                label_y = min(label_y, resolution[1] - edge_margin - max_label_height // 2)
                positions.append((label_x, label_y, label_info))
        
        return positions
    
    # Calculate all label positions
    all_positions = []
    for edge in ('top', 'bottom', 'left', 'right'):
        all_positions.extend(calculate_edge_positions(edge_labels[edge], edge))
    
    # Draw all labels
    for label_x, label_y, label_info in all_positions:
        part_id = label_info['part_id']
        matching_mask = label_info['matching_mask']
        color = label_info['color']
        edge = label_info['edge']
        
        label_text = str(part_id)
        
        # Convert color to integers
        label_color = tuple(int(c) for c in color[:3])
        
        # Calculate contrasting text color (black or white)
        brightness = (label_color[0] * 0.299 + label_color[1] * 0.587 + label_color[2] * 0.114)
        outline_color = (0, 0, 0, 255) if brightness > 128 else (255, 255, 255, 255)
        text_color = (0, 0, 0, 255) if brightness > 128 else (255, 255, 255, 255)
        
        # Calculate label bounding box
        half_w = max_label_width // 2 + padding
        half_h = max_label_height // 2 + padding
        bg_bbox = [
            label_x - half_w,
            label_y - half_h,
            label_x + half_w,
            label_y + half_h
        ]
        
        # Find anchor point on the part boundary
        anchor = find_anchor_on_boundary(matching_mask, label_x, label_y)
        if anchor:
            anchor_x, anchor_y = anchor
            # Store the actual boundary anchor in label_info for reuse
            label_info['boundary_anchor_x'] = anchor_x
            label_info['boundary_anchor_y'] = anchor_y
            
            # Determine the edge of the label box to connect from
            if edge == 'top':
                label_edge_x = label_x
                label_edge_y = bg_bbox[3]  # Bottom edge
            elif edge == 'bottom':
                label_edge_x = label_x
                label_edge_y = bg_bbox[1]  # Top edge
            elif edge == 'left':
                label_edge_x = bg_bbox[2]  # Right edge
                label_edge_y = label_y
            else:  # right
                label_edge_x = bg_bbox[0]  # Left edge
                label_edge_y = label_y
            
            # Draw connecting line from part to label
            line_color = label_color + (200,)
            draw.line([(int(anchor_x), int(anchor_y)), (int(label_edge_x), int(label_edge_y))], 
                     fill=line_color, width=8)
            
            # Draw small circle at anchor point on the part
            circle_radius = 6
            draw.ellipse(
                [int(anchor_x - circle_radius), int(anchor_y - circle_radius),
                 int(anchor_x + circle_radius), int(anchor_y + circle_radius)],
                fill=label_color + (255,), outline=outline_color, width=2
            )
        
        # Draw label background
        bg_color = label_color + (230,)
        draw.rectangle([int(x) for x in bg_bbox], fill=bg_color, outline=outline_color, width=2)
        
        # Draw label text centered in the box
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = label_x - text_width // 2
        text_y = label_y - text_height // 2
        draw.text((int(text_x), int(text_y)), label_text, fill=text_color, font=font)
    
    # Build label positions data for reuse on other images, will be used for rendering the original color images with the same labels
    label_positions = []
    for label_x, label_y, label_info in all_positions:
        # Use boundary anchor (computed by find_anchor_on_boundary)
        if 'boundary_anchor_x' not in label_info or 'boundary_anchor_y' not in label_info:
            raise ValueError(f"Boundary anchor not found for part {label_info['part_id']}. "
                           f"This should never happen - find_anchor_on_boundary failed.")
        label_positions.append({
            'label_x': label_x,
            'label_y': label_y,
            'part_id': label_info['part_id'],
            'anchor_x': label_info['boundary_anchor_x'],
            'anchor_y': label_info['boundary_anchor_y'],
            'edge': label_info['edge'],
            'color': label_info['color'],
            'max_label_width': max_label_width,
            'max_label_height': max_label_height,
        })
    
    return img, label_positions


def draw_labels_at_positions(img, label_positions, padding=8):
    """
    Draw labels on an image at pre-computed positions.
    Used to apply the same labels from segmented renders to original color renders.
    
    Args:
        img: PIL Image to draw on
        label_positions: List of dicts with label position info from add_part_labels
        padding: Padding around label text
    
    Returns:
        img: PIL Image with labels added
    """
    if not label_positions:
        return img
    
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Try to load a larger, bolder font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except Exception:
            font = ImageFont.load_default()
    
    for pos_info in label_positions:
        label_x = pos_info['label_x']
        label_y = pos_info['label_y']
        part_id = pos_info['part_id']
        anchor_x = pos_info['anchor_x']
        anchor_y = pos_info['anchor_y']
        edge = pos_info['edge']
        color = pos_info['color']
        max_label_width = pos_info['max_label_width']
        max_label_height = pos_info['max_label_height']
        
        label_text = str(part_id)
        
        # Convert color to integers
        label_color = tuple(int(c) for c in color[:3])
        
        # Calculate contrasting text color (black or white)
        brightness = (label_color[0] * 0.299 + label_color[1] * 0.587 + label_color[2] * 0.114)
        outline_color = (0, 0, 0, 255) if brightness > 128 else (255, 255, 255, 255)
        text_color = (0, 0, 0, 255) if brightness > 128 else (255, 255, 255, 255)
        
        # Calculate label bounding box
        half_w = max_label_width // 2 + padding
        half_h = max_label_height // 2 + padding
        bg_bbox = [
            label_x - half_w,
            label_y - half_h,
            label_x + half_w,
            label_y + half_h
        ]
        
        # Determine the edge of the label box to connect from
        if edge == 'top':
            label_edge_x = label_x
            label_edge_y = bg_bbox[3]  # Bottom edge
        elif edge == 'bottom':
            label_edge_x = label_x
            label_edge_y = bg_bbox[1]  # Top edge
        elif edge == 'left':
            label_edge_x = bg_bbox[2]  # Right edge
            label_edge_y = label_y
        else:  # right
            label_edge_x = bg_bbox[0]  # Left edge
            label_edge_y = label_y
        
        # Draw connecting line from anchor to label
        line_color = label_color + (200,)
        draw.line([(int(anchor_x), int(anchor_y)), (int(label_edge_x), int(label_edge_y))], 
                 fill=line_color, width=8)
        
        # Draw small circle at anchor point
        circle_radius = 6
        draw.ellipse(
            [int(anchor_x - circle_radius), int(anchor_y - circle_radius),
             int(anchor_x + circle_radius), int(anchor_y + circle_radius)],
            fill=label_color + (255,), outline=outline_color, width=2
        )
        
        # Draw label background
        bg_color = label_color + (230,)
        draw.rectangle([int(x) for x in bg_bbox], fill=bg_color, outline=outline_color, width=2)
        
        # Draw label text centered in the box
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = label_x - text_width // 2
        text_y = label_y - text_height // 2
        draw.text((int(text_x), int(text_y)), label_text, fill=text_color, font=font)
    
    return img


def load_partitioned_mesh(mesh_path, face_ids_path=None, verbose=False):
    """
    Load a mesh and its face_ids (part labels).
    
    Args:
        mesh_path: Path to the mesh file (.glb, .ply, etc.)
        face_ids_path: Path to face_ids numpy array. If None, try to infer from mesh_path
    
    Returns:
        mesh: trimesh.Trimesh object
        face_ids: numpy array of face labels
    """
    print(f"Loading mesh from: {mesh_path} for rendering")
    mesh = trimesh.load(mesh_path, force='mesh')
    
    # If mesh is a Scene, extract the main geometry
    if isinstance(mesh, trimesh.Scene):
        if verbose:
            print("Loaded a Scene, extracting main geometry...")
        mesh = mesh.dump(concatenate=True)
    
    if face_ids_path is None:
        # Try to infer face_ids path
        base_path = Path(mesh_path).parent
        face_ids_candidates = [
            base_path / 'final_face_ids.npy',
            Path(mesh_path).with_suffix('.npy'),
            Path(mesh_path.replace('.glb', '_face_ids.npy')),
            base_path / 'face_ids.npy',
        ]
        
        for candidate in face_ids_candidates:
            if candidate.exists():
                face_ids_path = str(candidate)
                break
        
        if face_ids_path is None:
            raise FileNotFoundError(
                f"Could not find face_ids file. Please specify with --face_ids. "
                f"Searched: {[str(c) for c in face_ids_candidates]}"
            )
    
    if verbose:
        print(f"Loading face_ids from: {face_ids_path}")
    face_ids = np.load(face_ids_path)
    
    if verbose:
        print(f"Mesh info: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"Face IDs: shape={face_ids.shape}, unique parts={len(np.unique(face_ids[face_ids >= 0]))}")
    
    return mesh, face_ids


def create_scene_from_parts(mesh, face_ids, verbose=False):
    """
    Create a trimesh.Scene with each part as a separate geometry.
    
    Args:
        mesh: trimesh.Trimesh object
        face_ids: numpy array of face labels
    
    Returns:
        scene: trimesh.Scene with separate geometries for each part
        part_info: dict mapping part_id -> (center, color, vertex_count)
        label_mapping: dict mapping old_label -> new_label (0, 1, 2, ...)
    """
    if verbose:
        print("Creating scene from parts...")
    scene = trimesh.Scene()
    part_info = {}
    
    unique_ids = np.unique(face_ids)
    valid_ids = unique_ids[unique_ids >= 0]  # Filter out -1, -2, etc.
    
    # Create label mapping: old_id -> new_id (0, 1, 2, ...)
    label_mapping = {int(old_id): new_id for new_id, old_id in enumerate(valid_ids)}
    if verbose:
        print(f"Created label mapping: {len(label_mapping)} parts (0 to {len(label_mapping)-1})")
    
    # Generate maximally distinct colors for each part
    # Use a fixed palette of colors that are far apart in RGB space
    # This avoids the golden ratio issue where similar hues can produce similar colors
    DISTINCT_COLORS = [
        (1.0, 0.0, 0.0),    # Red
        (0.0, 1.0, 0.0),    # Green
        (0.0, 0.0, 1.0),    # Blue
        (1.0, 1.0, 0.0),    # Yellow
        (1.0, 0.0, 1.0),    # Magenta
        (0.0, 1.0, 1.0),    # Cyan
        (1.0, 0.5, 0.0),    # Orange
        (0.5, 0.0, 1.0),    # Purple
        (0.0, 1.0, 0.5),    # Spring Green
        (1.0, 0.0, 0.5),    # Rose
        (0.5, 1.0, 0.0),    # Lime
        (0.0, 0.5, 1.0),    # Sky Blue
        (0.7, 0.3, 0.0),    # Brown
        (0.3, 0.7, 0.3),    # Sea Green
        (0.3, 0.3, 0.7),    # Slate Blue
        (0.8, 0.8, 0.0),    # Olive Yellow
        (0.8, 0.0, 0.8),    # Deep Magenta
        (0.0, 0.8, 0.8),    # Teal
        (1.0, 0.7, 0.4),    # Peach
        (0.4, 0.7, 1.0),    # Light Blue
    ]
    
    n_parts = len(valid_ids)
    colors = []
    for i in range(n_parts):
        if i < len(DISTINCT_COLORS):
            colors.append(DISTINCT_COLORS[i])
        else:
            # Fall back to golden ratio for additional colors beyond the palette
            hue = ((i - len(DISTINCT_COLORS)) * 0.618033988749895 + 0.1) % 1.0
            saturation = 0.7 + (i % 3) * 0.1  # Vary saturation
            value = 0.6 + (i % 4) * 0.1  # Vary value
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
    
    for idx, part_id in enumerate(valid_ids):
        # Extract faces for this part
        part_mask = face_ids == part_id
        part_faces_indices = np.where(part_mask)[0]
        
        if len(part_faces_indices) == 0:
            continue
        
        # Get the faces and create submesh
        part_faces = mesh.faces[part_faces_indices]
        
        # Find all vertices used by these faces
        vertices_used = np.unique(part_faces.flatten())
        
        # Create a mapping from old vertex indices to new ones
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(vertices_used)}
        
        # Extract vertices and remap faces
        part_vertices = mesh.vertices[vertices_used]
        part_faces_remapped = np.array([
            [vertex_map[v] for v in face] 
            for face in part_faces
        ])
        
        # Create submesh for this part
        part_mesh = trimesh.Trimesh(
            vertices=part_vertices,
            faces=part_faces_remapped,
            process=False
        )
        
        # Set color
        color_rgb = np.array(colors[idx % len(colors)]) * 255
        part_mesh.visual.face_colors = np.tile(
            np.append(color_rgb, 255), 
            (len(part_mesh.faces), 1)
        ).astype(np.uint8)
        
        # Calculate center and average normal direction
        center = part_vertices.mean(axis=0)
        
        # Calculate average normal (for visibility check)
        face_normals = np.cross(
            part_vertices[part_faces_remapped[:, 1]] - part_vertices[part_faces_remapped[:, 0]],
            part_vertices[part_faces_remapped[:, 2]] - part_vertices[part_faces_remapped[:, 0]]
        )
        face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-10)
        avg_normal = face_normals.mean(axis=0)
        avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-10)
        
        # Get new label ID (0, 1, 2, ...)
        new_label = label_mapping[int(part_id)]
        
        # Add to scene with new label name
        geom_name = f"part_{new_label}"
        scene.add_geometry(part_mesh, geom_name=geom_name)
        
        # Store info with new label
        part_info[new_label] = {
            'center': center,
            'color': color_rgb,
            'vertex_count': len(part_vertices),
            'geom_name': geom_name,
            'normal': avg_normal,
            'old_label': int(part_id)
        }
        
        if verbose:
            print(f"  Part {new_label} (was {part_id}): {len(part_vertices)} vertices, center={center}")
    
    if verbose:
        print(f"Created scene with {len(scene.geometry)} parts")
    return scene, part_info, label_mapping


def render_scene_with_labels(scene, part_info, camera_angles, output_dir, 
                              resolution=(1920, 1080),
                              renderer='pyrender', camera_mode='angles', num_views=None,
                              verbose=False, flat_shading=True, original_colors_source=None,
                              original_colors_explosion_factor=0.0, original_colors_face_ids=None):
    """
    Render the scene from multiple angles with part labels. Used for merging parts downstream.
    Also saves unlabeled versions to an 'unlabelled' subfolder.
    Optionally saves renders with original mesh colors to an 'original_colors' subfolder.
    
    Args:
        scene: trimesh.Scene to render
        part_info: dict with part information
        camera_angles: list of (azimuth, elevation) tuples in degrees (used if camera_mode='angles')
        output_dir: directory to save renders
        resolution: (width, height) tuple
        renderer: 'pyrender' or 'blender'
        camera_mode: 'angles' (manual azimuth/elevation), 'uniform' (random uniform), 
                     'standard', 'icosahedron', 'dodecahedron', etc.
        num_views: number of views (used for uniform/polyhedra modes)
        verbose: Print debug info
        flat_shading: If True (default), use flat/unlit rendering with no shadows.
                      This produces exact colors for reliable color-based label matching.
        original_colors_source: Optional. Can be:
                               - str/Path: Path to original GLB/mesh file (best for textured meshes)
                               - trimesh.Scene or trimesh.Trimesh: Mesh object with colors
                               If provided, renders are saved to 'original_colors' subfolder.
        original_colors_explosion_factor: Explosion factor for original colors (0 = no explosion)
        original_colors_face_ids: Face IDs for explosion (required if explosion_factor > 0)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create unlabelled subfolder
    unlabelled_dir = os.path.join(output_dir, 'unlabelled')
    os.makedirs(unlabelled_dir, exist_ok=True)
    
    # Create original_colors subfolder if original colors source is provided
    original_colors_dir = None
    original_scene = None
    if original_colors_source is not None:
        original_colors_dir = os.path.join(output_dir, 'original_colors')
        os.makedirs(original_colors_dir, exist_ok=True)
        

        if isinstance(original_colors_source, (str, Path)):
  
            if verbose:
                print(f"  Loading original mesh from path: {original_colors_source}")
            loaded = trimesh.load(str(original_colors_source))
            

            if isinstance(loaded, trimesh.Scene):
                # Extract geometry and apply scene graph transforms
                geom_names = list(loaded.geometry.keys())
                if len(geom_names) == 1:
                    geom_name = geom_names[0]
                    original_mesh = loaded.geometry[geom_name].copy()
                    
                    # Get transform from scene graph
                    # The graph stores (transform, parent_node) for each node
                    try:
                        transform, _ = loaded.graph[geom_name]
                        if transform is not None and not np.allclose(transform, np.eye(4)):
                            if verbose:
                                print(f"  Applying scene graph transform to geometry")
                            original_mesh.apply_transform(transform)
                    except Exception as e:
                        if verbose:
                            print(f"  Error applying scene graph transform: {e}")
                        raise e

                else:
                    # Multiple geometries - concatenate with transforms applied
                    transformed_geoms = []
                    for geom_name in geom_names:
                        geom = loaded.geometry[geom_name].copy()
                        try:
                            transform, _ = loaded.graph[geom_name]
                            if transform is not None and not np.allclose(transform, np.eye(4)):
                                geom.apply_transform(transform)
                        except (KeyError, TypeError, ValueError):
                            # ValueError can occur when trimesh hits iteration limit in complex graphs
                            pass
                        transformed_geoms.append(geom)
                    original_mesh = trimesh.util.concatenate(transformed_geoms)
            else:
                original_mesh = loaded
            
            if verbose:
                visual_type = type(original_mesh.visual).__name__
                has_uv = hasattr(original_mesh.visual, 'uv') and original_mesh.visual.uv is not None
                print(f"  Original mesh visual type: {visual_type}, has_uv: {has_uv}")
                print(f"  Bounds: {original_mesh.bounds}")
            
            
            # Apply explosion if requested
            if original_colors_explosion_factor > 0 and original_colors_face_ids is not None:
                if verbose:
                    print(f"  Exploding original mesh (factor={original_colors_explosion_factor})")
                original_mesh, _ = explode_mesh(original_mesh, original_colors_face_ids, original_colors_explosion_factor)
                if verbose:
                    visual_type = type(original_mesh.visual).__name__
                    has_uv = hasattr(original_mesh.visual, 'uv') and original_mesh.visual.uv is not None
                    print(f"  After explosion visual type: {visual_type}, has_uv: {has_uv}")
            
            original_scene = trimesh.Scene()
            original_scene.add_geometry(original_mesh, geom_name='original_mesh')
            
        elif isinstance(original_colors_source, trimesh.Scene):
            original_scene = original_colors_source
        else:
            # It's a Trimesh, wrap it in a Scene
            original_scene = trimesh.Scene()
            original_scene.add_geometry(original_colors_source, geom_name='original_mesh')

    if renderer == 'blender':
        raise Exception("Blender renderer depth buffer computation is not supported yet")

    # Calculate appropriate camera distance based on scene bounds
    scene_extents = scene.extents
    scene_scale = np.max(scene_extents)
    camera_distance = scene_scale * get_scale_multiplier(renderer)

    # Generate camera matrices using camera_utils
    lookat_position = torch.from_numpy(scene.centroid).float()

    camera_transforms = generate_camera_matrices(scene, camera_distance, camera_angles, lookat_position, camera_mode, num_views, verbose)

    for view_idx, (camera_transform, az_or_idx, el) in enumerate(tqdm(camera_transforms, desc="Rendering views")):
        if el is not None:
            # Manual angles mode
            if verbose:
                print(f"Rendering view {view_idx + 1}/{len(camera_transforms)}: "
                  f"azimuth={az_or_idx}°, elevation={el}°")
            view_name = f"az{az_or_idx}_el{el}"
        else:
            # Other modes
            if verbose:
                print(f"Rendering view {view_idx + 1}/{len(camera_transforms)}")
            view_name = f"view{view_idx:02d}"
        
        # Render using chosen backend
        output_path = os.path.join(output_dir, f"render_{view_name}.png")
        unlabelled_path = os.path.join(unlabelled_dir, f"render_{view_name}.png")
        img = None
        
        # Extract camera position and target for renderers that need it
        camera_eye = camera_transform[:3, 3]
        camera_target = scene.centroid

        # Render buffers dict will contain 'depth'
        render_buffers = None
        
        if renderer == "blender":
            img, depth_buffer = render_with_blender(scene, camera_eye, camera_target, resolution, output_path)
            render_buffers = {'depth': depth_buffer}
        elif renderer == "pyrender":
            img, render_buffers = render_with_pyrender_offscreen(scene, camera_transform, resolution, 
                                                                   verbose=verbose, flat_shading=flat_shading)

        # Save unlabelled version first
        img.save(unlabelled_path)
        
        # Add labels to the image (also returns positions for reuse)
        labeled_img, label_positions = add_part_labels(img.copy(), scene, part_info, camera_transform, resolution, render_buffers)
        
        # Save labeled image
        labeled_img.save(output_path)
        
        # Render with original colors if original mesh was provided
        if original_scene is not None:
            original_path = os.path.join(original_colors_dir, f"render_{view_name}.png")
            
            if renderer == "pyrender":
                # Render the original scene with the same camera transform
                # Use flat_shading=False for more realistic look with original colors
                original_img, _ = render_with_pyrender_offscreen(
                    original_scene, camera_transform, resolution, 
                    verbose=verbose, flat_shading=False
                )
            else:
                original_img = img  # Fallback
            
            # Apply the same labels to original colors render
            if original_img is not None and label_positions:
                original_img = draw_labels_at_positions(original_img, label_positions)
            
            original_img.save(original_path)
            if verbose:
                print(f"Saved original colors render to: {original_path}")
        
        if verbose:
            print(f"Saved renders to: {output_path} and {unlabelled_path}")
    
    print(f"\nAll renders saved to: {output_dir}")
    print(f"Unlabelled renders saved to: {unlabelled_dir}")
    if original_colors_dir is not None:
        print(f"Original colors renders saved to: {original_colors_dir}")


def render_with_pyrender_offscreen(scene, camera_transform, resolution, verbose=False, flat_shading=False):
    """
    Render using pyrender in offscreen mode (no pyglet needed).
    
    Args:
        scene: trimesh.Scene
        camera_transform: 4x4 camera matrix
        resolution: (width, height) tuple
        verbose: Print debug info
        flat_shading: If True, use flat/unlit rendering with no shadows or lighting effects.
                      This produces exact colors for reliable color matching.
    
    Returns:
        tuple: (PIL Image, dict with 'depth') or (None, None) if failed
    """
    try:
        import pyrender
        from io import BytesIO
        
        # Create pyrender scene
        if flat_shading:
            # Full ambient light, no shadows - colors will be exact
            pr_scene = pyrender.Scene(
                ambient_light=[1.0, 1.0, 1.0],  # Full ambient = no shading
                bg_color=[0.95, 0.95, 0.95, 1.0]
            )
            if verbose:
                print("  Using FLAT shading (no shadows/lighting)")
        else:
            # Normal lighting with shadows
            pr_scene = pyrender.Scene(
                ambient_light=[0.4, 0.4, 0.4],
                bg_color=[0.95, 0.95, 0.95, 1.0]
            )
        
        if verbose:
            print(f"  PyRender: Rendering {len(scene.geometry)} parts...")
        
        # Add all geometries from trimesh scene
        for geom_name, geometry in scene.geometry.items():
            if hasattr(geometry, 'vertices'):
                mesh_to_render = geometry
                
                if verbose:
                    print(f"    Part {geom_name}: {len(geometry.vertices)} vertices, {len(geometry.faces)} faces")
                    visual_kind = type(geometry.visual).__name__
                    print(f"      Visual type: {visual_kind}")
                
                # Check for TextureVisuals (PBR textures with UV coords)
                if hasattr(geometry.visual, 'uv') and geometry.visual.uv is not None:
                    # Mesh has texture - pyrender.Mesh.from_trimesh handles this directly
                    if verbose:
                        print(f"      Has UV texture mapping")
                    mesh_to_render = geometry  # Keep original with textures
                    
                elif hasattr(geometry.visual, 'face_colors') and geometry.visual.face_colors is not None:
                    # Convert face colors to vertex colors for pyrender compatibility
                    face_colors = geometry.visual.face_colors
                    vertex_colors = np.zeros((len(geometry.vertices), 4), dtype=np.uint8)
                    
                    # Average colors from all faces that use each vertex
                    vertex_count = np.zeros(len(geometry.vertices), dtype=np.int32)
                    vertex_color_sum = np.zeros((len(geometry.vertices), 4), dtype=np.float32)
                    
                    for face_idx, face in enumerate(geometry.faces):
                        for vertex_idx in face:
                            vertex_color_sum[vertex_idx] += face_colors[face_idx].astype(np.float32)
                            vertex_count[vertex_idx] += 1
                    
                    # Average the colors
                    for i in range(len(vertex_colors)):
                        if vertex_count[i] > 0:
                            vertex_colors[i] = (vertex_color_sum[i] / vertex_count[i]).astype(np.uint8)
                        else:
                            vertex_colors[i] = [200, 200, 200, 255]
                    
                    if verbose:
                        avg_color = np.mean(vertex_colors[:, :3], axis=0)
                        print(f"      Color: RGB({avg_color[0]:.0f}, {avg_color[1]:.0f}, {avg_color[2]:.0f})")
                    
                    # Create new mesh with vertex colors
                    mesh_to_render = trimesh.Trimesh(
                        vertices=geometry.vertices,
                        faces=geometry.faces,
                        vertex_colors=vertex_colors,
                        process=False
                    )
                
                elif hasattr(geometry.visual, 'vertex_colors') and geometry.visual.vertex_colors is not None:
                    # Already has vertex colors
                    if verbose:
                        print(f"      Has vertex colors")
                    mesh_to_render = geometry
                
                # Convert to pyrender mesh and add to scene
                try:
                    pr_mesh = pyrender.Mesh.from_trimesh(mesh_to_render, smooth=False)
                    pr_scene.add(pr_mesh)
                except Exception as e:
                    # Fallback: create mesh without colors
                    if verbose:
                        print(f"  Warning: Could not add colors for {geom_name}: {e}, using default material")
                    plain_mesh = trimesh.Trimesh(
                        vertices=geometry.vertices,
                        faces=geometry.faces,
                        process=False
                    )
                    pr_mesh = pyrender.Mesh.from_trimesh(plain_mesh, smooth=False)
                    pr_scene.add(pr_mesh)
        
        # Check if any meshes were added
        if len(pr_scene.meshes) == 0:
            if verbose:
                print("  Warning: No meshes added to pyrender scene!")
            return None, None
        
        if verbose:
            print(f"  Added {len(pr_scene.meshes)} meshes to scene")
        
        # Add camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        pr_scene.add(camera, pose=camera_transform)
        
        # Add lights (skip for flat shading - full ambient is enough)
        if not flat_shading:
            # Add multiple lights for better illumination from different angles
            # Main light from camera
            light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
            pr_scene.add(light1, pose=camera_transform)
            
            # Fill light from opposite side
            light_transform2 = camera_transform.copy()
            light_transform2[:3, 3] = -camera_transform[:3, 3] * 0.5
            light2 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
            pr_scene.add(light2, pose=light_transform2)
            
            # Top light
            light_transform3 = np.eye(4)
            light_transform3[:3, 3] = [0, 0, 10]
            light3 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
            pr_scene.add(light3, pose=light_transform3)
        
        # Render offscreen
        if verbose:
            print(f"  Rendering at {resolution[0]}x{resolution[1]}...")
        renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])
        color, depth = renderer.render(pr_scene)
        renderer.delete()
        
        if verbose:
            print(f"  Render complete")
        
        # Convert to PIL Image
        img = Image.fromarray(color)
        buffers = {'depth': depth}
        return img, buffers
        
    except Exception as e:
        print(f"  Pyrender offscreen failed: {e}")
        return None, None


def render_with_blender(scene, camera_pos, camera_target, resolution, output_path, glb_path=None):
    """
    Render using Blender (subprocess call).
    
    Args:
        scene: trimesh.Scene (used for camera calculations, not for rendering if glb_path is provided)
        camera_pos: camera position [x, y, z]
        camera_target: camera target [x, y, z]
        resolution: (width, height) tuple
        output_path: where to save the image
        glb_path: Optional path to original GLB file (preserves textures better than trimesh export)
    
    Returns:
        tuple: (PIL Image, depth_buffer) or (None, None) if failed
    """
    try:
        import subprocess
        import tempfile
        
        temp_dir = tempfile.mkdtemp()
        
        # Use original GLB if provided (preserves all textures), otherwise export from trimesh
        if glb_path and os.path.exists(glb_path):
            scene_path = glb_path
        else:
            scene_path = os.path.join(temp_dir, 'scene.glb')
            scene.export(scene_path)
        
        # Path for depth buffer output
        depth_path = os.path.join(temp_dir, 'depth_buffer.npy')
        
        # Create Blender Python script
        blender_script = f"""
import bpy
import math
import numpy as np

def direction_to_rotation(direction):
    import mathutils
    direction = mathutils.Vector(direction).normalized()
    up = mathutils.Vector((0, 0, 1))
    
    # Handle edge case where direction is parallel to up
    if abs(direction.dot(up)) > 0.999:
        up = mathutils.Vector((0, 1, 0))
    
    right = direction.cross(up).normalized()
    up = right.cross(direction).normalized()
    
    mat = mathutils.Matrix((right, up, -direction)).transposed().to_4x4()
    return mat.to_quaternion()

def compute_depth_buffer(camera_obj, scene, res_x, res_y):
    \"\"\"
    Compute depth buffer by ray casting (fallback method)

    Args:
        camera_obj: Camera object
        scene: Scene object
        res_x, res_y: Resolution

    Returns:
        2D array of depth values
    \"\"\"
    import mathutils
    from mathutils import Vector

    depsgraph = bpy.context.evaluated_depsgraph_get()
    depth_buffer = np.full((res_y, res_x), float('inf'))

    camera = camera_obj.data
    camera_matrix = camera_obj.matrix_world

    for y in range(res_y):
        for x in range(res_x):
            # Convert pixel to ray direction
            ndc_x = (2.0 * x / res_x) - 1.0
            ndc_y = 1.0 - (2.0 * y / res_y)

            if camera.type == 'PERSP':
                # Perspective projection
                aspect = res_x / res_y
                tan_half_fov = math.tan(camera.angle / 2.0)

                # Ray direction in camera space
                ray_dir_cam = Vector((
                    ndc_x * tan_half_fov * aspect,
                    ndc_y * tan_half_fov,
                    -1.0
                ))

                # Transform to world space
                ray_dir_world = camera_matrix.to_3x3() @ ray_dir_cam
                ray_dir_world.normalize()

                # Cast ray
                result = scene.ray_cast(depsgraph, camera_matrix.translation, ray_dir_world)

                if result[0]:  # Hit
                    distance = (result[1] - camera_matrix.translation).length
                    depth_buffer[y, x] = distance
            else:
                # Orthographic projection
                ray_origin = camera_matrix @ Vector((ndc_x, ndc_y, 0))
                ray_dir = camera_matrix.to_3x3() @ Vector((0, 0, -1))

                result = scene.ray_cast(depsgraph, ray_origin, ray_dir)

                if result[0]:  # Hit
                    distance = (result[1] - ray_origin).length
                    depth_buffer[y, x] = distance

    return depth_buffer

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import the scene
bpy.ops.import_scene.gltf(filepath='{scene_path}')

# Set up camera
cam_data = bpy.data.cameras.new('Camera')
cam_obj = bpy.data.objects.new('Camera', cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

# Set camera position and target
cam_obj.location = {tuple(camera_pos)}
target_location = {tuple(camera_target)}

# Point camera at target
direction = tuple(target_location[i] - cam_obj.location[i] for i in range(3))
rot_quat = direction_to_rotation(direction)
cam_obj.rotation_euler = rot_quat.to_euler()



cam_obj.rotation_euler = direction_to_rotation(direction).to_euler()

# Set up lighting
# Main key light from camera position
light_data1 = bpy.data.lights.new('KeyLight', 'SUN')
light_data1.energy = 5.0
light_obj1 = bpy.data.objects.new('KeyLight', light_data1)
bpy.context.scene.collection.objects.link(light_obj1)
light_obj1.location = cam_obj.location

# Fill light from opposite side (softer)
light_data2 = bpy.data.lights.new('FillLight', 'SUN')
light_data2.energy = 3.0
light_obj2 = bpy.data.objects.new('FillLight', light_data2)
bpy.context.scene.collection.objects.link(light_obj2)
fill_pos = tuple(-x * 0.5 for x in cam_obj.location)
light_obj2.location = fill_pos

# Top light 
light_data3 = bpy.data.lights.new('TopLight', 'SUN')
light_data3.energy = 3.0
light_obj3 = bpy.data.objects.new('TopLight', light_data3)
bpy.context.scene.collection.objects.link(light_obj3)
light_obj3.location = (0, 0, 10)

# Rim/back light 
light_data4 = bpy.data.lights.new('RimLight', 'SUN')
light_data4.energy = 2.0
light_obj4 = bpy.data.objects.new('RimLight', light_data4)
bpy.context.scene.collection.objects.link(light_obj4)
rim_pos = tuple(-x * 1.5 for x in cam_obj.location[:2]) + (cam_obj.location[2] + 5,)
light_obj4.location = rim_pos

# Add ambient/world lighting for overall brightness
bpy.context.scene.world.use_nodes = True
world_nodes = bpy.context.scene.world.node_tree.nodes
bg_node = world_nodes.get('Background')
if bg_node:
    bg_node.inputs['Color'].default_value = (0.4, 0.4, 0.4, 1.0)
    bg_node.inputs['Strength'].default_value = 1.5

# Set render settings
bpy.context.scene.render.resolution_x = {resolution[0]}
bpy.context.scene.render.resolution_y = {resolution[1]}
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.image_settings.color_mode = 'RGBA'  # Enable alpha channel output
bpy.context.scene.render.filepath = '{output_path}'

# Configure render engine
# Use Cycles for better transparency/PBR, or EEVEE for faster rendering
USE_CYCLES = True  # Set to True for better transparency handling

if USE_CYCLES:
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 64  # Lower samples for speed
    bpy.context.scene.cycles.use_denoising = True
    # Use GPU if available
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.scene.cycles.device = 'GPU'
    for device in bpy.context.preferences.addons['cycles'].preferences.devices:
        device.use = True
    print("Using Cycles renderer with GPU")
else:
    # Try EEVEE_NEXT for Blender 4.2+, fall back to BLENDER_EEVEE
    try:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    except:
        try:
            bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        except:
            print("Warning: Could not set EEVEE, using default engine")

# Configure EEVEE settings (handle API changes between versions)
eevee = bpy.context.scene.eevee
if hasattr(eevee, 'use_gtao'):
    eevee.use_gtao = True
    eevee.gtao_distance = 0.2
if hasattr(eevee, 'taa_render_samples'):
    eevee.taa_render_samples = 64
elif hasattr(eevee, 'taa_samples'):
    eevee.taa_samples = 64

# Connect alpha channel from base color texture to Principled BSDF Alpha input
# This enables proper transparency rendering for TRELLIS meshes exported as OPAQUE
for mat in bpy.data.materials:
    if mat.use_nodes:
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Find Principled BSDF node
        principled = None
        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                principled = node
                break
        
        if principled is None:
            continue
        
        # Find the image texture connected to Base Color
        base_color_input = principled.inputs.get('Base Color')
        if base_color_input and base_color_input.is_linked:
            from_node = base_color_input.links[0].from_node
            
            # Check if it's an image texture with alpha channel
            if from_node.type == 'TEX_IMAGE' and from_node.image:
                if from_node.image.channels == 4:  # RGBA image has alpha
                    # Connect Alpha output to Principled BSDF Alpha input
                    alpha_output = from_node.outputs.get('Alpha')
                    alpha_input = principled.inputs.get('Alpha')
                    
                    if alpha_output and alpha_input and not alpha_input.is_linked:
                        links.new(alpha_output, alpha_input)
                        print(f"Connected alpha for material: {{mat.name}}")
                        
                        # Set material blend mode for transparency
                        if hasattr(mat, 'blend_method'):
                            mat.blend_method = 'HASHED'  # HASHED works better than BLEND for mixed meshes
                        if hasattr(mat, 'shadow_method'):
                            mat.shadow_method = 'HASHED'
                        if hasattr(mat, 'use_backface_culling'):
                            mat.use_backface_culling = False
                        if hasattr(mat, 'show_transparent_back'):
                            mat.show_transparent_back = True

# Render
bpy.ops.render.render(write_still=True)

# Compute depth buffer
print("Computing depth buffer...")
depth_buffer = compute_depth_buffer(cam_obj, bpy.context.scene, {resolution[0]}, {resolution[1]})
np.save('{depth_path}', depth_buffer)
print(f"Depth buffer saved to {{'{depth_path}'}}")
"""
        
        script_path = os.path.join(temp_dir, 'render_script.py')
        with open(script_path, 'w') as f:
            f.write(blender_script)
        
        # Run Blender
        blender_cmd = ['blender', '--background', '--python', script_path]
        print("  Running Blender...")
        result = subprocess.run(
            blender_cmd,
            capture_output=True,
            timeout=120  # Increased timeout for depth computation
        )
        
        # Load results
        img = None
        depth_buffer = None
        
        # Debug: capture Blender output. NOTE: Blender emits Python tracebacks to
        # *stderr* and frequently still exits with returncode 0, so checking only
        # returncode / stdout silently swallows real errors (e.g. a missing numpy
        # in Blender's bundled Python crashing the glTF importer).
        stdout_text = result.stdout.decode(errors='replace') if result.stdout else ""
        stderr_text = result.stderr.decode(errors='replace') if result.stderr else ""
        combined_text = stdout_text + "\n" + stderr_text

        error_markers = ('Traceback', 'Error:', 'ModuleNotFoundError', 'ImportError')
        has_error = any(marker in combined_text for marker in error_markers)

        if result.returncode != 0 or has_error:
            print(f"  Blender reported a problem (returncode={result.returncode})")
            # Surface the actual error lines from whichever stream they appear in.
            for stream_name, stream_text in (('stderr', stderr_text), ('stdout', stdout_text)):
                if any(marker in stream_text for marker in error_markers):
                    error_lines = []
                    in_error = False
                    for line in stream_text.split('\n'):
                        if any(marker in line for marker in error_markers):
                            in_error = True
                        if in_error:
                            error_lines.append(line)
                    if error_lines:
                        print(f"  --- Blender {stream_name} (error excerpt) ---")
                        print("  " + "\n  ".join(error_lines[-25:]))

        if os.path.exists(output_path):
            img = Image.open(output_path)
        else:
            print(f"  Output image not found at: {output_path}")
            # Full diagnostics: the render produced no file, dump everything so the
            # failure is actionable instead of a bare "not found" warning.
            print(f"  Blender command: {' '.join(blender_cmd)}")
            print(f"  Return code: {result.returncode}")
            print(f"  Script preserved for inspection: {script_path}")
            print(f"  --- Blender stderr (last 1500 chars) ---")
            print(stderr_text[-1500:] if stderr_text else "  (empty)")
            print(f"  --- Blender stdout (last 1500 chars) ---")
            print(stdout_text[-1500:] if stdout_text else "  (empty)")
            
        if os.path.exists(depth_path):
            depth_buffer = np.load(depth_path)
        
        # Clean up only on success; keep temp_dir (incl. render_script.py) when the
        # render failed so the generated script and inputs can be inspected.
        import shutil
        if img is not None:
            shutil.rmtree(temp_dir)
            return img, depth_buffer
        else:
            print(f"  Temp dir preserved for debugging: {temp_dir}")
            return None, None
            
    except Exception as e:
        import traceback
        print(f"  Blender rendering failed: {e}")
        traceback.print_exc()
        return None, None

def get_scale_multiplier(renderer):
    if renderer == 'blender':
        return 3.0
    elif renderer == 'pyrender':
        return 1.25

def generate_camera_matrices(scene, camera_distance, camera_angles, lookat_position, camera_mode='angles', num_views=None, verbose=False):

    if camera_mode == 'angles':
        # Manual azimuth/elevation specification (backward compatibility)
        if verbose:
            print(f"Camera mode: manual angles ({len(camera_angles)} views)")
        camera_transforms = []
        for azimuth, elevation in camera_angles:
            azimuth_rad = np.radians(azimuth)
            elevation_rad = np.radians(elevation)
            
            # Calculate camera position
            camera_pos = np.array([
                camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad),
                camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad),
                camera_distance * np.sin(elevation_rad)
            ])
            camera_pos_torch = torch.from_numpy(camera_pos + scene.centroid).float()
            
            # Use camera_utils.view_matrix
            cam_matrix = view_matrix(
                camera_pos_torch,
                lookat_position,
                up=torch.tensor([0.0, 0.0, 1.0])
            )
            camera_transforms.append((cam_matrix[0].numpy(), azimuth, elevation))
        
    elif camera_mode == 'uniform':
        # Uniform random sampling
        n = num_views if num_views is not None else 12
        if verbose:
            print(f"Camera mode: uniform random ({n} views)")
        cam_matrices = sample_view_matrices(n, camera_distance, lookat_position)
        camera_transforms = [(cam_matrices[i].numpy(), i, None) for i in range(n)]
        
    elif camera_mode in ['standard', 'icosahedron', 'dodecahedron', 'octohedron', 'cube', 'tetrahedron']:
        # Polyhedra-based sampling
        kwargs = {}
        if camera_mode == 'standard':
            kwargs['n'] = num_views if num_views is not None else 8
            kwargs['elevation'] = 15
            if verbose:
                print(f"Camera mode: standard ({kwargs['n']} views, elevation={kwargs['elevation']}°)")
        else:
            if verbose:
                print(f"Camera mode: {camera_mode}")
        
        cam_matrices = sample_view_matrices_polyhedra(
            camera_mode,
            camera_distance,
            lookat_position,
            **kwargs
        )
        camera_transforms = [(cam_matrices[i].numpy(), i, None) for i in range(len(cam_matrices))]
    
    else:
        raise ValueError(f"Unknown camera_mode: {camera_mode}")

    return camera_transforms

def convert_hunyuan_to_partnet(input_glb: str, output_path: str = None, verbose: bool = False):
    """
    Copy TRELLIS/Hunyuan GLB to output path, preserving all PBR materials and textures.
    TRELLIS output already has correct orientation, so we just copy the file.
    
    Args:
        input_glb: Path to TRELLIS/Hunyuan generated GLB file
        output_path: Output path (default: same directory with _partnet suffix)
        verbose: Print debug output
    
    Returns:
        Path to output mesh (GLB format)
    """
    import shutil
    
    if output_path is None:
        input_path = Path(input_glb)
        output_path = str(input_path.parent / f"{input_path.stem}_partnet.glb")
    
    # Ensure output is GLB format for PBR preservation
    output_path_obj = Path(output_path)
    if output_path_obj.suffix.lower() != '.glb':
        output_path = str(output_path_obj.with_suffix('.glb'))
    
    if verbose:
        print(f"Converting: {input_glb}")
        print(f"Output: {output_path}")
    
    # Simply copy the original GLB to preserve ALL materials and textures
    # TRELLIS output already has the correct orientation
    shutil.copy(input_glb, output_path)
    
    # Load to verify and print bounds
    if verbose:
        loaded = trimesh.load(output_path)
        if isinstance(loaded, trimesh.Scene):
            print(f"  Input is Scene with {len(loaded.geometry)} geometries")
            print(f"  After applying embedded transforms:")
            print(f"    Bounds: {loaded.bounds[0]} to {loaded.bounds[1]}")
        else:
            print(f"  After applying embedded transforms:")
            print(f"    Bounds: {loaded.bounds[0]} to {loaded.bounds[1]}")
    
    if verbose:
        print(f"Converted to PartNet convention (GLB with PBR): {output_path}")
    
    return output_path




def create_original_colors_scene(mesh: trimesh.Trimesh, face_ids: np.ndarray = None, 
                                   explosion_factor: float = 0.0):
    """
    Create a scene from a mesh preserving its original colors.
    Optionally explodes the mesh if face_ids and explosion_factor are provided.
    
    Args:
        mesh: The mesh with original colors
        face_ids: Per-face segment IDs (required if explosion_factor > 0)
        explosion_factor: How much to explode (0 = no explosion)
        
    Returns:
        trimesh.Scene with original colors preserved
    """
    if explosion_factor > 0 and face_ids is not None:
        # Explode the mesh while preserving original colors
        exploded_mesh, _ = explode_mesh(mesh, face_ids, explosion_factor)
        scene = trimesh.Scene()
        scene.add_geometry(exploded_mesh, geom_name='original_colors_mesh')
        return scene
    else:
        # Just wrap the original mesh in a scene
        scene = trimesh.Scene()
        scene.add_geometry(mesh, geom_name='original_colors_mesh')
        return scene


def explode_mesh(mesh: trimesh.Trimesh, face_ids: np.ndarray, explosion_factor: float = 0.5):
    """
    Explode a segmented mesh by moving each segment outward from the mesh centroid.
    Preserves texture/UV information if present.
    
    Args:
        mesh: The mesh to explode
        face_ids: Per-face segment IDs
        explosion_factor: How much to explode (0 = no explosion, 1 = full segment distance)
        
    Returns:
        Exploded mesh with vertices duplicated per face, and new face_ids
    """
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    # Check if mesh has texture UVs
    has_texture = (hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and 
                   len(mesh.visual.uv) > 0)
    original_uvs = None
    if has_texture:
        original_uvs = np.array(mesh.visual.uv)
    
    # Compute mesh centroid
    mesh_centroid = vertices.mean(axis=0)
    
    # Compute face centroids
    face_centroids = vertices[faces].mean(axis=1)
    
    # Compute segment centroids
    unique_segments = np.unique(face_ids)
    segment_centroids = {}
    for seg_id in unique_segments:
        mask = face_ids == seg_id
        segment_centroids[seg_id] = face_centroids[mask].mean(axis=0)
    
    # Compute offset direction and distance for each segment
    segment_offsets = {}
    for seg_id in unique_segments:
        direction = segment_centroids[seg_id] - mesh_centroid
        distance = np.linalg.norm(direction)
        if distance > 1e-6:
            direction = direction / distance
        else:
            direction = np.array([0, 0, 0])
        segment_offsets[seg_id] = direction * explosion_factor * distance
    
    # Create new vertices by duplicating per face and applying offsets
    new_vertices = []
    new_faces = []
    new_face_ids = []
    new_uvs = [] if has_texture else None
    
    for face_idx, face in enumerate(faces):
        seg_id = face_ids[face_idx]
        offset = segment_offsets[seg_id]
        
        base_idx = len(new_vertices)
        for vertex_idx in face:
            new_vertices.append(vertices[vertex_idx] + offset)
            # Copy UV coordinate for each duplicated vertex
            if has_texture and vertex_idx < len(original_uvs):
                new_uvs.append(original_uvs[vertex_idx])
        
        new_faces.append([base_idx, base_idx + 1, base_idx + 2])
        new_face_ids.append(seg_id)
    
    new_vertices = np.array(new_vertices)
    new_faces = np.array(new_faces)
    new_face_ids = np.array(new_face_ids)
    
    # Create new mesh with texture if available
    if has_texture and new_uvs:
        new_uvs = np.array(new_uvs)
        # Preserve the material/texture from original mesh
        if hasattr(mesh.visual, 'material'):
            exploded_mesh = trimesh.Trimesh(
                vertices=new_vertices, 
                faces=new_faces, 
                process=False
            )
            # Create TextureVisuals with UV and material
            exploded_mesh.visual = trimesh.visual.TextureVisuals(
                uv=new_uvs,
                material=mesh.visual.material
            )
        else:
            exploded_mesh = trimesh.Trimesh(
                vertices=new_vertices, 
                faces=new_faces, 
                process=False
            )
    else:
        # Create new mesh without texture
        exploded_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
        
        # Copy face colors if available
        if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
            exploded_mesh.visual.face_colors = mesh.visual.face_colors
    
    return exploded_mesh, new_face_ids


def render_object(cfg: DictConfig, verbose=False):

    input_object_path = cfg.input_object_path
    object_path = cfg.object_path
    output_dir = cfg.out_dir
    resolution = cfg.resolution
    renderer = cfg.get('renderer', 'pyrender')  # pyrender is the default backend (no blender install needed)
    camera_mode = cfg.camera_mode
    num_views = cfg.num_views
    convert_to_partnet = cfg.convert_to_partnet


    os.makedirs(output_dir, exist_ok=True)
    
    if Path(output_dir + "/render_view00.png").exists() and not cfg.rerun:
        return
    if verbose:
        print(f"Rendering object")
   
    if convert_to_partnet:
        convert_hunyuan_to_partnet(input_object_path, object_path, verbose)
    else:
        shutil.copy(input_object_path, object_path)
    
    if verbose:
        print(f"Using renderer: {renderer}")
        print()

    
    scene = trimesh.load(object_path)

    # Calculate appropriate camera distance based on scene bounds
    scene_extents = scene.extents
    scene_scale = np.max(scene_extents)
    camera_distance = scene_scale * get_scale_multiplier(renderer)  # Distance as multiple of scene size
  
    lookat_position = torch.from_numpy(scene.centroid).float()

    if camera_mode == 'angles':
        camera_angles = [[az, el] for az in cfg.camera_angles for el in cfg.camera_angles]
    else:
        camera_angles = None

    camera_transforms = generate_camera_matrices(scene, camera_distance, camera_angles, lookat_position, camera_mode, num_views, verbose)


    # Render from differnet angles
    for view_idx, (camera_transform, az_or_idx, el) in enumerate(tqdm(camera_transforms, desc="Rendering views")):
        if el is not None:
            # Manual angles mode
            if verbose:
                print(f"Rendering view {view_idx + 1}/{len(camera_transforms)}: "
                  f"azimuth={az_or_idx}°, elevation={el}°")
            view_name = f"az{az_or_idx}_el{el}"
        else:
            # Other modes
            if verbose:
                print(f"Rendering view {view_idx + 1}/{len(camera_transforms)}")
            view_name = f"view{view_idx:02d}"
        
        output_path = os.path.join(output_dir, f"render_{view_name}.png")
        img = None
        
        # Extract camera position and target for renderers that need it
        camera_eye = camera_transform[:3, 3]
        camera_target = scene.centroid

        if renderer == "blender":
            # Pass original GLB path to Blender to preserve textures
            img, depth_buffer = render_with_blender(scene, camera_eye, camera_target, resolution, output_path, glb_path=object_path)
        elif renderer == "pyrender":
            img, depth_buffer = render_with_pyrender_offscreen(scene, camera_transform, resolution)

        if img is not None:
            img.save(output_path)
            if verbose:
                print(f"Saved render to: {output_path}")
        else:
            print(f"  Warning: Render failed for {view_name}")
    
    print(f"\nAll renders saved to: {output_dir}")

    return object_path


def explode_mesh_interactive(mesh, face_ids, max_explosion=2.0, verbose=False):
    """
    Create an interactive 3D visualization of an exploded mesh with a slider control.
    
    Args:
        mesh: trimesh.Trimesh object or path to mesh file
        face_ids: numpy array of face labels or path to face_ids .npy file
        max_explosion: maximum explosion factor for the slider
        verbose: print debug information
    
    Returns:
        fig: plotly figure object (also displays it)
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plotly is required for interactive visualization. Install with: pip install plotly")
    
    # Load mesh if path is provided
    if isinstance(mesh, (str, Path)):
        mesh = trimesh.load(str(mesh), force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
    
    # Load face_ids if path is provided
    if isinstance(face_ids, (str, Path)):
        face_ids = np.load(str(face_ids))
    
    # Get unique part IDs
    unique_ids = np.unique(face_ids)
    valid_ids = unique_ids[unique_ids >= 0]
    n_parts = len(valid_ids)
    
    if verbose:
        print(f"Found {n_parts} parts in mesh")
    
    # Calculate mesh centroid
    mesh_centroid = mesh.vertices.mean(axis=0)
    
    # Generate maximally distinct colors for each part
    DISTINCT_COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        (0, 255, 128), (255, 0, 128), (128, 255, 0), (0, 128, 255),
        (179, 77, 0), (77, 179, 77), (77, 77, 179), (204, 204, 0),
        (204, 0, 204), (0, 204, 204), (255, 179, 102), (102, 179, 255),
    ]
    colors = []
    for i in range(n_parts):
        if i < len(DISTINCT_COLORS):
            r, g, b = DISTINCT_COLORS[i]
        else:
            hue = ((i - len(DISTINCT_COLORS)) * 0.618033988749895 + 0.1) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.7 + (i % 3) * 0.1, 0.6 + (i % 4) * 0.1)
            r, g, b = int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
        colors.append(f'rgb({r},{g},{b})')
    
    # Extract part data
    parts_data = []
    for idx, part_id in enumerate(valid_ids):
        part_mask = face_ids == part_id
        part_faces_indices = np.where(part_mask)[0]
        
        if len(part_faces_indices) == 0:
            continue
        
        part_faces = mesh.faces[part_faces_indices]
        vertices_used = np.unique(part_faces.flatten())
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(vertices_used)}
        
        part_vertices = mesh.vertices[vertices_used].copy()
        part_faces_remapped = np.array([
            [vertex_map[v] for v in face] 
            for face in part_faces
        ])
        
        # Calculate part centroid
        part_centroid = part_vertices.mean(axis=0)
        
        # Direction from mesh center to part center (for explosion)
        explosion_dir = part_centroid - mesh_centroid
        if np.linalg.norm(explosion_dir) > 1e-6:
            explosion_dir = explosion_dir / np.linalg.norm(explosion_dir)
        else:
            explosion_dir = np.array([0, 0, 0])
        
        parts_data.append({
            'vertices': part_vertices,
            'faces': part_faces_remapped,
            'centroid': part_centroid,
            'explosion_dir': explosion_dir,
            'color': colors[idx % len(colors)],
            'label': f'Part {idx}'
        })
    
    # Calculate scene scale for explosion distance
    scene_scale = np.max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    
    # Create frames for different explosion levels
    n_steps = 51  # Number of slider steps
    explosion_values = np.linspace(0, max_explosion, n_steps)
    
    frames = []
    for exp_val in explosion_values:
        frame_data = []
        for part in parts_data:
            # Explode vertices
            offset = part['explosion_dir'] * exp_val * scene_scale * 0.3
            exploded_verts = part['vertices'] + offset
            
            # Create mesh3d trace
            frame_data.append(go.Mesh3d(
                x=exploded_verts[:, 0],
                y=exploded_verts[:, 1],
                z=exploded_verts[:, 2],
                i=part['faces'][:, 0],
                j=part['faces'][:, 1],
                k=part['faces'][:, 2],
                color=part['color'],
                opacity=0.95,
                name=part['label'],
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3),
                lightposition=dict(x=100, y=100, z=100)
            ))
        
        frames.append(go.Frame(data=frame_data, name=f'{exp_val:.2f}'))
    
    # Create initial figure with unexploded mesh
    initial_data = []
    for part in parts_data:
        initial_data.append(go.Mesh3d(
            x=part['vertices'][:, 0],
            y=part['vertices'][:, 1],
            z=part['vertices'][:, 2],
            i=part['faces'][:, 0],
            j=part['faces'][:, 1],
            k=part['faces'][:, 2],
            color=part['color'],
            opacity=0.95,
            name=part['label'],
            flatshading=True,
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3),
            lightposition=dict(x=100, y=100, z=100)
        ))
    
    fig = go.Figure(data=initial_data, frames=frames)
    
    # Create slider
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Explosion: ", "suffix": "x", "visible": True},
        pad={"t": 50, "b": 10},
        len=0.9,
        x=0.05,
        xanchor="left",
        steps=[
            dict(
                method="animate",
                args=[[f'{exp_val:.2f}'],
                      dict(mode="immediate",
                           frame=dict(duration=0, redraw=True),
                           transition=dict(duration=0))],
                label=f'{exp_val:.1f}'
            )
            for exp_val in explosion_values
        ]
    )]
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Interactive Mesh Explosion ({n_parts} parts)',
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(showgrid=True, gridcolor='lightgray', showbackground=True, backgroundcolor='rgb(240,240,240)'),
            yaxis=dict(showgrid=True, gridcolor='lightgray', showbackground=True, backgroundcolor='rgb(240,240,240)'),
            zaxis=dict(showgrid=True, gridcolor='lightgray', showbackground=True, backgroundcolor='rgb(240,240,240)'),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0)
            )
        ),
        sliders=sliders,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=0, r=0, t=50, b=100),
        width=1000,
        height=800
    )
    
    if verbose:
        print(f"Created interactive visualization with {n_parts} parts and {n_steps} explosion steps")
    
    fig.show()
    return fig


def explode_mesh_to_file(mesh, face_ids, output_path, explosion_factor=1.0, verbose=False):
    """
    Export an exploded mesh to a file.
    
    Args:
        mesh: trimesh.Trimesh object or path to mesh file
        face_ids: numpy array of face labels or path to face_ids .npy file
        output_path: path to save the exploded mesh (should end with .glb, .ply, .obj, etc.; .glb recommended for best compatibility)
        explosion_factor: how much to explode (0 = no explosion, 1 = moderate, 2 = more)
        verbose: print debug information
    
    Returns:
        scene: trimesh.Scene with exploded parts
    """
    # Load mesh if path is provided
    if isinstance(mesh, (str, Path)):
        mesh = trimesh.load(str(mesh), force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
    
    # Load face_ids if path is provided
    if isinstance(face_ids, (str, Path)):
        face_ids = np.load(str(face_ids))
    
    # Create scene from parts
    scene, part_info, label_mapping = create_scene_from_parts(mesh, face_ids, verbose=verbose)
    
    # Calculate mesh centroid and scale
    mesh_centroid = mesh.vertices.mean(axis=0)
    scene_scale = np.max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    
    # Explode each part
    exploded_scene = trimesh.Scene()
    for part_id, info in part_info.items():
        geom_name = info['geom_name']
        part_mesh = scene.geometry[geom_name].copy()
        
        # Calculate explosion direction and offset
        explosion_dir = info['center'] - mesh_centroid
        if np.linalg.norm(explosion_dir) > 1e-6:
            explosion_dir = explosion_dir / np.linalg.norm(explosion_dir)
        else:
            explosion_dir = np.array([0, 0, 0])
        
        offset = explosion_dir * explosion_factor * scene_scale * 0.3
        part_mesh.vertices += offset
        
        exploded_scene.add_geometry(part_mesh, geom_name=geom_name)
    
    # Export
    exploded_scene.export(output_path)
    if verbose:
        print(f"Exported exploded mesh to: {output_path}")
    
    return exploded_scene


def filter_small_parts(mesh, face_ids, min_faces=20, remove_faces=False, verbose=False):
    """
    Filter out small parts from a segmented mesh.
    
    Args:
        mesh: trimesh.Trimesh object or path to mesh file
        face_ids: numpy array of face labels or path to face_ids .npy file
        min_faces: minimum number of faces for a part to be kept (default: 20)
        remove_faces: if True, removes the faces entirely from the mesh;
                      if False, marks them as -1 in face_ids (default: False)
        verbose: print debug information
    
    Returns:
        If remove_faces=False:
            filtered_face_ids: numpy array with small parts marked as -1
            stats: dict with filtering statistics
        If remove_faces=True:
            filtered_mesh: trimesh.Trimesh with small parts removed
            filtered_face_ids: numpy array with updated labels
            stats: dict with filtering statistics
    """
    # Load mesh if path is provided
    if isinstance(mesh, (str, Path)):
        mesh = trimesh.load(str(mesh), force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
    
    # Load face_ids if path is provided
    if isinstance(face_ids, (str, Path)):
        face_ids = np.load(str(face_ids))
    
    # Make a copy to avoid modifying the original
    face_ids = face_ids.copy()
    
    # Get unique part IDs (excluding invalid ones like -1)
    unique_ids = np.unique(face_ids)
    valid_ids = unique_ids[unique_ids >= 0]
    
    # Count faces per part
    parts_to_remove = []
    parts_kept = []
    total_faces_removed = 0
    
    for part_id in valid_ids:
        part_mask = face_ids == part_id
        face_count = np.sum(part_mask)
        
        if face_count < min_faces:
            parts_to_remove.append((int(part_id), int(face_count)))
            total_faces_removed += face_count
            # Mark these faces as invalid
            face_ids[part_mask] = -1
        else:
            parts_kept.append((int(part_id), int(face_count)))
    
    stats = {
        'original_parts': len(valid_ids),
        'parts_removed': len(parts_to_remove),
        'parts_kept': len(parts_kept),
        'faces_removed': total_faces_removed,
        'removed_parts_detail': sorted(parts_to_remove, key=lambda x: x[1]),
        'kept_parts_detail': sorted(parts_kept, key=lambda x: x[1], reverse=True)
    }
    
    if verbose:
        print(f"Filtering small parts (min_faces={min_faces}):")
        print(f"  Original parts: {stats['original_parts']}")
        print(f"  Parts removed: {stats['parts_removed']} ({total_faces_removed} faces)")
        print(f"  Parts kept: {stats['parts_kept']}")
        if parts_to_remove:
            print(f"  Removed parts (id, faces): {stats['removed_parts_detail']}")
    
    if not remove_faces:
        return face_ids, stats
    
    # Remove the faces entirely and rebuild the mesh
    valid_face_mask = face_ids >= 0
    
    if verbose:
        print(f"  Removing {np.sum(~valid_face_mask)} faces from mesh...")
    
    # Get the faces we want to keep
    kept_faces = mesh.faces[valid_face_mask]
    kept_face_ids = face_ids[valid_face_mask]
    
    # Find all vertices used by kept faces
    vertices_used = np.unique(kept_faces.flatten())
    
    # Create vertex mapping
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(vertices_used)}
    
    # Remap faces to new vertex indices
    new_faces = np.array([
        [vertex_map[v] for v in face]
        for face in kept_faces
    ])
    
    # Create new mesh
    new_vertices = mesh.vertices[vertices_used]
    
    # Handle vertex colors/normals if present
    new_mesh_kwargs = {
        'vertices': new_vertices,
        'faces': new_faces,
        'process': False
    }
    
    # Try to preserve vertex normals
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
        try:
            new_mesh_kwargs['vertex_normals'] = mesh.vertex_normals[vertices_used]
        except:
            pass
    
    filtered_mesh = trimesh.Trimesh(**new_mesh_kwargs)
    
    # Relabel face_ids to be contiguous (0, 1, 2, ...)
    unique_kept = np.unique(kept_face_ids)
    label_map = {old_id: new_id for new_id, old_id in enumerate(unique_kept)}
    relabeled_face_ids = np.array([label_map[fid] for fid in kept_face_ids])
    
    if verbose:
        print(f"  New mesh: {len(new_vertices)} vertices, {len(new_faces)} faces")
        print(f"  Relabeled to {len(unique_kept)} contiguous part IDs (0 to {len(unique_kept)-1})")
    
    return filtered_mesh, relabeled_face_ids, stats


def save_filtered_segmentation(mesh_path, face_ids_path, output_dir, min_faces=20, remove_faces=False, verbose=True):
    """
    Convenience function to filter small parts and save results.
    
    Args:
        mesh_path: path to mesh file
        face_ids_path: path to face_ids .npy file
        output_dir: directory to save filtered results
        min_faces: minimum faces per part
        verbose: print info
    
    Returns:
        dict with paths to saved files and statistics
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load
    mesh = trimesh.load(mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    face_ids = np.load(face_ids_path)
    
    if verbose:
        print(f"Loaded mesh with {len(mesh.faces)} faces")
        print(f"Original segmentation has {len(np.unique(face_ids[face_ids >= 0]))} parts")
    
    if not remove_faces:
        filtered_face_ids, stats = filter_small_parts(mesh, face_ids, min_faces=min_faces, 
                                                       remove_faces=remove_faces, verbose=verbose)
    else:
        filtered_mesh, filtered_face_ids, stats = filter_small_parts(mesh, face_ids, min_faces=min_faces, 
                                                                      remove_faces=remove_faces, verbose=verbose)
        filtered_mesh.export(os.path.join(output_dir, 'filtered_mesh.glb'))
    
    # Save filtered face_ids
    filtered_face_ids_path = os.path.join(output_dir, 'filtered_face_ids.npy')
    np.save(filtered_face_ids_path, filtered_face_ids)
    
    # Also save a version with contiguous labels for parts that remain
    unique_valid = np.unique(filtered_face_ids[filtered_face_ids >= 0])
    label_map = {old_id: new_id for new_id, old_id in enumerate(unique_valid)}
    label_map[-1] = -1  # Keep -1 as -1
    relabeled = np.array([label_map[fid] for fid in filtered_face_ids])
    relabeled_path = os.path.join(output_dir, 'final_face_ids.npy')
    np.save(relabeled_path, relabeled)
    
    if verbose:
        print(f"\nSaved filtered segmentation:")
        print(f"  {filtered_face_ids_path} (original IDs, small parts marked as -1)")
        print(f"  {relabeled_path} (contiguous IDs: 0 to {len(unique_valid)-1})")
    
    return {
        'filtered_face_ids_path': filtered_face_ids_path,
        'relabeled_face_ids_path': relabeled_path,
        'stats': stats
    }