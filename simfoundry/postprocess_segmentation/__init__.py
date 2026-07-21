"""
Postprocess segmentation module.

Provides utilities for merging segmented mesh parts and interactive correction.
"""

from .merge import (
    merge_and_center_segmented_mesh,
    load_segmentation_data,
    get_base_segments,
    export_mesh_parts,
    create_unified_mesh,
)

from .interactive_ui import (
    interactive_segment_correction,
    SegmentCorrectionApp,
)

from .geometry import (
    SegmentGeometry,
    precompute_segment_geometry,
    compute_exploded_vertices,
)

from .visualization import SegmentFigureBuilder

from .styles import (
    STYLES,
    EXPLOSION_SCALE,
    GOLDEN_RATIO,
    generate_part_colors,
)


__all__ = [
    # Main functions
    'merge_and_center_segmented_mesh',
    'interactive_segment_correction',
    
    # Classes
    'SegmentCorrectionApp',
    'SegmentFigureBuilder',
    'SegmentGeometry',
    
    # Utilities
    'load_segmentation_data',
    'get_base_segments',
    'export_mesh_parts',
    'create_unified_mesh',
    'precompute_segment_geometry',
    'compute_exploded_vertices',
    'generate_part_colors',
    
    # Constants
    'STYLES',
    'EXPLOSION_SCALE',
    'GOLDEN_RATIO',
]
