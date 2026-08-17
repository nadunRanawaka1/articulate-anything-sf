"""Interactive refinement of published articulation results.

The articulation counterpart of postprocess_segmentation: a Dash web UI to
review and refine results/mobility.urdf — joint limits, pivot positions, axes,
joint types, and dynamic properties (damping, friction, mass) — with a live
3D preview of the articulated motion.

Entry points:
    - interactive_articulation_refinement: launch the UI over
      {object_name: results_dir}.
    - ArticulationModel: the non-UI URDF edit model (parse / FK / edits /
      validated save with backups and physics_overrides.json).
    - load_physics_overrides / merge_parts_properties: stdlib-only helpers for
      downstream consumers that re-author dynamics from other sources.
"""

from .physics_overrides import (
    OVERRIDES_FILENAME,
    load_physics_overrides,
    merge_parts_properties,
)
from .urdf_model import ArticulationModel

__all__ = [
    "ArticulationModel",
    "OVERRIDES_FILENAME",
    "load_physics_overrides",
    "merge_parts_properties",
    "interactive_articulation_refinement",
]


def interactive_articulation_refinement(*args, **kwargs):
    """Lazy wrapper so importing this package never requires dash/plotly."""
    from .interactive_ui import interactive_articulation_refinement as _impl

    return _impl(*args, **kwargs)
