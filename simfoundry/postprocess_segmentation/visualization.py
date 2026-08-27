"""Visualization utilities for segment correction UI."""

import plotly.graph_objects as go

from .geometry import SegmentGeometry, compute_exploded_vertices


class SegmentFigureBuilder:
    """Builds Plotly figures for segment visualization."""
    
    def __init__(
        self,
        segment_data: dict,
        part_colors: dict,
        scene_scale: float,
        get_segment_to_part: callable
    ):
        """
        Initialize the figure builder.
        
        Args:
            segment_data: Dict mapping segment IDs to SegmentGeometry
            part_colors: Dict mapping part names to RGB color strings
            scene_scale: Scale factor for consistent explosion
            get_segment_to_part: Callable that returns current segment->part mapping
        """
        self.segment_data = segment_data
        self.part_colors = part_colors
        self.scene_scale = scene_scale
        self.get_segment_to_part = get_segment_to_part
    
    def build(
        self,
        selected_segment: int = None,
        hovered_segment: int = None,
        explosion_factor: float = 0.3,
        selected_faces: list = None,
        face2label: 'np.ndarray' = None,
        mesh: 'trimesh.Trimesh' = None
    ) -> go.Figure:
        """
        Build the Plotly figure with segments colored by part assignment.
        
        Args:
            selected_segment: Currently selected segment ID
            hovered_segment: Currently hovered segment ID
            explosion_factor: How much to explode segments
            selected_faces: List of global face indices to highlight in black
            face2label: Per-face segment labels (needed for selected_faces)
            mesh: The mesh (needed for selected_faces)
            
        Returns:
            Plotly Figure object
        """
        import numpy as np
        
        segment_to_part = self.get_segment_to_part()
        traces = []
        
        # Create a set of selected faces for fast lookup
        selected_faces_set = set(selected_faces) if selected_faces else set()
        
        for seg_id, segment in self.segment_data.items():
            part_name = segment_to_part.get(seg_id, '_unassigned')
            base_color = self.part_colors.get(part_name, self.part_colors['_unassigned'])
            
            # Compute exploded vertices
            seg_vertices = compute_exploded_vertices(
                segment, explosion_factor, self.scene_scale
            )
            
            # Determine opacity based on state
            if seg_id == selected_segment:
                opacity = 1.0
            elif seg_id == hovered_segment:
                opacity = 0.6
            else:
                opacity = 1.0  # Full opacity to avoid see-through artifacts
            
            # Check if any faces in this segment are selected
            # We need to find which faces belong to this segment and if any are in selected_faces
            if selected_faces_set and face2label is not None:
                seg_global_faces = np.where(face2label == seg_id)[0]
                seg_selected = [f for f in seg_global_faces if f in selected_faces_set]
                
                if seg_selected:
                    # Create per-face colors: black for selected, base color for rest
                    # We need a mapping from global face idx to local face idx in segment
                    global_to_local = {gf: lf for lf, gf in enumerate(seg_global_faces)}
                    
                    # Create intensity array for facecolor
                    # Use 0 for selected (black), 1 for normal
                    intensities = np.ones(len(segment.faces))
                    for gf in seg_selected:
                        if gf in global_to_local:
                            intensities[global_to_local[gf]] = 0
                    
                    traces.append(go.Mesh3d(
                        x=seg_vertices[:, 0],
                        y=seg_vertices[:, 1],
                        z=seg_vertices[:, 2],
                        i=segment.faces[:, 0],
                        j=segment.faces[:, 1],
                        k=segment.faces[:, 2],
                        intensity=intensities,
                        # One intensity per FACE: without intensitymode='cell'
                        # plotly interprets the array per-vertex and colors the
                        # wrong geometry. cmin/cmax pinned so an all-selected
                        # segment still maps 0 -> black.
                        intensitymode='cell',
                        cmin=0,
                        cmax=1,
                        colorscale=[[0, 'rgb(20,20,20)'], [1, base_color]],  # Black to base color
                        showscale=False,
                        opacity=opacity,
                        name=f"Seg {seg_id}",
                        customdata=[seg_id],
                        hovertemplate=(
                            f"<b>Segment {seg_id}</b><br>"
                            f"Part: {part_name}<br>"
                            f"Faces: {segment.face_count}<extra></extra>"
                        ),
                        flatshading=True,
                        lighting=dict(
                            ambient=0.4,
                            diffuse=0.8,
                            specular=0.3,
                            roughness=0.5,
                            fresnel=0.2
                        ),
                        lightposition=dict(x=1000, y=1000, z=1000),
                    ))
                    continue
            
            traces.append(go.Mesh3d(
                x=seg_vertices[:, 0],
                y=seg_vertices[:, 1],
                z=seg_vertices[:, 2],
                i=segment.faces[:, 0],
                j=segment.faces[:, 1],
                k=segment.faces[:, 2],
                color=base_color,
                opacity=opacity,
                name=f"Seg {seg_id}",
                customdata=[seg_id],
                hovertemplate=(
                    f"<b>Segment {seg_id}</b><br>"
                    f"Part: {part_name}<br>"
                    f"Faces: {segment.face_count}<extra></extra>"
                ),
                flatshading=False,  # Smooth shading for better detail
                lighting=dict(
                    ambient=0.4,
                    diffuse=0.8,
                    specular=0.3,
                    roughness=0.5,
                    fresnel=0.2
                ),
                lightposition=dict(x=1000, y=1000, z=1000),
            ))
        
        fig = go.Figure(data=traces)
        fig.update_layout(
            scene=dict(
                xaxis=dict(showgrid=True, gridcolor='lightgray', showbackground=True),
                yaxis=dict(showgrid=True, gridcolor='lightgray', showbackground=True),
                zaxis=dict(showgrid=True, gridcolor='lightgray', showbackground=True),
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
            ),
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=600,
            uirevision='constant'
        )
        return fig
    
    def part_view_segments(self, part_name: str = None, current_assignment: dict = None) -> list:
        """Segment IDs shown by build_part_view, in trace order.

        Click handlers map clickData's curveNumber back to a segment through
        this list, so it must iterate exactly like build_part_view does.
        """
        current_assignment = current_assignment or {}
        if part_name and part_name in current_assignment:
            segments_to_show = set(current_assignment[part_name])
        else:
            segments_to_show = set(self.segment_data.keys())
        return [seg_id for seg_id in self.segment_data if seg_id in segments_to_show]

    def build_part_view(
        self,
        part_name: str = None,
        current_assignment: dict = None,
        explosion_factor: float = 0.0,
        selected_segment: int = None,
        selected_segments: list = None
    ) -> go.Figure:
        """
        Build a Plotly figure showing only segments belonging to a specific part.

        Args:
            part_name: Name of the part to display (None shows all parts)
            current_assignment: Dict mapping part_name -> list of segment IDs
            explosion_factor: How much to explode segments apart (0 = no explosion)
            selected_segment: Segment ID to highlight (or None for no highlight)
            selected_segments: Multiple segment IDs to highlight (editing selection)

        Returns:
            Plotly Figure object
        """
        from .styles import EXPLOSION_SCALE

        traces = []

        if current_assignment is None:
            current_assignment = {}

        highlight = set(selected_segments or [])
        if selected_segment is not None:
            highlight.add(selected_segment)

        # Get segments for this part
        if part_name and part_name in current_assignment:
            segments_to_show = set(current_assignment[part_name])
        else:
            # Show all segments
            segments_to_show = set(self.segment_data.keys())

        part_color = self.part_colors.get(part_name, self.part_colors.get('_unassigned', 'rgb(150,150,150)'))

        for seg_id, segment in self.segment_data.items():
            if seg_id not in segments_to_show:
                continue

            # Apply explosion if factor > 0
            if explosion_factor > 0:
                offset = segment.direction * explosion_factor * self.scene_scale * EXPLOSION_SCALE
                seg_vertices = segment.vertices_base + offset
            else:
                seg_vertices = segment.vertices_base

            # Determine color and opacity based on selection
            is_selected = seg_id in highlight
            if is_selected:
                color = 'rgb(255, 215, 0)'  # Gold for selected
                opacity = 1.0
            else:
                color = part_color
                opacity = 0.7 if highlight else 1.0
            
            traces.append(go.Mesh3d(
                x=seg_vertices[:, 0],
                y=seg_vertices[:, 1],
                z=seg_vertices[:, 2],
                i=segment.faces[:, 0],
                j=segment.faces[:, 1],
                k=segment.faces[:, 2],
                color=color,
                opacity=opacity,
                name=f"Seg {seg_id}",
                hovertemplate=(
                    f"<b>Segment {seg_id}</b><br>"
                    f"Part: {part_name}<br>"
                    f"Faces: {segment.face_count}<extra></extra>"
                ),
                flatshading=False,  # Smooth shading for better detail
                lighting=dict(
                    ambient=0.4,
                    diffuse=0.8,
                    specular=0.3,
                    roughness=0.5,
                    fresnel=0.2
                ),
                lightposition=dict(x=1000, y=1000, z=1000),
            ))
        
        fig = go.Figure(data=traces)
        fig.update_layout(
            scene=dict(
                xaxis=dict(showgrid=True, gridcolor='lightgray', showbackground=True),
                yaxis=dict(showgrid=True, gridcolor='lightgray', showbackground=True),
                zaxis=dict(showgrid=True, gridcolor='lightgray', showbackground=True),
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
            ),
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=600,
            uirevision='part-view'
        )
        return fig
