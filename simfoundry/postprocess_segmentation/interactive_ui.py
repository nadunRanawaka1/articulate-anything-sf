"""Interactive Dash-based segment correction UI."""

import json
import threading
import time
import webbrowser
import logging

import numpy as np
import trimesh

from .styles import STYLES, EXPLOSION_SCALE, generate_part_colors
from .geometry import (
    precompute_segment_geometry,
    find_coplanar_faces,
    split_segment_by_faces,
    split_by_connected_components,
    find_faces_by_normal,
    island_containing_face,
    stray_islands,
    merge_segments
)
from .visualization import SegmentFigureBuilder

# Suppress Dash's default HTTP request logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

HELP_STYLES = {
    'details': {
        'backgroundColor': '#fffde7',
        'border': '1px solid #ffe082',
        'borderRadius': '5px',
        'padding': '8px 14px',
        'margin': '0 auto 14px auto',
        'maxWidth': '1150px',
        'fontSize': '13px',
        'color': '#444',
        'textAlign': 'left',
    },
    'summary': {
        'cursor': 'pointer',
        'fontWeight': 'bold',
        'color': '#795548',
    },
    'tab_intro': {
        'fontSize': '12px',
        'color': '#555',
        'backgroundColor': '#f1f8e9',
        'border': '1px solid #dcedc8',
        'borderRadius': '4px',
        'padding': '6px 10px',
        'marginBottom': '10px',
    },
}


class SegmentCorrectionApp:
    """
    Interactive web UI to review and correct VLM's segment-to-part assignments.
    
    Provides a Dash web app with three tabs:
    - Overview: Assign segments to parts
    - Segments: Split and merge segments
    - Parts: View and edit by part
    """
    
    def __init__(
        self,
        mesh: trimesh.Trimesh,
        face2label: np.ndarray,
        label2face_mask: np.ndarray,
        part_segment_dict: dict,
        parts_list: list,
        explosion_factor: float = 0.3,
        verbose: bool = False
    ):
        self.mesh = mesh
        self.face2label = np.array(face2label)
        self.label2face_mask = np.array(label2face_mask)
        self.parts_list = parts_list if '_unassigned' in parts_list else ['_unassigned'] + list(parts_list)
        self.explosion_factor = explosion_factor
        self.verbose = verbose
        
        # Convert numpy types to Python types for JSON serialization
        self.current_assignment = {
            k: [int(x) for x in v] for k, v in part_segment_dict.items()
        }
        self.original_assignment = {
            k: [int(x) for x in v] for k, v in part_segment_dict.items()
        }
        
        # Get all valid segments
        self.all_segments = sorted([int(s) for s in np.unique(face2label) if s >= 0])
        
        # Generate colors
        self.part_colors = generate_part_colors(parts_list)
        
        # Precompute geometry
        self.segment_data, self.scene_scale, self.mesh_centroid = precompute_segment_geometry(
            mesh, face2label, self.all_segments
        )
        
        # Result holder for thread communication
        self.result_holder = {'result': None, 'done': False, 'cancelled': False}

        # Pristine copies returned on Cancel: splits/merges mutate
        # face2label/label2face_mask in place, and returning the mutated
        # arrays with the original assignment would silently drop the
        # split-off faces from their parts downstream.
        self._orig_face2label = self.face2label.copy()
        self._orig_label2face_mask = self.label2face_mask.copy()

        # Last user camera (from any viewer's relayoutData). Injected into
        # every rebuilt figure: plotly's uirevision-based camera preservation
        # breaks after the second orbit/rebuild cycle (its GUI-edit records
        # are consumed by a rebuild and re-recorded against a non-default
        # baseline, so the next rebuild treats the server's hardcoded default
        # camera as an intentional change and snaps back to it).
        self.last_camera = None
        
        # Undo history stack
        self.undo_history = []
        self.max_undo_history = 20
        
        # Create figure builder
        self.figure_builder = SegmentFigureBuilder(
            segment_data=self.segment_data,
            part_colors=self.part_colors,
            scene_scale=self.scene_scale,
            get_segment_to_part=self._get_segment_to_part
        )
        
        # Setup Dash app
        self._setup_app()
    
    def _get_segment_to_part(self) -> dict:
        """Create reverse mapping: segment_id -> part_name."""
        mapping = {}
        for part_name, segments in self.current_assignment.items():
            for seg_id in segments:
                mapping[int(seg_id)] = part_name
        return mapping
    
    def _save_undo_state(self):
        """Save current state to undo history."""
        import copy
        state = {
            'face2label': self.face2label.copy(),
            'label2face_mask': self.label2face_mask.copy(),
            'current_assignment': copy.deepcopy(self.current_assignment),
            'all_segments': list(self.all_segments),
            'segment_data': {k: copy.copy(v) for k, v in self.segment_data.items()},
        }
        self.undo_history.append(state)
        if len(self.undo_history) > self.max_undo_history:
            self.undo_history.pop(0)
    
    def _restore_undo_state(self) -> bool:
        """Restore previous state from undo history."""
        if not self.undo_history:
            return False
        state = self.undo_history.pop()
        self.face2label = state['face2label']
        self.label2face_mask = state['label2face_mask']
        self.current_assignment = state['current_assignment']
        self.all_segments = state['all_segments']
        self.segment_data = state['segment_data']
        return True
    
    def _remember_camera(self, relayout_data):
        """Capture the user's camera from a viewer's relayoutData payload."""
        if relayout_data and 'scene.camera' in relayout_data:
            self.last_camera = relayout_data['scene.camera']

    def _apply_camera(self, fig):
        """Ship the user's current camera with the figure so a rebuild can
        never snap the view back to the hardcoded default."""
        if self.last_camera:
            fig.update_layout(scene_camera=self.last_camera)
        return fig

    def _reassign_segments(self, segments: list, new_part: str) -> list:
        """Move segments to a part in current_assignment; returns (seg, old_part) pairs."""
        moved = []
        for seg_id in segments:
            old_part = None
            for part in self.current_assignment:
                if seg_id in self.current_assignment[part]:
                    old_part = part
                    self.current_assignment[part].remove(seg_id)
                    break
            if new_part not in self.current_assignment:
                self.current_assignment[new_part] = []
            if seg_id not in self.current_assignment[new_part]:
                self.current_assignment[new_part].append(seg_id)
            moved.append((seg_id, old_part))
        return moved

    def _get_assignment_html(self):
        """Generate HTML for current assignments."""
        from dash import html
        html_parts = []
        for part_name in self.parts_list:
            segments = self.current_assignment.get(part_name, [])
            color = self.part_colors.get(part_name, '#888')
            html_parts.append(
                html.Div([
                    html.Span("●", style={'color': color, 'fontSize': '20px', 'marginRight': '8px'}),
                    html.Strong(f"{part_name}: "),
                    html.Span(f"{sorted(segments)}")
                ], style={'marginBottom': '5px'})
            )
        return html_parts
    
    def _help_panel(self):
        """Collapsible walkthrough for first-time users (pure HTML, no callbacks)."""
        from dash import html

        li_style = {'marginBottom': '4px'}
        return html.Details([
            html.Summary("❓  New here? Click for a quick guide",
                         style=HELP_STYLES['summary']),
            html.P([
                html.B("What you're looking at: "),
                "the object's mesh was automatically cut into numbered patches called ",
                html.B("segments"), ". Your job is to make sure every segment belongs to the "
                "correct ", html.B("part"), " — the functional pieces of the object "
                "(base, door, drawer, …). Segments are colored by the part they are "
                "currently assigned to (gray = unassigned). Hover over the 3D view to see a "
                "segment's number, its part, and its size.",
            ], style={'marginBottom': '6px'}),
            html.P([
                html.B("3D view controls: "),
                "left-drag rotates · scroll zooms · right-drag pans. The ",
                html.B("Explosion"), " slider pulls the segments apart so you can see and "
                "click pieces hidden inside (0 = the object's true shape).",
            ], style={'marginBottom': '6px'}),
            html.B("Typical workflow:"),
            html.Ol([
                html.Li([html.B("Overview tab"), " — assign segments to parts: click segments "
                         "in the 3D view (click again to deselect) or pick them from the list, "
                         "choose the target part, press “Reassign All Selected”."],
                        style=li_style),
                html.Li([html.B("Segments tab"), " — only needed when one segment wrongly "
                         "covers two different parts (e.g. a drawer front fused to the cabinet "
                         "body): select the segment and cut it apart with one of the split "
                         "tools, then assign the new piece. Merging is optional — one part may "
                         "own many segments."], style=li_style),
                html.Li([html.B("Parts tab"), " — final check: view each part on its own and "
                         "move any segments that don't belong there."], style=li_style),
            ], style={'marginTop': '4px', 'marginBottom': '6px'}),
            html.P([
                html.B("Finishing (buttons on the Overview tab): "),
                "“Done — Accept Changes” saves everything (segments still unassigned are "
                "automatically given to the base part); “Cancel” discards all changes. "
                "Mistakes are fine — the ", html.B("Undo"), " button on the Segments tab "
                "reverts recent splits, merges, and reassignments from any tab.",
            ], style={'marginBottom': '2px'}),
        ], style=HELP_STYLES['details'])

    def _setup_app(self):
        """Setup the Dash application with three tabs."""
        from dash import Dash, html, dcc, Input, Output, State, callback_context
        from dash.exceptions import PreventUpdate
        
        self.app = Dash(__name__, suppress_callback_exceptions=True)
        
        # Shared references
        segment_data = self.segment_data
        part_colors = self.part_colors
        figure_builder = self.figure_builder
        
        # =====================================================
        # TAB 1: OVERVIEW - Simple segment-to-part assignment
        # =====================================================
        def make_overview_tab():
            return html.Div([
                html.Div([
                    # Left: 3D viewer
                    html.Div([
                        dcc.Graph(
                            id='overview-viewer',
                            figure=self._apply_camera(
                                figure_builder.build(explosion_factor=self.explosion_factor)),
                            style=STYLES['graph'],
                            config={'displayModeBar': True, 'scrollZoom': True}
                        ),
                        html.Div([
                            html.Label("Explosion:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                            dcc.Slider(
                                id='overview-explosion',
                                min=0, max=1.5, step=0.05, value=self.explosion_factor,
                                marks={0: '0', 0.5: '0.5', 1.0: '1.0', 1.5: '1.5'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], style=STYLES['slider_container']),
                    ], style=STYLES['left_panel']),
                
                    # Right: Controls
                    html.Div([
                        html.Div([
                            html.B("Goal: "),
                            "give every segment the right part. Click segments in the 3D "
                            "view (click again to deselect), pick the target part, press "
                            "Reassign. Repeat until each part is one solid color.",
                        ], style=HELP_STYLES['tab_intro']),
                        html.H4("Batch Assign Segments"),
                        html.P("Click segments in the 3D view or pick them from this list:",
                               style={'fontSize': '12px', 'color': '#666'}),
                        dcc.Dropdown(
                            id='overview-segments-dropdown',
                            options=[{'label': f"Segment {s}", 'value': s} for s in self.all_segments],
                            multi=True,
                            placeholder="Select segments or click on 3D view..."
                        ),
                        html.Div(id='overview-selected-display', 
                                 children="",
                                 style=STYLES['selected_display']),
                    
                        html.H4("Assign to Part", style={'marginTop': '20px'}),
                        dcc.Dropdown(
                            id='overview-part-dropdown',
                            options=[{'label': p, 'value': p} for p in self.parts_list],
                            placeholder="Select part..."
                        ),
                        html.Button("Reassign All Selected", id='overview-reassign-btn', n_clicks=0,
                                   title="Move every selected segment to the part chosen above",
                                   style={**STYLES['btn_reassign'], 'marginTop': '10px'}),
                        html.Div(id='overview-status', style={'marginTop': '10px'}),
                    
                        html.Hr(),
                    
                        html.H4("Current Assignments"),
                        html.Div(id='overview-assignments', children=self._get_assignment_html(),
                                 style=STYLES['assignments_container']),
                    
                        html.Hr(),
                    
                        html.Button("✓ Done - Accept Changes", id='done-btn', n_clicks=0,
                                   title="Save the assignment and close (unassigned segments "
                                         "go to the base part automatically)",
                                   style=STYLES['btn_done']),
                        html.Button("✗ Cancel", id='cancel-btn', n_clicks=0,
                                   title="Close and discard every change made in this session",
                                   style=STYLES['btn_cancel']),
                    ], style=STYLES['right_panel']),
                ], style=STYLES['main_layout']),
            
                # Hidden stores for overview tab
                dcc.Store(id='overview-selected-store', data=[]),  # Now stores list of selected segments
                dcc.Store(id='assignment-store', data=json.dumps(self.current_assignment)),
            ])
        
        # =====================================================
        # TAB 2: SEGMENTS - Split and merge operations
        # =====================================================
        def make_segments_tab():
            return html.Div([
                html.Div([
                    # Left: 3D viewer
                    html.Div([
                        dcc.Graph(
                            id='segments-viewer',
                            figure=self._apply_camera(
                                figure_builder.build(explosion_factor=self.explosion_factor)),
                            style=STYLES['graph'],
                            config={'displayModeBar': True, 'scrollZoom': True}
                        ),
                        html.Div([
                            html.Label("Explosion:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                            dcc.Slider(
                                id='segments-explosion',
                                min=0, max=1.5, step=0.05, value=self.explosion_factor,
                                marks={0: '0', 0.5: '0.5', 1.0: '1.0', 1.5: '1.5'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], style=STYLES['slider_container']),
                    ], style=STYLES['left_panel']),
                
                    # Right: Split/Merge controls
                    html.Div([
                        html.Div([
                            html.B("Goal: "),
                            "cut apart segments that wrongly span two different parts. "
                            "1) Select the segment (click it or use the dropdown). "
                            "2) Cut it with one of the tools below — the cut-off faces become "
                            "a new, unassigned (gray) segment. "
                            "3) Give the new segment its part on the Overview tab. "
                            "(Shortcut: after grabbing faces in Island mode, “Split & Assign "
                            "to Part” does steps 2 and 3 in one click.)",
                        ], style=HELP_STYLES['tab_intro']),
                        html.H4("Selected Segment"),
                        dcc.Dropdown(
                            id='segments-dropdown',
                            options=[{'label': f"Segment {s}", 'value': s} for s in self.all_segments],
                            placeholder="Select segment..."
                        ),
                        html.Div(id='segments-selected-display', style={'marginTop': '10px'}),

                        html.Hr(),

                        # Split by Planarity
                        html.H4("Split by Planarity"),
                        html.P("Cuts off one flat surface. Enter the mode, then click a flat "
                               "area (e.g. a drawer front): the faces lying on that plane are "
                               "split into a new segment.",
                               style={'fontSize': '12px', 'color': '#666'}),
                        html.Button("Enter Planar Split Mode", id='planar-split-btn', n_clicks=0,
                                   title="Then click a flat area of the selected segment to cut it off",
                                   style={**STYLES['btn_reassign'], 'backgroundColor': '#9C27B0', 'width': '100%'}),
                        html.Div(id='planar-split-status', style={'marginTop': '5px'}),
                        html.Div([
                            html.Label("Angle Threshold:", style={'fontSize': '12px'}),
                            dcc.Slider(id='angle-slider', min=5, max=45, step=5, value=15,
                                      marks={5: '5°', 15: '15°', 30: '30°', 45: '45°'}),
                            html.P("How far a face may tilt away from the clicked face and "
                                   "still be included (used by both split modes). Grabbing too "
                                   "much? Lower it. Too little? Raise it.",
                                   style={'fontSize': '10px', 'color': '#888', 'marginTop': '2px'}),
                        ], style={'marginTop': '10px'}),
                        dcc.Checklist(
                            id='cross-segment-check',
                            options=[{'label': ' Cross-segment', 'value': 'cross'}],
                            value=[], style={'fontSize': '12px', 'marginTop': '5px'}
                        ),
                        html.P("Cross-segment: take matching faces from other segments too, "
                               "not just the selected one. With Split by Planarity it only "
                               "grows across touching surfaces; with Split by Normal it "
                               "reaches matching faces anywhere on the object.",
                               style={'fontSize': '10px', 'color': '#888', 'marginTop': '2px'}),

                        html.Hr(),

                        # Split by Normal
                        html.H4("Split by Normal"),
                        html.P("Cuts off every face parallel to the one you click — facing "
                               "the same way or the exact opposite way (e.g. both the top and "
                               "the underside of a shelf) — even when they are not on the "
                               "same flat plane.",
                               style={'fontSize': '12px', 'color': '#666'}),
                        html.Button("Enter Normal Split Mode", id='normal-split-btn', n_clicks=0,
                                   title="Then click a face — all parallel faces split off",
                                   style={**STYLES['btn_reassign'], 'backgroundColor': '#3F51B5', 'width': '100%'}),
                        html.Div(id='normal-split-status', style={'marginTop': '5px'}),

                        html.Hr(),

                        # Split Connected Components
                        html.H4("Split Connected Components"),
                        html.P("If the selected segment is made of pieces that don't touch "
                               "each other, this turns each piece into its own segment "
                               "(no clicking in the 3D view needed).",
                               style={'fontSize': '12px', 'color': '#666'}),
                        html.Div([
                            html.Label("Min Size:", style={'fontSize': '12px'}),
                            dcc.Input(id='min-component-slider', type='number',
                                      min=1, max=10000, step=1, value=10,
                                      style={'width': '80px'}),
                            html.Span(" faces — smaller pieces stay with the main segment",
                                      style={'fontSize': '10px', 'color': '#888'}),
                        ]),
                        html.Div([
                            html.Label("Spatial Threshold:", style={'fontSize': '12px'}),
                            dcc.Input(id='spatial-threshold-input', type='number', 
                                      min=0, max=0.1, step=0.0001, value=0.0,
                                      style={'width': '80px', 'marginRight': '10px'}),
                            html.Span("(fraction of mesh diagonal, e.g. 0.001 = 0.1%)", 
                                      style={'fontSize': '10px', 'color': '#888'}),
                            html.P("Connect nearby faces even if not touching (prevents over-split)", 
                                   style={'fontSize': '10px', 'color': '#888', 'marginTop': '5px'}),
                        ]),
                        html.Button("Split Components", id='split-components-btn', n_clicks=0,
                                   title="Break the selected segment into its disconnected pieces",
                                   style={**STYLES['btn_reassign'], 'backgroundColor': '#FF5722', 'width': '100%', 'marginTop': '10px'}),

                        html.Hr(),

                        # Face selection: island grab / stray islands
                        html.H4("Select & Reassign Faces"),
                        html.P("Grab whole disconnected pieces (“islands”) of the selected "
                               "segment — they turn black — then split them off or send them "
                               "straight to another part. Ideal for stray debris far from "
                               "the main body.",
                               style={'fontSize': '12px', 'color': '#666'}),
                        html.Button("Enter Island Mode", id='island-mode-btn', n_clicks=0,
                                   title="Then click any disconnected piece of the selected "
                                         "segment to select all of it (click again to deselect)",
                                   style={**STYLES['btn_reassign'], 'backgroundColor': '#00BCD4', 'width': '100%', 'marginTop': '5px'}),
                        html.P("Island mode: one click selects a whole disconnected piece; "
                               "clicking a selected piece deselects it.",
                               style={'fontSize': '10px', 'color': '#888', 'marginTop': '2px'}),
                        html.Div(id='face-mode-status', style={'marginTop': '5px'}),
                        html.Button("Select All Stray Islands", id='stray-islands-btn', n_clicks=0,
                                   title="One click: select every piece except the segment's largest one",
                                   style={**STYLES['btn_reassign'], 'backgroundColor': '#8BC34A', 'width': '100%', 'marginTop': '5px'}),
                        html.Div([
                            html.Label("Max island size:", style={'fontSize': '12px'}),
                            dcc.Input(id='stray-max-input', type='number', min=1, step=1,
                                      placeholder='no limit',
                                      style={'width': '80px', 'marginLeft': '5px', 'marginRight': '5px'}),
                            html.Span("faces (blank = everything but the largest island)",
                                      style={'fontSize': '10px', 'color': '#888'}),
                        ], style={'marginTop': '5px'}),
                        html.Div(id='face-select-display', style={'marginTop': '8px'}),
                        html.Button("Clear Face Selection", id='clear-faces-btn', n_clicks=0,
                                   title="Deselect all selected (black) faces",
                                   style={**STYLES['btn_reassign'], 'backgroundColor': '#9E9E9E', 'width': '100%', 'marginTop': '5px'}),
                        html.Button("Split Selection → New Segment", id='split-selection-btn', n_clicks=0,
                                   title="Turn the selected (black) faces into a new, unassigned segment",
                                   style={**STYLES['btn_reassign'], 'backgroundColor': '#FF9800', 'width': '100%', 'marginTop': '5px'}),
                        dcc.Dropdown(
                            id='face-part-dropdown',
                            options=[{'label': p, 'value': p} for p in self.parts_list],
                            placeholder="Part to receive the faces...",
                        ),
                        html.Button("Split & Assign to Part", id='split-assign-btn', n_clicks=0,
                                   title="Split the selected (black) faces off and assign them "
                                         "to the part chosen above, in one step",
                                   style={**STYLES['btn_reassign'], 'width': '100%', 'marginTop': '5px'}),

                        html.Hr(),

                        # Merge Segments
                        html.H4("Merge Segments"),
                        html.P("Combine 2+ segments into one. Rarely required: a part may own "
                               "many segments, so merging is for tidiness only.",
                               style={'fontSize': '12px', 'color': '#666'}),
                        dcc.Dropdown(
                            id='merge-dropdown',
                            options=[{'label': f"Segment {s}", 'value': s} for s in self.all_segments],
                            multi=True,
                            placeholder="Select 2+ segments to merge..."
                        ),
                        html.Button("Merge", id='merge-btn', n_clicks=0,
                                   style={**STYLES['btn_reassign'], 'backgroundColor': '#795548', 'width': '100%', 'marginTop': '10px'}),
                    
                        html.Hr(),
                    
                        # Undo
                        html.Button("↶ Undo", id='undo-btn', n_clicks=0,
                                   title="Revert the last split / merge / reassignment "
                                         "(from any tab, up to 20 steps)",
                                   style={**STYLES['btn_reassign'], 'backgroundColor': '#607D8B', 'width': '100%'}),
                        html.Div(id='segments-status', style={'marginTop': '10px'}),
                    ], style=STYLES['right_panel']),
                ], style=STYLES['main_layout']),
            
                # Hidden stores
                dcc.Store(id='segments-selected-store', data=None),
                dcc.Store(id='planar-mode-store', data=False),
                dcc.Store(id='normal-mode-store', data=False),
                dcc.Store(id='island-mode-store', data=False),
                dcc.Store(id='face-select-store', data=[]),
            ])
        
        # =====================================================
        # TAB 3: PARTS - View by part
        # =====================================================
        def make_parts_tab():
            return html.Div([
                html.Div([
                    # Left: 3D viewer
                    html.Div([
                        dcc.Graph(
                            id='parts-viewer',
                            figure=self._apply_camera(
                                figure_builder.build(explosion_factor=self.explosion_factor)),
                            style=STYLES['graph'],
                            config={'displayModeBar': True, 'scrollZoom': True}
                        ),
                        html.Div([
                            html.Label("Explosion:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                            dcc.Slider(
                                id='parts-explosion',
                                min=0, max=1.5, step=0.05, value=self.explosion_factor,
                                marks={0: '0', 0.5: '0.5', 1.0: '1.0', 1.5: '1.5'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], style=STYLES['slider_container']),
                    ], style=STYLES['left_panel']),
                
                    # Right: Part controls
                    html.Div([
                        html.Div([
                            html.B("Goal: "),
                            "final check. Pick a part to see only its segments (the rest of "
                            "the object is hidden) and make sure nothing is missing or extra; "
                            "move any strays to their correct part.",
                        ], style=HELP_STYLES['tab_intro']),
                        html.H4("Select Part to View"),
                        dcc.Dropdown(
                            id='parts-dropdown',
                            options=[{'label': p, 'value': p} for p in self.parts_list],
                            placeholder="Select part..."
                        ),
                        html.Div(id='parts-info', style={'marginTop': '10px'}),

                        html.Hr(),

                        html.H4("Edit Segments of This Part"),
                        html.P("Click segments in the 3D view to select them (gold), "
                               "then move them to another part.",
                               style={'fontSize': '12px', 'color': '#666'}),
                        html.Div(id='parts-selected-display',
                                 children="No segments selected",
                                 style=STYLES['selected_display']),
                        dcc.Dropdown(
                            id='parts-move-dropdown',
                            options=[{'label': p, 'value': p} for p in self.parts_list],
                            placeholder="Move selected to part...",
                        ),
                        html.Button("Move Selected Segments", id='parts-move-btn', n_clicks=0,
                                   title="Move the gold segments to the part chosen above",
                                   style={**STYLES['btn_reassign'], 'marginTop': '10px'}),
                        html.Button("Clear Selection", id='parts-clear-btn', n_clicks=0,
                                   title="Deselect all gold segments",
                                   style={**STYLES['btn_reassign'], 'backgroundColor': '#9E9E9E'}),
                        html.Div(id='parts-status', style={'marginTop': '10px'}),

                        html.Hr(),

                        html.H4("Current Assignments"),
                        html.Div(id='parts-assignments', children=self._get_assignment_html(),
                                 style=STYLES['assignments_container']),
                    ], style=STYLES['right_panel']),
                ], style=STYLES['main_layout']),

                # Hidden stores for parts tab
                dcc.Store(id='parts-selected-store', data=[]),
            ])
        
        # =====================================================
        # MAIN LAYOUT WITH TABS
        # =====================================================
        self.app.layout = html.Div([
            html.H2("Segment-to-Part Assignment Editor",
                    style={'textAlign': 'center', 'marginBottom': '10px'}),

            self._help_panel(),

            dcc.Tabs(id='main-tabs', value='overview', children=[
                dcc.Tab(label='Overview (Assign)', value='overview'),
                dcc.Tab(label='Segments (Split/Merge)', value='segments'),
                dcc.Tab(label='Parts (View & Edit)', value='parts'),
            ]),
            
            html.Div(id='tab-content'),
            
            # Global stores
            dcc.Store(id='global-assignment', data=json.dumps(self.current_assignment)),
        ], style=STYLES['container'])
        
        # =====================================================
        # CALLBACKS
        # =====================================================
        app = self.app
        
        # Tab content switching
        @app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'value')]
        )
        def render_tab(tab):
            # Layouts are built per mount so dropdown options, assignment
            # displays, and figures always reflect the current segment set.
            if tab == 'overview':
                return make_overview_tab()
            elif tab == 'segments':
                return make_segments_tab()
            elif tab == 'parts':
                return make_parts_tab()
            return html.Div("Unknown tab")
        
        # =====================================================
        # OVERVIEW TAB CALLBACKS
        # =====================================================
        
        # Click to add segment to selection (or use dropdown for multi-select)
        @app.callback(
            [Output('overview-segments-dropdown', 'value'),
             Output('overview-selected-display', 'children')],
            [Input('overview-viewer', 'clickData'),
             Input('overview-segments-dropdown', 'value')],
            [State('overview-segments-dropdown', 'value')]
        )
        def overview_select(click_data, dropdown_value, current_selection):
            from dash import callback_context
            ctx = callback_context
            if not ctx.triggered:
                raise PreventUpdate
            
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]
            
            # Ensure current_selection is a list
            if current_selection is None:
                current_selection = []
            
            if trigger == 'overview-viewer' and click_data:
                try:
                    curve = click_data['points'][0]['curveNumber']
                    seg_id = list(segment_data.keys())[curve]
                    # Toggle: add if not present, remove if present
                    if seg_id in current_selection:
                        current_selection = [s for s in current_selection if s != seg_id]
                    else:
                        current_selection = list(current_selection) + [seg_id]
                except:
                    raise PreventUpdate
            elif trigger == 'overview-segments-dropdown':
                current_selection = dropdown_value or []
            
            # Generate display
            if not current_selection:
                display = html.Span("No segments selected", style={'color': '#888'})
            else:
                segment_to_part = self._get_segment_to_part()
                items = []
                for seg_id in sorted(current_selection):
                    part = segment_to_part.get(seg_id, '_unassigned')
                    items.append(html.Div([
                        html.Span(f"Segment {seg_id}: ", style={'fontWeight': 'bold'}),
                        html.Span(part, style={'color': part_colors.get(part, '#888')})
                    ]))
                display = html.Div(items)
            
            return current_selection, display
        
        # Update viewer on explosion change
        @app.callback(
            Output('overview-viewer', 'figure'),
            [Input('overview-explosion', 'value'),
             Input('global-assignment', 'data')],
            [State('overview-viewer', 'relayoutData')]
        )
        def update_overview_viewer(explosion, _, relayout_data):
            self.current_assignment = json.loads(_) if _ else self.current_assignment
            self._remember_camera(relayout_data)
            return self._apply_camera(figure_builder.build(explosion_factor=explosion))
        
        # Reassign multiple segments
        @app.callback(
            [Output('global-assignment', 'data'),
             Output('overview-assignments', 'children'),
             Output('overview-status', 'children'),
             Output('overview-segments-dropdown', 'value', allow_duplicate=True)],
            [Input('overview-reassign-btn', 'n_clicks')],
            [State('overview-segments-dropdown', 'value'),
             State('overview-part-dropdown', 'value'),
             State('global-assignment', 'data')],
            prevent_initial_call=True
        )
        def overview_reassign(n_clicks, selected_segments, new_part, assignment_json):
            if n_clicks == 0 or not selected_segments or new_part is None:
                raise PreventUpdate
            
            self._save_undo_state()
            self.current_assignment = json.loads(assignment_json)
            
            print(f"\nDEBUG: Reassigning segments {selected_segments} to '{new_part}'")
            
            # Process each selected segment
            for seg_id in selected_segments:
                # Find and remove from old part
                old_part = None
                for part in self.current_assignment:
                    if seg_id in self.current_assignment[part]:
                        old_part = part
                        self.current_assignment[part].remove(seg_id)
                        break
                
                # Add to new part
                if new_part not in self.current_assignment:
                    self.current_assignment[new_part] = []
                if seg_id not in self.current_assignment[new_part]:
                    self.current_assignment[new_part].append(seg_id)
                
                print(f"  Segment {seg_id}: {old_part} -> {new_part}")
            
            count = len(selected_segments)
            return (json.dumps(self.current_assignment), 
                    self._get_assignment_html(),
                    html.Span(f"✓ Reassigned {count} segment(s) to {new_part}", style={'color': 'green'}),
                    [])  # Clear selection after reassign
        
        # Done button
        @app.callback(
            Output('global-assignment', 'data', allow_duplicate=True),
            [Input('done-btn', 'n_clicks')],
            [State('global-assignment', 'data')],
            prevent_initial_call=True
        )
        def handle_done(n_clicks, assignment_json):
            if n_clicks == 0:
                raise PreventUpdate
            
            assignment = json.loads(assignment_json)
            
            # DEBUG: Print assignment from store
            print("\n" + "="*60)
            print("DEBUG: handle_done called")
            print("="*60)
            print("Assignment from global-assignment store:")
            for part, segs in sorted(assignment.items()):
                print(f"  {part}: {sorted(segs)}")
            print(f"\nall_segments tracked: {sorted(self.all_segments)}")
            print("="*60)
            
            # Find all segments assigned to real parts (exclude '_unassigned')
            assigned_segments = set()
            for part, segments in assignment.items():
                if part != '_unassigned':
                    assigned_segments.update(segments)
            
            # Find unassigned segments (not assigned to any real part OR in '_unassigned')
            unassigned_from_tracking = [s for s in self.all_segments if s not in assigned_segments]
            unassigned_explicit = assignment.get('_unassigned', [])
            all_unassigned = list(set(unassigned_from_tracking + unassigned_explicit))
            
            if all_unassigned:
                # Find the base part (usually "base" or contains "base" in name)
                base_part = None
                for part_name in assignment.keys():
                    if part_name != '_unassigned' and 'base' in part_name.lower():
                        base_part = part_name
                        break
                
                # If no "base" found, use the first non-_unassigned part
                if base_part is None:
                    real_parts = [p for p in assignment.keys() if p != '_unassigned']
                    base_part = real_parts[0] if real_parts else "base"
                    if base_part not in assignment:
                        assignment[base_part] = []
                
                # Assign all unassigned segments to base
                for seg in all_unassigned:
                    if seg not in assignment[base_part]:
                        assignment[base_part].append(seg)
                
                print(f"Auto-assigned {len(all_unassigned)} unassigned segments to '{base_part}': {sorted(all_unassigned)}")
            
            # Remove '_unassigned' from final assignment
            if '_unassigned' in assignment:
                del assignment['_unassigned']
            
            # DEBUG: Print final assignment
            print("\nFinal assignment being returned:")
            for part, segs in sorted(assignment.items()):
                print(f"  {part}: {sorted(segs)} ({len(segs)} segments)")
            print("="*60 + "\n")
            
            self.result_holder['result'] = assignment
            self.result_holder['done'] = True
            return json.dumps(assignment)
        
        # Cancel button
        @app.callback(
            Output('global-assignment', 'data', allow_duplicate=True),
            [Input('cancel-btn', 'n_clicks')],
            prevent_initial_call=True
        )
        def handle_cancel(n_clicks):
            if n_clicks == 0:
                raise PreventUpdate
            self.result_holder['result'] = self.original_assignment
            self.result_holder['cancelled'] = True
            self.result_holder['done'] = True
            raise PreventUpdate
        
        # =====================================================
        # SEGMENTS TAB CALLBACKS
        # =====================================================
        
        # Click or dropdown to select segment
        @app.callback(
            [Output('segments-selected-store', 'data'),
             Output('segments-selected-display', 'children'),
             Output('face-select-store', 'data', allow_duplicate=True),
             Output('face-select-display', 'children', allow_duplicate=True)],
            [Input('segments-viewer', 'clickData'),
             Input('segments-dropdown', 'value')],
            [State('segments-selected-store', 'data'),
             State('planar-mode-store', 'data'),
             State('normal-mode-store', 'data'),
             State('island-mode-store', 'data')],
            prevent_initial_call=True
        )
        def segments_select(click_data, dropdown_val, current, planar_mode, normal_mode,
                            island_mode):
            from dash import no_update
            ctx = callback_context
            if not ctx.triggered:
                raise PreventUpdate

            trigger = ctx.triggered[0]['prop_id'].split('.')[0]

            # If in a click-tool mode, don't change selection on click
            if trigger == 'segments-viewer' and (planar_mode or normal_mode or island_mode):
                raise PreventUpdate

            seg_id = None
            if trigger == 'segments-dropdown' and dropdown_val is not None:
                seg_id = dropdown_val
            elif trigger == 'segments-viewer' and click_data:
                try:
                    curve = click_data['points'][0]['curveNumber']
                    seg_id = list(segment_data.keys())[curve]
                except:
                    raise PreventUpdate

            if seg_id is None:
                raise PreventUpdate

            seg = segment_data.get(seg_id)
            face_count = seg.face_count if seg else 0
            display = html.Div([
                html.Strong(f"Segment {seg_id}"),
                html.Br(),
                html.Span(f"Faces: {face_count}")
            ])
            # Changing segment invalidates any face selection (it is scoped to
            # one segment); keep it when re-selecting the same segment.
            if seg_id == current:
                return seg_id, display, no_update, no_update
            return seg_id, display, [], ""

        # Update viewer on explosion change / assignment change / face selection
        @app.callback(
            Output('segments-viewer', 'figure'),
            [Input('segments-explosion', 'value'),
             Input('global-assignment', 'data'),
             Input('face-select-store', 'data')],
            [State('segments-viewer', 'relayoutData')]
        )
        def update_segments_viewer(explosion, _, selected_faces, relayout_data):
            if _:
                self.current_assignment = json.loads(_)
            self._remember_camera(relayout_data)
            return self._apply_camera(figure_builder.build(
                explosion_factor=explosion,
                selected_faces=selected_faces or [],
                face2label=self.face2label,
                mesh=self.mesh,
            ))
        
        # The three click-tool modes (planar/normal split, island) are
        # mutually exclusive: entering one disarms the others, INCLUDING their
        # buttons — a stale "Exit X Mode" button on a disarmed mode would
        # otherwise re-arm that mode when clicked.
        MODES = {
            'planar-mode-store': ('planar-split-btn', 'planar-split-status',
                                  "Enter Planar Split Mode", "Exit Planar Split Mode",
                                  '#9C27B0', "🔪 Click on segment {selected}"),
            'normal-mode-store': ('normal-split-btn', 'normal-split-status',
                                  "Enter Normal Split Mode", "Exit Normal Split Mode",
                                  '#3F51B5', "🎯 Click on segment {selected}"),
            'island-mode-store': ('island-mode-btn', 'face-mode-status',
                                  "Enter Island Mode", "Exit Island Mode",
                                  '#00BCD4', "🏝 Click a disconnected piece of segment {selected}"),
        }

        def _mode_base_style(color):
            return {**STYLES['btn_reassign'], 'backgroundColor': color, 'width': '100%'}

        def _make_mode_toggle(mode_id):
            btn_id, status_id, enter_label, exit_label, base_color, active_hint = MODES[mode_id]
            others = [m for m in MODES if m != mode_id]
            outputs = [Output(mode_id, 'data'),
                       Output(btn_id, 'children'),
                       Output(btn_id, 'style')]
            for other in others:
                other_btn = MODES[other][0]
                outputs += [Output(other, 'data', allow_duplicate=True),
                            Output(other_btn, 'children', allow_duplicate=True),
                            Output(other_btn, 'style', allow_duplicate=True)]
            outputs.append(Output(status_id, 'children', allow_duplicate=True))

            @app.callback(
                outputs,
                [Input(btn_id, 'n_clicks')],
                [State(mode_id, 'data'),
                 State('segments-selected-store', 'data')],
                prevent_initial_call=True
            )
            def toggle(n, current, selected):
                if not n:
                    raise PreventUpdate
                base_style = _mode_base_style(base_color)
                active_style = {**STYLES['btn_reassign'], 'backgroundColor': '#E91E63', 'width': '100%'}
                # Reset every other mode: store off, button back to its enter state.
                others_reset = []
                for other in others:
                    other_btn_id, _, other_enter, _, other_color, _ = MODES[other]
                    others_reset += [False, other_enter, _mode_base_style(other_color)]
                new_mode = not current
                if new_mode:
                    if selected is None:
                        return tuple([False, enter_label, base_style] + others_reset +
                                     [html.Span("⚠ Select a segment first", style={'color': 'orange'})])
                    return tuple([True, exit_label, active_style] + others_reset +
                                 [html.Span(active_hint.format(selected=selected),
                                            style={'color': base_color, 'fontWeight': 'bold'})])
                return tuple([False, enter_label, base_style] + others_reset + [""])
            return toggle

        for _mode_id in MODES:
            _make_mode_toggle(_mode_id)
        
        # Handle split click (planar or normal)
        @app.callback(
            [Output('global-assignment', 'data', allow_duplicate=True),
             Output('segments-status', 'children'),
             Output('segments-dropdown', 'options'),
             Output('merge-dropdown', 'options'),
             Output('face-select-store', 'data', allow_duplicate=True),
             Output('face-select-display', 'children', allow_duplicate=True)],
            [Input('segments-viewer', 'clickData')],
            [State('planar-mode-store', 'data'),
             State('normal-mode-store', 'data'),
             State('segments-selected-store', 'data'),
             State('global-assignment', 'data'),
             State('angle-slider', 'value'),
             State('cross-segment-check', 'value'),
             State('segments-explosion', 'value')],
            prevent_initial_call=True
        )
        def handle_split_click(click_data, planar_mode, normal_mode, selected, assignment_json,
                              angle, cross_val, explosion):
            if not (planar_mode or normal_mode) or selected is None or click_data is None:
                raise PreventUpdate
            
            def get_opts():
                return [{'label': f"Segment {s}", 'value': s} for s in self.all_segments]
            
            try:
                point = click_data['points'][0]
                curve = point['curveNumber']
                clicked_seg = list(segment_data.keys())[curve]
                
                from dash import no_update
                if clicked_seg != selected:
                    return assignment_json, html.Span(f"⚠ Click on segment {selected}", style={'color': 'orange'}), \
                           get_opts(), get_opts(), no_update, no_update
                
                # Find clicked face
                clicked_pos = np.array([point['x'], point['y'], point['z']])
                seg_mask = self.face2label == selected
                seg_faces = np.where(seg_mask)[0]
                
                vertices = np.array(self.mesh.vertices)
                faces = np.array(self.mesh.faces)
                centroids = vertices[faces[seg_faces]].mean(axis=1)
                
                seg = segment_data.get(selected)
                if seg:
                    # Use the slider's CURRENT explosion, not the initial one,
                    # or clicks in exploded view resolve to the wrong face.
                    factor = explosion if explosion is not None else self.explosion_factor
                    offset = seg.direction * factor * self.scene_scale * EXPLOSION_SCALE
                    centroids = centroids + offset
                
                dists = np.linalg.norm(centroids - clicked_pos, axis=1)
                clicked_face = seg_faces[np.argmin(dists)]
                
                cross = 'cross' in (cross_val or [])
                
                # Find faces to split
                if normal_mode:
                    faces_to_split = find_faces_by_normal(
                        self.mesh, self.face2label, selected, clicked_face,
                        angle_threshold_deg=float(angle), cross_segment=cross
                    )
                    split_type = "normal"
                else:
                    faces_to_split = find_coplanar_faces(
                        self.mesh, self.face2label, selected, clicked_face,
                        angle_threshold_deg=float(angle), distance_threshold_ratio=0.02,
                        cross_segment=cross
                    )
                    split_type = "planar"
                
                if len(faces_to_split) == 0:
                    return assignment_json, html.Span("No faces found", style={'color': 'red'}), \
                           get_opts(), get_opts(), no_update, no_update

                self._save_undo_state()

                # Create new segment. Never reuse a merged-away id: the id must
                # equal its label2face_mask row (see split_segment_by_faces).
                new_id = int(max(np.max(self.face2label) + 1, self.label2face_mask.shape[0]))
                if cross:
                    affected = set(int(self.face2label[f]) for f in faces_to_split)
                    for f in faces_to_split:
                        self.face2label[f] = new_id
                    if new_id >= self.label2face_mask.shape[0]:
                        new_mask = np.zeros((new_id + 1, self.label2face_mask.shape[1]), dtype=bool)
                        new_mask[:self.label2face_mask.shape[0], :] = self.label2face_mask
                        self.label2face_mask = new_mask
                    self.label2face_mask[new_id, faces_to_split] = True
                    for s in affected:
                        self.label2face_mask[s, faces_to_split] = False
                else:
                    self.face2label, self.label2face_mask, new_id = split_segment_by_faces(
                        self.face2label, self.label2face_mask, selected, faces_to_split
                    )
                
                # Update state
                self.all_segments = sorted([int(s) for s in np.unique(self.face2label) if s >= 0])
                self.segment_data, self.scene_scale, _ = precompute_segment_geometry(
                    self.mesh, self.face2label, self.all_segments
                )
                segment_data.clear()
                segment_data.update(self.segment_data)
                figure_builder.segment_data = self.segment_data
                figure_builder.scene_scale = self.scene_scale
                
                # Update assignment
                self.current_assignment = json.loads(assignment_json)
                # A split (especially cross-segment) may have emptied segments
                # entirely; drop their dead ids so the final assignment never
                # references segments without faces.
                live = set(self.all_segments)
                for part in self.current_assignment:
                    self.current_assignment[part] = [
                        s for s in self.current_assignment[part] if s in live
                    ]
                if '_unassigned' not in self.current_assignment:
                    self.current_assignment['_unassigned'] = []
                self.current_assignment['_unassigned'].append(new_id)

                return (json.dumps(self.current_assignment),
                       html.Span(f"✓ Split {len(faces_to_split)} faces ({split_type}) -> segment {new_id}",
                                style={'color': 'green'}),
                       get_opts(), get_opts(), [], "")
            except Exception as e:
                from dash import no_update as _nu
                return assignment_json, html.Span(f"Error: {e}", style={'color': 'red'}), \
                       get_opts(), get_opts(), _nu, _nu
        
        # Split by connected components
        @app.callback(
            [Output('global-assignment', 'data', allow_duplicate=True),
             Output('segments-status', 'children', allow_duplicate=True),
             Output('segments-dropdown', 'options', allow_duplicate=True),
             Output('merge-dropdown', 'options', allow_duplicate=True),
             Output('face-select-store', 'data', allow_duplicate=True),
             Output('face-select-display', 'children', allow_duplicate=True)],
            [Input('split-components-btn', 'n_clicks')],
            [State('segments-selected-store', 'data'),
             State('global-assignment', 'data'),
             State('min-component-slider', 'value'),
             State('spatial-threshold-input', 'value')],
            prevent_initial_call=True
        )
        def split_components(n, selected, assignment_json, min_size, spatial_threshold):
            from dash import no_update
            if not n or selected is None:
                raise PreventUpdate
            
            def get_opts():
                return [{'label': f"Segment {s}", 'value': s} for s in self.all_segments]
            
            # Handle None or empty inputs
            if min_size is None or min_size == '':
                min_size = 1
            if spatial_threshold is None or spatial_threshold == '':
                spatial_threshold = 0.0
            
            self._save_undo_state()
            self.face2label, self.label2face_mask, new_ids = split_by_connected_components(
                self.mesh, self.face2label, self.label2face_mask, selected, 
                min_component_size=int(min_size),
                spatial_threshold=float(spatial_threshold)
            )
            
            if not new_ids:
                # Nothing changed: discard the undo frame saved above so a
                # later Undo doesn't become a silent no-op.
                if self.undo_history:
                    self.undo_history.pop()
                return no_update, html.Span("Only 1 component", style={'color': 'orange'}), \
                       get_opts(), get_opts(), no_update, no_update

            # Update state
            self.all_segments = sorted([int(s) for s in np.unique(self.face2label) if s >= 0])
            self.segment_data, self.scene_scale, _ = precompute_segment_geometry(
                self.mesh, self.face2label, self.all_segments
            )
            segment_data.clear()
            segment_data.update(self.segment_data)
            figure_builder.segment_data = self.segment_data
            figure_builder.scene_scale = self.scene_scale

            self.current_assignment = json.loads(assignment_json)
            if '_unassigned' not in self.current_assignment:
                self.current_assignment['_unassigned'] = []
            for nid in new_ids:
                self.current_assignment['_unassigned'].append(nid)

            return (json.dumps(self.current_assignment),
                   html.Span(f"✓ Split into {len(new_ids)+1} components", style={'color': 'green'}),
                   get_opts(), get_opts(), [], "")

        # =====================================================
        # FACE SELECTION (island / stray) CALLBACKS
        # =====================================================

        def refresh_after_label_change():
            """Recompute per-segment geometry after face2label changed and
            propagate it to the shared dict the callbacks captured."""
            self.all_segments = sorted([int(s) for s in np.unique(self.face2label) if s >= 0])
            self.segment_data, self.scene_scale, _ = precompute_segment_geometry(
                self.mesh, self.face2label, self.all_segments
            )
            segment_data.clear()
            segment_data.update(self.segment_data)
            figure_builder.segment_data = self.segment_data
            figure_builder.scene_scale = self.scene_scale

        def face_selection_display(faces, seg_id):
            if not faces:
                return html.Span("No faces selected", style={'color': '#888'})
            return html.Span(f"{len(faces)} face(s) selected in segment {seg_id}",
                             style={'color': '#009688', 'fontWeight': 'bold'})

        def unexplode_click(point, explosion):
            """Clicked world position mapped back to un-exploded mesh coords
            (subtract the CLICKED segment's explosion offset)."""
            clicked_pos = np.array([point['x'], point['y'], point['z']])
            try:
                clicked_seg = list(segment_data.keys())[point['curveNumber']]
            except (KeyError, IndexError):
                return None, None
            factor = explosion if explosion is not None else self.explosion_factor
            seg_geo = segment_data.get(clicked_seg)
            if seg_geo is not None:
                clicked_pos = clicked_pos - seg_geo.direction * factor * self.scene_scale * EXPLOSION_SCALE
            return clicked_pos, clicked_seg

        # Island clicks accumulate a face selection within the selected
        # segment (rendered black in the viewer).
        @app.callback(
            [Output('face-select-store', 'data'),
             Output('face-select-display', 'children'),
             Output('face-mode-status', 'children', allow_duplicate=True)],
            [Input('segments-viewer', 'clickData')],
            [State('island-mode-store', 'data'),
             State('segments-selected-store', 'data'),
             State('face-select-store', 'data'),
             State('segments-explosion', 'value')],
            prevent_initial_call=True
        )
        def handle_face_click(click_data, island_mode, selected, selection, explosion):
            if not island_mode or click_data is None:
                raise PreventUpdate
            if selected is None:
                return [], face_selection_display([], None), \
                       html.Span("⚠ Select a segment first", style={'color': 'orange'})
            point = click_data['points'][0]
            if not all(k in point for k in ('x', 'y', 'z')):
                raise PreventUpdate

            base_pos, clicked_seg = unexplode_click(point, explosion)
            if base_pos is None:
                raise PreventUpdate

            selection = set(int(f) for f in (selection or []))
            seg_faces = np.where(self.face2label == selected)[0]
            if len(seg_faces) == 0:
                raise PreventUpdate

            if clicked_seg != selected:
                return (sorted(selection),
                        face_selection_display(sorted(selection), selected),
                        html.Span(f"⚠ Click on segment {selected} (clicked {clicked_seg})",
                                  style={'color': 'orange'}))
            vertices = np.array(self.mesh.vertices)
            faces_arr = np.array(self.mesh.faces)
            centroids = vertices[faces_arr[seg_faces]].mean(axis=1)
            seed = int(seg_faces[np.argmin(np.linalg.norm(centroids - base_pos, axis=1))])
            island = set(int(f) for f in island_containing_face(
                self.mesh, self.face2label, selected, seed))
            if island and island <= selection:
                selection -= island  # clicking a selected island deselects it
                msg = html.Span(f"Deselected island ({len(island)} faces)",
                                style={'color': '#00BCD4'})
            else:
                selection |= island
                msg = html.Span(f"🏝 Selected island ({len(island)} faces)",
                                style={'color': 'green'})

            sel_sorted = sorted(selection)
            return sel_sorted, face_selection_display(sel_sorted, selected), msg

        # One-shot: select every island of the segment except the largest.
        @app.callback(
            [Output('face-select-store', 'data', allow_duplicate=True),
             Output('face-select-display', 'children', allow_duplicate=True),
             Output('face-mode-status', 'children', allow_duplicate=True)],
            [Input('stray-islands-btn', 'n_clicks')],
            [State('segments-selected-store', 'data'),
             State('stray-max-input', 'value'),
             State('face-select-store', 'data')],
            prevent_initial_call=True
        )
        def select_stray_islands(n, selected, max_size, selection):
            if n == 0:
                raise PreventUpdate
            if selected is None:
                return [], face_selection_display([], None), \
                       html.Span("⚠ Select a segment first", style={'color': 'orange'})
            max_faces = int(max_size) if max_size else None
            stray = stray_islands(self.mesh, self.face2label, selected, max_faces=max_faces)
            current = set(int(f) for f in (selection or []))
            if len(stray) == 0:
                return (sorted(current), face_selection_display(sorted(current), selected),
                        html.Span("No stray islands found", style={'color': 'orange'}))
            merged = sorted(current | set(int(f) for f in stray))
            return (merged, face_selection_display(merged, selected),
                    html.Span(f"✓ Selected {len(stray)} faces in stray islands",
                              style={'color': 'green'}))

        @app.callback(
            [Output('face-select-store', 'data', allow_duplicate=True),
             Output('face-select-display', 'children', allow_duplicate=True)],
            [Input('clear-faces-btn', 'n_clicks')],
            prevent_initial_call=True
        )
        def clear_face_selection(n):
            if n == 0:
                raise PreventUpdate
            return [], face_selection_display([], None)

        # Split the face selection off — into a new (unassigned) segment, or
        # straight into a chosen part.
        @app.callback(
            [Output('global-assignment', 'data', allow_duplicate=True),
             Output('segments-status', 'children', allow_duplicate=True),
             Output('segments-dropdown', 'options', allow_duplicate=True),
             Output('merge-dropdown', 'options', allow_duplicate=True),
             Output('face-select-store', 'data', allow_duplicate=True),
             Output('face-select-display', 'children', allow_duplicate=True)],
            [Input('split-selection-btn', 'n_clicks'),
             Input('split-assign-btn', 'n_clicks')],
            [State('face-select-store', 'data'),
             State('segments-selected-store', 'data'),
             State('face-part-dropdown', 'value'),
             State('global-assignment', 'data')],
            prevent_initial_call=True
        )
        def split_face_selection(n_split, n_assign, selection, selected, target_part, assignment_json):
            trigger = callback_context.triggered[0]['prop_id'].split('.')[0]
            if (trigger == 'split-selection-btn' and not n_split) or \
               (trigger == 'split-assign-btn' and not n_assign):
                raise PreventUpdate

            def get_opts():
                return [{'label': f"Segment {s}", 'value': s} for s in self.all_segments]

            def keep(color, msg):
                return (assignment_json, html.Span(msg, style={'color': color}),
                        get_opts(), get_opts(), selection or [],
                        face_selection_display(sorted(selection or []), selected))

            if selected is None:
                return keep('orange', "⚠ Select a segment first")
            faces = np.array(sorted(set(int(f) for f in (selection or []))), dtype=int)
            if len(faces):
                # Defensive: only faces still belonging to the selected segment
                faces = faces[self.face2label[faces] == selected]
            if len(faces) == 0:
                return keep('orange', "No faces selected — grab islands first")

            assign_to_part = trigger == 'split-assign-btn'
            if assign_to_part and not target_part:
                return keep('orange', "Choose a part to receive the faces")

            self.current_assignment = json.loads(assignment_json)
            seg_total = int(np.sum(self.face2label == selected))
            if len(faces) == seg_total:
                # The whole segment is selected: no split needed.
                if not assign_to_part:
                    return keep('orange', "Selection covers the whole segment — nothing to split")
                self._save_undo_state()
                self._reassign_segments([selected], target_part)
                status = html.Span(f"✓ Moved all of segment {selected} to {target_part}",
                                   style={'color': 'green'})
            else:
                self._save_undo_state()
                self.face2label, self.label2face_mask, new_id = split_segment_by_faces(
                    self.face2label, self.label2face_mask, selected, faces
                )
                refresh_after_label_change()
                if assign_to_part:
                    self._reassign_segments([new_id], target_part)
                    status = html.Span(
                        f"✓ Split {len(faces)} faces → segment {new_id}, assigned to {target_part}",
                        style={'color': 'green'})
                else:
                    if '_unassigned' not in self.current_assignment:
                        self.current_assignment['_unassigned'] = []
                    self.current_assignment['_unassigned'].append(new_id)
                    status = html.Span(
                        f"✓ Split {len(faces)} faces → new segment {new_id} (unassigned)",
                        style={'color': 'green'})

            return (json.dumps(self.current_assignment), status, get_opts(), get_opts(),
                    [], face_selection_display([], selected))

        # Merge segments
        @app.callback(
            [Output('global-assignment', 'data', allow_duplicate=True),
             Output('segments-status', 'children', allow_duplicate=True),
             Output('segments-dropdown', 'options', allow_duplicate=True),
             Output('merge-dropdown', 'options', allow_duplicate=True),
             Output('merge-dropdown', 'value'),
             Output('face-select-store', 'data', allow_duplicate=True),
             Output('face-select-display', 'children', allow_duplicate=True)],
            [Input('merge-btn', 'n_clicks')],
            [State('merge-dropdown', 'value'),
             State('global-assignment', 'data')],
            prevent_initial_call=True
        )
        def merge_segs(n, to_merge, assignment_json):
            from dash import no_update

            def get_opts():
                return [{'label': f"Segment {s}", 'value': s} for s in self.all_segments]

            # n is 0 when the segments tab mounts (this callback's inputs live
            # inside the swapped-in tab while global-assignment does not, so
            # Dash fires it on mount despite prevent_initial_call).
            if not n:
                raise PreventUpdate
            if not to_merge or len(to_merge) < 2:
                return no_update, html.Span("Select 2+ segments", style={'color': 'orange'}), \
                       get_opts(), get_opts(), [], no_update, no_update
            
            self._save_undo_state()
            self.face2label, self.label2face_mask, merged = merge_segments(
                self.face2label, self.label2face_mask, to_merge
            )
            
            # Update state
            self.all_segments = sorted([int(s) for s in np.unique(self.face2label) if s >= 0])
            self.segment_data, self.scene_scale, _ = precompute_segment_geometry(
                self.mesh, self.face2label, self.all_segments
            )
            segment_data.clear()
            segment_data.update(self.segment_data)
            figure_builder.segment_data = self.segment_data
            figure_builder.scene_scale = self.scene_scale
            
            # Update assignment
            self.current_assignment = json.loads(assignment_json)
            for s in to_merge[1:]:
                for part in self.current_assignment:
                    if s in self.current_assignment[part]:
                        self.current_assignment[part].remove(s)
            
            return (json.dumps(self.current_assignment),
                   html.Span(f"✓ Merged {len(to_merge)} segments -> {merged}", style={'color': 'green'}),
                   get_opts(), get_opts(), [], [], "")
        
        # Undo
        @app.callback(
            [Output('global-assignment', 'data', allow_duplicate=True),
             Output('segments-status', 'children', allow_duplicate=True),
             Output('segments-dropdown', 'options', allow_duplicate=True),
             Output('merge-dropdown', 'options', allow_duplicate=True),
             Output('face-select-store', 'data', allow_duplicate=True),
             Output('face-select-display', 'children', allow_duplicate=True)],
            [Input('undo-btn', 'n_clicks')],
            prevent_initial_call=True
        )
        def undo(n):
            if n == 0:
                raise PreventUpdate

            def get_opts():
                return [{'label': f"Segment {s}", 'value': s} for s in self.all_segments]

            # Face indices may point at segments that no longer exist after the
            # undo; drop the selection.
            if self._restore_undo_state():
                segment_data.clear()
                segment_data.update(self.segment_data)
                figure_builder.segment_data = self.segment_data
                figure_builder.scene_scale = self.scene_scale
                return json.dumps(self.current_assignment), \
                       html.Span("↶ Undo successful", style={'color': 'green'}), \
                       get_opts(), get_opts(), [], ""
            return json.dumps(self.current_assignment), \
                   html.Span("Nothing to undo", style={'color': 'orange'}), \
                   get_opts(), get_opts(), [], ""
        
        # =====================================================
        # PARTS TAB CALLBACKS
        # =====================================================
        
        @app.callback(
            [Output('parts-viewer', 'figure'),
             Output('parts-info', 'children'),
             Output('parts-assignments', 'children')],
            [Input('parts-dropdown', 'value'),
             Input('parts-explosion', 'value'),
             Input('global-assignment', 'data'),
             Input('parts-selected-store', 'data')],
            [State('parts-viewer', 'relayoutData')]
        )
        def update_parts(selected_part, explosion, assignment_json, selected_segments,
                         relayout_data):
            if assignment_json:
                self.current_assignment = json.loads(assignment_json)

            self._remember_camera(relayout_data)
            fig = self._apply_camera(figure_builder.build_part_view(
                part_name=selected_part,
                current_assignment=self.current_assignment,
                explosion_factor=explosion,
                selected_segments=selected_segments or []
            ))

            info = ""
            if selected_part and selected_part in self.current_assignment:
                segs = self.current_assignment[selected_part]
                total = sum(segment_data[s].face_count for s in segs if s in segment_data)
                info = html.Div([
                    html.Strong(selected_part, style={'color': part_colors.get(selected_part, '#888')}),
                    html.Br(),
                    html.Span(f"Segments: {len(segs)}"),
                    html.Br(),
                    html.Span(f"Faces: {total}"),
                    html.Br(),
                    html.Span(f"IDs: {sorted(segs)}")
                ])

            return fig, info, self._get_assignment_html()

        # Click segments of the viewed part to select them (gold); part change
        # or the clear button resets the selection.
        @app.callback(
            [Output('parts-selected-store', 'data'),
             Output('parts-selected-display', 'children')],
            [Input('parts-viewer', 'clickData'),
             Input('parts-dropdown', 'value'),
             Input('parts-clear-btn', 'n_clicks')],
            [State('parts-selected-store', 'data'),
             State('global-assignment', 'data')],
            prevent_initial_call=True
        )
        def parts_select(click_data, selected_part, _n_clear, selection, assignment_json):
            ctx = callback_context
            if not ctx.triggered:
                raise PreventUpdate
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]

            if trigger in ('parts-dropdown', 'parts-clear-btn'):
                return [], "No segments selected"

            if not click_data:
                raise PreventUpdate
            if assignment_json:
                self.current_assignment = json.loads(assignment_json)

            shown = figure_builder.part_view_segments(selected_part, self.current_assignment)
            try:
                seg_id = shown[click_data['points'][0]['curveNumber']]
            except (KeyError, IndexError):
                raise PreventUpdate

            selection = list(selection or [])
            if seg_id in selection:
                selection.remove(seg_id)
            else:
                selection.append(seg_id)

            if not selection:
                return [], "No segments selected"
            total = sum(segment_data[s].face_count for s in selection if s in segment_data)
            display = html.Div([
                html.Span("Selected: ", style={'fontWeight': 'bold'}),
                html.Span(f"{sorted(selection)}"),
                html.Br(),
                html.Span(f"{total} faces", style={'fontSize': '12px', 'color': '#666'}),
            ])
            return selection, display

        # Move the selected segments to another part.
        @app.callback(
            [Output('global-assignment', 'data', allow_duplicate=True),
             Output('parts-status', 'children'),
             Output('parts-selected-store', 'data', allow_duplicate=True),
             Output('parts-selected-display', 'children', allow_duplicate=True)],
            [Input('parts-move-btn', 'n_clicks')],
            [State('parts-selected-store', 'data'),
             State('parts-move-dropdown', 'value'),
             State('global-assignment', 'data')],
            prevent_initial_call=True
        )
        def parts_move(n, selection, new_part, assignment_json):
            from dash import no_update
            if n == 0:
                raise PreventUpdate
            if not selection:
                return (assignment_json,
                        html.Span("⚠ Click segments in the 3D view first", style={'color': 'orange'}),
                        no_update, no_update)
            if not new_part:
                return (assignment_json,
                        html.Span("⚠ Choose a part to move them to", style={'color': 'orange'}),
                        no_update, no_update)

            self.current_assignment = json.loads(assignment_json)
            self._save_undo_state()
            moved = self._reassign_segments(list(selection), new_part)
            return (json.dumps(self.current_assignment),
                    html.Span(f"✓ Moved {len(moved)} segment(s) to {new_part}",
                              style={'color': 'green'}),
                    [], "No segments selected")
    
    def run(self, port: int = 8050) -> dict:
        """Run the Dash app and return the result when user is done."""
        import threading
        import socket
        from werkzeug.serving import make_server
        
        # Check if port is in use and find an available one
        def find_available_port(start_port, max_attempts=10):
            for offset in range(max_attempts):
                test_port = start_port + offset
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('127.0.0.1', test_port))
                        return test_port
                except OSError:
                    continue
            raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts}")
        
        # Find available port
        actual_port = find_available_port(port)
        if actual_port != port:
            print(f"Port {port} in use, using port {actual_port} instead")
        
        def open_browser():
            time.sleep(1.5)
            webbrowser.open_new(f"http://localhost:{actual_port}")
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Create a proper WSGI server that we can shutdown
        server = make_server('127.0.0.1', actual_port, self.app.server, threaded=True)
        
        def run_server():
            server.serve_forever()
        
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        
        print(f"\n{'='*60}")
        print(f"Interactive Segment Correction UI started at http://localhost:{actual_port}")
        print(f"{'='*60}")
        print("\nTabs:")
        print("  - Overview: Assign segments to parts")
        print("  - Segments: Split and merge segments; grab stray islands")
        print("              and send them to another part")
        print("  - Parts: View a part and move its segments to other parts")
        print("\nNew to this tool? Expand the '❓ New here?' guide at the top of the page.")
        print("\nClick 'Done' when finished, or 'Cancel' to discard changes.")
        print(f"{'='*60}\n")
        
        while not self.result_holder['done']:
            time.sleep(0.5)
        
        # Shutdown the server properly
        print("\nShutting down server...")
        server.shutdown()
        
        print("Segment correction complete!")
        # Return updated assignment, face2label, and label2face_mask.
        # On Cancel, return the pristine arrays so this session's splits and
        # merges are truly discarded along with the assignment edits.
        if self.result_holder.get('cancelled'):
            return {
                'assignment': self.result_holder['result'],
                'face2label': self._orig_face2label,
                'label2face_mask': self._orig_label2face_mask
            }
        return {
            'assignment': self.result_holder['result'],
            'face2label': self.face2label,
            'label2face_mask': self.label2face_mask
        }


def interactive_segment_correction(
    mesh: trimesh.Trimesh,
    face2label: np.ndarray,
    label2face_mask: np.ndarray,
    part_segment_dict: dict,
    parts_list: list,
    explosion_factor: float = 0.3,
    verbose: bool = False,
    port: int = 8050
) -> dict:
    """
    Interactive web UI to review and correct VLM's segment-to-part assignments.
    
    Returns:
        dict with keys:
            - 'assignment': Updated part_segment_dict
            - 'face2label': Updated face2label array (may have new segment IDs from splits)
            - 'label2face_mask': Updated label2face_mask array (expanded for new segments)
    """
    app = SegmentCorrectionApp(
        mesh=mesh,
        face2label=face2label,
        label2face_mask=label2face_mask,
        part_segment_dict=part_segment_dict,
        parts_list=parts_list,
        explosion_factor=explosion_factor,
        verbose=verbose
    )
    return app.run(port=port)
