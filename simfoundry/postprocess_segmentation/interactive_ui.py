"""Interactive Dash-based segment correction UI."""

import json
import threading
import time
import webbrowser
import logging

import numpy as np
import trimesh

from .styles import STYLES, generate_part_colors
from .geometry import (
    precompute_segment_geometry, 
    find_coplanar_faces, 
    split_segment_by_faces,
    split_by_connected_components,
    find_faces_by_normal,
    merge_segments
)
from .visualization import SegmentFigureBuilder

# Suppress Dash's default HTTP request logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


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
        self.result_holder = {'result': None, 'done': False}
        
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
        overview_tab = html.Div([
            html.Div([
                # Left: 3D viewer
                html.Div([
                    dcc.Graph(
                        id='overview-viewer',
                        figure=figure_builder.build(explosion_factor=self.explosion_factor),
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
                    html.H4("Batch Assign Segments"),
                    html.P("Enter segment numbers (comma-separated) or click to select:", 
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
                               style={**STYLES['btn_reassign'], 'marginTop': '10px'}),
                    html.Div(id='overview-status', style={'marginTop': '10px'}),
                    
                    html.Hr(),
                    
                    html.H4("Current Assignments"),
                    html.Div(id='overview-assignments', children=self._get_assignment_html(),
                             style=STYLES['assignments_container']),
                    
                    html.Hr(),
                    
                    html.Button("✓ Done - Accept Changes", id='done-btn', n_clicks=0,
                               style=STYLES['btn_done']),
                    html.Button("✗ Cancel", id='cancel-btn', n_clicks=0,
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
        segments_tab = html.Div([
            html.Div([
                # Left: 3D viewer
                html.Div([
                    dcc.Graph(
                        id='segments-viewer',
                        figure=figure_builder.build(explosion_factor=self.explosion_factor),
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
                    html.P("Click on a planar surface to split it off.", 
                           style={'fontSize': '12px', 'color': '#666'}),
                    html.Button("Enter Planar Split Mode", id='planar-split-btn', n_clicks=0,
                               style={**STYLES['btn_reassign'], 'backgroundColor': '#9C27B0', 'width': '100%'}),
                    html.Div(id='planar-split-status', style={'marginTop': '5px'}),
                    html.Div([
                        html.Label("Angle Threshold:", style={'fontSize': '12px'}),
                        dcc.Slider(id='angle-slider', min=5, max=45, step=5, value=15,
                                  marks={5: '5°', 15: '15°', 30: '30°', 45: '45°'}),
                    ], style={'marginTop': '10px'}),
                    dcc.Checklist(
                        id='cross-segment-check',
                        options=[{'label': ' Cross-segment', 'value': 'cross'}],
                        value=[], style={'fontSize': '12px', 'marginTop': '5px'}
                    ),
                    
                    html.Hr(),
                    
                    # Split by Normal
                    html.H4("Split by Normal"),
                    html.P("Split all faces with similar orientation.", 
                           style={'fontSize': '12px', 'color': '#666'}),
                    html.Button("Enter Normal Split Mode", id='normal-split-btn', n_clicks=0,
                               style={**STYLES['btn_reassign'], 'backgroundColor': '#3F51B5', 'width': '100%'}),
                    html.Div(id='normal-split-status', style={'marginTop': '5px'}),
                    
                    html.Hr(),
                    
                    # Split Connected Components
                    html.H4("Split Connected Components"),
                    html.Div([
                        html.Label("Min Size:", style={'fontSize': '12px'}),
                        dcc.Input(id='min-component-slider', type='number',
                                  min=1, max=10000, step=1, value=10,
                                  style={'width': '80px'}),
                        html.Span(" faces minimum", style={'fontSize': '10px', 'color': '#888'}),
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
                               style={**STYLES['btn_reassign'], 'backgroundColor': '#FF5722', 'width': '100%', 'marginTop': '10px'}),
                    
                    html.Hr(),
                    
                    # Merge Segments
                    html.H4("Merge Segments"),
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
                               style={**STYLES['btn_reassign'], 'backgroundColor': '#607D8B', 'width': '100%'}),
                    html.Div(id='segments-status', style={'marginTop': '10px'}),
                ], style=STYLES['right_panel']),
            ], style=STYLES['main_layout']),
            
            # Hidden stores
            dcc.Store(id='segments-selected-store', data=None),
            dcc.Store(id='planar-mode-store', data=False),
            dcc.Store(id='normal-mode-store', data=False),
        ])
        
        # =====================================================
        # TAB 3: PARTS - View by part
        # =====================================================
        parts_tab = html.Div([
            html.Div([
                # Left: 3D viewer
                html.Div([
                    dcc.Graph(
                        id='parts-viewer',
                        figure=figure_builder.build(explosion_factor=self.explosion_factor),
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
                    html.H4("Select Part to View"),
                    dcc.Dropdown(
                        id='parts-dropdown',
                        options=[{'label': p, 'value': p} for p in self.parts_list],
                        placeholder="Select part..."
                    ),
                    html.Div(id='parts-info', style={'marginTop': '10px'}),
                    
                    html.Hr(),
                    
                    html.H4("Current Assignments"),
                    html.Div(id='parts-assignments', children=self._get_assignment_html(),
                             style=STYLES['assignments_container']),
                ], style=STYLES['right_panel']),
            ], style=STYLES['main_layout']),
        ])
        
        # =====================================================
        # MAIN LAYOUT WITH TABS
        # =====================================================
        self.app.layout = html.Div([
            html.H2("Segment-to-Part Assignment Editor", 
                    style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            dcc.Tabs(id='main-tabs', value='overview', children=[
                dcc.Tab(label='Overview (Assign)', value='overview'),
                dcc.Tab(label='Segments (Split/Merge)', value='segments'),
                dcc.Tab(label='Parts (View by Part)', value='parts'),
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
            if tab == 'overview':
                return overview_tab
            elif tab == 'segments':
                return segments_tab
            elif tab == 'parts':
                return parts_tab
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
             Input('global-assignment', 'data')]
        )
        def update_overview_viewer(explosion, _):
            self.current_assignment = json.loads(_) if _ else self.current_assignment
            return figure_builder.build(explosion_factor=explosion)
        
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
            self.result_holder['done'] = True
            raise PreventUpdate
        
        # =====================================================
        # SEGMENTS TAB CALLBACKS
        # =====================================================
        
        # Click or dropdown to select segment
        @app.callback(
            [Output('segments-selected-store', 'data'),
             Output('segments-selected-display', 'children')],
            [Input('segments-viewer', 'clickData'),
             Input('segments-dropdown', 'value')],
            [State('segments-selected-store', 'data'),
             State('planar-mode-store', 'data'),
             State('normal-mode-store', 'data')]
        )
        def segments_select(click_data, dropdown_val, current, planar_mode, normal_mode):
            ctx = callback_context
            if not ctx.triggered:
                raise PreventUpdate
            
            trigger = ctx.triggered[0]['prop_id'].split('.')[0]
            
            # If in split mode, don't change selection on click
            if trigger == 'segments-viewer' and (planar_mode or normal_mode):
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
            return seg_id, display
        
        # Update viewer on explosion change
        @app.callback(
            Output('segments-viewer', 'figure'),
            [Input('segments-explosion', 'value'),
             Input('global-assignment', 'data')]
        )
        def update_segments_viewer(explosion, _):
            if _:
                self.current_assignment = json.loads(_)
            return figure_builder.build(explosion_factor=explosion)
        
        # Toggle planar split mode
        @app.callback(
            [Output('planar-mode-store', 'data'),
             Output('planar-split-btn', 'children'),
             Output('planar-split-btn', 'style'),
             Output('planar-split-status', 'children'),
             Output('normal-mode-store', 'data', allow_duplicate=True)],
            [Input('planar-split-btn', 'n_clicks')],
            [State('planar-mode-store', 'data'),
             State('segments-selected-store', 'data')],
            prevent_initial_call=True
        )
        def toggle_planar_mode(n, current, selected):
            if n == 0:
                raise PreventUpdate
            new_mode = not current
            if new_mode:
                if selected is None:
                    return False, "Enter Planar Split Mode", \
                           {**STYLES['btn_reassign'], 'backgroundColor': '#9C27B0', 'width': '100%'}, \
                           html.Span("⚠ Select a segment first", style={'color': 'orange'}), False
                return True, "Exit Planar Split Mode", \
                       {**STYLES['btn_reassign'], 'backgroundColor': '#E91E63', 'width': '100%'}, \
                       html.Span(f"🔪 Click on segment {selected}", style={'color': '#9C27B0', 'fontWeight': 'bold'}), False
            return False, "Enter Planar Split Mode", \
                   {**STYLES['btn_reassign'], 'backgroundColor': '#9C27B0', 'width': '100%'}, "", False
        
        # Toggle normal split mode
        @app.callback(
            [Output('normal-mode-store', 'data'),
             Output('normal-split-btn', 'children'),
             Output('normal-split-btn', 'style'),
             Output('normal-split-status', 'children'),
             Output('planar-mode-store', 'data', allow_duplicate=True)],
            [Input('normal-split-btn', 'n_clicks')],
            [State('normal-mode-store', 'data'),
             State('segments-selected-store', 'data')],
            prevent_initial_call=True
        )
        def toggle_normal_mode(n, current, selected):
            if n == 0:
                raise PreventUpdate
            new_mode = not current
            if new_mode:
                if selected is None:
                    return False, "Enter Normal Split Mode", \
                           {**STYLES['btn_reassign'], 'backgroundColor': '#3F51B5', 'width': '100%'}, \
                           html.Span("⚠ Select a segment first", style={'color': 'orange'}), False
                return True, "Exit Normal Split Mode", \
                       {**STYLES['btn_reassign'], 'backgroundColor': '#E91E63', 'width': '100%'}, \
                       html.Span(f"🎯 Click on segment {selected}", style={'color': '#3F51B5', 'fontWeight': 'bold'}), False
            return False, "Enter Normal Split Mode", \
                   {**STYLES['btn_reassign'], 'backgroundColor': '#3F51B5', 'width': '100%'}, "", False
        
        # Handle split click (planar or normal)
        @app.callback(
            [Output('global-assignment', 'data', allow_duplicate=True),
             Output('segments-status', 'children'),
             Output('segments-dropdown', 'options'),
             Output('merge-dropdown', 'options')],
            [Input('segments-viewer', 'clickData')],
            [State('planar-mode-store', 'data'),
             State('normal-mode-store', 'data'),
             State('segments-selected-store', 'data'),
             State('global-assignment', 'data'),
             State('angle-slider', 'value'),
             State('cross-segment-check', 'value')],
            prevent_initial_call=True
        )
        def handle_split_click(click_data, planar_mode, normal_mode, selected, assignment_json,
                              angle, cross_val):
            if not (planar_mode or normal_mode) or selected is None or click_data is None:
                raise PreventUpdate
            
            def get_opts():
                return [{'label': f"Segment {s}", 'value': s} for s in self.all_segments]
            
            try:
                point = click_data['points'][0]
                curve = point['curveNumber']
                clicked_seg = list(segment_data.keys())[curve]
                
                if clicked_seg != selected:
                    return assignment_json, html.Span(f"⚠ Click on segment {selected}", style={'color': 'orange'}), \
                           get_opts(), get_opts()
                
                # Find clicked face
                clicked_pos = np.array([point['x'], point['y'], point['z']])
                seg_mask = self.face2label == selected
                seg_faces = np.where(seg_mask)[0]
                
                vertices = np.array(self.mesh.vertices)
                faces = np.array(self.mesh.faces)
                centroids = vertices[faces[seg_faces]].mean(axis=1)
                
                seg = segment_data.get(selected)
                if seg:
                    from .styles import EXPLOSION_SCALE
                    offset = seg.direction * self.explosion_factor * self.scene_scale * EXPLOSION_SCALE
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
                           get_opts(), get_opts()
                
                self._save_undo_state()
                
                # Create new segment
                new_id = int(np.max(self.face2label) + 1)
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
                if '_unassigned' not in self.current_assignment:
                    self.current_assignment['_unassigned'] = []
                self.current_assignment['_unassigned'].append(new_id)
                
                return (json.dumps(self.current_assignment),
                       html.Span(f"✓ Split {len(faces_to_split)} faces ({split_type}) -> segment {new_id}", 
                                style={'color': 'green'}),
                       get_opts(), get_opts())
            except Exception as e:
                return assignment_json, html.Span(f"Error: {e}", style={'color': 'red'}), \
                       get_opts(), get_opts()
        
        # Split by connected components
        @app.callback(
            [Output('global-assignment', 'data', allow_duplicate=True),
             Output('segments-status', 'children', allow_duplicate=True),
             Output('segments-dropdown', 'options', allow_duplicate=True),
             Output('merge-dropdown', 'options', allow_duplicate=True)],
            [Input('split-components-btn', 'n_clicks')],
            [State('segments-selected-store', 'data'),
             State('global-assignment', 'data'),
             State('min-component-slider', 'value'),
             State('spatial-threshold-input', 'value')],
            prevent_initial_call=True
        )
        def split_components(n, selected, assignment_json, min_size, spatial_threshold):
            if n == 0 or selected is None:
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
                return assignment_json, html.Span("Only 1 component", style={'color': 'orange'}), \
                       get_opts(), get_opts()
            
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
                   get_opts(), get_opts())
        
        # Merge segments
        @app.callback(
            [Output('global-assignment', 'data', allow_duplicate=True),
             Output('segments-status', 'children', allow_duplicate=True),
             Output('segments-dropdown', 'options', allow_duplicate=True),
             Output('merge-dropdown', 'options', allow_duplicate=True),
             Output('merge-dropdown', 'value')],
            [Input('merge-btn', 'n_clicks')],
            [State('merge-dropdown', 'value'),
             State('global-assignment', 'data')],
            prevent_initial_call=True
        )
        def merge_segs(n, to_merge, assignment_json):
            def get_opts():
                return [{'label': f"Segment {s}", 'value': s} for s in self.all_segments]
            
            if n == 0 or not to_merge or len(to_merge) < 2:
                return assignment_json, html.Span("Select 2+ segments", style={'color': 'orange'}), \
                       get_opts(), get_opts(), []
            
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
                   get_opts(), get_opts(), [])
        
        # Undo
        @app.callback(
            [Output('global-assignment', 'data', allow_duplicate=True),
             Output('segments-status', 'children', allow_duplicate=True),
             Output('segments-dropdown', 'options', allow_duplicate=True),
             Output('merge-dropdown', 'options', allow_duplicate=True)],
            [Input('undo-btn', 'n_clicks')],
            prevent_initial_call=True
        )
        def undo(n):
            if n == 0:
                raise PreventUpdate
            
            def get_opts():
                return [{'label': f"Segment {s}", 'value': s} for s in self.all_segments]
            
            if self._restore_undo_state():
                segment_data.clear()
                segment_data.update(self.segment_data)
                figure_builder.segment_data = self.segment_data
                figure_builder.scene_scale = self.scene_scale
                return json.dumps(self.current_assignment), \
                       html.Span("↶ Undo successful", style={'color': 'green'}), \
                       get_opts(), get_opts()
            return json.dumps(self.current_assignment), \
                   html.Span("Nothing to undo", style={'color': 'orange'}), \
                   get_opts(), get_opts()
        
        # =====================================================
        # PARTS TAB CALLBACKS
        # =====================================================
        
        @app.callback(
            [Output('parts-viewer', 'figure'),
             Output('parts-info', 'children'),
             Output('parts-assignments', 'children')],
            [Input('parts-dropdown', 'value'),
             Input('parts-explosion', 'value'),
             Input('global-assignment', 'data')]
        )
        def update_parts(selected_part, explosion, assignment_json):
            if assignment_json:
                self.current_assignment = json.loads(assignment_json)
            
            fig = figure_builder.build_part_view(
                part_name=selected_part,
                current_assignment=self.current_assignment,
                explosion_factor=explosion
            )
            
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
        print("  - Segments: Split and merge segments")
        print("  - Parts: View by part")
        print("\nClick 'Done' when finished, or 'Cancel' to discard changes.")
        print(f"{'='*60}\n")
        
        while not self.result_holder['done']:
            time.sleep(0.5)
        
        # Shutdown the server properly
        print("\nShutting down server...")
        server.shutdown()
        
        print("Segment correction complete!")
        # Return updated assignment, face2label, and label2face_mask
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
