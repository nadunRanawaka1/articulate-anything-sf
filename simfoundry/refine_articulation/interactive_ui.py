"""Interactive Dash-based articulation refinement UI.

The articulation counterpart of postprocess_segmentation/interactive_ui.py:
where that app corrects the VLM's segment-to-part assignment before the URDF
exists, this one refines the *published* articulation result
(results/mobility.urdf) — joint limits, pivot positions, axes, joint types,
and dynamic properties (damping, friction, per-part mass) — with a live 3D
preview of the articulated motion.

Launch idiom mirrors SegmentCorrectionApp: a werkzeug server on 127.0.0.1
(port probe from the requested port), browser auto-open, and a blocking poll
on a result holder until the user clicks "Save & Finish" or "Cancel".
"""

import json
import logging
import math
import threading
import time
import webbrowser

import numpy as np

from postprocess_segmentation.styles import STYLES, generate_part_colors

from .urdf_model import ArticulationModel, MOVABLE_TYPES
from .visualization import ArticulationFigureBuilder

# Suppress Dash's default HTTP request logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

UNDO_LIMIT = 50

REFINE_STYLES = {
    'section': {
        'padding': '10px',
        'backgroundColor': '#f7f7f7',
        'borderRadius': '5px',
        'marginBottom': '12px',
    },
    'vec_input': {
        'width': '31%',
        'marginRight': '2%',
        'padding': '4px',
    },
    'num_input': {
        'width': '47%',
        'marginRight': '2%',
        'padding': '4px',
    },
    'btn_small': {
        'padding': '6px 10px',
        'fontSize': '13px',
        'backgroundColor': '#607d8b',
        'color': 'white',
        'border': 'none',
        'borderRadius': '4px',
        'cursor': 'pointer',
        'marginRight': '6px',
        'marginTop': '6px',
    },
    'btn_apply': {
        'padding': '6px 12px',
        'fontSize': '13px',
        'backgroundColor': '#4CAF50',
        'color': 'white',
        'border': 'none',
        'borderRadius': '4px',
        'cursor': 'pointer',
        'marginRight': '6px',
        'marginTop': '6px',
    },
    'hint': {
        'fontSize': '12px',
        'color': '#666',
        'marginTop': '4px',
    },
    'dirty_badge': {
        'color': '#e65100',
        'fontWeight': 'bold',
        'marginLeft': '12px',
    },
}


def _num(value, default=None):
    """Numeric input value -> float, treating empty/None as `default`."""
    if value is None or value == "":
        return default
    return float(value)


class ArticulationRefinementApp:
    """Web UI to refine articulation results for one or more objects."""

    def __init__(self, models: dict, verbose: bool = False):
        if not models:
            raise ValueError("No articulation models to refine")
        self.models: dict[str, ArticulationModel] = dict(models)
        self.verbose = verbose
        self.result_holder = {'result': None, 'done': False}
        self._builders: dict[str, ArticulationFigureBuilder] = {}
        self._undo: dict[str, list] = {name: [] for name in self.models}
        self._anim_dir: dict[str, float] = {}
        self._setup_app()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _model(self, obj_name: str) -> ArticulationModel:
        return self.models[obj_name]

    def _builder(self, obj_name: str) -> ArticulationFigureBuilder:
        if obj_name not in self._builders:
            self._builders[obj_name] = ArticulationFigureBuilder(self.models[obj_name])
        return self._builders[obj_name]

    def _push_undo(self, obj_name: str):
        stack = self._undo[obj_name]
        stack.append(self.models[obj_name].snapshot())
        if len(stack) > UNDO_LIMIT:
            stack.pop(0)

    def _apply_part_edits(self, obj_name: str, links: list, masses: list,
                          frictions: list, dampings: list):
        """Apply the per-part physics table (empty inputs clear the override)."""
        model = self.models[obj_name]
        for link, mass, friction, damping in zip(links, masses, frictions, dampings):
            model.set_part_properties(
                link,
                mass_kg=_num(mass),
                friction=_num(friction),
                joint_damping=_num(damping),
            )

    def _joint_options(self, obj_name: str) -> list:
        model = self._model(obj_name)
        return [
            {"label": f"{name}  [{model.joints[name].joint_type}]", "value": name}
            for name in model.editable_joints()
        ]

    def _default_joint(self, obj_name: str):
        model = self._model(obj_name)
        movable = model.movable_joints()
        if movable:
            return movable[0]
        editable = model.editable_joints()
        return editable[0] if editable else None

    def _slider_bounds(self, obj_name: str, joint_name):
        model = self._model(obj_name)
        if joint_name is None or joint_name not in model.joints:
            return 0.0, 1.0, True
        joint = model.joints[joint_name]
        if not joint.is_movable:
            return 0.0, 1.0, True
        if joint.joint_type == "continuous" or joint.limit is None:
            return -math.pi, math.pi, False
        return joint.limit["lower"], joint.limit["upper"], False

    def _parts_table(self, obj_name: str):
        from dash import html, dcc

        model = self._model(obj_name)
        colors = generate_part_colors(model.geometry_links())
        rows = [
            html.Div([
                html.Span(style={'display': 'inline-block', 'width': '90px'}),
                html.Span("mass (kg)", style={'display': 'inline-block', 'width': '80px', 'fontSize': '11px'}),
                html.Span("surface fric.", style={'display': 'inline-block', 'width': '80px', 'fontSize': '11px'}),
                html.Span("joint damp.", style={'display': 'inline-block', 'width': '80px', 'fontSize': '11px'}),
            ])
        ]
        for link_name in model.geometry_links():
            part = model.overrides["parts"].get(link_name, {})
            estimate = model.estimates["parts"].get(link_name, {})

            def hint(key):
                # Show the pipeline's estimate as the placeholder so users see
                # what "no override" means; blank input keeps the estimate.
                value = estimate.get(key)
                return f"{value:.3g}" if isinstance(value, (int, float)) else "auto"

            rows.append(html.Div([
                html.Span("■ ", style={'color': colors.get(link_name, '#999')}),
                html.Span(link_name, style={
                    'display': 'inline-block', 'width': '80px', 'fontSize': '12px',
                    'overflow': 'hidden', 'textOverflow': 'ellipsis', 'verticalAlign': 'middle',
                }, title=link_name),
                dcc.Input(id={'type': 'part-mass', 'link': link_name}, type='number',
                          value=part.get('mass_kg'), min=0, placeholder=hint('mass_kg'),
                          style={'width': '72px', 'marginRight': '8px'}),
                dcc.Input(id={'type': 'part-friction', 'link': link_name}, type='number',
                          value=part.get('friction'), min=0, placeholder=hint('friction'),
                          style={'width': '72px', 'marginRight': '8px'}),
                dcc.Input(id={'type': 'part-damping', 'link': link_name}, type='number',
                          value=part.get('joint_damping'), min=0, placeholder=hint('joint_damping'),
                          style={'width': '72px'}),
            ], style={'marginTop': '4px'}))
        return rows

    # ------------------------------------------------------------------
    # App / layout
    # ------------------------------------------------------------------

    def _setup_app(self):
        from dash import Dash, html, dcc

        self.app = Dash(__name__, suppress_callback_exceptions=True)
        self.app.title = "Articulation Refinement"
        object_names = list(self.models.keys())
        first = object_names[0]

        def vec_inputs(prefix):
            return html.Div([
                dcc.Input(id=f'{prefix}-x', type='number', placeholder='x',
                          style=REFINE_STYLES['vec_input']),
                dcc.Input(id=f'{prefix}-y', type='number', placeholder='y',
                          style=REFINE_STYLES['vec_input']),
                dcc.Input(id=f'{prefix}-z', type='number', placeholder='z',
                          style=REFINE_STYLES['vec_input']),
            ])

        def num_field(label, input_id, **input_kwargs):
            """A number input with a persistent label above it (placeholders
            disappear once a value is filled in, leaving fields unlabeled)."""
            return html.Div([
                html.Label(label, style={'fontSize': '11px', 'color': '#555',
                                         'display': 'block', 'marginBottom': '2px'}),
                dcc.Input(id=input_id, type='number',
                          style={'width': '100%', 'padding': '4px',
                                 'boxSizing': 'border-box'},
                          **input_kwargs),
            ], style={'display': 'inline-block', 'width': '47%',
                      'marginRight': '2%', 'verticalAlign': 'top'})

        left_panel = html.Div([
            dcc.Graph(id='viewport', config={'displayModeBar': True, 'scrollZoom': True},
                      style=STYLES['graph']),
            html.Div([
                html.B("Test joint motion"),
                html.Span(id='q-readout', style={'marginLeft': '10px', 'fontSize': '13px'}),
                dcc.Slider(id='q-slider', min=0.0, max=1.0, step=0.001, value=0.0,
                           marks=None, updatemode='drag',
                           tooltip={"placement": "bottom", "always_visible": False}),
                html.Div([
                    html.Button("q = 0", id='btn-q-zero', style=REFINE_STYLES['btn_small']),
                    html.Button("→ lower", id='btn-q-lower', style=REFINE_STYLES['btn_small']),
                    html.Button("→ upper", id='btn-q-upper', style=REFINE_STYLES['btn_small']),
                    dcc.Checklist(id='chk-ghosts',
                                  options=[{'label': ' Show limit ghosts', 'value': 'ghosts'}],
                                  value=['ghosts'],
                                  style={'display': 'inline-block', 'marginLeft': '14px'}),
                    dcc.Checklist(id='chk-animate',
                                  options=[{'label': ' Animate', 'value': 'animate'}],
                                  value=[],
                                  style={'display': 'inline-block', 'marginLeft': '14px'}),
                    dcc.RadioItems(id='color-mode',
                                   options=[{'label': ' Parts', 'value': 'segmented'},
                                            {'label': ' Textured', 'value': 'textured'}],
                                   value='segmented', inline=True,
                                   style={'display': 'inline-block', 'marginLeft': '14px'}),
                ]),
                dcc.Interval(id='animate-interval', interval=120, disabled=True),
            ], style=STYLES['slider_container']),
        ], style=STYLES['left_panel'])

        right_panel = html.Div([
            html.Div([
                html.B("Joint"),
                dcc.Dropdown(id='joint-dropdown', options=self._joint_options(first),
                             value=self._default_joint(first), clearable=False),
                html.Div(id='joint-info', style={**STYLES['selected_display'], 'marginTop': '8px'}),
                html.Label("Type", style={'fontSize': '12px'}),
                dcc.Dropdown(id='joint-type-dropdown', clearable=False,
                             options=[{'label': t, 'value': t}
                                      for t in list(MOVABLE_TYPES) + ['fixed']]),
            ], style=REFINE_STYLES['section']),

            html.Div([
                html.B("Axis"),
                html.Div("Direction in the parent-link frame (normalized on apply).",
                         style=REFINE_STYLES['hint']),
                vec_inputs('axis'),
                html.Div([
                    html.Button("Apply", id='btn-axis-apply', style=REFINE_STYLES['btn_apply']),
                    html.Button("Flip", id='btn-axis-flip', style=REFINE_STYLES['btn_small']),
                    html.Button("World X", id='btn-axis-world-x', style=REFINE_STYLES['btn_small']),
                    html.Button("World Y", id='btn-axis-world-y', style=REFINE_STYLES['btn_small']),
                    html.Button("World Z", id='btn-axis-world-z', style=REFINE_STYLES['btn_small']),
                ]),
                html.Div(id='axis-world-readout', style=REFINE_STYLES['hint']),
            ], style=REFINE_STYLES['section'], id='axis-section'),

            html.Div([
                html.B("Pivot / origin"),
                html.Div("Joint origin in the parent-link frame.", style=REFINE_STYLES['hint']),
                vec_inputs('origin'),
                html.Div([
                    html.Button("Apply", id='btn-origin-apply', style=REFINE_STYLES['btn_apply']),
                    html.Button("Pick point on mesh", id='btn-pick-pivot',
                                style=REFINE_STYLES['btn_small']),
                ]),
                dcc.Checklist(
                    id='chk-compensate',
                    options=[{'label': ' Keep part in place (compensate child geometry)',
                              'value': 'comp'}],
                    value=['comp'], style=REFINE_STYLES['hint']),
            ], style=REFINE_STYLES['section'], id='pivot-section'),

            html.Div([
                html.B("Limits"),
                html.Div(id='limit-units-hint', style=REFINE_STYLES['hint']),
                html.Div([
                    num_field("lower limit", 'limit-lower', placeholder='lower'),
                    num_field("upper limit", 'limit-upper', placeholder='upper'),
                ]),
                html.Div([
                    num_field("effort (max force/torque)", 'limit-effort',
                              placeholder='effort', min=0),
                    num_field("max velocity", 'limit-velocity',
                              placeholder='velocity', min=0),
                ], style={'marginTop': '6px'}),
                html.Div(id='limit-degrees', style=REFINE_STYLES['hint']),
                html.Button("Apply", id='btn-limits-apply', style=REFINE_STYLES['btn_apply']),
            ], style=REFINE_STYLES['section'], id='limits-section'),

            html.Div([
                html.B("Joint dynamics"),
                html.Div([
                    num_field("damping", 'dyn-damping', placeholder='damping', min=0),
                    num_field("friction", 'dyn-friction', placeholder='friction', min=0),
                ]),
                html.Button("Apply", id='btn-dyn-apply', style=REFINE_STYLES['btn_apply']),
                html.Div("Also saved to physics_overrides.json so the sim-ready "
                         "importer keeps them.", style=REFINE_STYLES['hint']),
            ], style=REFINE_STYLES['section'], id='dynamics-section'),

            html.Details([
                html.Summary("Per-part physics (mass / surface friction / joint damping)"),
                html.Div("Blank = keep the pipeline's automatic estimate.",
                         style=REFINE_STYLES['hint']),
                html.Div(id='parts-table', children=self._parts_table(first)),
                html.Button("Apply part physics", id='btn-parts-apply',
                            style=REFINE_STYLES['btn_apply']),
            ], style=REFINE_STYLES['section']),

            html.Div([
                html.Button("Undo", id='btn-undo', style=REFINE_STYLES['btn_small']),
                html.Button("Reset joint", id='btn-reset-joint', style=REFINE_STYLES['btn_small']),
                html.Button("Reset all", id='btn-reset-all', style=REFINE_STYLES['btn_small']),
            ]),
            html.Button("Save", id='btn-save', style={**STYLES['btn_reassign'], 'marginTop': '10px'}),
            html.Button("Save & Finish", id='btn-done', style=STYLES['btn_done']),
            html.Button("Cancel (discard unsaved)", id='btn-cancel', style=STYLES['btn_cancel']),
        ], style=STYLES['right_panel'])

        self.app.layout = html.Div([
            html.H2("Articulation Refinement", style=STYLES['header']),
            html.Div([
                html.Label("Object: ", style={'marginRight': '8px', 'fontWeight': 'bold'}),
                dcc.Dropdown(id='object-dropdown',
                             options=[{'label': n, 'value': n} for n in object_names],
                             value=first, clearable=False,
                             style={'width': '340px', 'display': 'inline-block',
                                    'verticalAlign': 'middle'}),
                html.Span(id='dirty-badge', style=REFINE_STYLES['dirty_badge']),
                html.Span(id='save-status', style={'marginLeft': '16px'}),
            ], style={'textAlign': 'center', 'marginBottom': '6px'}),
            html.Div([left_panel, right_panel], style=STYLES['main_layout']),
            dcc.Store(id='store-object', data=first),
            dcc.Store(id='store-selected-joint', data=self._default_joint(first)),
            dcc.Store(id='store-q', data=json.dumps({})),
            dcc.Store(id='store-pick-mode', data=False),
            dcc.Store(id='store-refresh', data=0),
        ], style=STYLES['container'])

        self._setup_callbacks()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _setup_callbacks(self):
        from dash import Input, Output, State, ALL, callback_context, html, no_update
        from dash.exceptions import PreventUpdate

        app = self.app

        def ok(msg):
            return html.Span(msg, style={'color': 'green'})

        def err(msg):
            return html.Span(str(msg), style={'color': 'red'})

        # ---------- object switch ----------
        @app.callback(
            Output('joint-dropdown', 'options'),
            Output('joint-dropdown', 'value'),
            Output('store-object', 'data'),
            Output('store-q', 'data'),
            Output('parts-table', 'children'),
            Output('store-refresh', 'data'),
            Output('store-pick-mode', 'data', allow_duplicate=True),
            Output('btn-pick-pivot', 'children', allow_duplicate=True),
            Output('btn-pick-pivot', 'style', allow_duplicate=True),
            Input('object-dropdown', 'value'),
            State('store-refresh', 'data'),
            prevent_initial_call=True,
        )
        def on_object_change(obj_name, refresh):
            # Disarm pick mode: a click on the new object's mesh must not
            # inherit the previous object's pending pivot pick.
            return (
                self._joint_options(obj_name),
                self._default_joint(obj_name),
                obj_name,
                json.dumps({}),
                self._parts_table(obj_name),
                (refresh or 0) + 1,
                False, "Pick point on mesh", REFINE_STYLES['btn_small'],
            )

        # ---------- joint panel sync ----------
        @app.callback(
            Output('store-selected-joint', 'data'),
            Output('joint-info', 'children'),
            Output('joint-type-dropdown', 'value'),
            Output('axis-x', 'value'), Output('axis-y', 'value'), Output('axis-z', 'value'),
            Output('origin-x', 'value'), Output('origin-y', 'value'), Output('origin-z', 'value'),
            Output('limit-lower', 'value'), Output('limit-upper', 'value'),
            Output('limit-effort', 'value'), Output('limit-velocity', 'value'),
            Output('dyn-damping', 'value'), Output('dyn-friction', 'value'),
            Output('q-slider', 'min'), Output('q-slider', 'max'),
            Output('q-slider', 'value'), Output('q-slider', 'disabled'),
            Output('axis-section', 'style'), Output('pivot-section', 'style'),
            Output('limits-section', 'style'), Output('dynamics-section', 'style'),
            Output('axis-world-readout', 'children'),
            Output('limit-units-hint', 'children'),
            Output('limit-degrees', 'children'),
            Output('dirty-badge', 'children'),
            Input('joint-dropdown', 'value'),
            Input('store-refresh', 'data'),
            State('store-object', 'data'),
            State('store-q', 'data'),
        )
        def sync_joint_panel(joint_name, _refresh, obj_name, q_json):
            model = self._model(obj_name)
            dirty = "● unsaved edits" if model.dirty else ""
            hidden = {'display': 'none'}
            section = REFINE_STYLES['section']
            if not joint_name or joint_name not in model.joints:
                return (None, "This object has no editable joints.", None,
                        None, None, None, None, None, None,
                        None, None, None, None, None, None,
                        0.0, 1.0, 0.0, True,
                        hidden, hidden, hidden, hidden, "", "", "", dirty)

            joint = model.joints[joint_name]
            q_values = json.loads(q_json or "{}")
            lo, hi, disabled = self._slider_bounds(obj_name, joint_name)
            q = min(max(float(q_values.get(joint_name, 0.0)), lo), hi)

            info = html.Div([
                html.Div([html.B(joint.name)]),
                html.Div(f"{joint.parent}  →  {joint.child}", style={'fontSize': '12px'}),
            ])
            axis = joint.axis if joint.axis is not None else np.array([1.0, 0.0, 0.0])
            origin = joint.origin_xyz
            limit = joint.limit or {}
            dynamics = joint.dynamics or {}

            movable = joint.is_movable
            has_limits = joint.joint_type in ("revolute", "prismatic")
            _, axis_world, _ = model.joint_world_frame(joint_name, q_values)
            axis_readout = ("World frame: [" + ", ".join(f"{v:+.3f}" for v in axis_world) + "]") if movable else ""
            if joint.joint_type == "revolute":
                units_hint = "Radians (q=0 is the as-scanned pose)."
                degrees = ""
                if limit:
                    degrees = (f"= {math.degrees(limit['lower']):.1f}° … "
                               f"{math.degrees(limit['upper']):.1f}°")
            elif joint.joint_type == "prismatic":
                units_hint = "Meters, in canonical-mesh scale (q=0 is the as-scanned pose)."
                degrees = ""
            else:
                units_hint = ""
                degrees = ""

            return (
                joint_name, info, joint.joint_type,
                round(float(axis[0]), 6), round(float(axis[1]), 6), round(float(axis[2]), 6),
                round(float(origin[0]), 6), round(float(origin[1]), 6), round(float(origin[2]), 6),
                limit.get('lower'), limit.get('upper'),
                limit.get('effort'), limit.get('velocity'),
                dynamics.get('damping'), dynamics.get('friction'),
                lo, hi, q, disabled,
                section if movable else hidden,
                section if movable else hidden,
                section if has_limits else hidden,
                section if movable else hidden,
                axis_readout, units_hint, degrees, dirty,
            )

        # ---------- joint type ----------
        @app.callback(
            Output('store-refresh', 'data', allow_duplicate=True),
            Output('save-status', 'children', allow_duplicate=True),
            Output('joint-dropdown', 'options', allow_duplicate=True),
            Input('joint-type-dropdown', 'value'),
            State('store-selected-joint', 'data'),
            State('store-object', 'data'),
            State('store-refresh', 'data'),
            prevent_initial_call=True,
        )
        def on_type_change(new_type, joint_name, obj_name, refresh):
            if not new_type or not joint_name:
                raise PreventUpdate
            model = self._model(obj_name)
            if joint_name not in model.joints or model.joints[joint_name].joint_type == new_type:
                raise PreventUpdate
            self._push_undo(obj_name)
            try:
                model.set_joint_type(joint_name, new_type)
            except ValueError as exc:
                self._undo[obj_name].pop()
                return no_update, err(exc), no_update
            return (refresh or 0) + 1, ok(f"{joint_name} is now {new_type}"), self._joint_options(obj_name)

        # ---------- axis ----------
        @app.callback(
            Output('store-refresh', 'data', allow_duplicate=True),
            Output('save-status', 'children', allow_duplicate=True),
            Input('btn-axis-apply', 'n_clicks'),
            Input('btn-axis-flip', 'n_clicks'),
            Input('btn-axis-world-x', 'n_clicks'),
            Input('btn-axis-world-y', 'n_clicks'),
            Input('btn-axis-world-z', 'n_clicks'),
            State('axis-x', 'value'), State('axis-y', 'value'), State('axis-z', 'value'),
            State('store-selected-joint', 'data'),
            State('store-object', 'data'),
            State('store-q', 'data'),
            State('store-refresh', 'data'),
            prevent_initial_call=True,
        )
        def on_axis_edit(_a, _f, _wx, _wy, _wz, ax, ay, az, joint_name, obj_name, q_json, refresh):
            if not joint_name:
                raise PreventUpdate
            trigger = callback_context.triggered[0]['prop_id'].split('.')[0]
            model = self._model(obj_name)
            joint = model.joints[joint_name]
            q_values = json.loads(q_json or "{}")
            try:
                if trigger == 'btn-axis-apply':
                    new_axis = [_num(ax, 0.0), _num(ay, 0.0), _num(az, 0.0)]
                elif trigger == 'btn-axis-flip':
                    current = joint.axis if joint.axis is not None else np.array([1.0, 0.0, 0.0])
                    new_axis = (-np.asarray(current)).tolist()
                else:
                    world_dir = {'btn-axis-world-x': [1.0, 0.0, 0.0],
                                 'btn-axis-world-y': [0.0, 1.0, 0.0],
                                 'btn-axis-world-z': [0.0, 0.0, 1.0]}[trigger]
                    new_axis = model.world_dir_to_parent_frame(joint_name, world_dir, q_values).tolist()
            except ValueError as exc:
                return no_update, err(exc)
            self._push_undo(obj_name)
            try:
                model.set_axis(joint_name, new_axis)
            except ValueError as exc:
                self._undo[obj_name].pop()
                return no_update, err(exc)
            return (refresh or 0) + 1, ok(f"Axis of {joint_name} updated")

        # ---------- origin ----------
        @app.callback(
            Output('store-refresh', 'data', allow_duplicate=True),
            Output('save-status', 'children', allow_duplicate=True),
            Input('btn-origin-apply', 'n_clicks'),
            State('origin-x', 'value'), State('origin-y', 'value'), State('origin-z', 'value'),
            State('chk-compensate', 'value'),
            State('store-selected-joint', 'data'),
            State('store-object', 'data'),
            State('store-refresh', 'data'),
            prevent_initial_call=True,
        )
        def on_origin_apply(_n, ox, oy, oz, compensate, joint_name, obj_name, refresh):
            if not joint_name:
                raise PreventUpdate
            model = self._model(obj_name)
            try:
                self._push_undo(obj_name)
                model.set_origin(
                    joint_name,
                    [_num(ox, 0.0), _num(oy, 0.0), _num(oz, 0.0)],
                    compensate=bool(compensate),
                )
            except ValueError as exc:
                return no_update, err(exc)
            return (refresh or 0) + 1, ok(f"Pivot of {joint_name} updated")

        # ---------- pivot picking ----------
        @app.callback(
            Output('store-pick-mode', 'data'),
            Output('btn-pick-pivot', 'children'),
            Output('btn-pick-pivot', 'style'),
            Input('btn-pick-pivot', 'n_clicks'),
            State('store-pick-mode', 'data'),
            prevent_initial_call=True,
        )
        def on_pick_toggle(_n, active):
            active = not bool(active)
            style = dict(REFINE_STYLES['btn_small'])
            label = "Pick point on mesh"
            if active:
                style['backgroundColor'] = '#e65100'
                label = "Click the 3D view… (click here to cancel)"
            return active, label, style

        @app.callback(
            Output('store-refresh', 'data', allow_duplicate=True),
            Output('save-status', 'children', allow_duplicate=True),
            Output('store-pick-mode', 'data', allow_duplicate=True),
            Output('btn-pick-pivot', 'children', allow_duplicate=True),
            Output('btn-pick-pivot', 'style', allow_duplicate=True),
            Input('viewport', 'clickData'),
            State('store-pick-mode', 'data'),
            State('chk-compensate', 'value'),
            State('store-selected-joint', 'data'),
            State('store-object', 'data'),
            State('store-q', 'data'),
            State('store-refresh', 'data'),
            prevent_initial_call=True,
        )
        def on_viewport_click(click_data, pick_mode, compensate, joint_name, obj_name, q_json, refresh):
            if not pick_mode or not joint_name or not click_data:
                raise PreventUpdate
            point = click_data['points'][0]
            if not all(k in point for k in ('x', 'y', 'z')):
                raise PreventUpdate
            model = self._model(obj_name)
            q_values = json.loads(q_json or "{}")
            world_point = [point['x'], point['y'], point['z']]
            # customdata carries the clicked link's name: a pick on the joint's
            # own (possibly displaced) subtree is mapped back to that feature's
            # rest position, so picking works at any test-slider q.
            clicked_link = point.get('customdata')
            if isinstance(clicked_link, (list, tuple)):
                clicked_link = clicked_link[0] if clicked_link else None
            new_origin = model.world_point_to_pivot(
                joint_name, world_point, q_values, clicked_link=clicked_link)
            self._push_undo(obj_name)
            model.set_origin(joint_name, new_origin, compensate=bool(compensate))
            return ((refresh or 0) + 1,
                    ok(f"Pivot of {joint_name} set from picked point"),
                    False, "Pick point on mesh", REFINE_STYLES['btn_small'])

        # ---------- limits ----------
        @app.callback(
            Output('store-refresh', 'data', allow_duplicate=True),
            Output('save-status', 'children', allow_duplicate=True),
            Input('btn-limits-apply', 'n_clicks'),
            State('limit-lower', 'value'), State('limit-upper', 'value'),
            State('limit-effort', 'value'), State('limit-velocity', 'value'),
            State('store-selected-joint', 'data'),
            State('store-object', 'data'),
            State('store-refresh', 'data'),
            prevent_initial_call=True,
        )
        def on_limits_apply(_n, lower, upper, effort, velocity, joint_name, obj_name, refresh):
            if not joint_name:
                raise PreventUpdate
            model = self._model(obj_name)
            try:
                self._push_undo(obj_name)
                model.set_limits(joint_name, _num(lower, 0.0), _num(upper, 0.0),
                                 effort=_num(effort), velocity=_num(velocity))
            except ValueError as exc:
                self._undo[obj_name].pop()
                return no_update, err(exc)
            warns = [w for w in model.warnings() if w.startswith(joint_name)]
            if warns:
                return (refresh or 0) + 1, html.Span(warns[0], style={'color': 'orange'})
            return (refresh or 0) + 1, ok(f"Limits of {joint_name} updated")

        # ---------- joint dynamics ----------
        @app.callback(
            Output('store-refresh', 'data', allow_duplicate=True),
            Output('save-status', 'children', allow_duplicate=True),
            Input('btn-dyn-apply', 'n_clicks'),
            State('dyn-damping', 'value'), State('dyn-friction', 'value'),
            State('store-selected-joint', 'data'),
            State('store-object', 'data'),
            State('store-refresh', 'data'),
            prevent_initial_call=True,
        )
        def on_dynamics_apply(_n, damping, friction, joint_name, obj_name, refresh):
            if not joint_name:
                raise PreventUpdate
            model = self._model(obj_name)
            try:
                self._push_undo(obj_name)
                model.set_joint_dynamics(joint_name, _num(damping), _num(friction))
            except ValueError as exc:
                self._undo[obj_name].pop()
                return no_update, err(exc)
            return (refresh or 0) + 1, ok(f"Dynamics of {joint_name} updated")

        # ---------- per-part physics ----------
        @app.callback(
            Output('store-refresh', 'data', allow_duplicate=True),
            Output('save-status', 'children', allow_duplicate=True),
            Input('btn-parts-apply', 'n_clicks'),
            State({'type': 'part-mass', 'link': ALL}, 'value'),
            State({'type': 'part-mass', 'link': ALL}, 'id'),
            State({'type': 'part-friction', 'link': ALL}, 'value'),
            State({'type': 'part-damping', 'link': ALL}, 'value'),
            State('store-object', 'data'),
            State('store-refresh', 'data'),
            prevent_initial_call=True,
        )
        def on_parts_apply(_n, masses, mass_ids, frictions, dampings, obj_name, refresh):
            self._push_undo(obj_name)
            try:
                self._apply_part_edits(
                    obj_name, [mass_id['link'] for mass_id in mass_ids],
                    masses, frictions, dampings)
            except ValueError as exc:
                # The loop may have applied earlier rows already — roll the
                # model back to the pre-apply snapshot instead of dropping it.
                self._model(obj_name).restore(self._undo[obj_name].pop())
                return (refresh or 0) + 1, err(exc)
            return (refresh or 0) + 1, ok("Part physics updated")

        # ---------- q slider / test motion ----------
        @app.callback(
            Output('store-q', 'data', allow_duplicate=True),
            Output('q-readout', 'children'),
            Input('q-slider', 'value'),
            State('store-selected-joint', 'data'),
            State('store-object', 'data'),
            State('store-q', 'data'),
            prevent_initial_call=True,
        )
        def on_q_change(value, joint_name, obj_name, q_json):
            if not joint_name or value is None:
                raise PreventUpdate
            model = self._model(obj_name)
            q_values = json.loads(q_json or "{}")
            q_values[joint_name] = float(value)
            joint = model.joints.get(joint_name)
            if joint is not None and joint.joint_type == "revolute":
                readout = f"q = {value:.4f} rad ({math.degrees(value):.1f}°)"
            else:
                readout = f"q = {value:.4f}"
            return json.dumps(q_values), readout

        @app.callback(
            Output('q-slider', 'value', allow_duplicate=True),
            Input('btn-q-zero', 'n_clicks'),
            Input('btn-q-lower', 'n_clicks'),
            Input('btn-q-upper', 'n_clicks'),
            State('store-selected-joint', 'data'),
            State('store-object', 'data'),
            prevent_initial_call=True,
        )
        def on_q_buttons(_z, _l, _u, joint_name, obj_name):
            if not joint_name:
                raise PreventUpdate
            trigger = callback_context.triggered[0]['prop_id'].split('.')[0]
            lo, hi, disabled = self._slider_bounds(obj_name, joint_name)
            if disabled:
                raise PreventUpdate
            if trigger == 'btn-q-lower':
                return lo
            if trigger == 'btn-q-upper':
                return hi
            return min(max(0.0, lo), hi)

        @app.callback(
            Output('animate-interval', 'disabled'),
            Input('chk-animate', 'value'),
        )
        def on_animate_toggle(value):
            return 'animate' not in (value or [])

        @app.callback(
            Output('q-slider', 'value', allow_duplicate=True),
            Input('animate-interval', 'n_intervals'),
            State('q-slider', 'value'),
            State('store-selected-joint', 'data'),
            State('store-object', 'data'),
            prevent_initial_call=True,
        )
        def on_animate_tick(_n, value, joint_name, obj_name):
            # Bounds come from the model (not the slider's possibly-stale
            # min/max State): a tick racing an object switch must not write an
            # out-of-range q, and fixed joints must not animate at all.
            if not joint_name or value is None:
                raise PreventUpdate
            model = self._model(obj_name)
            if joint_name not in model.joints:
                raise PreventUpdate
            lo, hi, disabled = self._slider_bounds(obj_name, joint_name)
            if disabled or hi <= lo:
                raise PreventUpdate
            value = min(max(float(value), lo), hi)
            step = (hi - lo) / 40.0
            direction = self._anim_dir.get(joint_name, 1.0)
            new_value = value + direction * step
            if new_value >= hi:
                new_value, direction = hi, -1.0
            elif new_value <= lo:
                new_value, direction = lo, 1.0
            self._anim_dir[joint_name] = direction
            return new_value

        # ---------- figure ----------
        @app.callback(
            Output('viewport', 'figure'),
            Input('store-refresh', 'data'),
            Input('store-q', 'data'),
            Input('store-selected-joint', 'data'),
            Input('chk-ghosts', 'value'),
            Input('color-mode', 'value'),
            State('store-object', 'data'),
        )
        def update_figure(_refresh, q_json, joint_name, ghosts, color_mode, obj_name):
            q_values = json.loads(q_json or "{}")
            return self._builder(obj_name).build(
                q_values,
                selected_joint=joint_name,
                show_ghosts='ghosts' in (ghosts or []),
                color_mode=color_mode or 'segmented',
            )

        # ---------- undo / reset ----------
        @app.callback(
            Output('store-refresh', 'data', allow_duplicate=True),
            Output('save-status', 'children', allow_duplicate=True),
            Output('parts-table', 'children', allow_duplicate=True),
            Output('joint-dropdown', 'options', allow_duplicate=True),
            Input('btn-undo', 'n_clicks'),
            Input('btn-reset-joint', 'n_clicks'),
            Input('btn-reset-all', 'n_clicks'),
            State('store-selected-joint', 'data'),
            State('store-object', 'data'),
            State('store-refresh', 'data'),
            prevent_initial_call=True,
        )
        def on_undo_reset(_u, _rj, _ra, joint_name, obj_name, refresh):
            trigger = callback_context.triggered[0]['prop_id'].split('.')[0]
            model = self._model(obj_name)
            if trigger == 'btn-undo':
                stack = self._undo[obj_name]
                if not stack:
                    return (no_update, html.Span("Nothing to undo", style={'color': 'orange'}),
                            no_update, no_update)
                model.restore(stack.pop())
                msg = ok("Undid last edit")
            elif trigger == 'btn-reset-joint':
                if not joint_name:
                    raise PreventUpdate
                self._push_undo(obj_name)
                model.reset_joint(joint_name)
                msg = ok(f"Reset {joint_name} to the loaded state")
            else:
                self._push_undo(obj_name)
                model.reset_all()
                msg = ok("Reset all joints to the loaded state")
            # Dropdown labels embed the joint type, which undo/reset may revert.
            return ((refresh or 0) + 1, msg, self._parts_table(obj_name),
                    self._joint_options(obj_name))

        # ---------- save / done / cancel ----------
        @app.callback(
            Output('store-refresh', 'data', allow_duplicate=True),
            Output('save-status', 'children', allow_duplicate=True),
            Input('btn-save', 'n_clicks'),
            State('store-object', 'data'),
            State('store-refresh', 'data'),
            prevent_initial_call=True,
        )
        def on_save(_n, obj_name, refresh):
            model = self._model(obj_name)
            try:
                summary = model.save()
            except ValueError as exc:
                return no_update, err(exc)
            return ((refresh or 0) + 1,
                    ok(f"Saved refinement v{summary['version']} "
                       f"({len(summary['changed_joints'])} joint(s) changed)"))

        @app.callback(
            Output('store-refresh', 'data', allow_duplicate=True),
            Output('save-status', 'children', allow_duplicate=True),
            Input('btn-done', 'n_clicks'),
            State('store-refresh', 'data'),
            prevent_initial_call=True,
        )
        def on_done(_n, refresh):
            summaries = {}
            for name, model in self.models.items():
                if model.dirty:
                    try:
                        summaries[name] = model.save()
                    except ValueError as exc:
                        return no_update, err(f"{name}: {exc}")
            self.result_holder['result'] = {'saved': summaries, 'cancelled': False}
            self.result_holder['done'] = True
            return ((refresh or 0) + 1,
                    ok("Saved. You can close this tab and return to the terminal."))

        @app.callback(
            Output('save-status', 'children', allow_duplicate=True),
            Input('btn-cancel', 'n_clicks'),
            prevent_initial_call=True,
        )
        def on_cancel(_n):
            self.result_holder['result'] = {'saved': {}, 'cancelled': True}
            self.result_holder['done'] = True
            return html.Span("Cancelled — unsaved edits discarded. You can close this tab.",
                             style={'color': 'orange'})

    # ------------------------------------------------------------------
    # Server lifecycle (mirrors SegmentCorrectionApp.run)
    # ------------------------------------------------------------------

    def run(self, port: int = 8060, open_browser: bool = True) -> dict:
        """Run the Dash app and return the result when the user is done."""
        import socket
        from werkzeug.serving import make_server

        def find_available_port(start_port, max_attempts=10):
            for offset in range(max_attempts):
                test_port = start_port + offset
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('127.0.0.1', test_port))
                        return test_port
                except OSError:
                    continue
            raise RuntimeError(
                f"Could not find available port in range {start_port}-{start_port + max_attempts}")

        actual_port = find_available_port(port)
        if actual_port != port:
            print(f"Port {port} in use, using port {actual_port} instead")

        if open_browser:
            def open_tab():
                time.sleep(1.5)
                webbrowser.open_new(f"http://localhost:{actual_port}")

            browser_thread = threading.Thread(target=open_tab)
            browser_thread.daemon = True
            browser_thread.start()

        server = make_server('127.0.0.1', actual_port, self.app.server, threaded=True)
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        print(f"\n{'=' * 60}")
        print(f"Articulation Refinement UI started at http://localhost:{actual_port}")
        print(f"{'=' * 60}")
        print(f"\nObjects: {list(self.models.keys())}")
        print("Refine joint axes, pivots, limits and dynamics; use the test")
        print("slider to preview motion. 'Save' writes results/mobility.urdf")
        print("(originals kept as mobility_original.urdf + versioned copies)")
        print("and physics_overrides.json.")
        print("\nClick 'Save & Finish' when done, or 'Cancel' to discard unsaved edits.")
        print(f"{'=' * 60}\n")

        while not self.result_holder['done']:
            time.sleep(0.5)

        print("\nShutting down server...")
        server.shutdown()

        result = self.result_holder['result'] or {'saved': {}, 'cancelled': True}
        if result.get('cancelled'):
            print("Articulation refinement cancelled (unsaved edits discarded).")
        else:
            print(f"Articulation refinement complete! "
                  f"Saved {len(result['saved'])} object(s) with unsaved edits.")
        return result


def interactive_articulation_refinement(
    results_dirs: dict,
    port: int = 8060,
    open_browser: bool = True,
    max_faces_per_link: int = 40000,
    verbose: bool = False,
) -> dict:
    """Launch the refinement UI over ``{object_name: results_dir}``.

    Each results_dir must contain mobility.urdf and meshes/. Returns
    ``{'saved': {object_name: save summary}, 'cancelled': bool}``.
    """
    models = {}
    for name, results_dir in results_dirs.items():
        try:
            models[name] = ArticulationModel(results_dir, max_faces_per_link=max_faces_per_link)
        except FileNotFoundError as exc:
            print(f"Skipping {name}: {exc}")
    if not models:
        raise RuntimeError("No refinable articulation results found.")
    app = ArticulationRefinementApp(models, verbose=verbose)
    return app.run(port=port, open_browser=open_browser)
