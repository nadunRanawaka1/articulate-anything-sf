"""Plotly figure construction for the articulation refinement UI."""

import numpy as np
import plotly.graph_objects as go

from postprocess_segmentation.styles import generate_part_colors

from .urdf_model import ArticulationModel, make_transform, rotation_about_axis

GHOST_COLOR = "rgb(120,144,180)"
AXIS_COLOR = "#d62728"
ARC_COLOR = "#ff7f0e"
PIVOT_COLOR = "#111111"
MESH_LIGHTING = dict(ambient=0.4, diffuse=0.8, specular=0.3, roughness=0.5, fresnel=0.2)
MESH_LIGHTPOSITION = dict(x=1000, y=1000, z=1000)


def _transform_points(points: np.ndarray, tf: np.ndarray) -> np.ndarray:
    return points @ tf[:3, :3].T + tf[:3, 3]


class ArticulationFigureBuilder:
    """Builds the 3D scene: FK-posed link meshes, joint gizmos, limit ghosts."""

    def __init__(self, model: ArticulationModel):
        self.model = model
        self.link_colors = generate_part_colors(model.geometry_links())
        self.radius = model.bounding_radius()

    def _link_visual_transform(self, link_name: str, q_values: dict) -> np.ndarray:
        link = self.model.links[link_name]
        tf = self.model.link_world_transform(link_name, q_values)
        if link.geoms:
            first = link.geoms[0]
            tf = tf @ make_transform(first.xyz, first.rpy)
        return tf

    def _mesh_trace(self, link_name: str, q_values: dict, color_mode: str,
                    opacity: float = 1.0, ghost: bool = False, name_suffix: str = "") -> go.Mesh3d | None:
        mesh = self.model.link_mesh(link_name)
        if mesh is None or len(mesh["vertices"]) == 0:
            return None
        tf = self._link_visual_transform(link_name, q_values)
        verts = _transform_points(mesh["vertices"], tf)
        faces = mesh["faces"]

        kwargs = dict(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            flatshading=False,
            lighting=MESH_LIGHTING,
            lightposition=MESH_LIGHTPOSITION,
            opacity=opacity,
            name=f"{link_name}{name_suffix}",
        )
        if ghost:
            kwargs.update(color=GHOST_COLOR, hoverinfo="skip")
        else:
            parent_joint = self.model._child_to_joint.get(link_name, "")
            kwargs.update(
                customdata=[link_name] * len(verts),
                hovertemplate=(
                    f"<b>{link_name}</b><br>joint: {parent_joint}"
                    "<br>x=%{x:.4f} y=%{y:.4f} z=%{z:.4f}<extra></extra>"
                ),
            )
            if color_mode == "textured" and mesh["vertex_colors"] is not None:
                kwargs.update(vertexcolor=mesh["vertex_colors"])
            else:
                kwargs.update(color=self.link_colors.get(link_name, "rgb(150,150,150)"))
        return go.Mesh3d(**kwargs)

    def _axis_gizmo_traces(self, joint_name: str, q_values: dict) -> list:
        joint = self.model.joints[joint_name]
        if not joint.is_movable:
            return []
        pivot, axis_world, _ = self.model.joint_world_frame(joint_name, q_values)
        length = max(0.6 * self.radius, 0.05)
        start = pivot - axis_world * length
        end = pivot + axis_world * length

        traces = [
            go.Scatter3d(
                x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
                mode="lines", line=dict(color=AXIS_COLOR, width=7),
                hoverinfo="skip", name="axis",
            ),
            go.Cone(
                x=[end[0]], y=[end[1]], z=[end[2]],
                u=[axis_world[0]], v=[axis_world[1]], w=[axis_world[2]],
                sizemode="absolute", sizeref=0.14 * length, anchor="tail",
                colorscale=[[0, AXIS_COLOR], [1, AXIS_COLOR]],
                showscale=False, hoverinfo="skip", name="axis-dir",
            ),
            go.Scatter3d(
                x=[pivot[0]], y=[pivot[1]], z=[pivot[2]],
                mode="markers", marker=dict(size=7, color=PIVOT_COLOR, symbol="diamond"),
                # Not hoverable: in pick-pivot mode a hoverable marker would
                # intercept clicks aimed at mesh points near the current pivot.
                hoverinfo="skip",
                name="pivot",
            ),
        ]
        traces.extend(self._range_traces(joint_name, q_values, pivot, axis_world))
        return traces

    def _child_rest_centroid(self, joint_name: str, q_values: dict) -> np.ndarray | None:
        """World centroid of the child mesh with this joint at q=0."""
        child = self.model.joints[joint_name].child
        mesh = self.model.link_mesh(child)
        if mesh is None:
            return None
        q_rest = dict(q_values)
        q_rest[joint_name] = 0.0
        tf = self._link_visual_transform(child, q_rest)
        return tf[:3, :3] @ mesh["centroid"] + tf[:3, 3]

    def _range_traces(self, joint_name: str, q_values: dict,
                      pivot: np.ndarray, axis_world: np.ndarray) -> list:
        """Sweep indicator: an arc (revolute) or travel segment (prismatic)
        through the child part's rest centroid, from lower to upper limit."""
        joint = self.model.joints[joint_name]
        centroid = self._child_rest_centroid(joint_name, q_values)
        if centroid is None:
            return []
        limit = joint.limit
        traces = []
        if joint.joint_type in ("revolute", "continuous"):
            lower, upper = (limit["lower"], limit["upper"]) if limit else (-np.pi, np.pi)
            thetas = np.linspace(lower, upper, 40)
            rel = centroid - pivot
            pts = np.array([
                pivot + rotation_about_axis(axis_world, t)[:3, :3] @ rel for t in thetas
            ])
            if np.linalg.norm(np.cross(rel, axis_world)) < 1e-6:
                return []  # centroid on the axis: no meaningful arc
            traces.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
                line=dict(color=ARC_COLOR, width=5, dash="dot"),
                hoverinfo="skip", name="range",
            ))
            ends, labels = [pts[0], pts[-1]], [f"lower {lower:.3f}", f"upper {upper:.3f}"]
        else:  # prismatic
            lower, upper = (limit["lower"], limit["upper"]) if limit else (0.0, 0.1)
            p0 = centroid + axis_world * lower
            p1 = centroid + axis_world * upper
            traces.append(go.Scatter3d(
                x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]], mode="lines",
                line=dict(color=ARC_COLOR, width=5, dash="dot"),
                hoverinfo="skip", name="range",
            ))
            ends, labels = [p0, p1], [f"lower {lower:.3f}", f"upper {upper:.3f}"]
        ends = np.array(ends)
        traces.append(go.Scatter3d(
            x=ends[:, 0], y=ends[:, 1], z=ends[:, 2],
            mode="markers+text", text=labels, textposition="top center",
            textfont=dict(size=10, color=ARC_COLOR),
            marker=dict(size=4, color=ARC_COLOR),
            hoverinfo="skip", name="range-ends",
        ))
        return traces

    def _world_triad_traces(self) -> list:
        size = 0.25 * self.radius
        traces = []
        for axis_vec, color, label in (
            ((size, 0, 0), "red", "X"),
            ((0, size, 0), "green", "Y"),
            ((0, 0, size), "blue", "Z"),
        ):
            traces.append(go.Scatter3d(
                x=[0, axis_vec[0]], y=[0, axis_vec[1]], z=[0, axis_vec[2]],
                mode="lines+text", text=["", label],
                textfont=dict(size=10, color=color),
                line=dict(color=color, width=3),
                hoverinfo="skip", name=f"world-{label}",
            ))
        return traces

    def build(self, q_values: dict, selected_joint: str | None = None,
              show_ghosts: bool = True, color_mode: str = "segmented") -> go.Figure:
        traces = []
        selected_links = set()
        if selected_joint and selected_joint in self.model.joints:
            selected_links = set(self.model.descendant_links(selected_joint))

        for link_name in self.model.geometry_links():
            if selected_links and link_name not in selected_links:
                opacity = 0.55
            else:
                opacity = 1.0
            trace = self._mesh_trace(link_name, q_values, color_mode, opacity=opacity)
            if trace is not None:
                traces.append(trace)

        if selected_joint and selected_joint in self.model.joints:
            joint = self.model.joints[selected_joint]
            if show_ghosts and joint.is_movable:
                if joint.limit is not None:
                    ghost_qs = [joint.limit["lower"], joint.limit["upper"]]
                elif joint.joint_type == "continuous":
                    ghost_qs = [-np.pi / 2, np.pi / 2]
                else:
                    ghost_qs = []
                for ghost_q in ghost_qs:
                    if abs(ghost_q - float(q_values.get(selected_joint, 0.0))) < 1e-9:
                        continue
                    q_ghost = dict(q_values)
                    q_ghost[selected_joint] = ghost_q
                    for link_name in selected_links:
                        trace = self._mesh_trace(
                            link_name, q_ghost, color_mode,
                            opacity=0.22, ghost=True, name_suffix=f"@{ghost_q:.2f}",
                        )
                        if trace is not None:
                            traces.append(trace)
            traces.extend(self._axis_gizmo_traces(selected_joint, q_values))

        traces.extend(self._world_triad_traces())

        fig = go.Figure(data=traces)
        fig.update_layout(
            scene=dict(
                aspectmode="data",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
                xaxis=dict(showgrid=True, gridcolor="lightgray", showbackground=True),
                yaxis=dict(showgrid=True, gridcolor="lightgray", showbackground=True),
                zaxis=dict(showgrid=True, gridcolor="lightgray", showbackground=True),
            ),
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=640,
            uirevision="constant",
        )
        return fig
