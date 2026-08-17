"""Tests for the segmentation-UI face-selection tools (islands).

Covers the pure-geometry engine (segment_islands, island_containing_face,
stray_islands) on a synthetic mesh with a main body and far-away debris
islands — the exact "stray faces" scenario the tools exist for — plus
app-level wiring (selection -> split -> reassignment).

Run with pytest:  pytest tests/test_segment_face_selection.py
"""

import os
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simfoundry"))

from postprocess_segmentation.geometry import (  # noqa: E402
    island_containing_face,
    merge_segments,
    precompute_segment_geometry,
    segment_islands,
    split_segment_by_faces,
    stray_islands,
)


def make_debris_mesh():
    """A main box at the origin plus two small far-away debris islands.

    Returns (mesh, face2label) with everything labelled segment 0 — the
    situation in the user's screenshot: one segment whose faces include
    disconnected junk floating far from the body.
    """
    body = trimesh.creation.box(extents=(1.0, 1.0, 1.0))  # 12 faces
    debris1 = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
    debris1.apply_translation([3.0, -2.0, 0.0])  # 12 faces, far away
    # A single lonely triangle even further out
    debris2 = trimesh.Trimesh(
        vertices=[[5.0, 5.0, 0.0], [5.1, 5.0, 0.0], [5.0, 5.1, 0.0]],
        faces=[[0, 1, 2]], process=False)
    mesh = trimesh.util.concatenate([body, debris1, debris2])
    face2label = np.zeros(len(mesh.faces), dtype=int)
    return mesh, face2label


@pytest.fixture
def debris():
    return make_debris_mesh()


# ---------------------------------------------------------------------------
# Geometry engine
# ---------------------------------------------------------------------------

def test_segment_islands_topology_only(debris):
    mesh, face2label = debris
    islands = segment_islands(mesh, face2label, 0)
    assert [len(i) for i in islands] == [12, 12, 1]  # largest-first
    # Every face accounted for exactly once
    all_faces = np.concatenate(islands)
    assert sorted(all_faces) == list(range(len(mesh.faces)))


def test_segment_islands_respects_labels(debris):
    mesh, face2label = debris
    face2label[:12] = 7  # body belongs to another segment now
    islands = segment_islands(mesh, face2label, 0)
    assert [len(i) for i in islands] == [12, 1]
    assert segment_islands(mesh, face2label, 99) == []


def test_island_containing_face(debris):
    mesh, face2label = debris
    lone_face = len(mesh.faces) - 1
    island = island_containing_face(mesh, face2label, 0, lone_face)
    assert list(island) == [lone_face]
    body_island = island_containing_face(mesh, face2label, 0, 0)
    assert len(body_island) == 12 and 0 in body_island
    # Seed outside the segment -> empty
    face2label2 = face2label.copy()
    face2label2[lone_face] = 3
    assert len(island_containing_face(mesh, face2label2, 0, lone_face)) == 0


def test_stray_islands_selects_all_but_largest(debris):
    mesh, face2label = debris
    stray = stray_islands(mesh, face2label, 0)
    assert len(stray) == 13  # debris cube (12) + lone triangle (1)
    assert set(stray) == set(range(12, 25))
    # Max size filter keeps big secondary islands as real geometry
    only_tiny = stray_islands(mesh, face2label, 0, max_faces=5)
    assert list(only_tiny) == [len(mesh.faces) - 1]
    # Single-island segments have no strays
    face2label3 = face2label.copy()
    face2label3[12:] = 1
    assert len(stray_islands(mesh, face2label3, 0)) == 0


def test_split_selected_faces_roundtrip(debris):
    """The end-to-end data flow of 'Select All Stray Islands' -> split."""
    mesh, face2label = debris
    label2face_mask = np.zeros((1, len(mesh.faces)), dtype=bool)
    label2face_mask[0, :] = True

    stray = stray_islands(mesh, face2label, 0)
    face2label, label2face_mask, new_id = split_segment_by_faces(
        face2label, label2face_mask, 0, stray)
    assert new_id == 1
    assert set(np.where(face2label == 1)[0]) == set(stray)
    assert not label2face_mask[0, stray].any()
    assert label2face_mask[1, stray].all()
    # After the split, the source segment is one clean island
    assert len(segment_islands(mesh, face2label, 0)) == 1


def test_split_after_merge_keeps_mask_row_consistent(debris):
    """Regression: merging away the highest segment id used to make the next
    split write its mask row at the wrong index (id != row), silently losing
    the split faces at export time."""
    mesh, face2label = debris
    # Segments: 0 = body, 1 = debris cube, 2 = lone triangle
    face2label[12:24] = 1
    face2label[24] = 2
    label2face_mask = np.zeros((3, len(mesh.faces)), dtype=bool)
    for seg in range(3):
        label2face_mask[seg] = face2label == seg

    # Merge [1, 2]: id 2 vacated, max(face2label) drops to 1, mask keeps 3 rows.
    face2label, label2face_mask, _ = merge_segments(face2label, label2face_mask, [1, 2])
    assert np.max(face2label) == 1 and label2face_mask.shape[0] == 3

    # Split two faces off segment 0: the new id must index its own mask row.
    face2label, label2face_mask, new_id = split_segment_by_faces(
        face2label, label2face_mask, 0, np.array([0, 1]))
    assert new_id >= 3  # never reuses the vacated id
    assert new_id < label2face_mask.shape[0]
    np.testing.assert_array_equal(
        label2face_mask[new_id], face2label == new_id)
    # Every live segment's mask row matches face2label exactly.
    for seg in np.unique(face2label):
        np.testing.assert_array_equal(label2face_mask[seg], face2label == seg)


def make_seam_mesh():
    """Two triangles sharing an edge, but with duplicated seam vertices that
    differ only in UV coordinates — a texture seam. Physically one island."""
    vertices = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],   # tri A
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],   # tri B (dup seam verts)
    ])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    uv = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
                   [0.9, 0.0], [1.0, 0.1], [1.0, 1.0]])  # differs at the seam
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
    return mesh


def test_islands_bridge_texture_seams():
    """Regression: default merge_vertices keeps UV-seam duplicates apart, so
    physically connected geometry fragmented into fake 'islands'."""
    mesh = make_seam_mesh()
    face2label = np.zeros(2, dtype=int)
    islands = segment_islands(mesh, face2label, 0)
    assert len(islands) == 1
    assert len(stray_islands(mesh, face2label, 0)) == 0
    assert len(island_containing_face(mesh, face2label, 0, 0)) == 2


def test_face_highlight_uses_cell_intensity(debris):
    """Regression: without intensitymode='cell' the black selected-face
    highlight was interpreted per-vertex and colored the wrong geometry."""
    pytest.importorskip("plotly")
    from postprocess_segmentation.visualization import SegmentFigureBuilder
    from postprocess_segmentation.styles import generate_part_colors

    mesh, face2label = debris
    seg_data, scale, _ = precompute_segment_geometry(mesh, face2label, [0])
    builder = SegmentFigureBuilder(
        segment_data=seg_data,
        part_colors=generate_part_colors(["body"]),
        scene_scale=scale,
        get_segment_to_part=lambda: {0: "body"})
    fig = builder.build(selected_faces=[12, 13], face2label=face2label, mesh=mesh)
    trace = fig.data[0]
    assert trace.intensitymode == 'cell'
    assert len(trace.intensity) == len(mesh.faces)  # one per face
    assert (trace.cmin, trace.cmax) == (0, 1)
    assert list(trace.intensity[12:14]) == [0, 0]


# ---------------------------------------------------------------------------
# App-level wiring (no server)
# ---------------------------------------------------------------------------

def make_app(mesh, face2label):
    pytest.importorskip("dash")
    pytest.importorskip("plotly")
    from postprocess_segmentation.interactive_ui import SegmentCorrectionApp

    label2face_mask = np.zeros((1, len(mesh.faces)), dtype=bool)
    label2face_mask[0, :] = True
    return SegmentCorrectionApp(
        mesh=mesh,
        face2label=face2label,
        label2face_mask=label2face_mask,
        part_segment_dict={"trash_can_body": [0], "lid": []},
        parts_list=["trash_can_body", "lid"],
    )


def test_app_constructs_with_new_controls(debris):
    mesh, face2label = debris
    app = make_app(mesh, face2label)
    layout_ids = []

    def collect(component):
        comp_id = getattr(component, 'id', None)
        if isinstance(comp_id, str):
            layout_ids.append(comp_id)
        for child in getattr(component, 'children', None) or []:
            if hasattr(child, 'children') or hasattr(child, 'id'):
                collect(child)

    # The tab layouts are swapped in dynamically; verify the callbacks exist.
    callback_ids = "\n".join(app.app.callback_map.keys())
    for needle in ("face-select-store", "parts-selected-store",
                   "island-mode-store", "parts-viewer"):
        assert needle in callback_ids, f"no callback wired for {needle}"


def test_camera_persistence(debris):
    """Regression: rebuilt figures must carry the user's last camera (from
    relayoutData), or plotly snaps to the default view on the second
    orbit/rebuild cycle (its uirevision GUI-edit records are consumed by the
    first rebuild and re-baselined by the next orbit)."""
    pytest.importorskip("plotly")
    mesh, face2label = debris
    app = make_app(mesh, face2label)

    user_cam = {"eye": {"x": 0.2, "y": 2.5, "z": 0.7},
                "up": {"x": 0, "y": 0, "z": 1},
                "center": {"x": 0, "y": 0, "z": 0}}
    # No orbit yet: figures keep the default camera.
    fig = app._apply_camera(app.figure_builder.build(explosion_factor=0.3))
    assert fig.layout.scene.camera.eye.x == 1.5

    # Orbit happened: relayoutData carries scene.camera; every later figure
    # (including other viewers and tab-remount initials) ships it.
    app._remember_camera({"scene.camera": user_cam})
    fig = app._apply_camera(app.figure_builder.build(explosion_factor=0.3))
    assert fig.layout.scene.camera.eye.y == 2.5
    part_fig = app._apply_camera(app.figure_builder.build_part_view(
        part_name="trash_can_body", current_assignment=app.current_assignment))
    assert part_fig.layout.scene.camera.eye.y == 2.5

    # Payloads without a camera (e.g. dragmode changes, None on remount) are ignored.
    app._remember_camera(None)
    app._remember_camera({"scene.dragmode": "pan"})
    fig = app._apply_camera(app.figure_builder.build(explosion_factor=0.3))
    assert fig.layout.scene.camera.eye.y == 2.5

    # The three viewer callbacks read their graph's relayoutData as State.
    for viewer in ("overview-viewer", "segments-viewer", "parts-viewer"):
        entry = next(v for k, v in app.app.callback_map.items()
                     if k.startswith(f"{viewer}.figure") or f"{viewer}.figure" in k)
        assert {"id": viewer, "property": "relayoutData"} in entry["state"], viewer


def test_app_reassign_segments(debris):
    mesh, face2label = debris
    app = make_app(mesh, face2label)
    moved = app._reassign_segments([0], "lid")
    assert moved == [(0, "trash_can_body")]
    assert app.current_assignment["lid"] == [0]
    assert app.current_assignment["trash_can_body"] == []
    # Re-moving to the same part is a no-op append
    app._reassign_segments([0], "lid")
    assert app.current_assignment["lid"] == [0]


def test_part_view_segment_order_matches_click_mapping(debris):
    """part_view_segments must iterate exactly like build_part_view so
    clickData curveNumber -> segment mapping stays correct."""
    pytest.importorskip("plotly")
    mesh, face2label = debris
    # Split debris into segment 1 so the part has two segments
    label2face_mask = np.zeros((1, len(mesh.faces)), dtype=bool)
    label2face_mask[0, :] = True
    stray = stray_islands(mesh, face2label, 0)
    face2label, label2face_mask, _ = split_segment_by_faces(
        face2label, label2face_mask, 0, stray)

    from postprocess_segmentation.interactive_ui import SegmentCorrectionApp
    app = SegmentCorrectionApp(
        mesh=mesh, face2label=face2label, label2face_mask=label2face_mask,
        part_segment_dict={"trash_can_body": [0, 1], "lid": []},
        parts_list=["trash_can_body", "lid"])

    assignment = app.current_assignment
    shown = app.figure_builder.part_view_segments("trash_can_body", assignment)
    fig = app.figure_builder.build_part_view(
        part_name="trash_can_body", current_assignment=assignment)
    assert len(fig.data) == len(shown)
    for trace, seg_id in zip(fig.data, shown):
        assert trace.name == f"Seg {seg_id}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
