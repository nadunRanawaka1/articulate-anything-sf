"""Constants and style definitions for the segment correction UI."""

import colorsys

# Geometry constants
EXPLOSION_SCALE = 0.3
GOLDEN_RATIO = 0.618033988749895

# UI Styles
STYLES = {
    'container': {
        'fontFamily': 'Arial, sans-serif'
    },
    'header': {
        'textAlign': 'center',
        'marginBottom': '10px',
        'color': '#333'
    },
    'main_layout': {
        'display': 'flex',
        'padding': '20px',
        'backgroundColor': '#e8e8e8',
        'minHeight': '100vh'
    },
    'left_panel': {
        'flex': '2',
        'marginRight': '20px'
    },
    'right_panel': {
        'flex': '1',
        'minWidth': '300px',
        'padding': '15px',
        'backgroundColor': '#fff',
        'borderRadius': '10px',
        'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'
    },
    'graph': {
        'height': '600px'
    },
    'slider_container': {
        'padding': '15px',
        'backgroundColor': '#fff',
        'borderRadius': '5px',
        'marginTop': '10px'
    },
    'selected_display': {
        'padding': '10px',
        'backgroundColor': '#f0f0f0',
        'borderRadius': '5px',
        'marginBottom': '15px'
    },
    'assignments_container': {
        'maxHeight': '200px',
        'overflowY': 'auto',
        'padding': '10px',
        'backgroundColor': '#f9f9f9',
        'borderRadius': '5px'
    },
    'btn_reassign': {
        'width': '100%',
        'padding': '10px',
        'fontSize': '14px',
        'backgroundColor': '#4CAF50',
        'color': 'white',
        'border': 'none',
        'borderRadius': '5px',
        'cursor': 'pointer',
        'marginBottom': '10px'
    },
    'btn_done': {
        'width': '100%',
        'padding': '15px',
        'fontSize': '16px',
        'backgroundColor': '#2196F3',
        'color': 'white',
        'border': 'none',
        'borderRadius': '5px',
        'cursor': 'pointer',
        'marginBottom': '10px'
    },
    'btn_cancel': {
        'width': '100%',
        'padding': '10px',
        'fontSize': '14px',
        'backgroundColor': '#f44336',
        'color': 'white',
        'border': 'none',
        'borderRadius': '5px',
        'cursor': 'pointer'
    }
}


def generate_part_colors(parts_list: list) -> dict:
    """
    Generate distinct colors for each part using golden ratio spacing.
    
    Args:
        parts_list: List of part names
        
    Returns:
        Dict mapping part names to RGB color strings
    """
    colors = {'_unassigned': 'rgb(150,150,150)'}
    for i, part_name in enumerate(parts_list):
        hue = (i * GOLDEN_RATIO) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.9)
        colors[part_name] = f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'
    return colors
