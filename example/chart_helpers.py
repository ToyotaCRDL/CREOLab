"""
Chart style and helper functions for evaluation pipeline.
"""

import os
import json
import matplotlib.pyplot as plt


# ============================================================================
# Chart Style Constants
# ============================================================================

# Chart text styles
CHART_TITLE_STYLE = {'fontsize': 14, 'fontweight': 'bold'}
CHART_LABEL_STYLE = {'fontsize': 12, 'fontweight': 'bold'}

# Grid style
CHART_GRID_STYLE = {'alpha': 0.3, 'axis': 'y'}

# Default deduction category style (fallback)
DEFAULT_DEDUCTION_STYLE = {
    'color': '#F4F4F4', 
    'pattern': '', 
    'edgecolor': '#666666', 
    'linewidth': 0.5
}

# Deduction bar style parameters
DEDUCTION_BAR_ALPHA = 0.9

# Reference line parameters
REFERENCE_LINE_STYLE = {'color': 'black', 'linestyle': '--', 'alpha': 0.5, 'linewidth': 1}
REFERENCE_LINE_TEXT_STYLE = {'va': 'center', 'ha': 'left', 'fontsize': 10}


# ============================================================================
# Chart Style Functions
# ============================================================================

def apply_chart_style(ax, title, ylabel='Points', xlabel=None, ylim=(0, 105), 
                      xticks=None, xticklabels=None, add_reference_line=False, reference_x_pos=None):
    """
    Apply consistent chart style settings.
    
    Args:
        ax: Matplotlib axis object
        title: Chart title
        ylabel: Y-axis label (default: 'Points')
        xlabel: X-axis label (optional)
        ylim: Y-axis limits as tuple (default: (0, 105))
        xticks: X-axis tick positions (optional)
        xticklabels: X-axis tick labels (optional)
        add_reference_line: Whether to add 100-point reference line (default: False)
        reference_x_pos: X position for reference line text (required if add_reference_line=True)
    """
    ax.set_title(title, **CHART_TITLE_STYLE, pad=20)
    ax.set_ylabel(ylabel, **CHART_LABEL_STYLE)
    if xlabel:
        ax.set_xlabel(xlabel, **CHART_LABEL_STYLE)
    ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    ax.grid(True, **CHART_GRID_STYLE)
    
    # Add 100-point reference line if requested
    if add_reference_line and reference_x_pos is not None:
        ax.axhline(y=100, **REFERENCE_LINE_STYLE)
        ax.text(reference_x_pos, 100, '100', **REFERENCE_LINE_TEXT_STYLE)


def save_chart_with_error_handling(save_path, message_prefix="Chart"):
    """Save chart with consistent error handling."""
    plt.tight_layout()
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{message_prefix} saved: {save_path}")
        return save_path
    except Exception as e:
        print(f"Error saving {message_prefix.lower()}: {e}")
        plt.close()
        return None


def save_chart_data_json(chart_image_path, data_dict):
    """
    Save chart data as JSON file alongside the chart image.
    
    Args:
        chart_image_path: Path to the chart image file (e.g., chart.png)
        data_dict: Dictionary containing chart data to save
    """
    if not chart_image_path or not data_dict:
        return None
    
    # Replace image extension with .json
    json_path = os.path.splitext(chart_image_path)[0] + '.json'
    
    try:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)
        print(f"Chart data saved: {json_path}")
        return json_path
    except Exception as e:
        print(f"Error saving chart data JSON: {e}")
        return None


# ============================================================================
# Deduction Category Styles
# ============================================================================

def get_rubric_category_order():
    """Get the standard order for rubric evaluation categories (6-1 reversed for proper chart display)."""
    return [
        "Ambiguous Terminology",             # 6. -5 points each (displayed at top)
        "Incorrect Terminology",             # 5. -10 points each
        "Incomplete Step Descriptions",      # 4. -5 points each
        "Unnecessary Additional Steps",      # 3. -8 points each
        "Incorrect Step Sequence",           # 2. -12 points each  
        "Critical Step Omissions"            # 1. -15 points each (displayed at bottom)
    ]


def get_category_visual_style():
    """Get visual styles (colors and patterns) for deduction categories that are colorblind-friendly and subtle."""
    category_order = get_rubric_category_order()
    
    # Medium pastel colors that are visible but won't compete with the red final score
    colors = [
        '#B8D4E8',  # Light blue-gray (Ambiguous Terminology)
        '#D0B8E8',  # Light purple (Incorrect Terminology) 
        '#B8E8B8',  # Light green (Incomplete Step Descriptions)
        '#E8D0B8',  # Light orange (Unnecessary Additional Steps)
        '#E8B8B8',  # Light pink (Incorrect Step Sequence)
        '#D4D4D4'   # Light gray (Critical Step Omissions)
    ]
    
    # Hatch patterns for colorblind accessibility
    patterns = [
        '...',      # Dots (Ambiguous Terminology)
        '///',      # Diagonal lines (Incorrect Terminology)
        '|||',      # Vertical lines (Incomplete Step Descriptions) 
        '---',      # Horizontal lines (Unnecessary Additional Steps)
        '+++',      # Plus signs (Incorrect Step Sequence)
        'xxx'       # X pattern (Critical Step Omissions)
    ]
    
    # Create mapping
    style_mapping = {}
    for i, category in enumerate(category_order):
        style_mapping[category] = {
            'color': colors[i],
            'pattern': patterns[i],
            'edgecolor': 'white',    # White pattern color
            'linewidth': 1.0         # Slightly thicker for better visibility
        }
    
    return style_mapping

