"""
Module: plotting_utils
======================
Contains specialized plotting functions for the Clustering/ASD analysis pipeline.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap
from typing import List, Dict, Optional

# --- GLOBAL CONSTANTS ---
# Standard-Mapping für dein Projekt.
# Wird automatisch genutzt, wenn keine eigene Map übergeben wird.
DEFAULT_PARAM_MAP = {
    'clus_threshold': 'Th',
    'clus_linkage': 'Link',
    'clus_metric': 'Metric',
    'segm_max_chunk_size': 'MaxSize',
    'segm_min_chunk_size': 'MinSize',
    'segm_min_duration_on': 'MinOn',
    'segm_min_duration_off': 'MinOff',
    'segm_onset': 'Onset',
    'segm_offset': 'Offset'
}

def plot_session_heatmap(df: pd.DataFrame,
                         param_map: Optional[Dict[str, str]] = None,
                         exclude_params: Optional[List[str]] = None,
                         show_param_names: bool = False,
                         metric_col: str = "AVG_SPEAKER_F1",
                         title: str = "Session Success Heatmap",
                         verbose: bool = True) -> None:
    """
    Generates a heatmap visualizing the success (Perfect/Failed) of sessions.

    Args:
        df (pd.DataFrame): Data loaded from overview csv.
        param_map (Dict): Mapping for column names. If None, uses DEFAULT_PARAM_MAP.
        exclude_params (List): Columns to hide.
        show_param_names (bool): True for 'Th=0.7', False for '0.7'.
        metric_col (str): Column to sort by.
        title (str): Plot title.
        verbose (bool): Print debug info.
    """
    # 1. Defaults
    if param_map is None:
        if verbose: print("Using DEFAULT_PARAM_MAP from plotting_utils.")
        param_map = DEFAULT_PARAM_MAP
    if exclude_params is None:
        exclude_params = []

    # 2. Check Metric
    if metric_col not in df.columns:
        raise ValueError(f"Metric column '{metric_col}' not found in DataFrame.")

    # 3. Sort Data
    df_plot = df.sort_values(by=metric_col, ascending=False).copy()

    # 4. Identify Parameter Columns
    all_cols = df_plot.columns
    param_cols = [c for c in all_cols if c.startswith('segm_') or c.startswith('clus_')]
    param_cols = sorted([c for c in param_cols if c not in exclude_params])

    if verbose:
        print(f"Plotting parameters: {param_cols}")

    # 5. Helper to Create Row Labels (Simple Join, No Padding)
    def _create_label(row):
        # Metric first
        val_metric = row[metric_col]
        # Format metric simply
        label_parts = [f"{val_metric:.3f}" if isinstance(val_metric, (float, int)) else str(val_metric)]

        for col in param_cols:
            val = row[col]

            # Mapped Name
            display_name = param_map.get(col, col.split('_', 1)[1] if '_' in col else col)

            if show_param_names:
                # Option A: "Th=0.7"
                label_parts.append(f"{display_name}={val}")
            else:
                # Option B: "0.7" (Value only)
                label_parts.append(f"{val}")

        return " | ".join(label_parts)

    df_plot['label'] = df_plot.apply(_create_label, axis=1)
    df_plot = df_plot.set_index('label')

    # 6. Construct Header String (Simple Join)
    if show_param_names:
        y_axis_title = "Experiment Parameters"
    else:
        # Build the header string: "F1 | Th | Link | ..."
        short_metric = "F1" if "F1" in metric_col else metric_col
        header_parts = [short_metric]

        for col in param_cols:
            name = param_map.get(col, col.split('_', 1)[1] if '_' in col else col)
            header_parts.append(name)
        y_axis_title = " | ".join(header_parts)

    # 7. Filter Heatmap Data
    heatmap_data = df_plot.filter(regex='_eva$')
    if heatmap_data.empty:
        print("Warning: No columns ending with '_eva' found.")
        return
    heatmap_data.columns = [c.replace('_eva', '') for c in heatmap_data.columns]

    # 8. Plotting
    plt.figure(figsize=(22, max(6, len(df_plot) * 0.5)))
    my_cmap = ListedColormap(['#ffcccc', '#ccffcc'])

    ax = sns.heatmap(
        heatmap_data,
        cmap=my_cmap,
        cbar=False,
        linewidths=1,
        linecolor='white',
        annot=False,
        square=False
    )

    plt.title(title, fontsize=16, pad=20)

    # Standard Matplotlib Y-Label (Vertical, Centered) - wie in deinem Code gewünscht
    plt.ylabel(y_axis_title, fontsize=12, fontweight='bold')

    # Rows are Monospace to keep numbers readable
    plt.yticks(rotation=0, fontfamily='monospace')
    plt.xlabel("Sessions", fontsize=12)
    plt.tight_layout()
    plt.show()
