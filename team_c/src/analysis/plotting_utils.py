"""
Module: plotting_utils
======================
Contains specialized plotting functions for the Clustering/ASD analysis pipeline.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Optional

# --- GLOBAL CONSTANTS ---
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

# --- HELPER (Intern) ---
def _prepare_labels(df_plot, param_cols, param_map, show_param_names, metric_col, col_widths):
    """Interne Hilfsfunktion, um die Y-Achsen-Labels zu bauen."""
    def _create_label(row):
        parts = []
        # Metric
        val_metric = row[metric_col]
        val_str = f"{val_metric:.3f}" if isinstance(val_metric, (float, int)) else str(val_metric)
        parts.append(f"{val_str:>{col_widths['__metric__']}}")

        for col in param_cols:
            val = row[col]
            val_str = str(val)
            width = col_widths[col]
            display_name = param_map.get(col, col.split('_', 1)[1] if '_' in col else col)

            if show_param_names:
                parts.append(f"{display_name}={val}")
            else:
                parts.append(f"{val_str:>{width}}")
        return " | ".join(parts)
    return df_plot.apply(_create_label, axis=1)

def _calculate_widths(df_plot, param_cols, param_map, metric_col):
    """Berechnet die Spaltenbreiten für das Layout."""
    col_widths = {}
    metric_vals = [f"{x:.3f}" if isinstance(x, (float, int)) else str(x) for x in df_plot[metric_col]]
    short_metric = "F1" if "F1" in metric_col else metric_col
    col_widths['__metric__'] = max([len(v) for v in metric_vals] + [len(short_metric)])

    for col in param_cols:
        vals = [str(x) for x in df_plot[col]]
        header_name = param_map.get(col, col.split('_', 1)[1] if '_' in col else col)
        col_widths[col] = max([len(v) for v in vals] + [len(header_name)])
    return col_widths

# --- HAUPTFUNKTION 1: Standard Heatmap (Richtig/Falsch) ---
def plot_session_heatmap(df: pd.DataFrame,
                         param_map: Optional[Dict[str, str]] = None,
                         exclude_params: Optional[List[str]] = None,
                         show_param_names: bool = False,
                         metric_col: str = "AVG_SPEAKER_F1",
                         title: str = "Session Success Heatmap",
                         row_height: float = 0.3,
                         verbose: bool = True) -> None:
    # (Dieser Code bleibt wie oben definiert - hier der Kürze halber ausgeblendet,
    #  aber du kannst den Code aus der vorherigen Antwort hier lassen)
    pass # Platzhalter - nutze den Code aus der vorherigen Antwort für diese Funktion!


# --- HAUPTFUNKTION 2: Confidence Heatmap (NEU!) ---
def plot_confidence_heatmap(df: pd.DataFrame,
                            param_map: Optional[Dict[str, str]] = None,
                            exclude_params: Optional[List[str]] = None,
                            show_param_names: bool = False,
                            metric_col: str = "AVG_SPEAKER_F1",
                            title: str = "Session Confidence Heatmap",
                            row_height: float = 0.3,
                            verbose: bool = True) -> None:
    """
    Erstellt eine Heatmap, die Erfolg UND Wahrscheinlichkeit kombiniert.

    Farblogik:
    - Hellgrün: Richtig + Hohe Confidence (Optimal)
    - Dunkelgrün: Richtig + Niedrige Confidence
    - Dunkelrot: Falsch + Niedrige Confidence
    - Hellrot: Falsch + Hohe Confidence (Kritischer Fehler)
    """
    # 1. Setup
    if param_map is None: param_map = DEFAULT_PARAM_MAP
    if exclude_params is None: exclude_params = []

    # 2. Sortieren
    df_plot = df.sort_values(by=metric_col, ascending=False).copy()

    # 3. Parameter finden & Breiten berechnen (für saubere Labels)
    all_cols = df_plot.columns
    param_cols = sorted([c for c in all_cols if (c.startswith('segm_') or c.startswith('clus_')) and c not in exclude_params])
    col_widths = _calculate_widths(df_plot, param_cols, param_map, metric_col)

    # 4. Labels erstellen
    df_plot['label'] = _prepare_labels(df_plot, param_cols, param_map, show_param_names, metric_col, col_widths)
    df_plot = df_plot.set_index('label')

    # 5. Header bauen
    short_metric = "F1" if "F1" in metric_col else metric_col
    header_parts = [f"{short_metric:>{col_widths['__metric__']}}"]
    for col in param_cols:
        name = param_map.get(col, col.split('_', 1)[1] if '_' in col else col)
        header_parts.append(f"{name:>{col_widths[col]}}")
    y_axis_title = " | ".join(header_parts) if not show_param_names else "Experiment Parameters"

    # --- 6. DATEN FÜR CONFIDENCE VORBEREITEN ---
    # Wir brauchen Paare von Spalten: s40_eva und s40_conf
    # Logik:
    # Wenn eva == 1: Wert = +confidence (0 bis 1)
    # Wenn eva == 0: Wert = -confidence (0 bis -1)

    # Suche alle Session-IDs
    eva_cols = [c for c in df_plot.columns if c.endswith('_eva')]
    sessions = [c.replace('_eva', '') for c in eva_cols]

    conf_data = pd.DataFrame(index=df_plot.index)

    found_any = False
    for sess in sessions:
        col_eva = f"{sess}_eva"
        col_conf = f"{sess}_conf"

        if col_conf in df_plot.columns:
            found_any = True
            # Vektor-Rechnung: (2 * eva - 1) macht aus 0->-1 und aus 1->1
            # Dann mal Confidence multiplizieren.
            conf_data[sess] = (2 * df_plot[col_eva] - 1) * df_plot[col_conf]
        else:
            # Fallback, falls Confidence fehlt: Hart 1.0 oder -1.0
            conf_data[sess] = (2 * df_plot[col_eva] - 1)

    if not found_any and verbose:
        print("Warnung: Keine '_conf' Spalten gefunden. Zeige Standard-Heatmap.")

    # 7. Custom Colormap definieren (Hellrot -> Schwarz -> Hellgrün)
    # 0.0 (Links) = Hellrot (#ff4d4d)
    # 0.5 (Mitte) = Fast Schwarz (#1a1a1a) -> "Dunkel"
    # 1.0 (Rechts) = Hellgrün (#4dff4d)
    colors = ["#ff4d4d", "#260000", "#002600", "#4dff4d"]
    # Positionen: -1, nahe 0 (negativ), nahe 0 (positiv), +1
    # Wir machen es linear:
    cmap_name = 'conf_diverging'
    # Wir definieren 3 Ankerpunkte: 0.0 (Rot), 0.5 (Schwarz), 1.0 (Grün)
    my_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    # 8. Plotten
    total_height = max(4, len(df_plot) * row_height)
    plt.figure(figsize=(22, total_height))

    ax = sns.heatmap(
        conf_data,
        cmap=my_cmap,
        vmin=-1.0, vmax=1.0, # WICHTIG: Skala festnageln
        cbar=True,           # Farblegende anzeigen!
        cbar_kws={'label': 'Confidence (Negative=Error, Positive=Success)', 'shrink': 0.5},
        linewidths=0.5,
        linecolor='#333333', # Dunkle Linien für besseren Kontrast
        annot=False
    )

    plt.title(title, fontsize=16, pad=30)
    plt.ylabel(y_axis_title, fontsize=11, fontweight='bold',
               fontfamily='monospace', rotation=0, loc='top', labelpad=10)
    plt.yticks(rotation=0, fontfamily='monospace', fontsize=10)
    plt.xlabel("Sessions", fontsize=12)
    plt.tight_layout()
    plt.show()
