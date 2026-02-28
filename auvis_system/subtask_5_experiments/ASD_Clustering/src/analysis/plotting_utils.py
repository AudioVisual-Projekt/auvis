"""
Module: plotting_utils
======================
Contains specialized plotting functions for the Clustering/ASD analysis pipeline.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


# --- HELPER ---
def _create_simple_labels(df_plot):
    """Erstellt einfache Labels: 'Config 01', 'Config 02'."""
    labels = []
    for i in range(len(df_plot)):
        labels.append(f"Config {i+1:02d}")
    return labels

def _ensure_plot_dir():
    path = os.path.join("analysis", "plots")
    os.makedirs(path, exist_ok=True)
    return path

# --- HAUPTFUNKTION 1: Binäre Heatmap ---
def plot_session_heatmap(df: pd.DataFrame,
                         metric_col: str = "AVG_SPEAKER_F1",
                         title: str = "Session Success (Binary)",
                         row_height: float = 0.4,
                         top_n: int = 20): # <--- NEU: Standardmäßig Top 20
    """
    Zeigt die Top N Configs als Rot/Grün Matrix.
    """
    # 1. Sortieren
    df_plot = df.sort_values(by=metric_col, ascending=False).copy()

    # 2. Top N filtern (falls gewünscht)
    if top_n is not None and top_n < len(df_plot):
        df_plot = df_plot.head(top_n)

    # 3. Labels
    df_plot['label'] = _create_simple_labels(df_plot)
    df_plot = df_plot.set_index('label')

    # 4. Daten
    eva_cols = [c for c in df_plot.columns if c.endswith('_eva')]
    heatmap_data = df_plot[eva_cols].rename(columns=lambda x: x.replace('_eva', ''))

    # 5. Plot
    cmap = ListedColormap(['#ff4d4d', '#4dff4d'])
    total_height = max(6, len(df_plot) * row_height)

    plt.figure(figsize=(22, total_height))

    ax = sns.heatmap(
        heatmap_data,
        cmap=cmap,
        vmin=0, vmax=1,
        cbar=True,
        cbar_kws={'ticks': [0.25, 0.75], 'shrink': 0.7},
        linewidths=0.5,
        linecolor='#e0e0e0',
        square=False
    )

    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(['Error', 'Perfect'])

    plt.title(f"{title} (Top {len(df_plot)})", fontsize=16, pad=20)
    plt.ylabel("Experiment Configurations", fontsize=12)
    plt.xlabel("Sessions", fontsize=12)
    plt.yticks(rotation=0, fontfamily='monospace')

    plt.tight_layout()

    save_dir = _ensure_plot_dir()
    save_path = os.path.join(save_dir, "session_heatmap_binary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Heatmap gespeichert: {save_path}")
    plt.show()

# --- HAUPTFUNKTION 2: Confidence Heatmap ---
def plot_confidence_heatmap(df: pd.DataFrame,
                            metric_col: str = "AVG_SPEAKER_F1",
                            title: str = "Confidence (Brightness = Certainty)",
                            row_height: float = 0.4,
                            top_n: int = 20): # <--- NEU: Standardmäßig Top 20
    """
    Zeigt die Top N Configs mit Confidence.
    """
    # 1. Sortieren
    df_plot = df.sort_values(by=metric_col, ascending=False).copy()

    # 2. Top N filtern
    if top_n is not None and top_n < len(df_plot):
        df_plot = df_plot.head(top_n)

    # 3. Labels
    df_plot['label'] = _create_simple_labels(df_plot)
    df_plot = df_plot.set_index('label')

    # 4. Daten berechnen
    eva_cols = [c for c in df_plot.columns if c.endswith('_eva')]
    sessions = [c.replace('_eva', '') for c in eva_cols]

    conf_data = pd.DataFrame(index=df_plot.index)
    for sess in sessions:
        col_eva = f"{sess}_eva"
        col_conf = f"{sess}_conf"
        if col_conf in df_plot.columns:
            direction = (2 * df_plot[col_eva] - 1)
            confidence = df_plot[col_conf]
            conf_data[sess] = direction * confidence
        else:
            conf_data[sess] = (2 * df_plot[col_eva] - 1)

    # 5. Plot
    colors = ["#ff4d4d", "#330000", "#003300", "#4dff4d"]
    my_cmap = LinearSegmentedColormap.from_list('conf_map', colors, N=100)

    total_height = max(6, len(df_plot) * row_height)

    plt.figure(figsize=(22, total_height))

    ax = sns.heatmap(
        conf_data,
        cmap=my_cmap,
        vmin=-1.0, vmax=1.0,
        cbar=True,
        cbar_kws={'label': 'Confidence', 'shrink': 0.7, 'ticks': [-1, 0, 1]},
        linewidths=0.5,
        linecolor='#222222'
    )

    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(['Sicher Falsch', 'Unsicher', 'Sicher Richtig'])

    plt.title(f"{title} (Top {len(df_plot)})", fontsize=16, pad=20)
    plt.ylabel("Experiment Configurations", fontsize=12)
    plt.xlabel("Sessions", fontsize=12)
    plt.yticks(rotation=0, fontfamily='monospace')

    plt.tight_layout()

    save_dir = _ensure_plot_dir()
    save_path = os.path.join(save_dir, "session_heatmap_confidence.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Heatmap gespeichert: {save_path}")
    plt.show()
