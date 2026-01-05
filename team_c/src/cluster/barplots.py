import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import math
from pathlib import Path

def plot_barchart(
    ax,
    df,
    bar_col,
    x_group_col,
    color_group_col,
    *,
    x_order=None,
    color_order=None,
    aggfunc='mean',
    colors=None,
    labelmap=None,
    paramap=None,
    yrange=None,
    gap=0.15,
    title=None,
    annotate=True,
    annotate_fmt="{:.3f}",
    annotate_offset=0.005,
    rotation=45,
    legend=True,
    legend_title=None,
    highlight_max_xticks=False,
    highlight_color="red",
    highlight_fontweight="bold",
    xtick_fontsize=None,
    ytick_fontsize=None,
):
    # ---------- Validierung ----------
    required_cols = {bar_col, x_group_col, color_group_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten im DataFrame: {missing}")

    if df.empty:
        raise ValueError("DataFrame ist leer")

    # ---------- Gruppieren ----------
    grouped = (
        df
        .groupby([x_group_col, color_group_col], dropna=False)[bar_col]
        .agg(aggfunc)
        .reset_index()
    )

    # ---------- Reihenfolge ----------
    if x_order is None:
        x_vals = grouped[x_group_col].unique()
    else:
        x_vals = x_order

    if color_order is None:
        color_groups = grouped[color_group_col].unique()
    else:
        color_groups = color_order

    x_pos = np.arange(len(x_vals))
    x_pos_map = {v: i for i, v in enumerate(x_vals)}

    # ---------- Balkenbreite ----------
    usable_width = 1 - gap
    bar_width = usable_width / len(color_groups)

    # ---------- Plot ----------
    max_x_positions = set()  # sammelt x-Positionen der Maxima

    for i, cg in enumerate(color_groups):
        sub = grouped[grouped[color_group_col] == cg]
        heights = [
            sub.loc[sub[x_group_col] == xv, bar_col].iloc[0]
            if xv in sub[x_group_col].values else np.nan
            for xv in x_vals
        ]

        color = None
        if isinstance(colors, dict):
            color = colors.get(cg)
        elif isinstance(colors, (list, tuple)):
            color = colors[i]

        ax.bar(
            x_pos - usable_width / 2 + i * bar_width + bar_width / 2,
            heights,
            width=bar_width,
            label=str(cg),
            color=color
        )

        # ---------- Annotation ----------
        if annotate and np.any(~np.isnan(heights)):
            max_idx = np.nanargmax(heights)
            y = heights[max_idx]
            max_x_positions.add(max_idx)

            x_text = (
                x_pos[max_idx]
                - usable_width / 2
                + i * bar_width
                + bar_width / 2
            )
            ax.text(
                x_text,
                y + annotate_offset,
                f"max({cg})\n{annotate_fmt.format(y)}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    # ---------- Achsen ----------
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_vals, rotation=rotation)

    # ---- Hervorhebung der x-Tick-Labels ----
    if highlight_max_xticks:
        for i, label in enumerate(ax.get_xticklabels()):
            if i in max_x_positions:
                label.set_color(highlight_color)
                label.set_fontweight(highlight_fontweight)

    # ---- Optional: Fontsize für x-Ticks ----
    if xtick_fontsize is not None:
        for label in ax.get_xticklabels():
            label.set_fontsize(xtick_fontsize)

    # ---- Optional: Fontsize für y-Ticks ----
    if ytick_fontsize is not None:
        for label in ax.get_yticklabels():
            label.set_fontsize(ytick_fontsize)

    if yrange:
        ax.set_ylim(*yrange)

    ax.set_xlabel(paramap.get(x_group_col, x_group_col) if paramap else x_group_col)
    ax.set_ylabel(labelmap.get(bar_col, bar_col) if labelmap else bar_col)

    if title:
        ax.set_title(title)

    if legend:
        ax.legend(title=legend_title)

if __name__ == '__main__':
    BASE_DIR = Path(__file__).resolve().parents[2] ## team_c
    OUTPUT_DIR = BASE_DIR / "results_data_plots"

    labelmap = {'Pairwise_F1_mean': 'Mittlerer Pairwise-F1 Score',
                'Macro-F1_per_speaker_mean': 'Mittlerer F1 Score pro Sprecher'}
    parametermap = {'threshold': 'Score - Threshold',
                    'linkage': 'Linkage',
                    "tolerance": "Overlap - Toleranz",
                    "non_linear":'Overlap-Berechnung',
                    "weight_by_length":'Gewichtung von Overlaps bei längeren Segmenten'
                    }

    df_all_teil1 = pd.read_csv("df_grid_search_preprocess_and_cluster_Teil1.csv", delimiter=";")
    df_all_teil2 = pd.read_csv("df_grid_search_preprocess_and_cluster_Teil2.csv", delimiter=";")
    df_all = pd.concat([df_all_teil1, df_all_teil2], ignore_index=True).drop_duplicates(ignore_index=True)
    df_all = df_all.drop(
        columns=["Micro-F1_per_speaker_mean", "ARI_std", "Pairwise_F1_std", "Macro-F1_per_speaker_std",
                 "Micro-F1_per_speaker_std"])
    #############################################################################################
    # ### ab hier  Barplot f1 scores bei variablem threshold und linkage
    # df_part = df_all.query("tolerance == 0 and non_linear.isna() and weight_by_length == False")
    # df_part = df_part.drop(columns=["tolerance", "non_linear", "weight_by_length", "ARI_mean", "Unnamed: 0"])
    # df_part = df_part.drop_duplicates(ignore_index=True)
    # df_part.reset_index()
    # df_part = df_part.reset_index(drop=True)
    # x_group = 'threshold'  ## für die Gruppierung entlang der x-Achse
    # color_group = 'linkage'  ## für die Farben der Balken
    # 
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    # 
    # balkenhoehe_1 = 'Pairwise_F1_mean'
    # # title = f'{labelmap[balkenhoehe]} \nbei verschiedenen Clusterparametern\n{parametermap[x_group]} und {parametermap[color_group]}'
    # title_1 = f'{labelmap[balkenhoehe_1]}'
    # balkenhoehe_2 = 'Macro-F1_per_speaker_mean'
    # # title = f'{labelmap[balkenhoehe]} \nbei verschiedenen Clusterparametern\n{parametermap[x_group]} und {parametermap[color_group]}'
    # title_2 = f'{labelmap[balkenhoehe_2]}'
    # plot_barchart(
    #     axes[0],
    #     df_part,
    #     bar_col=balkenhoehe_1,  ## legt die Balkenhöhe fest, = y-Wert
    #     x_group_col=x_group,  ## für die Gruppierung entlang der x-Achse
    #     color_group_col=color_group,  ## für die Farben der Balken
    #     labelmap=labelmap,
    #     paramap=parametermap,
    #     yrange=[0.6, 1],
    #     gap=0.15,
    #     title=title_1,
    #     legend_title=parametermap['linkage'],
    #     highlight_max_xticks=True,
    #     highlight_fontweight='bold',
    #     xtick_fontsize=8,
    #     ytick_fontsize=8
    # )
    # 
    # plot_barchart(
    #     axes[1],
    #     df_part,
    #     bar_col=balkenhoehe_2,  ## legt die Balkenhöhe fest, = y-Wert
    #     x_group_col=x_group,  ## für die Gruppierung entlang der x-Achse
    #     color_group_col=color_group,  ## für die Farben der Balken
    #     labelmap=labelmap,
    #     paramap=parametermap,
    #     yrange=[0.6, 1],
    #     gap=0.15,
    #     title=title_2,
    #     legend_title=parametermap['linkage'],
    #     highlight_max_xticks=True,
    #     highlight_fontweight='bold',
    #     xtick_fontsize=8,
    #     ytick_fontsize=8
    # )
    # 
    # outfilename = f'{balkenhoehe_1}_und_{balkenhoehe_2}_at_variablem_{x_group}_und_{color_group}'
    # plotfile_out = OUTPUT_DIR / outfilename
    # 
    # fig.savefig(plotfile_out, dpi=300, bbox_inches="tight")
    # plt.show()

    ################################################################################################
    ### ab hier  Barplot f1 scores bei variablem preprocessing
    '''
        tolerances: Liste der Toleranzwerte in sek bei denen Überlappendes Sprechen nicht als Überlapp gewertet wird
        weight_options: Liste False und True, falls True, werden Overlaps bei längeren Segmenten stärker bestraft.
        non_linears: Liste mit Optionen für die Gewichtung von Overlap vs. Non-Overlap (z. B. statt linear, mit einer Sigmoid- oder Log-Funktion).
    '''
    df_part_weight_false = df_all.query("linkage == 'complete' and threshold == 0.74 and weight_by_length == False")  ## weight_by_length == False
    df_part_weight_false = df_part_weight_false.drop(columns=["linkage", "threshold","ARI_mean", "weight_by_length", "Unnamed: 0"])
    df_part_weight_false = df_part_weight_false.fillna({'non_linear':'linear'})
    df_part_weight_false = df_part_weight_false.drop_duplicates(ignore_index=True)
    df_part_weight_false.reset_index()
    df_part_weight_false = df_part_weight_false.reset_index(drop=True)

    # print(df_part_weight_false)
    df_part_weight_true = df_all.query("linkage == 'complete' and threshold == 0.74 and weight_by_length == True")  ## weight_by_length == True
    df_part_weight_true = df_part_weight_true.drop(columns=["ARI_mean", "weight_by_length", "Unnamed: 0"])
    df_part_weight_true = df_part_weight_true.fillna({'non_linear': 'linear'})
    df_part_weight_true = df_part_weight_true.drop_duplicates(ignore_index=True)
    df_part_weight_true.reset_index()
    df_part_weight_true = df_part_weight_true.reset_index(drop=True)

    x_group = 'tolerance'  ## für die Gruppierung entlang der x-Achse
    color_group = 'non_linear'  ## für die Farben der Balken

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharey=True)
    # axes ist 2x2 Array:
    # [[axes[0,0], axes[0,1]],
    #  [axes[1,0], axes[1,1]]]

    balkenhoehe_1 = 'Pairwise_F1_mean'
    title_0 = f'{labelmap[balkenhoehe_1]}\n{parametermap["weight_by_length"]} = False'
    title_2 = f'{labelmap[balkenhoehe_1]}\n{parametermap["weight_by_length"]} = True'
    balkenhoehe_2 = 'Macro-F1_per_speaker_mean'
    title_1 = f'{labelmap[balkenhoehe_2]}\n{parametermap["weight_by_length"]} = False'
    title_3 = f'{labelmap[balkenhoehe_2]}\n{parametermap["weight_by_length"]} = True'
    plot_barchart(
        axes[0,0],
        df_part_weight_false,
        bar_col=balkenhoehe_1,  ## legt die Balkenhöhe fest, = y-Wert
        x_group_col=x_group,  ## für die Gruppierung entlang der x-Achse
        color_group_col=color_group,  ## für die Farben der Balken
        labelmap=labelmap,
        paramap=parametermap,
        yrange=[0.6, 1],
        gap=0.15,
        title=title_0,
        legend_title=parametermap['non_linear'],
        highlight_max_xticks=False,
        xtick_fontsize=8,
        ytick_fontsize=8
    )

    plot_barchart(
        axes[0,1],
        df_part_weight_false,
        bar_col=balkenhoehe_2,  ## legt die Balkenhöhe fest, = y-Wert
        x_group_col=x_group,  ## für die Gruppierung entlang der x-Achse
        color_group_col=color_group,  ## für die Farben der Balken
        labelmap=labelmap,
        paramap=parametermap,
        yrange=[0.6, 1],
        gap=0.15,
        title=title_1,
        legend_title=parametermap['non_linear'],
        highlight_max_xticks=False,
        xtick_fontsize=8,
        ytick_fontsize=8
    )
    plot_barchart(
        axes[1,0],
        df_part_weight_true,
        bar_col=balkenhoehe_1,  ## legt die Balkenhöhe fest, = y-Wert
        x_group_col=x_group,  ## für die Gruppierung entlang der x-Achse
        color_group_col=color_group,  ## für die Farben der Balken
        labelmap=labelmap,
        paramap=parametermap,
        yrange=[0.6, 1],
        gap=0.15,
        title=title_2,
        legend_title=parametermap['non_linear'],
        highlight_max_xticks=False,
        xtick_fontsize=8,
        ytick_fontsize=8
    )

    plot_barchart(
        axes[1,1],
        df_part_weight_true,
        bar_col=balkenhoehe_2,  ## legt die Balkenhöhe fest, = y-Wert
        x_group_col=x_group,  ## für die Gruppierung entlang der x-Achse
        color_group_col=color_group,  ## für die Farben der Balken
        labelmap=labelmap,
        paramap=parametermap,
        yrange=[0.6, 1],
        gap=0.15,
        title=title_3,
        legend_title=parametermap['non_linear'],
        highlight_max_xticks=False,
        xtick_fontsize=8,
        ytick_fontsize=8
    )
    plt.tight_layout()
    outfilename = f'{balkenhoehe_1}_und_{balkenhoehe_2}_at_variablem_{x_group}_und_{color_group}'
    plotfile_out = OUTPUT_DIR / outfilename

    fig.savefig(plotfile_out, dpi=300, bbox_inches="tight")
    plt.show()
    ################################################################################################