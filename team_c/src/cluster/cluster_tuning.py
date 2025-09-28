import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from typing import Dict, List
import matplotlib.pyplot as plt
from team_c.src.cluster.eval import pairwise_f1_score, pairwise_f1_score_per_speaker  # changed absolute path


def compute_macro_micro_f1_per_speaker(true_labels: List[int], pred_labels: List[int]) -> Dict[str, float]:
    per_speaker_f1 = pairwise_f1_score_per_speaker(true_labels, pred_labels)
    macro_f1 = sum(per_speaker_f1.values()) / len(per_speaker_f1)

    # Micro-F1
    n = len(true_labels)
    total_tp = total_fp = total_fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            true_same = (true_labels[i] == true_labels[j])
            pred_same = (pred_labels[i] == pred_labels[j])
            if pred_same and true_same:
                total_tp += 1
            elif pred_same and not true_same:
                total_fp += 1
            elif not pred_same and true_same:
                total_fn += 1
    if total_tp == 0:
        micro_f1 = 0.0
    else:
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"macro_f1_per_speaker": macro_f1, "micro_f1_per_speaker": micro_f1}


def grid_search_agglomerative_clustering(sessions: list[dict],
                               thresholds: list[float] = [0.5, 0.6, 0.7, 0.8],
                               linkages: list[str] = ["complete", "average"]) -> pd.DataFrame:
    """
    Grid Search für mehrere Sessions gleichzeitig.

    Args:
        sessions: Liste von Dicts, jedes mit {"scores": NxN np.ndarray, "true_labels": List[int]}
        thresholds: Liste der Threshold-Werte
        linkages: Liste der Linkage-Methoden

    Returns:
        DataFrame mit durchschnittlichen Metriken über alle Sessions
    """
    results = []

    for th, linkage in itertools.product(thresholds, linkages):
        ari_list, macro_list_per_speaker, micro_list_per_speaker, macro_list, micro_list = [], [], [], [], []

        for session in sessions:
            scores = session["scores"]
            true_labels = session["true_labels"]

            distances = 1 - scores
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1 - th,
                metric="precomputed",
                linkage=linkage
            )
            pred_labels = clustering.fit_predict(distances)

            # Metriken
            ari = adjusted_rand_score(true_labels, pred_labels)
            f1_scores_per_speaker = compute_macro_micro_f1_per_speaker(true_labels, pred_labels)

            ari_list.append(ari)
            macro_list_per_speaker.append(f1_scores_per_speaker ["macro_f1_per_speaker"])
            micro_list_per_speaker.append(f1_scores_per_speaker ["micro_f1_per_speaker"])


        # Durchschnitt über Sessions
        results.append({
            "threshold": th,
            "linkage": linkage,
            "ARI_mean": np.mean(ari_list),
            "Macro-F1_per_Speaker_mean": np.mean(macro_list_per_speaker),
            "Micro-F1_per_Speaker_mean": np.mean(micro_list_per_speaker),
            "ARI_std": np.std(ari_list),  # optional: Varianz anschauen
            "Macro-F1_per_Speaker_std": np.std(macro_list_per_speaker),
            "Micro-F1_per_Speaker_std": np.std(micro_list_per_speaker),
        })
    # # Heatmap für ARI
    # plot_heatmap(pd.DataFrame(results), metric="ARI_mean")
    #
    # # Heatmap für Macro-F1
    # plot_heatmap(pd.DataFrame(results), metric="Macro-F1_mean")
    #
    # # Heatmap für Micro-F1
    # plot_heatmap(pd.DataFrame(results), metric="Micro-F1_mean")
    return pd.DataFrame(results)

# ----------------------------
# Heatmap Plot
# ----------------------------
def plot_heatmap(df: pd.DataFrame, metric: str = "ARI"):
    pivot = df.pivot(index="threshold", columns="linkage", values=metric)

    plt.figure(figsize=(6,4))
    plt.imshow(pivot, cmap="viridis", aspect="auto", origin="lower")
    plt.colorbar(label=metric)
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Linkage")
    plt.ylabel("Threshold")
    plt.title(f"Heatmap: {metric}")
    for i, th in enumerate(pivot.index):
        for j, link in enumerate(pivot.columns):
            value = pivot.loc[th, link]
            plt.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=9)
    plt.show()
