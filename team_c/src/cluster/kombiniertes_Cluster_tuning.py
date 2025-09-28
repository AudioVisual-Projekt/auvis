import itertools
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from typing import List, Dict

from team_c.src.cluster.new_score_calc import (calculate_conversation_scores)
from team_c.src.cluster.cluster_tuning import (pairwise_f1_score_per_speaker, compute_macro_micro_f1_per_speaker)



def grid_search_full(
    sessions: List[Dict],  # Jede Session: {"speaker_segments": dict, "true_labels": List[int]}
    thresholds: List[float] = [0.5, 0.6, 0.7, 0.8],
    linkages: List[str] = ["complete", "average"],
    tolerances: List[float] = [0.0, 0.2, 0.5],
    weight_options: List[bool] = [False, True],
    non_linears: List[str] = [None, "sigmoid", "log"]
) -> pd.DataFrame:
    """
    Grid Search über Preprocessing (Overlap-Berechnung) und Agglomerative Clustering Parameter.
    Gibt DataFrame mit durchschnittlichen Metriken über alle Sessions zurück.
    """
    results = []

    # Alle Kombinationen durchprobieren
    for th, linkage, tol, weight, non_lin in itertools.product(
        thresholds, linkages, tolerances, weight_options, non_linears
    ):
        ari_list, macro_list_per_speaker, micro_list_per_speaker = [], [], []

        for session in sessions:
            speaker_segments = session["speaker_segments"]
            true_labels = session["true_labels"]

            # 1️⃣ Konversation-Scores berechnen mit den Preprocessing-Parametern
            scores = calculate_conversation_scores(
                speaker_segments,
                non_linear=non_lin,
                tolerance=tol,
                weight_by_length=weight
            )

            # 2️⃣ Agglomerative Clustering mit threshold/linkage
            distances = 1 - scores
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1-th,
                metric="precomputed",
                linkage=linkage
            )
            pred_labels = clustering.fit_predict(distances)

            # 3️⃣ Metriken
            ari_list.append(adjusted_rand_score(true_labels, pred_labels))
            f1_scores = compute_macro_micro_f1_per_speaker(true_labels, pred_labels)
            macro_list_per_speaker.append(f1_scores["macro_f1_per_speaker"])
            micro_list_per_speaker.append(f1_scores["micro_f1_per_speaker"])

        # Durchschnitt über alle Sessions
        results.append({
            "threshold": th,
            "linkage": linkage,
            "tolerance": tol,
            "weight_by_length": weight,
            "non_linear": non_lin,
            "ARI_mean": np.mean(ari_list),
            "Macro-F1_per_speaker_mean": np.mean(macro_list_per_speaker),
            "Micro-F1_per_speaker_mean": np.mean(micro_list_per_speaker),
            "ARI_std": np.std(ari_list),
            "Macro-F1_per_speaker_std": np.std(macro_list_per_speaker),
            "Micro-F1_per_speaker_std": np.std(micro_list_per_speaker)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Beispiel mit 2 Sessions
    sessions = [
        {
            "speaker_segments": {
                "spk_0": [[0,2],[5,7]],
                "spk_1": [[1,3],[6,8]],
                "spk_2": [[10,12]]
            },
            "true_labels": [0,0,1]
        },
        {
            "speaker_segments": {
                "spk_0": [[0,3],[7,8]],
                "spk_1": [[1,4]],
                "spk_2": [[9,11]]
            },
            "true_labels": [0,0,1]
        }
    ]

    df_results = grid_search_full(
        sessions,
        thresholds=[0.7, 0.8],
        linkages=["complete", "average"],
        tolerances=[0.0, 0.5],
        weight_options=[False, True],
        non_linears=[None, "sigmoid"]
    )

    # Ergebnisse sortieren nach ARI_mean
    df_results = df_results.sort_values("ARI_mean", ascending=False)
    print(df_results[["threshold","linkage","tolerance","weight_by_length","non_linear","ARI_mean","Macro-F1_per_speaker_mean","Micro-F1_per_speaker_mean"]])
