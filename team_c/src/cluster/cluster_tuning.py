import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from typing import Dict, List
from team_c.src.cluster.eval import pairwise_f1_score, pairwise_f1_score_per_speaker, plot_and_save_clustered_segs # changed absolute path
from team_c.script.main import inference
from team_c.src.cluster.new_score_calc import calculate_overlap_duration, calculate_conversation_scores

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

def grid_search_agglomerative_clustering(
        df: pd.DataFrame,
        thresholds: list[float] = np.arange(0.70, 0.8 + 0.001, 0.01).round(2).tolist(),
        linkages: list[str] = ["complete", "average"]
) -> pd.DataFrame:
    """
        Grid Search für beste Parameter des agglomerative clustering

        Args:
            df: DataFrame u.a. mit Spalten ["session_name", "true_clusters", "session_scores"]
            thresholds: Liste der Threshold-Werte
            linkages: Liste der Linkage-Methoden

        Returns:
            DataFrame mit durchschnittlichen Metriken über alle Sessions
        """
    results = []

    for th, linkage in itertools.product(thresholds, linkages):
        ari_list, macro_list_per_speaker, micro_list_per_speaker, pairwise_f1_list = [], [], [], []

        for _, row in df.iterrows():
            scores = row["session_scores"]
            true_labels = list(row["true_clusters"].values())

            # --- Agglomerative Clustering ---
            distances = 1 - scores
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1 - th,
                metric="precomputed",
                linkage=linkage
            )
            pred_labels = clustering.fit_predict(distances)

            # # --- Skip, wenn nur ein Cluster vorhanden ist ---
            # if len(set(true_labels)) <= 1 or len(set(pred_labels)) <= 1:
            #     continue

            # --- ARI ---
            ari = adjusted_rand_score(true_labels, pred_labels)

            # --- Speaker-F1 ---
            f1_scores_per_speaker = compute_macro_micro_f1_per_speaker(true_labels, pred_labels)

            # --- Pairwise-F1 ---
            pw_f1 = pairwise_f1_score(true_labels, pred_labels)

            # --- Sammeln ---
            ari_list.append(ari)
            macro_list_per_speaker.append(f1_scores_per_speaker["macro_f1_per_speaker"])
            micro_list_per_speaker.append(f1_scores_per_speaker["micro_f1_per_speaker"])
            pairwise_f1_list.append(pw_f1)

        # Durchschnitt über alle Sessions
        results.append({
            "threshold": th,
            "linkage": linkage,
            "ARI_mean": np.mean(ari_list),
            "Pairwise_F1_mean": np.mean(pairwise_f1_list),
            "Macro-F1_per_Speaker_mean": np.mean(macro_list_per_speaker),
            "Micro-F1_per_Speaker_mean": np.mean(micro_list_per_speaker),
            "ARI_std": np.std(ari_list),
            "Pairwise_F1_std": np.std(pairwise_f1_list),
            "Macro-F1_per_Speaker_std": np.std(macro_list_per_speaker),
            "Micro-F1_per_Speaker_std": np.std(micro_list_per_speaker),
        })
    df_results = pd.DataFrame(results)
    # Ergebnisse sortieren nach ARI_mean
    df_results = df_results.sort_values("Pairwise_F1_mean", ascending=False)
    return df_results

def grid_search_preprocessing_and_clustering(
    df: pd.DataFrame,  # DataFrame mit Spalten ["session_name", "speaker_segments", "true_clusters"]
    # thresholds: List[float] = np.arange(0.70, 0.76 + 0.001, 0.005).round(3).tolist(),
    thresholds: List[float] = np.arange(0.76, 0.8 + 0.001, 0.005).round(3).tolist(),
    linkages: List[str] = ["complete", "average"],
    tolerances: List[float] = [0.0, 0.2, 0.5,1],
    weight_options: List[bool] = [False],
    non_linears: List[str] = [None, "sigmoid", "log"]
) -> pd.DataFrame:
    """
    Grid Search über Preprocessing (Overlap-Berechnung) und Agglomerative Clustering Parameter
    Berechnet ARI, Pairwise-F1, Macro- und Micro-F1 pro Speaker.

    Args:
        df: DataFrame u.a. mit Spalten ["session_name", "true_clusters", "session_scores","session_speaker_segments"]
        thresholds: Liste der Threshold-Werte (threshold = Minimum score to consider speakers in same conversation)
        linkages: Liste der Linkage-Methoden des agglomerative clustering
        tolerances: Liste der Toleranzwerte in sek bei denen Überlappendes Sprechen nicht als Überlapp gewertet wird
        weight_options: Liste False und True, falls True, werden Overlaps bei längeren Segmenten stärker bestraft.
        non_linears: Liste mit Optionen für die Gewichtung von Overlap vs. Non-Overlap (z. B. statt linear, mit einer Sigmoid- oder Log-Funktion).

    Returns:
        DataFrame mit durchschnittlichen Metriken über alle Sessions

    """
    results = []
    i = 0
    for th, linkage, tol, weight, non_lin in itertools.product(
        thresholds, linkages, tolerances, weight_options, non_linears
    ):
        ari_list, macro_list_per_speaker, micro_list_per_speaker, pairwise_f1_list = [], [], [], []
        i += 1
        print(f"Step {i}: ",th, linkage, tol, weight, non_lin)
        for _, row in df.iterrows():
            speaker_segments = row["session_speaker_segments"]
            true_labels = list(row["true_clusters"].values())


            # --- Scores mit Preprocessing berechnen ---
            scores = calculate_conversation_scores(
                speaker_segments,
                non_linear=non_lin,
                tolerance=tol,
                weight_by_length=weight
            )

            # --- Agglomerative Clustering ---
            distances = 1 - scores
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1 - th,
                metric="precomputed",
                linkage=linkage
            )
            pred_labels = clustering.fit_predict(distances)

            # --- Metriken ---
            ari_list.append(adjusted_rand_score(true_labels, pred_labels))
            f1_scores = compute_macro_micro_f1_per_speaker(true_labels, pred_labels)
            macro_list_per_speaker.append(f1_scores["macro_f1_per_speaker"])
            micro_list_per_speaker.append(f1_scores["micro_f1_per_speaker"])
            pairwise_f1_list.append(pairwise_f1_score(true_labels, pred_labels))

        # --- Durchschnitt über alle Sessions ---
        results.append({
            "threshold": th,
            "linkage": linkage,
            "tolerance": tol,
            "weight_by_length": weight,
            "non_linear": non_lin,
            "ARI_mean": np.mean(ari_list),
            "Pairwise_F1_mean": np.mean(pairwise_f1_list),
            "Macro-F1_per_speaker_mean": np.mean(macro_list_per_speaker),
            "Micro-F1_per_speaker_mean": np.mean(micro_list_per_speaker),
            "ARI_std": np.std(ari_list),
            "Pairwise_F1_std": np.std(pairwise_f1_list),
            "Macro-F1_per_speaker_std": np.std(macro_list_per_speaker),
            "Micro-F1_per_speaker_std": np.std(micro_list_per_speaker)
        })
    df_results = pd.DataFrame(results)
    # Ergebnisse sortieren nach ARI_mean
    df_results = df_results.sort_values("Pairwise_F1_mean", ascending=False)
    return df_results




if __name__ == "__main__":
    result = inference()
    df_grid_search_cluster = grid_search_agglomerative_clustering(result)
    print(df_grid_search_cluster.head(10))
    df_grid_search_cluster.to_csv("df_grid_search_cluster.csv", sep=";", index=True, encoding="utf-8-sig")

    df_grid_search_preprocess_and_cluster = grid_search_preprocessing_and_clustering(result)
    print(df_grid_search_preprocess_and_cluster.head(10))
    df_grid_search_preprocess_and_cluster.to_csv("df_grid_search_preprocess_and_cluster.csv", sep=";", index=True, encoding="utf-8-sig")


