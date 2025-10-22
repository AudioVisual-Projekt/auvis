import itertools
import os
from typing import List
import pandas as pd
import matplotlib.pyplot as plt

# --- 1️⃣ Importe ---
from team_c.src.talking_detector.segmentation import CENTRAL_ASD_CHUNKING_PARAMETERS
from team_c.src.cluster.eval import plot_and_save_clustered_segs, run_inference, pairwise_f1_score


# ---- Parametergrid ----
asd_param_grid = {
    "onset": [0.8, 1.0, 1.2],
    "offset": [0.6, 0.8, 1.0],
    "min_duration_on": [0.5, 1.0, 1.5, 2.0],
    "min_duration_off": [0.3, 0.5,1.0, 1.5],
}
cluster_thresholds = [0.6, 0.7, 0.74, 0.8, 0.9]

asd_combinations = list(itertools.product(
    asd_param_grid["onset"],
    asd_param_grid["offset"],
    asd_param_grid["min_duration_on"],
    asd_param_grid["min_duration_off"]
))

# --- 3️⃣ Hauptschleife ---
base_output_root = "results_gridsearch"
summary_rows = []  # F1-Ergebnisse für Vergleich

#
# ---- Hauptschleife ----
for onset, offset, min_on, min_off in asd_combinations:
    for cluster_thr in cluster_thresholds:
        CENTRAL_ASD_CHUNKING_PARAMETERS.update({
            "onset": onset,
            "offset": offset,
            "min_duration_on": min_on,
            "min_duration_off": min_off,
        })

        run_name = f"on{onset}_off{offset}_mon{min_on}_moff{min_off}_thr{cluster_thr}"
        output_dir = os.path.join(base_output_root, run_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n🚀 Starte Lauf: {run_name}")
        inference_result = run_inference(cluster_threshold=cluster_thr)

        # ---- F1 Score ----
        f1_scores = []
        for _, row in inference_result.iterrows():
            true_dict = row["true_clusters"]
            pred_dict = row["pred_clusters"]
            common_speakers = sorted(set(true_dict.keys()) & set(pred_dict.keys()))
            true_labels = [true_dict[s] for s in common_speakers]
            pred_labels = [pred_dict[s] for s in common_speakers]
            f1_scores.append(pairwise_f1_score(true_labels, pred_labels))

        inference_result["f1_score"] = f1_scores
        mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        # Speichern
        inference_result.to_csv(os.path.join(output_dir, "inference_results.csv"), index=False)

        # Plots
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        os.chdir(plots_dir)
        plot_and_save_clustered_segs(
            inference_result["session_name"],
            inference_result["session_speaker_segments"],
            inference_result["true_clusters"],
            inference_result["pred_clusters"]
        )
        os.chdir("../../..")

        summary_rows.append({
            "run_name": run_name,
            "onset": onset,
            "offset": offset,
            "min_duration_on": min_on,
            "min_duration_off": min_off,
            "cluster_threshold": cluster_thr,
            "mean_f1": mean_f1
        })

        print(f"✅ Lauf abgeschlossen: {run_name} (mean F1={mean_f1:.3f})")

# ---- Gesamtübersicht ----
summary_df = pd.DataFrame(summary_rows)
summary_df.sort_values("mean_f1", ascending=False, inplace=True)
summary_df.to_csv(os.path.join(base_output_root, "summary_f1.csv"), index=False)
print("\n🏁 Grid-Search abgeschlossen. Ergebnisse in summary_f1.csv gespeichert.")
