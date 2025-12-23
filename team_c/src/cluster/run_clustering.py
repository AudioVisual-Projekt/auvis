import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
# import matplotlib.pyplot as plt
# from sklearn.cluster import AgglomerativeClustering
# from scipy.cluster.hierarchy import linkage, dendrogram
# from scipy.spatial.distance import squareform
import os
# import json
# import glob
# from typing import List, Dict
from team_c.src.talking_detector.segmentation import CENTRAL_ASD_CHUNKING_PARAMETERS
from team_c.src.cluster.eval import plot_and_save_clustered_segs, pairwise_f1_score, compute_macro_micro_f1_per_speaker
from team_c.src.cluster.dendrogramme import plot_colored_dendrograms
from team_c.src.cluster.max_distance import max_clustering_distance
from team_c.script.main import read_cluster_labels_from_json
from pathlib import Path



def run_inference(cluster_threshold: float = 0.7): # zur Vermeidung eines zirkulären Importerrors
    from team_c.script.main import inference
    return inference(cluster_threshold)

if __name__ == '__main__':
    train_flag = False
    dev_flag = True
    ######################################
    #### ACHTUNG:
    #### in main unter inference() das default session_dir entsprechend manuell anpassen !!!!!!!!!!!!!
    #######################################

    max_distance_overall = 0.7  ## Wird zur Berechnung des trust_scores verwendet

    #### Parameter  Info
    LINKAGE = "complete"  ## dient ausschließlich Infozwecken. Die Linkage muss in conv_spks.py festgelegt werden !!!!!!
    #### Parameter  festlegen
    threshold = 0.74
    CENTRAL_ASD_CHUNKING_PARAMETERS.update({
        "onset": 1.2,
        "offset": 1.0,
        "min_duration_on": 1.5,
        "min_duration_off": 0.5,
    })
    #### Pfade definieren (relativ zu diesem Skript)
    run_name = (
        f"on{CENTRAL_ASD_CHUNKING_PARAMETERS['onset']}_"
        f"off{CENTRAL_ASD_CHUNKING_PARAMETERS['offset']}_"
        f"mon{CENTRAL_ASD_CHUNKING_PARAMETERS['min_duration_on']}_"
        f"moff{CENTRAL_ASD_CHUNKING_PARAMETERS['min_duration_off']}_"
        f"thr{threshold}"
    )

    BASE_DIR = Path(__file__).resolve().parents[2]
    if train_flag:
        print("Train Data werden prozessiert")
        DATA_DIR = BASE_DIR / "data-bin" / "train"
        OUTPUT_DIR = BASE_DIR / "data-bin" / "output_best_params_train" /run_name
        IMAGE_DIR = OUTPUT_DIR / "dendrogramme_best_params_train"
        OUTFILE_METRIKEN = OUTPUT_DIR / "metriken_best_params_train.csv"
        OUTFILE_METRIKEN_PER_SESSION = OUTPUT_DIR / "metriken_best_params_per_session_train.csv"
        OUTFILE_RESULT_PICKLE = OUTPUT_DIR / "inference_result_train.pkl"
        OUTFILE_RESULT_CSV = OUTPUT_DIR / "inference_result_train.csv"
    elif dev_flag:
        print("Dev Data werden prozessiert")
        DATA_DIR = BASE_DIR / "data-bin" / "dev"
        OUTPUT_DIR = BASE_DIR / "data-bin" / "output_best_params_dev" /run_name
        IMAGE_DIR = OUTPUT_DIR / "dendrogramme_best_params_dev"
        OUTFILE_METRIKEN = OUTPUT_DIR / "metriken_best_params_dev.csv"
        OUTFILE_METRIKEN_PER_SESSION = OUTPUT_DIR / "metriken_best_params_per_session_dev.csv"
        OUTFILE_RESULT_PICKLE = OUTPUT_DIR / "inference_result_dev.pkl"
        OUTFILE_RESULT_CSV = OUTPUT_DIR / "inference_result_dev.csv"
    else:
        print("Bitte eine flag (train_flag oder dev_flag) auf True setzen.")

    SESSION_GLOB = "session_*"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    # Ergebnisse des Clustering laden oder neu clustern
    if os.path.exists(OUTFILE_RESULT_PICKLE):
        result = pd.read_pickle(OUTFILE_RESULT_PICKLE)
    elif os.path.exists(OUTFILE_RESULT_CSV):
        df = pd.read_csv(OUTFILE_RESULT_CSV)
    else:
        ### Clustering durchführen
        result = run_inference(threshold)
        result.to_pickle(OUTFILE_RESULT_PICKLE)
        result.to_csv(OUTFILE_RESULT_CSV, index=False)

    # if max_sessions is not None:
    #     df = results.head(max_sessions).copy(deep=True)
    #     print(f"Erstelle Dendrogramme für {max_sessions} Sessions\n")
    # else:
    #     df = results.copy(deep=True)



    ##### Metriken berechnen und speichern
    metriken_best_params = []
    metriken_best_params_per_session = []
    max_total_distances = []
    ari_list, macro_list_per_speaker, micro_list_per_speaker, pairwise_f1_list, max_cluster_distance_list, trust_score_list = [], [], [], [], [], []
    for idx, row in result.iterrows():  ## Iteration über alle Sessions
        plot_colored_dendrograms(row, 1-threshold, IMAGE_DIR, parametersatz=run_name)
        max_cluster_distance_session = max_clustering_distance(row, 1 - threshold)
        # max_total_session_distance = max_clustering_distance(row, 0)    ## hier keine Limitierung durch trehshold, wurde nur 1 mal zur Ermittlung der max_distance_overall verwendet

        trust_score_session = (1 - max_cluster_distance_session / max_distance_overall) * 100 ## Wert in Prozent

        true_labels = list(row["true_clusters"].values())
        pred_labels_pre = list(row["pred_clusters"].values())
        session_name = row["session_name"]

        # --- ARI ---
        ari = adjusted_rand_score(true_labels, pred_labels_pre)

        # --- Speaker-F1 ---
        f1_scores_per_speaker = compute_macro_micro_f1_per_speaker(true_labels, pred_labels_pre)

        # --- Pairwise-F1 ---
        pw_f1 = pairwise_f1_score(true_labels, pred_labels_pre)

        # --- Sammeln ---
        ari_list.append(ari)
        macro_list_per_speaker.append(f1_scores_per_speaker["macro_f1_per_speaker"])
        micro_list_per_speaker.append(f1_scores_per_speaker["micro_f1_per_speaker"])
        pairwise_f1_list.append(pw_f1)
        max_cluster_distance_list.append(max_cluster_distance_session)
        trust_score_list.append(trust_score_session)
        metriken_best_params_per_session.append({
            "session_name": session_name,
            "threshold": threshold,
            "linkage": LINKAGE,
            "ARI": ari,
            "Pairwise_F1": pw_f1,
            "Macro-F1_per_Speaker": f1_scores_per_speaker["macro_f1_per_speaker"],
            "Micro-F1_per_Speaker": f1_scores_per_speaker["micro_f1_per_speaker"],
            "Max_Cluster_Distance": max_cluster_distance_session,
            "Trust_Score_Session_[%]": trust_score_session,
        })
    # Durchschnitt über alle Sessions
    metriken_best_params.append({
        "session_name": session_name,
        "threshold": threshold,
        "linkage": LINKAGE,
        "ARI_mean": np.mean(ari_list),
        "Pairwise_F1_mean": np.mean(pairwise_f1_list),
        "Macro-F1_per_Speaker_mean": np.mean(macro_list_per_speaker),
        "Micro-F1_per_Speaker_mean": np.mean(micro_list_per_speaker),
        "Max_Cluster_Distance_mean": np.mean(max_cluster_distance_list),
        "Trust_Score_Session_[%]_mean": np.mean(trust_score_session),
        "ARI_std": np.std(ari_list),
        "Pairwise_F1_std": np.std(pairwise_f1_list),
        "Macro-F1_per_Speaker_std": np.std(macro_list_per_speaker),
        "Micro-F1_per_Speaker_std": np.std(micro_list_per_speaker),
        "Max_Cluster_Distance_std": np.std(max_cluster_distance_list),
        "Trust_Score_Session_[%]_std": np.std(trust_score_list),
    })
    df_metriken_best_params = pd.DataFrame(metriken_best_params)
    df_metriken_best_params = df_metriken_best_params.sort_values("Pairwise_F1_mean", ascending=False)
    df_metriken_best_params.to_csv(OUTFILE_METRIKEN, sep=";", encoding="utf-8-sig")

    df_metriken_best_params_per_session = pd.DataFrame(metriken_best_params_per_session)
    df_metriken_best_params_per_session.to_csv(OUTFILE_METRIKEN_PER_SESSION, sep=";", encoding="utf-8-sig")
    # print(df_metriken_best_params_per_Session)

    plot_and_save_clustered_segs(
        result["session_name"],
        result["session_speaker_segments"],
        result["true_clusters"],
        result["pred_clusters"]
    )

