import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import math
from pathlib import Path
from team_c.src.talking_detector.segmentation import CENTRAL_ASD_CHUNKING_PARAMETERS



### Hilfsfunktion
def round_up_first_decimal(x):
    return math.ceil(x * 10) / 10

def run_inference(cluster_threshold: float = 0.7): # zur Vermeidung eines zirkulären Importerrors
    from team_c.script.main import inference
    return inference(cluster_threshold)

def max_clustering_distance(row, threshold):
    """
    Berechnet die größte tatsächlich verwendete Clusteringdistanz einer Session.
    Dieser Wert ist kleiner als oder gleich dem distance-threshold.

    Args:
        row: Eine Zeile des DataFrame mit inference_results
            (Spalten: "session_name", "true_clusters", "pred_clusters", "session_scores", "session_speaker_segments")
        threshold: Distanz - threshold (= 1 - score_threshold), float
    """

    session_name = row["session_name"]
    scores = row["session_scores"]
    pred_clusters = row["pred_clusters"]
    speaker_ids = list(row['pred_clusters'].keys())

    # --- Safety parsing ---
    if isinstance(scores, str):
        try:
            scores_clean = re.sub(r'\s+', ',', scores.strip())
            scores_clean = scores_clean.replace('[,', '[').replace(',]', ']')
            scores = np.array(eval(scores_clean))
        except Exception:
            print(f"⚠️ Konnte Scores für {session_name} nicht lesen.")
            return

    if isinstance(pred_clusters, str):
        try:
            pred_clusters = eval(pred_clusters)
        except Exception:
            print(f"⚠️ Konnte pred_clusters für {session_name} nicht lesen.")
            return

    scores = np.array(scores)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        print(f"⚠️ Ungültige Score-Matrix für {session_name}")
        return



    # --- Distanzmatrix & Linkage ---
    distances = 1 - scores
    condensed_distances = squareform(distances, checks=False)
    linkage_matrix = linkage(condensed_distances, method="complete")
    cluster_distances = linkage_matrix[:, 2]
    max_cluster_distance = cluster_distances[cluster_distances <= threshold].max()

    return max_cluster_distance



if __name__ == "__main__":
    train_flag = True
    dev_flag = False
    ######################################
    #### ACHTUNG:
    #### in main unter inference() das default session_dir entsprechend manuell anpassen !!!!!!!!!!!!!
    #######################################
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
        OUTPUT_DIR = BASE_DIR / "data-bin" / "output_best_params_train" / run_name
        IMAGE_DIR = OUTPUT_DIR / "dendrogramme_best_params_train"
        OUTFILE_METRIKEN = OUTPUT_DIR / "metriken_best_params_train.csv"
        OUTFILE_METRIKEN_PER_SESSION = OUTPUT_DIR / "metriken_best_params_per_session_train.csv"
        OUTFILE_RESULT_PICKLE = OUTPUT_DIR / "inference_result_train.pkl"
        OUTFILE_RESULT_CSV = OUTPUT_DIR / "inference_result_train.csv"
    elif dev_flag:
        print("Dev Data werden prozessiert")
        DATA_DIR = BASE_DIR / "data-bin" / "dev"
        OUTPUT_DIR = BASE_DIR / "data-bin" / "output_best_params_dev" / run_name
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


    for idx, row in result.iterrows():
        max_cluster_distance = max_clustering_distance(row, 1 - threshold)
        print(f"{max_cluster_distance:.3f}")
