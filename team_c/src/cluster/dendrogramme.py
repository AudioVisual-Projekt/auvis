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

def plot_colored_dendrograms(row, threshold, dendro_dir, parametersatz):
    """
    Erzeugt Dendrogramme für alle Sessions eines bestimmten Grid-Search-Laufs,
    farblich markiert nach den vorhergesagten Clustern.

    Args:
        row: Eine Zeile des DataFrame mit inference_results
            (Spalten: "session_name", "true_clusters", "pred_clusters", "session_scores", "session_speaker_segments")
        dendro_dir: Das Directory, in dem die Dendrogramme abgespeichert werden.
        threshold: Distanz - threshold (= 1 - score_threshold), float
        parametersatz: Info in Überschrift über die verwendeten ASD und Clustering Parameter, string
    """
    cmap = plt.get_cmap("tab10")  # 10 gut unterscheidbare Farben

    session_name = row["session_name"]
    scores = row["session_scores"]
    pred_clusters = row["pred_clusters"]
    speaker_ids = list(row['pred_clusters'].keys())
    print(f"speaker_ids: {speaker_ids}")

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
    condensed_dist = squareform(distances, checks=False)
    linkage_matrix = linkage(condensed_dist, method="complete")

    # --- Dendrogramm plotten ---
    fig, ax = plt.subplots(figsize=(10, 6))
    dendro = dendrogram(
        linkage_matrix,
        truncate_mode='lastp',
        labels=speaker_ids,
        # leaf_rotation=90,
        color_threshold=0.24,
        above_threshold_color="lightgrey",
    )

    ax.set_title(f"Dendrogramm für {session_name}\n({parametersatz})", fontsize=12)
    ax.set_xlabel("Sprecher")
    ax.set_ylabel("Distanz")
    # Automatische y-Grenzen aus Matplotlib (aktuell)
    y_min, y_max = ax.get_ylim()
    # Neue y_max wählen: mindestens Threshold, sonst Standard
    new_y_max = max(round_up_first_decimal(threshold+0.01), y_max)
    # y-Achse anpassen
    ax.set_ylim(0, new_y_max)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Distance-Threshold={threshold}')
    # Annotation hinzufügen
    plt.text(
        x=0.5,             # x-Position (relativ zur Achse)
        y=threshold + 0.02,  # leicht oberhalb der Linie
        s=f"Obere Grenze für Clusterbildung = {threshold}",
        color='r',
        fontsize=8
    )
    plt.tight_layout()

    # --- Speichern ---
    os.makedirs(dendro_dir, exist_ok=True)
    out_path = os.path.join(dendro_dir, f"{session_name}_dendrogram.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Dendrogramm gespeichert: {out_path}")



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
        plot_colored_dendrograms(row, 1 - threshold, IMAGE_DIR, parametersatz=run_name)