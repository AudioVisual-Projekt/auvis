import os
import sys
import json
import glob
from typing import Dict, List

import numpy as np
from sklearn.metrics import adjusted_rand_score

# src auf sys.path legen (analog zu main.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.cluster.eval import pairwise_f1_score  # type: ignore

# Pfade (analog zu main.py, aber für dein team_c-Setup)
TEAMC_DATABIN = os.path.join(PROJECT_ROOT, "data-bin", "dev")
OUTPUT_BASE = os.path.join(TEAMC_DATABIN, "output")

# Sessions mit Ground-Truth-Labels
SESSION_DEV_GLOB = os.path.join(TEAMC_DATABIN, "session_*")


def load_json(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def eval_session(session_dir: str) -> Dict[str, float]:
    """
    Evaluierung einer Session:
    - Ground Truth: speaker_to_cluster.json (Baseline-Labels)
    - Vorhersagen: Sitz-basierte Heuristiken aus seating_speaker_baselines.py
      * speaker_to_cluster_seat_neighbors.json
      * speaker_to_cluster_seat_opposites.json
      * speaker_to_cluster_seat_halves.json
      * speaker_to_cluster_seat_dist_components.json
      * speaker_to_cluster_seat_dist_k2.json
    """
    session_name = os.path.basename(os.path.normpath(session_dir))

    # Ground Truth
    gt_path = os.path.join(session_dir, "labels", "speaker_to_cluster.json")

    # Predictions (Output deiner Sitz-Baselines)
    pred_dir = os.path.join(OUTPUT_BASE, session_name)

    neighbors_path = os.path.join(pred_dir, "speaker_to_cluster_seat_neighbors.json")
    opposites_path = os.path.join(pred_dir, "speaker_to_cluster_seat_opposites.json")
    halves_path = os.path.join(pred_dir, "speaker_to_cluster_seat_halves.json")
    dist_comp_path = os.path.join(pred_dir, "speaker_to_cluster_seat_dist_components.json")
    dist_k2_path = os.path.join(pred_dir, "speaker_to_cluster_seat_dist_k2.json")

    if not os.path.isfile(gt_path):
        print(f"[WARN] Skip {session_name}: missing GT {gt_path}")
        return {}

    if not (os.path.isfile(neighbors_path)
            and os.path.isfile(opposites_path)
            and os.path.isfile(halves_path)
            and os.path.isfile(dist_comp_path)
            and os.path.isfile(dist_k2_path)):
        print(f"[WARN] Skip {session_name}: missing baseline files in {pred_dir}")
        return {}

    gt = load_json(gt_path)
    pred_neighbors = load_json(neighbors_path)
    pred_opposites = load_json(opposites_path)
    pred_halves = load_json(halves_path)
    pred_dist_comp = load_json(dist_comp_path)
    pred_dist_k2 = load_json(dist_k2_path)

    # Sprecher-Reihenfolge fixieren (wie beim Kollegen)
    speakers: List[str] = sorted(gt.keys())

    # Robustheitscheck: alle Preds müssen für alle GT-Speaker ein Label haben
    for s in speakers:
        if (s not in pred_neighbors
                or s not in pred_opposites
                or s not in pred_halves
                or s not in pred_dist_comp
                or s not in pred_dist_k2):
            print(f"[WARN] Skip {session_name}: missing prediction for speaker {s}")
            return {}

    gt_labels = [gt[s] for s in speakers]
    neighbors_labels = [pred_neighbors[s] for s in speakers]
    opposites_labels = [pred_opposites[s] for s in speakers]
    halves_labels = [pred_halves[s] for s in speakers]
    dist_comp_labels = [pred_dist_comp[s] for s in speakers]
    dist_k2_labels = [pred_dist_k2[s] for s in speakers]

    def metrics(pred: List[int]):
        f1 = pairwise_f1_score(gt_labels, pred)
        ari = adjusted_rand_score(gt_labels, pred)
        return f1, ari

    f1_neighbors, ari_neighbors = metrics(neighbors_labels)
    f1_opposites, ari_opposites = metrics(opposites_labels)
    f1_halves, ari_halves = metrics(halves_labels)
    f1_dist_comp, ari_dist_comp = metrics(dist_comp_labels)
    f1_dist_k2, ari_dist_k2 = metrics(dist_k2_labels)

    return {
        "session": session_name,
        "f1_neighbors": f1_neighbors,
        "ari_neighbors": ari_neighbors,
        "f1_opposites": f1_opposites,
        "ari_opposites": ari_opposites,
        "f1_halves": f1_halves,
        "ari_halves": ari_halves,
        "f1_dist_components": f1_dist_comp,
        "ari_dist_components": ari_dist_comp,
        "f1_dist_k2": f1_dist_k2,
        "ari_dist_k2": ari_dist_k2,
    }


def main():
    session_dirs = [
        p for p in glob.glob(SESSION_DEV_GLOB)
        if os.path.isdir(p)
    ]
    session_dirs = sorted(session_dirs)

    print(f"Found {len(session_dirs)} sessions to evaluate.")

    rows = []
    for sd in session_dirs:
        res = eval_session(sd)
        if res:
            rows.append(res)

    if not rows:
        print("No sessions evaluated.")
        return

    # Übersicht pro Session (analog zum Kollegen, nur mit mehr Spalten)
    print("\nPer-session scores:")
    print(
        "session\t"
        "f1_neighbors\tf1_opposites\tf1_halves\tf1_dist_components\tf1_dist_k2\t"
        "ari_neighbors\tari_opposites\tari_halves\tari_dist_components\tari_dist_k2"
    )
    for r in rows:
        print(
            f"{r['session']}\t"
            f"{r['f1_neighbors']:.3f}\t{r['f1_opposites']:.3f}\t{r['f1_halves']:.3f}\t"
            f"{r['f1_dist_components']:.3f}\t{r['f1_dist_k2']:.3f}\t"
            f"{r['ari_neighbors']:.3f}\t{r['ari_opposites']:.3f}\t{r['ari_halves']:.3f}\t"
            f"{r['ari_dist_components']:.3f}\t{r['ari_dist_k2']:.3f}"
        )

    # Globale Mittelwerte
    f1_neighbors_mean = np.mean([r["f1_neighbors"] for r in rows])
    f1_opposites_mean = np.mean([r["f1_opposites"] for r in rows])
    f1_halves_mean = np.mean([r["f1_halves"] for r in rows])
    f1_dist_comp_mean = np.mean([r["f1_dist_components"] for r in rows])
    f1_dist_k2_mean = np.mean([r["f1_dist_k2"] for r in rows])

    ari_neighbors_mean = np.mean([r["ari_neighbors"] for r in rows])
    ari_opposites_mean = np.mean([r["ari_opposites"] for r in rows])
    ari_halves_mean = np.mean([r["ari_halves"] for r in rows])
    ari_dist_comp_mean = np.mean([r["ari_dist_components"] for r in rows])
    ari_dist_k2_mean = np.mean([r["ari_dist_k2"] for r in rows])

    print("\nMean scores over all sessions:")
    print(f"F1_neighbors        = {f1_neighbors_mean:.3f}")
    print(f"F1_opposites        = {f1_opposites_mean:.3f}")
    print(f"F1_halves           = {f1_halves_mean:.3f}")
    print(f"F1_dist_components  = {f1_dist_comp_mean:.3f}")
    print(f"F1_dist_k2          = {f1_dist_k2_mean:.3f}")
    print(f"ARI_neighbors       = {ari_neighbors_mean:.3f}")
    print(f"ARI_opposites       = {ari_opposites_mean:.3f}")
    print(f"ARI_halves          = {ari_halves_mean:.3f}")
    print(f"ARI_dist_components = {ari_dist_comp_mean:.3f}")
    print(f"ARI_dist_k2         = {ari_dist_k2_mean:.3f}")


if __name__ == "__main__":
    main()
