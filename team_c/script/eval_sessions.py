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

from src.cluster.eval import pairwise_f1_score  # :contentReference[oaicite:0]{index=0}

# Pfade (analog zu main.py) :contentReference[oaicite:1]{index=1}
TEAMC_DATABIN = os.path.join(PROJECT_ROOT, "data-bin")
OUTPUT_BASE = os.path.join(TEAMC_DATABIN, "output_semantic_test")

BASELINE_DEV_GLOB = os.path.join(
    PROJECT_ROOT,
    "../../../mcorec_baseline/data-bin/dev/session_*"
)


def load_json(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def eval_session(session_dir: str) -> Dict[str, float]:
    """
    Evaluierung einer Session:
    - Ground Truth: speaker_to_cluster.json (Baseline)
    - Vorhersagen: time / semantic / hybrid (Team C Output)
    """
    session_name = os.path.basename(os.path.normpath(session_dir))

    gt_path = os.path.join(session_dir, "labels", "speaker_to_cluster.json")
    pred_dir = os.path.join(OUTPUT_BASE, session_name)

    time_path = os.path.join(pred_dir, "speaker_to_cluster_time.json")
    sem_path = os.path.join(pred_dir, "speaker_to_cluster_semantic.json")
    hyb_path = os.path.join(pred_dir, "speaker_to_cluster_hybrid.json")

    if not (os.path.isfile(gt_path)
            and os.path.isfile(time_path)
            and os.path.isfile(sem_path)
            and os.path.isfile(hyb_path)):
        print(f"[WARN] Skip {session_name}: missing files")
        return {}

    gt = load_json(gt_path)
    pred_time = load_json(time_path)
    pred_sem = load_json(sem_path)
    pred_hyb = load_json(hyb_path)

    # Sprecher-Reihenfolge fixieren
    speakers: List[str] = sorted(gt.keys())

    gt_labels = [gt[s] for s in speakers]
    time_labels = [pred_time[s] for s in speakers]
    sem_labels = [pred_sem[s] for s in speakers]
    hyb_labels = [pred_hyb[s] for s in speakers]

    def metrics(pred: List[int]):
        f1 = pairwise_f1_score(gt_labels, pred)
        ari = adjusted_rand_score(gt_labels, pred)
        return f1, ari

    f1_time, ari_time = metrics(time_labels)
    f1_sem, ari_sem = metrics(sem_labels)
    f1_hyb, ari_hyb = metrics(hyb_labels)

    return {
        "session": session_name,
        "f1_time": f1_time,
        "ari_time": ari_time,
        "f1_sem": f1_sem,
        "ari_sem": ari_sem,
        "f1_hyb": f1_hyb,
        "ari_hyb": ari_hyb,
    }


def main():
    session_dirs = [
        p for p in glob.glob(BASELINE_DEV_GLOB)
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

    # Übersicht pro Session
    print("\nPer-session scores:")
    print("session\tf1_time\tf1_sem\tf1_hyb\tari_time\tari_sem\tari_hyb")
    for r in rows:
        print(
            f"{r['session']}\t"
            f"{r['f1_time']:.3f}\t{r['f1_sem']:.3f}\t{r['f1_hyb']:.3f}\t"
            f"{r['ari_time']:.3f}\t{r['ari_sem']:.3f}\t{r['ari_hyb']:.3f}"
        )

    # Globale Mittelwerte
    f1_time_mean = np.mean([r["f1_time"] for r in rows])
    f1_sem_mean = np.mean([r["f1_sem"] for r in rows])
    f1_hyb_mean = np.mean([r["f1_hyb"] for r in rows])

    ari_time_mean = np.mean([r["ari_time"] for r in rows])
    ari_sem_mean = np.mean([r["ari_sem"] for r in rows])
    ari_hyb_mean = np.mean([r["ari_hyb"] for r in rows])

    print("\nMean scores over all sessions:")
    print(f"F1_time  = {f1_time_mean:.3f}")
    print(f"F1_sem   = {f1_sem_mean:.3f}")
    print(f"F1_hyb   = {f1_hyb_mean:.3f}")
    print(f"ARI_time = {ari_time_mean:.3f}")
    print(f"ARI_sem  = {ari_sem_mean:.3f}")
    print(f"ARI_hyb  = {ari_hyb_mean:.3f}")


if __name__ == "__main__":
    main()
