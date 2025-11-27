# team_c/script/eval_asd_overlap_baseline.py

import os
import sys
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import adjusted_rand_score  # optional, nur falls du querchecken willst

# PROJECT_ROOT = .../team_c
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

TEAMC_DATABIN = os.path.join(PROJECT_ROOT, "data-bin", "dev")
OUTPUT_BASE = os.path.join(TEAMC_DATABIN, "output_asd_overlap")

import auvis.team_c.src.cluster.conv_spks as baseline


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gt_labels(session_dir: str) -> Dict[str, int]:
    """
    Ground-Truth:
        <session_dir>/labels/speaker_to_cluster.json
      Format:
        { "spk_0": 0, "spk_1": 0, "spk_2": 1, ... }
    """
    gt_path = os.path.join(session_dir, "labels", "speaker_to_cluster.json")
    if not os.path.isfile(gt_path):
        return {}
    data = load_json(gt_path)
    gt: Dict[str, int] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not k.startswith("spk_"):
            continue
        try:
            gt[k] = int(v)
        except (TypeError, ValueError):
            continue
    return gt


def find_speaker_asd_paths(session_dir: str) -> Dict[str, List[str]]:
    """
    ASD-Scores pro Speaker finden.

    Erwartetes Layout (wie dein Beispiel):
        <session_dir>/speakers/spk_0/central_crops/track_00_asd.json
        <session_dir>/speakers/spk_1/central_crops/track_00_asd.json
        ...

    Rückgabe:
        { "spk_0": [ ".../track_00_asd.json", ... ], ... }
    """
    speakers_root = os.path.join(session_dir, "speakers")
    if not os.path.isdir(speakers_root):
        return {}

    spk_to_paths: Dict[str, List[str]] = {}
    for spk_dir in sorted(glob.glob(os.path.join(speakers_root, "spk_*"))):
        spk_id = os.path.basename(spk_dir)
        cc_dir = os.path.join(spk_dir, "central_crops")
        if not os.path.isdir(cc_dir):
            continue
        # [Inference] robust: alle *_asd.json unter central_crops nehmen
        paths = sorted(glob.glob(os.path.join(cc_dir, "*_asd.json")))
        if paths:
            spk_to_paths[spk_id] = paths
    return spk_to_paths


def infer_uem_bounds(spk_asd_paths: Dict[str, List[str]]) -> Tuple[float, float]:
    """
    [Inference] UEM-Fenster aus ASD-Frames ableiten.

    Wir nehmen die min/max-Frame-ID über alle Speaker,
    wandeln mit 25 fps in Sekunden um (wie in get_speaker_activity_segments),
    und geben (uem_start, uem_end) zurück.
    """
    min_frame = None
    max_frame = None

    for paths in spk_asd_paths.values():
        for asd_path in paths:
            data = load_json(asd_path)
            for k in data.keys():
                try:
                    fi = int(k)
                except ValueError:
                    continue
                if min_frame is None or fi < min_frame:
                    min_frame = fi
                if max_frame is None or fi > max_frame:
                    max_frame = fi

    if max_frame is None:
        return 0.0, 0.0

    fps = 25.0
    if min_frame is None:
        min_frame = 0
    uem_start = float(min_frame) / fps
    uem_end = float(max_frame) / fps
    return uem_start, uem_end


def build_speaker_segments(
    spk_asd_paths: Dict[str, List[str]],
    uem_start: float,
    uem_end: float,
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Für jeden Speaker:
      ASD-JSONs -> get_speaker_activity_segments -> Zeitsegmente.
    """
    segments: Dict[str, List[Tuple[float, float]]] = {}
    for spk_id, paths in spk_asd_paths.items():
        segs = baseline.get_speaker_activity_segments(
            pycrop_asd_path=paths,
            uem_start=uem_start,
            uem_end=uem_end,
        )
        segments[spk_id] = segs
    return segments


def cluster_session(
    session_dir: str,
    n_clusters: int | None = None,
) -> Dict[str, int]:
    """
    Voller Overlap-Baseline-Step für EINE Session:

      1. ASD-JSONs finden
      2. Segmente via get_speaker_activity_segments
      3. conversation scores via calculate_conversation_scores
      4. cluster_speakers (Agglomerative, average, k = n_clusters)
    """
    spk_asd_paths = find_speaker_asd_paths(session_dir)
    if not spk_asd_paths:
        raise RuntimeError(f"No ASD JSONs found in {session_dir}")

    uem_start, uem_end = infer_uem_bounds(spk_asd_paths)
    speaker_segments = build_speaker_segments(spk_asd_paths, uem_start, uem_end)

    baseline.validate_speaker_segments(speaker_segments)

    scores = baseline.calculate_conversation_scores(speaker_segments)
    speaker_ids = list(speaker_segments.keys())

    clusters = baseline.cluster_speakers(
        scores=scores,
        speaker_ids=speaker_ids,
        n_clusters=n_clusters,          # für Eval: k = #GT-Cluster
        algo="agglomerative",
        algo_kwargs={"linkage": "average"},
    )
    return clusters


def eval_session(session_dir: str) -> Dict[str, float]:
    """
    Eine Session auswerten:
      - GT laden (speaker_to_cluster.json)
      - Overlap-Baseline laufen lassen
      - F1 / ARI berechnen
      - Pred-Cluster als JSON wegschreiben
    """
    session_name = os.path.basename(os.path.normpath(session_dir))

    gt_labels = load_gt_labels(session_dir)
    if not gt_labels:
        print(f"[WARN] Skip {session_name}: missing ground-truth labels")
        return {}

    # k = #GT-Cluster
    true_clusters = list(set(gt_labels.values()))
    n_clusters = len(true_clusters)

    try:
        pred_clusters = cluster_session(session_dir, n_clusters=n_clusters)
    except Exception as e:
        print(f"[WARN] Skip {session_name}: error in clustering: {e}")
        return {}

    # Nur Speaker nutzen, die sowohl in GT als auch in Predikt sind
    speakers = sorted(set(gt_labels.keys()) & set(pred_clusters.keys()))
    if len(speakers) < 2:
        print(f"[WARN] Skip {session_name}: not enough overlapping speakers")
        return {}

    # F1 / ARI mit denselben Helpern wie dein Kollege
    f1 = baseline.get_clustering_f1_score(
        conversation_clusters_label={s: pred_clusters[s] for s in speakers},
        true_clusters_label={s: gt_labels[s] for s in speakers},
    )
    ari = baseline.get_clustering_ari_score(
        conversation_clusters_label={s: pred_clusters[s] for s in speakers},
        true_clusters_label={s: gt_labels[s] for s in speakers},
    )

    # Vorhersage für spätere Auswertungen speichern
    out_dir = os.path.join(OUTPUT_BASE, session_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "speaker_to_cluster_asd_overlap.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pred_clusters, f, indent=2)

    return {
        "session": session_name,
        "f1_overlap": float(f1),
        "ari_overlap": float(ari),
        "n_speakers": len(speakers),
        "n_clusters": int(n_clusters),
    }


def main():
    session_dirs = [
        p for p in glob.glob(os.path.join(TEAMC_DATABIN, "session_*"))
        if os.path.isdir(p)
    ]
    session_dirs = sorted(session_dirs)

    print(f"Found {len(session_dirs)} sessions to evaluate (ASD overlap baseline).")

    rows = []
    for sd in session_dirs:
        res = eval_session(sd)
        if res:
            rows.append(res)

    if not rows:
        print("No sessions evaluated.")
        return

    # Per-Session-Overview
    print("\nPer-session scores:")
    print("session\tf1_overlap\tari_overlap\tn_speakers\tn_clusters")
    for r in rows:
        print(
            f"{r['session']}\t"
            f"{r['f1_overlap']:.3f}\t{r['ari_overlap']:.3f}\t"
            f"{r['n_speakers']}\t{r['n_clusters']}"
        )

    # Mittelwerte
    f1_mean = float(np.mean([r["f1_overlap"] for r in rows]))
    ari_mean = float(np.mean([r["ari_overlap"] for r in rows]))

    print("\nMean scores over all sessions:")
    print(f"F1_overlap = {f1_mean:.3f}")
    print(f"ARI_overlap = {ari_mean:.3f}")


if __name__ == "__main__":
    main()
