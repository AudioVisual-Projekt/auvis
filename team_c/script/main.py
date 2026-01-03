# /home/ercel001/AUVIS/task5_semantic_prototype/auvis/team_c/script/main.py
# Pipeline (dev):
#  1) Texte aus Baseline-Output-VTTs extrahieren  -> dev_texts.json
#  2) Sentence-Embeddings erzeugen               -> E.npy, ids.json
#  3) Cosine-Distanzmatrix bauen                 -> D.npy
#  4) Clustering: Zeit (Baseline) + Semantik     -> speaker_to_cluster.json je Session
#  5) Evaluation aller Sessions                  -> F1 / ARI (Zeit, Semantik)
#  6) Evaluationsergebnis mit Zeitstempel        -> evaluation_YYYYMMDD_HHMMSS.txt

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.cluster.conv_spks import (
    get_speaker_activity_segments,
    calculate_conversation_scores,
    cluster_speakers,
    get_clustering_f1_score,
    get_clustering_ari_score,
)
from src.semantic.semantic_clustering import semantic_cluster_speakers


# -------------------------
# Pfade (Team C)
# -------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAMC_DATABIN = os.path.join(PROJECT_ROOT, "data-bin")

# Baseline-Kopie (dein lokaler Mirror der Baseline-Session-Struktur)
BASELINE_DEV_ROOT = os.path.join(TEAMC_DATABIN, "baseline", "dev")

# Output-Roots (getrennt: Zeit vs. Semantik)
OUT_DEV_ROOT = os.path.join(TEAMC_DATABIN, "_output", "dev")
OUT_TIME_ROOT = os.path.join(OUT_DEV_ROOT, "asd_clustering")
OUT_SEM_ROOT = os.path.join(OUT_DEV_ROOT, "semantik_clustering")

# Semantik-Artefakte (global, liegen im Root von OUT_SEM_ROOT)
DEV_TEXTS_JSON = os.path.join(OUT_SEM_ROOT, "dev_texts.json")
IDS_JSON = os.path.join(OUT_SEM_ROOT, "ids.json")
E_NPY = os.path.join(OUT_SEM_ROOT, "E.npy")
D_NPY = os.path.join(OUT_SEM_ROOT, "D.npy")  # session_distances.py default erwartet "D.npy"


# -------------------------
# Step 1: Text-Extraktion
# -------------------------
def _read_vtt_file(path: str) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("WEBVTT"):
                continue
            if "-->" in line:
                continue
            if line.isdigit():  # optionale Cue-IDs
                continue
            lines.append(line)
    return lines


def extract_texts_from_baseline_outputs(
    baseline_split_root: str,
    output_dir_name: str = "output",
) -> Dict[str, List[str]]:
    """
    Liest System-Hypothesen aus:
      <baseline_split_root>/session_*/<output_dir_name>/spk_*.vtt

    Erzeugt:
      {"ids":[...], "texts":[...]}
      mit id = "{session}_{spk}", z. B. "session_40_spk_0"
    """
    baseline_split_root = os.path.abspath(os.path.expanduser(baseline_split_root))
    if not os.path.isdir(baseline_split_root):
        raise FileNotFoundError(f"Baseline split root nicht gefunden: {baseline_split_root}")

    all_ids: List[str] = []
    all_texts: List[str] = []

    sessions = sorted(
        s for s in os.listdir(baseline_split_root)
        if s.startswith("session_") and os.path.isdir(os.path.join(baseline_split_root, s))
    )

    for session in sessions:
        out_dir = os.path.join(baseline_split_root, session, output_dir_name)
        if not os.path.isdir(out_dir):
            # Für diese Session gibt es (noch) keine VTT-Hypothesen -> skip
            continue

        vtts = sorted(
            f for f in os.listdir(out_dir)
            if f.endswith(".vtt") and f.startswith("spk_")
        )

        for fn in vtts:
            spk = fn[:-4]  # remove ".vtt" -> "spk_0"
            fp = os.path.join(out_dir, fn)

            text = " ".join(_read_vtt_file(fp)).strip()
            if not text:
                continue

            all_ids.append(f"{session}_{spk}")
            all_texts.append(text)

    return {"ids": all_ids, "texts": all_texts}


# -------------------------
# Step 2: Embeddings
# -------------------------
def build_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    """
    Erzeugt Sentence-Embeddings für eine Liste von Texten.
    Rückgabe: E mit Shape (N, dim)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    # Keine Gradienten erforderlich (Inference)
    torch.set_grad_enabled(False)

    embs: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        embs.append(
            model.encode(
                batch,
                batch_size=len(batch),
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
        )

    return np.vstack(embs)


# -------------------------
# Step 3: Distanzmatrix
# -------------------------
def compute_blockwise_cosine_distance(
    E: np.ndarray,
    block_size: int = 512,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Baut eine vollständige Cosine-Distanzmatrix D (N x N) aus E (N x d).

    D[i, j] = 1 - cos_sim(E[i], E[j])

    Blockweise Berechnung über Zeilen (Speicher-/Zeit-Kompromiss).
    """
    E = E.astype(np.float32, copy=False)
    N = E.shape[0]

    norms = np.linalg.norm(E, axis=1, keepdims=True).astype(np.float32)
    D = np.empty((N, N), dtype=np.float32)

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        E_block = E[start:end]
        norms_block = norms[start:end]

        sim = E_block @ E.T
        denom = norms_block @ norms.T
        sim = sim / (denom + eps)

        D[start:end, :] = (1.0 - sim).astype(np.float32)

    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)  # numerische Symmetrie
    return D.astype(np.float32)


# -------------------------
# Step 4: Clustering je Session
# -------------------------
def _load_metadata(session_dir: str) -> dict:
    p = os.path.join(session_dir, "metadata.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_speaker_segments(session_dir: str, metadata: dict) -> Dict[str, List[Tuple[float, float]]]:
    """
    WICHTIG: Wir erzwingen eine deterministische Sprecher-Reihenfolge (spk_0, spk_1, ...),
    damit die interne Reihenfolge in calculate_conversation_scores() konsistent ist.
    """
    speaker_segments: Dict[str, List[Tuple[float, float]]] = {}

    for speaker_name in sorted(metadata.keys()):
        speaker_data = metadata[speaker_name]
        list_tracks_asd = [
            os.path.join(session_dir, track["asd"])
            for track in speaker_data["central"]["crops"]
        ]
        uem_start = speaker_data["central"]["uem"]["start"]
        uem_end = speaker_data["central"]["uem"]["end"]

        speaker_segments[speaker_name] = get_speaker_activity_segments(
            list_tracks_asd, uem_start, uem_end
        )

    return speaker_segments


def run_time_and_semantic_clustering_for_all_sessions(
    baseline_split_root: str,
    out_time_root: str,
    out_sem_root: str,
    sem_data_dir: str,
) -> List[str]:
    baseline_split_root = os.path.abspath(os.path.expanduser(baseline_split_root))
    sessions = sorted(
        s for s in os.listdir(baseline_split_root)
        if s.startswith("session_") and os.path.isdir(os.path.join(baseline_split_root, s))
    )

    processed: List[str] = []
    for session in sessions:
        session_dir = os.path.join(baseline_split_root, session)

        # Output-Pfade je Session
        time_session_dir = os.path.join(out_time_root, session)
        sem_session_dir = os.path.join(out_sem_root, session)
        os.makedirs(time_session_dir, exist_ok=True)
        os.makedirs(sem_session_dir, exist_ok=True)

        out_time_path = os.path.join(time_session_dir, "speaker_to_cluster.json")
        out_sem_path = os.path.join(sem_session_dir, "speaker_to_cluster.json")

        metadata = _load_metadata(session_dir)
        speaker_segments = _build_speaker_segments(session_dir, metadata)
        speaker_ids = list(speaker_segments.keys())

        # (4a) Zeitcluster (Baseline)
        scores_time = calculate_conversation_scores(speaker_segments)
        clusters_time = cluster_speakers(scores_time, speaker_ids)
        with open(out_time_path, "w", encoding="utf-8") as f:
            json.dump(clusters_time, f, indent=2, ensure_ascii=False)

        # (4b) Semantikcluster (liest ids.json + D.npy aus sem_data_dir = OUT_SEM_ROOT)
        clusters_sem = semantic_cluster_speakers(
            data_dir=sem_data_dir,
            session_name=session,
            speaker_names=speaker_ids,
        )
        with open(out_sem_path, "w", encoding="utf-8") as f:
            json.dump(clusters_sem, f, indent=2, ensure_ascii=False)

        processed.append(session)

    return processed


# -------------------------
# Step 5+6: Evaluation
# -------------------------
def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_all_sessions(
    baseline_split_root: str,
    out_time_root: str,
    out_sem_root: str,
) -> str:
    baseline_split_root = os.path.abspath(os.path.expanduser(baseline_split_root))
    sessions = sorted(
        s for s in os.listdir(baseline_split_root)
        if s.startswith("session_") and os.path.isdir(os.path.join(baseline_split_root, s))
    )

    rows: List[str] = []
    f1_time_all: List[float] = []
    ari_time_all: List[float] = []
    f1_sem_all: List[float] = []
    ari_sem_all: List[float] = []

    for session in sessions:
        gt_path = os.path.join(baseline_split_root, session, "labels", "speaker_to_cluster.json")
        pred_time_path = os.path.join(out_time_root, session, "speaker_to_cluster.json")
        pred_sem_path = os.path.join(out_sem_root, session, "speaker_to_cluster.json")

        if not (os.path.isfile(gt_path) and os.path.isfile(pred_time_path) and os.path.isfile(pred_sem_path)):
            continue

        gt = _load_json(gt_path)
        pred_time = _load_json(pred_time_path)
        pred_sem = _load_json(pred_sem_path)

        f1_time = get_clustering_f1_score(pred_time, gt)
        ari_time = get_clustering_ari_score(pred_time, gt)
        f1_sem = get_clustering_f1_score(pred_sem, gt)
        ari_sem = get_clustering_ari_score(pred_sem, gt)

        f1_time_all.append(f1_time)
        ari_time_all.append(ari_time)
        f1_sem_all.append(f1_sem)
        ari_sem_all.append(ari_sem)

        rows.append(
            f"{session}: "
            f"time(F1={f1_time:.4f}, ARI={ari_time:.4f}) | "
            f"sem(F1={f1_sem:.4f}, ARI={ari_sem:.4f})"
        )

    def _avg(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    summary = [
        "=== Evaluation Summary (dev) ===",
        f"N_sessions_evaluated: {len(f1_time_all)}",
        f"AVG time:     F1={_avg(f1_time_all):.4f}  ARI={_avg(ari_time_all):.4f}",
        f"AVG semantic: F1={_avg(f1_sem_all):.4f}  ARI={_avg(ari_sem_all):.4f}",
        "",
        "=== Per Session ===",
        *rows,
        "",
    ]
    return "\n".join(summary)


def main() -> None:
    ap = argparse.ArgumentParser(description="Team C Pipeline (dev): texts -> embeddings -> D -> clustering -> eval")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--force_rebuild", action="store_true", help="Artefakte neu erzeugen (dev_texts/E/D).")
    args = ap.parse_args()

    os.makedirs(OUT_DEV_ROOT, exist_ok=True)
    os.makedirs(OUT_TIME_ROOT, exist_ok=True)
    os.makedirs(OUT_SEM_ROOT, exist_ok=True)

    # (1) dev_texts.json
    if args.force_rebuild or not os.path.isfile(DEV_TEXTS_JSON):
        data = extract_texts_from_baseline_outputs(BASELINE_DEV_ROOT, output_dir_name="output")
        with open(DEV_TEXTS_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # (2) E.npy, ids.json
    if args.force_rebuild or not (os.path.isfile(E_NPY) and os.path.isfile(IDS_JSON)):
        d = _load_json(DEV_TEXTS_JSON)
        ids = d.get("ids", [])
        texts = d.get("texts", [])

        if not (isinstance(ids, list) and isinstance(texts, list) and len(ids) == len(texts)):
            raise ValueError("dev_texts.json hat ein unerwartetes Format (erwarte ids/texts gleicher Länge).")

        # Embeddings hängen nur an den Texten; ids werden separat persistent gemacht
        E = build_embeddings(texts, args.model, args.batch, args.normalize)

        np.save(E_NPY, E)
        with open(IDS_JSON, "w", encoding="utf-8") as f:
            json.dump(ids, f, indent=2, ensure_ascii=False)

    # (3) D.npy
    if args.force_rebuild or not os.path.isfile(D_NPY):
        E = np.load(E_NPY)
        D = compute_blockwise_cosine_distance(E, block_size=args.block_size)
        np.save(D_NPY, D)

    # (4) Clustering pro Session (Zeit + Semantik), getrennte Output-Roots
    run_time_and_semantic_clustering_for_all_sessions(
        baseline_split_root=BASELINE_DEV_ROOT,
        out_time_root=OUT_TIME_ROOT,
        out_sem_root=OUT_SEM_ROOT,
        sem_data_dir=OUT_SEM_ROOT,  # wichtig: ids.json + D.npy liegen im Root von OUT_SEM_ROOT
    )

    # (5) Evaluation
    report = evaluate_all_sessions(
        baseline_split_root=BASELINE_DEV_ROOT,
        out_time_root=OUT_TIME_ROOT,
        out_sem_root=OUT_SEM_ROOT,
    )

    # (6) Report speichern (Timestamp)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(OUT_DEV_ROOT, f"evaluation_{ts}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"→ Report gespeichert: {report_path}")


if __name__ == "__main__":
    main()
