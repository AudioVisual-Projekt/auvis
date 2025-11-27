# team_c/script/conv_confidence.py

import os
import sys
from typing import Dict, List, Any

import numpy as np
import math

# Projekt-Root wie in conv_spks.py auf sys.path legen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# [Inference] Passe den Import ggf. an dein tatsächliches Package-Layout an
from auvis.team_c.src.cluster.conv_spks import (
    calculate_conversation_scores,
    cluster_speakers,
)


def compute_conversation_cluster_quality(
    scores: np.ndarray,
    speaker_ids: List[str],
    clusters_dict: Dict[str, int],
) -> Dict[str, Any]:
    """
    Berechnet eine unsupervised Qualitätsmetrik für ein bereits vorhandenes Clustering.

    Inputs:
        scores:
            NxN Similarity-Matrix, wie von calculate_conversation_scores()
            (höher = eher gleiche Konversation).
        speaker_ids:
            Liste der Speaker-IDs in der Reihenfolge der Matrix-Indices.
        clusters_dict:
            Mapping {speaker_id -> cluster_label} wie von cluster_speakers().

    Output-Dict:
        {
          "n_speakers": int,
          "n_valid_speakers": int,
          "n_clusters": int,
          "cluster_sizes": {cluster_label: size, ...},
          "within_mean": float | None,
          "between_mean": float | None,
          "quality_within_minus_between": float | None,
          "silhouette_mean": float | None,
          "silhouette_per_speaker": {spk_id: silhouette | None, ...},
          "silhouette_per_cluster": {cluster_label: mean_silhouette | None, ...},
        }

    Idee:
        - within_mean: mittlere Similarity innerhalb der Cluster
        - between_mean: mittlere Similarity zwischen Clustern
        - quality_within_minus_between: within_mean - between_mean
        - silhouette_mean: Silhouette-ähnlicher Score, basierend auf Similarity
    """

    scores = np.asarray(scores, dtype=float)
    n = scores.shape[0]

    if scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be square NxN")

    if len(speaker_ids) != n:
        raise ValueError("len(speaker_ids) must match scores.shape[0]")

    # Map speaker_id -> Index
    spk_to_idx = {spk: i for i, spk in enumerate(speaker_ids)}

    # Label-Array (ein Eintrag pro Speaker)
    labels = np.full(n, -1, dtype=int)
    for spk, lbl in clusters_dict.items():
        if spk not in spk_to_idx:
            # ignorieren, falls Clustering Speaker enthält, die nicht in speaker_ids sind
            continue
        labels[spk_to_idx[spk]] = int(lbl)

    # Nur Speaker mit gültigem Label
    valid_mask = labels >= 0
    if not np.any(valid_mask):
        return {
            "n_speakers": int(n),
            "n_valid_speakers": 0,
            "n_clusters": 0,
            "cluster_sizes": {},
            "within_mean": None,
            "between_mean": None,
            "quality_within_minus_between": None,
            "silhouette_mean": None,
            "silhouette_per_speaker": {},
            "silhouette_per_cluster": {},
        }

    # Mapping cluster -> Indices
    cluster_to_indices: Dict[int, List[int]] = {}
    for idx, lbl in enumerate(labels):
        if lbl < 0:
            continue
        cluster_to_indices.setdefault(int(lbl), []).append(idx)

    # --- Within / Between Similarities ---

    within_vals: List[float] = []
    between_vals: List[float] = []

    for i in range(n):
        li = labels[i]
        if li < 0:
            continue
        for j in range(i + 1, n):
            lj = labels[j]
            if lj < 0:
                continue
            if li == lj:
                within_vals.append(float(scores[i, j]))
            else:
                between_vals.append(float(scores[i, j]))

    def _mean_or_none(arr: List[float]) -> float | None:
        return float(np.mean(arr)) if arr else None

    within_mean = _mean_or_none(within_vals)
    between_mean = _mean_or_none(between_vals)

    quality = None
    if within_mean is not None and between_mean is not None:
        quality = float(within_mean - between_mean)

    # --- Silhouette-ähnlicher Score (auf Similarity-Basis) ---

    silhouettes = np.full(n, np.nan, dtype=float)

    for c, idxs in cluster_to_indices.items():
        if len(idxs) == 1:
            # Singletons: definieren Silhouette = 0
            i = idxs[0]
            silhouettes[i] = 0.0
            continue

        other_clusters = [cc for cc in cluster_to_indices.keys() if cc != c]

        for i in idxs:
            # a(i): mittlere Similarity zu anderen im gleichen Cluster
            same = [j for j in idxs if j != i]
            if same:
                a = float(np.mean(scores[i, same]))
            else:
                a = math.nan

            # b(i): maximale mittlere Similarity zu anderen Clustern
            b_vals: List[float] = []
            for oc in other_clusters:
                oidxs = cluster_to_indices[oc]
                if not oidxs:
                    continue
                b_vals.append(float(np.mean(scores[i, oidxs])))
            b = max(b_vals) if b_vals else math.nan

            if math.isnan(a) or math.isnan(b) or (a <= 0 and b <= 0):
                silhouettes[i] = 0.0
            else:
                denom = max(a, b)
                silhouettes[i] = (a - b) / denom if denom != 0 else 0.0

    valid_sil = silhouettes[labels >= 0]
    sil_mean = float(np.mean(valid_sil)) if valid_sil.size > 0 else None

    per_spk_sil: Dict[str, float | None] = {}
    for i, spk in enumerate(speaker_ids):
        if labels[i] < 0:
            continue
        val = silhouettes[i]
        per_spk_sil[spk] = float(val) if not math.isnan(val) else None

    per_cluster_sil: Dict[int, float | None] = {}
    for c, idxs in cluster_to_indices.items():
        vals = [silhouettes[i] for i in idxs if not math.isnan(silhouettes[i])]
        per_cluster_sil[int(c)] = float(np.mean(vals)) if vals else None

    return {
        "n_speakers": int(n),
        "n_valid_speakers": int(np.sum(labels >= 0)),
        "n_clusters": int(len(cluster_to_indices)),
        "cluster_sizes": {int(c): int(len(idxs)) for c, idxs in cluster_to_indices.items()},
        "within_mean": within_mean,
        "between_mean": between_mean,
        "quality_within_minus_between": quality,
        "silhouette_mean": sil_mean,
        "silhouette_per_speaker": per_spk_sil,
        "silhouette_per_cluster": per_cluster_sil,
    }


def run_baseline_with_quality(
    speaker_segments: Dict[str, List[tuple]],
    threshold: float = 0.7,
    n_clusters: int | None = None,
    algo: str = "agglomerative",
    algo_kwargs: dict | None = None,
) -> Dict[str, Any]:
    """
    Convenience-Wrapper:
      1) berechnet Konversations-Scores (ASD-basierte Baseline)
      2) führt Clustering durch (cluster_speakers)
      3) berechnet die Qualitätsmetrik (compute_conversation_cluster_quality)

    Returns:
      {
        "scores": np.ndarray,         # NxN Similarity
        "speaker_ids": List[str],     # Reihenfolge zu scores
        "clusters": Dict[str, int],   # speaker_id -> cluster_label
        "quality": { ... },           # Output von compute_conversation_cluster_quality
      }
    """
    algo_kwargs = algo_kwargs or {}

    # 1) Scores
    scores = calculate_conversation_scores(speaker_segments)
    speaker_ids = list(speaker_segments.keys())

    # 2) Clustering
    clusters = cluster_speakers(
        scores=scores,
        speaker_ids=speaker_ids,
        threshold=threshold,
        n_clusters=n_clusters,
        algo=algo,
        algo_kwargs=algo_kwargs,
    )

    # 3) Qualitätsmetrik
    quality = compute_conversation_cluster_quality(
        scores=scores,
        speaker_ids=speaker_ids,
        clusters_dict=clusters,
    )

    return {
        "scores": scores,
        "speaker_ids": speaker_ids,
        "clusters": clusters,
        "quality": quality,
    }
