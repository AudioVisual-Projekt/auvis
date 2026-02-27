"""
src/semantic/services/semantic_clustering_service.py

Schritt 4 (Semantic): Clustering *pro Session* auf Basis einer *vorberechneten*
Distanzmatrix (distance_matrix.npy) mittels AgglomerativeClustering (metric="precomputed").

Abgrenzung / Architektur:
- Rein semantisch: Keine ASD-/Zeitlogik.
- Diese Datei enthält bewusst *keine Run-Orchestrierung* (keine Session-Iteration, kein Run-Meta).
  Das übernimmt ausschließlich: build_active_speaker_segments_semantic.py

Inputs pro Session (aus out_dir/session_*/):
- distance_matrix.npy  (n x n)
- distance_ids.json    (bindende Reihenfolge; 1:1 zu Zeilen/Spalten)
- optional: distance_meta.json

Outputs pro Session:
- speaker_to_cluster.json              (uid -> cluster_id)
- optional: semantic_clustering_meta.json
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from semantic.models.semantic_clustering_models import SemanticClusteringConfig

LOGGER = logging.getLogger(__name__)

# Input-Dateinamen (müssen zum Schritt-3-Output passen)
DISTANCE_MATRIX_NAME = "distance_matrix.npy"
DISTANCE_IDS_NAME = "distance_ids.json"
DISTANCE_META_NAME = "distance_meta.json"

# Output-Dateinamen (Schritt 4)
OUT_SPEAKER_TO_CLUSTER = "speaker_to_cluster.json"
OUT_SESSION_META = "semantic_clustering_meta.json"


# ----------------------------
# Zeit / Atomisches Schreiben
# ----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, obj: Any, *, indent: int = 2, sort_keys: bool = True) -> None:
    """
    Schreibt JSON atomisch (tmp -> os.replace), deterministisch formatiert.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path_str = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False, sort_keys=sort_keys)
        os.replace(str(tmp_path), str(path))  # atomisch (innerhalb des Filesystems)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


# ----------------------------
# Laden Inputs pro Session
# ----------------------------

def load_distance_artifacts(session_dir: Path) -> Tuple[np.ndarray, List[str], Optional[dict]]:
    """
    Lädt pro Session:
      - distance_matrix.npy
      - distance_ids.json  (Liste von Objekten mit mindestens {"uid": ...})
      - optional distance_meta.json (dict)

    Rückgabe:
      (D, uids, distance_meta_dict_or_None)
    """
    p_matrix = session_dir / DISTANCE_MATRIX_NAME
    p_ids = session_dir / DISTANCE_IDS_NAME
    p_meta = session_dir / DISTANCE_META_NAME

    if not p_matrix.is_file():
        raise FileNotFoundError(f"[{session_dir.name}] Pflichtdatei fehlt: {p_matrix}")
    if not p_ids.is_file():
        raise FileNotFoundError(f"[{session_dir.name}] Pflichtdatei fehlt: {p_ids}")

    D = np.load(p_matrix)

    ids_obj = json.loads(p_ids.read_text(encoding="utf-8"))
    if not isinstance(ids_obj, list):
        raise ValueError(f"[{session_dir.name}] {p_ids.name} ist keine Liste.")

    uids: List[str] = []
    for i, item in enumerate(ids_obj):
        if not isinstance(item, dict):
            raise ValueError(f"[{session_dir.name}] ids[{i}] ist kein Objekt (dict).")
        uid = item.get("uid")
        if not isinstance(uid, str) or not uid.strip():
            raise ValueError(f"[{session_dir.name}] ids[{i}].uid fehlt oder ist ungültig.")
        uids.append(uid)

    meta: Optional[dict] = None
    if p_meta.is_file():
        meta_obj = json.loads(p_meta.read_text(encoding="utf-8"))
        if isinstance(meta_obj, dict):
            meta = meta_obj

    return D, uids, meta


# ----------------------------
# Validierung
# ----------------------------

def validate_distance_matrix(D: np.ndarray, uids: List[str], *, session_name: str, strict: bool) -> None:
    """
    Validierung für D:
    - quadratisch
    - Shape passt zu uids
    - optional (strict): NaN/Inf, Diagonale ~0, Symmetrie ~
    """
    if not isinstance(D, np.ndarray):
        raise TypeError(f"[{session_name}] distance_matrix ist kein np.ndarray.")

    if D.ndim != 2:
        raise ValueError(f"[{session_name}] distance_matrix hat ndim={D.ndim}, erwartet 2.")

    n0, n1 = D.shape
    if n0 != n1:
        raise ValueError(f"[{session_name}] distance_matrix ist nicht quadratisch: {D.shape}.")
    if len(uids) != n0:
        raise ValueError(
            f"[{session_name}] IDs passen nicht zur Matrix: len(distance_ids)={len(uids)} vs n={n0}."
        )

    if not strict or n0 == 0:
        return

    if not np.isfinite(D).all():
        raise ValueError(f"[{session_name}] distance_matrix enthält NaN/Inf.")

    diag = np.diag(D)
    if not np.allclose(diag, 0.0, atol=1e-6):
        max_diag = float(np.max(np.abs(diag)))
        raise ValueError(f"[{session_name}] Diagonale ist nicht ~0 (max |diag|={max_diag}, tol=1e-6).")

    if not np.allclose(D, D.T, atol=1e-6):
        diff = float(np.max(np.abs(D - D.T)))
        raise ValueError(f"[{session_name}] distance_matrix ist nicht symmetrisch (max diff={diff}, tol=1e-6).")


def check_distance_meta(
    distance_meta: Optional[dict],
    config: SemanticClusteringConfig,
    *,
    session_name: str,
) -> None:
    """
    Optionale Meta-Prüfung:
    - Wenn expected_distance_type gesetzt ist und Meta vorhanden: muss matchen.
    """
    if distance_meta is None:
        return
    expected = config.expected_distance_type
    if expected is None:
        return

    actual = distance_meta.get("distance_type")
    if actual is None:
        LOGGER.warning("[%s] distance_meta.json ohne 'distance_type' (expected=%s).", session_name, expected)
        return
    if str(actual) != str(expected):
        raise ValueError(f"[{session_name}] distance_type mismatch: meta={actual!r} expected={expected!r}.")


# ----------------------------
# Clustering (Determinismus)
# ----------------------------

def build_agglomerative_model(config: SemanticClusteringConfig) -> AgglomerativeClustering:
    """
    Erstellt das scikit-learn Clustering-Objekt.
    Aktiviert compute_distances=True für Confidence-Berechnung.
    """
    kwargs: Dict[str, Any] = {
        "linkage": config.linkage,
        "distance_threshold": config.distance_threshold,
        "n_clusters": config.n_clusters,
        "compute_distances": True  # WICHTIG: Erlaubt Zugriff auf model.distances_
    }

    # Für distance_threshold ist compute_full_tree in manchen Versionen nötig/empfohlen
    if config.distance_threshold is not None:
        kwargs["compute_full_tree"] = True

    try:
        return AgglomerativeClustering(metric="precomputed", **kwargs)
    except TypeError:
        return AgglomerativeClustering(affinity="precomputed", **kwargs)


def renumber_cluster_labels(labels: np.ndarray) -> np.ndarray:
    """
    Deterministische Renummerierung:
    - Cluster werden nach dem kleinsten Index eines Elements im Cluster sortiert.
    - Neue Labels: 0..K-1 in dieser Reihenfolge.
    """
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        return labels

    clusters: Dict[int, List[int]] = {}
    for i, lbl in enumerate(labels.tolist()):
        clusters.setdefault(lbl, []).append(i)

    order = sorted(clusters.keys(), key=lambda lbl: min(clusters[lbl]))
    remap = {old: new for new, old in enumerate(order)}
    return np.asarray([remap[int(lbl)] for lbl in labels.tolist()], dtype=int)


def cluster_session(
    D: np.ndarray,
    uids: List[str],
    config: SemanticClusteringConfig,
    *,
    session_name: str,
) -> Tuple[Dict[str, int], float, float]:
    """
    Führt Agglomerative Clustering auf einer vorgegebenen Distanzmatrix durch.
    
    Rückgabe:
        (Mapping uid->cluster_id, max_dist_used, session_confidence)
    """
    n = len(uids)

    # Edge-Cases: konsistent definiert
    if n == 0:
        return {}, 0.0, 1.0
    if n == 1:
        return {uids[0]: 0}, 0.0, 1.0

    validate_distance_matrix(D, uids, session_name=session_name, strict=config.strict_checks)

    model = build_agglomerative_model(config)
    try:
        labels = model.fit_predict(D)
    except Exception as e:
        raise RuntimeError(f"[{session_name}] Clustering fehlgeschlagen: {e}")

    # -----------------------------------------------------------
    # Confidence Berechnung (Logik analog zu Team C / clustering_engine.py)
    # -----------------------------------------------------------
    max_dist_used = 0.0
    
    # model.children_ enthält die Merge-Schritte.
    # model.distances_ enthält die Distanzen dieser Merges.
    if hasattr(model, 'children_') and hasattr(model, 'distances_'):
        for i, (child_a, child_b) in enumerate(model.children_):
            dist = model.distances_[i]
            
            # Wenn Threshold gesetzt ist, schneidet sklearn zwar ab, aber wir prüfen sicherheitshalber
            # gegen den konfigurierten Threshold, falls full_tree berechnet wurde.
            if config.distance_threshold is not None:
                if dist > config.distance_threshold:
                    # Wir interessieren uns nur für Merges, die tatsächlich stattgefunden haben
                    # im Kontext des Thresholds. Da sklearn bei 'labels_' das Ergebnis
                    # schon geschnitten hat, ist dieser Check hier eher informativ,
                    # aber für max_dist_used relevant.
                    # HINWEIS: Bei n_clusters=None + distance_threshold bricht sklearn
                    # das Clustering ab. model.distances_ enthält dann nur die Distanzen
                    # der ausgeführten Merges (sofern compute_full_tree nicht alles erzwingt).
                    # Zur Sicherheit:
                    pass 

            # Wir nehmen das Max über alle Distanzen, die wir finden.
            # Da wir Labels von fit_predict nutzen, gehen wir davon aus, dass der 
            # Baum konsistent ist. Bei distance_threshold Filterung wäre es sauberer,
            # nur Distanzen <= threshold zu betrachten.
            if config.distance_threshold is not None and dist > config.distance_threshold:
                # Dieser Merge hat für das finale Ergebnis nicht stattgefunden 
                # (oder würde Cluster verbinden, die getrennt bleiben sollen).
                # Wir ignorieren Distanzen oberhalb des Thresholds.
                continue

            max_dist_used = max(max_dist_used, float(dist))

    # Session Confidence: 1.0 (sicher) bis 0.0 (unsicher)
    session_confidence = 1.0 - max_dist_used
    # Clipping auf [0, 1] zur Sicherheit
    session_confidence = max(0.0, min(1.0, session_confidence))

    if config.renumber_labels:
        labels = renumber_cluster_labels(labels)

    mapping = {uid: int(lbl) for uid, lbl in zip(uids, labels)}
    return mapping, max_dist_used, session_confidence


# ----------------------------
# Schreiben Outputs pro Session
# ----------------------------

def write_speaker_to_cluster(session_dir: Path, mapping: Dict[str, int]) -> Path:
    """
    Schreibt speaker_to_cluster.json atomisch.
    """
    out_path = session_dir / OUT_SPEAKER_TO_CLUSTER
    atomic_write_json(out_path, mapping, indent=2, sort_keys=True)
    return out_path


def build_session_meta(
    *,
    session_name: str,
    n: int,
    config: SemanticClusteringConfig,
    distance_meta: Optional[dict],
    max_merge_dist: float,
    confidence: float
) -> Dict[str, Any]:
    """
    Baut den Meta-Block für semantic_clustering_meta.json (Session-spezifisch).
    Erweitert um 'max_merge_distance' und 'session_confidence'.
    """
    meta: Dict[str, Any] = {
        "step": 4,
        "session": session_name,
        "created_at": utc_now_iso(),
        "n": int(n),
        "config": asdict(config),
        "inputs": {
            "distance_matrix": DISTANCE_MATRIX_NAME,
            "distance_ids": DISTANCE_IDS_NAME,
            "distance_meta_present": bool(distance_meta is not None),
        },
        # NEU: Metriken für Vergleichbarkeit
        "max_merge_distance": max_merge_dist,
        "session_confidence": confidence,
    }
    if distance_meta is not None:
        meta["distance_meta"] = distance_meta
    return meta


def write_session_meta(session_dir: Path, meta: Dict[str, Any]) -> Path:
    """
    Schreibt semantic_clustering_meta.json atomisch.
    """
    out_path = session_dir / OUT_SESSION_META
    atomic_write_json(out_path, meta, indent=2, sort_keys=True)
    return out_path


def run_clustering_for_session_dir(
    session_dir: Path,
    config: SemanticClusteringConfig,
    *,
    write_meta: bool = True,
) -> Dict[str, Any]:
    """
    Führt *nur* für eine einzelne Session das Clustering aus und schreibt Outputs.

    Rückgabe (für Orchestrator):
      {
        "session": "session_40",
        "n": 123,
        "out_mapping": "speaker_to_cluster.json",
        "out_meta": "semantic_clustering_meta.json" | None
      }
    """
    session_name = session_dir.name

    D, uids, dist_meta = load_distance_artifacts(session_dir)
    check_distance_meta(dist_meta, config, session_name=session_name)
    validate_distance_matrix(D, uids, session_name=session_name, strict=config.strict_checks)

    # Hier kommen jetzt 3 Werte zurück
    mapping, max_dist, confidence = cluster_session(D=D, uids=uids, config=config, session_name=session_name)
    
    write_speaker_to_cluster(session_dir, mapping)

    out_meta_name: Optional[str] = None
    if write_meta:
        meta = build_session_meta(
            session_name=session_name,
            n=len(uids),
            config=config,
            distance_meta=dist_meta,
            max_merge_dist=max_dist,  # Übergabe
            confidence=confidence     # Übergabe
        )
        write_session_meta(session_dir, meta)
        out_meta_name = OUT_SESSION_META

    return {
        "session": session_name,
        "n": len(uids),
        "out_mapping": OUT_SPEAKER_TO_CLUSTER,
        "out_meta": out_meta_name,
        "confidence": confidence # Kann nützlich für Logs sein
    }