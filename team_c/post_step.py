# -*- coding: utf-8 -*-
"""
post_step.py

Standalone post-processing step (no CLI) that operates on already-produced per-session outputs.

It does:
1) evaluate_existing_assignments():
   - loads existing baseline predictions: speaker_to_cluster_seat_*.json (per session)
   - evaluates them against GT labels (labels/speaker_to_cluster.json)
   - writes per-session + global means into output_root/speaker_clustering_eval.json

2) build_distance_matrices():
   - builds speaker distance matrices using seat geometry + ASD seat mapping
   - optionally adjusts distances using gaze (head-yaw proxy)
   - NEW: also builds a gaze-only interaction distance matrix
   - saves per session:
        speaker_distance_seat.npy
        speaker_distance_seat_gaze.npy
        speaker_distance_gaze_only.npy
        speaker_distance_meta.json (speaker order)
     and a global JSON dump (for quick inspection):
        output_root/speaker_distance_matrices.json

3) evaluate_agglo_sweep():
   - for each session builds agglomerative clustering from the precomputed distances:
        - seat-only
        - seat+gaze
        - NEW: gaze-only
   - chooses k WITHOUT using GT by maximizing silhouette over k=2..min(4,n-1)
   - evaluates chosen clustering vs GT (ARI + pairwise F1)
   - writes predictions per session:
        speaker_to_cluster_agglo_seat_only.json
        speaker_to_cluster_agglo_seat_gaze.json
        speaker_to_cluster_agglo_gaze_only.json

4) evaluate_spectral_mutual() + evaluate_spectral_similarity():
   - builds gaze similarity graphs from meta['gaze_pair_stats']
       a) mutual=min(p(i->j), p(j->i))
       b) avg_look=0.5*(p(i->j)+p(j->i))
   - chooses k WITHOUT using GT by maximizing weighted modularity over k=2..4
   - runs SpectralClustering (precomputed affinity)
   - evaluates chosen clustering vs GT (ARI + pairwise F1)
   - writes predictions per session:
        speaker_to_cluster_spectral_mutual.json
        speaker_to_cluster_spectral_mutual_info.json
        speaker_to_cluster_spectral_similarity.json
        speaker_to_cluster_spectral_similarity_info.json

Run in PyCharm via the Run button: execute this file directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

# --------------------------
# USER CONFIG (no CLI)
# --------------------------

# If None, roots are auto-discovered relative to the repo "team_c" directory.
OUTPUT_ROOT: Optional[Path] = None  # e.g., Path(r"C:\...\team_c\data-bin\output_gaze")
LABELS_ROOT: Optional[Path] = None  # e.g., Path(r"C:\...\team_c\data-bin\dev")

# Clustering constraints
MIN_K: int = 2
MAX_K: int = 4

# Gaze integration (head-yaw proxy)
YAW_SIGN: float = 1.0
GAZE_MIN_SAMPLES: int = 10
BASE_GAZE_TOL_DEG: float = 25.0
MAX_GAZE_TOL_DEG: float = 50.0
W_MUTUAL: float = 0.45  # distance reduction if mutual gaze
W_ONEWAY: float = 0.20  # distance reduction if one-way gaze

# Spectral clustering on mutual-gaze graph (no CLI)
SPECTRAL_K_MIN: int = 2
SPECTRAL_K_MAX: int = 4
SPECTRAL_KNN: int = 2  # keep top-k edges per node to avoid disconnected graphs
SPECTRAL_GAMMA: float = 2.0  # similarity sharpening: mutual**gamma


# --------------------------
# Small utilities
# --------------------------

def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _discover_team_c_root(start: Path) -> Path:
    p = start.resolve()
    while p.name != "team_c" and p.parent != p:
        p = p.parent
    return p


def _norm_spk_id(x) -> str:
    """
    Normalize speaker identifiers to 'spk_<int>'.
    Accepts: 'spk_0', 0, '0', 'speaker_0' (best-effort).
    """
    if x is None:
        return ""
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("spk_"):
            return s
        digits = "".join(ch for ch in s if ch.isdigit() or ch == "-")
        if digits != "" and digits != "-":
            try:
                i = int(digits)
                return f"spk_{i}"
            except Exception:
                pass
        return s
    try:
        i = int(x)
        return f"spk_{i}"
    except Exception:
        return str(x)


def _intersection_keys(a: Dict[str, int], b: Dict[str, int]) -> List[str]:
    return sorted(set(a.keys()).intersection(set(b.keys())))


# --------------------------
# Metrics
# --------------------------

def pairwise_f1(gt_labels: List[int], pred_labels: List[int]) -> float:
    """
    Pairwise F1 over all speaker pairs.
    """
    n = len(gt_labels)
    if n < 2:
        return 0.0

    tp = fp = fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            gt_same = (gt_labels[i] == gt_labels[j])
            pr_same = (pred_labels[i] == pred_labels[j])
            if pr_same and gt_same:
                tp += 1
            elif pr_same and not gt_same:
                fp += 1
            elif (not pr_same) and gt_same:
                fn += 1

    if tp == 0:
        return 0.0

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if (prec + rec) == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


# --------------------------
# Geometry + ASD + Gaze loaders
# --------------------------

def load_seat_geometry(npz_path: Path) -> dict:
    """
    Expected keys in seat_geometry.npz:
      - dist_seat: NxN normalized circular seat distance
      - theta_deg: N seat angles (deg)
      - person_ids: N person IDs (ints)
    """
    dat = np.load(str(npz_path), allow_pickle=True)
    required = ("dist_seat", "theta_deg", "person_ids")
    missing = [k for k in required if k not in dat.files]
    if missing:
        raise KeyError(f"seat_geometry.npz missing keys {missing} in {npz_path}")
    return {
        "dist_seat": np.asarray(dat["dist_seat"], float),
        "theta_deg": np.asarray(dat["theta_deg"], float),
        "person_ids": np.asarray(dat["person_ids"], int),
    }


def load_asd_mapping(asd_json_path):
    with open(asd_json_path, "r", encoding="utf-8") as f:
        j = json.load(f)

    asd_ids = [int(x) for x in j.get("asd_track_ids", [])]
    asd_to_person = {}

    # Preferred: 1-to-1 assignments (Hungarian)
    assignments = j.get("assignments", None)
    if isinstance(assignments, list) and assignments and isinstance(assignments[0], dict):
        for a in assignments:
            tid = a.get("asd_track_id", None)
            pid = a.get("person_id", None)
            if tid is None:
                continue
            tid = int(tid)

            # per your requirement: unconfident -> still set
            asd_to_person[tid] = None if pid is None else int(pid)

        for tid in asd_ids:
            asd_to_person.setdefault(tid, None)

        return asd_ids, asd_to_person

    # Fallback if only assignments_all exists (not ideal)
    assignments_all = j.get("assignments_all", None)
    if isinstance(assignments_all, list) and assignments_all and isinstance(assignments_all[0], dict):
        # pick best per asd_track_id by min angle_diff_deg
        best = {}
        for a in assignments_all:
            tid = a.get("asd_track_id", None)
            pid = a.get("person_id", None)
            ang = a.get("angle_diff_deg", float("inf"))
            if tid is None:
                continue
            tid = int(tid)
            cur = best.get(tid)
            if cur is None or ang < cur["ang"]:
                best[tid] = {"pid": pid, "ang": ang}

        for tid, v in best.items():
            asd_to_person[tid] = None if v["pid"] is None else int(v["pid"])

        for tid in asd_ids:
            asd_to_person.setdefault(tid, None)

        return asd_ids, asd_to_person

    raise ValueError(f"Unrecognized ASD mapping schema in {asd_json_path}")


def load_gaze(gaze_path: Path) -> Dict[int, dict]:
    """
    Loads output_gaze/session_*/gaze_tracks.json.

    Supports both schemas:
      - summary-only: yaw_deg_median, yaw_deg_iqr, n_samples
      - sample-based: additionally yaw_deg_samples (preferred)

    Returns: person_id -> {
        "yaw_med": float|None,
        "yaw_iqr": float|None,
        "n": int,
        "yaw_samples": Optional[List[float]],   # yaw degrees samples if present
    }
    """
    if not gaze_path.exists():
        return {}

    j = _read_json(gaze_path)
    persons = j.get("persons", {}) or {}
    out: Dict[int, dict] = {}

    for pid_str, info in persons.items():
        try:
            pid = int(pid_str)
        except Exception:
            continue
        if not isinstance(info, dict):
            continue

        yaw_med = info.get("yaw_deg_median", None)
        yaw_iqr = info.get("yaw_deg_iqr", None)
        n_samp = info.get("n_samples", None)

        # NEW: full yaw sample series (if exported)
        yaw_samples_raw = info.get("yaw_deg_samples", None)
        yaw_samples = None
        if isinstance(yaw_samples_raw, list) and len(yaw_samples_raw) > 0:
            ys = []
            for v in yaw_samples_raw:
                if v is None:
                    continue
                try:
                    ys.append(float(v))
                except Exception:
                    continue
            yaw_samples = ys if ys else None

        # backward compatible fallback keys
        if yaw_med is None or n_samp is None:
            yaw_med = info.get("yaw_med", yaw_med)
            yaw_iqr = info.get("yaw_iqr", yaw_iqr)
            n_samp = info.get("n", n_samp)

        # if we have samples but missing summary, derive
        if yaw_samples is not None:
            arr = np.asarray(yaw_samples, float)
            if yaw_med is None:
                yaw_med = float(np.median(arr))
            if yaw_iqr is None:
                q75, q25 = np.percentile(arr, [75, 25])
                yaw_iqr = float(q75 - q25)
            if n_samp is None:
                n_samp = int(arr.size)

        out[pid] = {
            "yaw_med": None if yaw_med is None else float(yaw_med),
            "yaw_iqr": None if yaw_iqr is None else float(yaw_iqr),
            "n": int(n_samp) if n_samp is not None else 0,
            "yaw_samples": yaw_samples,
        }

    return out


# --------------------------
# Circular helpers
# --------------------------

def circ_abs_diff_deg(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def circ_signed_diff_deg(a: float, b: float) -> float:
    d = (b - a) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


# --------------------------
# Distance building
# --------------------------

def build_speaker_distance_matrices(
        geom: dict,
        asd_ids: List[int],
        asd_to_person: Dict[int, Optional[int]],
        gaze_by_person: Dict[int, dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Build three speaker-level distance matrices from seating geometry + optional gaze tracks.

    Matrices (MxM, M=len(asd_ids), order=asd_ids):
      - D_seat:       pure seat distance (0..1)
      - D_seat_gaze:  seat distance scaled by a continuous gaze interaction factor
      - D_gaze_only:  interaction distance only (base=1.0 scaled by gaze factor)

    Gaze evaluation (sample-first):
      For a directed pair i->j:
        p_ij = fraction of i's gaze-direction samples that fall within tol_i around the target direction to j
      mutual(i,j) = min(p_ij, p_ji)
      oneway(i,j) = max(p_ij, p_ji) - mutual(i,j)

      dist_factor = 1 - (W_MUTUAL * mutual + W_ONEWAY * oneway)
      (clipped to [0,1])

    The important bit: distances become CONTINUOUS whenever yaw_deg_samples are present,
    instead of collapsing to {1.0, 1-W_ONEWAY, 1-W_MUTUAL}.
    """
    dist_seat = np.asarray(geom["dist_seat"], float)  # NxN normalized
    theta_deg = np.asarray(geom["theta_deg"], float)  # N
    person_ids = np.asarray(geom["person_ids"], int)  # N
    pid_to_index = {int(pid): i for i, pid in enumerate(person_ids)}

    M = len(asd_ids)
    D_seat = np.zeros((M, M), float)
    D_seat_gaze = np.zeros((M, M), float)
    D_gaze_only = np.zeros((M, M), float)

    seat_idx: List[Optional[int]] = []
    gaze_dir_med_abs: List[Optional[float]] = []         # abs gaze direction from yaw median (fallback)
    gaze_dir_samples_abs: List[Optional[List[float]]] = []  # abs gaze direction samples (preferred)
    gaze_tol: List[Optional[float]] = []

    seat_assignment: Dict[str, Optional[int]] = {}
    gaze_used: Dict[str, dict] = {}

    def _frac_looks(
            samples_abs: Optional[List[float]],
            med_abs: Optional[float],
            target_abs: float,
            tol_deg: float,
    ) -> float:
        """Return fraction of gaze directions within tol of target. Fallback to median-only if samples missing."""
        if isinstance(samples_abs, list) and len(samples_abs) > 0:
            hits = 0
            for gdir in samples_abs:
                if circ_abs_diff_deg(float(gdir), float(target_abs)) <= float(tol_deg):
                    hits += 1
            return hits / float(len(samples_abs))
        if med_abs is None:
            return 0.0
        return 1.0 if circ_abs_diff_deg(float(med_abs), float(target_abs)) <= float(tol_deg) else 0.0

    # --------------------
    # Precompute per speaker
    # --------------------
    for sid in asd_ids:
        spk = _norm_spk_id(sid)

        pid = asd_to_person.get(int(sid), None)
        idx = pid_to_index.get(int(pid), None) if pid is not None else None

        seat_idx.append(idx)
        seat_assignment[spk] = None if pid is None else int(pid)

        g = gaze_by_person.get(int(pid), None) if (pid is not None) else None
        if (pid is None) or (idx is None) or (not g) or (g.get("n", 0) < GAZE_MIN_SAMPLES):
            gaze_dir_med_abs.append(None)
            gaze_dir_samples_abs.append(None)
            gaze_tol.append(None)
            continue

        yaw_med = g.get("yaw_med", None)
        yaw_iqr = g.get("yaw_iqr", None)
        yaw_samples = g.get("yaw_samples", None)

        # tolerance = base + 0.5*IQR (clipped)
        iqr = float(yaw_iqr) if yaw_iqr is not None else 0.0
        tol = float(np.clip(BASE_GAZE_TOL_DEG + 0.5 * iqr, BASE_GAZE_TOL_DEG, MAX_GAZE_TOL_DEG))

        # median abs gaze dir (fallback)
        if yaw_med is not None:
            gd_med = float((theta_deg[idx] + YAW_SIGN * float(yaw_med)) % 360.0)
        else:
            gd_med = None

        # sample abs gaze dirs (preferred)
        gds = None
        if isinstance(yaw_samples, list) and len(yaw_samples) > 0:
            tmp = []
            for ys in yaw_samples:
                if ys is None:
                    continue
                try:
                    tmp.append(float((theta_deg[idx] + YAW_SIGN * float(ys)) % 360.0))
                except Exception:
                    continue
            gds = tmp if tmp else None

        gaze_dir_med_abs.append(gd_med)
        gaze_dir_samples_abs.append(gds)
        gaze_tol.append(tol)

        gaze_used[spk] = {
            "person_id": int(pid),
            "seat_theta_deg": float(theta_deg[idx]),
            "yaw_med_deg": None if yaw_med is None else float(yaw_med),
            "yaw_iqr_deg": None if yaw_iqr is None else float(yaw_iqr),
            "tol_deg": float(tol),
            "n": int(g.get("n", 0)),
            "has_samples": bool(gds is not None),
        }

    # --------------------
    # Build matrices + per-pair stats
    # --------------------
    pair_stats: List[dict] = []
    mutual_vals: List[float] = []
    avg_vals: List[float] = []

    for i in range(M):
        for j in range(M):
            if i == j:
                continue

            ii = seat_idx[i]
            jj = seat_idx[j]

            # Seat-only base
            base_seat = 1.0 if (ii is None or jj is None) else float(dist_seat[ii, jj])
            D_seat[i, j] = base_seat

            # Default factors
            dist_factor = 1.0
            p_ij = p_ji = mutual = avg_look = 0.0

            if ii is not None and jj is not None:
                # target direction i->j (absolute)
                dir_ij = circ_signed_diff_deg(theta_deg[ii], theta_deg[jj])
                target_i = float((theta_deg[ii] + dir_ij) % 360.0)

                # target direction j->i
                dir_ji = circ_signed_diff_deg(theta_deg[jj], theta_deg[ii])
                target_j = float((theta_deg[jj] + dir_ji) % 360.0)

                ti = gaze_tol[i] if gaze_tol[i] is not None else BASE_GAZE_TOL_DEG
                tj = gaze_tol[j] if gaze_tol[j] is not None else BASE_GAZE_TOL_DEG

                p_ij = _frac_looks(gaze_dir_samples_abs[i], gaze_dir_med_abs[i], target_i, ti)
                p_ji = _frac_looks(gaze_dir_samples_abs[j], gaze_dir_med_abs[j], target_j, tj)

                mutual = min(p_ij, p_ji)
                avg_look = 0.5 * (p_ij + p_ji)
                oneway = max(p_ij, p_ji) - mutual

                dist_factor = 1.0 - (W_MUTUAL * mutual + W_ONEWAY * oneway)
                dist_factor = float(np.clip(dist_factor, 0.0, 1.0))

                pair_stats.append({
                    "spk_i": _norm_spk_id(asd_ids[i]),
                    "spk_j": _norm_spk_id(asd_ids[j]),
                    "p_i_looks_j": float(p_ij),
                    "p_j_looks_i": float(p_ji),
                    "mutual": float(mutual),
                    "avg_look": float(avg_look),
                    "tol_i_deg": float(ti),
                    "tol_j_deg": float(tj),
                })
                mutual_vals.append(mutual)
                avg_vals.append(avg_look)

            D_seat_gaze[i, j] = float(np.clip(base_seat * dist_factor, 0.0, 1.0))
            D_gaze_only[i, j] = float(np.clip(1.0 * dist_factor, 0.0, 1.0))

    # Symmetrize (Agglo expects symmetric distances)
    D_seat = 0.5 * (D_seat + D_seat.T)
    D_seat_gaze = 0.5 * (D_seat_gaze + D_seat_gaze.T)
    D_gaze_only = 0.5 * (D_gaze_only + D_gaze_only.T)

    np.fill_diagonal(D_seat, 0.0)
    np.fill_diagonal(D_seat_gaze, 0.0)
    np.fill_diagonal(D_gaze_only, 0.0)

    def _summary(vals: List[float]) -> dict:
        if not vals:
            return {"n": 0, "p50": None, "p75": None, "p90": None, "p95": None, "mean": None}
        arr = np.asarray(vals, float)
        return {
            "n": int(arr.size),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "mean": float(np.mean(arr)),
        }

    meta = {
        "speaker_order": [_norm_spk_id(sid) for sid in asd_ids],
        "asd_ids": [int(x) for x in asd_ids],
        "seat_assignment_person_id": seat_assignment,  # spk_* -> person_id (or None)
        "gaze_used": gaze_used,
        "gaze_pair_stats": pair_stats,
        "gaze_mutual_summary": _summary(mutual_vals),
        "gaze_avglook_summary": _summary(avg_vals),
        "params": {
            "yaw_sign": float(YAW_SIGN),
            "gaze_min_samples": int(GAZE_MIN_SAMPLES),
            "base_gaze_tol_deg": float(BASE_GAZE_TOL_DEG),
            "max_gaze_tol_deg": float(MAX_GAZE_TOL_DEG),
            "w_mutual": float(W_MUTUAL),
            "w_oneway": float(W_ONEWAY),
            "distance_factor": "1 - (W_MUTUAL*mutual + W_ONEWAY*(max(p_ij,p_ji)-mutual))",
            "note": "Distances become continuous when yaw_deg_samples exist; median-only fallback yields discrete steps.",
        }
    }
    return D_seat, D_seat_gaze, D_gaze_only, meta


# --------------------------
# GT + prediction loaders
# --------------------------

def load_gt_clusters(labels_root: Path, session: str) -> Dict[str, int]:
    p = labels_root / session / "labels" / "speaker_to_cluster.json"
    if not p.exists():
        return {}
    j = _read_json(p)
    out: Dict[str, int] = {}
    if isinstance(j, dict):
        for k, v in j.items():
            sid = _norm_spk_id(k)
            try:
                out[sid] = int(v)
            except Exception:
                continue
    return out


def load_pred_clusters(pred_path: Path) -> Dict[str, int]:
    j = _read_json(pred_path)
    out: Dict[str, int] = {}
    if isinstance(j, dict):
        for k, v in j.items():
            sid = _norm_spk_id(k)
            try:
                out[sid] = int(v)
            except Exception:
                continue
    return out


# --------------------------
# Agglomerative clustering + k selection (NO GT)
# --------------------------

def agglo_labels_from_precomputed(D: np.ndarray, k: int) -> np.ndarray:
    n = int(D.shape[0])
    if k > n:
        k = n
    if k < 2:
        k = 2 if n >= 2 else 1

    try:
        model = AgglomerativeClustering(
            n_clusters=int(k),
            metric="precomputed",
            linkage="average",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=int(k),
            affinity="precomputed",
            linkage="average",
        )
    return model.fit_predict(D)


def choose_k_without_gt(D: np.ndarray, k_min: int, k_max: int) -> int:
    """
    Choose k purely from D by maximizing silhouette (metric='precomputed').

    Hard guarantee:
      - never returns None
      - always returns an int in [2, min(k_max, n)] if n>=2
    """
    n = int(D.shape[0])
    if n < 2:
        return 1

    k_min_eff = max(2, int(k_min))
    k_max_eff = min(int(k_max), n)

    if k_min_eff > k_max_eff:
        # If n==2, both become 2; otherwise clamp
        return int(max(2, min(n, k_max_eff)))

    best_k: Optional[int] = None
    best_s = -1e9

    for k in range(k_min_eff, k_max_eff + 1):
        try:
            labels = agglo_labels_from_precomputed(D, k=int(k))
            # silhouette needs at least 2 clusters and no singleton-only weirdness
            if len(set(labels.tolist())) < 2:
                continue
            s = float(silhouette_score(D, labels, metric="precomputed"))
        except Exception:
            continue

        if (s > best_s) or (abs(s - best_s) < 1e-12 and (best_k is None or k < best_k)):
            best_s = s
            best_k = k

    # Fallback if *everything* failed
    if best_k is None:
        return int(k_min_eff)

    return int(best_k)



# --------------------------
# Spectral clustering on mutual-gaze similarity graph (NO GT for k-choice)
# --------------------------

def build_mutual_similarity_knn(meta: dict, knn: int = 2, gamma: float = 2.0) -> np.ndarray:
    """
    Build symmetric similarity S from meta['gaze_pair_stats'] using mutual gaze:
        mutual = min(p_i_looks_j, p_j_looks_i)

    Uses best mutual per unordered pair, applies sharpening mutual**gamma,
    then optional undirected kNN sparsification.
    """
    spk_order = meta.get("speaker_order", [])
    n = len(spk_order)
    idx = {spk: i for i, spk in enumerate(spk_order)}
    S = np.zeros((n, n), dtype=float)

    pair_stats = meta.get("gaze_pair_stats", [])
    if not isinstance(pair_stats, list) or n == 0:
        return S

    best = {}
    for r in pair_stats:
        a = r.get("spk_i")
        b = r.get("spk_j")
        m = r.get("mutual", None)
        if a not in idx or b not in idx or a == b or m is None:
            continue
        key = tuple(sorted((a, b)))
        best[key] = max(float(m), best.get(key, 0.0))

    for (a, b), m in best.items():
        i, j = idx[a], idx[b]
        w = float(m) ** float(gamma)
        S[i, j] = w
        S[j, i] = w

    np.fill_diagonal(S, 0.0)

    # kNN sparsify (undirected)
    if knn is not None and knn > 0 and n > 2:
        S_knn = np.zeros_like(S)
        for i in range(n):
            nbrs = np.argsort(S[i])[::-1]
            nbrs = [j for j in nbrs if j != i and S[i, j] > 0]
            for j in nbrs[:knn]:
                S_knn[i, j] = S[i, j]
                S_knn[j, i] = max(S_knn[j, i], S[i, j])
        S = S_knn

    return S


def build_similarity_avg_knn(meta: dict, knn: int = 2, gamma: float = 2.0) -> np.ndarray:
    """
    Build symmetric similarity S from meta['gaze_pair_stats'] using symmetric gaze similarity:
        sim = 0.5*(p_i_looks_j + p_j_looks_i)

    Less strict than mutual, so graphs don't collapse to empty when gaze is mostly one-way.
    Uses best(sim) per unordered pair, applies sim**gamma, then optional undirected kNN sparsification.
    """
    spk_order = meta.get("speaker_order", [])
    n = len(spk_order)
    idx = {spk: i for i, spk in enumerate(spk_order)}
    S = np.zeros((n, n), dtype=float)

    pair_stats = meta.get("gaze_pair_stats", [])
    if not isinstance(pair_stats, list) or n == 0:
        return S

    best = {}
    for r in pair_stats:
        a = r.get("spk_i")
        b = r.get("spk_j")
        pij = r.get("p_i_looks_j", None)
        pji = r.get("p_j_looks_i", None)
        if a not in idx or b not in idx or a == b or pij is None or pji is None:
            continue
        sim = 0.5 * (float(pij) + float(pji))
        key = tuple(sorted((a, b)))
        best[key] = max(sim, best.get(key, 0.0))

    for (a, b), sim in best.items():
        i, j = idx[a], idx[b]
        w = float(sim) ** float(gamma)
        S[i, j] = w
        S[j, i] = w

    np.fill_diagonal(S, 0.0)

    if knn is not None and knn > 0 and n > 2:
        S_knn = np.zeros_like(S)
        for i in range(n):
            nbrs = np.argsort(S[i])[::-1]
            nbrs = [j for j in nbrs if j != i and S[i, j] > 0]
            for j in nbrs[:knn]:
                S_knn[i, j] = S[i, j]
                S_knn[j, i] = max(S_knn[j, i], S[i, j])
        S = S_knn

    return S


def modularity_weighted(S: np.ndarray, labels: np.ndarray) -> float:
    """Weighted modularity Q for undirected weighted graph with adjacency S."""
    S = np.asarray(S, float)
    labels = np.asarray(labels, int)
    m = S.sum() / 2.0
    if m <= 0:
        return -1.0
    k = S.sum(axis=1)  # weighted degrees
    n = int(S.shape[0])

    Q = 0.0
    for i in range(n):
        for j in range(n):
            if labels[i] != labels[j]:
                continue
            Q += S[i, j] - (k[i] * k[j]) / (2.0 * m)
    Q /= (2.0 * m)
    return float(Q)


def spectral_cluster_auto_k(S: np.ndarray, k_min: int = 2, k_max: int = 4) -> Tuple[np.ndarray, dict]:
    """Choose k in [k_min,k_max] by maximizing weighted modularity on similarity S."""
    n = int(S.shape[0])
    if n <= 1:
        return np.zeros((n,), dtype=int), {"chosen_k": 1, "reason": "n<=1"}

    k_min_eff = max(2, int(k_min))
    k_max_eff = min(int(k_max), n)

    best = {"Q": -1e9, "k": None, "labels": None}
    scores = {}

    for k in range(k_min_eff, k_max_eff + 1):
        try:
            model = SpectralClustering(
                n_clusters=int(k),
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=0,
            )
            labels = model.fit_predict(S).astype(int)
            Q = modularity_weighted(S, labels)
        except Exception:
            labels = np.zeros((n,), dtype=int)
            Q = -1e9

        scores[str(k)] = {"modularity_Q": float(Q)}
        if Q > best["Q"]:
            best = {"Q": Q, "k": k, "labels": labels}

    if best["labels"] is None:
        best["labels"] = np.zeros((n,), dtype=int)
        best["k"] = 1

    info = {
        "chosen_k": int(best["k"]),
        "scores": scores,
        "k_range": [int(k_min_eff), int(k_max_eff)],
        "knn": int(SPECTRAL_KNN),
        "gamma": float(SPECTRAL_GAMMA),
    }
    return best["labels"], info


def labels_to_pred_map(speaker_order: List[str], labels: np.ndarray) -> Dict[str, int]:
    return {speaker_order[i]: int(labels[i]) for i in range(len(speaker_order))}




# --------------------------
# Step 1: evaluate existing baseline predictions
# --------------------------

def evaluate_existing_assignments(output_root: Path, labels_root: Path) -> Tuple[List[dict], List[dict]]:
    rows: List[dict] = []
    acc: Dict[str, List[Tuple[float, float]]] = {}

    for sdir in sorted([p for p in output_root.glob("session_*") if p.is_dir()]):
        session = sdir.name
        gt = load_gt_clusters(labels_root, session)
        if not gt:
            continue

        pred_files = sorted(sdir.glob("speaker_to_cluster_seat_*.json"))
        for pf in pred_files:
            pred = load_pred_clusters(pf)
            keys = _intersection_keys(gt, pred)
            if len(keys) < 2:
                continue

            gt_labels = [gt[k] for k in keys]
            pr_labels = [pred[k] for k in keys]

            ari = float(adjusted_rand_score(gt_labels, pr_labels))
            f1 = float(pairwise_f1(gt_labels, pr_labels))

            suffix = pf.stem.replace("speaker_to_cluster_seat_", "")
            approach = f"baseline_{suffix}"

            rows.append({
                "session": session,
                "approach": approach,
                "k": int(len(set(pr_labels))),
                "n_eval": int(len(keys)),
                "ari": ari,
                "pairwise_f1": f1,
                "pred_file": str(pf),
            })
            acc.setdefault(approach, []).append((ari, f1))

    means: List[dict] = []
    for approach, vals in sorted(acc.items(), key=lambda x: x[0]):
        aris = [v[0] for v in vals]
        f1s = [v[1] for v in vals]
        means.append({
            "approach": approach,
            "n_sessions": int(len(vals)),
            "ari_mean": float(np.mean(aris)) if aris else None,
            "ari_std": float(np.std(aris)) if aris else None,
            "f1_mean": float(np.mean(f1s)) if f1s else None,
            "f1_std": float(np.std(f1s)) if f1s else None,
        })
    return rows, means


# --------------------------
# Step 2: build distance matrices (seat-only, seat+gaze, gaze-only)
# --------------------------

def build_distance_matrices(output_root: Path) -> dict:
    global_dump = {}

    for sdir in sorted([p for p in output_root.glob("session_*") if p.is_dir()]):
        session = sdir.name
        seat_npz = sdir / "seat_geometry.npz"
        asd_json = sdir / "asd_seat_matching.json"
        gaze_json = sdir / "gaze_tracks.json"

        if not seat_npz.exists() or not asd_json.exists():
            continue

        geom = load_seat_geometry(seat_npz)
        asd_ids, asd_to_person = load_asd_mapping(asd_json)
        if len(asd_ids) < 2:
            continue

        gaze_by_person = load_gaze(gaze_json) if gaze_json.exists() else {}
        D_seat, D_seat_gaze, D_gaze_only, meta = build_speaker_distance_matrices(
            geom=geom,
            asd_ids=asd_ids,
            asd_to_person=asd_to_person,
            gaze_by_person=gaze_by_person,
        )

        np.save(str(sdir / "speaker_distance_seat.npy"), D_seat)
        np.save(str(sdir / "speaker_distance_seat_gaze.npy"), D_seat_gaze)
        np.save(str(sdir / "speaker_distance_gaze_only.npy"), D_gaze_only)
        _write_json(sdir / "speaker_distance_meta.json", meta)

        # Build the two spectral graph distances (for JSON dump only):
        #   - MUTUAL graph:     S_ij = mutual**gamma (after kNN sparsification)
        #   - SIMILARITY graph: S_ij = avg_look**gamma (after kNN sparsification)
        # SpectralClustering itself consumes S (affinity). Here we store D = 1 - S (clipped) for inspection.
        S_mutual = build_mutual_similarity_knn(meta, knn=SPECTRAL_KNN, gamma=SPECTRAL_GAMMA)
        D_spec_mutual = np.clip(1.0 - S_mutual, 0.0, 1.0)
        D_spec_mutual = 0.5 * (D_spec_mutual + D_spec_mutual.T)
        np.fill_diagonal(D_spec_mutual, 0.0)

        S_similarity = build_similarity_avg_knn(meta, knn=SPECTRAL_KNN, gamma=SPECTRAL_GAMMA)
        D_spec_similarity = np.clip(1.0 - S_similarity, 0.0, 1.0)
        D_spec_similarity = 0.5 * (D_spec_similarity + D_spec_similarity.T)
        np.fill_diagonal(D_spec_similarity, 0.0)

        global_dump[session] = {
            "speaker_order": meta["speaker_order"],
            "distance_seat": D_seat.tolist(),
            "distance_seat_gaze": D_seat_gaze.tolist(),
            "distance_gaze_only": D_gaze_only.tolist(),
            # NEW: store the *spectral* distances as well (derived from the on-the-fly similarity graphs)
            # SpectralClustering uses an affinity/similarity matrix S; for inspection/debug we store D = 1 - S.
            "distance_spectral_mutual": D_spec_mutual.tolist(),
            "distance_spectral_similarity": D_spec_similarity.tolist(),
        }

    out_path = output_root / "speaker_distance_matrices.json"
    _write_json(out_path, global_dump)
    print(f"[post_step] wrote global distance JSON: {out_path}")
    return global_dump


# --------------------------
# Step 3: evaluate agglo with k chosen WITHOUT GT
# --------------------------

def evaluate_agglo_sweep(output_root: Path, labels_root: Path) -> Tuple[List[dict], List[dict]]:
    rows: List[dict] = []
    acc: Dict[str, List[Tuple[float, float]]] = {}

    for sdir in sorted([p for p in output_root.glob("session_*") if p.is_dir()]):
        session = sdir.name
        meta_path = sdir / "speaker_distance_meta.json"
        if not meta_path.exists():
            continue
        meta = _read_json(meta_path)
        spk_order = meta.get("speaker_order", [])
        if not spk_order or len(spk_order) < 2:
            continue

        d_seat_path = sdir / "speaker_distance_seat.npy"
        d_seat_gaze_path = sdir / "speaker_distance_seat_gaze.npy"
        d_gaze_only_path = sdir / "speaker_distance_gaze_only.npy"
        if not d_seat_path.exists() or not d_seat_gaze_path.exists() or not d_gaze_only_path.exists():
            continue

        D_seat = np.load(str(d_seat_path))
        D_seat_gaze = np.load(str(d_seat_gaze_path))
        D_gaze_only = np.load(str(d_gaze_only_path))

        gt = load_gt_clusters(labels_root, session)

        for approach, D in [
            ("agglo_seat_only", D_seat),
            ("agglo_seat_gaze", D_seat_gaze),
            ("agglo_gaze_only", D_gaze_only),
        ]:
            n = int(D.shape[0])
            if n < 2:
                continue

            k_min_eff = max(2, int(MIN_K))
            k_max_eff = min(int(MAX_K), n)
            if k_min_eff > k_max_eff:
                k_min_eff = k_max_eff

            chosen_k = choose_k_without_gt(D, k_min=k_min_eff, k_max=k_max_eff)
            chosen_k = int(min(max(chosen_k, 2), min(MAX_K, n)))

            pred_labels = agglo_labels_from_precomputed(D, k=chosen_k).tolist()
            pred_map = {spk_order[i]: int(pred_labels[i]) for i in range(len(spk_order))}

            pred_file = sdir / f"speaker_to_cluster_{approach}.json"
            _write_json(pred_file, pred_map)

            if gt:
                keys = _intersection_keys(gt, pred_map)
                if len(keys) >= 2:
                    gt_labels = [gt[k] for k in keys]
                    pr_labels = [pred_map[k] for k in keys]
                    ari = float(adjusted_rand_score(gt_labels, pr_labels))
                    f1 = float(pairwise_f1(gt_labels, pr_labels))
                else:
                    ari = None
                    f1 = None
            else:
                ari = None
                f1 = None

            rows.append({
                "session": session,
                "approach": approach,
                "k_chosen": int(chosen_k),
                "k_range": [int(k_min_eff), int(k_max_eff)],
                "n_total": int(n),
                "n_eval": int(len(_intersection_keys(gt, pred_map))) if gt else 0,
                "ari": ari,
                "pairwise_f1": f1,
                "pred_file": str(pred_file),
            })

            if (ari is not None) and (f1 is not None):
                acc.setdefault(approach, []).append((float(ari), float(f1)))

    means: List[dict] = []
    for approach, vals in sorted(acc.items(), key=lambda x: x[0]):
        aris = [v[0] for v in vals]
        f1s = [v[1] for v in vals]
        means.append({
            "approach": approach,
            "n_sessions": int(len(vals)),
            "ari_mean": float(np.mean(aris)) if aris else None,
            "ari_std": float(np.std(aris)) if aris else None,
            "f1_mean": float(np.mean(f1s)) if f1s else None,
            "f1_std": float(np.std(f1s)) if f1s else None,
        })
    return rows, means


# --------------------------
# Step 4: evaluate spectral clustering on mutual-gaze graph (k auto WITHOUT GT)
# --------------------------

def evaluate_spectral_mutual(output_root: Path, labels_root: Path) -> Tuple[List[dict], List[dict]]:
    """
    Spectral clustering on a MUTUAL-gaze graph:
        mutual = min(p_i_looks_j, p_j_looks_i)

    k is chosen WITHOUT GT by maximizing weighted modularity over k=2..4.
    Always writes per-session outputs (fallback single cluster if graph is empty or stats missing).
    """
    rows: List[dict] = []
    acc: Dict[str, List[Tuple[float, float]]] = {}

    for sdir in sorted([p for p in output_root.glob("session_*") if p.is_dir()]):
        session = sdir.name
        meta_path = sdir / "speaker_distance_meta.json"
        if not meta_path.exists():
            continue
        meta = _read_json(meta_path)
        spk_order = meta.get("speaker_order", [])
        if not spk_order or len(spk_order) < 2:
            continue

        n = int(len(spk_order))
        S = build_mutual_similarity_knn(meta, knn=SPECTRAL_KNN, gamma=SPECTRAL_GAMMA)

        if float(np.sum(S)) <= 0.0:
            labels = np.zeros((n,), dtype=int)
            info = {
                "chosen_k": 1,
                "reason": "empty_or_missing_mutual_graph",
                "k_range": [max(2, int(SPECTRAL_K_MIN)), min(int(SPECTRAL_K_MAX), n)],
                "knn": int(SPECTRAL_KNN),
                "gamma": float(SPECTRAL_GAMMA),
                "similarity": "mutual=min(p_i_looks_j, p_j_looks_i)",
                "sum_S": float(np.sum(S)),
            }
        else:
            k_min_eff = max(2, int(SPECTRAL_K_MIN))
            k_max_eff = min(int(SPECTRAL_K_MAX), n)
            if k_min_eff > k_max_eff:
                k_min_eff = k_max_eff
            labels, info = spectral_cluster_auto_k(S, k_min=k_min_eff, k_max=k_max_eff)
            info["similarity"] = "mutual=min(p_i_looks_j, p_j_looks_i)"
            info["sum_S"] = float(np.sum(S))

        pred_map = labels_to_pred_map(spk_order, labels)

        pred_file = sdir / "speaker_to_cluster_spectral_mutual.json"
        info_file = sdir / "speaker_to_cluster_spectral_mutual_info.json"
        _write_json(pred_file, pred_map)
        _write_json(info_file, info)

        gt = load_gt_clusters(labels_root, session)
        if gt:
            keys = _intersection_keys(gt, pred_map)
            if len(keys) >= 2:
                gt_labels = [gt[k] for k in keys]
                pr_labels = [pred_map[k] for k in keys]
                ari = float(adjusted_rand_score(gt_labels, pr_labels))
                f1 = float(pairwise_f1(gt_labels, pr_labels))
            else:
                ari = None
                f1 = None
        else:
            ari = None
            f1 = None

        approach = "spectral_mutual"
        rows.append({
            "session": session,
            "approach": approach,
            "k_chosen": int(info.get("chosen_k", len(set(labels.tolist())))),
            "k_range": info.get("k_range", [int(SPECTRAL_K_MIN), int(min(SPECTRAL_K_MAX, n))]),
            "n_total": int(n),
            "n_eval": int(len(_intersection_keys(gt, pred_map))) if gt else 0,
            "ari": ari,
            "pairwise_f1": f1,
            "pred_file": str(pred_file),
            "info_file": str(info_file),
            "sum_S": float(np.sum(S)),
        })

        if (ari is not None) and (f1 is not None):
            acc.setdefault(approach, []).append((float(ari), float(f1)))

    means: List[dict] = []
    for approach, vals in sorted(acc.items(), key=lambda x: x[0]):
        aris = [v[0] for v in vals]
        f1s = [v[1] for v in vals]
        means.append({
            "approach": approach,
            "n_sessions": int(len(vals)),
            "ari_mean": float(np.mean(aris)) if aris else None,
            "ari_std": float(np.std(aris)) if aris else None,
            "f1_mean": float(np.mean(f1s)) if f1s else None,
            "f1_std": float(np.std(f1s)) if f1s else None,
        })
    return rows, means

def evaluate_spectral_similarity(output_root: Path, labels_root: Path) -> Tuple[List[dict], List[dict]]:
    """
    Spectral clustering on a symmetric gaze similarity graph:
        sim = 0.5*(p_i_looks_j + p_j_looks_i)

    k is chosen WITHOUT GT by maximizing weighted modularity over k=2..4.
    Always writes per-session outputs (fallback single cluster if graph is empty).
    """
    rows: List[dict] = []
    acc: Dict[str, List[Tuple[float, float]]] = {}

    for sdir in sorted([p for p in output_root.glob("session_*") if p.is_dir()]):
        session = sdir.name
        meta_path = sdir / "speaker_distance_meta.json"
        if not meta_path.exists():
            continue
        meta = _read_json(meta_path)
        spk_order = meta.get("speaker_order", [])
        if not spk_order or len(spk_order) < 2:
            continue

        n = int(len(spk_order))

        S = build_similarity_avg_knn(meta, knn=SPECTRAL_KNN, gamma=SPECTRAL_GAMMA)

        if float(np.sum(S)) <= 0.0:
            labels = np.zeros((n,), dtype=int)
            info = {
                "chosen_k": 1,
                "reason": "empty_or_missing_similarity_graph",
                "k_range": [max(2, int(SPECTRAL_K_MIN)), min(int(SPECTRAL_K_MAX), n)],
                "knn": int(SPECTRAL_KNN),
                "gamma": float(SPECTRAL_GAMMA),
                "similarity": "avg_look=0.5*(p_i_looks_j+p_j_looks_i)",
            }
        else:
            k_min_eff = max(2, int(SPECTRAL_K_MIN))
            k_max_eff = min(int(SPECTRAL_K_MAX), n)
            if k_min_eff > k_max_eff:
                k_min_eff = k_max_eff
            labels, info = spectral_cluster_auto_k(S, k_min=k_min_eff, k_max=k_max_eff)
            info["similarity"] = "avg_look=0.5*(p_i_looks_j+p_j_looks_i)"

        pred_map = labels_to_pred_map(spk_order, labels)

        pred_file = sdir / "speaker_to_cluster_spectral_similarity.json"
        info_file = sdir / "speaker_to_cluster_spectral_similarity_info.json"
        _write_json(pred_file, pred_map)
        _write_json(info_file, info)

        gt = load_gt_clusters(labels_root, session)
        if gt:
            keys = _intersection_keys(gt, pred_map)
            if len(keys) >= 2:
                gt_labels = [gt[k] for k in keys]
                pr_labels = [pred_map[k] for k in keys]
                ari = float(adjusted_rand_score(gt_labels, pr_labels))
                f1 = float(pairwise_f1(gt_labels, pr_labels))
            else:
                ari = None
                f1 = None
        else:
            ari = None
            f1 = None

        approach = "spectral_similarity"
        rows.append({
            "session": session,
            "approach": approach,
            "k_chosen": int(info.get("chosen_k", len(set(labels.tolist())))),
            "k_range": info.get("k_range", [int(SPECTRAL_K_MIN), int(min(SPECTRAL_K_MAX, n))]),
            "n_total": int(n),
            "n_eval": int(len(_intersection_keys(gt, pred_map))) if gt else 0,
            "ari": ari,
            "pairwise_f1": f1,
            "pred_file": str(pred_file),
            "info_file": str(info_file),
            "sum_S": float(np.sum(S)),
        })

        if (ari is not None) and (f1 is not None):
            acc.setdefault(approach, []).append((float(ari), float(f1)))

    means: List[dict] = []
    for approach, vals in sorted(acc.items(), key=lambda x: x[0]):
        aris = [v[0] for v in vals]
        f1s = [v[1] for v in vals]
        means.append({
            "approach": approach,
            "n_sessions": int(len(vals)),
            "ari_mean": float(np.mean(aris)) if aris else None,
            "ari_std": float(np.std(aris)) if aris else None,
            "f1_mean": float(np.mean(f1s)) if f1s else None,
            "f1_std": float(np.std(f1s)) if f1s else None,
        })
    return rows, means


# --------------------------
# Ordering helpers (for eval JSON)
# --------------------------

def _session_num(s: str) -> int:
    try:
        return int(str(s).split("_")[-1])
    except Exception:
        return 10 ** 9


def _sort_eval_rows(base_rows: List[dict], agg_rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    Sort per-session rows so that for each session all approaches are consecutive,
    with a stable approach order.
    """
    APPROACH_ORDER = [
        # baselines (common)
        "baseline_dist_components",
        "baseline_dist_k2",
        "baseline_halves",
        "baseline_neighbors",
        "baseline_opposites",
        # agglo
        "agglo_seat_only",
        "agglo_seat_gaze",
        "agglo_gaze_only",
        "spectral_mutual",
        "spectral_similarity",
    ]
    rank = {a: i for i, a in enumerate(APPROACH_ORDER)}

    per_rows = base_rows + agg_rows
    per_rows = sorted(
        per_rows,
        key=lambda r: (
            _session_num(r.get("session", "")),
            rank.get(r.get("approach", ""), 10 ** 6),
            r.get("approach", ""),
        ),
    )
    return per_rows, APPROACH_ORDER


def _sort_means_rows(base_means: List[dict], agg_means: List[dict]) -> List[dict]:
    APPROACH_ORDER = [
        "baseline_dist_components",
        "baseline_dist_k2",
        "baseline_halves",
        "baseline_neighbors",
        "baseline_opposites",
        "agglo_seat_only",
        "agglo_seat_gaze",
        "agglo_gaze_only",
        "spectral_mutual",
        "spectral_similarity",
    ]
    rank = {a: i for i, a in enumerate(APPROACH_ORDER)}
    rows = base_means + agg_means
    return sorted(rows, key=lambda r: rank.get(r.get("approach", ""), 10 ** 6))


# --------------------------
# Orchestration
# --------------------------

def main() -> None:
    script_dir = Path(__file__).resolve().parent
    team_c_root = _discover_team_c_root(script_dir)

    data_root = team_c_root / "data-bin"
    output_root = OUTPUT_ROOT if OUTPUT_ROOT else (data_root / "output_gaze")
    labels_root = LABELS_ROOT if LABELS_ROOT else (data_root / "dev")

    if not output_root.exists():
        raise SystemExit(f"[post_step][ERR] output_root not found: {output_root}")
    if not labels_root.exists():
        raise SystemExit(f"[post_step][ERR] labels_root not found: {labels_root}")

    print(f"[post_step] output_root = {output_root}")
    print(f"[post_step] labels_root = {labels_root}")

    # 1) evaluate existing baselines
    base_rows, base_means = evaluate_existing_assignments(output_root, labels_root)

    # 2) build distances
    build_distance_matrices(output_root)

    # 3) agglo (k chosen without GT)
    agg_rows, agg_means = evaluate_agglo_sweep(output_root, labels_root)

    # 4) spectral (k chosen without GT via modularity)
    spec_rows, spec_means = evaluate_spectral_mutual(output_root, labels_root)
    spec_sim_rows, spec_sim_means = evaluate_spectral_similarity(output_root, labels_root)

    # 5) sort JSON: group by session, stable approach order
    per_rows, _ = _sort_eval_rows(base_rows, agg_rows + spec_rows + spec_sim_rows)
    means_rows = _sort_means_rows(base_means, agg_means + spec_means + spec_sim_means)

    out = {
        "output_root": str(output_root),
        "labels_root": str(labels_root),
        "k_constraints": {"min_k": int(MIN_K), "max_k": int(MAX_K)},
        "gaze_params": {
            "yaw_sign": float(YAW_SIGN),
            "gaze_min_samples": int(GAZE_MIN_SAMPLES),
            "base_gaze_tol_deg": float(BASE_GAZE_TOL_DEG),
            "max_gaze_tol_deg": float(MAX_GAZE_TOL_DEG),
            "w_mutual": float(W_MUTUAL),
            "w_oneway": float(W_ONEWAY),
        },
        "spectral_params": {
            "k_min": int(SPECTRAL_K_MIN),
            "k_max": int(SPECTRAL_K_MAX),
            "knn": int(SPECTRAL_KNN),
            "gamma": float(SPECTRAL_GAMMA),
            "k_choice": "maximize weighted modularity on graph similarity (precomputed affinity)",
            "graphs": {
                "spectral_mutual": "mutual=min(p(i->j), p(j->i))",
                "spectral_similarity": "avg_look=0.5*(p(i->j)+p(j->i))",
            },
        },

        "per_session": per_rows,
        "means": means_rows,
        "notes": {
            "existing_assignments": "baseline_* are evaluated from speaker_to_cluster_seat_*.json already present per session.",
            "distance_matrices": "Per session .npy are written: speaker_distance_seat.npy, speaker_distance_seat_gaze.npy, speaker_distance_gaze_only.npy (+ meta JSON).",
            "agglo_k_selection": "k is selected WITHOUT GT via silhouette_score on precomputed distances over k=2..min(4,n-1).",
            "spectral_k_selection": "k is selected WITHOUT GT via modularity maximization over k=2..4 on the gaze graph (kNN sparsified); evaluated for both mutual and avg_look similarities.",
            "metrics": "ARI + pairwise F1 computed against GT for the intersection of speakers in prediction and GT.",
            "gaze_only": "Agglo gaze-only clusters from an interaction distance (base=1.0 reduced by mutual/one-way gaze). Seat angles are used only to define gaze targets.",
        }
    }

    out_path = output_root / "speaker_clustering_eval.json"
    _write_json(out_path, out)

    print(f"[post_step] wrote eval JSON: {out_path}")
    print(
        f"[post_step] baseline rows: {len(base_rows)} | agglo rows: {len(agg_rows)} | spectral(mutu) rows: {len(spec_rows)} | spectral(sim) rows: {len(spec_sim_rows)}")


if __name__ == "__main__":
    main()
