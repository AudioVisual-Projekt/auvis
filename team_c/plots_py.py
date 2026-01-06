# -*- coding: utf-8 -*-
"""
Standalone evaluation script: Distance bins vs GT for distance matrices.

Reads:
- <OUTPUT_ROOT>/speaker_distance_matrices.json

Creates (for multiple distance keys):
- Per-session + per-anchor stacked bar histograms over distance bins (bin width = BIN_WIDTH)
  * Y-axis is PERCENT of all anchor-other pairs in that plot (not absolute)
  * Title includes: correct/incorrect counts and percentages
- Global aggregated plot over all sessions
  * Additionally annotates per-bin green% and red% ABOVE the stacked bar (non-overlapping)
- Extra plot: ONLY pairs with distance < THRESH (default SESSION_LT_THRESH), aggregated per session
  * Sessions with 0 pairs below THRESH are excluded
  * Y-axis is percent within-threshold (stacked green/red to 100%)
- summary.json files with counts + fractions

No CLI. Edit OUTPUT_ROOT / LABELS_ROOT below and run the file in PyCharm.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Headless-safe plotting (works in PyCharm too)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# USER CONFIG
# =========================
OUTPUT_ROOT = Path(r"C:\Users\seana\PycharmProjects\auvis\team_c\data-bin\output_gaze_plots_spectral")
LABELS_ROOT = Path(r"C:\Users\seana\PycharmProjects\auvis\team_c\data-bin\dev")

BIN_WIDTH = 0.025
MAKE_PER_SPEAKER_PLOTS = True
OUT_DIRNAME = "gaze_distance_gt_bins"

# Extra plot: only consider pairs with distance < THRESH (ignore the rest completely)
SESSION_LT_THRESH = 0.2

# If you want to restrict sessions, list them here like ["session_132", "session_40"]. Otherwise keep None.
SESSIONS: Optional[List[str]] = None

# Which distance matrices to evaluate (must exist as keys per session in speaker_distance_matrices.json)
# - distance_spectral_mutual / distance_spectral_similarity: your spectral clustering distances
# - distance_gaze_only: gaze-only distance matrix
# - distance_seat_gaze: seat + gaze mix distance matrix
DISTANCE_KEYS: List[str] = [
    "distance_spectral_mutual",
    "distance_spectral_similarity",
    "distance_gaze_only",
    "distance_seat_gaze",
]


# =========================
# IO / HELPERS
# =========================
def _read_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _norm_spk_id(x) -> str:
    """
    Normalizes speaker IDs to the form "spk_<int>".
    Accepts:
      - int (e.g., 3)
      - "3"
      - "spk_3"
      - "spk03" (will try to parse)
    """
    if x is None:
        return ""
    if isinstance(x, int):
        return f"spk_{x}"
    s = str(x).strip()
    if not s:
        return ""
    if s.startswith("spk_"):
        tail = s[4:]
        try:
            return f"spk_{int(tail)}"
        except Exception:
            return s
    try:
        return f"spk_{int(s)}"
    except Exception:
        return s


def load_gt_clusters(labels_root: Path, session: str) -> Dict[str, int]:
    """
    Loads GT mapping speaker_id -> cluster_id for a session.

    Tries multiple common locations / filenames.
    Returns {} if nothing found.
    """
    sess_dir = labels_root / session
    candidates = [
        sess_dir / "labels" / "speaker_to_cluster.json",
        sess_dir / "speaker_to_cluster.json",
        sess_dir / "labels" / "clusters.json",
        sess_dir / "clusters.json",
        sess_dir / "labels" / "speaker_clusters.json",
        sess_dir / "speaker_clusters.json",
    ]

    gt_path = next((p for p in candidates if p.exists()), None)
    if gt_path is None:
        return {}

    j = _read_json(gt_path)
    if not isinstance(j, dict):
        return {}

    out: Dict[str, int] = {}
    for k, v in j.items():
        spk = _norm_spk_id(k)
        try:
            out[spk] = int(v)
        except Exception:
            if isinstance(v, dict) and "cluster" in v:
                try:
                    out[spk] = int(v["cluster"])
                except Exception:
                    pass
    return out


def _bin_setup(bin_width: float) -> Tuple[np.ndarray, np.ndarray, int]:
    edges = np.arange(0.0, 1.0 + bin_width, bin_width)
    centers = (edges[:-1] + edges[1:]) / 2.0
    n_bins = len(centers)
    return edges, centers, n_bins


def _bin_index(d: float, bin_width: float, n_bins: int) -> int:
    if d <= 0.0:
        return 0
    if d >= 1.0:
        return n_bins - 1
    idx = int(d / bin_width)
    return min(max(idx, 0), n_bins - 1)


def _pct_stacked(g: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    """
    Convert counts to percentages of total pairs in this plot.
    Returns: g_pct, r_pct, total_sum, green_sum, red_sum
    """
    total = g + r
    total_sum = int(total.sum())
    green_sum = int(g.sum())
    red_sum = int(r.sum())
    if total_sum <= 0:
        return np.zeros_like(g, float), np.zeros_like(r, float), 0, 0, 0
    g_pct = 100.0 * (g.astype(float) / float(total_sum))
    r_pct = 100.0 * (r.astype(float) / float(total_sum))
    return g_pct, r_pct, total_sum, green_sum, red_sum


def _ensure_key_exists(distance_key: str, sess_obj: dict) -> bool:
    return isinstance(sess_obj, dict) and (distance_key in sess_obj)


# =========================
# MAIN PLOTTING
# =========================
def plot_distance_bins_vs_gt(
    output_root: Path,
    labels_root: Path,
    distance_key: str,
    bin_width: float = 0.025,
    sessions: Optional[List[str]] = None,
    make_per_speaker_plots: bool = True,
    out_dirname: str = "gaze_distance_gt_bins",
) -> Dict[str, object]:
    """
    Per-session and global stacked bar histograms over distance bins.
    Y-axis is PERCENT of all anchor-other pairs in that plot (not absolute).
    """
    distance_key = (distance_key or "").strip()
    if not distance_key:
        raise ValueError("distance_key must be a non-empty key name")

    dist_json_path = output_root / "speaker_distance_matrices.json"
    if not dist_json_path.exists():
        raise FileNotFoundError(f"Missing distance json: {dist_json_path}")

    all_dist = _read_json(dist_json_path)
    sess_list = sessions if sessions is not None else sorted(all_dist.keys())

    edges, centers, n_bins = _bin_setup(bin_width)

    global_green = np.zeros(n_bins, dtype=int)
    global_red = np.zeros(n_bins, dtype=int)

    per_session_summary: Dict[str, object] = {}

    out_root = output_root / out_dirname / distance_key
    out_root.mkdir(parents=True, exist_ok=True)

    for session in sess_list:
        sess_obj = all_dist.get(session, None)
        if not _ensure_key_exists(distance_key, sess_obj):
            continue

        speaker_order = [_norm_spk_id(x) for x in sess_obj.get("speaker_order", [])]
        if not speaker_order:
            continue

        D = np.asarray(sess_obj[distance_key], dtype=float)
        if D.ndim != 2 or D.shape[0] != D.shape[1] or D.shape[0] != len(speaker_order):
            continue

        gt_map = load_gt_clusters(labels_root, session)
        gt_available = [spk for spk in speaker_order if spk in gt_map]
        if len(gt_available) < 2:
            continue

        idx_of = {spk: i for i, spk in enumerate(speaker_order)}

        sess_green = np.zeros(n_bins, dtype=int)
        sess_red = np.zeros(n_bins, dtype=int)

        if make_per_speaker_plots:
            session_out_dir = output_root / session / out_dirname / distance_key
            session_out_dir.mkdir(parents=True, exist_ok=True)

        anchor_summaries: Dict[str, object] = {}

        for anchor in gt_available:
            i = idx_of[anchor]
            g = np.zeros(n_bins, dtype=int)
            r = np.zeros(n_bins, dtype=int)

            for other in gt_available:
                if other == anchor:
                    continue
                j = idx_of[other]

                d = float(D[i, j])
                b = _bin_index(d, bin_width, n_bins)

                same_gt = (gt_map[anchor] == gt_map[other])
                if same_gt:
                    g[b] += 1
                else:
                    r[b] += 1

            # aggregate
            sess_green += g
            sess_red += r
            global_green += g
            global_red += r

            # per-anchor plot (percent of all anchor-other pairs)
            if make_per_speaker_plots:
                g_pct, r_pct, total_sum, green_sum, red_sum = _pct_stacked(g, r)
                if total_sum > 0:
                    green_frac = green_sum / total_sum
                    red_frac = red_sum / total_sum

                    plt.figure(figsize=(12, 4))
                    plt.bar(centers, g_pct, width=bin_width * 0.95, color="green", label="GT same cluster (green)")
                    plt.bar(centers, r_pct, bottom=g_pct, width=bin_width * 0.95, color="red", label="GT different cluster (red)")
                    plt.xlabel("Distance bin (center)")
                    plt.ylabel("Share of pairs (%)")
                    plt.title(
                        f"{session} | anchor={anchor} | {distance_key}\n"
                    )
                    plt.xlim(0.0, 1.0)
                    plt.ylim(0.0, max(1.0, float((g_pct + r_pct).max()) * 1.10))
                    plt.legend(loc="upper right")
                    plt.tight_layout()
                    plt.savefig(session_out_dir / f"anchor_{anchor}.png", dpi=160)
                    plt.close()

            denom = (g + r).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                frac_green = np.where(denom > 0, g / denom, np.nan)
                frac_red = np.where(denom > 0, r / denom, np.nan)

            anchor_summaries[anchor] = {
                "counts_green": g.tolist(),
                "counts_red": r.tolist(),
                "frac_green": frac_green.tolist(),
                "frac_red": frac_red.tolist(),
            }

        denom_s = (sess_green + sess_red).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            sess_frac_green = np.where(denom_s > 0, sess_green / denom_s, np.nan)
            sess_frac_red = np.where(denom_s > 0, sess_red / denom_s, np.nan)

        per_session_summary[session] = {
            "distance_key": distance_key,
            "speaker_order": speaker_order,
            "gt_available_speakers": gt_available,
            "counts_green": sess_green.tolist(),
            "counts_red": sess_red.tolist(),
            "frac_green": sess_frac_green.tolist(),
            "frac_red": sess_frac_red.tolist(),
            "anchors": anchor_summaries,
        }

    # global plot (percent of all pairs across all sessions)
    g_pct, r_pct, total_sum, green_sum, red_sum = _pct_stacked(global_green, global_red)
    if total_sum > 0:
        green_frac = green_sum / total_sum
        red_frac = red_sum / total_sum

        plt.figure(figsize=(12, 4))
        plt.bar(centers, g_pct, width=bin_width * 0.95, color="green", label="GT same cluster (green)")
        plt.bar(centers, r_pct, bottom=g_pct, width=bin_width * 0.95, color="red", label="GT different cluster (red)")
        plt.xlabel("Distance bin (center)")
        plt.ylabel("Share of pairs (%)")
        plt.title(
            f"ALL SESSIONS | {distance_key}\n"
        )
        plt.xlim(0.0, 1.0)

        max_stack = float((g_pct + r_pct).max())
        plt.ylim(0.0, max(5.0, max_stack + 10.0))

        # ---- annotate per-bin green% and red% ABOVE the stacked bar ----
        # - keep it tiny
        # - avoid overlap by slightly shifting x for green vs red
        min_seg_pct = 0.75  # hide tiny segments to keep readability
        x_off = bin_width * 0.18
        y_pad = 0.6

        for x, g, r, total in zip(centers, g_pct, r_pct, (g_pct + r_pct)):
            if total <= 0.0:
                continue

            y = total + y_pad

            if g >= min_seg_pct:
                plt.text(
                    x - x_off,
                    y,
                    f"{g:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                    color="green",
                    clip_on=False,
                )
            if r >= min_seg_pct:
                plt.text(
                    x + x_off,
                    y,
                    f"{r:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                    color="red",
                    clip_on=False,
                )

        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(out_root / "ALL.png", dpi=180)
        plt.close()

    denom = (global_green + global_red).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        global_frac_green = np.where(denom > 0, global_green / denom, np.nan)
        global_frac_red = np.where(denom > 0, global_red / denom, np.nan)

    summary = {
        "distance_key": distance_key,
        "bin_width": float(bin_width),
        "bin_edges": edges.tolist(),
        "bin_centers": centers.tolist(),
        "global": {
            "counts_green": global_green.tolist(),
            "counts_red": global_red.tolist(),
            "frac_green": global_frac_green.tolist(),
            "frac_red": global_frac_red.tolist(),
        },
        "per_session": per_session_summary,
    }
    _write_json(out_root / "summary.json", summary)
    return summary


def plot_sessions_under_threshold(
    output_root: Path,
    labels_root: Path,
    distance_key: str,
    thr: float,
    sessions: Optional[List[str]] = None,
    out_dirname: str = "gaze_distance_gt_bins",
) -> Dict[str, object]:
    """
    Extra overview plot:
    - For each session: only counts pairs with distance < thr (ignore all others)
    - Excludes sessions with 0 such pairs
    - Stacked % bars per session (green+red = 100% within-threshold)
    """
    distance_key = (distance_key or "").strip()
    if not distance_key:
        raise ValueError("distance_key must be a non-empty key name")

    dist_json_path = output_root / "speaker_distance_matrices.json"
    all_dist = _read_json(dist_json_path)
    sess_list = sessions if sessions is not None else sorted(all_dist.keys())

    rows = []  # (session, green, red)
    for session in sess_list:
        sess_obj = all_dist.get(session, None)
        if not _ensure_key_exists(distance_key, sess_obj):
            continue

        speaker_order = [_norm_spk_id(x) for x in sess_obj.get("speaker_order", [])]
        if not speaker_order:
            continue

        D = np.asarray(sess_obj[distance_key], dtype=float)
        if D.ndim != 2 or D.shape[0] != D.shape[1] or D.shape[0] != len(speaker_order):
            continue

        gt_map = load_gt_clusters(labels_root, session)
        gt_available = [spk for spk in speaker_order if spk in gt_map]
        if len(gt_available) < 2:
            continue

        idx_of = {spk: i for i, spk in enumerate(speaker_order)}

        green = 0
        red = 0
        for a in gt_available:
            i = idx_of[a]
            for b in gt_available:
                if b == a:
                    continue
                j = idx_of[b]
                d = float(D[i, j])
                if d >= thr:
                    continue
                if gt_map[a] == gt_map[b]:
                    green += 1
                else:
                    red += 1

        if green + red > 0:
            rows.append((session, green, red))

    # sort sessions by number of within-thr pairs (desc)
    rows.sort(key=lambda x: (x[1] + x[2]), reverse=True)

    out_root = output_root / out_dirname / distance_key
    out_root.mkdir(parents=True, exist_ok=True)

    if not rows:
        _write_json(out_root / f"sessions_lt_{thr:.2f}.json", {
            "distance_key": distance_key, "thr": float(thr), "sessions": []
        })
        return {"distance_key": distance_key, "thr": float(thr), "sessions": []}

    sessions_x = [r[0] for r in rows]
    g_counts = np.array([r[1] for r in rows], dtype=float)
    r_counts = np.array([r[2] for r in rows], dtype=float)
    totals = g_counts + r_counts

    g_pct = 100.0 * (g_counts / totals)
    r_pct = 100.0 * (r_counts / totals)

    plt.figure(figsize=(max(10, 0.35 * len(sessions_x)), 4))
    x = np.arange(len(sessions_x))
    plt.bar(x, g_pct, color="green", label="GT same cluster (green)")
    plt.bar(x, r_pct, bottom=g_pct, color="red", label="GT different cluster (red)")
    plt.ylabel(f"Share within d<{thr:.2f} (%)")
    plt.xlabel("Session")

    total_green = int(g_counts.sum())
    total_red = int(r_counts.sum())
    total_pairs = int(total_green + total_red)
    plt.title(
        f"Sessions with distances < {thr:.2f} | {distance_key}\n"
    )
    plt.xticks(x, sessions_x, rotation=60, ha="right")
    plt.ylim(0, 100)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_root / f"SESSIONS_LT_{thr:.2f}.png", dpi=180)
    plt.close()

    out = {
        "distance_key": distance_key,
        "thr": float(thr),
        "sessions": [
            {
                "session": s,
                "correct": int(g),
                "wrong": int(r),
                "total": int(g + r),
                "correct_frac": float(g / (g + r)),
            }
            for s, g, r in rows
        ],
        "global": {
            "correct": int(total_green),
            "wrong": int(total_red),
            "total": int(total_pairs),
            "correct_frac": float(total_green / total_pairs),
        }
    }
    _write_json(out_root / f"sessions_lt_{thr:.2f}.json", out)
    return out




def plot_correct_rate_vs_threshold(
    output_root: Path,
    labels_root: Path,
    distance_key: str,
    thresholds: Optional[np.ndarray] = None,
    sessions: Optional[List[str]] = None,
    out_dirname: str = "gaze_distance_gt_bins",
) -> Dict[str, object]:
    """
    Simple overview: Correct-Rate vs Threshold.

    For each threshold t:
      consider ONLY directed pairs (anchor->other) with distance <= t
      correct = GT same cluster
      wrong   = GT different cluster

    Plot:
      x: threshold t
      y: correct_rate = correct / (correct + wrong)
      Also write a JSON with counts for reproducibility.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)  # step 0.01

    dist_json_path = output_root / "speaker_distance_matrices.json"
    if not dist_json_path.exists():
        raise FileNotFoundError(f"Missing distance json: {dist_json_path}")

    all_dist = _read_json(dist_json_path)
    sess_list = sessions if sessions is not None else sorted(all_dist.keys())

    correct_counts = np.zeros(len(thresholds), dtype=int)
    wrong_counts = np.zeros(len(thresholds), dtype=int)

    for session in sess_list:
        sess_obj = all_dist.get(session, None)
        if not isinstance(sess_obj, dict) or distance_key not in sess_obj:
            continue

        speaker_order = [_norm_spk_id(x) for x in sess_obj.get("speaker_order", [])]
        if not speaker_order:
            continue

        D = np.asarray(sess_obj[distance_key], dtype=float)
        if D.ndim != 2 or D.shape[0] != D.shape[1] or D.shape[0] != len(speaker_order):
            continue

        gt_map = load_gt_clusters(labels_root, session)
        gt_available = [spk for spk in speaker_order if spk in gt_map]
        if len(gt_available) < 2:
            continue

        idx_of = {spk: i for i, spk in enumerate(speaker_order)}

        # Collect all directed pair distances + correctness once per session
        dists = []
        is_correct = []
        for a in gt_available:
            i = idx_of[a]
            for b in gt_available:
                if b == a:
                    continue
                j = idx_of[b]
                d = float(D[i, j])
                dists.append(d)
                is_correct.append(1 if gt_map[a] == gt_map[b] else 0)

        if not dists:
            continue

        dists = np.asarray(dists, dtype=float)
        is_correct = np.asarray(is_correct, dtype=int)

        # For each threshold, count pairs <= t
        # Vectorized using broadcasting over thresholds
        # mask shape: (T, Npairs)
        mask = dists[None, :] <= thresholds[:, None]
        # counts per threshold
        corr = (mask & (is_correct[None, :] == 1)).sum(axis=1)
        wrong = (mask & (is_correct[None, :] == 0)).sum(axis=1)

        correct_counts += corr.astype(int)
        wrong_counts += wrong.astype(int)

    totals = correct_counts + wrong_counts
    with np.errstate(divide="ignore", invalid="ignore"):
        correct_rate = np.where(totals > 0, correct_counts / totals, np.nan)

    # Plot
    out_root = output_root / out_dirname / distance_key
    out_root.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(thresholds, correct_rate * 100.0)  # percent
    plt.xlabel("Distance threshold t (pairs with d <= t)")
    plt.ylabel("Correct rate (%)")
    total_corr = int(correct_counts[-1])
    total_wrong = int(wrong_counts[-1])
    total_all = int(total_corr + total_wrong)
    title = f"Correct-Rate vs Threshold | {distance_key}\n"
    if total_all > 0:
        title += f" "
    else:
        title += "No pairs available"
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 100.0)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_root / "CORRECT_RATE_VS_THRESHOLD.png", dpi=180)
    plt.close()

    out = {
        "distance_key": distance_key,
        "thresholds": thresholds.tolist(),
        "correct_counts": correct_counts.tolist(),
        "wrong_counts": wrong_counts.tolist(),
        "correct_rate": correct_rate.tolist(),
    }
    _write_json(out_root / "correct_rate_vs_threshold.json", out)
    return out


def main() -> None:
    print(f"[eval] OUTPUT_ROOT={OUTPUT_ROOT}")
    print(f"[eval] LABELS_ROOT={LABELS_ROOT}")
    print(f"[eval] BIN_WIDTH={BIN_WIDTH} | per-speaker plots={MAKE_PER_SPEAKER_PLOTS}")
    print(f"[eval] SESSION_LT_THRESH={SESSION_LT_THRESH}")
    print(f"[eval] DISTANCE_KEYS={DISTANCE_KEYS}")

    dist_json_path = OUTPUT_ROOT / "speaker_distance_matrices.json"
    if not dist_json_path.exists():
        raise FileNotFoundError(f"Missing distance json: {dist_json_path}")

    # quick sanity: show which keys are present at least once
    all_dist = _read_json(dist_json_path)
    any_session = next(iter(all_dist.values()), {})
    present_keys = sorted([k for k in any_session.keys() if k.startswith("distance_")])
    print(f"[eval] example distance keys in json: {present_keys}")

    for distance_key in DISTANCE_KEYS:
        print(f"[eval] running distance_key={distance_key} (bins)")
        plot_distance_bins_vs_gt(
            output_root=OUTPUT_ROOT,
            labels_root=LABELS_ROOT,
            distance_key=distance_key,
            bin_width=BIN_WIDTH,
            sessions=SESSIONS,
            make_per_speaker_plots=MAKE_PER_SPEAKER_PLOTS,
            out_dirname=OUT_DIRNAME,
        )
        # Correct-Rate vs Threshold overview
        plot_correct_rate_vs_threshold(
            output_root=OUTPUT_ROOT,
            labels_root=LABELS_ROOT,
            distance_key=distance_key,
            thresholds=np.linspace(0.0, 1.0, 101),
            sessions=SESSIONS,
            out_dirname=OUT_DIRNAME,
        )
        print(f"[eval] running distance_key={distance_key} (sessions under threshold)")
        plot_sessions_under_threshold(
            output_root=OUTPUT_ROOT,
            labels_root=LABELS_ROOT,
            distance_key=distance_key,
            thr=SESSION_LT_THRESH,
            sessions=SESSIONS,
            out_dirname=OUT_DIRNAME,
        )

    print("[eval] done")


if __name__ == "__main__":
    main()
