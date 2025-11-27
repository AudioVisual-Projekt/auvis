import os
import sys
import json
import numpy as np

from pathlib import Path

# Sitzungs-Qualitätsmetrik (Similarity-basiert)
from auvis.team_c.src.cluster.conv_confidence import compute_conversation_cluster_quality


# === conv_spks importieren (Baseline-Sachen deines Kollegen) ===
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import auvis.team_c.src.cluster.conv_spks as baseline


# --- Helper ---
def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_gt_labels(session_dir: Path):
    """
    Ground-Truth:
        <session_dir>/labels/speaker_to_cluster.json
      Format:
        { "spk_0": 0, "spk_1": 0, "spk_2": 1, ... }
    """
    path = session_dir / "labels" / "speaker_to_cluster.json"
    if not path.exists():
        return {}
    return load_json(path)


def load_asd_matching(session_out_dir: Path):
    """
    asd_seat_matching.json aus dem Video-Seating-Skript:
        <databin_root>/output/session_XX/asd_seat_matching.json
    """
    path = session_out_dir / "asd_seat_matching.json"
    if not path.exists():
        return None
    return load_json(path)


def load_A(session_out_dir: Path):
    """
    A.npy: Sitz-Ähnlichkeitsmatrix (0..1, Diagonale ~1)
    """
    return np.load(session_out_dir / "A.npy")


# ============================================================
#   Video-basierte Sitzcluster-Varianten (Seat-Space)
# ============================================================

def seat_cluster_neighbors(n: int):
    """
    Dummy-Variante: jeder Seat eigene Cluster-ID (nur als Platzhalter).
    Wenn du echte Nachbarn-Clustern willst, kannst du hier noch Logik ergänzen.
    """
    clusters = {}
    label = 0
    for i in range(1, n + 1):
        clusters[f"seat_{i}"] = label
        label += 1
    return clusters


def seat_cluster_halves(n: int):
    """
    2 Cluster: erste Hälfte vs. zweite Hälfte im Sitzindex.
    """
    clusters = {}
    for i in range(1, n + 1):
        clusters[f"seat_{i}"] = 0 if i <= n // 2 else 1
    return clusters


def seat_cluster_opposites(n: int):
    """
    Alternierende Cluster: 0,1,0,1,... – "gegenüber" als grobe Heuristik.
    """
    clusters = {}
    for i in range(1, n + 1):
        clusters[f"seat_{i}"] = (i % 2)
    return clusters


def seat_cluster_k2(A: np.ndarray):
    """
    Hierarchisches Clustering (k=2) im Seat-Space auf Basis von Distanzen.
    A ist ÄHNLICHKEIT (1 = identisch, ~0 = weit weg).
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    # Distanzmatrix: D = 1 - A, Diagonale = 0
    D = 1.0 - A.astype(float)
    np.fill_diagonal(D, 0.0)

    # squareform braucht eine gültige Distanzmatrix
    dist = squareform(D, checks=False)
    Z = linkage(dist, method="average")
    labels = fcluster(Z, 2, criterion="maxclust")
    clusters = {}
    for i, c in enumerate(labels, 1):
        clusters[f"seat_{i}"] = int(c - 1)
    return clusters


def convert_seat_clusters_to_speaker_clusters(
    seat_clusters: dict, asd_matching: dict
):
    """
    seat_id -> cluster  UND  person_id -> asd_track_id
    liefert: speaker_id ("spk_X") -> cluster_id
    """
    assignments = asd_matching.get("assignments", [])

    mapping = {}

    for a in assignments:
        pid = int(a["person_id"])         # 1-basiert
        spk = int(a["asd_track_id"])      # X in spk_X
        key = f"seat_{pid}"
        if key not in seat_clusters:
            continue
        cluster = seat_clusters[key]
        mapping[f"spk_{spk}"] = int(cluster)

    return mapping


def build_speaker_similarity_from_A(A: np.ndarray, asd_matching: dict):
    """
    Similarity-Matrix im Raum der SPEAKER IDs konstruieren,
    basierend auf der Sitz-Ähnlichkeitsmatrix A.

    Vorgehen:
      - assignments: person_id (Seat) -> asd_track_id (Speaker)
      - für jede Speaker-Paarung (i,j):
            score = A[seat_i, seat_j]
    """
    assignments = asd_matching.get("assignments", [])
    if not assignments:
        return [], np.zeros((0, 0), float)

    person_ids = []
    speaker_ids = []
    for a in assignments:
        pid = int(a["person_id"])
        spk = int(a["asd_track_id"])
        person_ids.append(pid)
        speaker_ids.append(f"spk_{spk}")

    m = len(person_ids)
    scores = np.zeros((m, m), float)

    for i in range(m):
        pi = person_ids[i] - 1  # Seats sind 1-basiert, A 0-basiert
        for j in range(m):
            pj = person_ids[j] - 1
            scores[i, j] = float(A[pi, pj])  # A ist Similarity

    return speaker_ids, scores


# ============================================================
#   ASD-Overlap-Baseline (wie bei deinem Kollegen)
# ============================================================

def find_speaker_asd_paths(session_dir: Path):
    """
    ASD-Scores pro Speaker finden.

    Erwartetes Layout:
        <session_dir>/speakers/spk_0/central_crops/*_asd.json
        <session_dir>/speakers/spk_1/central_crops/*_asd.json
        ...

    Rückgabe:
        { "spk_0": [Path(...), ...], ... }
    """
    speakers_root = session_dir / "speakers"
    if not speakers_root.exists():
        return {}

    spk_to_paths = {}
    for spk_dir in sorted(speakers_root.glob("spk_*")):
        cc_dir = spk_dir / "central_crops"
        if not cc_dir.exists():
            continue
        paths = sorted(cc_dir.glob("*_asd.json"))
        if paths:
            spk_to_paths[spk_dir.name] = paths
    return spk_to_paths


def infer_uem_bounds(spk_asd_paths: dict):
    """
    UEM-Fenster aus ASD-Frames ableiten:
      min/max Frame-ID über alle Speaker, dann in Sekunden bei 25 fps.
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

    if min_frame is None:
        min_frame = 0

    fps = 25.0
    uem_start = float(min_frame) / fps
    uem_end = float(max_frame) / fps
    return uem_start, uem_end


def build_speaker_segments(
    spk_asd_paths: dict,
    uem_start: float,
    uem_end: float,
):
    """
    Für jeden Speaker:
      ASD-JSONs -> get_speaker_activity_segments -> Zeitsegmente.
    """
    segments = {}
    for spk_name, paths in spk_asd_paths.items():
        segs = baseline.get_speaker_activity_segments(
            pycrop_asd_path=[str(p) for p in paths],
            uem_start=uem_start,
            uem_end=uem_end,
        )
        segments[spk_name] = segs
    return segments


def eval_asd_overlap_session(session_dir: Path, out_root: Path):
    """
    Voller Overlap-Baseline-Step für EINE Session:

      1. ASD-JSONs finden
      2. Segmente via get_speaker_activity_segments
      3. conversation scores via calculate_conversation_scores
      4. cluster_speakers (Agglomerative, average, k = #GT-Cluster)
      5. F1 / ARI gegen GT berechnen
      6. Pred-Cluster als JSON speichern
    """
    session_name = session_dir.name

    gt_labels = load_gt_labels(session_dir)
    if not gt_labels:
        print(f"[ASD] {session_name}: no GT labels -> skip")
        return None

    spk_asd_paths = find_speaker_asd_paths(session_dir)
    if not spk_asd_paths:
        print(f"[ASD] {session_name}: no ASD JSONs -> skip")
        return None

    uem_start, uem_end = infer_uem_bounds(spk_asd_paths)
    speaker_segments = build_speaker_segments(spk_asd_paths, uem_start, uem_end)

    baseline.validate_speaker_segments(speaker_segments)

    scores = baseline.calculate_conversation_scores(speaker_segments)
    speaker_ids = list(speaker_segments.keys())

    true_clusters = list(set(gt_labels.values()))
    n_clusters = len(true_clusters)

    clusters = baseline.cluster_speakers(
        scores=scores,
        speaker_ids=speaker_ids,
        n_clusters=n_clusters,
        algo="agglomerative",
        algo_kwargs={"linkage": "average"},
    )

    # Schnittmenge
    speakers = sorted(set(gt_labels.keys()) & set(clusters.keys()))
    if len(speakers) < 2:
        print(f"[ASD] {session_name}: not enough overlapping speakers -> skip")
        return None

    f1 = baseline.get_clustering_f1_score(
        conversation_clusters_label={s: clusters[s] for s in speakers},
        true_clusters_label={s: gt_labels[s] for s in speakers},
    )
    ari = baseline.get_clustering_ari_score(
        conversation_clusters_label={s: clusters[s] for s in speakers},
        true_clusters_label={s: gt_labels[s] for s in speakers},
    )

    # Vorhersage speichern
    out_dir = out_root / session_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "speaker_to_cluster_asd_overlap.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2)

    return {
        "f1_overlap": float(f1),
        "ari_overlap": float(ari),
        "n_speakers": len(speakers),
        "n_clusters": int(n_clusters),
    }


# ============================================================
#   Video-Eval pro Session (Quality + F1/ARI)
# ============================================================

def evaluate_video_session(session_name: str, databin_root: Path):
    """
    Evaluierung einer Session im Video-Space:
      - lädt A.npy + asd_seat_matching.json
      - konstruiert seat-basierte Cluster-Varianten
      - mappt über ASD-Matching auf Speaker
      - berechnet Quality-Metriken + F1/ARI im Speaker-Raum
    """
    session_dir = databin_root / session_name
    out_dir = databin_root / "output" / session_name

    if not session_dir.exists():
        print(f"[VIDEO] {session_name}: session dir not found -> {session_dir}")
        return None
    if not out_dir.exists():
        print(f"[VIDEO] {session_name}: output dir not found -> {out_dir}")
        return None

    gt = load_gt_labels(session_dir)
    A = load_A(out_dir)
    asd = load_asd_matching(out_dir)

    if asd is None:
        print(f"[VIDEO] {session_name}: no asd_seat_matching.json")
        return None

    n = A.shape[0]

    seat_methods = {
        "neighbors": seat_cluster_neighbors(n),
        "halves": seat_cluster_halves(n),
        "opposites": seat_cluster_opposites(n),
        "dist_k2": seat_cluster_k2(A),
    }

    results = {}

    # Speaker-Similarity aus Sitz-Ähnlichkeitsmatrix bauen
    spk_ids, scores = build_speaker_similarity_from_A(A, asd)
    if scores.size == 0:
        print(f"[VIDEO] {session_name}: no speaker similarity matrix from A")
        return None

    for name, seat_cl in seat_methods.items():
        # seat -> speaker Cluster-Mapping
        pred = convert_seat_clusters_to_speaker_clusters(seat_cl, asd)

        # evtl. gibt es weniger Speaker als Seats -> auf gemeinsame Speaker einschränken
        common_spk = sorted(set(spk_ids) & set(pred.keys()))
        if len(common_spk) < 2:
            continue

        # Indizes für die Reduced-Matrix
        idx_map = {spk: i for i, spk in enumerate(spk_ids)}
        sel_idx = [idx_map[s] for s in common_spk]
        scores_sub = scores[np.ix_(sel_idx, sel_idx)]

        pred_sub = {s: pred[s] for s in common_spk}

        # Quality-Metrik (Silhouette + within-between)
        q = compute_conversation_cluster_quality(
            scores=scores_sub,
            speaker_ids=common_spk,
            clusters_dict=pred_sub,
        )

        # F1 / ARI VIDEO vs GT (falls GT vorhanden)
        f1_vid = None
        ari_vid = None
        if gt:
            inter = sorted(set(common_spk) & set(gt.keys()))
            if len(inter) >= 2:
                f1_vid = baseline.get_clustering_f1_score(
                    conversation_clusters_label={s: pred_sub[s] for s in inter},
                    true_clusters_label={s: gt[s] for s in inter},
                )
                ari_vid = baseline.get_clustering_ari_score(
                    conversation_clusters_label={s: pred_sub[s] for s in inter},
                    true_clusters_label={s: gt[s] for s in inter},
                )

        results[name] = {
            "n_speakers_quality": q["n_speakers"],
            "n_clusters_quality": q["n_clusters"],
            "silhouette_mean": q["silhouette_mean"],
            "quality_within_minus_between": q["quality_within_minus_between"],
            "f1_video": None if f1_vid is None else float(f1_vid),
            "ari_video": None if ari_vid is None else float(ari_vid),
        }

    return results


# ----------------- MAIN -----------------
def main():
    # Root für deine Sessions (da liegen session_40, session_137, ...)
    databin_root = Path(
        r"C:\Users\seana\PycharmProjects\auvis\team_c\data-bin\dev"
    )
    asd_out_root = databin_root / "output_asd_overlap"
    video_summary_out = databin_root / "output" / "video_vs_asd_eval.json"
    tsv_out = databin_root / "output" / "video_vs_asd_eval.tsv"

    session_dirs = sorted(
        [p for p in databin_root.glob("session_*") if p.is_dir()],
        key=lambda x: x.name,
    )

    all_results = {}
    rows = []

    print(f"Found {len(session_dirs)} sessions.\n")

    for sdir in session_dirs:
        session_name = sdir.name
        print(f"===== {session_name} =====")

        # ASD-Baseline
        asd_res = eval_asd_overlap_session(sdir, asd_out_root)
        if asd_res:
            print(
                f"  ASD_overlap: F1={asd_res['f1_overlap']:.3f}, "
                f"ARI={asd_res['ari_overlap']:.3f}, "
                f"n_speakers={asd_res['n_speakers']}, "
                f"n_clusters={asd_res['n_clusters']}"
            )
        else:
            print("  ASD_overlap: no result")

        # Video-Varianten
        vid_res = evaluate_video_session(session_name, databin_root)
        if vid_res:
            for variant, q in vid_res.items():
                print(
                    f"  VIDEO {variant:12s} | "
                    f"sil={q['silhouette_mean']!r:>8} | "
                    f"Δ(within-between)={q['quality_within_minus_between']!r:>8} | "
                    f"F1_vid={q['f1_video']!r:>8} | "
                    f"ARI_vid={q['ari_video']!r:>8}"
                )
        else:
            print("  VIDEO: no result")

        # beste Video-Variante nach F1 (falls vorhanden)
        best_variant = None
        best_f1 = None
        if vid_res:
            for v_name, q in vid_res.items():
                f1v = q["f1_video"]
                if f1v is None:
                    continue
                if best_f1 is None or f1v > best_f1:
                    best_f1 = f1v
                    best_variant = v_name

        all_results[session_name] = {
            "asd_overlap": asd_res,
            "video": vid_res,
            "best_video_variant": best_variant,
            "best_video_f1": best_f1,
        }

        row = {
            "session": session_name,
            "asd_f1": None if not asd_res else asd_res["f1_overlap"],
            "asd_ari": None if not asd_res else asd_res["ari_overlap"],
            "best_video_variant": best_variant,
            "best_video_f1": best_f1,
        }
        # optional: noch ein paar Quality-Zahlen pro Session reinkippen (z.B. dist_k2)
        if vid_res and "dist_k2" in vid_res:
            row["dist_k2_sil"] = vid_res["dist_k2"]["silhouette_mean"]
            row["dist_k2_f1"] = vid_res["dist_k2"]["f1_video"]
        rows.append(row)

        print()

    # JSON speichern
    video_summary_out.parent.mkdir(parents=True, exist_ok=True)
    with video_summary_out.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"[INFO] wrote JSON to {video_summary_out}")

    # TSV speichern (flattened)
    with tsv_out.open("w", encoding="utf-8") as f:
        header = [
            "session",
            "asd_f1",
            "asd_ari",
            "best_video_variant",
            "best_video_f1",
            "dist_k2_sil",
            "dist_k2_f1",
        ]
        f.write("\t".join(header) + "\n")
        for r in rows:
            vals = [
                r.get("session", ""),
                "" if r.get("asd_f1") is None else f"{r['asd_f1']:.3f}",
                "" if r.get("asd_ari") is None else f"{r['asd_ari']:.3f}",
                r.get("best_video_variant") or "",
                "" if r.get("best_video_f1") is None else f"{r['best_video_f1']:.3f}",
                "" if r.get("dist_k2_sil") is None else f"{r['dist_k2_sil']:.3f}",
                "" if r.get("dist_k2_f1") is None else f"{r['dist_k2_f1']:.3f}",
            ]
            f.write("\t".join(vals) + "\n")
    print(f"[INFO] wrote TSV to {tsv_out}")


if __name__ == "__main__":
    main()
