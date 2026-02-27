#!/usr/bin/env python3
"""
GridSearch Orchestrator V3 (Master-CSV Edition)
-----------------------------------------------
Features:
1. Fix: Manuelle ID-Normalisierung (spk_0 statt session_X_spk_0).
2. Neu: Erfassung der Session-Confidence.
3. Neu: 'Master-CSV' (grid_details.csv) mit ALLEN Datenpunkten (Session x Parameter) für Plots.
"""

import argparse
import csv
import json
import logging
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any

# --- Setup Imports ---
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Projekt-Imports
from semantic.models.semantic_clustering_models import SemanticClusteringConfig
from semantic.services.semantic_clustering_service import (
    load_distance_artifacts,
    cluster_session,
    write_speaker_to_cluster,
    build_session_meta,
    write_session_meta
)
from semantic.models.evaluation_semantic_models import EvaluationConfig
from semantic.services.evaluation_semantic_service import evaluate_one_session

# Optional: Matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
LOGGER = logging.getLogger("gridsearch")


def _generate_run_id(linkage: str, threshold: float) -> str:
    return f"run_{linkage}_t{threshold:.2f}"


def _plot_heatmap(session_dir: Path, out_path: Path, D: np.ndarray, title: str):
    if not HAS_MATPLOTLIB:
        return
    try:
        plt.figure(figsize=(10, 8))
        plt.imshow(D, cmap='viridis_r', interpolation='nearest')
        plt.colorbar(label="Cosine Distance")
        plt.title(f"{title}\nN={D.shape[0]}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=100)
        plt.close()
    except Exception as e:
        LOGGER.warning(f"Heatmap fehlgeschlagen für {session_dir.name}: {e}")


def run_grid_search_session(
    session_dir: Path,
    baseline_dir: Path,
    run_dir: Path,
    config: SemanticClusteringConfig,
    eval_config: EvaluationConfig
) -> Dict[str, Any]:
    session_name = session_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clustering
    try:
        D, uids, dist_meta = load_distance_artifacts(session_dir)
        # Cluster Session gibt (mapping, max_dist, confidence) zurück
        mapping, max_dist, confidence = cluster_session(D=D, uids=uids, config=config, session_name=session_name)
        
        # --- FIX: IDs manuell normalisieren ---
        prefix = f"{session_name}_"
        mapping_fixed = {}
        for uid, cluster_id in mapping.items():
            if uid.startswith(prefix):
                new_uid = uid[len(prefix):]
            else:
                new_uid = uid
            mapping_fixed[new_uid] = cluster_id
        
        write_speaker_to_cluster(run_dir, mapping_fixed)
        
        meta = build_session_meta(
            session_name=session_name,
            n=len(uids),
            config=config,
            distance_meta=dist_meta,
            max_merge_dist=max_dist,
            confidence=confidence
        )
        write_session_meta(run_dir, meta)

    except Exception as e:
        LOGGER.error(f"[{session_name}] Clustering failed: {e}")
        return {"error": str(e)}

    # 2. Evaluation
    try:
        gt_session_dir = baseline_dir / session_name
        
        res = evaluate_one_session(
            session_dir_pred=run_dir,
            session_dir_gt=gt_session_dir,
            config=eval_config,
            force=True
        )

        out_eval = run_dir / "evaluation.json"
        with out_eval.open("w", encoding="utf-8") as f:
            json.dump(res.to_dict(), f, indent=2, sort_keys=True)
            
        return {
            "session": session_name,
            "pairwise_f1": res.pairwise_metrics.f1 if res.pairwise_metrics else 0.0,
            "pairwise_precision": res.pairwise_metrics.precision if res.pairwise_metrics else 0.0,
            "pairwise_recall": res.pairwise_metrics.recall if res.pairwise_metrics else 0.0,
            "speaker_f1_macro": res.speaker_f1_macro if res.speaker_f1_macro is not None else 0.0,
            "session_confidence": res.session_confidence if res.session_confidence is not None else 0.0,
            "n_speakers": res.n_speakers,
            "error": None
        }

    except Exception as e:
        LOGGER.error(f"[{session_name}] Evaluation failed: {e}")
        return {"error": str(e)}


def _write_csv_summary(path, rows):
    if not rows: return
    # Sortiert nach F1 absteigend für schnelle Übersicht
    rows_sorted = sorted(rows, key=lambda x: (x["avg_pairwise_f1"], x["avg_pairwise_precision"]), reverse=True)
    keys = ["run_id", "linkage", "threshold", "avg_pairwise_f1", "avg_pairwise_precision", "avg_pairwise_recall", "avg_session_confidence", "n_sessions"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows_sorted)


def _append_csv_details(path, rows):
    """Hängt neue Detail-Zeilen an die Master-CSV an."""
    if not rows: return
    
    file_exists = path.exists()
    keys = [
        "run_id", "linkage", "threshold", 
        "session", "n_speakers", 
        "pairwise_f1", "pairwise_precision", "pairwise_recall", 
        "speaker_f1_macro", "session_confidence"
    ]
    
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="GridSearch Semantic Clustering V3 (Master CSV)")
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sessions", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    baseline_root = Path(args.baseline_root).resolve()
    output_root = Path(args.output_root).resolve()
    
    thresholds = [round(x, 2) for x in np.arange(0.1, 1.01, 0.02)]
    linkages = ["average", "complete"]

    all_sessions = sorted([p for p in output_root.glob("session_*") if p.is_dir()])
    if args.sessions:
        whitelist = set(args.sessions.split(","))
        all_sessions = [p for p in all_sessions if p.name in whitelist]
    
    if not all_sessions:
        LOGGER.error("Keine Sessions gefunden.")
        return

    LOGGER.info(f"Starte GridSearch auf {len(all_sessions)} Sessions.")
    
    grid_root = output_root / "_gridsearch"
    grid_root.mkdir(parents=True, exist_ok=True)
    
    # Pfade für die beiden CSVs
    summary_csv_path = grid_root / "grid_results.csv"
    details_csv_path = grid_root / "grid_details.csv"
    
    # Falls wir neu starten, löschen wir die Details-CSV vorher, damit wir nicht doppelt appenden
    if not args.skip_existing and details_csv_path.exists():
        details_csv_path.unlink()

    summary_rows = []
    total_runs = len(linkages) * len(thresholds)
    current_run = 0

    for linkage in linkages:
        for thr in thresholds:
            current_run += 1
            run_id = _generate_run_id(linkage, thr)
            LOGGER.info(f"=== Run {current_run}/{total_runs}: {run_id} ===")

            clust_conf = SemanticClusteringConfig(linkage=linkage, distance_threshold=thr, renumber_labels=True)
            eval_conf = EvaluationConfig(split="dev", baseline_root=str(baseline_root), semantic_output_root="", write_csv=False, write_summary_json=False)

            f1_scores, prec_scores, rec_scores = [], [], []
            conf_scores = [] 
            
            # Temporäre Liste für die Detail-Zeilen dieses Runs
            current_run_details = []
            session_metrics = [] # Für Heatmap-Auswahl

            for s_dir in all_sessions:
                run_dir = s_dir / "_gridsearch" / run_id
                
                # Resume Check (einfach)
                if args.skip_existing and (run_dir / "evaluation.json").exists():
                    # Hier könnte man Logik einbauen, um Werte aus json zu laden, 
                    # wenn man resume wirklich robust will. Für jetzt skippen wir die Berechnung.
                    pass 

                res = run_grid_search_session(s_dir, baseline_root, run_dir, clust_conf, eval_conf)
                
                if res.get("error"): continue
                
                # Metriken sammeln für Durchschnitt
                f1_scores.append(res["pairwise_f1"])
                prec_scores.append(res["pairwise_precision"])
                rec_scores.append(res["pairwise_recall"])
                conf_scores.append(res["session_confidence"])

                session_metrics.append({"session": s_dir.name, "f1": res["pairwise_f1"], "dir": s_dir})

                # Detail-Zeile bauen für Master CSV
                detail_row = {
                    "run_id": run_id,
                    "linkage": linkage,
                    "threshold": thr,
                    "session": res["session"],
                    "n_speakers": res["n_speakers"],
                    "pairwise_f1": res["pairwise_f1"],
                    "pairwise_precision": res["pairwise_precision"],
                    "pairwise_recall": res["pairwise_recall"],
                    "speaker_f1_macro": res["speaker_f1_macro"],
                    "session_confidence": res["session_confidence"]
                }
                current_run_details.append(detail_row)

            # 1. Details schreiben (append)
            _append_csv_details(details_csv_path, current_run_details)

            # 2. Durchschnitt berechnen & Summary updaten
            avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
            avg_prec = np.mean(prec_scores) if prec_scores else 0.0
            avg_rec = np.mean(rec_scores) if rec_scores else 0.0
            avg_conf = np.mean(conf_scores) if conf_scores else 0.0

            summary_rows.append({
                "run_id": run_id, "linkage": linkage, "threshold": thr,
                "avg_pairwise_f1": avg_f1, "avg_pairwise_precision": avg_prec, 
                "avg_pairwise_recall": avg_rec, "avg_session_confidence": avg_conf,
                "n_sessions": len(f1_scores)
            })
            _write_csv_summary(summary_csv_path, summary_rows)

            # 3. Heatmaps für die schlechtesten (optional)
            if HAS_MATPLOTLIB and session_metrics:
                worst = sorted(session_metrics, key=lambda x: x["f1"])[:3]
                for item in worst:
                    try:
                        D, _, _ = load_distance_artifacts(item["dir"])
                        _plot_heatmap(item["dir"], item["dir"] / "_gridsearch" / run_id / "heatmap_dist.png", D, f"{item['dir'].name} | F1={item['f1']:.2f}")
                    except: pass

    LOGGER.info(f"GridSearch abgeschlossen. Ergebnisse:")
    LOGGER.info(f" - Summary: {summary_csv_path}")
    LOGGER.info(f" - Master-CSV: {details_csv_path}")

if __name__ == "__main__":
    main()