# script/build_active_speaker_segments_semantic.py
"""
Orchestrator für die semantische Pipeline (DEV) – Team C (Semantic-only)

Pipeline (sessionweise)
-----------------------
1) Text-Extraktion (sessionweise)
   Output je Session:
     - out_dir/session_*/extracted_texts.json
   Run-weit:
     - out_dir/run_meta_extract_texts.json

2) Embeddings (sessionweise)
   Output je Session:
     - out_dir/session_*/embeddings.npy
     - out_dir/session_*/embedding_ids.json
     - optional: out_dir/session_*/embed_meta.json
   Run-weit:
     - out_dir/run_meta_embed_texts.json

3) Distanzmatrix (sessionweise)
   Inputs je Session:
     - out_dir/session_*/embeddings.npy
     - out_dir/session_*/embedding_ids.json
   Output je Session:
     - out_dir/session_*/distance_matrix.npy
     - out_dir/session_*/distance_ids.json
     - optional: out_dir/session_*/distance_meta.json
   Run-weit:
     - out_dir/run_meta_distance_matrix.json

4) Semantic Clustering (sessionweise)
   Inputs je Session:
     - out_dir/session_*/distance_matrix.npy
     - out_dir/session_*/distance_ids.json
   Output je Session:
     - out_dir/session_*/speaker_to_cluster.json
     - optional: out_dir/session_*/semantic_clustering_meta.json
   Run-weit:
     - out_dir/run_meta_semantic_clustering.json

5) Evaluation (Clustering-only; CHiME Task-5-relevant)
   Inputs je Session:
     - Prediction: out_dir/session_*/speaker_to_cluster.json
     - Ground Truth: baseline-dev-dir/session_*/labels/speaker_to_cluster.json
   Output je Session:
     - out_dir/session_*/evaluation.json
   Run-weit:
     - out_dir/evaluation_summary.json
     - out_dir/run_meta_evaluation_semantic.json
     - optional: out_dir/evaluation_sessions.csv, out_dir/evaluation_errors.csv

Hinweis:
- Dieser Orchestrator ist strikt "semantic-only" (keine ASD/Timing-Logik).
- WER/JACER sind in Schritt 5 bewusst nicht implementiert (siehe evaluation_semantic_service.py).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np  # (bewusst toleriert; in manchen Umgebungen für Debug/Checks nützlich)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger("semantic.orchestrator")

# ---------------------------------------------------------------------------
# Import-Pfad-Setup
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from semantic.models.extract_texts_models import SCHEMA_VERSION
from semantic.models.semantic_clustering_models import SemanticClusteringConfig
from semantic.services.embed_texts_service import EmbedConfig, embed_sessions
from semantic.services.extract_texts_service import extract_texts_by_session, write_session_extracted_texts
from semantic.services.distance_matrix_service import build_distance_matrix_session_from_files
from semantic.services.semantic_clustering_service import run_clustering_for_session_dir
from semantic.models.evaluation_semantic_models import EvaluationConfig
from semantic.services.evaluation_semantic_service import evaluate_semantic_clustering

# ---------------------------------------------------------------------------
# Constants (Artefakte)
# ---------------------------------------------------------------------------
TEXTS_FILE = "extracted_texts.json"

EMB_FILE = "embeddings.npy"
EMB_IDS_FILE = "embedding_ids.json"
EMB_META_FILE = "embed_meta.json"

DIST_FILE = "distance_matrix.npy"
DIST_IDS_FILE = "distance_ids.json"
DIST_META_FILE = "distance_meta.json"

CLUSTER_FILE = "speaker_to_cluster.json"
CLUSTER_META_FILE = "semantic_clustering_meta.json"

SESSION_EVAL_FILE = "evaluation.json"
EVAL_SUMMARY_FILE = "evaluation_summary.json"

META_STEP1 = "run_meta_extract_texts.json"
META_STEP2 = "run_meta_embed_texts.json"
META_STEP3 = "run_meta_distance_matrix.json"
META_STEP4 = "run_meta_semantic_clustering.json"
META_STEP5 = "run_meta_evaluation_semantic.json"  # wird vom Service geschrieben

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Semantische Pipeline (DEV, sessionweise): Schritte 1–5 einzeln oder als Gesamtlauf.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--run",
        required=True,
        help="Auszuführender Schritt: 1..5, 'all' oder kommaseparierte Liste (z.B. 1,2,3,4,5).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Überschreibt Artefakte (setzt intern step-spezifische Force-Flags).",
    )

    p.add_argument("--baseline-dev-dir", default="data-bin/baseline/dev")
    p.add_argument("--out-dir", default="data-bin/_output/dev/semantic_clustering")
    p.add_argument("--output-dir-name", default="output")

    # Schritt 2
    p.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    p.add_argument("--embed-batch-size", type=int, default=32)
    p.add_argument("--embed-device", default=None)
    p.add_argument("--embed-normalize", action="store_true")
    p.add_argument("--embed-force", action="store_true")
    p.add_argument("--embed-no-meta", action="store_true")

    # Schritt 3
    p.add_argument("--dist-type", default="cosine", choices=["cosine"])
    p.add_argument("--dist-block-size", type=int, default=2048)
    p.add_argument("--dist-eps", type=float, default=1e-12)
    p.add_argument("--dist-force", action="store_true")
    p.add_argument("--dist-no-meta", action="store_true")
    p.add_argument("--dist-no-strict-checks", action="store_true")
    p.add_argument("--dist-show-progress", action="store_true")

    # Schritt 4
    p.add_argument("--cluster-linkage", default="average", choices=["complete", "average", "single"])
    p.add_argument("--cluster-distance-threshold", type=float, default=0.76)
    p.add_argument("--cluster-n-clusters", type=int, default=None)
    p.add_argument("--cluster-force", action="store_true")
    p.add_argument("--cluster-no-meta", action="store_true")
    p.add_argument("--cluster-no-strict-checks", action="store_true")
    p.add_argument("--cluster-no-renumber", action="store_true")
    p.add_argument("--cluster-expected-distance-type", default="cosine")

    # Schritt 5
    p.add_argument("--eval-force", action="store_true")
    p.add_argument("--eval-write-csv", action="store_true")
    p.add_argument("--eval-collect-pair-errors", action="store_true")
    p.add_argument("--eval-no-speaker-f1", action="store_true")
    p.add_argument("--eval-no-id-normalization", action="store_true")

    return p.parse_args()


def _parse_run_spec(run_spec: str) -> List[int]:
    """Parst --run ("all", "3" oder "1,2,3") zu einer Step-Liste in Ausführungsreihenfolge."""
    spec = (run_spec or "").strip().lower()
    if not spec:
        raise ValueError("--run darf nicht leer sein.")

    if spec == "all":
        return [1, 2, 3, 4, 5]

    if "," in spec:
        raw = [s.strip() for s in spec.split(",") if s.strip()]
        steps: List[int] = []
        seen = set()
        for part in raw:
            if not part.isdigit():
                raise ValueError(f"Ungültiger --run Eintrag: {part!r} (erwartet Ziffer 1..5).")
            v = int(part)
            if v < 1 or v > 5:
                raise ValueError(f"Ungültiger --run Schritt: {v} (erwartet 1..5).")
            if v not in seen:
                steps.append(v)
                seen.add(v)
        if not steps:
            raise ValueError("Ungültiger --run Wert: keine Schritte erkannt.")
        return steps

    if not spec.isdigit():
        raise ValueError(f"Ungültiger --run Wert: {run_spec!r} (erwartet 1..5, 'all' oder Liste).")
    v = int(spec)
    if v < 1 or v > 5:
        raise ValueError(f"Ungültiger --run Schritt: {v} (erwartet 1..5).")
    return [v]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _serialize(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    return str(obj)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(_serialize(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        LOGGER.exception("Konnte JSON nicht lesen: %s", path)
        return None


def _list_session_dirs(out_dir: Path) -> List[Path]:
    """Deterministisch: session_* Ordner lexikographisch sortiert."""
    if out_dir.is_dir() and out_dir.name.startswith("session_"):
        return [out_dir]
    if not out_dir.exists():
        return []
    sessions = [p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("session_")]
    return sorted(sessions, key=lambda p: p.name)


def _build_clustering_config(args: argparse.Namespace) -> SemanticClusteringConfig:
    # XOR-Regel: wenn n_clusters gesetzt ist, muss distance_threshold None sein.
    distance_threshold = None if args.cluster_n_clusters is not None else float(args.cluster_distance_threshold)
    n_clusters = int(args.cluster_n_clusters) if args.cluster_n_clusters is not None else None

    return SemanticClusteringConfig(
        linkage=str(args.cluster_linkage),
        distance_threshold=distance_threshold,
        n_clusters=n_clusters,
        strict_checks=not bool(args.cluster_no_strict_checks),
        renumber_labels=not bool(args.cluster_no_renumber),
        expected_distance_type=str(args.cluster_expected_distance_type),
    )


# ---------------------------------------------------------------------------
# Step 1
# ---------------------------------------------------------------------------
def _run_step1_extract(args: argparse.Namespace, *, baseline_dir: str, out_dir: Path) -> dict:
    sessions = extract_texts_by_session(
        baseline_dir=baseline_dir,
        output_dir_name=args.output_dir_name,
    )

    sessions_total = 0
    sessions_built = 0
    sessions_failed = 0
    failed_sessions: List[str] = []

    for extracted in sessions:
        sessions_total += 1
        session_name = extracted.session_name
        try:
            write_session_extracted_texts(extracted=extracted, out_dir=out_dir, filename=TEXTS_FILE)
            sessions_built += 1
        except Exception as e:
            sessions_failed += 1
            failed_sessions.append(session_name)
            LOGGER.exception("Step1 failed for %s: %s", session_name, e)

    run_meta = {
        "timestamp": _now(),
        "schema_version": int(SCHEMA_VERSION),
        "baseline_dev_dir": str(baseline_dir),
        "out_dir": str(out_dir),
        "output_dir_name": str(args.output_dir_name),
        "sessions_total": int(sessions_total),
        "sessions_built": int(sessions_built),
        "sessions_failed": int(sessions_failed),
        "failed_sessions": failed_sessions,
    }
    _atomic_write_json(out_dir / META_STEP1, run_meta)

    print(f"[OK] Schritt 1: built={sessions_built}/{sessions_total}, failed={sessions_failed}")
    print(f"[OK] Schritt 1: Meta {out_dir / META_STEP1}")
    return run_meta


# ---------------------------------------------------------------------------
# Step 2
# ---------------------------------------------------------------------------
def _run_step2_embed(args: argparse.Namespace, *, out_dir: Path) -> dict:
    session_dirs = _list_session_dirs(out_dir)
    if not session_dirs:
        raise FileNotFoundError(f"Keine session_* Ordner in out_dir gefunden: {out_dir}")

    cfg = EmbedConfig(
        model_name=args.embed_model,
        batch_size=int(args.embed_batch_size),
        device=args.embed_device,
        normalize_embeddings=bool(args.embed_normalize),
        force=bool(args.embed_force),
        write_meta=not bool(args.embed_no_meta),
    )

    meta = embed_sessions(session_dirs=session_dirs, out_dir=out_dir, config=cfg)

    _atomic_write_json(out_dir / META_STEP2, meta)

    m = _serialize(meta)
    sessions_total = m.get("sessions_total")
    sessions_embedded = m.get("sessions_embedded")
    sessions_skipped = m.get("sessions_skipped")
    failed = len(m.get("errors") or [])
    print(f"[OK] Schritt 2: embedded={sessions_embedded}/{sessions_total}, skipped={sessions_skipped}, failed={failed}")
    print(f"[OK] Schritt 2: Meta {out_dir / META_STEP2}")
    return m


# ---------------------------------------------------------------------------
# Step 3
# ---------------------------------------------------------------------------
def _run_step3_distance_matrix(args: argparse.Namespace, *, out_dir: Path) -> dict:
    session_dirs = _list_session_dirs(out_dir)
    if not session_dirs:
        raise FileNotFoundError(f"Keine session_* Ordner in out_dir gefunden: {out_dir}")

    sessions_total = len(session_dirs)
    built = skipped = failed = 0
    failed_sessions: List[str] = []

    for sdir in session_dirs:
        if not args.dist_force:
            have_main = (sdir / DIST_FILE).is_file() and (sdir / DIST_IDS_FILE).is_file()
            have_meta = bool(args.dist_no_meta) or (sdir / DIST_META_FILE).is_file()
            if have_main and have_meta:
                skipped += 1
                LOGGER.info("Step3 skip %s (cached)", sdir.name)
                continue

        try:
            embed_meta = _read_json_if_exists(sdir / EMB_META_FILE) or {}
            input_fingerprint = embed_meta.get("input_fingerprint")
            model_name = embed_meta.get("model_name")

            artifacts = build_distance_matrix_session_from_files(
                embeddings_path=sdir / EMB_FILE,
                ids_path=sdir / EMB_IDS_FILE,
                distance_type=str(args.dist_type),
                block_size=int(args.dist_block_size),
                eps=float(args.dist_eps),
                show_progress=bool(args.dist_show_progress),
                input_fingerprint=input_fingerprint,
                model_name=model_name,
                strict_numeric_checks=not bool(args.dist_no_strict_checks),
            )

            artifacts.to_files(
                sdir,
                distance_matrix_name=DIST_FILE,
                distance_ids_name=DIST_IDS_FILE,
                distance_meta_name=DIST_META_FILE,
                overwrite=bool(args.dist_force),
                write_meta=not bool(args.dist_no_meta),
                strict_numeric_checks=not bool(args.dist_no_strict_checks),
            )

            built += 1
        except Exception as e:
            failed += 1
            failed_sessions.append(sdir.name)
            LOGGER.exception("Step3 failed for %s: %s", sdir.name, e)

    run_meta = {
        "timestamp": _now(),
        "out_dir": str(out_dir),
        "config": {
            "distance_type": str(args.dist_type),
            "block_size": int(args.dist_block_size),
            "eps": float(args.dist_eps),
            "show_progress": bool(args.dist_show_progress),
            "strict_numeric_checks": not bool(args.dist_no_strict_checks),
        },
        "write_meta": not bool(args.dist_no_meta),
        "sessions_total": int(sessions_total),
        "sessions_built": int(built),
        "sessions_skipped": int(skipped),
        "sessions_failed": int(failed),
        "failed_sessions": failed_sessions,
    }
    _atomic_write_json(out_dir / META_STEP3, run_meta)

    print(f"[OK] Schritt 3: built={built}/{sessions_total}, skipped={skipped}, failed={failed}")
    print(f"[OK] Schritt 3: Meta {out_dir / META_STEP3}")
    return run_meta


# ---------------------------------------------------------------------------
# Step 4
# ---------------------------------------------------------------------------
def _run_step4_semantic_clustering(args: argparse.Namespace, *, out_dir: Path) -> dict:
    session_dirs = _list_session_dirs(out_dir)
    if not session_dirs:
        raise FileNotFoundError(f"Keine session_* Ordner in out_dir gefunden: {out_dir}")

    config = _build_clustering_config(args)
    write_meta = not bool(args.cluster_no_meta)

    sessions_total = len(session_dirs)
    built = skipped = failed = 0
    failed_sessions: List[str] = []

    for sdir in session_dirs:
        if not args.cluster_force:
            if (sdir / CLUSTER_FILE).is_file() and (not write_meta or (sdir / CLUSTER_META_FILE).is_file()):
                skipped += 1
                LOGGER.info("Step4 skip %s (cached)", sdir.name)
                continue

        try:
            run_clustering_for_session_dir(
                session_dir=sdir,
                config=config,
                write_meta=write_meta,
            )
            built += 1
        except Exception as e:
            failed += 1
            failed_sessions.append(sdir.name)
            LOGGER.exception("Step4 failed for %s: %s", sdir.name, e)

    run_meta = {
        "timestamp": _now(),
        "out_dir": str(out_dir),
        "config": _serialize(config),
        "write_meta": bool(write_meta),
        "sessions_total": int(sessions_total),
        "sessions_built": int(built),
        "sessions_skipped": int(skipped),
        "sessions_failed": int(failed),
        "failed_sessions": failed_sessions,
    }
    _atomic_write_json(out_dir / META_STEP4, run_meta)

    print(f"[OK] Schritt 4: built={built}/{sessions_total}, skipped={skipped}, failed={failed}")
    print(f"[OK] Schritt 4: Meta {out_dir / META_STEP4}")
    return run_meta


# ---------------------------------------------------------------------------
# Step 5
# ---------------------------------------------------------------------------
def _run_step5_evaluate(args: argparse.Namespace, *, baseline_dir: str, out_dir: Path) -> dict:
    session_dirs = _list_session_dirs(out_dir)
    if not session_dirs:
        raise FileNotFoundError(f"Keine session_* Ordner in out_dir gefunden: {out_dir}")

    cfg = EvaluationConfig(
        split="dev",
        baseline_root=str(Path(baseline_dir)),
        semantic_output_root=str(out_dir),
        strict_speaker_matching=True,
        compute_pairwise_metrics=True,
        compute_speaker_f1=not bool(args.eval_no_speaker_f1),
        collect_pair_errors=bool(args.eval_collect_pair_errors),
        write_session_json=True,
        write_summary_json=True,
        write_csv=bool(args.eval_write_csv),
        enable_id_normalization=not bool(args.eval_no_id_normalization),
    )

    summary = evaluate_semantic_clustering(cfg, force=bool(args.eval_force))

    s = _serialize(summary)
    sessions_total = s.get("sessions_total")
    evaluated = len(s.get("sessions_evaluated") or [])
    failed = len(s.get("sessions_failed") or [])
    avg_pw_f1 = s.get("avg_pairwise_f1")

    print(f"[OK] Schritt 5: sessions_total={sessions_total}, evaluated={evaluated}, failed={failed}")
    if isinstance(avg_pw_f1, float):
        print(f"[OK] Schritt 5: Avg Pairwise F1 (über Sessions): {avg_pw_f1:.4f}")
    else:
        print("[OK] Schritt 5: Avg Pairwise F1: n/a")

    if not args.eval_no_speaker_f1:
        avg_spk_macro = s.get("avg_speaker_f1_macro")
        if isinstance(avg_spk_macro, float):
            print(f"[OK] Schritt 5: Avg Per-Speaker F1 Macro (über Sessions): {avg_spk_macro:.4f}")
        else:
            print("[OK] Schritt 5: Avg Per-Speaker F1 Macro: n/a")

    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        args = _parse_args()

        # global --force propagieren (step-spezifisch)
        if args.force:
            args.embed_force = True
            args.dist_force = True
            args.cluster_force = True
            args.eval_force = True

        steps = _parse_run_spec(args.run)

        baseline_dir = os.path.abspath(args.baseline_dev_dir)
        out_dir = Path(os.path.abspath(args.out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)

        failures = 0

        for step in steps:
            if step == 1:
                m = _run_step1_extract(args, baseline_dir=baseline_dir, out_dir=out_dir)
                failures += int(m.get("sessions_failed", 0))
            elif step == 2:
                m = _run_step2_embed(args, out_dir=out_dir)
                failures += len(m.get("errors") or [])
            elif step == 3:
                m = _run_step3_distance_matrix(args, out_dir=out_dir)
                failures += int(m.get("sessions_failed", 0))
            elif step == 4:
                m = _run_step4_semantic_clustering(args, out_dir=out_dir)
                failures += int(m.get("sessions_failed", 0))
            elif step == 5:
                m = _run_step5_evaluate(args, baseline_dir=baseline_dir, out_dir=out_dir)
                failures += len(m.get("sessions_failed") or [])
            else:
                raise ValueError(f"Unbekannter Step: {step}")

        # Exit-Codes:
        # 0 = alles ok
        # 2 = mindestens ein Session-Fehler (teilweise erfolgreich)
        return 0 if failures == 0 else 2

    except Exception as e:
        LOGGER.exception("Pipeline abgebrochen: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
