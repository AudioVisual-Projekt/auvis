# src/semantic/services/evaluation_semantic_service.py
"""
Schritt 5 (Evaluation) – rein semantische Pipeline (Team C, Task 5 Clustering).

Implementiert:
- scope=dev
- matching=strict
- speaker_f1=yes
- wer/jacer=off

Inputs pro Session:
- Prediction:  <semantic_output_root>/session_*/speaker_to_cluster.json
- Meta (Step 4): <semantic_output_root>/session_*/semantic_clustering_meta.json (für Confidence)
- Ground Truth: <baseline_root>/session_*/labels/speaker_to_cluster.json

Outputs:
- pro Session: <semantic_output_root>/session_*/evaluation.json
- run-weit:    <semantic_output_root>/evaluation_summary.json
- run-weit:    <semantic_output_root>/run_meta_evaluation_semantic.json
- optional (config.write_csv): evaluation_sessions.csv, evaluation_errors.csv
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from semantic.models.evaluation_semantic_models import (
    EvaluationConfig,
    EvaluationSummary,
    PairwiseCounts,
    PairwiseMetrics,
    SessionEvaluationResult,
    SessionMetricRow,
    SpeakerF1Score,
    SpeakerPairError,
    Step5RunMeta,
    ValidationReport,
)

LOGGER = logging.getLogger("semantic.evaluation")

PRED_SPEAKER_TO_CLUSTER = "speaker_to_cluster.json"
GT_SPEAKER_TO_CLUSTER = "speaker_to_cluster.json"
# NEU: Wir lesen auch die Meta-Daten aus Schritt 4, um die Confidence zu holen
CLUSTERING_META_FILE = "semantic_clustering_meta.json"

GT_LABELS_DIR = "labels"

OUT_SESSION_EVAL = "evaluation.json"
OUT_SUMMARY = "evaluation_summary.json"
OUT_RUN_META = "run_meta_evaluation_semantic.json"

OUT_CSV_SESSIONS = "evaluation_sessions.csv"
OUT_CSV_ERRORS = "evaluation_errors.csv"


# ----------------------------
# Zeit / Atomisches Schreiben
# ----------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_write_json(path: Path, obj: Any, *, indent: int = 2, sort_keys: bool = True) -> None:
    """JSON atomisch schreiben (tmp -> replace), deterministisch formatiert."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=sort_keys) + "\n"
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(data, encoding="utf-8")
        tmp.replace(path)
    except Exception: # Fallback für Windows PermissionError u.ä.
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
    finally:
        # Falls replace fehlschlug, aufräumen
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _atomic_write_csv(path: Path, header: List[str], rows: List[Dict[str, Any]]) -> None:
    """CSV atomisch schreiben (tmp -> replace). Spaltenreihenfolge = header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                # None-Werte als Leerstring, sonst String-Repräsentation
                row_dict = {}
                for k in header:
                    val = r.get(k)
                    row_dict[k] = "" if val is None else val
                w.writerow(row_dict)
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# ----------------------------
# Discovery / IO
# ----------------------------

def _list_session_dirs(root: Path) -> List[Path]:
    """
    Deterministische Ermittlung von session_* Verzeichnissen.
    """
    if not root.exists():
        return []
    if root.is_dir() and root.name.startswith("session_"):
        return [root]
    sessions = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("session_")]
    return sorted(sessions, key=lambda p: p.name)


def _read_json_dict(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"JSON ist kein dict: {path}")
    return obj


def _coerce_cluster_map(raw: Dict[str, Any], *, path: Path) -> Dict[str, int]:
    """Erwartet speaker_id -> cluster_id (cluster_id int oder int-konvertierbar)."""
    out: Dict[str, int] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"Ungültiger Speaker-Key in {path}: {repr(k)}")
        if isinstance(v, bool) or v is None:
            raise ValueError(f"Ungültige Cluster-ID (bool/None) für {k} in {path}: {repr(v)}")
        try:
            out[k] = int(v)
        except Exception as e:
            raise ValueError(f"Cluster-ID nicht int-konvertierbar für {k} in {path}: {repr(v)}") from e
    return out


# ----------------------------
# ID-Normalisierung / Matching
# ----------------------------

def normalize_pred_speaker_id(session_name: str, pred_id: str) -> str:
    """
    Normalisiert Prediction-IDs:
    - "session_40_spk_0" -> "spk_0"
    - "spk_0" bleibt "spk_0"
    """
    prefix = f"{session_name}_"
    if pred_id.startswith(prefix):
        return pred_id[len(prefix):]
    return pred_id


def _normalize_map(
    mapping: Dict[str, int],
    *,
    session_name: str,
    enable: bool,
    strict: bool,
) -> Tuple[Dict[str, int], List[str]]:
    """
    Normalisiert Keys. Erkennt Kollisionen.
    """
    if not enable:
        return dict(mapping), []

    out: Dict[str, int] = {}
    warnings: List[str] = []

    for k, v in mapping.items():
        nk = normalize_pred_speaker_id(session_name, k)
        if nk in out and out[nk] != v:
            msg = f"ID-Normalisierung kollidiert: {k} -> {nk} (alt={out[nk]}, neu={v})"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)
            # deterministic choice: keep first
            continue
        out[nk] = v

    return out, warnings


def _validate_speaker_sets_strict(
    gt_map: Dict[str, int],
    pred_map_norm: Dict[str, int],
) -> ValidationReport:
    gt = set(gt_map.keys())
    pr = set(pred_map_norm.keys())

    errors: List[str] = []
    warnings: List[str] = []

    only_pred = sorted(pr - gt)
    only_gt = sorted(gt - pr)

    if only_pred:
        errors.append(f"Prediction enthält Speaker, die nicht in GT sind: {only_pred[:10]} (n={len(only_pred)})")
    if only_gt:
        errors.append(f"GT enthält Speaker, die nicht in Prediction sind: {only_gt[:10]} (n={len(only_gt)})")

    ok = len(errors) == 0
    return ValidationReport(ok=ok, errors=errors, warnings=warnings)


# ----------------------------
# Metriken
# ----------------------------

def _safe_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def compute_pairwise_metrics(
    gt_map: Dict[str, int],
    pred_map: Dict[str, int],
    *,
    collect_pair_errors: bool,
    max_pair_errors: int = 5000,
) -> Tuple[PairwiseCounts, PairwiseMetrics, List[SpeakerPairError]]:
    """Pairwise Precision/Recall/F1 gemäß CHiME (alle ungeordneten Sprecherpaare)."""
    speakers = sorted(gt_map.keys())
    n = len(speakers)
    n_pairs = n * (n - 1) // 2

    tp = fp = fn = 0
    pair_errors: List[SpeakerPairError] = []

    for a, b in combinations(speakers, 2):
        true_same = (gt_map[a] == gt_map[b])
        pred_same = (pred_map[a] == pred_map[b])

        if pred_same and true_same:
            tp += 1
        elif pred_same and not true_same:
            fp += 1
            if collect_pair_errors and len(pair_errors) < max_pair_errors:
                pair_errors.append(
                    SpeakerPairError(
                        speaker_a=a,
                        speaker_b=b,
                        error_type="FP",
                        gt_same_cluster=true_same,
                        pred_same_cluster=pred_same,
                    )
                )
        elif (not pred_same) and true_same:
            fn += 1
            if collect_pair_errors and len(pair_errors) < max_pair_errors:
                pair_errors.append(
                    SpeakerPairError(
                        speaker_a=a,
                        speaker_b=b,
                        error_type="FN",
                        gt_same_cluster=true_same,
                        pred_same_cluster=pred_same,
                    )
                )

    p, r, f1 = _safe_prf(tp, fp, fn)
    return (
        PairwiseCounts(tp=tp, fp=fp, fn=fn, n_pairs_total=n_pairs),
        PairwiseMetrics(precision=p, recall=r, f1=f1),
        pair_errors,
    )


def compute_per_speaker_f1(
    gt_map: Dict[str, int],
    pred_map: Dict[str, int],
) -> Tuple[List[SpeakerF1Score], float]:
    """Per-Speaker one-vs-rest F1 gemäß CHiME 3.1 + Macro-Average."""
    speakers = sorted(gt_map.keys())
    scores: List[SpeakerF1Score] = []

    for i in speakers:
        tp = fp = fn = 0
        for j in speakers:
            if j == i:
                continue
            true_same = (gt_map[i] == gt_map[j])
            pred_same = (pred_map[i] == pred_map[j])

            if pred_same and true_same:
                tp += 1
            elif pred_same and not true_same:
                fp += 1
            elif (not pred_same) and true_same:
                fn += 1

        p, r, f1 = _safe_prf(tp, fp, fn)
        scores.append(
            SpeakerF1Score(
                speaker_id=i,
                precision=p,
                recall=r,
                f1=f1,
                tp=tp,
                fp=fp,
                fn=fn,
            )
        )

    macro = (sum(s.f1 for s in scores) / len(scores)) if scores else 0.0
    return scores, macro


# ----------------------------
# Session-Evaluation
# ----------------------------

def evaluate_one_session(
    *,
    session_dir_pred: Path,
    session_dir_gt: Path,
    config: EvaluationConfig,
    force: bool,
) -> SessionEvaluationResult:
    session_name = session_dir_pred.name

    pred_file = session_dir_pred / PRED_SPEAKER_TO_CLUSTER
    meta_file = session_dir_pred / CLUSTERING_META_FILE  # Step 4 Meta
    gt_file = session_dir_gt / GT_LABELS_DIR / GT_SPEAKER_TO_CLUSTER

    errors: List[str] = []
    warnings: List[str] = []

    if not pred_file.is_file():
        errors.append(f"Missing prediction file: {pred_file}")
    if not gt_file.is_file():
        errors.append(f"Missing GT file: {gt_file}")

    if errors:
        vr = ValidationReport(ok=False, errors=errors, warnings=warnings)
        return SessionEvaluationResult(
            session_name=session_name,
            n_speakers=0,
            validation=vr,
            pairwise_counts=None,
            pairwise_metrics=None,
            speaker_f1=[],
            speaker_f1_macro=None,
            session_confidence=None, # Leer
            pair_errors=[],
            notes=[],
        )

    pred_raw = _read_json_dict(pred_file)
    gt_raw = _read_json_dict(gt_file)
    
    # NEU: Confidence aus Step 4 Meta lesen
    session_confidence: Optional[float] = None
    if meta_file.is_file():
        try:
            meta_raw = _read_json_dict(meta_file)
            val = meta_raw.get("session_confidence")
            if isinstance(val, (int, float)):
                session_confidence = float(val)
        except Exception:
            warnings.append(f"Konnte {meta_file.name} nicht lesen/parsen, ignoriere Confidence.")

    pred_map = _coerce_cluster_map(pred_raw, path=pred_file)
    gt_map = _coerce_cluster_map(gt_raw, path=gt_file)

    pred_map_norm, norm_warnings = _normalize_map(
        pred_map,
        session_name=session_name,
        enable=config.enable_id_normalization,
        strict=config.strict_speaker_matching,
    )
    warnings.extend(norm_warnings)

    vr = _validate_speaker_sets_strict(gt_map=gt_map, pred_map_norm=pred_map_norm)
    # merge warnings (deterministisch)
    vr = ValidationReport(ok=vr.ok, errors=vr.errors, warnings=sorted(set(vr.warnings + warnings)))

    if config.strict_speaker_matching and not vr.ok:
        return SessionEvaluationResult(
            session_name=session_name,
            n_speakers=0,
            validation=vr,
            pairwise_counts=None,
            pairwise_metrics=None,
            speaker_f1=[],
            speaker_f1_macro=None,
            session_confidence=session_confidence, # Wert übergeben auch bei Matching-Fehler
            pair_errors=[],
            notes=["strict_matching_failed"],
        )

    # strict: Sets identisch. Stabiler Align auf GT-Keys.
    speakers = sorted(gt_map.keys())
    missing = [s for s in speakers if s not in pred_map_norm]
    if missing:
        vr2 = ValidationReport(
            ok=False,
            errors=[f"Prediction fehlt Speaker nach Normalisierung: {missing[:10]} (n={len(missing)})"],
            warnings=list(vr.warnings),
        )
        return SessionEvaluationResult(
            session_name=session_name,
            n_speakers=0,
            validation=vr2,
            pairwise_counts=None,
            pairwise_metrics=None,
            speaker_f1=[],
            speaker_f1_macro=None,
            session_confidence=session_confidence,
            pair_errors=[],
            notes=["alignment_failed"],
        )

    pred_aligned = {s: pred_map_norm[s] for s in speakers}
    gt_aligned = {s: gt_map[s] for s in speakers}

    notes: List[str] = []
    pair_counts: Optional[PairwiseCounts] = None
    pair_metrics: Optional[PairwiseMetrics] = None
    pair_errors: List[SpeakerPairError] = []

    if config.compute_pairwise_metrics:
        pair_counts, pair_metrics, pair_errors = compute_pairwise_metrics(
            gt_aligned,
            pred_aligned,
            collect_pair_errors=config.collect_pair_errors,
        )

    spk_scores: List[SpeakerF1Score] = []
    spk_macro: Optional[float] = None
    if config.compute_speaker_f1:
        spk_scores, spk_macro_val = compute_per_speaker_f1(gt_aligned, pred_aligned)
        spk_macro = float(spk_macro_val)

    return SessionEvaluationResult(
        session_name=session_name,
        n_speakers=len(speakers),
        validation=vr,
        pairwise_counts=pair_counts,
        pairwise_metrics=pair_metrics,
        speaker_f1=spk_scores,
        speaker_f1_macro=spk_macro,
        session_confidence=session_confidence, # Hier wird der Wert übergeben
        pair_errors=pair_errors if config.collect_pair_errors else [],
        notes=notes,
    )


# ----------------------------
# Run-Evaluation (Public API)
# ----------------------------

def evaluate_semantic_clustering(
    config: EvaluationConfig,
    *,
    force: bool = False,
    logger: Optional[logging.Logger] = None,
) -> EvaluationSummary:
    """Führt Schritt 5 (Evaluation) über alle Sessions aus (split=dev)."""
    lg = logger or LOGGER
    config.validate()

    pred_root = Path(config.semantic_output_root)
    gt_root = Path(config.baseline_root)

    session_pred_dirs = _list_session_dirs(pred_root)
    sessions_total = len(session_pred_dirs)
    if sessions_total == 0:
        raise FileNotFoundError(f"Keine session_* Ordner gefunden in semantic_output_root: {pred_root}")

    session_rows: List[SessionMetricRow] = []
    sessions_evaluated: List[str] = []
    sessions_failed: List[str] = []
    run_errors: List[str] = []
    run_warnings: List[str] = []
    error_rows: List[Dict[str, Any]] = []

    pw_p: List[float] = []
    pw_r: List[float] = []
    pw_f1: List[float] = []
    spk_macro_list: List[float] = []

    for sdir_pred in session_pred_dirs:
        session_name = sdir_pred.name
        sdir_gt = gt_root / session_name

        try:
            res = evaluate_one_session(
                session_dir_pred=sdir_pred,
                session_dir_gt=sdir_gt,
                config=config,
                force=force,
            )

            # Session JSON schreiben (auch bei Fehlern -> Debugbarkeit)
            if config.write_session_json:
                _atomic_write_json(sdir_pred / OUT_SESSION_EVAL, res.to_dict(), indent=2, sort_keys=True)

            if not res.validation.ok:
                sessions_failed.append(session_name)
                for err in res.validation.errors:
                    error_rows.append({"session": session_name, "level": "error", "message": err})
                for warn in res.validation.warnings:
                    error_rows.append({"session": session_name, "level": "warning", "message": warn})
                continue

            sessions_evaluated.append(session_name)

            row = SessionMetricRow(
                session_name=session_name,
                n_speakers=res.n_speakers,
                pairwise_precision=(res.pairwise_metrics.precision if res.pairwise_metrics else None),
                pairwise_recall=(res.pairwise_metrics.recall if res.pairwise_metrics else None),
                pairwise_f1=(res.pairwise_metrics.f1 if res.pairwise_metrics else None),
                speaker_f1_macro=res.speaker_f1_macro,
                session_confidence=res.session_confidence, # NEU: Übertrag in die CSV-Zeile
            )
            session_rows.append(row)

            if res.pairwise_metrics is not None:
                pw_p.append(res.pairwise_metrics.precision)
                pw_r.append(res.pairwise_metrics.recall)
                pw_f1.append(res.pairwise_metrics.f1)
            if res.speaker_f1_macro is not None:
                spk_macro_list.append(float(res.speaker_f1_macro))

        except Exception as e:
            sessions_failed.append(session_name)
            msg = f"Session {session_name} failed: {repr(e)}"
            lg.exception(msg)
            run_errors.append(msg)
            error_rows.append({"session": session_name, "level": "error", "message": msg})

    avg_pw_p = (sum(pw_p) / len(pw_p)) if pw_p else None
    avg_pw_r = (sum(pw_r) / len(pw_r)) if pw_r else None
    avg_pw_f1 = (sum(pw_f1) / len(pw_f1)) if pw_f1 else None
    avg_spk_macro = (sum(spk_macro_list) / len(spk_macro_list)) if spk_macro_list else None

    summary = EvaluationSummary(
        split=config.split,
        config=config,
        sessions_total=sessions_total,
        sessions_evaluated=sessions_evaluated,
        sessions_failed=sessions_failed,
        avg_pairwise_precision=avg_pw_p,
        avg_pairwise_recall=avg_pw_r,
        avg_pairwise_f1=avg_pw_f1,
        avg_speaker_f1_macro=avg_spk_macro,
        session_rows=session_rows,
        errors=run_errors,
        warnings=run_warnings,
        created_at_utc=_utc_now_iso(),
    )
    summary.validate()

    if config.write_summary_json:
        _atomic_write_json(pred_root / OUT_SUMMARY, summary.to_dict(), indent=2, sort_keys=True)

    run_meta = Step5RunMeta(
        created_at_utc=_utc_now_iso(),
        split=config.split,
        baseline_root=str(gt_root),
        semantic_output_root=str(pred_root),
        sessions_evaluated=sessions_evaluated,
        sessions_failed=sessions_failed,
        parameters={
            "strict_speaker_matching": config.strict_speaker_matching,
            "compute_pairwise_metrics": config.compute_pairwise_metrics,
            "compute_speaker_f1": config.compute_speaker_f1,
            "collect_pair_errors": config.collect_pair_errors,
            "enable_id_normalization": config.enable_id_normalization,
            "force": force,
        },
        notes=[],
    )
    run_meta.validate()
    _atomic_write_json(pred_root / OUT_RUN_META, run_meta.to_dict(), indent=2, sort_keys=True)

    if config.write_csv:
        rows_sessions: List[Dict[str, Any]] = []
        for r in session_rows:
            rows_sessions.append(
                {
                    "session": r.session_name,
                    "n_speakers": r.n_speakers,
                    "pairwise_precision": r.pairwise_precision,
                    "pairwise_recall": r.pairwise_recall,
                    "pairwise_f1": r.pairwise_f1,
                    "speaker_f1_macro": r.speaker_f1_macro,
                    "session_confidence": r.session_confidence, # NEU: Für CSV Output
                }
            )

        _atomic_write_csv(
            pred_root / OUT_CSV_SESSIONS,
            # NEU: Spalte session_confidence im Header
            header=["session", "n_speakers", "pairwise_precision", "pairwise_recall", "pairwise_f1", "speaker_f1_macro", "session_confidence"],
            rows=rows_sessions,
        )
        _atomic_write_csv(
            pred_root / OUT_CSV_ERRORS,
            header=["session", "level", "message"],
            rows=error_rows,
        )

    lg.info(
        "Step5 done: sessions_total=%d evaluated=%d failed=%d avg_pairwise_f1=%s avg_speaker_f1_macro=%s",
        sessions_total,
        len(sessions_evaluated),
        len(sessions_failed),
        f"{avg_pw_f1:.4f}" if isinstance(avg_pw_f1, float) else "n/a",
        f"{avg_spk_macro:.4f}" if isinstance(avg_spk_macro, float) else "n/a",
    )
    return summary