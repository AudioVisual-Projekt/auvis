# src/semantic/models/evaluation_semantic_models.py
"""
Datenmodelle für Schritt 5 (Evaluation) der rein semantischen Pipeline (Team C, Semantic-only).

Wichtig:
- Diese Datei enthält KEINE Metriklogik (keine Berechnung).
- Sie definiert nur stabile, deterministisch serialisierbare Ergebnis-/Konfigurationsobjekte.
- Sie ist kompatibel zu:
  - script/build_active_speaker_segments_semantic.py (Run 5)
  - src/semantic/services/evaluation_semantic_service.py (Clustering-only Evaluation)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# -----------------------------
# Hilfsfunktionen (Zeit/Serialisierung)
# -----------------------------

def utc_now_iso() -> str:
    """Aktueller UTC-Zeitstempel als ISO-String (sekundengenau, deterministisch)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _serialize(obj: Any) -> Any:
    """
    Wandelt Python-/Dataclass-Objekte in JSON-serialisierbare Strukturen um (deterministisch).

    Regeln:
    - Dataclasses rekursiv -> Dict
    - datetime -> ISO-String
    - Path -> str
    - set -> sortierte Liste (Determinismus)
    - dict -> Keys sortiert (Determinismus)
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, datetime):
        return obj.isoformat()

    if isinstance(obj, Path):
        return str(obj)

    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in obj.__dict__.items()}

    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            out[str(k)] = _serialize(obj[k])
        return out

    if isinstance(obj, set):
        return [_serialize(x) for x in sorted(obj, key=lambda x: str(x))]

    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]

    return str(obj)


@dataclass(frozen=True)
class JsonExportMixin:
    """Mixin: deterministische Dict/JSON Exporte."""
    def to_dict(self) -> Dict[str, Any]:
        return _serialize(self)

    def to_json_str(self, indent: int = 2, sort_keys: bool = True) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=sort_keys, ensure_ascii=False)

    def write_json(self, path: Union[str, Path], indent: int = 2) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json_str(indent=indent, sort_keys=True), encoding="utf-8")


# -----------------------------
# Konfiguration
# -----------------------------

@dataclass(frozen=True)
class EvaluationConfig(JsonExportMixin):
    """
    Konfiguration für Schritt 5 (Clustering-only Evaluation).

    baseline_root:
      Root mit session_* (z. B. .../data-bin/baseline/dev)
      Erwartet: session_*/labels/speaker_to_cluster.json

    semantic_output_root:
      Root mit session_* (z. B. .../data-bin/_output/dev/semantic_clustering)
      Erwartet: session_*/speaker_to_cluster.json
    """
    split: str = "dev"
    baseline_root: str = ""
    semantic_output_root: str = ""

    strict_speaker_matching: bool = True
    compute_pairwise_metrics: bool = True
    compute_speaker_f1: bool = True

    collect_pair_errors: bool = False
    write_session_json: bool = True
    write_summary_json: bool = True
    write_csv: bool = False

    enable_id_normalization: bool = True

    def validate(self) -> None:
        if not isinstance(self.split, str) or not self.split.strip():
            raise ValueError("EvaluationConfig.split muss ein nicht-leerer String sein.")
        for name in ("baseline_root", "semantic_output_root"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"EvaluationConfig.{name} muss ein nicht-leerer Pfad-String sein.")

        for name in (
            "strict_speaker_matching",
            "compute_pairwise_metrics",
            "compute_speaker_f1",
            "collect_pair_errors",
            "write_session_json",
            "write_summary_json",
            "write_csv",
            "enable_id_normalization",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"EvaluationConfig.{name} muss bool sein.")

        if not self.compute_pairwise_metrics and not self.compute_speaker_f1:
            raise ValueError("Mindestens eine Metrik muss aktiv sein (pairwise oder speaker_f1).")


# -----------------------------
# Validierung / Fehlerobjekte
# -----------------------------

@dataclass(frozen=True)
class ValidationReport(JsonExportMixin):
    """Validierung pro Session (für matching=strict)."""
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if not isinstance(self.ok, bool):
            raise TypeError("ValidationReport.ok muss bool sein.")
        if not isinstance(self.errors, list) or not all(isinstance(x, str) for x in self.errors):
            raise TypeError("ValidationReport.errors muss list[str] sein.")
        if not isinstance(self.warnings, list) or not all(isinstance(x, str) for x in self.warnings):
            raise TypeError("ValidationReport.warnings muss list[str] sein.")


@dataclass(frozen=True)
class SpeakerPairError(JsonExportMixin):
    """
    Fehler auf Paar-Ebene.

    error_type:
      - "FP": Prediction sagt gleiches Cluster, GT sagt verschieden
      - "FN": Prediction sagt verschieden, GT sagt gleich
    """
    speaker_a: str
    speaker_b: str
    error_type: str  # "FP" oder "FN"
    gt_same_cluster: bool
    pred_same_cluster: bool

    def validate(self) -> None:
        for k in ("speaker_a", "speaker_b", "error_type"):
            v = getattr(self, k)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"SpeakerPairError.{k} muss ein nicht-leerer String sein.")
        if self.error_type not in ("FP", "FN"):
            raise ValueError("SpeakerPairError.error_type muss 'FP' oder 'FN' sein.")
        if not isinstance(self.gt_same_cluster, bool) or not isinstance(self.pred_same_cluster, bool):
            raise TypeError("SpeakerPairError.gt_same_cluster/pred_same_cluster müssen bool sein.")


# -----------------------------
# Pairwise Metriken
# -----------------------------

@dataclass(frozen=True)
class PairwiseCounts(JsonExportMixin):
    """Zählwerte für Pairwise-Auswertung."""
    tp: int
    fp: int
    fn: int
    n_pairs_total: int

    def validate(self) -> None:
        for k in ("tp", "fp", "fn", "n_pairs_total"):
            v = getattr(self, k)
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"PairwiseCounts.{k} muss eine nicht-negative Ganzzahl sein.")


@dataclass(frozen=True)
class PairwiseMetrics(JsonExportMixin):
    """Pairwise Precision/Recall/F1."""
    precision: float
    recall: float
    f1: float

    def validate(self) -> None:
        for k in ("precision", "recall", "f1"):
            v = getattr(self, k)
            if not isinstance(v, (int, float)):
                raise TypeError(f"PairwiseMetrics.{k} muss float sein.")
            if v < 0.0 or v > 1.0:
                raise ValueError(f"PairwiseMetrics.{k} muss in [0,1] liegen.")


# -----------------------------
# Per-Speaker One-vs-Rest
# -----------------------------

@dataclass(frozen=True)
class SpeakerF1Score(JsonExportMixin):
    """Per-Speaker One-vs-Rest F1 (CHiME 3.1)."""
    speaker_id: str
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def validate(self) -> None:
        if not isinstance(self.speaker_id, str) or not self.speaker_id.strip():
            raise ValueError("SpeakerF1Score.speaker_id muss ein nicht-leerer String sein.")
        for k in ("precision", "recall", "f1"):
            v = getattr(self, k)
            if not isinstance(v, (int, float)):
                raise TypeError(f"SpeakerF1Score.{k} muss float sein.")
            if v < 0.0 or v > 1.0:
                raise ValueError(f"SpeakerF1Score.{k} muss in [0,1] liegen.")
        for k in ("tp", "fp", "fn"):
            v = getattr(self, k)
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"SpeakerF1Score.{k} muss eine nicht-negative Ganzzahl sein.")


# -----------------------------
# Session- und Run-Ergebnisse
# -----------------------------

@dataclass(frozen=True)
class SessionEvaluationResult(JsonExportMixin):
    """Ergebnis pro Session."""
    session_name: str
    n_speakers: int
    validation: ValidationReport

    pairwise_counts: Optional[PairwiseCounts] = None
    pairwise_metrics: Optional[PairwiseMetrics] = None

    speaker_f1: List[SpeakerF1Score] = field(default_factory=list)
    speaker_f1_macro: Optional[float] = None
    
    # NEU: Confidence aus Schritt 4 (für Vergleich mit Team-Time)
    session_confidence: Optional[float] = None

    pair_errors: List[SpeakerPairError] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if not isinstance(self.session_name, str) or not self.session_name.strip():
            raise ValueError("SessionEvaluationResult.session_name muss ein nicht-leerer String sein.")
        if not isinstance(self.n_speakers, int) or self.n_speakers < 0:
            raise ValueError("SessionEvaluationResult.n_speakers muss eine nicht-negative Ganzzahl sein.")

        self.validation.validate()

        if self.pairwise_counts is not None:
            self.pairwise_counts.validate()
        if self.pairwise_metrics is not None:
            self.pairwise_metrics.validate()

        if not isinstance(self.speaker_f1, list) or not all(isinstance(x, SpeakerF1Score) for x in self.speaker_f1):
            raise TypeError("SessionEvaluationResult.speaker_f1 muss list[SpeakerF1Score] sein.")
        for s in self.speaker_f1:
            s.validate()

        if self.speaker_f1_macro is not None:
            if not isinstance(self.speaker_f1_macro, (int, float)):
                raise TypeError("SessionEvaluationResult.speaker_f1_macro muss float oder None sein.")
            if self.speaker_f1_macro < 0.0 or self.speaker_f1_macro > 1.0:
                raise ValueError("SessionEvaluationResult.speaker_f1_macro muss in [0,1] liegen.")
        
        # Validierung für Confidence
        if self.session_confidence is not None:
            if not isinstance(self.session_confidence, (int, float)):
                raise TypeError("SessionEvaluationResult.session_confidence muss float oder None sein.")

        if not isinstance(self.pair_errors, list) or not all(isinstance(x, SpeakerPairError) for x in self.pair_errors):
            raise TypeError("SessionEvaluationResult.pair_errors muss list[SpeakerPairError] sein.")
        for e in self.pair_errors:
            e.validate()

        if not isinstance(self.notes, list) or not all(isinstance(x, str) for x in self.notes):
            raise TypeError("SessionEvaluationResult.notes muss list[str] sein.")


@dataclass(frozen=True)
class SessionMetricRow(JsonExportMixin):
    """Kompakte Zeile pro Session (CSV/Plots)."""
    session_name: str
    n_speakers: int
    pairwise_precision: Optional[float]
    pairwise_recall: Optional[float]
    pairwise_f1: Optional[float]
    speaker_f1_macro: Optional[float]
    
    # NEU: Confidence für CSV
    session_confidence: Optional[float]

    def validate(self) -> None:
        if not isinstance(self.session_name, str) or not self.session_name.strip():
            raise ValueError("SessionMetricRow.session_name muss ein nicht-leerer String sein.")
        if not isinstance(self.n_speakers, int) or self.n_speakers < 0:
            raise ValueError("SessionMetricRow.n_speakers muss eine nicht-negative Ganzzahl sein.")
        
        # Alle Metriken prüfen (inkl. Confidence)
        for k in ("pairwise_precision", "pairwise_recall", "pairwise_f1", "speaker_f1_macro", "session_confidence"):
            v = getattr(self, k)
            if v is None:
                continue
            if not isinstance(v, (int, float)):
                raise TypeError(f"SessionMetricRow.{k} muss float oder None sein.")
            # Confidence kann 1.0 sein, Scores auch. [0,1] sollte passen.
            if v < 0.0 or v > 1.0:
                 pass # Erlauben wir technisch, auch wenn ungewöhnlich.


@dataclass(frozen=True)
class EvaluationSummary(JsonExportMixin):
    """Aggregierte Auswertung über alle Sessions."""
    split: str
    config: EvaluationConfig

    sessions_total: int
    sessions_evaluated: List[str] = field(default_factory=list)
    sessions_failed: List[str] = field(default_factory=list)

    avg_pairwise_precision: Optional[float] = None
    avg_pairwise_recall: Optional[float] = None
    avg_pairwise_f1: Optional[float] = None

    avg_speaker_f1_macro: Optional[float] = None

    session_rows: List[SessionMetricRow] = field(default_factory=list)

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    created_at_utc: str = field(default_factory=utc_now_iso)

    def validate(self) -> None:
        if not isinstance(self.split, str) or not self.split.strip():
            raise ValueError("EvaluationSummary.split muss ein nicht-leerer String sein.")
        self.config.validate()
        if not isinstance(self.sessions_total, int) or self.sessions_total < 0:
            raise ValueError("EvaluationSummary.sessions_total muss eine nicht-negative Ganzzahl sein.")

        for k in ("sessions_evaluated", "sessions_failed", "errors", "warnings"):
            v = getattr(self, k)
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise TypeError(f"EvaluationSummary.{k} muss list[str] sein.")

        for k in ("avg_pairwise_precision", "avg_pairwise_recall", "avg_pairwise_f1", "avg_speaker_f1_macro"):
            v = getattr(self, k)
            if v is None:
                continue
            if not isinstance(v, (int, float)):
                raise TypeError(f"EvaluationSummary.{k} muss float oder None sein.")
            if v < 0.0 or v > 1.0:
                raise ValueError(f"EvaluationSummary.{k} muss in [0,1] liegen.")

        if not isinstance(self.session_rows, list) or not all(isinstance(x, SessionMetricRow) for x in self.session_rows):
            raise TypeError("EvaluationSummary.session_rows muss list[SessionMetricRow] sein.")
        for r in self.session_rows:
            r.validate()

        if not isinstance(self.created_at_utc, str) or not self.created_at_utc.strip():
            raise ValueError("EvaluationSummary.created_at_utc muss ein nicht-leerer String sein.")


@dataclass(frozen=True)
class Step5RunMeta(JsonExportMixin):
    """Run-Metadaten für Schritt 5 (Evaluation)."""
    created_at_utc: str = field(default_factory=utc_now_iso)
    split: str = "dev"
    baseline_root: str = ""
    semantic_output_root: str = ""
    sessions_evaluated: List[str] = field(default_factory=list)
    sessions_failed: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def validate(self) -> None:
        for k in ("created_at_utc", "split", "baseline_root", "semantic_output_root"):
            v = getattr(self, k)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"Step5RunMeta.{k} muss ein nicht-leerer String sein.")
        for k in ("sessions_evaluated", "sessions_failed", "notes"):
            v = getattr(self, k)
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise TypeError(f"Step5RunMeta.{k} muss list[str] sein.")
        if not isinstance(self.parameters, dict):
            raise TypeError("Step5RunMeta.parameters muss dict sein.")