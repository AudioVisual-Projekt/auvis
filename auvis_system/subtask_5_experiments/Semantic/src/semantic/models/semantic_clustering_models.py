"""
src/semantic/models/semantic_clustering_models.py

Modelle/Konfiguration für Schritt 4 (Semantic Clustering) – sessionweise.

Wichtig:
- Wir arbeiten mit einer *vorberechneten Distanzmatrix* (metric="precomputed").
- Es darf genau *eine* Stopp-Logik aktiv sein:
  (a) distance_threshold  ODER  (b) n_clusters
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Literal


Linkage = Literal["complete", "average", "single"]


@dataclass(frozen=True)
class SemanticClusteringConfig:
    """
    Konfiguration für Agglomerative Clustering auf einer vorgegebenen Distanzmatrix.

    Attribute:
        linkage:
            Wie der Abstand zwischen Clustern gemessen wird:
            - "complete": konservativ (max. Paarabstand)
            - "average" : robust (Durchschnitt)
            - "single"  : kann Kettenbildung verursachen

        distance_threshold:
            Distanz-Schwelle auf D (klein = ähnlich).
            Nur verwenden, wenn n_clusters = None.

        n_clusters:
            Feste Anzahl Cluster.
            Nur verwenden, wenn distance_threshold = None.

        strict_checks:
            Wenn True: zusätzliche Validierungen (z. B. NaN/Inf) sollen im Service
            streng geprüft werden (die eigentliche Prüfung erfolgt dort).

        renumber_labels:
            Wenn True: Cluster-Labels werden nach dem Clustering deterministisch
            renummeriert (z. B. nach kleinster UID/Index je Cluster; Umsetzung im Service).

        expected_distance_type:
            Optionaler Erwartungswert (z. B. "cosine") für distance_meta.json.
            Wenn None: keine Prüfung.
    """

    linkage: Linkage = "average"
    distance_threshold: Optional[float] = 0.76
    n_clusters: Optional[int] = None

    strict_checks: bool = True
    renumber_labels: bool = True
    expected_distance_type: Optional[str] = "cosine"

    def __post_init__(self) -> None:
        """
        Validiert die Konfiguration.

        Regeln:
        - Genau eine der Optionen distance_threshold oder n_clusters muss gesetzt sein.
        - distance_threshold muss > 0 sein (falls gesetzt).
        - n_clusters muss >= 1 sein (falls gesetzt).
        - linkage muss einer der erlaubten Werte sein.
        """
        if self.linkage not in ("complete", "average", "single"):
            raise ValueError("linkage muss 'complete', 'average' oder 'single' sein.")

        # Genau eine Stopp-Option muss gesetzt sein (XOR)
        if (self.distance_threshold is None) == (self.n_clusters is None):
            raise ValueError(
                "Ungültige Konfiguration: Setze entweder "
                "'distance_threshold' ODER 'n_clusters' (genau eine Option)."
            )

        # Plausibilitätschecks
        if self.distance_threshold is not None:
            if not isinstance(self.distance_threshold, (float, int)):
                raise TypeError("distance_threshold muss eine Zahl (float/int) sein.")
            if float(self.distance_threshold) <= 0.0:
                raise ValueError("distance_threshold muss > 0 sein.")

        if self.n_clusters is not None:
            if not isinstance(self.n_clusters, int):
                raise TypeError("n_clusters muss ein int sein.")
            if self.n_clusters < 1:
                raise ValueError("n_clusters muss >= 1 sein.")

        if not isinstance(self.strict_checks, bool):
            raise TypeError("strict_checks muss bool sein.")
        if not isinstance(self.renumber_labels, bool):
            raise TypeError("renumber_labels muss bool sein.")
        if self.expected_distance_type is not None and not isinstance(self.expected_distance_type, str):
            raise TypeError("expected_distance_type muss str oder None sein.")


@dataclass(frozen=True)
class SemanticClusteringMeta:
    """
    Metadaten zum Clustering-Ergebnis einer Session.
    Wird als semantic_clustering_meta.json gespeichert.
    """
    session_name: str
    n_speakers: int
    config: SemanticClusteringConfig
    created_at_utc: str
    
    # NEU: Metriken zur Vergleichbarkeit mit Team-Time (analog zu clustering_engine.py)
    # session_confidence = 1.0 - max_merge_distance
    max_merge_distance: Optional[float] = None
    session_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_name": self.session_name,
            "n_speakers": self.n_speakers,
            "config": asdict(self.config),
            "created_at_utc": self.created_at_utc,
            "max_merge_distance": self.max_merge_distance,
            "session_confidence": self.session_confidence,
        }

    def to_json_str(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)