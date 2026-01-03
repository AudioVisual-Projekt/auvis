"""
Module: evaluation_models
=========================
Defines data structures to hold evaluation results in memory.
This prevents "write-only" debugging where we lose access to calculated metrics
after saving them to disk.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SessionEvaluationResult:
    """
    Holds the evaluation metrics for a SINGLE session.

    Attributes:
        session_id (str): The unique session identifier (e.g., 'session_123').
        f1_score (float): Pairwise F1 Score (0.0 to 1.0).
        ari_score (float): Adjusted Rand Index (-1.0 to 1.0).
        is_perfect (bool): True if the clustering matches Ground Truth exactly.

        # Optional: Detailed info
        max_merge_distance (float): The max distance used for merging (uncertainty metric).
        per_speaker_f1 (Dict[str, float]): F1 score for each individual speaker.
    """
    session_id: str
    f1_score: float
    precision: float
    recall: float
    ari_score: float
    max_merge_distance: float = 0.0
    per_speaker_f1: Dict[str, float] = field(default_factory=dict)

    @property
    def is_perfect(self) -> bool:
        """Helper to quickly check for 100% accuracy (considering float precision)."""
        return self.f1_score > 0.9999 and self.ari_score > 0.9999


@dataclass
class ExperimentEvaluationSummary:
    """
    Aggregates the results of ALL sessions for ONE specific clustering experiment.

    Attributes:
        clustering_hash (str): ID of the clustering config evaluated.
        parent_seg_hash (str): ID of the underlying segmentation.
        session_results (List[SessionEvaluationResult]): List of individual session results.
    """
    clustering_hash: str
    parent_seg_hash: str
    session_results: List[SessionEvaluationResult] = field(default_factory=list)

    def get_average_f1(self) -> float:
        """Calculates global average F1 score."""
        if not self.session_results:
            return 0.0
        return sum(r.f1_score for r in self.session_results) / len(self.session_results)

    def get_average_ari(self) -> float:
        """Calculates global average ARI score."""
        if not self.session_results:
            return 0.0
        return sum(r.ari_score for r in self.session_results) / len(self.session_results)

    def add_result(self, result: SessionEvaluationResult):
        self.session_results.append(result)
