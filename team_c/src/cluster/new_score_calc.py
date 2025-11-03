from typing import List, Tuple
from typing import List, Tuple, Dict
import numpy as np


def calculate_overlap_duration(
        segments1: List[Tuple[float, float]],
        segments2: List[Tuple[float, float]],
        tolerance: float = 0.0,
        weight_by_length: bool = False
) -> Tuple[float, float]:
    """
    Erweiterte Berechnung von Overlap und Non-Overlap-Dauer zwischen zwei Sprechern.

    Args:
        segments1: List of (start, end) tuples for first speaker
        segments2: List of (start, end) tuples for second speaker
        tolerance: Kleine Überlappungen (in Sekunden), die ignoriert werden (z. B. 0.5).
        weight_by_length: Falls True, werden Overlaps bei längeren Segmenten stärker bestraft.

    Returns:
        Tuple of (total_overlap_duration, total_non_overlap_duration)
    """
    total_overlap = 0.0
    total_non_overlap = 0.0

    # Gesamtdauer beider Sprecher
    total_duration1 = sum(end - start for start, end in segments1)
    total_duration2 = sum(end - start for start, end in segments2)

    # Calculate overlaps
    for start1, end1 in segments1:
        for start2, end2 in segments2:
            # Calculate overlap
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            overlap = max(0.0, overlap_end - overlap_start)

            # Toleranzschwelle
            if overlap <= tolerance:
                continue

            if weight_by_length:
                # längere Segmente werden stärker gewichtet
                # d.h. Overlaps werden stärker bestraft, wenn ein Sprecher lange spricht
                seg_len = max(end1 - start1, end2 - start2)
                total_overlap += overlap * np.log1p(seg_len)
            else:
                total_overlap += overlap

    # Non-Overlap = Restdauer
    total_non_overlap = total_duration1 + total_duration2 - 2 * total_overlap

    return total_overlap, total_non_overlap

def calculate_conversation_scores(
    speaker_segments: Dict[str, List[Tuple[float, float]]],
    non_linear: str = None,
    tolerance: float = 0.0,
    weight_by_length: bool = False
) -> np.ndarray:
    """
    Calculate extended conversation likelihood scores between all pairs of speakers.

    Args:
        speaker_segments: Dict {speaker_id -> [(start, end), ...]}
        non_linear: transformation of Overlap-Ratio ('sigmoid', 'log', None)
        tolerance: ignore small overlaps (i.e. overlap < tolerance in sec)
        weight_by_length: longer speech segments get higher weight

    Returns:
        NxN numpy array of conversation scores (values between 0 and 1: 1 = same conversation, 0 = different conversation)
    """
    n_speakers = len(speaker_segments)
    scores = np.zeros((n_speakers, n_speakers))
    speaker_ids = list(speaker_segments.keys())

    for i in range(n_speakers):
        for j in range(i + 1, n_speakers):
            spk1 = speaker_ids[i]
            spk2 = speaker_ids[j]

            overlap, non_overlap = calculate_overlap_duration(
                speaker_segments[spk1],
                speaker_segments[spk2],
                tolerance=tolerance,
                weight_by_length=weight_by_length
            )

            # Calculate conversation likelihood score
            # Higher score when there's less overlap (more likely to be in same conversation)
            total = overlap + non_overlap
            if total > 0:
                overlap_ratio = overlap / total
                score = 1 - overlap_ratio
            else:
                score = 0.0

            # Nichtlineare Transformationen
            if non_linear == "sigmoid":
                score = 1 / (1 + np.exp(-10 * (score - 0.5)))  # steilere Mitte
            elif non_linear == "log":
                score = np.clip(score, 0.0, 1.0)
                score = np.log1p(score) / np.log(2)  # log-skaliert [0..1]

            scores[i, j] = score
            scores[j, i] = score   # Symmetric

    return scores
