import numpy as np
from typing import List, Dict, Tuple


class MatrixCalculator:
    """
    Stateless service class that handles the calculation of interaction matrices
    based on speaker segments (Overlap-Model).
    """

    def calculate_session_matrix(self, speakers_data: Dict) -> Dict:
        """
        Main entry point: Converts speaker segments into a full distance matrix structure.

        Args:
            speakers_data (Dict): The 'speakers' dictionary from the segments.json
                                  Structure: { "spk_id": { "segments": [[s,e], ...] } }

        Returns:
            Dict: Containing 'distance_matrix' (List[List[float]]), 'durations', and 'speaker_ids'.
        """
        # 1. Sort speaker IDs to ensure consistent matrix indices
        speaker_ids = sorted(list(speakers_data.keys()))
        n_speakers = len(speaker_ids)

        matrix = np.zeros((n_speakers, n_speakers))
        durations = {}

        # 2. Iterate Pairwise
        for i in range(n_speakers):
            id_i = speaker_ids[i]
            segs_i = speakers_data[id_i]["segments"]

            # Save duration stats
            durations[id_i] = self._calculate_total_duration(segs_i)

            for j in range(i + 1, n_speakers):
                id_j = speaker_ids[j]
                segs_j = speakers_data[id_j]["segments"]

                # Calculate Score
                score = self._compute_pairwise_score(segs_i, segs_j)

                # Convert to Distance (1 - Score)
                # High Overlap => High Score => Low Distance
                distance = round(1.0 - score, 5)

                matrix[i, j] = distance
                matrix[j, i] = distance  # Symmetric

        return {
            "speaker_ids": speaker_ids,
            "durations_seconds": durations,
            "distance_matrix": matrix.tolist()
        }

    def _calculate_total_duration(self, segments: List[List[float]]) -> float:
        """Helper: Sums up total speaking time."""
        return round(sum(end - start for start, end in segments), 5)

    def _compute_pairwise_score(self, segs1: List[List[float]], segs2: List[List[float]]) -> float:
        """
        Core Logic: Calculates the conversation score based on overlap.
        Returns score between 0.0 and 1.0.
        """
        dur1 = self._calculate_total_duration(segs1)
        dur2 = self._calculate_total_duration(segs2)

        total_overlap = 0.0

        # Calculate Overlap
        for start1, end1 in segs1:
            for start2, end2 in segs2:
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                if overlap_end > overlap_start:
                    total_overlap += overlap_end - overlap_start

        # Calculate Non-Overlap (Time where strictly one or both speak, but NOT both at once?)
        # NOTE: Using the formula from your old code:
        # non_overlap = dur1 + dur2 - 2 * overlap
        # Logic: (A_total - overlap) + (B_total - overlap) = parts where they speak alone.
        total_non_overlap = dur1 + dur2 - (2 * total_overlap)

        if total_overlap + total_non_overlap > 0:
            total_active_duration = total_overlap + total_non_overlap
            overlap_ratio = total_overlap / total_active_duration
            return 1.0 - overlap_ratio

        return 0.0
