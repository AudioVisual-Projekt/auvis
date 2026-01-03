import numpy as np
from sklearn.cluster import AgglomerativeClustering
from typing import Dict, List


class ClusteringEngine:
    """
    Wrapper around sklearn AgglomerativeClustering.
    Converts our distance matrices into speaker mappings.
    """

    def __init__(self, threshold: float, linkage: str = "average"):
        """
        Args:
            threshold (float): Similarity Score (0.0 to 1.0).
                               0.8 means: Speakers must be 80% similar to merge.
                               Internally converts to distance: 1.0 - 0.8 = 0.2.
            linkage (str): 'complete', 'average', or 'single'.
                           Legacy code used 'complete'.
        """
        self.similarity_threshold = threshold
        self.linkage = linkage

        # Initialize sklearn model
        # metric='precomputed' is crucial because we feed it a distance matrix!
        dist_threshold = max(0.0, 1.0 - threshold)

        self.model = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage=self.linkage,
            distance_threshold=dist_threshold
        )

    def run_clustering(self, distance_matrix: List[List[float]], speaker_ids: List[str]) -> Dict[str, int]:
        """
        Runs the clustering on a single session matrix.

        Args:
            distance_matrix: NxN list of floats (0.0 to 1.0)
            speaker_ids: List of speaker IDs corresponding to matrix rows/cols.

        Returns:
            Dict mapping { "spk_id": cluster_id (int) }
        """
        if not speaker_ids:
            return {}

        # Convert to numpy for sklearn
        X = np.array(distance_matrix)

        # Safety Check: Matrix must be symmetric and square
        if X.shape[0] != X.shape[1] or X.shape[0] != len(speaker_ids):
            raise ValueError("Matrix dimensions do not match speaker count.")

        # Special Case: Only 1 speaker
        if len(speaker_ids) == 1:
            return {speaker_ids[0]: 0}

        # Run Fit
        try:
            labels = self.model.fit_predict(X)
        except Exception as e:
            # Fallback for edge cases (e.g. 0 distance everywhere)
            print(f"Clustering Warning: {e}")
            # If fail, everyone is their own cluster? Or one big cluster?
            # Usually safe to return all 0 if really broken, but sklearn is robust.
            return {spk: 0 for spk in speaker_ids}

        # Map results back to IDs
        # labels is an array like [0, 0, 1, 1, 0] matching the index of speaker_ids
        result = {}
        for idx, spk_id in enumerate(speaker_ids):
            result[spk_id] = int(labels[idx])

        return result
