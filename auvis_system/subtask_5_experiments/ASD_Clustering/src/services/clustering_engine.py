import numpy as np
from sklearn.cluster import AgglomerativeClustering
from typing import Dict, List, Tuple, Any


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
        """
        self.similarity_threshold = threshold
        self.linkage = linkage

        # Initialize sklearn model
        # metric='precomputed' is crucial because we feed it a distance matrix!
        # compute_distances=True is crucial to extract the merge history log!
        self.dist_threshold = max(0.0, 1.0 - threshold)

        self.model = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage=self.linkage,
            distance_threshold=self.dist_threshold,
            compute_distances=True
        )

    def run_clustering(self,
                       distance_matrix: List[List[float]],
                       speaker_ids: List[str]) -> Tuple[Dict[str, int], float, List[Dict[str, Any]]]:
        """
        Runs the clustering on a single session matrix.

        Args:
            distance_matrix: NxN list of floats (0.0 to 1.0)
            speaker_ids: List of speaker IDs corresponding to matrix rows/cols.

        Returns:
            Tuple containing:
            1. Dict mapping { "spk_id": cluster_label (int) }
            2. float: The maximum distance used for an actual merge (confidence metric).
            3. List[Dict]: The readable merge history log.
        """
        if not speaker_ids:
            return {}, 0.0, []

        # Convert to numpy for sklearn
        X = np.array(distance_matrix)

        # Safety Check: Matrix must be symmetric and square
        if X.shape[0] != X.shape[1] or X.shape[0] != len(speaker_ids):
            raise ValueError("Matrix dimensions do not match speaker count.")

        # Special Case: Only 1 speaker
        if len(speaker_ids) == 1:
            return {speaker_ids[0]: 0}, 0.0, []

        # Run Fit
        try:
            self.model.fit(X)
            labels = self.model.labels_
        except Exception as e:
            # Fallback for edge cases (e.g. 0 distance everywhere)
            print(f"Clustering Warning: {e}")
            return {spk: 0 for spk in speaker_ids}, 0.0, []

        # 1. Map results back to IDs
        result = {}
        for idx, spk_id in enumerate(speaker_ids):
            result[spk_id] = int(labels[idx])

        # 2. Extract History and Max Merge Distance
        history, max_dist = self._generate_merge_history(speaker_ids)

        return result, max_dist, history

    def _generate_merge_history(self, speaker_ids: List[str]) -> Tuple[List[Dict[str, Any]], float]:
        """
        Translates scikit-learn's internal dendrogram structure into a readable JSON log.
        Also determines the maximum distance that was actually used for a merge.
        """
        history = []
        max_dist_used = 0.0

        # Scikit-Learn stores the hierarchy using indices:
        # Indices 0 to n_samples-1 are the original samples (leaves).
        # Indices >= n_samples are the newly created clusters.
        n_samples = len(speaker_ids)

        # Map index -> Name (initially just the speaker IDs)
        node_names = {i: spk for i, spk in enumerate(speaker_ids)}

        if not hasattr(self.model, 'children_'):
            return [], 0.0

        # Iterate over the children_ array (each row is a merge step)
        for i, (child_a, child_b) in enumerate(self.model.children_):
            # model.distances_ is available because we set compute_distances=True
            dist = self.model.distances_[i]

            # Since 'distance_threshold' is used, the model might compute the full tree
            # internally. We only care about merges that satisfy the threshold.
            if dist > self.dist_threshold:
                break

            max_dist_used = max(max_dist_used, dist)

            # Resolve names
            name_a = node_names.get(child_a, f"cluster_idx_{child_a}")
            name_b = node_names.get(child_b, f"cluster_idx_{child_b}")

            # Assign a name to the new cluster (index = n_samples + i)
            new_node_idx = n_samples + i
            new_node_name = f"cluster_step_{i + 1}"
            node_names[new_node_idx] = new_node_name

            step_entry = {
                "step": i + 1,
                "merge": [name_a, name_b],
                "distance": round(float(dist), 5),
                "resulting_cluster": new_node_name
            }
            history.append(step_entry)

        return history, max_dist_used
