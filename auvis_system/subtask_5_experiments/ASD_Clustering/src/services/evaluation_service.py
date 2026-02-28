import os
import json
import glob
import logging
from typing import Dict, List, Any, Optional

from auvis_system.subtask_5_experiments.ASD_Clustering.src.models.clustering_experiment import ClusteringExperimentModel
from auvis_system.subtask_5_experiments.ASD_Clustering.src.models.evaluation_models import SessionEvaluationResult, ExperimentEvaluationSummary
from auvis_system.subtask_5_experiments.ASD_Clustering.src.utils.clustering_metrics import (
    calculate_pairwise_metrics,
    calculate_ari,
    calculate_per_speaker_f1
)


class EvaluationService:
    """
    Service to compare Clustering Predictions against Ground Truth.

    It matches sessions by ID, loads the ground truth from the input data directory,
    aligns the speaker lists, and computes the metrics defined in evaluation_models.
    """

    def __init__(self, project_root: str):
        """
        Initialize the service.

        Args:
            project_root (str): Path to the root of the project (to locate data-bin).
        """
        self.project_root = project_root

    def evaluate_experiment(self,
                            experiment: ClusteringExperimentModel,
                            dataset_name: str,
                            logger: logging.Logger) -> ExperimentEvaluationSummary:
        """
        Evaluates a single clustering experiment (which contains multiple sessions).
        Iterates over all output files of the experiment, compares them to ground truth,
        and logs the progress using the provided experiment-specific logger.

        Args:
            experiment: The clustering experiment model containing output paths.
            dataset_name: Name of the dataset (e.g., 'dev' or 'train') to find GT.
            logger: The experiment-specific logger instance to write logs to.

        Returns:
            ExperimentEvaluationSummary: Aggregated results for this experiment.
        """
        # Create the summary container.
        # We access the parent segmentation ID via the object reference.
        summary = ExperimentEvaluationSummary(
            clustering_hash=experiment.clust_exp_id,
            parent_seg_hash=experiment.parent_seg_exp.seg_exp_id
        )

        # 1. Find all prediction files in the experiment output folder
        # Structure: .../<CLUST_HASH>/session_123_clustering.json
        search_pattern = os.path.join(experiment.output_dir, "*_clustering.json")
        pred_files = glob.glob(search_pattern)

        if not pred_files:
            logger.warning(f"Evaluation: No prediction files found in {experiment.output_dir}")
            return summary

        # 2. Iterate through predictions and compare with GT
        for pred_file in pred_files:
            try:
                # Load Prediction JSON
                with open(pred_file, 'r') as f:
                    pred_json = json.load(f)

                session_id = pred_json.get("session_id")

                # Extract predicted labels: {"spk_0": 0, "spk_1": 1}
                pred_labels_map = pred_json.get("clustering", {})

                # Extract max_dist (if available from the clustering step)
                max_dist = pred_json.get("max_merge_distance", 0.0)

                # Basic validation
                if not session_id or not pred_labels_map:
                    # We skip silent/empty files or log debug info if necessary
                    continue

                # 3. Load Ground Truth for this session
                gt_labels_map = self._load_ground_truth(dataset_name, session_id)

                if not gt_labels_map:
                    logger.warning(f"Missing Ground Truth for {session_id}. Skipping evaluation.")
                    continue

                # 4. Compute Metrics
                result = self._compute_session_metrics(
                    session_id,
                    pred_labels_map,
                    gt_labels_map,
                    max_dist
                )

                # 5. Add to Summary
                summary.add_result(result)

                # LOGGING: Log progress for this specific session
                logger.info(f"Evaluated {session_id}: F1={result.f1_score:.4f}, ARI={result.ari_score:.4f}")

            except Exception as e:
                logger.error(f"Error evaluating {os.path.basename(pred_file)}: {e}")

        return summary

    def _load_ground_truth(self, dataset: str, session_id: str) -> Optional[Dict[str, int]]:
        """
        Constructs the file path to the Ground Truth labels and loads the JSON.
        Path convention: data-bin/<dataset>/<session_id>/labels/speaker_to_cluster.json

        Args:
            dataset: The dataset name (e.g., 'dev').
            session_id: The session identifier.

        Returns:
            Dict mapping speaker IDs to cluster IDs, or None if file is missing.
        """
        gt_path = os.path.join(
            self.project_root,
            "data-bin",
            dataset,
            session_id,
            "labels",
            "speaker_to_cluster.json"
        )

        if not os.path.exists(gt_path):
            return None

        try:
            with open(gt_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def _compute_session_metrics(self,
                                 session_id: str,
                                 pred_map: Dict[str, int],
                                 gt_map: Dict[str, int],
                                 max_dist: float) -> SessionEvaluationResult:
        """
        Aligns speaker IDs between prediction and ground truth, then calculates
        all required metrics (F1, Precision, Recall, ARI).

        Args:
            session_id: The session identifier.
            pred_map: Dictionary of predicted labels.
            gt_map: Dictionary of ground truth labels.
            max_dist: The uncertainty metric from the clustering step.

        Returns:
            SessionEvaluationResult: A data object containing all calculated metrics.
        """
        # 1. Align Speakers (Intersection)
        # We only evaluate speakers that exist in BOTH prediction and GT.
        # Ideally, these sets are identical.
        common_speakers = sorted(list(set(pred_map.keys()) & set(gt_map.keys())))

        if not common_speakers:
            # Return zeroed result if no overlap found
            return SessionEvaluationResult(session_id, 0.0, 0.0, 0.0, 0.0, 0.0, {})

        # 2. Create parallel lists of integers for the metric functions
        # e.g. [0, 0, 1] vs [0, 1, 1] corresponding to spk_0, spk_1, spk_2
        gt_vec = [gt_map[spk] for spk in common_speakers]
        pred_vec = [pred_map[spk] for spk in common_speakers]

        # 3. Calculate Global Session Metrics
        # Note: calculate_pairwise_metrics returns (precision, recall, f1)
        precision, recall, f1 = calculate_pairwise_metrics(gt_vec, pred_vec)
        ari = calculate_ari(gt_vec, pred_vec)

        # 4. Calculate Per-Speaker F1 (mapped back to speaker IDs)
        # Result is Dict[int_index, score] -> We need Dict[str_id, score]
        per_spk_scores_idx = calculate_per_speaker_f1(gt_vec, pred_vec)

        per_spk_scores_str = {}
        for idx, score in per_spk_scores_idx.items():
            spk_id = common_speakers[idx]
            per_spk_scores_str[spk_id] = round(score, 4)

        return SessionEvaluationResult(
            session_id=session_id,
            f1_score=f1,
            precision=precision,
            recall=recall,
            ari_score=ari,
            max_merge_distance=max_dist,
            per_speaker_f1=per_spk_scores_str
        )
