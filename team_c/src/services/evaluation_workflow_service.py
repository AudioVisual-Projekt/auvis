"""
Module: evaluation_workflow_service
===================================
Orchestrates the evaluation process for the clustering pipeline.

This service:
1. Iterates over clustering experiments.
2. Uses EvaluationService to compare predictions vs. Ground Truth.
3. Aggregates results (F1, Precision, Recall, ARI).
4. Saves detailed CSV reports for each experiment.
5. Saves a comprehensive JSON summary for each experiment (Experiment Level).
6. Generates a Global Overview CSV comparing all experiments (Competition View).
"""

import os
import csv
import json
import glob
import logging
from datetime import datetime
from typing import List, Dict, Any

from team_c.src.models.clustering_experiment import ClusteringExperimentModel
from team_c.src.services.evaluation_service import EvaluationService
from team_c.src.models.evaluation_models import ExperimentEvaluationSummary, SessionEvaluationResult


def run_evaluation_pipeline(experiments: List[ClusteringExperimentModel],
                            dataset: str,
                            global_logger: logging.Logger) -> None:
    """
    Main entry point for the evaluation stage.

    Args:
        experiments: List of clustering experiment models to evaluate.
        dataset: The dataset name (e.g., 'dev') to locate Ground Truth labels.
        global_logger: Main logger for high-level output.
    """
    global_logger.info("--- Service: Running Evaluation Pipeline ---")

    # 1. Setup Service
    # We need the project root to find the data-bin folder for Ground Truth
    project_root = _get_project_root()
    eval_service = EvaluationService(project_root)

    for exp_model in experiments:
        clust_id = exp_model.clust_exp_id

        # get segmentation id (hash)
        seg_id = exp_model.parent_seg_exp.seg_exp_id

        # Check if we actually have results to evaluate
        if not os.path.exists(exp_model.output_dir):
            global_logger.warning(f"Skipping evaluation for {clust_id}: Output directory missing.")
            continue

        with exp_model.experiment_logger() as logger:
            logger.info(f"Starting Evaluation against Ground Truth ({dataset})...")

            # 2. Run Evaluation Logic (In-Memory)
            # This returns a structured summary object containing all metrics
            summary: ExperimentEvaluationSummary = eval_service.evaluate_experiment(
                experiment=exp_model,
                dataset_name=dataset,
                logger=logger
            )

            if not summary.session_results:
                logger.warning("No sessions evaluated (maybe missing GT or Pred files?).")
                continue

            # 3. Calculate Global Averages
            # Metric A: Conversational Pairwise F1 (Average of Session-Averages)
            avg_f1 = summary.get_average_f1()
            avg_ari = summary.get_average_ari()

            # Metric B: Per-Speaker F1 (Average across ALL speakers in ALL sessions)
            # This corresponds to the CHiME Metric logic.
            all_speakers_scores = []
            for res in summary.session_results:
                # We collect every single speaker score from every session into one giant list
                if res.per_speaker_f1:
                    all_speakers_scores.extend(res.per_speaker_f1.values())

            # Compute average over ALL speakers
            avg_speaker_f1 = 0.0
            if all_speakers_scores:
                avg_speaker_f1 = sum(all_speakers_scores) / len(all_speakers_scores)

            # Count perfect sessions
            perfect_count = sum(1 for r in summary.session_results if r.is_perfect)
            total_count = len(summary.session_results)

            # Log Summary
            # We log both F1 scores now to distinguish between Pairwise and Speaker-based metrics
            stats_msg = (
                f"Evaluation Results -> "
                f"Avg Speaker F1 (CHiME): {avg_speaker_f1:.4f} | "
                f"Avg Pairwise F1: {avg_f1:.4f} | "
                f"Avg ARI: {avg_ari:.4f} | "
                f"Perfect Sessions: {perfect_count}/{total_count}"
            )
            logger.info(stats_msg)
            global_logger.info(f"seg_hash[{seg_id}] + clust_hash[{clust_id}] - {stats_msg}")

            # 4. Save Detailed CSV Report (Local to experiment)
            _save_session_csv(exp_model.output_dir, summary.session_results, logger)

            # 5. Save Comprehensive JSON Result (Source for Overview CSV)
            # This JSON includes parameters and the calculated avg_speaker_f1
            _save_experiment_result_json(
                exp_model,
                summary,
                dataset,
                avg_speaker_f1,
                logger
            )

    global_logger.info("--> Evaluation Pipeline finished.")


def generate_global_overview_csv(experiments_root: str, global_logger: logging.Logger) -> None:
    """
    Scans the entire output directory for 'experiment_result.json' files
    and compiles them into a single 'experiments_overview.csv'.

    This CSV allows sorting by 'AVG_SPEAKER_F1' to find the competition winner.

    Args:
        experiments_root: The root directory containing all experiment hash folders
                          (e.g., .../data-bin/_output/av_asd).
        global_logger: Logger for output.
    """
    global_logger.info("--- Aggregating Experiment Results into Overview CSV ---")

    # 1. Search recursively for all JSON results
    # Pattern: experiments_root/**/experiment_result.json
    search_pattern = os.path.join(experiments_root, "**", "experiment_result.json")
    json_files = glob.glob(search_pattern, recursive=True)

    if not json_files:
        global_logger.warning("No experiment_result.json files found. Nothing to aggregate.")
        return

    aggregated_data = []
    all_session_keys = set()
    all_param_keys = set()

    # 2. Load data and flatten
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)

            # Basic Info & Metrics
            row = {
                "dataset": data["meta"]["dataset"],
                "seg_hash": data["meta"]["seg_hash"],
                "clust_hash": data["meta"]["clust_hash"],
                "created": data["meta"]["creation_date"],

                # METRICS
                # AVG_SPEAKER_F1 is the CHiME primary clustering metric
                "AVG_SPEAKER_F1": data["metrics"].get("avg_speaker_f1", 0.0),
                "AVG_CONV_F1": data["metrics"].get("avg_conversational_f1", 0.0),
                "AVG_ARI": data["metrics"].get("avg_ari", 0.0),
                "perfect_sessions": data["metrics"]["perfect_sessions_count"]
            }

            # Parameters (flattened)
            # e.g., seg_onset, clust_threshold
            for p_type in ["segmentation", "clustering"]:
                for key, val in data["parameters"].get(p_type, {}).items():
                    col_name = f"{p_type[:4]}_{key}"  # e.g., seg_onset
                    row[col_name] = val
                    all_param_keys.add(col_name)

            # Heatmap Data (Sessions)
            # 1 = Perfect, 0 = Imperfect (for easy conditional formatting/sorting)
            for sess_id, is_perfect in data.get("session_heatmap", {}).items():
                row[sess_id] = 1 if is_perfect else 0
                all_session_keys.add(sess_id)

            aggregated_data.append(row)

        except Exception as e:
            global_logger.warning(f"Skipping corrupted JSON {jf}: {e}")

    if not aggregated_data:
        return

    # 3. Organize Columns
    sorted_sessions = sorted(list(all_session_keys))
    sorted_params = sorted(list(all_param_keys))

    # Order: IDs -> Critical Metric -> Other Metrics -> Params -> Heatmap
    fieldnames = [
        "dataset", "seg_hash", "clust_hash",
        "AVG_SPEAKER_F1", "AVG_CONV_F1", "AVG_ARI",
        "perfect_sessions", "created"
    ] + sorted_params + sorted_sessions

    # 4. Write CSV
    output_csv_path = os.path.join(experiments_root, "experiments_overview.csv")

    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in aggregated_data:
                writer.writerow(row)

        global_logger.info(f"Successfully wrote overview to: {output_csv_path}")
        global_logger.info(f"--> Sort this CSV by 'AVG_SPEAKER_F1' to find the best model!")

    except Exception as e:
        global_logger.error(f"Failed to write overview CSV: {e}")


def _save_session_csv(output_dir: str,
                      results: List[SessionEvaluationResult],
                      logger: logging.Logger) -> None:
    """
    Writes a CSV file containing metrics for every single session in this experiment.
    Useful for debugging specific failures.
    """
    filename = "evaluation_sessions.csv"
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, mode='w', newline='') as csvfile:
            # Define CSV Columns
            fieldnames = [
                'session_id',
                'f1_score',
                'precision',
                'recall',
                'ari_score',
                'max_merge_dist',
                'is_perfect'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Write rows
            for res in results:
                writer.writerow({
                    'session_id': res.session_id,
                    'f1_score': f"{res.f1_score:.4f}",
                    'precision': f"{res.precision:.4f}",
                    'recall': f"{res.recall:.4f}",
                    'ari_score': f"{res.ari_score:.4f}",
                    'max_merge_dist': f"{res.max_merge_distance:.4f}",
                    'is_perfect': res.is_perfect
                })

        logger.info(f"Saved detailed evaluation report to {filename}")

    except Exception as e:
        logger.error(f"Failed to write CSV report: {e}")


def _save_experiment_result_json(exp_model: ClusteringExperimentModel,
                                 summary: ExperimentEvaluationSummary,
                                 dataset: str,
                                 avg_speaker_f1: float,
                                 logger: logging.Logger) -> None:
    """
    Saves a JSON file containing all metadata, parameters, and aggregated metrics.
    Includes the 'avg_speaker_f1' (cpF1) which is critical for CHiME evaluation.
    This file serves as the single source of truth for the 'experiments_overview.csv'.
    """
    filename = "experiment_result.json"
    filepath = os.path.join(exp_model.output_dir, filename)

    try:
        # 1. Extract Parameters safely
        seg_params = exp_model.parent_seg_exp.seg_config.__dict__
        clust_params = exp_model.clust_config.__dict__

        # 2. Prepare Session Heatmap Data (Boolean status)
        session_status_map = {
            res.session_id: res.is_perfect
            for res in summary.session_results
        }

        # 3. Build the structure
        data = {
            "meta": {
                "creation_date": datetime.now().isoformat(),
                "dataset": dataset,
                "seg_hash": exp_model.parent_seg_exp.seg_exp_id,
                "clust_hash": exp_model.clust_exp_id
            },
            "parameters": {
                "segmentation": seg_params,
                "clustering": clust_params
            },
            "metrics": {
                "avg_conversational_f1": summary.get_average_f1(), # Pairwise (Metric A)
                "avg_speaker_f1": round(avg_speaker_f1, 4),        # CHiME (Metric B)
                "avg_ari": summary.get_average_ari(),
                "perfect_sessions_count": sum(1 for r in summary.session_results if r.is_perfect),
                "total_sessions": len(summary.session_results)
            },
            "session_heatmap": session_status_map
        }

        # 4. Save to JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

        logger.info(f"Saved experiment summary to {filename}")

    except Exception as e:
        logger.error(f"Failed to save JSON summary: {e}")


def _get_project_root() -> str:
    """
    Helper to find the project root directory.
    Assumes this file is in src/services/
    """
    current_file = os.path.abspath(__file__)
    # Up 3 levels: src/services/ -> src/ -> team_c/ -> ROOT
    return os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
