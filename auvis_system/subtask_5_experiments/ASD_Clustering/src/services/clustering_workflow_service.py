"""
Module: clustering_workflow_service
===================================
Orchestrates the workflow for the clustering stage of the pipeline.

This service acts as the bridge between the segmentation results and the clustering logic.
It currently handles the batch processing of interaction matrices calculation.

Functions:
    - run_matrix_pipeline: Iterates over experiments and triggers matrix calculation.
    - discover_clustering_experiments: Finds existing clustering runs.
    - run_clustering_pipeline: Runs the actual clustering (with history and max_dist).
"""

import os
import json
import logging
from glob import glob
from typing import List
from dataclasses import fields

# Import Logic Class
from ..services.matrix_calculator import MatrixCalculator
# Import Data Model for Type Hinting
from ..models.segmentation_experiment import SegmentationExperimentModel

from ..models.clustering_experiment import ClusteringExperimentModel, ClusteringConfig
from ..services.clustering_engine import ClusteringEngine


def run_matrix_pipeline(experiments: List[SegmentationExperimentModel], global_logger: logging.Logger) -> None:
    """
    Step 2 of the Pipeline: Calculates distance matrices for existing segmentation results.
    """
    global_logger.info("--- Service: Running Matrix Calculation Pipeline ---")

    # Instantiate the logic engine (stateless)
    calculator = MatrixCalculator()

    for exp_model in experiments:
        exp_id = exp_model.seg_exp_id

        # We now look specifically into the 'segments' subdirectory
        segments_source_dir = exp_model.segments_dir
        # And we target the 'matrices' subdirectory for output
        matrices_target_dir = exp_model.matrices_dir

        global_logger.info(f"Processing Matrices for Experiment: {exp_id}")

        # Open the experiment-specific logger context
        with exp_model.experiment_logger() as exp_logger:

            # Search for segment files using glob in the segments subdirectory
            search_pattern = os.path.join(segments_source_dir, "*_segments.json")
            segment_files = glob(search_pattern)

            if not segment_files:
                msg = f"No segment files found in {segments_source_dir}. Skipping Matrix Step."
                global_logger.warning(msg)
                exp_logger.warning(msg)
                continue

            exp_logger.info(f"=== Starting Matrix Calculation for {len(segment_files)} Sessions ===")

            count_new = 0

            for seg_file in segment_files:
                # Extract clean session ID to build the new path
                session_filename = os.path.basename(seg_file)
                session_id = session_filename.replace("_segments.json", "")

                # Construct output filename: session_123_matrix.json
                output_filename = f"{session_id}_matrix.json"
                # Construct full path in the new 'matrices' directory
                matrix_file_path = os.path.join(matrices_target_dir, output_filename)

                try:
                    # 1. Load Segments
                    with open(seg_file, 'r') as f:
                        data = json.load(f)

                    loaded_session_id = data.get("session_id", session_id)
                    speakers_data = data.get("speakers", {})

                    # 2. Run Calculation Logic
                    result_payload = calculator.calculate_session_matrix(speakers_data)

                    # 3. Add Metadata for traceability
                    result_payload["session_id"] = loaded_session_id
                    result_payload["hash"] = exp_id
                    result_payload["source_segments_file"] = session_filename

                    # 4. Save Matrix Result into the matrices directory
                    with open(matrix_file_path, 'w') as f:
                        json.dump(result_payload, f, indent=4)

                    exp_logger.info(f"Matrix calculated: {loaded_session_id}")
                    count_new += 1

                except Exception as e:
                    error_msg = f"Error calculating matrix for {session_filename}: {e}"
                    global_logger.error(f"[Exp {exp_id}] {error_msg}")
                    exp_logger.error(error_msg)

            # Summary for this experiment
            summary = f"Matrix Step Finished. Processed: {count_new}"
            exp_logger.info(summary)
            global_logger.info(f"-> {exp_id}: {summary}")

    global_logger.info("--> Matrix Pipeline execution finished.")


def discover_clustering_experiments(parents: List[SegmentationExperimentModel],
                                    global_logger) -> List[ClusteringExperimentModel]:
    """
    Scans the output directories of the parent segmentation experiments to find
    existing clustering results on disk.
    """
    discovered_experiments = []

    # Dynamically retrieve allowed fields from the Config class
    allowed_fields = {f.name for f in fields(ClusteringConfig)}

    for parent in parents:
        parent_dir = parent.output_dir

        try:
            if not os.path.exists(parent_dir):
                continue
            # Scan for directories only
            subdirs = [f.path for f in os.scandir(parent_dir) if f.is_dir()]
        except Exception:
            continue

        for subdir in subdirs:
            # We identify a valid clustering experiment by the existence of its config file
            config_path = os.path.join(subdir, "clust_config.json")

            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        raw_data = json.load(f)

                    clean_config_data = {
                        k: v for k, v in raw_data.items()
                        if k in allowed_fields
                    }

                    config = ClusteringConfig(**clean_config_data)
                    model = ClusteringExperimentModel(config, parent)
                    discovered_experiments.append(model)

                except Exception as e:
                    global_logger.warning(f"Found clustering folder {os.path.basename(subdir)} but failed to load: {e}")

    global_logger.info(f"Discovered {len(discovered_experiments)} existing clustering experiments on disk.")
    return discovered_experiments


def run_clustering_pipeline(experiments, param_grid_clustering, global_logger):
    """
    Step 3: Runs Clustering on the calculated matrices.
    Creates sub-folders for each clustering configuration.

    UPDATED: Now handles (labels, max_dist, history) from engine.
    """
    global_logger.info("--- Service: Running Clustering Pipeline ---")

    # 1. Generate Clustering Configs from Grid
    cluster_configs = []
    for th in param_grid_clustering['threshold']:
        for li in param_grid_clustering['linkage']:
            cluster_configs.append(ClusteringConfig(threshold=th, linkage=li))

    # 2. Iterate over PARENT experiments (Segmentation)
    for seg_exp in experiments:
        seg_id = seg_exp.seg_exp_id

        # Use the property 'matrices_dir' to locate the correct source folder
        matrix_source_dir = seg_exp.matrices_dir

        search_pattern = os.path.join(matrix_source_dir, "*_matrix.json")
        matrix_files = glob(search_pattern)

        if not matrix_files:
            global_logger.warning(f"Skipping {seg_id}: No matrices found in {matrix_source_dir}.")
            continue

        # 3. Iterate over CHILD experiments (Clustering variations)
        for clust_conf in cluster_configs:
            # Init Child Model
            clust_exp = ClusteringExperimentModel(clust_conf, seg_exp)
            clust_exp.create_structure()

            # Init Engine
            engine = ClusteringEngine(
                threshold=clust_conf.threshold,
                linkage=clust_conf.linkage
            )

            with clust_exp.experiment_logger() as logger:
                logger.info(f"--- Clustering: Th={clust_conf.threshold}, Link={clust_conf.linkage} ---")

                # Container for history logs for this experiment
                all_histories = {}
                count = 0

                for mat_file in matrix_files:
                    try:
                        # Load Matrix
                        with open(mat_file, 'r') as f:
                            data = json.load(f)

                        session_id = data.get('session_id')
                        # Note: Check if your matrix calculator saves it as 'distance_matrix' or just 'matrix'
                        # Based on previous code, it seems to be 'distance_matrix' or 'matrix'
                        # We try 'distance_matrix' first (as per your snippet), then fallback if needed
                        matrix = data.get('distance_matrix')
                        if matrix is None:
                             matrix = data.get('matrix')

                        spk_ids = data.get('speaker_ids')

                        if session_id is None or matrix is None or spk_ids is None:
                            logger.error(f"Invalid data structure in {os.path.basename(mat_file)}")
                            continue

                        # --- NEW: Catch all 3 return values ---
                        clustering_map, max_dist, history = engine.run_clustering(matrix, spk_ids)

                        # Store history
                        all_histories[session_id] = history

                        # Prepare Result Payload (Including max_merge_distance)
                        output_payload = {
                            "session_id": session_id,
                            "parent_hash": seg_id,
                            "cluster_hash": clust_exp.clust_exp_id,
                            "max_merge_distance": max_dist,  # <--- NEW
                            "clustering": clustering_map
                        }

                        # Define output path
                        out_name = f"{session_id}_clustering.json"
                        out_path = os.path.join(clust_exp.output_dir, out_name)

                        # Save Result
                        with open(out_path, 'w') as f_out:
                            json.dump(output_payload, f_out, indent=4)

                        logger.info(f"Clustered session: {session_id}")
                        count += 1

                    except Exception as e:
                        fname = os.path.basename(mat_file)
                        logger.error(f"Failed session {fname}: {e}")

                # --- NEW: Save Global History Log ---
                hist_path = os.path.join(clust_exp.output_dir, "clustering_history.json")
                with open(hist_path, 'w') as f:
                    json.dump(all_histories, f, indent=4)

                logger.info(f"Clustered {count} sessions. History saved.")

    global_logger.info("--> Clustering Pipeline finished.")
