"""
Module: build_active_speaker_segments
=====================================

Main entry point for the Active Speaker Segmentation pipeline (Modal: AV_ASD).

Workflow:
1. Setup: Generate experiment configurations based on a parameter grid.
2. Discovery: Load existing experiments from disk.
3. Data Loading: Load session data (Input), applying preprocessing/caching via the Loader Service.
4. Execution: Run the segmentation algorithm (Future Step).

Usage:
    Run this script directly to execute the pipeline.
"""

import os
import time
import pandas as pd
import itertools
from team_c.src.utils.logging_utils import setup_logger_context
from team_c.src.services.asd_scores_preprocess_service import load_sessions
from team_c.src.services.segmentation_workflow_service import (
    generate_configs,
    load_existing_experiments,
    update_experiment_registry,
    run_segmentation_pipeline
)
from team_c.src.services.clustering_workflow_service import (
    run_matrix_pipeline,
    run_clustering_pipeline,
    discover_clustering_experiments
)
from team_c.src.services.evaluation_workflow_service import (
    run_evaluation_pipeline,
    generate_global_overview_csv
)

# --- CONTROL ROOM: PIPELINE STEERING ---
# Set these flags to True/False to control which parts of the pipeline should run.
# NOTE: Experiment loading happens automatically to ensure subsequent steps have access to previous data.
DO_SETUP        = True   # Generate configs & update registry
DO_SEGMENTATION = True  # Run the expensive segmentation algorithm (Neural Network/Signal Processing)
DO_MATRIX       = True  # Calculate distance matrices (required for clustering)
DO_CLUSTERING   = True   # Run the agglomerative clustering
DO_EVALUATION   = True   # Run evaluation & generate CSV overview


def preview_grid_search(seg_grid, clust_grid):
    """
    Helper function to visualize the complexity of the grid search before execution.
    Prints the total number of combinations to the console.
    """
    print("\n--- GRID SEARCH PREVIEW ---")

    # 1. Expand Segmentation Grid
    seg_keys = list(seg_grid.keys())
    seg_values = list(seg_grid.values())
    seg_combinations = list(itertools.product(*seg_values))

    # 2. Expand Clustering Grid
    clust_keys = list(clust_grid.keys())
    clust_values = list(clust_grid.values())
    clust_combinations = list(itertools.product(*clust_values))

    total_experiments = len(seg_combinations) * len(clust_combinations)

    print(f"Segmentation configurations: {len(seg_combinations)}")
    print(f"Clustering configurations (per seg): {len(clust_combinations)}")
    print(f"TOTAL PLANNED EXPERIMENTS: {total_experiments}")
    print("-" * 30 + "\n")


# --- MAIN ---
def main():
    # dev for using dev session data & train for using train session data
    datasets = ["dev"]  # dev | train

    # Define the parameter grid for Segmentation
    seg_param_grid = {
        # --- ASD THRESHOLDS ---
        'onset': [1.0, 1.2, 1.7],  # default: 1.0
        'offset': [0.8, 1.0, 1.7, 2.25],  # default: 0.8

        # --- SMOOTHING ---
        'min_duration_on': [0.94, 1.0, 1.5],  # default: 1.0
        'min_duration_off': [0.26, 0.5, 1.0],  # default: 0.5

        # --- CHUNKING ---
        'max_chunk_size': [10, 14.12],  # default: 10
        'min_chunk_size': [1.0, 1.28]  # default: 1
    }

    # Define the parameter grid for Clustering
    clust_param_grid = {
        'threshold': [0.7, 0.74, 0.76, 0.78],
        'linkage': ["complete"]  # "complete", "average", "single"
    }

    # Show a preview of the grid search complexity
    preview_grid_search(seg_param_grid, clust_param_grid)

    # Setup global logger using the new utility
    global_ctx = setup_logger_context(
        name="Global",
        log_file_path="pipeline_execution.log",
        console=True
    )

    with global_ctx as main_logger:
        main_logger.info(">>> Starting Pipeline Control <<<")

        # ---------------------------------------------------------
        # 1. SETUP & LOADING
        # ---------------------------------------------------------
        experiments = []

        if DO_SETUP:
            main_logger.info("--- [1] Setup & Config Generation ---")
            generate_configs(datasets, seg_param_grid, main_logger)
            update_experiment_registry(main_logger)

        # We ALWAYS load experiments so that later steps (like Clustering) know what exists on disk,
        # even if we skip the Segmentation step in this run.
        experiments = load_existing_experiments(main_logger)

        if experiments:
            main_logger.info(f"Registry loaded. Ready to process {len(experiments)} base experiments.")
        else:
            main_logger.warning("No experiments found in registry.")

        # ---------------------------------------------------------
        # 2. SEGMENTATION PIPELINE
        # ---------------------------------------------------------
        if DO_SEGMENTATION and experiments:
            main_logger.info("--- [2] Running Segmentation Pipeline ---")
            t_start = time.time()

            # Using the renamed Session Loader (handles caching, merging, UEM cutting)
            target_dataset = datasets[0]
            sessions = load_sessions(target_dataset, main_logger)

            # Execution (Algorithm)
            if sessions:
                run_segmentation_pipeline(experiments, sessions, main_logger)
            else:
                main_logger.warning("Skipping execution: No sessions loaded.")

            t_end = time.time()
            main_logger.info(f"DONE: Segmentation took {t_end - t_start:.2f} seconds.")
        else:
            main_logger.info("SKIPPING: Segmentation Pipeline (DO_SEGMENTATION=False)")

        # ---------------------------------------------------------
        # 3. DISTANCE MATRIX PIPELINE
        # ---------------------------------------------------------
        # build distance matrix
        if DO_MATRIX and experiments:
            main_logger.info("--- [3] Running Matrix Pipeline ---")
            t_start = time.time()

            run_matrix_pipeline(experiments, main_logger)

            t_end = time.time()
            main_logger.info(f"DONE: Matrix calculation took {t_end - t_start:.2f} seconds.")
        else:
            main_logger.info("SKIPPING: Matrix Pipeline (DO_MATRIX=False)")

        # ---------------------------------------------------------
        # 4. CLUSTERING PIPELINE
        # ---------------------------------------------------------
        # do agglomerative clustering
        if DO_CLUSTERING and experiments:
            main_logger.info("--- [4] Running Clustering Pipeline ---")
            t_start = time.time()

            run_clustering_pipeline(experiments, clust_param_grid, main_logger)

            t_end = time.time()
            main_logger.info(f"DONE: Clustering took {t_end - t_start:.2f} seconds.")
        else:
            main_logger.info("SKIPPING: Clustering Pipeline (DO_CLUSTERING=False)")

        # ---------------------------------------------------------
        # 5. EVALUATION PIPELINE
        # ---------------------------------------------------------
        # evaluation
        if DO_EVALUATION and experiments:
            main_logger.info("--- [5] Running Evaluation Pipeline ---")
            t_start = time.time()

            # target_dataset is e.g. "dev" (where the labels are located)
            target_dataset = datasets[0]
            clustering_experiments = discover_clustering_experiments(experiments, main_logger)

            if clustering_experiments:
                # 1. Run Evaluation
                run_evaluation_pipeline(clustering_experiments, target_dataset, main_logger)

                # 2. Generate Global Overview CSV
                # Determine root output directory (e.g., .../data-bin/_output/av_asd)
                # We use the first experiment as an anchor and navigate up two levels
                # (from .../<seg_hash>/<clust_hash> up to .../av_asd)
                first_exp = clustering_experiments[0]
                root_output_dir = os.path.dirname(os.path.dirname(first_exp.output_dir))

                generate_global_overview_csv(root_output_dir, main_logger)
            else:
                main_logger.warning("Skipping Evaluation: No clustering experiments found on disk.")

            t_end = time.time()
            main_logger.info(f"DONE: Evaluation took {t_end - t_start:.2f} seconds.")
        else:
            main_logger.info("SKIPPING: Evaluation Pipeline (DO_EVALUATION=False)")

        main_logger.info(">>> Pipeline Finished <<<")


if __name__ == "__main__":
    main()
