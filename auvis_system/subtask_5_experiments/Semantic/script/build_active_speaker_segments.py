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
    discover_clustering_experiments)
from team_c.src.services.evaluation_workflow_service import run_evaluation_pipeline


# --- MAIN ---
def main():
    # dev for using dev session data & train for using train session data
    datasets = ["dev"]  # dev | train

    seg_param_grid = {
        # --- ASD THRESHOLDS ---
        'onset': [1.0, 0.8],
        'offset': [0.8],

        # --- SMOOTHING ---
        'min_duration_on': [1.0],
        'min_duration_off': [0.5, 1.0],

        # --- CHUNKING ---
        'max_chunk_size': [10, 15],
        'min_chunk_size': [1]
    }

    clust_param_grid = {
        'threshold': [0.5, 0.7],
        'linkage': ["complete"]  # "complete", "average", "single"
    }

    # Setup global logger using the new utility
    global_ctx = setup_logger_context(
        name="Global",
        log_file_path="pipeline_execution.log",
        console=True
    )

    with global_ctx as main_logger:
        main_logger.info("Starting Pipeline...")

        # --- SETUP ---
        generate_configs(datasets, seg_param_grid, main_logger)

        update_experiment_registry(main_logger)

        # --- LOADING EXPERIMENTS ---
        experiments = load_existing_experiments(main_logger)

        if experiments:
            main_logger.info(f"Ready to process {len(experiments)} experiments.")

        # --- LOADING DATA ---
        # Using the renamed Session Loader (handles caching, merging, UEM cutting)
        target_dataset = datasets[0]
        sessions = load_sessions(target_dataset, main_logger)

        # --- EXECUTION (Algorithm) ---
        if experiments and sessions:
            run_segmentation_pipeline(experiments, sessions, main_logger)
        else:
            main_logger.warning("Skipping execution: No experiments or sessions loaded.")

        # build distance matrix
        run_matrix_pipeline(experiments, main_logger)

        # do agglomerative clustering
        run_clustering_pipeline(experiments, clust_param_grid, main_logger)

        # evaluation
        # target_dataset ist z.B. "dev" (wo die Labels liegen)
        clustering_experiments = discover_clustering_experiments(experiments, main_logger)
        if clustering_experiments:
            run_evaluation_pipeline(clustering_experiments, target_dataset, main_logger)
        else:
            main_logger.warning("Skipping Evaluation: No clustering experiments found on disk.")

        main_logger.info("Pipeline Finished.")


if __name__ == "__main__":
    main()
