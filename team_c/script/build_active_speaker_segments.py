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
from team_c.src.services.experiment_service import generate_configs, load_existing_experiments


# --- MAIN ---
def main():
    # dev for using dev session data & train for using train session data
    datasets = ["dev"]  # dev | train

    param_grid = {
        # --- ASD THRESHOLDS ---
        'onset': [1.0, 0.8],
        'offset': [0.8],

        # --- SMOOTHING ---
        'min_duration_on': [1.0],
        'min_duration_off': [0.5],

        # --- CHUNKING ---
        'max_chunk_size': [10, 15],
        'min_chunk_size': [1]
    }

    # Setup global logger using the new utility
    global_ctx = setup_logger_context(
        name="Global",
        log_file_path="pipeline_execution.log",
        console=True
    )

    with global_ctx as main_logger:
        main_logger.info("Starting Pipeline...")

        # --- STEP 1: SETUP ---
        # Logic moved to src/services/experiment_service.py
        generate_configs(datasets, param_grid, main_logger)

        # --- STEP 2: LOADING EXPERIMENTS ---
        # Logic moved to src/services/experiment_service.py
        experiments = load_existing_experiments(main_logger)

        if experiments:
            main_logger.info(f"Ready to process {len(experiments)} experiments.")

        # --- STEP 3: LOADING DATA ---
        # Using the renamed Session Loader (handles caching, merging, UEM cutting)
        target_dataset = datasets[0]
        sessions = load_sessions(target_dataset, main_logger)

        # --- STEP 4: EXECUTION (Algorithm) ---
        # TODO: Here we will iterate over experiments and sessions
        # execution_service.run_pipeline(experiments, sessions, main_logger)

        # --- DEINE TODO LISTE (Status Update) ---

        # TODO: save the results [ERLEDIGT durch loader_service: speichert _preprocessed.json]
        # TODO: merged asd scores for each speaker in each session [ERLEDIGT: Passiert im Loader]
        # ToDo: only save the relevant part (UEM start and end!) [ERLEDIGT: Passiert durch _apply_uem_cut im Loader]
        # TODO: UEM start and end time needs to be saved as well [ERLEDIGT: Ist in SessionSpeakerData Attributen und im JSON]
        # QUESTION: how to save both parts - together? [ERLEDIGT: Ja, in einer JSON Datei pro Session]

        # TODO: .csv-Übersichtsdatei über Experimente einbauen [OFFEN -> Nächster Schritt?]

        # TODO: weitere tests einbauen [OFFEN -> Nächster Schritt?]
        # TODO: fusion der asd-scores [ERLEDIGT: Logik ist in loader_service, aber Test fehlt noch]
        # QUESTION: wie überprüfen...
        # TODO: split von load_sessions() in load_json(), merge_tracks(), apply_uem_cut() [ERLEDIGT: Service ist gesplittet]
        # TODO: richtiges "Beschneiden" der ASD-Scores [ERLEDIGT: Logik ist implementiert, Test fehlt]
        # TODO: ... weitere Tests ...

        # TODO: Logging erweitern [ERLEDIGT: utils/logging_utils ist da]
        # TODO: Erstellung der .csv-Experiment-Übersichtsdatei [OFFEN]

        # TODO: Klassen und Funktionen in Dateien splitten [ERLEDIGT: src/models, src/services]
        # QUESTION: Wie soll die Ordnerstruktur organisiert werden? [ERLEDIGT]

    main_logger.info("Pipeline Finished.")


if __name__ == "__main__":
    main()