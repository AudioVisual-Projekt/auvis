"""
Module: experiment_service
==========================

Handles the management of experiment configurations.
Responsibilites:
1. Generating folder structures from a parameter grid (Grid Search Setup).
2. Discovering and loading existing experiment configurations from disk.
"""

import csv
import os
import json
from glob import glob
from itertools import product
from typing import List, Dict, Any

from ..models.segmentation_experiment import SegmentationConfig, SegmentationExperimentModel
from ..services.segmentation_algorithm import SegmentationAlgorithm


def _get_project_root() -> str:
    """Calculates the project root directory (team_c)."""
    # src/services/segmentation_workflow_service.py -> src/services -> src -> team_c
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))


def generate_configs(datasets: List[str], param_grid: Dict[str, List[Any]], global_logger) -> None:
    """
    Step 1: Generates the folder structure based on the parameter grid.
    """
    global_logger.info("--- Service: Generating Configurations ---")

    # Generate combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    created_count = 0
    skipped_count = 0

    for ds in datasets:
        for params in combinations:
            cfg = SegmentationConfig(dataset=ds, **params)
            model = SegmentationExperimentModel(cfg)

            if model.create_structure():
                created_count += 1
            else:
                skipped_count += 1

    global_logger.info(f"--> Done. Created: {created_count}, Skipped: {skipped_count}")


def load_existing_experiments(global_logger) -> List[SegmentationExperimentModel]:
    """
    Step 2: Scans the output directory for existing configurations.
    """
    global_logger.info("--- Service: Loading Experiments from Disk ---")

    root = _get_project_root()

    # Path: data-bin/_output/av_asd/*/*/seg_config.json
    search_pattern = os.path.join(root, "data-bin", "_output", "av_asd", "*", "*", "seg_config.json")

    config_files = glob(search_pattern)

    if not config_files:
        global_logger.warning("No configurations found! Did you run generate_configs first?")
        return []

    global_logger.info(f"Found {len(config_files)} config files. Re-initializing models...")

    loaded_experiments = []

    for json_path in config_files:
        try:
            model = SegmentationExperimentModel.from_json(json_path)
            loaded_experiments.append(model)
        except Exception as e:
            global_logger.error(f"Failed to load {json_path}: {e}")

    global_logger.info(f"--> Successfully loaded {len(loaded_experiments)} experiments.\n")
    return loaded_experiments


def update_experiment_registry(global_logger):
    """
    Scans all existing experiment configurations and creates/updates a central
    CSV overview file: 'experiments_overview.csv'.
    """
    global_logger.info("--- Service: Updating Experiment Registry (CSV) ---")

    root = _get_project_root()
    base_output_dir = os.path.join(root, "data-bin", "_output", "av_asd")

    # Target CSV file
    csv_path = os.path.join(base_output_dir, "experiments_overview.csv")

    # 1. Find all config files
    search_pattern = os.path.join(base_output_dir, "*", "*", "seg_config.json")
    config_files = glob(search_pattern)

    if not config_files:
        global_logger.warning("No experiments found to list in CSV.")
        return

    # 2. Collect data
    registry_data = []

    # Read the first file to determine headers (column names).
    try:
        sample_model = SegmentationExperimentModel.from_json(config_files[0])
        # Get all config keys and sort them
        all_param_keys = sorted(sample_model.seg_config.__dict__.keys())

        # Remove 'dataset' from the generic list to place it explicitly at the start
        if "dataset" in all_param_keys:
            all_param_keys.remove("dataset")

    except Exception as e:
        global_logger.error(f"Error reading sample config: {e}")
        return

    # Define final columns: Dataset first, then ID, then remaining parameters, then metrics
    metric_keys = ["precision", "recall", "f1_score"]
    headers = ["dataset", "seg_experiment_id"] + all_param_keys + metric_keys

    for json_path in config_files:
        try:
            model = SegmentationExperimentModel.from_json(json_path)

            # Build the row
            row = {
                "dataset": model.seg_config.dataset,
                "seg_experiment_id": model.seg_exp_id
            }

            # Add remaining parameters
            row.update(model.seg_config.__dict__)

            # Initialize metrics with empty strings
            for mk in metric_keys:
                row[mk] = ""

            registry_data.append(row)
        except Exception as e:
            global_logger.error(f"Skipping corrupted config {json_path}: {e}")

    # 3. Sort rows (by dataset first, then by experiment ID)
    registry_data.sort(key=lambda x: (x.get("dataset", ""), x["seg_experiment_id"]))

    # 4. Write to CSV
    try:
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for row in registry_data:
                writer.writerow(row)

        global_logger.info(f"--> Registry updated: {csv_path}")
        global_logger.info(f"--> Listed {len(registry_data)} experiments.\n")

    except IOError as e:
        global_logger.error(f"Could not write CSV registry: {e}")


def run_segmentation_pipeline(experiments, sessions, global_logger):
    """
    Executes the segmentation for all loaded experiments on the provided session data.
    Saves the results as JSON files in the corresponding experiment folders.
    """
    global_logger.info("--- Service: Running Segmentation Pipeline ---")

    for exp_model in experiments:
        exp_id = exp_model.seg_exp_id
        global_logger.info(f"Processing Experiment: {exp_id}")

        # 1. Initialize Algorithm with this experiment's config
        algo = SegmentationAlgorithm(exp_model.seg_config.__dict__)

        # writing in .../<hash>/experiment.log
        with exp_model.experiment_logger() as exp_logger:
            exp_logger.info(f"--- Starting Processing for {len(sessions)} Sessions (from ASD to ActiveSpeakerSegments in s) ---")

            # 2. Iterate over all loaded sessions
            for session_id, session_model in sessions.items():

                # Prepare output structure
                output_data = {
                    "session_id": session_id,
                    "hash": exp_id,
                    "parameters": exp_model.seg_config.__dict__,  # Save params for reference
                    "speakers": {}
                }

                # Access the structured speaker data from the SessionModel
                # Attribute is named 'input_data' (Dict[str, SessionSpeakerData])
                speakers_data = session_model.input_data

                for spk_id, spk_data_obj in speakers_data.items():
                    # Extract inputs from SessionSpeakerData object
                    uem_start = spk_data_obj.uem_start
                    uem_end = spk_data_obj.uem_end

                    # FIX: Attribute is named 'asd_scores' in SessionSpeakerData
                    scores = spk_data_obj.asd_scores

                    # NOTE: Your SessionSpeakerData uses int keys (dict[int, float]).
                    # The SegmentationAlgorithm logic should handle int keys correctly.
                    # If the Algorithm expects strings (legacy JSON), this might return empty results.
                    # Assuming Algorithm handles int keys or scores are compliant:
                    segments = algo.run(scores)

                    # Save to output structure
                    output_data["speakers"][spk_id] = {
                        "uem_start": uem_start,
                        "uem_end": uem_end,
                        "segments": segments
                    }

                # 3. Save Result to Disk
                output_dir = exp_model.segments_dir

                output_filename = f"{session_id}_segments.json"
                output_path = os.path.join(output_dir, output_filename)

                try:
                    with open(output_path, 'w') as f:
                        json.dump(output_data, f, indent=4)

                    # LOGGING: Erfolg in das lokale Log schreiben
                    exp_logger.info(f"Processed session: {session_id}")

                except Exception as e:
                    error_msg = f"Failed to save {output_filename}: {e}"
                    global_logger.error(error_msg)
                    # LOGGING: Fehler auch ins lokale Log
                    exp_logger.error(error_msg)

            exp_logger.info("--- Experiment Processing Finished ---")

    global_logger.info("--> Segmentation Pipeline execution finished.")
