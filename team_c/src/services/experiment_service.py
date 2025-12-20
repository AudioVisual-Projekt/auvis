"""
Module: experiment_service
==========================

Handles the management of experiment configurations.
Responsibilites:
1. Generating folder structures from a parameter grid (Grid Search Setup).
2. Discovering and loading existing experiment configurations from disk.
"""

import os
from glob import glob
from itertools import product
from typing import List, Dict, Any

from ..models.segmentation_experiment import SegmentationConfig, SegmentationExperimentModel


def _get_project_root() -> str:
    """Calculates the project root directory (team_c)."""
    # src/services/experiment_service.py -> src/services -> src -> team_c
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
