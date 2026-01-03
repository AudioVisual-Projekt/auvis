"""
Module: segmentation_experiment
===============================

Manages the configuration and execution environment for Active Speaker Segmentation experiments.
It acts as the "Controller" for a specific experimental run.

Responsibilities:
    1. Define parameters (SegmentationConfig).
    2. Manage file system paths (creation of output folders in data-bin/av_asd).
    3. Hold the calculation results in memory.
    4. Persist the configuration to disk.

Classes:
    - SegmentationConfig: Immutable configuration parameters.
    - SegmentationExperimentModel: The manager class for one experiment instance.
"""

import hashlib
import json
import os
from dataclasses import dataclass, asdict

# Relative import within the models package
from .segmentation_results import SegmentationResult

# Relative import from the utils package
from ..utils.logging_utils import setup_logger_context


@dataclass(frozen=True)
class SegmentationConfig:
    """
    Configuration parameters for the Active Speaker Segmentation algorithm.
    This class is frozen (immutable) to ensure parameters don't change during runtime.

    Attributes:
        dataset (str): Name of the dataset (e.g., 'dev', 'train').
        onset (float): Threshold to START a segment (Hysteresis High). Higher = stricter.
                       Default: 1.0.
        offset (float): Threshold to STOP a segment (Hysteresis Low).
                        Default: 0.8.
        min_duration_on (float): Minimum duration (in seconds) to keep a speech segment.
                                 Strategy: "DROP short speech". Default: 1.0.
        min_duration_off (float): Minimum duration (in seconds) of silence to split segments.
                                  Strategy: "FILL short gaps". Default: 0.5.
        max_chunk_size (int): Max speech length in seconds before forcing a split. Default: 10.
        min_chunk_size (int): Min chunk length allowed in seconds. Default: 1.
    """
    dataset: str
    onset: float = 1.0
    offset: float = 0.8
    min_duration_on: float = 1.0
    min_duration_off: float = 0.5
    max_chunk_size: int = 10
    min_chunk_size: int = 1

    def get_hash(self) -> str:
        """
        Generates a unique 8-char MD5 hash based on the configuration parameters.

        Note:
            The 'dataset' attribute is EXCLUDED from the hash calculation.
            This ensures that the same parameters produce the same ID across
            different datasets (e.g., onset=1.0 on 'dev' == onset=1.0 on 'train').

        Returns:
            str: The first 8 characters of the MD5 hash.
        """
        # convert dataclass to dict
        params = asdict(self)

        # IMPORTANT: Remove 'dataset' from the hash calculation.
        if 'dataset' in params:
            del params['dataset']

        # sort keys to ensure consistent hashing
        unique_str = json.dumps(params, sort_keys=True)

        # convert dictionary to String for using md5 on it
        return hashlib.md5(unique_str.encode('utf-8')).hexdigest()[:8]


class SegmentationExperimentModel:
    """
    Manages the environment for a specific segmentation experiment.
    Handles path creation, configuration persistence, and result storage in memory.

    Attributes:
        seg_config (SegmentationConfig): The configuration object used.
        seg_exp_id (str): The unique hash ID of the configuration.
        output_dir (str): The full path to the experiment output folder.
                          Format: .../data-bin/av_asd/_output/<dataset>/<hash>/
        results (dict[str, SegmentationResult]): In-memory storage of processed sessions.
            Key: session_id.
    """

    def __init__(self, seg_config: SegmentationConfig):
        self.seg_config = seg_config
        self.seg_exp_id = seg_config.get_hash()

        # Resolve Project Root.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))

        # Construct path: data-bin/av_asd/_output/<dataset>/<hash>/
        self.output_dir = os.path.join(
            project_root,
            "data-bin", "_output", "av_asd",
            self.seg_config.dataset,
            self.seg_exp_id
        )

        # Container for calculation results to allow in-memory access/debugging
        self.results: dict[str, SegmentationResult] = {}

    @property
    def segments_dir(self) -> str:
        """Sub-directory for segmentation results (.json)."""
        path = os.path.join(self.output_dir, "segments")
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def matrices_dir(self) -> str:
        """Sub-directory for distance matrices (.json)."""
        path = os.path.join(self.output_dir, "matrices")
        os.makedirs(path, exist_ok=True)
        return path

    def experiment_logger(self):
        """
        Context Manager: Sets up the local logger for this specific experiment.
        Writes to 'experiment.log' inside the output directory.

        Returns:
            The context manager from setup_logger_context.
        """
        if setup_logger_context is None:
            raise ImportError("Logger utility not found. Did you complete Step 2 (utils)?")

        log_path = os.path.join(self.output_dir, "experiment.log")
        logger_name = f"Local_{self.seg_exp_id}"

        return setup_logger_context(
            name=logger_name,
            log_file_path=log_path,
            console=False
        )

    def create_structure(self) -> bool:
        """
        Initializes the experiment folder and saves the config.json.

        Returns:
            bool: True if created, False if skipped (already exists).
        """
        if os.path.exists(self.output_dir):
            return False  # skip if already exists

        os.makedirs(self.output_dir, exist_ok=True)

        # We need the logger here, so we call the context manager
        with self.experiment_logger() as logger:
            logger.info("--- Init Experiment Structure ---")
            logger.info(f"Dataset: {self.seg_config.dataset}")
            logger.info(f"Params: {self.seg_config}")

            config_path = os.path.join(self.output_dir, "seg_config.json")
            with open(config_path, "w") as f:
                json.dump(asdict(self.seg_config), f, indent=4)

            logger.info(f"Config saved to {config_path}")

            # TODO: Here we will later call self.update_overview_csv()

        return True

    def add_result(self, result: SegmentationResult):
        """Stores a calculation result in memory."""
        self.results[result.session_id] = result

    def get_result(self, session_id: str) -> SegmentationResult:
        """Retrieves a result for a specific session."""
        return self.results.get(session_id)

    @classmethod
    def from_json(cls, json_path: str):
        """
        Factory method: Creates a Model instance by reading a config.json file.
        Used to reload experiments from disk.

        Args:
            json_path (str): Path to the seg_config.json file.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Create Config object from dictionary
        # **data unpacks the dict keys to arguments (onset=..., offset=...)
        config = SegmentationConfig(**data)

        return cls(config)

    def __repr__(self):
        return f"<SegmentationExperiment ID={self.seg_exp_id} | Dataset={self.seg_config.dataset}>"