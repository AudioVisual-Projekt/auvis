"""
Module: clustering_experiment
=============================
Manages configuration and execution environment for Clustering experiments.
Depends on a parent SegmentationExperiment.
"""

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

# Importiere das Parent-Modell, um darauf aufzubauen
from team_c.src.models.segmentation_experiment import SegmentationExperimentModel
from team_c.src.utils.logging_utils import setup_logger_context


@dataclass(frozen=True)
class ClusteringConfig:
    """
    Configuration for Agglomerative Clustering.
    """
    threshold: float  # Distance threshold (e.g., 0.5)
    linkage: str = "complete"  # 'average', 'complete', 'single'
    metric: str = "precomputed"  # Always precomputed for our matrix approach

    def get_hash(self) -> str:
        """Generates unique ID for these clustering params."""
        params = asdict(self)
        unique_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(unique_str.encode('utf-8')).hexdigest()[:8]


class ClusteringExperimentModel:
    """
    Manages a specific clustering run.
    It is ALWAYS tied to a specific SegmentationExperimentModel (Parent),
    because it needs the matrices produced by that parent.
    """

    def __init__(self, clust_config: ClusteringConfig, parent_seg_exp: SegmentationExperimentModel):
        self.clust_config = clust_config
        self.parent_seg_exp = parent_seg_exp

        self.clust_exp_id = clust_config.get_hash()

        # Output folder is a SUB-FOLDER of the segmentation experiment
        # .../data-bin/_output/.../<SEG_HASH>/<CLUST_HASH>/
        self.output_dir = os.path.join(
            parent_seg_exp.output_dir,
            self.clust_exp_id
        )

    def experiment_logger(self):
        """Logger specifically for this clustering run."""
        os.makedirs(self.output_dir, exist_ok=True)
        log_path = os.path.join(self.output_dir, "clustering.log")
        logger_name = f"Clust_{self.clust_exp_id}"
        return setup_logger_context(name=logger_name, log_file_path=log_path, console=False)

    def create_structure(self) -> bool:
        """Creates the sub-folder and saves clust_config.json."""
        if os.path.exists(self.output_dir):
            return False

        os.makedirs(self.output_dir, exist_ok=True)

        config_path = os.path.join(self.output_dir, "clust_config.json")
        with open(config_path, "w") as f:
            # 1. Config in Dictionary umwandeln
            data = asdict(self.clust_config)

            # 2. Metadaten hinzufügen (Parent Hash UND eigener Hash)
            data["parent_seg_hash"] = self.parent_seg_exp.seg_exp_id
            data["cluster_hash"] = self.clust_exp_id  # <--- HIER IST DER FIX

            json.dump(data, f, indent=4)

        return True
