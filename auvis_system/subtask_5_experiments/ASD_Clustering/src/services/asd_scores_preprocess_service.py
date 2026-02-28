"""
Module: loader_service
======================

Handles the loading and preprocessing of session data.
Implements an ETL (Extract, Transform, Load) pipeline with caching.

The logic is:
1. Check if a preprocessed JSON exists in 'data-bin/av_asd/_output/.../00_preprocessed'.
2. If YES: Load it directly (Fast).
3. If NO:
    - Extract: Load raw metadata and stitching ASD track files.
    - Transform: Merge tracks and filter frames outside valid UEM times.
    - Load: Save the clean data to JSON for future use.
"""

import json
import os
import math
from glob import glob
from typing import Dict

# Relative import from the models package
from ..models.session_input import SessionModel, SessionSpeakerData


def _get_project_root() -> str:
    """Calculates the project root directory based on this file's location."""
    # src/services/asd_scores_preprocess_service.py -> src/services -> src -> team_c
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))


def _get_preprocessed_path(dataset_name: str, session_id: str) -> str:
    """
    Constructs the path for the cached JSON file.
    Structure: data-bin/av_asd/_output/<dataset>/00_preprocessed/<session_id>.json
    """
    root = _get_project_root()
    output_dir = os.path.join(root, "data-bin", "_output", "av_asd", dataset_name, "00_preprocessed")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{session_id}.json")


def _load_raw_session_data(session_folder: str, global_logger) -> SessionModel:
    """
    Internal helper: Loads raw metadata and stitches ASD tracks together.
    Does NOT apply UEM filtering yet.
    """
    folder_name = os.path.basename(session_folder)
    meta_path = os.path.join(session_folder, "metadata.json")

    with open(meta_path, 'r') as f:
        meta_json = json.load(f)

    parsed_speakers = {}

    # Iterate over speakers (spk_0, spk_1...)
    for spk_id, spk_data in meta_json.items():
        if "central" not in spk_data:
            continue

        # Extract UEM limits directly
        uem_info = spk_data["central"].get("uem", {})
        u_start = uem_info.get("start", 0.0)
        u_end = uem_info.get("end", 0.0)

        # Init Container
        speaker_obj = SessionSpeakerData(
            speaker_id=spk_id,
            uem_start=u_start,
            uem_end=u_end,
            asd_scores={}
        )

        # Stitch Tracks (Crops)
        crops_list = spk_data["central"].get("crops", [])
        for crop in crops_list:
            rel_asd_path = crop.get("asd")
            if not rel_asd_path:
                continue

            full_asd_path = os.path.join(session_folder, rel_asd_path)
            if os.path.exists(full_asd_path):
                with open(full_asd_path, 'r') as f:
                    asd_json = json.load(f)

                # Merge into main dict (keys are absolute frame IDs)
                # We cast keys to int and values to float for safety
                for k, v in asd_json.items():
                    speaker_obj.asd_scores[int(k)] = float(v)

        parsed_speakers[spk_id] = speaker_obj

    return SessionModel(
        session_id=folder_name,
        folder_path=session_folder,
        metadata=meta_json,
        input_data=parsed_speakers
    )


def _apply_uem_cut(session: SessionModel, global_logger) -> SessionModel:
    """
    Internal helper: Removes all ASD scores that are outside the speaker's UEM range.
    STRATEGY: Strict Inside.
    - Start: Round UP (ceil). Don't include frames starting before UEM start.
    - End: Round DOWN (floor). Don't include frames starting after UEM end.
    """
    fps = 25.0

    for spk_id, spk_data in session.input_data.items():
        start_time = spk_data.uem_start
        end_time = spk_data.uem_end

        if start_time == 0.0 and end_time == 0.0:
            continue

        # Convert time to frames

        # START: Round UP (Ceiling) to avoid pre-speech noise.
        # Example: Start 1.5s (at 25fps) = 37.5 frames -> Ceil -> Frame 38.
        # Frame 37 starts at 1.48s (OUT). Frame 38 starts at 1.52s (IN).
        start_frame = int(math.ceil(start_time * fps))

        # END: Round DOWN (Floor).
        # Example: End 2.5s (at 25fps) = 62.5 frames -> Floor -> Frame 62.
        # Frame 62 starts at 2.48s (IN). Frame 63 starts at 2.52s (OUT/Whistle).
        end_frame = int(math.floor(end_time * fps))

        # Filter logic: Keep only frames strictly within [start_frame, end_frame]
        spk_data.asd_scores = {
            f: s for f, s in spk_data.asd_scores.items()
            if start_frame <= f <= end_frame
        }

    return session


def _save_preprocessed_json(session: SessionModel, filepath: str):
    """
    Internal helper: Saves the cleaned session data to a single JSON file.
    """
    data = {
        "session_id": session.session_id,
        "folder_path": session.folder_path,
        "speakers": {}
    }

    for spk_id, spk_data in session.input_data.items():
        data["speakers"][spk_id] = {
            "uem_start": spk_data.uem_start,
            "uem_end": spk_data.uem_end,
            "scores": spk_data.asd_scores  # Keys will become strings in JSON automatically
        }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)  # needs more space if using indent!


def _load_preprocessed_json(filepath: str) -> SessionModel:
    """
    Internal helper: Fast loading from the clean JSON cache.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    parsed_speakers = {}
    for spk_id, val in data["speakers"].items():
        # Convert keys back to int (JSON stores keys as strings)
        scores_int_keys = {int(k): v for k, v in val["scores"].items()}

        parsed_speakers[spk_id] = SessionSpeakerData(
            speaker_id=spk_id,
            uem_start=val["uem_start"],
            uem_end=val["uem_end"],
            asd_scores=scores_int_keys
        )

    return SessionModel(
        session_id=data["session_id"],
        folder_path=data.get("folder_path", ""),
        metadata={},  # We don't need full raw metadata anymore for processing
        input_data=parsed_speakers
    )


def load_sessions(dataset_name: str, global_logger) -> Dict[str, SessionModel]:
    """
    Main Entry Point for Data Loading.

    1. Checks if preprocessed JSON exists.
    2. If yes -> Load fast.
    3. If no -> Load Raw, Stitch, Cut, Save, Return.

    Args:
        dataset_name (str): The name of the dataset (e.g., 'dev', 'train').
        global_logger: Logger instance for status updates.

    Returns:
        Dict[str, SessionModel]: Dictionary of loaded session objects.
    """
    global_logger.info(f"--- Loading Sessions for '{dataset_name}' (Modal: AV_ASD) ---")

    # Path construction: data-bin/av_asd/<dataset>
    root = _get_project_root()
    raw_dataset_dir = os.path.join(root, "data-bin", dataset_name)

    if not os.path.exists(raw_dataset_dir):
        global_logger.error(f"Dataset directory not found: {raw_dataset_dir}")
        return {}

    # Find session folders
    search_pattern = os.path.join(raw_dataset_dir, "session_*")
    session_folders = glob(search_pattern)

    loaded_sessions = {}

    # Progress counters
    total = len(session_folders)

    for idx, folder in enumerate(session_folders, 1):
        session_id = os.path.basename(folder)
        preproc_path = _get_preprocessed_path(dataset_name, session_id)

        try:
            if os.path.exists(preproc_path):
                # CACHE HIT
                # global_logger.info(f"[{idx}/{total}] Loading cached: {session_id}")
                session = _load_preprocessed_json(preproc_path)
            else:
                # CACHE MISS
                global_logger.info(f"[{idx}/{total}] Processing raw: {session_id} ...")
                # 1. Load Raw
                session = _load_raw_session_data(folder, global_logger)
                # 2. Cut UEM
                session = _apply_uem_cut(session, global_logger)
                # 3. Save Cache
                _save_preprocessed_json(session, preproc_path)

            loaded_sessions[session_id] = session

        except Exception as e:
            global_logger.error(f"Error loading {session_id}: {e}")

    global_logger.info(f"--> Successfully loaded {len(loaded_sessions)} sessions.\n")
    return loaded_sessions
