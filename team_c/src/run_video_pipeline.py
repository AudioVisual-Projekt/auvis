"""
ORCHESTRATOR: Team_c gaze + distance + clustering pipeline with STEP SELECTION.

KEY FEATURE: Run individual steps independently without waiting for earlier steps.
- Edit constants at top to select which steps to run
- Run from PyCharm with NO CLI arguments
- Automatically skips prerequisites that are missing; gracefully warns

USAGE:
1. Edit RUN_STEP* flags and SESSION_FILTER near top
2. Click Run in PyCharm
3. Watch console output
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

# ============================================================
# STEP SELECTION CONFIGURATION (EDIT THESE)
# ============================================================

# Which steps to run (set to False to skip)
RUN_STEP1_SEAT_PRIOR = False           # Seat extraction from 360 video
RUN_STEP2A_GAZE = True                  # HEAD-YAW extraction (NEW ROBUST VERSION)
RUN_STEP2B_MATRICES = True            # Build distance matrices
RUN_STEP2C_AGGLOM = True             # Agglomerative clustering
RUN_STEP3_BASELINES = True             # Seating baseline heuristics

# Session filtering (None = all sessions, or list like ["session_57"])
SESSION_FILTER = ["session_132"]                  # Try None for all, or ["session_40"] for one

# ============================================================
# Auto-Detect Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent  # src/
SRC_DIR = SCRIPT_DIR

# Find TEAM_C_ROOT
TEAM_C_ROOT = SRC_DIR
while TEAM_C_ROOT.name != "team_c" and TEAM_C_ROOT.parent != TEAM_C_ROOT:
    TEAM_C_ROOT = TEAM_C_ROOT.parent

# Data roots
CANDIDATE_DATA_ROOTS = [
    TEAM_C_ROOT / "data-bin" / "dev",
    TEAM_C_ROOT / "data-bin",
]

DATA_ROOT = None
VIDEOS_ROOT = None

for dr in CANDIDATE_DATA_ROOTS:
    vr = dr / "dev_central_videos"
    if vr.exists():
        DATA_ROOT = dr
        VIDEOS_ROOT = vr
        break

if VIDEOS_ROOT is None:
    DATA_ROOT = CANDIDATE_DATA_ROOTS[0]
    VIDEOS_ROOT = DATA_ROOT / "dev_central_videos"

# Output roots
OLD_OUTPUT_ROOT = DATA_ROOT / "output"
NEW_OUTPUT_ROOT = TEAM_C_ROOT / "data-bin" / "Output_blickrichtung"

# Import pipeline modules
sys.path.insert(0, str(SRC_DIR))

from video_360.build_seat_prior_from_360_video import main as main_seat_prior
from video_360.gaze_estimation import (
    extract_gaze_for_session,
    save_gaze_result,
)

try:
    from cluster.speaker_distances import build_speaker_distance_matrices, save_global_distance_matrices
    from cluster.agglomerative import cluster_session_matrices, save_clustering_results
    _CLUSTER_MODULES_OK = True
except ImportError:
    _CLUSTER_MODULES_OK = False

try:
    from cluster.seating_speaker_baselines import build_seating_speaker_clusters_for_session
    _BASELINE_MODULE_OK = True
except ImportError:
    _BASELINE_MODULE_OK = False

# ============================================================
# Utility Functions
# ============================================================

def log(msg: str, level: str = "INFO"):
    """Simple logging."""
    print(f"[main.py] [{level}] {msg}", flush=True)

def find_sessions(vroot: Path, session_filter: List[str] | None = None) -> List[Path]:
    """Find all session_* directories with central_video.mp4."""
    sessions = []

    for p in sorted(vroot.glob("session_*")):
        if (p / "central_video.mp4").exists():
            # Apply filter if specified
            if session_filter is not None and p.name not in session_filter:
                continue
            sessions.append(p)

    return sessions


def check_prerequisites_for_gaze(session_out_dir: Path, session_name: str) -> bool:
    """Check if session has prerequisites for gaze extraction."""
    import os  # ADD THIS

    required_files = ["seat_order.json"]
    missing = []
    for fname in required_files:
        if not (session_out_dir / fname).exists():
            missing.append(fname)

    # WINDOWS FIX: Use os.listdir instead of glob
    session_name_only = session_out_dir.name
    possible_asd_root = session_out_dir.parent.parent / session_name_only
    central_crops_dir = possible_asd_root / "speakers" / "spk_0" / "central_crops"

    has_tracking = False
    if central_crops_dir.exists():
        try:
            for filename in os.listdir(str(central_crops_dir)):  # WINDOWS-SAFE!
                if filename.startswith("track_") and filename.endswith("_bbox.json"):
                    has_tracking = True
                    break
        except:
            pass

    if not has_tracking:
        missing.append("tracking data (spk_0/central_crops/track_*_bbox.json)")

    if missing:
        log(f"[{session_name}] SKIP gaze (missing: {', '.join(missing)})", "WARN")
        return False

    return True


def check_prerequisites_for_matrices(session_out_dir: Path, session_name: str) -> bool:
    """Check if session has prerequisites for distance matrices."""
    required_files = ["gaze_tracks.json", "seat_geometry.npz", "asd_seat_matching.json"]

    missing = [f for f in required_files if not (session_out_dir / f).exists()]

    if missing:
        log(f"[{session_name}] SKIP matrices (missing: {', '.join(missing)})", "WARN")
        return False

    return True

# ============================================================
# Main Pipeline
# ============================================================

def main():
    """Run selected steps of the pipeline."""

    log("=" * 70, "START")
    log("TEAM_C GAZE + DISTANCE + CLUSTERING PIPELINE (Step Selection Mode)", "INFO")
    log("=" * 70, "START")

    log(f"VIDEOS_ROOT: {VIDEOS_ROOT}", "CONFIG")
    log(f"OLD_OUTPUT_ROOT: {OLD_OUTPUT_ROOT}", "CONFIG")
    log(f"NEW_OUTPUT_ROOT: {NEW_OUTPUT_ROOT}", "CONFIG")

    # Display which steps will run
    log("", "INFO")
    log("SELECTED STEPS:", "INFO")
    log(f"  Step 1 (Seat Prior):        {RUN_STEP1_SEAT_PRIOR}", "INFO")
    log(f"  Step 2a (Gaze):             {RUN_STEP2A_GAZE}", "INFO")
    log(f"  Step 2b (Distance Matrices): {RUN_STEP2B_MATRICES}", "INFO")
    log(f"  Step 2c (Agglomerative):    {RUN_STEP2C_AGGLOM}", "INFO")
    log(f"  Step 3 (Seating Baselines): {RUN_STEP3_BASELINES}", "INFO")
    log(f"  SESSION_FILTER: {SESSION_FILTER}", "INFO")
    log("", "INFO")

    if not VIDEOS_ROOT or not VIDEOS_ROOT.exists():
        log(f"ERROR: VIDEOS_ROOT not found: {VIDEOS_ROOT}", "ERROR")
        return

    NEW_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # STEP 1: Seat Prior (OPTIONAL)
    # ================================================================

    if RUN_STEP1_SEAT_PRIOR:
        log("-" * 70, "STEP1")
        log("STEP 1: SEAT PRIOR EXTRACTION", "STEP")
        log("-" * 70, "STEP1")

        main_seat_prior(
            videos_root=VIDEOS_ROOT,
            output_root=OLD_OUTPUT_ROOT,
            recompute_geometry=False,
        )
    else:
        log("-" * 70, "STEP1")
        log("STEP 1: SKIPPED (RUN_STEP1_SEAT_PRIOR = False)", "SKIP")
        log("-" * 70, "STEP1")

    # ================================================================
    # STEP 2: Per-Session Processing
    # ================================================================

    sessions = find_sessions(VIDEOS_ROOT, SESSION_FILTER)

    if not sessions:
        log(f"No sessions found (filter={SESSION_FILTER})", "ERROR")
        return

    log(f"Found {len(sessions)} session(s) to process", "INFO")

    # ================================================================
    # STEP 2a: Gaze Extraction (NEW ROBUST)
    # ================================================================

    if RUN_STEP2A_GAZE:
        log("-" * 70, "STEP2a")
        log("STEP 2a: GAZE EXTRACTION (HEAD-YAW)", "STEP")
        log("-" * 70, "STEP2a")

        for session_dir in sessions:
            session_name = session_dir.name

            # Determine output directory
            # Try new output first, fall back to old
            session_new_out_dir = NEW_OUTPUT_ROOT / session_name
            session_old_out_dir = OLD_OUTPUT_ROOT / session_name

            # Use new output if gaze_only, otherwise use existing old output

            session_out_dir = session_new_out_dir
            session_out_dir.mkdir(parents=True, exist_ok=True)

            log(f"[{session_name}] Checking prerequisites...", "INFO")

            # WINDOWS FIX: Copy seat_order.json from old to new output dir
            old_seat_order = session_old_out_dir / "seat_order.json"
            new_seat_order = session_out_dir / "seat_order.json"
            if old_seat_order.exists() and not new_seat_order.exists():
                import shutil
                shutil.copy2(str(old_seat_order), str(new_seat_order))
                log(f"[{session_name}] Copied seat_order.json to new output dir", "INFO")

            if not check_prerequisites_for_gaze(session_out_dir, session_name):
                continue

            log(f"[{session_name}] Extracting gaze...", "INFO")

            try:
                video_path = session_dir / "central_video.mp4"

                gaze_result = extract_gaze_for_session(
                    session_name=session_name,
                    session_out_dir=session_out_dir,
                    video_path=video_path,
                )

                # Save result (in old output for now; can move later)
                save_gaze_result(session_out_dir, gaze_result)

                # Report
                total_samples = gaze_result.get("total_samples", 0)
                n_persons = len(gaze_result.get("persons", {}))
                n_with_yaw = sum(
                    1 for p in gaze_result.get("persons", {}).values()
                    if isinstance(p, dict) and p.get("n_samples", 0) > 0
                )

                log(
                    f"[{session_name}] → {n_persons} persons, {n_with_yaw} with yaw, "
                    f"{total_samples} total samples",
                    "SUCCESS" if total_samples > 0 else "WARN"
                )

                if "error" in gaze_result:
                    log(f"[{session_name}]   Error: {gaze_result['error']}", "WARN")

            except Exception as e:
                log(f"[{session_name}] ERROR: {e}", "ERROR")

    else:
        log("-" * 70, "STEP2a")
        log("STEP 2a: SKIPPED (RUN_STEP2A_GAZE = False)", "SKIP")
        log("-" * 70, "STEP2a")

    # ================================================================
    # STEP 2b: Distance Matrices (OPTIONAL)
    # ================================================================

    if RUN_STEP2B_MATRICES:
        if not _CLUSTER_MODULES_OK:
            log("STEP 2b: SKIPPED (cluster modules not available)", "WARN")
        else:
            log("-" * 70, "STEP2b")
            log("STEP 2b: BUILDING DISTANCE MATRICES", "STEP")
            log("-" * 70, "STEP2b")

            all_sessions_matrices = {}

            for session_dir in sessions:
                session_name = session_dir.name
                session_out_dir = OLD_OUTPUT_ROOT / session_name

                if not session_out_dir.exists():
                    log(f"[{session_name}] SKIP matrices (no output dir from step 1)", "WARN")
                    continue

                if not check_prerequisites_for_matrices(session_out_dir, session_name):
                    continue

                log(f"[{session_name}] Building matrices...", "INFO")

                try:
                    matrices_result = build_speaker_distance_matrices(session_out_dir)

                    n_spk = matrices_result.get("n_speakers", 0)
                    n_dropped = len(matrices_result.get("dropped_speakers", []))
                    n_gaze_missing = len(matrices_result.get("missing", {}).get("gaze_speakers", []))

                    log(
                        f"[{session_name}] → {n_spk} speakers (dropped: {n_dropped}), "
                        f"{n_gaze_missing} missing gaze",
                        "SUCCESS" if n_spk > 0 else "WARN"
                    )

                    all_sessions_matrices[session_name] = matrices_result

                except Exception as e:
                    log(f"[{session_name}] ERROR: {e}", "ERROR")

            # Save global matrices
            if all_sessions_matrices:
                try:
                    matrices_path = save_global_distance_matrices(
                        output_root=OLD_OUTPUT_ROOT,
                        all_sessions_matrices=all_sessions_matrices,
                    )
                    log(f"Saved global: {matrices_path}", "SUCCESS")
                except Exception as e:
                    log(f"ERROR saving global matrices: {e}", "ERROR")

    else:
        log("-" * 70, "STEP2b")
        log("STEP 2b: SKIPPED (RUN_STEP2B_MATRICES = False)", "SKIP")
        log("-" * 70, "STEP2b")

    # ================================================================
    # STEP 2c: Agglomerative Clustering (OPTIONAL)
    # ================================================================

    if RUN_STEP2C_AGGLOM:
        if not _CLUSTER_MODULES_OK:
            log("STEP 2c: SKIPPED (cluster modules not available)", "WARN")
        else:
            log("-" * 70, "STEP2c")
            log("STEP 2c: AGGLOMERATIVE CLUSTERING", "STEP")
            log("-" * 70, "STEP2c")

            # Re-load matrices from global file if available
            global_matrices_file = OLD_OUTPUT_ROOT / "speaker_distance_matrices_all_sessions.json"

            if not global_matrices_file.exists():
                log("No global matrices file found; skipping clustering", "WARN")
            else:
                try:
                    with global_matrices_file.open("r", encoding="utf-8") as f:
                        global_data = json.load(f)

                    for session_name, matrices_result in global_data.get("sessions", {}).items():
                        session_out_dir = OLD_OUTPUT_ROOT / session_name

                        speakers = matrices_result.get("speakers", [])
                        matrices = matrices_result.get("matrices", {})

                        if not speakers or not matrices:
                            log(f"[{session_name}] SKIP clustering (no speakers/matrices)", "WARN")
                            continue

                        log(f"[{session_name}] Clustering {len(speakers)} speakers...", "INFO")

                        try:
                            clustering_result = cluster_session_matrices(
                                session_name=session_name,
                                speakers=speakers,
                                matrices=matrices,
                            )

                            k_combined = clustering_result.get("k_combined", 0)
                            k_seat = clustering_result.get("k_seat", 0)
                            k_gaze = clustering_result.get("k_gaze_interaction", 0)

                            log(
                                f"[{session_name}] → k_combined={k_combined}, k_seat={k_seat}, k_gaze={k_gaze}",
                                "SUCCESS"
                            )

                            save_clustering_results(session_out_dir, clustering_result)

                        except Exception as e:
                            log(f"[{session_name}] ERROR: {e}", "ERROR")

                except Exception as e:
                    log(f"ERROR loading global matrices: {e}", "ERROR")

    else:
        log("-" * 70, "STEP2c")
        log("STEP 2c: SKIPPED (RUN_STEP2C_AGGLOM = False)", "SKIP")
        log("-" * 70, "STEP2c")

    # ================================================================
    # STEP 3: Seating Baselines (OPTIONAL)
    # ================================================================

    if RUN_STEP3_BASELINES:
        if not _BASELINE_MODULE_OK:
            log("STEP 3: SKIPPED (seating baseline module not available)", "WARN")
        else:
            log("-" * 70, "STEP3")
            log("STEP 3: SEATING-BASED CLUSTERING HEURISTICS", "STEP")
            log("-" * 70, "STEP3")

            for session_dir in sessions:
                session_name = session_dir.name
                session_out_dir = OLD_OUTPUT_ROOT / session_name

                if not session_out_dir.exists():
                    log(f"[{session_name}] SKIP (no output dir)", "WARN")
                    continue

                log(f"[{session_name}] Building seating baseline heuristics...", "INFO")

                try:
                    results = build_seating_speaker_clusters_for_session(session_name)
                    variants = ", ".join(sorted(results.keys()))
                    log(f"[{session_name}] → variants: {variants}", "SUCCESS")
                except Exception as e:
                    log(f"[{session_name}] ERROR: {e}", "ERROR")

    else:
        log("-" * 70, "STEP3")
        log("STEP 3: SKIPPED (RUN_STEP3_BASELINES = False)", "SKIP")
        log("-" * 70, "STEP3")

    # ================================================================
    # Done
    # ================================================================

    log("=" * 70, "DONE")
    log("PIPELINE COMPLETE", "SUCCESS")
    log("=" * 70, "DONE")

if __name__ == "__main__":
    main()
