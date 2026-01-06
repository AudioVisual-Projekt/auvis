"""
One-click runner for Team C seat prior + gaze + speaker clustering.

Run this file from PyCharm's ▶ Run button (no CLI required).
It will:
1) Build per-session seat priors (and ASD seat matching) from 360° central videos
   and also extract head-yaw ("gaze") summaries into gaze_tracks.json.
2) Build seat-based baselines + new agglomerative speaker clustering variants
   using seat/gaze/combined speaker distance matrices.
3) Write a global output: data-bin/dev/output/speaker_distance_matrices.json

This runner uses dynamic imports because the upstream scripts live in src/
and one of them contains a hyphen in the filename.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _import_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    team_c_root = Path(__file__).resolve().parent

    build_path = team_c_root / "src" / "video_360" / "build_seat_prior_from_360_video.py"
    baseline_path = team_c_root / "src" / "cluster" / "seating_speaker_baselines.py"

    if not build_path.exists():
        raise FileNotFoundError(f"Missing: {build_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing: {baseline_path}")

    build_mod = _import_from_path("build_seat_prior", build_path)
    baseline_mod = _import_from_path("seating_speaker_baselines", baseline_path)

    print("[main] Step 1/2: build seat priors + ASD matching + gaze ...")
    build_mod.main()  # uses internal defaults / root discovery

    print("[main] Step 2/2: speaker clustering baselines + agglomerative + global JSON ...")
    baseline_mod.main_all_sessions()

    print("[main] Done.")


if __name__ == "__main__":
    main()
