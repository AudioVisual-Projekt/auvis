import os
import argparse
import json
import glob

# Pfad: .../task5_semantic_prototype/auvis/team_c/script
# -> eine Ebene hoch: .../team_c
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Semantische Daten liegen in: .../team_c/data-bin
TEAMC_DATABIN = os.path.join(PROJECT_ROOT, "data-bin")

# src auf sys.path legen
os.sys.path.append(os.path.join(PROJECT_ROOT))

from src.cluster.conv_spks import (
    get_speaker_activity_segments,
    calculate_conversation_scores,
    cluster_speakers,
    get_clustering_f1_score,
    semantic_cluster_speakers,
    hybrid_cluster_speakers,
)


class InferenceEngine:
    """Main inference engine that handles model selection and processing"""

    def __init__(self, max_length=15):
        self.max_length = max_length

    def mcorec_session_infer(self, session_dir, output_dir):
        """Process a complete MCoReC session"""

        # ---- 1) Session-Metadaten laden ----
        with open(os.path.join(session_dir, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # ---- 2) Sprecher-Segmente aus ASD holen ----
        speaker_segments = {}
        for speaker_name, speaker_data in metadata.items():
            list_tracks_asd = [
                os.path.join(session_dir, track["asd"])
                for track in speaker_data["central"]["crops"]
            ]

            uem_start = speaker_data["central"]["uem"]["start"]
            uem_end = speaker_data["central"]["uem"]["end"]

            speaker_activity_segments = get_speaker_activity_segments(
                list_tracks_asd, uem_start, uem_end
            )
            speaker_segments[speaker_name] = speaker_activity_segments

        # Sprecherreihenfolge fixieren
        speaker_ids = list(speaker_segments.keys())

        # Session-Name (für IDs wie dev0001_SPK01 wichtig)
        session_name = os.path.basename(os.path.normpath(session_dir))

        # ---- 3) Baseline: Zeit-basiertes Clustering ----
        scores_time = calculate_conversation_scores(speaker_segments)
        clusters_time = cluster_speakers(scores_time, speaker_ids)

        with open(os.path.join(output_dir, "speaker_to_cluster_time.json"), "w") as f:
            json.dump(clusters_time, f, indent=4)

        # ---- 4) Rein semantisches Clustering ----
        clusters_semantic = semantic_cluster_speakers(
            data_dir=TEAMC_DATABIN,          # <<-- WICHTIG: immer team_c/data-bin
            session_name=session_name,
            speaker_names=speaker_ids,
        )

        with open(
            os.path.join(output_dir, "speaker_to_cluster_semantic.json"), "w"
        ) as f:
            json.dump(clusters_semantic, f, indent=4)

        # ---- 5) Hybrid-Clustering (Zeit + Semantik) ----
        clusters_hybrid = hybrid_cluster_speakers(
            speaker_segments=speaker_segments,
            data_dir=TEAMC_DATABIN,          # <<-- ebenfalls team_c/data-bin
            session_name=session_name,
            alpha=0.5,
        )

        with open(
            os.path.join(output_dir, "speaker_to_cluster_hybrid.json"), "w"
        ) as f:
            json.dump(clusters_hybrid, f, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Unified inference script for conversation clustering (Zeit / Semantik / Hybrid)"
    )
    parser.add_argument(
        "--session_dir",
        type=str,
        required=True,
        help="Glob-Pfad zu den MCoRec-Session-Ordnern, z. B. '../../mcorec_baseline/data-bin/dev/session_*'",
    )
    parser.add_argument(
        "--output_dir_name",
        type=str,
        default="output_semantic",
        help="Name des Output-Ordners unterhalb von team_c/data-bin",
    )
    args = parser.parse_args()

    # Session-Glob relativ zum PROJECT_ROOT (team_c) interpretieren
    session_dir_arg = os.path.join(PROJECT_ROOT, args.session_dir)

    # Alle Session-Dirs sammeln
    all_session_dirs = [p for p in glob.glob(session_dir_arg) if os.path.isdir(p)]

    print(f"Inferring {len(all_session_dirs)} sessions using time / semantic / hybrid clustering")
    if not all_session_dirs:
        print(f"⚠ Keine Sessions gefunden für Pattern: {session_dir_arg}")

    engine = InferenceEngine()

    # Outputs nach team_c/data-bin/<output_dir_name>/session_xxx schreiben
    output_base = os.path.join(TEAMC_DATABIN, args.output_dir_name)

    for session_dir in all_session_dirs:
        session_name = os.path.basename(os.path.normpath(session_dir))
        output_dir = os.path.join(output_base, session_name)

        print("Session dir:", session_dir)
        print("Output dir will be:", output_dir)

        os.makedirs(output_dir, exist_ok=True)

        print(f"Processing session {session_name}")
        engine.mcorec_session_infer(session_dir, output_dir)


if __name__ == "__main__":
    main()
