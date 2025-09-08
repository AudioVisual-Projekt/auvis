import os
import argparse
import json
import glob

# Add src to path
os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from team_c.src.cluster.conv_spks import (
    get_speaker_activity_segments,
    calculate_conversation_scores,
    cluster_speakers,
    get_clustering_f1_score
)


class InferenceEngine:
    """Main inference engine that handles model selection and processing"""

    def __init__(self, max_length=15):
        self.max_length = max_length


    def mcorec_session_infer(self, session_dir, output_dir):
        """Process a complete MCoReC session"""
        # Load session metadata
        session_dir2 = session_dir

        with open(os.path.join(session_dir, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Process speaker clustering
        speaker_segments = {}
        for speaker_name, speaker_data in metadata.items():
            list_tracks_asd = []
            for track in speaker_data['central']['crops']:
                list_tracks_asd.append(
                    os.path.join(session_dir, track['asd']))
            uem_start = speaker_data['central']['uem']['start']
            uem_end = speaker_data['central']['uem']['end']
            speaker_activity_segments = get_speaker_activity_segments(
                list_tracks_asd, uem_start, uem_end)
            speaker_segments[speaker_name] = speaker_activity_segments

        scores = calculate_conversation_scores(speaker_segments)
        clusters = cluster_speakers(scores, list(speaker_segments.keys()))
        output_clusters_file = os.path.join(output_dir,
                                            "speaker_to_cluster.json")
        with open(output_clusters_file, "w") as f:
            json.dump(clusters, f, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Unified inference script for multiple AVSR models"
    )
    parser.add_argument(
        '--session_dir',
        type=str,
        default='data-bin/dev/session_*',
        help='Glob path to session directories (supports *)'
    )
    parser.add_argument(
        '--output_dir_name',
        type=str,
        default='output',
        help='Name of the output directory at the data-bin level'
    )
    args = parser.parse_args()

    # Projekt-Root bestimmen
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    session_dir_arg = os.path.join(project_root, args.session_dir)

    # Alle Session-Dirs sammeln
    all_session_dirs = [p for p in glob.glob(session_dir_arg) if os.path.isdir(p)]

    print(f"Inferring {len(all_session_dirs)} sessions using NO model")

    engine = InferenceEngine()

    for session_dir in all_session_dirs:
        # Session-Name robust extrahieren
        session_name = os.path.basename(os.path.normpath(session_dir))

        # data-bin finden
        data_bin_dir = os.path.dirname(os.path.dirname(session_dir))
        output_base = os.path.join(data_bin_dir, args.output_dir_name)

        # Ziel: .../data-bin/output/session_xyz
        output_dir = os.path.join(output_base, session_name)

        print("Session dir:", session_dir)
        print("Output dir will be:", output_dir)

        os.makedirs(output_dir, exist_ok=True)
        print(f"Processing session {session_name}")

        engine.mcorec_session_infer(session_dir, output_dir)


if __name__ == '__main__':
    main()

