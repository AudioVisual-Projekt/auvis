import os
import argparse
import json
import glob
import pandas as pd

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


    def mcorec_session_infer(self, session_dir, output_dir, threshold: float = 0.7):
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
        clusters = cluster_speakers(scores, list(speaker_segments.keys()), threshold=threshold)
        output_clusters_file = os.path.join(output_dir,
                                            "speaker_to_cluster.json")
        with open(output_clusters_file, "w") as f:
            json.dump(clusters, f, indent=4)
        return scores, clusters, speaker_segments

def read_cluster_labels_from_json(label_path):
    with open(os.path.join(label_path, "speaker_to_cluster.json"), "r") as f:
        label_data = json.load(f)
    return label_data

def inference(cluster_threshold: float = 0.7) -> pd.DataFrame:
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
    result = []
    for session_dir in all_session_dirs:
        # Session-Name robust extrahieren
        session_name = os.path.basename(os.path.normpath(session_dir))

        # data-bin finden
        data_bin_dir = os.path.dirname(os.path.dirname(session_dir))
        output_base = os.path.join(data_bin_dir, args.output_dir_name)
        label_dir = os.path.join(session_dir,"labels")

        # Ziel: .../data-bin/output/session_xyz
        output_dir = os.path.join(output_base, session_name)

        print("Session dir:", session_dir)
        print("Output dir will be:", output_dir)

        os.makedirs(output_dir, exist_ok=True)
        print(f"Processing session {session_name}")

        session_true_clusters = read_cluster_labels_from_json(label_dir)
        session_scores, session_pred_clusters, session_speaker_segments = engine.mcorec_session_infer(session_dir, output_dir, threshold=cluster_threshold)
        result.append({"session_name": session_name,
                       "true_clusters": session_true_clusters,
                       "pred_clusters": session_pred_clusters,
                       "session_scores": session_scores,
                       "session_speaker_segments": session_speaker_segments
                       })

    return pd.DataFrame(result)
if __name__ == '__main__':
    result = inference()
    print(result.iloc[0]['pred_clusters'])

