import os
import argparse
import json
import glob

from numpy import ndarray
from sympy.physics.units import metric_ton

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
        # load session metadata
        with open(os.path.join(session_dir, "metadata.json"), "r", encoding="utf-8") as f:
            # save content of json file into dictionary ("metadata")
            metadata = json.load(f)

        # process speaker clustering for each speaker
        speaker_segments = {}
        for speaker_name, speaker_data in metadata.items():
            list_tracks_asd = []

            # get all track asd files in appended list (from "metadata.json")
            # there can be multiple track files per speaker - todo: Why?
            for track in speaker_data['central']['crops']:
                list_tracks_asd.append(
                    os.path.join(session_dir, track['asd']))

            # take start and end time from metadata (whistle sound)
            uem_start = speaker_data['central']['uem']['start']
            uem_end = speaker_data['central']['uem']['end']

            # get all segments where the speaker is speaking (regarding asd scores)
            # => using function from conv_spks.py file
            speaker_activity_segments = get_speaker_activity_segments(list_tracks_asd, uem_start, uem_end)

            # save speaker activity segments as dictionary with speaker name as the key (value is a list of the segments)
            speaker_segments[speaker_name] = speaker_activity_segments

        # NxN numpy array of conversation scores
        scores = calculate_conversation_scores(speaker_segments)
        # dictionary - mapping cluster IDs to lists of speaker IDs
        clusters = cluster_speakers(scores=scores,
                                    speaker_ids=list(speaker_segments.keys()))
        output_clusters_file = os.path.join(output_dir,
                                            "speaker_to_cluster.json")
        with open(output_clusters_file, "w") as f:
            json.dump(clusters, f, indent=4)


def main():
    # dev for using dev session data & train for using train session data
    dataset = "dev"  # dev | train

    session_dir = f"data-bin\\{dataset}\\session_*"
    output_dir = f"_output\\{dataset}"

    # determine project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    session_dir_arg = os.path.join(project_root, session_dir)

    # gather all session directories
    all_session_dirs = [p for p in glob.glob(session_dir_arg) if os.path.isdir(p)]

    print(f"Inferring {len(all_session_dirs)} sessions using NO model")

    # create an instance of InferenceEngine() where each session can be processed
    engine = InferenceEngine()

    # loop over all session dirs
    for session_dir in all_session_dirs:
        # get session name, e.g. "session_132"
        session_name = os.path.basename(os.path.normpath(session_dir))

        # data-bin finden
        data_bin_dir = os.path.dirname(os.path.dirname(session_dir))
        output_base = os.path.join(data_bin_dir, output_dir)

        # Ziel: .../data-bin/output/session_xyz
        output_dir = os.path.join(output_base, session_name)

        print("Session dir:", session_dir)
        print("Output dir will be:", output_dir)

        os.makedirs(output_dir, exist_ok=True)
        print(f"Processing session {session_name}")

        engine.mcorec_session_infer(session_dir, output_dir)

        print(f"Session {session_name} was processed.")


if __name__ == '__main__':
    main()

