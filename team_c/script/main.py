import os
import argparse
import json
import glob
import pickle

# Add src to path
os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from team_c.src.cluster.conv_spks import (
    get_speaker_activity_segments,
    # calculate_conversation_scores,
    cluster_speakers,
    get_clustering_f1_score
)

from team_c.src.cluster.cluster_tuning import (
    grid_search_agglomerative_clustering
)

from team_c.src.cluster.new_score_calc import (calculate_conversation_scores)
from team_c.src.cluster.kombiniertes_Cluster_tuning import (grid_search_full)

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

        return scores, clusters, speaker_segments

def read_true_cluster_labels(label_path):
    with open(os.path.join(label_path, "speaker_to_cluster.json"), "r") as f:
        label_data = json.load(f)
    return label_data

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
    ################# AB hier Code eingefügt für Grid-Search (DS)
    ### sessions = [
    ###    {"scores": scores1, "true_labels": [0,0,1,1,2,2]},
    ###    {"scores": scores2, "true_labels": [0,1,1,2,2,0]},
    ###    {"scores": scores3, "true_labels": [0,0,0,1,1,2]}
    ### ]
    ### mit scores: NxN numpy array of conversation scores
    ### und true_labels = values der speaker_to_cluster.json
    ###
    ###sessions2 = [
    ###    {
    ###        "speaker_segments": {
    ###             "spk_0": [[0,2],[5,7]],
    ###            "spk_1": [[1,3],[6,8]],
    ###            "spk_2": [[10,12]]
    ###        },
    ###        "true_labels": [0,0,1]
    ###    },
    ###    {
    ###        "speaker_segments": {
    ###            "spk_0": [[0,3],[7,8]],
    ###            "spk_1": [[1,4]],
    ###            "spk_2": [[9,11]]
    ###        },
    ###        "true_labels": [0,0,1]
    ###    }
    ### ]
    ### mit speaker_segments = dict of speakers timeslots when they talk
    ### und true_labels = values der speaker_to_cluster.json
    ###
    ###
    sessions_pkl_file = os.path.join(project_root,"data-bin","output","sessions.pkl")
    sessions2_pkl_file = os.path.join(project_root, "data-bin", "output", "sessions2.pkl")
    if os.path.exists(sessions_pkl_file) and os.path.exists(sessions2_pkl_file):
        with open(sessions_pkl_file, "rb") as f:
            sessions = pickle.load(f)
        with open(sessions2_pkl_file, "rb") as f:
            sessions2 = pickle.load(f)
    else:
        sessions = []
        sessions2 = []
        ########################## BIS hier Code eingefügt
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

            # engine.mcorec_session_infer(session_dir, output_dir)  ## auskommentiert für grid-Search (DS)

            ############################  AB hier Code für Grid-Search für gute Parameter des Cluster - Algorithmus eingefügt (DS)
            session_label_path = os.path.join(session_dir,'labels')
            session_true_cluster_dict = read_true_cluster_labels(session_label_path)
            session_true_cluster_labels = list(session_true_cluster_dict.values())
            session_scores, session_pred_clusters, session_speaker_segments = engine.mcorec_session_infer(session_dir, output_dir)

            sessions.append({
                "scores": session_scores,
                "true_labels": session_true_cluster_labels
            })

            sessions2.append({"speaker_segments": session_speaker_segments,
                "true_labels": session_true_cluster_labels})

            # Ordner anlegen (falls nicht vorhanden)
            os.makedirs(os.path.dirname(sessions_pkl_file), exist_ok=True)
            os.makedirs(os.path.dirname(sessions2_pkl_file), exist_ok=True)
            with open(sessions_pkl_file, 'wb') as f:
                pickle.dump(sessions, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(sessions2_pkl_file, 'wb') as f:
                pickle.dump(sessions2, f, protocol=pickle.HIGHEST_PROTOCOL)



    df_grid_search = grid_search_agglomerative_clustering(sessions)
    print(df_grid_search.sort_values(by="ARI_mean", ascending=False)[["threshold", "linkage", "ARI_mean", "Macro-F1_per_Speaker_mean", "Micro-F1_per_Speaker_mean"]])
    df_grid_search.to_csv("df_grid_search.csv", sep=";", index=True, encoding="utf-8-sig")

    ############# kombiniertes Cluster_tuning
    df_results = grid_search_full(
        sessions2,
        thresholds=[0.7, 0.8],
        linkages=["complete", "average"],
        tolerances=[0.0, 0.5],
        weight_options=[False, True],
        non_linears=[None, "sigmoid"]
    )

    # Ergebnisse sortieren nach ARI_mean
    df_results = df_results.sort_values("ARI_mean", ascending=False)
    print(df_results.sort_values(by="ARI_mean", ascending=False)[["threshold","linkage","tolerance","weight_by_length","non_linear","ARI_mean","Macro-F1_per_speaker_mean","Micro-F1_per_speaker_mean"]])
    df_results.to_csv("df_results.csv", sep=";", index=True, encoding="utf-8-sig")
    ############################  BIS hier Code für Grid-Search für gute Parameter des Cluster - Algorithmus eingefügt

if __name__ == '__main__':
    main()

