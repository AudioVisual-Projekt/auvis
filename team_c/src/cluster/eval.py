import itertools
from typing import List, Tuple, Dict
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import matplotlib.pyplot as plt


def pairwise_f1_score(true_labels: List[int], pred_labels: List[int]) -> float:
    """
    Compute the pairwise F1 score for clustering evaluation.
    
    Args:
        true_labels (List[int]): Ground truth cluster labels.
        pred_labels (List[int]): Predicted cluster labels.
    
    Returns:
        float: Pairwise F1 score.
    """
    # Generate all unique unordered pairs of indices
    pairs = list(itertools.combinations(range(len(true_labels)), 2))
    
    # Initialize counts
    tp = fp = fn = 0
    
    for i, j in pairs:
        # True same-cluster?
        true_same = (true_labels[i] == true_labels[j])
        # Predicted same-cluster?
        pred_same = (pred_labels[i] == pred_labels[j])
        
        if pred_same and true_same:
            tp += 1
        elif pred_same and not true_same:
            fp += 1
        elif not pred_same and true_same:
            fn += 1
        # True negatives (not same in both) are not used in F1
    # print(tp, fp, fn)
    # Handle edge cases
    if tp == 0:
        return 0.0
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1

def pairwise_f1_score_per_speaker(true_labels: List[int], pred_labels: List[int]) -> Dict[int, float]:
    """
    Compute the pairwise F1 score for each speaker (one-vs-rest style) in clustering evaluation.
    
    Args:
        true_labels (List[int]): Ground truth cluster labels.
        pred_labels (List[int]): Predicted cluster labels.
    
    Returns:
        Dict[int, float]: Mapping from speaker index to their pairwise F1 score.
    """
    n = len(true_labels)
    scores = {}

    for i in range(n):
        tp = fp = fn = 0
        for j in range(n):
            if i == j:
                continue

            # True and predicted same-cluster relationships between i and j
            true_same = (true_labels[i] == true_labels[j])
            pred_same = (pred_labels[i] == pred_labels[j])

            if pred_same and true_same:
                tp += 1
            elif pred_same and not true_same:
                fp += 1
            elif not pred_same and true_same:
                fn += 1

        # Compute F1 for this speaker
        if tp == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        scores[i] = f1

    return scores




def plot_and_save_clustered_segs(all_session_names, all_spk_segs, all_true_clusters, all_pred_clusters):
    '''
    Plottet die Sprechersegmente auf der Zeitachse,
    Farbcodierung der Sprecher gemäß Clusterzugehörigkeit
    speichert die Plots in Dateien

    Args:
        (werden übernommen aus der inference:  inference_result = inference() )

        all_session_names
        all_session_names ist eine Series mit Index = Session und Value = session_name

        all_spk_segs = inference_result['session_speaker_segments']
        all_spk_segs ist eine Series mit Index = Session und Value = dict(speaker -> List of Lists mit jeweils Start und Ende des Sprechsegments)
        z.B. {'spk_0': [[3.65, 10.21], [10.25, 17.17], [18.37, 27.61], [27.65, 36.89],'spk_0': [[...]...]...}

        all_true_clusters = inference_result['true_clusters']
        all_true_clusters ist eine Series mit Index = session und Value = dict(speaker -> clustergroup)

        all_pred_clusters = inference_result['pred_clusters']
        all_pred_clusters ist eine Series mit Index = session und Value = dict(speaker -> clustergroup)

    Returns:
        None
    '''

    session_names = dict(zip(all_spk_segs.index, all_session_names))

    # bringe die all_spk_segs Series in einen DataFrame all_spk_segs_df der Form:
    #         session     speaker     start       end
    # 0           0       spk_0       3.651       10.2111
    # 1           0       spk_0       10.251      17.1711
    # 2           0       spk_0       18.371      27.6111
    # 3           0       spk_0       27.651      36.8911
    # 4           0       spk_0       36.931      46.1711
    rows = []
    for idx, seg_dict in all_spk_segs.items():  # idx = Index der Series all_spk_segs
        for spk, segs in seg_dict.items():
            for seg in segs:
                rows.append({
                    "session": idx,
                    "speaker": spk,
                    "start": seg[0],
                    "end": seg[1]
                })
    all_spk_segs_df = pd.DataFrame(rows)

    for session_id, seg_df in all_spk_segs_df.groupby("session"):
        # Clusterzuordnung für diese Session holen (true und predicted)
        # group_map = Clusterzuordnung: Sprecher -> Clustergruppe
        group_map_true = all_true_clusters.loc[session_id]
        group_map_pred = all_pred_clusters.loc[session_id]

        # Sprecherliste
        speakers = seg_df['speaker'].unique()

        # Colormap auswählen
        cmap_true = plt.get_cmap('tab10', len(set(group_map_true.values())))
        cmap_pred = plt.get_cmap('tab10', len(set(group_map_pred.values())))

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        bar_height = 0.4  # Höhe der Balken (kleiner = schmaler)

        # --- Erster Plot ---
        ax = axes[0]
        for i, spk in enumerate(speakers):
            segs = seg_df[seg_df['speaker'] == spk]
            intervals = [(row['start'], row['end'] - row['start']) for _, row in segs.iterrows()]

            # Farbe nach Clusterzuordnung
            group_id = group_map_true.get(spk, 0)  # fallback: Gruppe 0
            color = cmap_true(group_id)

            # Mittelpunkt um i, Höhe = bar_height
            ax.broken_barh(intervals, (i - bar_height / 2, bar_height), facecolors=color)

        # Y-Achse
        ax.set_yticks(range(len(speakers)))
        ax.set_yticklabels(speakers)

        # Labels & Titel
        ax.set_ylabel("Sprecher")
        ax.set_title(f"Farbkodierung nach true Clusters")

        # --- Zweiter Plot ---
        ax = axes[1]
        for i, spk in enumerate(speakers):
            segs = seg_df[seg_df['speaker'] == spk]
            intervals = [(row['start'], row['end'] - row['start']) for _, row in segs.iterrows()]
            group_id = group_map_pred.get(spk, 0)
            color = cmap_pred(group_id)
            ax.broken_barh(intervals, (i - bar_height / 2, bar_height), facecolors=color)

        ax.set_yticks(range(len(speakers)))
        ax.set_yticklabels(speakers)
        ax.set_xlabel("Zeit (s)")
        ax.set_ylabel("Sprecher")
        ax.set_title(f"Farbkodierung nach predicted Clusters")

        # Gemeinsamer Titel für den ganzen Plot
        fig.suptitle(f"Sprecher-Segmente mit zwei verschiedenen Clusterzuordnungen {session_names[session_id]}",
                     fontsize=14, weight="bold")

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Platz für den gemeinsamen Titel lassen

        # # --- lokal speichern ---
        plt.savefig(f"plot_case_{session_id}.png", dpi=150)
        plt.close(fig)  # Speicher freigeben

def run_inference(cluster_threshold: float = 0.7): # zur Vermeidung eines zirkulären Importerrors
    from team_c.script.main import inference
    return inference(cluster_threshold)


if __name__ == "__main__":
    # Example usage
    examples: List[Tuple[List[int], List[int]]] = [
        ([0, 0, 1, 1], [0, 0, 2, 2]),
        ([0, 0, 1, 1], [1, 1, 0, 0]),
        ([0, 0, 1, 2], [0, 0, 1, 1]),
        ([0, 0, 0, 0], [0, 1, 2, 3]),
        ([0, 0, 1, 1], [0, 1, 0, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 0]),
        ([0, 0, 0, 0], [1, 1, 0, 0]),
        ([0, 0, 0, 0, 1, 2], [1, 1, 0, 0, 2, 2]),
        ([0, 0, 1, 1, 2, 2], [0, 0, 0, 1, 1, 1])
    ]

    # Compute and display results
    results = [(true, pred, pairwise_f1_score(true, pred), adjusted_rand_score(true, pred)) for true, pred in examples]
    for true, pred, f1, ari in results:
        print(f"True: {true}, Pred: {pred}, F1: {f1}, ARI: {ari}")

    # Compute per-speaker F1 scores
    for true, pred in examples:
        per_speaker_f1 = pairwise_f1_score_per_speaker(true, pred)
        print(f"True: {true}, Pred: {pred}, Per-Speaker F1: {per_speaker_f1}")

    # --- Plot Cluster results predicted vs true ---

    # Get inference_result data:
    inference_result = run_inference()

    # Get session names from inference_result data:
    all_session_names = inference_result['session_name']

    # Get speaker segments from inference_result data:
    all_spk_segs = inference_result['session_speaker_segments']

    # Get true cluster labels from inference_result data:
    all_true_clusters = inference_result['true_clusters']

    # Get predictid cluster labels from inference_result data:
    all_pred_clusters = inference_result['pred_clusters']

    plot_and_save_clustered_segs(all_session_names, all_spk_segs, all_true_clusters, all_pred_clusters)