import itertools
from typing import List, Tuple, Dict
from sklearn.metrics import adjusted_rand_score


def pairwise_f1_score(true_labels: List[int], pred_labels: List[int]) -> float:
    """
    Pairwise-F1 für Clustering:
    - Betrachte alle ungeordneten Paare (i, j)
    - Positiv, wenn i und j im selben Cluster sind
    """

    tp = fp = fn = 0

    for i, j in itertools.combinations(range(len(true_labels)), 2):
        true_same = (true_labels[i] == true_labels[j])
        pred_same = (pred_labels[i] == pred_labels[j])

        if pred_same and true_same:
            tp += 1
        elif pred_same and not true_same:
            fp += 1
        elif (not pred_same) and true_same:
            fn += 1

    # Randfall: Keine positiven Paare in GT und Prediction -> in Pairwise-Sicht perfekt
    if (tp + fp) == 0 and (tp + fn) == 0:
        return 1.0

    # Keine True Positives, aber es gibt positive Paare in GT oder Prediction -> F1 = 0
    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def pairwise_f1_score_per_speaker(true_labels: List[int], pred_labels: List[int]) -> Dict[int, float]:
    """
    Pairwise-F1 pro Sprecher i (one-vs-rest über Paare (i, j)).
    """
    n = len(true_labels)
    scores: Dict[int, float] = {}

    for i in range(n):
        tp = fp = fn = 0

        for j in range(n):
            if i == j:
                continue

            true_same = (true_labels[i] == true_labels[j])
            pred_same = (pred_labels[i] == pred_labels[j])

            if pred_same and true_same:
                tp += 1
            elif pred_same and not true_same:
                fp += 1
            elif (not pred_same) and true_same:
                fn += 1

        # Randfall für Sprecher i: keine positiven Beziehungen in GT und Prediction
        if (tp + fp) == 0 and (tp + fn) == 0:
            scores[i] = 1.0
            continue

        if tp == 0:
            scores[i] = 0.0
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores[i] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return scores


if __name__ == "__main__":
    examples: List[Tuple[List[int], List[int]]] = [
        ([0, 0, 1, 1], [0, 0, 2, 2]),
        ([0, 0, 1, 1], [1, 1, 0, 0]),
        ([0, 0, 1, 2], [0, 0, 1, 1]),
        ([0, 0, 0, 0], [0, 1, 2, 3]),
        ([0, 0, 1, 1], [0, 1, 0, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 0]),
        ([0, 0, 0, 0], [1, 1, 0, 0]),
        ([0, 0, 0, 0, 1, 2], [1, 1, 0, 0, 2, 2]),
        ([0, 0, 1, 1, 2, 2], [0, 0, 0, 1, 1, 1]),
    ]

    results = [
        (true, pred, pairwise_f1_score(true, pred), adjusted_rand_score(true, pred))
        for true, pred in examples
    ]
    for true, pred, f1, ari in results:
        print(f"True: {true}, Pred: {pred}, F1: {f1}, ARI: {ari}")

    for true, pred in examples:
        per_speaker_f1 = pairwise_f1_score_per_speaker(true, pred)
        print(f"True: {true}, Pred: {pred}, Per-Speaker F1: {per_speaker_f1}")
