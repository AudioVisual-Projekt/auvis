"""
Module: clustering_metrics
==========================
Pure mathematical functions to calculate clustering performance.
Migrated and cleaned up from legacy 'eval.py' and 'conv_spks.py'.
"""

import itertools
from typing import List, Dict, Tuple
from sklearn.metrics import adjusted_rand_score


def calculate_pairwise_metrics(true_labels: List[int], pred_labels: List[int]) -> Tuple[float, float, float]:
    """
    Computes Pairwise Precision, Recall, and F1 Score.

    Returns:
        Tuple[precision, recall, f1]
    """
    n = len(true_labels)
    # Edge case: Less than 2 items means no pairs possible.
    # We define this as perfect match if lists are identical.
    if n < 2:
        val = 1.0 if true_labels == pred_labels else 0.0
        return val, val, val

    pairs = list(itertools.combinations(range(n), 2))

    tp = 0
    fp = 0
    fn = 0

    for i, j in pairs:
        true_same = (true_labels[i] == true_labels[j])
        pred_same = (pred_labels[i] == pred_labels[j])

        if pred_same and true_same:
            tp += 1
        elif pred_same and not true_same:
            fp += 1
        elif not pred_same and true_same:
            fn += 1

    # Calculate Precision
    if (tp + fp) == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    # Calculate Recall
    if (tp + fn) == 0:
        recall = 0.0  # Should ideally be handled based on context (if no true pairs existed)
        # If there were NO pairs in GT (everyone alone), and we found NO pairs, Recall is technically 1.0
        # Check if GT had any pairs:
        gt_pairs = sum(1 for i, j in pairs if true_labels[i] == true_labels[j])
        if gt_pairs == 0:
            recall = 1.0
    else:
        recall = tp / (tp + fn)

    # Calculate F1
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def calculate_ari(true_labels: List[int], pred_labels: List[int]) -> float:
    """
    Wrapper for sklearn's Adjusted Rand Index.
    Measures similarity between two clusterings.
    1.0 = Perfect match.
    0.0 = Random labeling.
    """
    return adjusted_rand_score(true_labels, pred_labels)


def calculate_per_speaker_f1(true_labels: List[int], pred_labels: List[int]) -> Dict[int, float]:
    """
    Computes F1 score from the perspective of each speaker (One-vs-Rest).
    Useful to find out WHICH speaker was wrongly assigned.

    Returns:
        Dict mapping index (0 to N-1) to F1 score.
    """
    n = len(true_labels)
    scores = {}

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
            elif not pred_same and true_same:
                fn += 1

        if tp == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            val = precision + recall
            f1 = (2 * precision * recall / val) if val > 0 else 0.0

        scores[i] = f1

    return scores
