"""Evaluation metrics for RQ1: organizational structure extraction."""

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import normalized_mutual_info_score


def dominance_f1(
    ground_truth: pd.DataFrame,
    extracted_ranks: dict[str, float],
) -> dict[str, float]:
    """Pair-classification accuracy on the Agarwal dominance-pair task.

    For each ground-truth (superior, subordinate) pair, check whether
    extracted_ranks order them correctly. Pairs missing at least one node
    count as wrong (the model failed to recover the relation).

    The canonical metric is ``pair_accuracy = correct / total_GT_pairs``,
    reported alongside ``coverage`` (fraction of GT pairs evaluable). The
    older keys ``precision``, ``recall``, ``f1`` are preserved for
    backward compatibility but now reflect the honest accounting:

    * ``precision`` is ``1.0`` by construction — every prediction is on a
      GT pair, so there are no false positives in this task formulation.
    * ``recall`` equals ``pair_accuracy`` (correct / total).
    * ``f1`` equals ``pair_accuracy``. Earlier code reported
      ``2 * recall / (1 + recall)`` here, which inflated the number; that
      shape is no longer used.
    """
    correct = wrong_order = missing = 0

    for _, row in ground_truth.iterrows():
        sup = row["superior"]
        sub = row["subordinate"]

        if sup not in extracted_ranks or sub not in extracted_ranks:
            missing += 1
            continue

        if extracted_ranks[sup] > extracted_ranks[sub]:
            correct += 1
        else:
            wrong_order += 1

    total = correct + wrong_order + missing
    pair_accuracy = correct / total if total else 0.0
    coverage = (correct + wrong_order) / total if total else 0.0

    return {
        "pair_accuracy": pair_accuracy,
        "coverage": coverage,
        "correct": correct,
        "wrong_order": wrong_order,
        "missing": missing,
        "total_pairs": total,
        "precision": 1.0,
        "recall": pair_accuracy,
        "f1": pair_accuracy,
    }


def hierarchy_spearman(
    ground_truth_levels: dict[str, int],
    extracted_ranks: dict[str, float],
) -> float:
    """Compute Spearman correlation between ground truth hierarchy levels and extracted ranks.

    Returns Spearman rho correlation coefficient.
    """
    common = set(ground_truth_levels) & set(extracted_ranks)
    if len(common) < 3:
        return 0.0

    gt_values = [ground_truth_levels[k] for k in common]
    ex_values = [extracted_ranks[k] for k in common]

    rho, _ = spearmanr(gt_values, ex_values)
    return float(rho)


def community_nmi(
    true_labels: dict[str, int],
    predicted_labels: dict[str, int],
) -> float:
    """Compute Normalized Mutual Information between true and predicted communities.

    Returns NMI score (0.0 to 1.0).
    """
    common = sorted(set(true_labels) & set(predicted_labels))
    if len(common) < 2:
        return 0.0

    y_true = [true_labels[k] for k in common]
    y_pred = [predicted_labels[k] for k in common]

    return float(normalized_mutual_info_score(y_true, y_pred))
