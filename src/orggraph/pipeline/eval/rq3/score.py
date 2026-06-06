"""Scoring for the RQ3 R1-vs-R2 pilot."""

from __future__ import annotations

import random
from statistics import mean

from orggraph.evaluation.text_metrics import normalize_answer


def recall_at_k_fuzzy(
    retrieved_bodies: list[str], source_email_str: str, threshold: float = 0.5
) -> float:
    """Return 1.0 iff any retrieved body has Jaccard token-overlap ≥ *threshold*
    with *source_email_str*, else 0.0.
    """
    src_tokens = set(normalize_answer(source_email_str).split())
    if not src_tokens or not retrieved_bodies:
        return 0.0
    for body in retrieved_bodies:
        body_tokens = set(normalize_answer(body).split())
        if not body_tokens:
            continue
        jaccard = len(src_tokens & body_tokens) / len(src_tokens | body_tokens)
        if jaccard >= threshold:
            return 1.0
    return 0.0


def paired_bootstrap(
    r1_scores: list[float],
    r2_scores: list[float],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """Paired bootstrap over (R2 - R1) per-question deltas.

    Returns (mean_delta, ci_low, ci_high, one_sided_p) for ``H1: R2 > R1``.
    The CI is the percentile interval at the requested ``alpha``.
    The one-sided p-value is the fraction of bootstrap resamples in which
    the resampled mean delta is ≤ 0.
    """
    if len(r1_scores) != len(r2_scores):
        raise ValueError("r1_scores and r2_scores must have equal length")
    if not r1_scores:
        raise ValueError("score lists must be non-empty")
    deltas = [b - a for a, b in zip(r1_scores, r2_scores)]
    observed_mean = mean(deltas)

    rng = random.Random(seed)
    n = len(deltas)
    boot_means: list[float] = []
    n_negative = 0
    n_zero = 0
    for _ in range(n_resamples):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        m = sum(sample) / n
        boot_means.append(m)
        if m < 0:
            n_negative += 1
        elif m == 0.0:
            n_zero += 1

    boot_means.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    ci_low = boot_means[max(0, lo_idx)]
    ci_high = boot_means[min(n_resamples - 1, hi_idx)]
    # Mid-p correction: count strict negatives + half the ties at 0
    p_one_sided = (n_negative + 0.5 * n_zero) / n_resamples
    return observed_mean, ci_low, ci_high, p_one_sided
