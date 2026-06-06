"""Unsupervised hierarchy extraction from communication network centrality."""

import numpy as np
import pandas as pd


def assign_hierarchy_tiers(
    centrality: pd.DataFrame,
    n_tiers: int = 5,
    weights: dict[str, float] | None = None,
    invert: set[str] | None = None,
    log_transform: set[str] | None = None,
) -> pd.DataFrame:
    """Assign hierarchy tiers based on composite score of network features.

    Combines centrality metrics and communication pattern features into a
    single composite score, then assigns tiers using quantile binning.

    Args:
        centrality: DataFrame with a ``node`` column plus one column per
            feature listed in *weights*.
        n_tiers: Number of hierarchy tiers (default 5).
        weights: Feature-name -> weight mapping. Defaults to the original
            three centrality features with equal weight.
        invert: Set of column names where *lower* raw values indicate
            *higher* rank.  These columns are flipped (1 - norm) after
            min-max normalisation so that the composite score treats them
            consistently (higher = more senior).
        log_transform: Set of column names to apply log1p before
            normalisation.  Useful for highly skewed features like
            cc_frequency or in_degree.

    Returns:
        DataFrame with added columns: composite_score, tier (0=lowest,
        n_tiers-1=highest).
    """
    if weights is None:
        weights = {"pagerank": 1.0, "betweenness": 1.0, "in_degree": 1.0}
    if invert is None:
        invert = set()
    if log_transform is None:
        log_transform = set()

    df = centrality.copy()

    # Optional log1p transform for skewed features
    for col in log_transform:
        if col in weights:
            df[col] = np.log1p(df[col])

    # Min-max normalize each metric to [0, 1]
    for col in weights:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max > col_min:
            df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[f"{col}_norm"] = 0.0
        # Invert if lower raw value means higher hierarchy
        if col in invert:
            df[f"{col}_norm"] = 1.0 - df[f"{col}_norm"]

    # Composite score as weighted sum of normalized metrics
    df["composite_score"] = sum(
        w * df[f"{col}_norm"] for col, w in weights.items()
    )

    # Assign tiers using quantile binning
    df["tier"] = pd.qcut(
        df["composite_score"],
        q=n_tiers,
        labels=range(n_tiers),
        duplicates="drop",
    ).astype(int)

    # Drop intermediate norm columns
    norm_cols = [f"{col}_norm" for col in weights]
    df = df.drop(columns=norm_cols)

    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)
