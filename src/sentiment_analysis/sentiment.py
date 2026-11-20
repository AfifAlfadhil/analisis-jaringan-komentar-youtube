"""Multi-annotator sentiment agreement (Fleiss' Kappa) and Majority Vote utilities."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

LABELS = ["positif", "netral", "negatif"]


def normalize_label(value: object) -> str | None:
    """Normalize raw LLM or manual label text into positif / netral / negatif."""
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    text = str(value).strip().lower()
    aliases = {
        "pos": "positif",
        "positive": "positif",
        "positivo": "positif",
        "1": "positif",
        "+1": "positif",
        "net": "netral",
        "neutral": "netral",
        "neutro": "netral",
        "0": "netral",
        "neg": "negatif",
        "negative": "negatif",
        "negativo": "negatif",
        "-1": "negatif",
    }
    return aliases.get(text, text if text in LABELS else None)


def interpret_landis_koch(kappa: float | None) -> str:
    """Interpret Kappa value using Landis & Koch (1977) benchmark scale."""
    if kappa is None or np.isnan(kappa):
        return "N/A"
    if kappa < 0.0:
        return "Poor"
    if kappa <= 0.20:
        return "Slight"
    if kappa <= 0.40:
        return "Fair"
    if kappa <= 0.60:
        return "Moderate"
    if kappa <= 0.80:
        return "Substantial"
    return "Almost Perfect"


def _build_count_matrix(
    ratings_df: pd.DataFrame,
    annotator_cols: Sequence[str],
    labels: Sequence[str] = LABELS,
) -> np.ndarray:
    """Build N x k count matrix where n_ij = number of raters assigning item i to label j."""
    valid = ratings_df[list(annotator_cols)].dropna()
    if valid.empty:
        return np.zeros((0, len(labels)), dtype=float)
    matrix = np.zeros((len(valid), len(labels)), dtype=float)
    values = valid.to_numpy()
    for j, label in enumerate(labels):
        matrix[:, j] = (values == label).sum(axis=1)
    n_raters = len(annotator_cols)
    complete_mask = matrix.sum(axis=1) == n_raters
    return matrix[complete_mask]


def fleiss_kappa(
    ratings_df: pd.DataFrame,
    annotator_cols: Sequence[str],
    labels: Sequence[str] = LABELS,
) -> float:
    """Compute Fleiss' Kappa (Fleiss, 1971) for >= 2 categorical raters."""
    matrix = _build_count_matrix(ratings_df, annotator_cols, labels)
    n_items = matrix.shape[0]
    n_raters = len(annotator_cols)
    if n_items == 0 or n_raters < 2:
        return float("nan")

    p_i = (np.sum(matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(np.mean(p_i))

    p_j = np.sum(matrix, axis=0) / (n_items * n_raters)
    p_e = float(np.sum(p_j ** 2))

    if np.isclose(1.0 - p_e, 0.0):
        return 1.0 if np.isclose(p_bar, 1.0) else 0.0
    return float((p_bar - p_e) / (1.0 - p_e))


def pairwise_percent_agreement(
    ratings_df: pd.DataFrame,
    annotator_cols: Sequence[str],
    labels: Sequence[str] = LABELS,
) -> float:
    """Average pairwise percent agreement (P_bar) across all annotator pairs."""
    matrix = _build_count_matrix(ratings_df, annotator_cols, labels)
    n_items = matrix.shape[0]
    n_raters = len(annotator_cols)
    if n_items == 0 or n_raters < 2:
        return float("nan")
    p_i = (np.sum(matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    return float(np.mean(p_i))


def majority_vote(
    row_labels: Iterable[str | None],
    tie_breaker: str = "netral",
) -> tuple[str, bool]:
    """Resolve final label via Majority Vote (seri 1-1-1 -> tie_breaker)."""
    valid = [lbl for lbl in row_labels if lbl in LABELS]
    if not valid:
        return tie_breaker, True
    counts = Counter(valid)
    top_label, top_count = counts.most_common(1)[0]
    runners = [lbl for lbl, cnt in counts.items() if cnt == top_count]
    if len(runners) == 1:
        return top_label, False
    return tie_breaker, True


def resolve_majority_vote(
    df: pd.DataFrame,
    annotator_cols: Sequence[str],
    tie_breaker: str = "netral",
) -> tuple[pd.Series, pd.Series]:
    """Apply majority_vote row-wise and return (sentiment_series, is_tie_series)."""
    results = [
        majority_vote(row, tie_breaker=tie_breaker)
        for row in df[list(annotator_cols)].itertuples(index=False, name=None)
    ]
    labels = pd.Series([r[0] for r in results], index=df.index, dtype="object")
    ties = pd.Series([r[1] for r in results], index=df.index, dtype="bool")
    return labels, ties


def compute_agreement_summary(
    df: pd.DataFrame,
    annotator_cols: Sequence[str],
    target_kappa: float = 0.85,
) -> dict:
    """Compute Fleiss' Kappa, percent agreement, and Landis & Koch interpretation."""
    valid = df[list(annotator_cols)].dropna()
    valid = valid[valid.apply(lambda r: all(x in LABELS for x in r), axis=1)]
    n_comments = len(valid)
    if n_comments == 0:
        return {
            "n_comments": 0,
            "fleiss_kappa": float("nan"),
            "percent_agreement": float("nan"),
            "interpretation": "N/A",
            "meets_target_85": False,
        }

    kappa = fleiss_kappa(valid, annotator_cols)
    p_agree = pairwise_percent_agreement(valid, annotator_cols)

    return {
        "n_comments": int(n_comments),
        "fleiss_kappa": round(kappa, 4),
        "percent_agreement": round(p_agree, 4),
        "interpretation": interpret_landis_koch(kappa),
        "meets_target_85": bool(kappa >= target_kappa),
    }
