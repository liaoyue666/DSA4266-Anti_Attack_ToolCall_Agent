"""Call-level metrics and validation-only threshold selection."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix


def binary_metrics(labels: Iterable[int], probabilities: Iterable[float], threshold: float) -> dict:
    y_true = np.asarray(list(labels), dtype=int)
    y_score = np.asarray(list(probabilities), dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "rows": int(len(y_true)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy": float((tp + tn) / len(y_true)),
        "unauthorized_precision": float(precision),
        "unauthorized_recall": float(recall),
        "unauthorized_f1": float(f1),
        "false_block_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def select_threshold(
    labels: Iterable[int],
    probabilities: Iterable[float],
    *,
    minimum_recall: float = 0.80,
) -> tuple[float, dict]:
    """Prefer minimum false-block rate subject to recall, then higher F1."""

    labels = list(labels)
    probabilities = list(probabilities)
    candidates = [round(value / 100, 2) for value in range(5, 96)]
    scored = [binary_metrics(labels, probabilities, threshold) for threshold in candidates]
    feasible = [row for row in scored if row["unauthorized_recall"] >= minimum_recall]
    if feasible:
        best = min(
            feasible,
            key=lambda row: (
                row["false_block_rate"],
                -row["unauthorized_f1"],
                -row["threshold"],
            ),
        )
    else:
        best = max(
            scored,
            key=lambda row: (
                row["unauthorized_f1"],
                row["unauthorized_recall"],
                -row["false_block_rate"],
            ),
        )
    return best["threshold"], best
