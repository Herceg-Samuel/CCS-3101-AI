from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """Compute the rubric metrics for system-level diagnoses."""
    labels = sorted(set(y_true) | set(y_pred))
    if not y_true:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "labels": labels,
            "confusion_matrix": [],
            "classification_report": {},
            "roc_auc": "not_applicable_for_multiclass_label_output",
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(
            y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(
            y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(
            y_true, y_pred, average="weighted", zero_division=0)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0,
            output_dict=True),
        "roc_auc": "not_applicable_for_multiclass_label_output",
    }


def save_metrics_csv(metrics: Dict, output_path: str | Path):
    """Save a one-row metrics summary for the final report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        key: value
        for key, value in metrics.items()
        if key in {"accuracy", "precision", "recall", "f1_score", "roc_auc"}
    }
    pd.DataFrame([summary]).to_csv(output_path, index=False)
