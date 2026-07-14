from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_confusion_matrix(metrics: Dict, output_path: str | Path):
    """Generate the final-system confusion matrix chart."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = metrics.get("labels", [])
    matrix = metrics.get("confusion_matrix", [])
    if not labels or not matrix:
        return None

    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Full System Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_module_comparison(reports, output_path: str | Path):
    """Plot average confidence per module across the evaluated patients."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for report in reports:
        for name, result in report.get("module_results", {}).items():
            if isinstance(result, dict) and "confidence" in result:
                rows.append({
                    "module": name,
                    "confidence": float(result["confidence"]),
                })
    if not rows:
        return None

    df = pd.DataFrame(rows)
    summary = df.groupby("module", as_index=False)["confidence"].mean()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=summary, x="module", y="confidence", hue="module",
                legend=False, palette="viridis")
    plt.ylim(0, 1)
    plt.title("Average Module Confidence")
    plt.xlabel("Module")
    plt.ylabel("Average confidence")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path
