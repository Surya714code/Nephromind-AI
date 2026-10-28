"""Binary classification metrics and diagnostic plot helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_binary(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "sensitivity": recall_score(y_true, predictions, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "brier_score": brier_score_loss(y_true, probabilities),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def save_validation_plots(results: dict[str, dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # ROC and PR plots share all models for direct comparison.
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(result["y_true"], result["probabilities"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={result['metrics']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Validation ROC curves")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(output_dir / "validation_roc_curve.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, result in results.items():
        precision, recall, _ = precision_recall_curve(result["y_true"], result["probabilities"])
        ax.plot(recall, precision, label=f"{name} (AP={result['metrics']['pr_auc']:.3f})")
    prevalence = next(iter(results.values()))["y_true"].mean()
    ax.axhline(prevalence, linestyle="--", color="grey", label=f"Positive prevalence ({prevalence:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Validation precision-recall curves")
    ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(output_dir / "validation_pr_curve.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, result) in zip(axes, results.items()):
        ConfusionMatrixDisplay(np.array(result["metrics"]["confusion_matrix"]), display_labels=["No", "Yes"]).plot(ax=ax, colorbar=False)
        ax.set_title(name)
    fig.suptitle("Validation confusion matrices", y=1.03)
    fig.tight_layout(); fig.savefig(output_dir / "validation_confusion_matrices.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, result in results.items():
        observed, predicted = calibration_curve(result["y_true"], result["probabilities"], n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction", title="Validation calibration curves")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(output_dir / "validation_calibration_curve.png", dpi=160); plt.close(fig)
