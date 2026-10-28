"""
NephroMind AI — Step 4: Evaluate DNN
======================================
Evaluates the trained DNN on held-out test data.
- Threshold optimized on VALIDATION data, then LOCKED
- Test set evaluated exactly ONCE with locked threshold
- All metrics computed from actual predictions (no fabrication)

Saves:
  models/metrics.json          — full evaluation results
  models/dnn_threshold.json    — locked threshold
  models/figures/              — ROC, PR, Confusion Matrix plots

Also compares DNN vs Gradient Boosting (baseline).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    brier_score_loss, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FIGS_DIR   = ROOT / "models" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")  # evaluate on CPU for reproducibility

# ── Load model architecture ───────────────────────────────────────────────────
# Re-define architecture (same as training script) for loading
class NephroMindDNN(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.30),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.25),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

print("=" * 60)
print("NephroMind AI — DNN Evaluation")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────────────
X_val  = np.load(MODELS_DIR / "X_val.npy")
X_test = np.load(MODELS_DIR / "X_test.npy")
y_val  = np.load(MODELS_DIR / "y_val.npy")
y_test = np.load(MODELS_DIR / "y_test.npy")

with open(MODELS_DIR / "nephromind_features.json") as f:
    features_meta = json.load(f)
feature_names = features_meta["feature_names"]

# ── Load DNN checkpoint ───────────────────────────────────────────────────────
checkpoint = torch.load(MODELS_DIR / "nephromind_dnn.pt", map_location=DEVICE,
                         weights_only=False)
n_features = checkpoint["n_features"]
model = NephroMindDNN(n_features=n_features)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"\nLoaded DNN checkpoint: best_epoch={checkpoint['best_epoch']}  "
      f"val_loss={checkpoint['best_val_loss']:.4f}")

# ── Get probabilities ─────────────────────────────────────────────────────────
def get_probs(X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()

val_probs  = get_probs(X_val)
test_probs = get_probs(X_test)

# ── Helper: compute all metrics ───────────────────────────────────────────────
def compute_metrics(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy":    round(accuracy_score(y_true, preds), 6),
        "precision":   round(precision_score(y_true, preds, zero_division=0), 6),
        "recall":      round(recall_score(y_true, preds, zero_division=0), 6),
        "f1":          round(f1_score(y_true, preds, zero_division=0), 6),
        "roc_auc":     round(roc_auc_score(y_true, probs), 6),
        "pr_auc":      round(average_precision_score(y_true, probs), 6),
        "sensitivity": round(recall_score(y_true, preds, zero_division=0), 6),
        "specificity": round(tn / (tn + fp) if (tn + fp) else 0.0, 6),
        "brier_score": round(brier_score_loss(y_true, probs), 6),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

# ── Step 1: Find best threshold on VALIDATION data (Youden's J = Sensitivity + Specificity - 1) ──
print("\nSearching for best threshold on VALIDATION data (optimize Youden's J = Sensitivity + Specificity - 1)...")
best_t, best_j = 0.5, -1.0
for t in np.linspace(0.05, 0.95, 91):
    preds = (val_probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    j = sens + spec - 1.0
    if j > best_j:
        best_j = j
        best_t = float(t)

print(f"  Best Youden's J: {best_j:.4f} at threshold={best_t:.3f}")
print(f"  Validation metrics at this threshold:")
val_at_best = compute_metrics(y_val, val_probs, threshold=best_t)
for k in ("sensitivity","specificity","f1","roc_auc"):
    print(f"    {k:<15}: {val_at_best[k]}")
print(f"  → Locking threshold at {best_t:.3f}")

# Save threshold
with open(MODELS_DIR / "dnn_threshold.json", "w") as f:
    json.dump({"locked_threshold": best_t,
               "optimized_on": "validation_Youden_J",
               "val_youden_j_at_threshold": round(best_j, 6)}, f, indent=2)

# ── Step 2: Evaluate on TEST set (one time, locked threshold) ─────────────────
val_metrics_default = compute_metrics(y_val, val_probs, threshold=0.5)
val_metrics_tuned   = compute_metrics(y_val, val_probs, threshold=best_t)
test_metrics        = compute_metrics(y_test, test_probs, threshold=best_t)

print(f"\n{'='*60}")
print(f"VALIDATION metrics (threshold=0.5):")
for k, v in val_metrics_default.items():
    if k != "confusion_matrix":
        print(f"  {k:<15}: {v}")
print(f"\nVALIDATION metrics (threshold={best_t:.3f}):")
for k, v in val_metrics_tuned.items():
    if k != "confusion_matrix":
        print(f"  {k:<15}: {v}")
print(f"\nTEST metrics (locked threshold={best_t:.3f}) — evaluated ONCE:")
for k, v in test_metrics.items():
    if k != "confusion_matrix":
        print(f"  {k:<15}: {v}")
cm = test_metrics["confusion_matrix"]
print(f"  Confusion matrix: TN={cm['tn']}  FP={cm['fp']}  FN={cm['fn']}  TP={cm['tp']}")

# ── Step 3: Load GB baseline results for comparison ───────────────────────────
baseline_summary_path = ROOT / "ml" / "reports" / "pipeline_summary.json"
gb_baseline = {}
if baseline_summary_path.exists():
    with open(baseline_summary_path) as f:
        summary = json.load(f)
    gb_baseline = summary.get("test_metrics_tuned", {})
    gb_baseline["model"] = "Gradient Boosting (Tuned) — Baseline"
    print(f"\n{'='*60}")
    print(f"BASELINE (Gradient Boosting, test set):")
    for k, v in gb_baseline.items():
        if k not in ("confusion_matrix", "model"):
            print(f"  {k:<15}: {v}")

# ── Step 4: Save metrics.json ─────────────────────────────────────────────────
metrics_output = {
    "model": "NephroMind DNN (PyTorch MLP)",
    "model_version": "nephromind_dnn_v1",
    "feature_vector": feature_names,
    "n_features": n_features,
    "locked_threshold": best_t,
    "threshold_optimized_on": "validation_F1",
    "dnn_val_metrics_default_threshold": val_metrics_default,
    "dnn_val_metrics_tuned_threshold": val_metrics_tuned,
    "dnn_test_metrics": test_metrics,
    "baseline_gb_test_metrics": gb_baseline,
    "class_distribution": {
        "val_positive": int(y_val.sum()),
        "val_total": int(len(y_val)),
        "test_positive": int(y_test.sum()),
        "test_total": int(len(y_test)),
    },
    "disclaimer": (
        "PROTOTYPE ONLY. This model calculates a risk score from physiological "
        "biomarkers. It is NOT a clinical diagnostic tool and has NOT been clinically validated."
    )
}
with open(MODELS_DIR / "metrics.json", "w") as f:
    json.dump(metrics_output, f, indent=2)
print(f"\nSaved: models/metrics.json")

# ── Step 5: Plots ─────────────────────────────────────────────────────────────
STYLE = {"figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
         "axes.edgecolor": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
         "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
         "text.color": "#e0e0e0", "grid.color": "#2a2a4a", "grid.alpha": 0.5}

def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {path.name}")

# ROC curve
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=STYLE["figure.facecolor"])
    ax.set_facecolor(STYLE["axes.facecolor"])
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    auc = test_metrics["roc_auc"]
    ax.plot(fpr, tpr, color="#e040fb", lw=2, label=f"DNN (AUC={auc:.3f})")
    if gb_baseline:
        gb_auc = gb_baseline.get("roc_auc", None)
        if gb_auc:
            ax.axhline(0, color="none")
            ax.plot([0], [0], color="#40c4ff", lw=2,
                    label=f"GB Baseline (AUC={gb_auc:.3f})")
    ax.plot([0,1],[0,1],"--",color="grey",lw=1,label="Chance")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve — NephroMind DNN vs GB Baseline")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, FIGS_DIR / "dnn_roc_curve.png")

# PR curve
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=STYLE["figure.facecolor"])
    ax.set_facecolor(STYLE["axes.facecolor"])
    prec, rec, _ = precision_recall_curve(y_test, test_probs)
    pr_auc = test_metrics["pr_auc"]
    ax.plot(rec, prec, color="#e040fb", lw=2, label=f"DNN (AP={pr_auc:.3f})")
    prevalence = y_test.mean()
    ax.axhline(prevalence, linestyle="--", color="grey", lw=1,
               label=f"Prevalence ({prevalence:.2f})")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Precision-Recall Curve — NephroMind DNN")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, FIGS_DIR / "dnn_pr_curve.png")

# Confusion matrix
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=STYLE["figure.facecolor"])
    ax.set_facecolor(STYLE["axes.facecolor"])
    preds_test = (test_probs >= best_t).astype(int)
    cm_arr = confusion_matrix(y_test, preds_test, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm_arr, display_labels=["No Disease (0)", "Kidney Disease (1)"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — DNN (threshold={best_t:.3f})",
                 color="#e0e0e0")
    ax.set_facecolor(STYLE["axes.facecolor"])
    fig.tight_layout()
    save_fig(fig, FIGS_DIR / "dnn_confusion_matrix.png")

# Training loss curve
history_path = MODELS_DIR / "training_history.json"
if history_path.exists():
    with open(history_path) as f:
        hist = json.load(f)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=STYLE["figure.facecolor"])
        ax.set_facecolor(STYLE["axes.facecolor"])
        epochs = list(range(1, len(hist["train_loss"]) + 1))
        ax.plot(epochs, hist["train_loss"], color="#e040fb", lw=2, label="Train Loss")
        ax.plot(epochs, hist["val_loss"],   color="#40c4ff", lw=2, label="Val Loss")
        be = hist["best_epoch"]
        ax.axvline(be, color="#69f0ae", linestyle="--", lw=1,
                   label=f"Best epoch ({be})")
        ax.set(xlabel="Epoch", ylabel="BCE Loss (weighted)",
               title="DNN Training History")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_fig(fig, FIGS_DIR / "dnn_training_history.png")

print(f"\n✓ EVALUATION COMPLETE")
print(f"\n{'='*60}")
print(f"FINAL SUMMARY (Test Set, threshold={best_t:.3f}):")
print(f"  Accuracy   : {test_metrics['accuracy']:.4f}")
print(f"  Precision  : {test_metrics['precision']:.4f}")
print(f"  Recall     : {test_metrics['recall']:.4f}")
print(f"  Specificity: {test_metrics['specificity']:.4f}")
print(f"  F1-score   : {test_metrics['f1']:.4f}")
print(f"  ROC-AUC    : {test_metrics['roc_auc']:.4f}")
print(f"  PR-AUC     : {test_metrics['pr_auc']:.4f}")
if gb_baseline:
    print(f"\nBaseline (GB, threshold={gb_baseline.get('threshold','?')}):")
    print(f"  ROC-AUC    : {gb_baseline.get('roc_auc','N/A')}")
    print(f"  F1-score   : {gb_baseline.get('f1','N/A')}")
