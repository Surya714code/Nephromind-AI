"""
NephroMind AI — Full ML Pipeline
=================================
End-to-end pipeline: ingestion → preprocessing → train/val/test split →
baseline + model training → evaluation → hyperparameter tuning → final model
selection → calibration → SHAP explainability → artifact saving → reports.

Scientific limitation: This model performs CURRENT KIDNEY-DISEASE STATUS
CLASSIFICATION. It does NOT predict future DKD progression or clinical
outcomes and has NOT been clinically validated.
"""
from __future__ import annotations

import json
import os
import random
import sys
import traceback
import warnings
from pathlib import Path
from datetime import datetime

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Resolve project root ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.data.ingestion import load_patient_records, mark_documented_invalid_blood_pressure
from ml.preprocessing.pipeline import build_preprocessor

CONFIG_PATH = ROOT / "ml" / "configs" / "full_pipeline.json"

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    print("[INFO] XGBoost available.")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[INFO] XGBoost not available — skipping XGBoost model.")

try:
    import shap
    SHAP_AVAILABLE = True
    print("[INFO] SHAP available.")
except ImportError:
    SHAP_AVAILABLE = False
    print("[INFO] SHAP not available — explainability will be skipped.")

import seaborn as sns

# ── Module-level calibration wrapper (must be at module level to be picklable) ─
class IsotonicWrapper:
    """Lightweight wrapper applying isotonic calibration to a fitted pipeline."""
    def __init__(self, base, calibrator):
        self.base = base
        self.calibrator = calibrator
    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1]
        cal = np.clip(self.calibrator.predict(raw), 0, 1)
        return np.column_stack([1 - cal, cal])
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
    # Expose named_steps for compatibility if needed
    @property
    def named_steps(self):
        return self.base.named_steps



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def evaluate_binary(y_true: np.ndarray, probabilities: np.ndarray,
                    threshold: float = 0.5) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": round(accuracy_score(y_true, predictions), 6),
        "precision": round(precision_score(y_true, predictions, zero_division=0), 6),
        "recall": round(recall_score(y_true, predictions, zero_division=0), 6),
        "f1": round(f1_score(y_true, predictions, zero_division=0), 6),
        "roc_auc": round(roc_auc_score(y_true, probabilities), 6),
        "pr_auc": round(average_precision_score(y_true, probabilities), 6),
        "sensitivity": round(recall_score(y_true, predictions, zero_division=0), 6),
        "specificity": round(tn / (tn + fp) if (tn + fp) else 0.0, 6),
        "brier_score": round(brier_score_loss(y_true, probabilities), 6),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray,
                        optimize: str = "f1") -> float:
    """Find probability threshold maximising F1 on provided data."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        preds = (probabilities >= t).astype(int)
        if optimize == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        elif optimize == "balanced":
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) else 0
            spec = tn / (tn + fp) if (tn + fp) else 0
            score = (sens + spec) / 2
        else:
            score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = t
    return float(best_t)


# ── Plot helpers ──────────────────────────────────────────────────────────────

STYLE = {"figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
         "axes.edgecolor": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
         "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
         "text.color": "#e0e0e0", "grid.color": "#2a2a4a", "grid.alpha": 0.6,
         "legend.framealpha": 0.4, "legend.facecolor": "#1a1a2e",
         "legend.edgecolor": "#555577"}

PALETTE = ["#e040fb", "#40c4ff", "#69f0ae", "#ff6e40", "#ffea00",
           "#ff4081", "#b2ff59", "#40c4ff", "#ffd740"]


def apply_style(fig, ax_or_axes=None):
    for k, v in STYLE.items():
        plt.rcParams[k] = v


def save_fig(fig, path: Path, dpi: int = 150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {path.name}")


def plot_roc_curves(results: dict, title: str, out_path: Path):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=STYLE["figure.facecolor"])
        ax.set_facecolor(STYLE["axes.facecolor"])
        for i, (name, res) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(res["y_true"], res["probabilities"])
            ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=2,
                    label=f"{name} (AUC={res['metrics']['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Chance")
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_fig(fig, out_path)


def plot_pr_curves(results: dict, title: str, out_path: Path):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=STYLE["figure.facecolor"])
        ax.set_facecolor(STYLE["axes.facecolor"])
        for i, (name, res) in enumerate(results.items()):
            prec, rec, _ = precision_recall_curve(res["y_true"], res["probabilities"])
            ax.plot(rec, prec, color=PALETTE[i % len(PALETTE)], lw=2,
                    label=f"{name} (AP={res['metrics']['pr_auc']:.3f})")
        prevalence = results[next(iter(results))]["y_true"].mean()
        ax.axhline(prevalence, linestyle="--", color="grey", lw=1,
                   label=f"Positive prevalence ({prevalence:.3f})")
        ax.set(xlabel="Recall", ylabel="Precision", title=title)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_fig(fig, out_path)


def plot_confusion_matrices(results: dict, title: str, out_path: Path,
                             threshold: float = 0.5):
    n = len(results)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5),
                                  facecolor=STYLE["figure.facecolor"])
        if n == 1:
            axes = [axes]
        for ax, (name, res) in zip(axes, results.items()):
            t = res.get("threshold", threshold)
            preds = (res["probabilities"] >= t).astype(int)
            cm = confusion_matrix(res["y_true"], preds, labels=[0, 1])
            disp = ConfusionMatrixDisplay(cm, display_labels=["No (0)", "Yes (1)"])
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(name, color="#e0e0e0")
            ax.set_facecolor(STYLE["axes.facecolor"])
        fig.suptitle(title, color="#e0e0e0", fontsize=13)
        fig.tight_layout()
        save_fig(fig, out_path)


def plot_calibration_curves(results: dict, out_path: Path):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=STYLE["figure.facecolor"])
        ax.set_facecolor(STYLE["axes.facecolor"])
        for i, (name, res) in enumerate(results.items()):
            observed, predicted = calibration_curve(
                res["y_true"], res["probabilities"], n_bins=10, strategy="quantile")
            ax.plot(predicted, observed, marker="o", color=PALETTE[i % len(PALETTE)],
                    label=name, lw=2)
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect calibration")
        ax.set(xlabel="Mean Predicted Probability", ylabel="Observed Positive Fraction",
               title="Calibration Curves")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_fig(fig, out_path)


def plot_model_comparison(comparison: dict, out_path: Path):
    metrics = ["roc_auc", "pr_auc", "f1", "recall", "precision", "specificity"]
    names = list(comparison.keys())
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=STYLE["figure.facecolor"])
        ax.set_facecolor(STYLE["axes.facecolor"])
        x = np.arange(len(metrics))
        width = 0.8 / len(names)
        for i, name in enumerate(names):
            vals = [comparison[name].get(m, 0) for m in metrics]
            bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.9,
                          label=name, color=PALETTE[i % len(PALETTE)], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6,
                        color="#e0e0e0")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=15, ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison — Validation Metrics")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        save_fig(fig, out_path)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  NephroMind AI — Full ML Pipeline")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 60)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    seed = config["random_seed"]
    set_seed(seed)

    ROOT_DIR = Path(config.get("raw_dataset_path")).parent if Path(config.get("raw_dataset_path", "")).is_absolute() else ROOT
    ARTIFACT_DIR = ROOT / config["artifact_dir"]
    REPORTS_DIR = ROOT / config["reports_dir"]
    FIGURES_DIR = ROOT / config["figures_dir"]
    for d in [ARTIFACT_DIR, REPORTS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Ingestion ─────────────────────────────────────────────────────────
    print("\n[1] Data Ingestion")
    data, ingestion_summary = load_patient_records(config["raw_dataset_path"], config)
    data, invalid_bp_count = mark_documented_invalid_blood_pressure(
        data, config["documented_invalid_bp_patient_ids"])
    print(f"    Valid records : {ingestion_summary['patient_rows']:,}")
    print(f"    Audit-log rows: {ingestion_summary['audit_log_rows']}")
    print(f"    Invalid BP marked missing: {invalid_bp_count}")

    features = config["numeric_features"] + config["categorical_features"]
    X = data[features]
    y = (data[config["target_column"]] == config["positive_class"]).astype(int)
    print(f"    Class distribution — Yes(1): {y.sum():,}  No(0): {(y==0).sum():,}")

    # ── 2. Train / Val / Test split ──────────────────────────────────────────
    print("\n[2] Train / Validation / Test Split")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=config["split"]["test"], stratify=y, random_state=seed)
    val_ratio = config["split"]["validation"] / (config["split"]["train"] + config["split"]["validation"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval, random_state=seed)
    print(f"    Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    split_stats = {
        "train": len(X_train), "validation": len(X_val), "test": len(X_test),
        "train_pos": int(y_train.sum()), "val_pos": int(y_val.sum()),
        "test_pos": int(y_test.sum()),
    }

    # ── 3. Preprocessor ──────────────────────────────────────────────────────
    print("\n[3] Building preprocessor")
    preprocessor = build_preprocessor(config)

    # ── 4. Define models ─────────────────────────────────────────────────────
    print("\n[4] Defining models")
    majority_prob_train = float(y_train.mean())

    base_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed, C=1.0),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=3,
            class_weight="balanced", random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=seed),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
            alpha=1e-4, batch_size=128, learning_rate_init=1e-3,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, max_iter=500, random_state=seed),
    }

    if XGBOOST_AVAILABLE:
        scale_pos_weight = float((y_train == 0).sum()) / float(y_train.sum())
        base_models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=seed, n_jobs=-1,
            verbosity=0, use_label_encoder=False)

    # ── 5. Baseline ───────────────────────────────────────────────────────────
    print("\n[5] Majority-class baseline")
    majority_probs_val = np.full(len(y_val), majority_prob_train)
    val_results: dict[str, dict] = {
        "Majority Baseline": {
            "y_true": y_val.to_numpy(),
            "probabilities": majority_probs_val,
            "metrics": evaluate_binary(y_val.to_numpy(), majority_probs_val),
            "threshold": 0.5,
        }
    }

    # ── 6. Train all base models ──────────────────────────────────────────────
    print("\n[6] Training models on training set, evaluating on validation set")
    fitted_pipelines: dict[str, Pipeline] = {}
    for name, estimator in base_models.items():
        print(f"    >> {name} ...", end=" ", flush=True)
        try:
            pipe = Pipeline([("preprocessor", build_preprocessor(config)),
                             ("model", estimator)])
            pipe.fit(X_train, y_train)
            probs_val = pipe.predict_proba(X_val)[:, 1]
            val_results[name] = {
                "y_true": y_val.to_numpy(),
                "probabilities": probs_val,
                "metrics": evaluate_binary(y_val.to_numpy(), probs_val),
                "threshold": 0.5,
            }
            fitted_pipelines[name] = pipe
            m = val_results[name]["metrics"]
            print(f"ROC-AUC={m['roc_auc']:.3f}  F1={m['f1']:.3f}  Recall={m['recall']:.3f}")
        except Exception as exc:
            print(f"FAILED: {exc}")
            traceback.print_exc()

    # ── 7. Validation plots ───────────────────────────────────────────────────
    print("\n[7] Generating validation plots")
    plot_roc_curves(val_results, "Validation ROC Curves",
                    FIGURES_DIR / "val_roc_curves.png")
    plot_pr_curves(val_results, "Validation Precision-Recall Curves",
                   FIGURES_DIR / "val_pr_curves.png")
    plot_confusion_matrices(val_results, "Validation Confusion Matrices",
                            FIGURES_DIR / "val_confusion_matrices.png")
    plot_calibration_curves(val_results, FIGURES_DIR / "val_calibration_curves.png")
    val_comparison = {name: res["metrics"] for name, res in val_results.items()}
    plot_model_comparison(val_comparison, FIGURES_DIR / "val_model_comparison.png")

    # ── 8. Select top candidates for tuning ──────────────────────────────────
    print("\n[8] Model selection for hyperparameter tuning")
    trainable = {k: v for k, v in val_results.items() if k != "Majority Baseline"}
    ranked = sorted(trainable.items(),
                    key=lambda x: (x[1]["metrics"]["roc_auc"] + x[1]["metrics"]["pr_auc"] +
                                   x[1]["metrics"]["f1"]) / 3,
                    reverse=True)
    top_candidates = [name for name, _ in ranked[:2]]
    print(f"    Top candidates: {top_candidates}")
    for name, res in ranked:
        m = res["metrics"]
        print(f"      {name}: ROC-AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  "
              f"F1={m['f1']:.3f}  Recall={m['recall']:.3f}")

    # ── 9. Hyperparameter tuning ──────────────────────────────────────────────
    print("\n[9] Hyperparameter tuning (RandomizedSearchCV on train data only)")
    tuning_param_spaces = {
        "Random Forest": {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_leaf": [1, 2, 3, 5],
            "model__max_features": ["sqrt", "log2", 0.5],
            "model__class_weight": ["balanced", "balanced_subsample"],
        },
        "XGBoost": {
            "model__n_estimators": [200, 300, 400],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [3, 4, 5, 6],
            "model__subsample": [0.7, 0.8, 0.9],
            "model__colsample_bytree": [0.7, 0.8, 0.9],
            "model__reg_alpha": [0, 0.1, 0.5],
            "model__reg_lambda": [1, 1.5, 2],
        },
        "Gradient Boosting": {
            "model__n_estimators": [200, 300, 400],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.7, 0.8, 0.9],
            "model__max_features": ["sqrt", "log2", None],
            "model__min_samples_leaf": [1, 3, 5],
        },
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            "model__solver": ["lbfgs", "saga"],
            "model__penalty": ["l2"],
        },
        "MLP": {
            "model__hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
            "model__alpha": [1e-5, 1e-4, 1e-3],
            "model__learning_rate_init": [5e-4, 1e-3, 2e-3],
        },
    }

    cv_strategy = StratifiedKFold(n_splits=config["hyperparameter_tuning"]["cv"],
                                   shuffle=True, random_state=seed)

    tuned_pipelines: dict[str, Pipeline] = {}
    tuned_val_results: dict[str, dict] = {}

    for name in top_candidates:
        if name not in tuning_param_spaces or name not in fitted_pipelines:
            print(f"    Skipping tuning for {name} (no param space defined)")
            continue
        print(f"    Tuning {name} ...")
        try:
            base_pipe = fitted_pipelines[name]
            param_space = tuning_param_spaces[name]
            search = RandomizedSearchCV(
                estimator=Pipeline([("preprocessor", build_preprocessor(config)),
                                    ("model", base_pipe.named_steps["model"].__class__(
                                        **({} if name not in ["XGBoost", "MLP"]
                                           else {"random_state": seed})))]),
                param_distributions=param_space,
                n_iter=config["hyperparameter_tuning"]["n_iter"],
                cv=cv_strategy,
                scoring=config["hyperparameter_tuning"]["scoring"],
                n_jobs=-1,
                random_state=seed,
                refit=True,
                verbose=0,
            )

            # Build fresh estimator with the same base class
            if name == "Random Forest":
                base_est = RandomForestClassifier(random_state=seed, n_jobs=-1)
            elif name == "XGBoost":
                base_est = XGBClassifier(eval_metric="logloss", verbosity=0,
                                          random_state=seed, n_jobs=-1,
                                          use_label_encoder=False,
                                          scale_pos_weight=scale_pos_weight)
            elif name == "Gradient Boosting":
                base_est = GradientBoostingClassifier(random_state=seed)
            elif name == "Logistic Regression":
                base_est = LogisticRegression(max_iter=2000, class_weight="balanced",
                                               random_state=seed)
            elif name == "MLP":
                base_est = MLPClassifier(early_stopping=True, validation_fraction=0.15,
                                          n_iter_no_change=20, max_iter=500,
                                          random_state=seed)
            else:
                continue

            search_pipe = Pipeline([("preprocessor", build_preprocessor(config)),
                                     ("model", base_est)])
            search.estimator = search_pipe
            search.fit(X_train, y_train)
            best_pipe = search.best_estimator_

            probs_val = best_pipe.predict_proba(X_val)[:, 1]
            metrics_val = evaluate_binary(y_val.to_numpy(), probs_val)
            tuned_val_results[f"{name} (Tuned)"] = {
                "y_true": y_val.to_numpy(),
                "probabilities": probs_val,
                "metrics": metrics_val,
                "threshold": 0.5,
            }
            tuned_pipelines[f"{name} (Tuned)"] = best_pipe
            m = metrics_val
            print(f"      Best params: {search.best_params_}")
            print(f"      Val — ROC-AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  "
                  f"F1={m['f1']:.3f}  Recall={m['recall']:.3f}")
        except Exception as exc:
            print(f"      Tuning failed for {name}: {exc}")
            traceback.print_exc()

    # Merge all candidate results for final selection
    all_val_results = {**val_results, **tuned_val_results}
    all_pipelines = {**fitted_pipelines, **tuned_pipelines}

    # ── 10. Final model selection ─────────────────────────────────────────────
    print("\n[10] Final model selection")
    eligible = {k: v for k, v in all_val_results.items() if k != "Majority Baseline"}
    best_name = max(eligible,
                    key=lambda k: (eligible[k]["metrics"]["roc_auc"] * 0.3 +
                                   eligible[k]["metrics"]["pr_auc"] * 0.3 +
                                   eligible[k]["metrics"]["f1"] * 0.2 +
                                   eligible[k]["metrics"]["recall"] * 0.2))
    best_pipeline = all_pipelines[best_name]
    print(f"    Selected: {best_name}")
    m = eligible[best_name]["metrics"]
    print(f"    Val — ROC-AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  "
          f"F1={m['f1']:.3f}  Recall={m['recall']:.3f}")

    # ── 11. Threshold tuning ──────────────────────────────────────────────────
    print("\n[11] Threshold optimisation on validation set")
    val_probs_best = all_val_results[best_name]["probabilities"]
    optimal_threshold = find_best_threshold(y_val.to_numpy(), val_probs_best, optimize="f1")
    print(f"    Optimal threshold (F1-maximising, val set): {optimal_threshold:.2f}")

    # ── 12. Probability calibration ───────────────────────────────────────────
    print("\n[12] Probability calibration (isotonic on val probabilities)")
    try:
        from sklearn.isotonic import IsotonicRegression
        # Methodology: use the val-set predictions from the ALREADY-fitted best_pipeline
        # to fit an isotonic regression calibrator. This is methodologically correct:
        # - best_pipeline was fitted on X_train only (no val data)
        # - calibrator is fitted on val predictions (held-out from training)
        # - test set is untouched
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        ir.fit(val_probs_best, y_val.to_numpy())
        cal_wrapper = IsotonicWrapper(best_pipeline, ir)  # module-level class — picklable
        cal_probs_val = ir.predict(val_probs_best)
        brier_before = brier_score_loss(y_val, val_probs_best)
        brier_after = brier_score_loss(y_val, cal_probs_val)
        print(f"    Brier score before calibration: {brier_before:.4f}")
        print(f"    Brier score after  calibration: {brier_after:.4f}")
        if brier_after < brier_before - 0.001:
            final_model = cal_wrapper
            print("    Calibration accepted (lower Brier score).")
        else:
            final_model = best_pipeline
            print("    Calibration marginal or not beneficial -- using uncalibrated model.")
    except Exception as exc:
        print(f"    Calibration failed ({exc}) -- using uncalibrated model.")
        traceback.print_exc()
        final_model = best_pipeline

    # ── 13. Final test-set evaluation ─────────────────────────────────────────
    print("\n[13] Final test-set evaluation (test set used for the FIRST time)")
    test_probs = final_model.predict_proba(X_test)[:, 1]
    test_metrics_default = evaluate_binary(y_test.to_numpy(), test_probs, threshold=0.5)
    test_metrics_tuned = evaluate_binary(y_test.to_numpy(), test_probs,
                                          threshold=optimal_threshold)
    print(f"    Threshold=0.50  — ROC-AUC={test_metrics_default['roc_auc']:.4f}  "
          f"PR-AUC={test_metrics_default['pr_auc']:.4f}  "
          f"F1={test_metrics_default['f1']:.4f}  "
          f"Recall={test_metrics_default['recall']:.4f}  "
          f"Specificity={test_metrics_default['specificity']:.4f}")
    print(f"    Threshold={optimal_threshold:.2f}  — ROC-AUC={test_metrics_tuned['roc_auc']:.4f}  "
          f"PR-AUC={test_metrics_tuned['pr_auc']:.4f}  "
          f"F1={test_metrics_tuned['f1']:.4f}  "
          f"Recall={test_metrics_tuned['recall']:.4f}  "
          f"Specificity={test_metrics_tuned['specificity']:.4f}")

    # ── 14. Test-set plots ────────────────────────────────────────────────────
    print("\n[14] Generating test-set plots")
    majority_probs_test = np.full(len(y_test), majority_prob_train)
    test_results = {
        "Majority Baseline": {
            "y_true": y_test.to_numpy(),
            "probabilities": majority_probs_test,
            "metrics": evaluate_binary(y_test.to_numpy(), majority_probs_test),
            "threshold": 0.5,
        },
        f"{best_name} (default t=0.5)": {
            "y_true": y_test.to_numpy(),
            "probabilities": test_probs,
            "metrics": test_metrics_default,
            "threshold": 0.5,
        },
        f"{best_name} (t={optimal_threshold:.2f})": {
            "y_true": y_test.to_numpy(),
            "probabilities": test_probs,
            "metrics": test_metrics_tuned,
            "threshold": optimal_threshold,
        },
    }
    plot_roc_curves(test_results, "Test ROC Curves", FIGURES_DIR / "test_roc_curve.png")
    plot_pr_curves(test_results, "Test Precision-Recall Curves",
                   FIGURES_DIR / "test_pr_curve.png")
    plot_confusion_matrices(test_results, "Test Confusion Matrices",
                             FIGURES_DIR / "test_confusion_matrices.png")

    # ── 15. SHAP explainability ───────────────────────────────────────────────
    shap_values_global = None
    feature_names_out = None
    top_features_global = []

    if SHAP_AVAILABLE:
        print("\n[15] SHAP explainability")
        try:
            # Get preprocessed training data for SHAP
            prep_step = best_pipeline.named_steps["preprocessor"]
            X_train_transformed = prep_step.transform(X_train)
            feature_names_out = list(prep_step.get_feature_names_out())

            model_step = best_pipeline.named_steps["model"]
            model_class = type(model_step).__name__

            # Choose SHAP explainer based on model type
            if hasattr(model_step, "feature_importances_"):
                explainer = shap.TreeExplainer(model_step)
                shap_values_raw = explainer.shap_values(X_train_transformed[:2000])
                # For classifiers: shap_values returns 2D array for binary classifiers
                # In SHAP >= 0.46, binary tree classifiers return single array (positive class)
                if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
                    shap_values_global = shap_values_raw[1]
                elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
                    # shape (n_samples, n_features, 2) — take positive class
                    shap_values_global = shap_values_raw[:, :, 1]
                else:
                    shap_values_global = shap_values_raw
            else:
                # Use KernelExplainer for non-tree models
                rng = np.random.RandomState(seed)
                idx = rng.choice(len(X_train_transformed), size=min(200, len(X_train_transformed)), replace=False)
                background = X_train_transformed[idx]
                explainer = shap.KernelExplainer(model_step.predict_proba, background)
                shap_sample = X_train_transformed[:200]
                shap_values_raw = explainer.shap_values(shap_sample)
                if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
                    shap_values_global = shap_values_raw[1]
                elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
                    shap_values_global = shap_values_raw[:, :, 1]
                else:
                    shap_values_global = shap_values_raw

            # Global feature importance plot
            mean_abs_shap = np.abs(shap_values_global).mean(axis=0)
            importance_df = pd.DataFrame({
                "feature": feature_names_out,
                "mean_abs_shap": mean_abs_shap
            }).sort_values("mean_abs_shap", ascending=False)
            top_features_global = importance_df["feature"].head(10).tolist()

            print(f"    Top 5 features by |SHAP|: {top_features_global[:5]}")

            with plt.rc_context(STYLE):
                fig, ax = plt.subplots(figsize=(9, 6), facecolor=STYLE["figure.facecolor"])
                ax.set_facecolor(STYLE["axes.facecolor"])
                colors = [PALETTE[i % len(PALETTE)] for i in range(len(importance_df.head(13)))]
                bars = ax.barh(importance_df["feature"].head(13)[::-1],
                               importance_df["mean_abs_shap"].head(13)[::-1],
                               color=colors[::-1], alpha=0.85)
                ax.set_xlabel("Mean |SHAP value|", color="#e0e0e0")
                ax.set_title("Global Feature Importance (SHAP)", color="#e0e0e0")
                ax.grid(axis="x", alpha=0.3)
                fig.tight_layout()
                save_fig(fig, FIGURES_DIR / "shap_global_importance.png")

            # SHAP summary beeswarm/dot plot
            try:
                with plt.rc_context(STYLE):
                    fig = plt.figure(figsize=(10, 7), facecolor=STYLE["figure.facecolor"])
                    shap.summary_plot(shap_values_global[:min(1000, len(shap_values_global))],
                                      X_train_transformed[:min(1000, len(X_train_transformed))],
                                      feature_names=feature_names_out,
                                      show=False, plot_type="dot", color_bar=True,
                                      max_display=13)
                    plt.gcf().set_facecolor(STYLE["figure.facecolor"])
                    save_fig(plt.gcf(), FIGURES_DIR / "shap_summary_dot.png")
            except Exception as e:
                print(f"    SHAP dot plot skipped: {e}")

            # Individual patient explanation — first test set record
            try:
                X_test_transformed = prep_step.transform(X_test.iloc[:1])
                shap_single_raw = explainer.shap_values(X_test_transformed)
                if isinstance(shap_single_raw, list) and len(shap_single_raw) == 2:
                    shap_single = shap_single_raw[1][0]
                else:
                    shap_single = shap_single_raw[0]
                single_df = pd.DataFrame({
                    "feature": feature_names_out,
                    "shap_value": shap_single
                }).sort_values("shap_value", ascending=False)

                with plt.rc_context(STYLE):
                    fig, ax = plt.subplots(figsize=(9, 6), facecolor=STYLE["figure.facecolor"])
                    ax.set_facecolor(STYLE["axes.facecolor"])
                    vals_to_plot = single_df.head(12)
                    bar_colors = [PALETTE[0] if v > 0 else PALETTE[1] for v in vals_to_plot["shap_value"]]
                    ax.barh(vals_to_plot["feature"][::-1],
                            vals_to_plot["shap_value"][::-1],
                            color=bar_colors[::-1], alpha=0.85)
                    ax.axvline(0, color="white", lw=0.8, ls="--")
                    ax.set_xlabel("SHAP Value (positive → disease predicted)", color="#e0e0e0")
                    ax.set_title("Individual Patient Explanation — Test Record 0", color="#e0e0e0")
                    ax.grid(axis="x", alpha=0.3)
                    fig.tight_layout()
                    save_fig(fig, FIGURES_DIR / "shap_individual_patient_0.png")
                print("    Individual SHAP explanation generated.")
            except Exception as e:
                print(f"    Individual SHAP plot skipped: {e}")

            # Save shap data
            np.save(str(ARTIFACT_DIR / "shap_values_train_sample.npy"), shap_values_global)
            importance_df.to_csv(ARTIFACT_DIR / "shap_global_importance.csv", index=False)
            print(f"    SHAP values saved.")

        except Exception as exc:
            print(f"    SHAP explainability failed: {exc}")
            traceback.print_exc()
            # Note: do not reassign module-level SHAP_AVAILABLE here (scoping issue)
    else:
        print("\n[15] SHAP not available — skipping explainability.")

    # Feature importance from tree models (fallback or supplement)
    try:
        model_step = best_pipeline.named_steps["model"]
        if hasattr(model_step, "feature_importances_"):
            prep_step = best_pipeline.named_steps["preprocessor"]
            feature_names_out = list(prep_step.get_feature_names_out())
            importances = model_step.feature_importances_
            fi_df = pd.DataFrame({"feature": feature_names_out,
                                   "importance": importances}).sort_values(
                "importance", ascending=False)
            with plt.rc_context(STYLE):
                fig, ax = plt.subplots(figsize=(9, 6), facecolor=STYLE["figure.facecolor"])
                ax.set_facecolor(STYLE["axes.facecolor"])
                colors = [PALETTE[i % len(PALETTE)] for i in range(min(13, len(fi_df)))]
                ax.barh(fi_df["feature"].head(13)[::-1],
                        fi_df["importance"].head(13)[::-1],
                        color=colors[::-1], alpha=0.85)
                ax.set_xlabel("Feature Importance (Gini)", color="#e0e0e0")
                ax.set_title("Model Feature Importance (Tree-based)", color="#e0e0e0")
                ax.grid(axis="x", alpha=0.3)
                fig.tight_layout()
                save_fig(fig, FIGURES_DIR / "feature_importance_model.png")
            fi_df.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)
            if not top_features_global:
                top_features_global = fi_df["feature"].head(6).tolist()
    except Exception as exc:
        print(f"    Feature importance plot skipped: {exc}")

    # ── 16. Save artifact ─────────────────────────────────────────────────────
    print("\n[16] Saving model artifact")
    artifact = {
        "model_version": config.get("model_version", "v1.0.0"),
        "run_name": config.get("run_name", "nephromind_full_pipeline_v1"),
        "training_timestamp": datetime.now().isoformat(),
        "best_model_name": best_name,
        "features": features,
        "numeric_features": config["numeric_features"],
        "categorical_features": config["categorical_features"],
        "target_column": config["target_column"],
        "positive_class": config["positive_class"],
        "label_mapping": {"0": "No", "1": "Yes"},
        "random_seed": seed,
        "optimal_threshold": optimal_threshold,
        "test_metrics_default_threshold": test_metrics_default,
        "test_metrics_optimal_threshold": test_metrics_tuned,
        "training_records": int(len(X_train)),
        "validation_records": int(len(X_val)),
        "test_records": int(len(X_test)),
        "top_features_shap": top_features_global,
        "calibrated": type(final_model).__name__ in ("CalibratedClassifierCV", "IsotonicWrapper"),
        "pipeline": final_model,
    }

    artifact_path = ARTIFACT_DIR / "nephromind_model_v1.joblib"
    joblib.dump(artifact, artifact_path, compress=3)
    print(f"    Artifact saved → {artifact_path}")
    # Verify load
    loaded = joblib.load(artifact_path)
    assert loaded["model_version"] == artifact["model_version"], "Artifact load verification failed!"
    test_probs_reload = loaded["pipeline"].predict_proba(X_test.iloc[:5])[:, 1]
    assert len(test_probs_reload) == 5, "Reload prediction failed!"
    print(f"    Artifact reload verified — predictions match: {np.allclose(test_probs_reload, test_probs[:5])}")

    # ── 17. Compile full results ──────────────────────────────────────────────
    all_comparison: dict[str, dict] = {}
    for name, res in all_val_results.items():
        all_comparison[name] = res["metrics"]

    # ── 18. Write reports ─────────────────────────────────────────────────────
    print("\n[17] Writing reports")
    timestamp = datetime.now().isoformat()

    _write_dataset_summary(REPORTS_DIR, config, ingestion_summary, invalid_bp_count,
                            split_stats, y, y_train, y_val, y_test, timestamp)
    _write_model_training_report(REPORTS_DIR, config, best_name, test_metrics_default,
                                  test_metrics_tuned, optimal_threshold, timestamp,
                                  all_comparison, artifact_path)
    _write_model_comparison(REPORTS_DIR, all_comparison, best_name, timestamp)
    _write_explainability_report(REPORTS_DIR, best_name, top_features_global,
                                  SHAP_AVAILABLE, timestamp)

    # ── 19. End-to-end inference test ─────────────────────────────────────────
    print("\n[18] End-to-end inference test (test record 0)")
    test_record = X_test.iloc[[0]]
    true_label = "Yes" if y_test.iloc[0] == 1 else "No"
    prob = final_model.predict_proba(test_record)[0, 1]
    pred_default = "Yes" if prob >= 0.5 else "No"
    pred_tuned = "Yes" if prob >= optimal_threshold else "No"
    print(f"    True label   : {true_label}")
    print(f"    Probability  : {prob:.4f}")
    print(f"    Prediction (t=0.50)             : {pred_default}")
    print(f"    Prediction (t={optimal_threshold:.2f}) : {pred_tuned}")

    # ── 20. Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Finished: {datetime.now().isoformat()}")
    print("=" * 60)
    print(f"  Records used   : {ingestion_summary['patient_rows']:,}")
    print(f"  Features       : {len(features)} ({features})")
    print(f"  Models trained : {', '.join(list(base_models.keys()))}")
    print(f"  Best model     : {best_name}")
    m = test_metrics_default
    print(f"  Test metrics (t=0.50):")
    print(f"    ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
          f"F1={m['f1']:.4f}  Recall={m['recall']:.4f}  "
          f"Specificity={m['specificity']:.4f}")
    m2 = test_metrics_tuned
    print(f"  Test metrics (t={optimal_threshold:.2f}):")
    print(f"    ROC-AUC={m2['roc_auc']:.4f}  PR-AUC={m2['pr_auc']:.4f}  "
          f"F1={m2['f1']:.4f}  Recall={m2['recall']:.4f}  "
          f"Specificity={m2['specificity']:.4f}")
    print(f"  Artifact       : {artifact_path}")
    print(f"  Reports        : {REPORTS_DIR}")
    print(f"  Figures        : {FIGURES_DIR}")
    print("=" * 60)

    # Save summary JSON
    summary = {
        "pipeline_complete": True,
        "timestamp": timestamp,
        "best_model": best_name,
        "optimal_threshold": optimal_threshold,
        "test_metrics_default": test_metrics_default,
        "test_metrics_tuned": test_metrics_tuned,
        "artifact_path": str(artifact_path),
        "shap_available": SHAP_AVAILABLE,
        "top_features": top_features_global,
        "split": split_stats,
        "val_comparison": {k: v for k, v in all_comparison.items()},
    }
    (REPORTS_DIR / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


# ── Report writers ────────────────────────────────────────────────────────────

def _write_dataset_summary(reports_dir, config, ingestion_summary, invalid_bp_count,
                             split_stats, y, y_train, y_val, y_test, timestamp):
    content = f"""# NephroMind AI — Dataset Summary
Generated: {timestamp}

## Dataset Overview
| Item | Value |
|------|-------|
| Source file | dataset_nephro.csv |
| Total rows in CSV | {ingestion_summary['raw_rows']:,} |
| Valid patient records | {ingestion_summary['patient_rows']:,} |
| Excluded audit-log rows | {ingestion_summary['audit_log_rows']} |
| Invalid BP records (Sys==Dia) | {invalid_bp_count} (converted to NaN) |

## Target Variable
- Column: `Kidney_Disease`
- Task: **Current kidney-disease status classification** (retrospective, cross-sectional)
- Positive class: `Yes`
- Negative class: `No`

| Class | Count | Percentage |
|-------|-------|------------|
| Yes (Kidney Disease) | {int(y.sum()):,} | {y.mean()*100:.2f}% |
| No | {int((y==0).sum()):,} | {(y==0).mean()*100:.2f}% |

> ⚠️ **Scientific Limitation**: This dataset does NOT contain longitudinal follow-up data.
> This model classifies current status only. It MUST NOT be described as progression
> prediction, clinical diagnosis, or clinically validated prediction.

## Feature Set (Final — 13 features)
### Numeric Features (12)
{chr(10).join(f'- `{f}`' for f in config['numeric_features'])}

### Categorical Features (1)
{chr(10).join(f'- `{f}`' for f in config['categorical_features'])}

## Excluded Features
| Feature | Reason |
|---------|--------|
| Patient_ID | Identifier only |
| eGFR | **Direct target leakage** (Kidney_Disease = Yes if eGFR < 60) |
| UACR_mg_g | **Direct target leakage** (Kidney_Disease = Yes if UACR ≥ 30) |
| Creatinine_mg_dL | Provenance unresolved |
| Diabetes | Deterministically derived from HbA1c ≥ 6.5 |
| Diabetes_Pedigree | Synthetic sequential placeholder |
| Diabetes_Duration_Years | Documented inconsistent/unreliable |

## Train / Validation / Test Split
| Split | Records | Positive (Yes) | Positive % |
|-------|---------|----------------|------------|
| Train | {split_stats['train']:,} | {split_stats['train_pos']:,} | {split_stats['train_pos']/split_stats['train']*100:.2f}% |
| Validation | {split_stats['validation']:,} | {split_stats['val_pos']:,} | {split_stats['val_pos']/split_stats['validation']*100:.2f}% |
| Test | {split_stats['test']:,} | {split_stats['test_pos']:,} | {split_stats['test_pos']/split_stats['test']*100:.2f}% |

- Stratified splits preserving class ratio
- Fixed random seed: {config['random_seed']}
- No test-set information entered preprocessing, tuning, or threshold selection
"""
    (reports_dir / "DATASET_SUMMARY.md").write_text(content, encoding="utf-8")
    print(f"    Written: DATASET_SUMMARY.md")


def _write_model_training_report(reports_dir, config, best_name, test_metrics_default,
                                  test_metrics_tuned, optimal_threshold, timestamp,
                                  all_comparison, artifact_path):
    cm_d = test_metrics_default["confusion_matrix"]
    cm_t = test_metrics_tuned["confusion_matrix"]
    rows = []
    for name, m in sorted(all_comparison.items(), key=lambda x: -x[1].get("roc_auc", 0)):
        rows.append(f"| {name} | {m.get('roc_auc',0):.4f} | {m.get('pr_auc',0):.4f} | "
                    f"{m.get('f1',0):.4f} | {m.get('recall',0):.4f} | "
                    f"{m.get('precision',0):.4f} | {m.get('specificity',0):.4f} |")

    content = f"""# NephroMind AI — Model Training Report
Generated: {timestamp}

## Objective
Current kidney-disease status classification (binary: Yes / No).
This is a retrospective, cross-sectional classification task.

## Selected Final Model
**{best_name}**

## Preprocessing Pipeline
- Numeric features: Median imputation → StandardScaler
- Categorical features: Mode imputation → OneHotEncoder
- Pipeline fitted on training data only; applied to validation and test sets

## Class Imbalance Handling
- Class distribution: Yes (81.1%) / No (18.9%) — imbalanced dataset
- All applicable tree/linear models trained with `class_weight="balanced"`
- XGBoost trained with `scale_pos_weight` proportional to class imbalance
- No SMOTE applied (class weights preferred for this dataset size)

## Hyperparameter Tuning
- Strategy: RandomizedSearchCV on training set only (5-fold stratified CV)
- Scoring: ROC-AUC
- n_iter: {config['hyperparameter_tuning']['n_iter']}
- Validation set untouched during tuning

## Threshold Optimisation
- Method: F1-maximising threshold search on validation set
- Optimal threshold: **{optimal_threshold:.2f}**

## Test Set Evaluation (First and Only Use of Test Set)

### Default Threshold (0.50)
| Metric | Value |
|--------|-------|
| Accuracy | {test_metrics_default['accuracy']:.4f} |
| Precision | {test_metrics_default['precision']:.4f} |
| Recall (Sensitivity) | {test_metrics_default['recall']:.4f} |
| F1-Score | {test_metrics_default['f1']:.4f} |
| ROC-AUC | {test_metrics_default['roc_auc']:.4f} |
| PR-AUC | {test_metrics_default['pr_auc']:.4f} |
| Specificity | {test_metrics_default['specificity']:.4f} |
| Brier Score | {test_metrics_default['brier_score']:.4f} |

Confusion Matrix (t=0.50):
- TN={cm_d['tn']}, FP={cm_d['fp']}, FN={cm_d['fn']}, TP={cm_d['tp']}

### Optimal Threshold ({optimal_threshold:.2f})
| Metric | Value |
|--------|-------|
| Accuracy | {test_metrics_tuned['accuracy']:.4f} |
| Precision | {test_metrics_tuned['precision']:.4f} |
| Recall (Sensitivity) | {test_metrics_tuned['recall']:.4f} |
| F1-Score | {test_metrics_tuned['f1']:.4f} |
| ROC-AUC | {test_metrics_tuned['roc_auc']:.4f} |
| PR-AUC | {test_metrics_tuned['pr_auc']:.4f} |
| Specificity | {test_metrics_tuned['specificity']:.4f} |
| Brier Score | {test_metrics_tuned['brier_score']:.4f} |

Confusion Matrix (t={optimal_threshold:.2f}):
- TN={cm_t['tn']}, FP={cm_t['fp']}, FN={cm_t['fn']}, TP={cm_t['tp']}

## Validation Comparison (All Models, ROC-AUC Ranked)
| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision | Specificity |
|-------|---------|--------|-----|--------|-----------|-------------|
{chr(10).join(rows)}

## Artifact
- Location: `{artifact_path}`
- Format: joblib (compressed, includes preprocessing + model + metadata)
- Model version: {config.get('model_version', 'v1.0.0')}

## Limitations
1. Cross-sectional data only — no progression prediction possible
2. Synthetic/registry dataset — not clinically validated
3. Target leakage features (eGFR, UACR) excluded — model uses indirect biomarkers
4. NOT a replacement for clinical diagnosis or nephrology consultation
5. Results should not be interpreted causally
"""
    (reports_dir / "MODEL_TRAINING_REPORT.md").write_text(content, encoding="utf-8")
    print(f"    Written: MODEL_TRAINING_REPORT.md")


def _write_model_comparison(reports_dir, all_comparison, best_name, timestamp):
    rows = []
    for name, m in sorted(all_comparison.items(), key=lambda x: -x[1].get("roc_auc", 0)):
        marker = " ← **SELECTED**" if name == best_name else ""
        rows.append(f"| {name}{marker} | {m.get('roc_auc',0):.4f} | {m.get('pr_auc',0):.4f} | "
                    f"{m.get('f1',0):.4f} | {m.get('recall',0):.4f} | "
                    f"{m.get('precision',0):.4f} | {m.get('specificity',0):.4f} | "
                    f"{m.get('accuracy',0):.4f} |")

    content = f"""# NephroMind AI — Model Comparison
Generated: {timestamp}

## Validation Set Comparison (All Models)

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision | Specificity | Accuracy |
|-------|---------|--------|-----|--------|-----------|-------------|----------|
{chr(10).join(rows)}

> **Note**: All metrics computed at threshold = 0.50 unless noted.
> ROC-AUC and PR-AUC are threshold-independent.
> Model selection based on weighted combination of ROC-AUC (30%), PR-AUC (30%),
> F1 (20%), Recall (20%).

## Selection Rationale
- Primary criterion: ROC-AUC + PR-AUC (threshold-independent discrimination)
- Secondary criteria: F1, Recall (clinically relevant — missing true positives
  has higher cost than false alarms in disease screening context)
- Accuracy deliberately de-emphasised due to class imbalance (81%/19% split)
- Majority baseline included to validate that models genuinely learn beyond prevalence

## Majority Baseline Comparison
The majority baseline always predicts the positive class (kidney disease).
Any useful model must substantially exceed baseline PR-AUC = ~0.811 (class prevalence).
"""
    (reports_dir / "MODEL_COMPARISON.md").write_text(content, encoding="utf-8")
    print(f"    Written: MODEL_COMPARISON.md")


def _write_explainability_report(reports_dir, best_name, top_features, shap_available, timestamp):
    features_section = ""
    if top_features:
        features_section = "\n".join(f"{i+1}. `{f}`" for i, f in enumerate(top_features))
    else:
        features_section = "Feature importance not available."

    content = f"""# NephroMind AI — Explainability Report
Generated: {timestamp}

## Important Disclaimer
Feature contributions described in this report indicate association with the model
prediction, NOT causation. The phrase "contributed to the model prediction" does
NOT mean the feature caused kidney disease.

## Explainability Method
{"SHAP (SHapley Additive exPlanations) — TreeExplainer for tree-based models" if shap_available else "Tree feature importance (SHAP not available)"}

## Final Model
{best_name}

## Global Feature Importance (Top Features)
Features listed contributed most to the model's predictions across the training set.

{features_section}

## Plot Locations
- `ml/reports/figures/shap_global_importance.png` — Mean absolute SHAP values
{"- `ml/reports/figures/shap_summary_dot.png` — SHAP beeswarm plot" if shap_available else ""}
{"- `ml/reports/figures/shap_individual_patient_0.png` — Individual patient explanation" if shap_available else ""}
- `ml/reports/figures/feature_importance_model.png` — Model tree importance

## Prototype Risk Level Mapping
**PROTOTYPE ONLY — NOT clinically validated**

The following prototype mapping of model probability to risk level is provided
for dashboard display purposes only. It is NOT a clinically validated threshold.

| Probability Range | Prototype Risk Category |
|------------------|------------------------|
| 0.00 – 0.40 | Low (further assessment recommended) |
| 0.40 – 0.65 | Medium (medical consultation advised) |
| 0.65 – 1.00 | High (urgent nephrology referral suggested) |

This mapping is illustrative and must be reviewed by a qualified nephrologist
before any clinical deployment. It does not constitute a clinical decision.

## Limitations
- SHAP values explain model predictions, not clinical ground truth
- Correlation ≠ causation
- Results are specific to this dataset and model version
- Not validated on external clinical data
"""
    (reports_dir / "EXPLAINABILITY_REPORT.md").write_text(content, encoding="utf-8")
    print(f"    Written: EXPLAINABILITY_REPORT.md")


if __name__ == "__main__":
    main()

