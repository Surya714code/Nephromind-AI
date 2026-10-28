"""First validation-only model comparison for NephroMind AI.

This script deliberately leaves the held-out test set untouched. It does not perform
hyperparameter tuning or final model selection.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from ml.data.ingestion import load_patient_records, mark_documented_invalid_blood_pressure
from ml.evaluation.metrics import evaluate_binary, save_validation_plots
from ml.preprocessing.pipeline import build_preprocessor


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "ml" / "configs" / "first_run.json"
OUTPUT_DIR = ROOT / "ml" / "reports" / "first_validation_comparison"


def model_definitions(seed: int) -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=2, random_state=seed),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=1e-4,
            batch_size=128, learning_rate_init=1e-3, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, max_iter=400, random_state=seed,
        ),
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    seed = config["random_seed"]
    random.seed(seed); np.random.seed(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data, ingestion_summary = load_patient_records(config["raw_dataset_path"], config)
    data, invalid_bp_count = mark_documented_invalid_blood_pressure(data, config["documented_invalid_bp_patient_ids"])
    features = config["numeric_features"] + config["categorical_features"]
    x = data[features]
    y = (data[config["target_column"]] == config["positive_class"]).astype(int)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=config["split"]["test"], stratify=y, random_state=seed
    )
    validation_share_of_train_val = config["split"]["validation"] / (config["split"]["train"] + config["split"]["validation"])
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_val, y_train_val, test_size=validation_share_of_train_val,
        stratify=y_train_val, random_state=seed,
    )

    run_summary = {
        "config": config,
        "ingestion": ingestion_summary,
        "documented_invalid_bp_rows_marked_missing": invalid_bp_count,
        "split_rows": {"train": len(x_train), "validation": len(x_validation), "test_untouched": len(x_test)},
        "class_distribution": {
            "all": y.value_counts().sort_index().to_dict(),
            "train": y_train.value_counts().sort_index().to_dict(),
            "validation": y_validation.value_counts().sort_index().to_dict(),
            "test_untouched": y_test.value_counts().sort_index().to_dict(),
        },
    }

    # A naive majority predictor always returns the training positive prevalence.
    majority_probability = np.full(len(y_validation), y_train.mean())
    validation_results: dict[str, dict[str, object]] = {
        "Majority baseline": {
            "y_true": y_validation.to_numpy(), "probabilities": majority_probability,
            "metrics": evaluate_binary(y_validation.to_numpy(), majority_probability),
        }
    }
    reports = {}
    for name, estimator in model_definitions(seed).items():
        pipeline = Pipeline([("preprocessor", build_preprocessor(config)), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_validation)[:, 1]
        validation_results[name] = {
            "y_true": y_validation.to_numpy(), "probabilities": probabilities,
            "metrics": evaluate_binary(y_validation.to_numpy(), probabilities),
        }
        reports[name] = classification_report(y_validation, (probabilities >= 0.5).astype(int), target_names=["No", "Yes"], output_dict=True, zero_division=0)

    save_validation_plots(validation_results, OUTPUT_DIR)
    serializable_results = {name: result["metrics"] for name, result in validation_results.items()}
    run_summary["validation_metrics"] = serializable_results
    run_summary["classification_reports"] = reports
    (OUTPUT_DIR / "first_run_results.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))
    print("Test set was split and retained untouched; no test metrics were calculated.")


if __name__ == "__main__":
    main()
