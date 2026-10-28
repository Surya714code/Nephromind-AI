"""
NephroMind AI — Inference Module
==================================
Production-ready inference for the trained kidney-disease status classifier.

Scientific limitation: This model performs CURRENT KIDNEY-DISEASE STATUS
CLASSIFICATION only. It is NOT a clinical diagnosis tool and has NOT been
clinically validated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_PATH = ROOT / "ml" / "artifacts" / "nephromind_model_v1.joblib"

# Module-level cache to avoid reloading on every call
_artifact_cache: Optional[dict] = None
_artifact_path_cache: Optional[Path] = None


def load_artifact(artifact_path: Optional[str | Path] = None) -> dict:
    """Load (or return cached) model artifact."""
    global _artifact_cache, _artifact_path_cache
    path = Path(artifact_path) if artifact_path else DEFAULT_ARTIFACT_PATH
    if _artifact_cache is None or _artifact_path_cache != path:
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {path}. "
                "Run ml/training/full_pipeline.py first."
            )
        _artifact_cache = joblib.load(path)
        _artifact_path_cache = path
    return _artifact_cache


def predict_patient(
    patient_data: dict[str, Any],
    artifact_path: Optional[str | Path] = None,
    threshold: Optional[float] = None,
) -> dict[str, Any]:
    """
    Run inference for a single patient.

    Parameters
    ----------
    patient_data : dict
        Feature values for one patient. Required keys are the 13 approved features:
        Age, Gender, BMI, Systolic_BP, Diastolic_BP, Heart_Rate, SpO2,
        Daily_Steps, Sleep_Duration_hr, Activity_Minutes,
        Resting_Heart_Rate, HRV_ms, HbA1c

    artifact_path : str or Path, optional
        Path to the saved .joblib artifact. Defaults to ml/artifacts/nephromind_model_v1.joblib.

    threshold : float, optional
        Classification threshold. If None, uses the model's optimal threshold
        (tuned on validation set).

    Returns
    -------
    dict with keys:
        prediction      : "Yes" / "No"
        probability     : float  (probability of kidney disease = Yes)
        model_version   : str
        threshold_used  : float
        top_features    : list[str]   (global top features from training)
        disclaimer      : str
    """
    artifact = load_artifact(artifact_path)
    pipeline = artifact["pipeline"]
    model_version = artifact["model_version"]
    label_mapping = artifact["label_mapping"]
    features = artifact["features"]
    optimal_threshold = threshold if threshold is not None else artifact["optimal_threshold"]
    top_features = artifact.get("top_features_shap", [])

    # Validate and construct feature dataframe
    missing = [f for f in features if f not in patient_data]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    df = pd.DataFrame([{f: patient_data[f] for f in features}])

    # Coerce numeric types
    numeric_features = artifact["numeric_features"]
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    prob = float(pipeline.predict_proba(df)[0, 1])
    predicted_int = int(prob >= optimal_threshold)
    prediction = label_mapping[str(predicted_int)]

    return {
        "prediction": prediction,
        "probability": round(prob, 4),
        "model_version": model_version,
        "threshold_used": round(optimal_threshold, 4),
        "top_features": top_features[:6] if top_features else [],
        "disclaimer": (
            "PROTOTYPE — This model classifies current kidney-disease status "
            "based on clinical features. It is NOT a clinical diagnosis and "
            "has NOT been clinically validated. Results must be reviewed by a "
            "qualified nephrologist before any clinical action."
        ),
    }


def predict_batch(
    patient_records: list[dict[str, Any]],
    artifact_path: Optional[str | Path] = None,
    threshold: Optional[float] = None,
) -> list[dict[str, Any]]:
    """
    Run inference for multiple patients.

    Parameters
    ----------
    patient_records : list of dicts, each containing the 13 required features.

    Returns
    -------
    list of result dicts (same format as predict_patient).
    """
    artifact = load_artifact(artifact_path)
    pipeline = artifact["pipeline"]
    features = artifact["features"]
    numeric_features = artifact["numeric_features"]
    model_version = artifact["model_version"]
    label_mapping = artifact["label_mapping"]
    optimal_threshold = threshold if threshold is not None else artifact["optimal_threshold"]
    top_features = artifact.get("top_features_shap", [])

    records_df = pd.DataFrame([{f: r.get(f) for f in features} for r in patient_records])
    for col in numeric_features:
        if col in records_df.columns:
            records_df[col] = pd.to_numeric(records_df[col], errors="coerce")

    probs = pipeline.predict_proba(records_df)[:, 1]
    results = []
    for i, prob in enumerate(probs):
        prob = float(prob)
        predicted_int = int(prob >= optimal_threshold)
        results.append({
            "patient_index": i,
            "prediction": label_mapping[str(predicted_int)],
            "probability": round(prob, 4),
            "model_version": model_version,
            "threshold_used": round(optimal_threshold, 4),
            "top_features": top_features[:6] if top_features else [],
        })
    return results


def get_model_info(artifact_path: Optional[str | Path] = None) -> dict[str, Any]:
    """Return model metadata without running inference."""
    artifact = load_artifact(artifact_path)
    return {
        "model_version": artifact["model_version"],
        "run_name": artifact.get("run_name", ""),
        "training_timestamp": artifact.get("training_timestamp", ""),
        "best_model_name": artifact.get("best_model_name", ""),
        "features": artifact["features"],
        "numeric_features": artifact["numeric_features"],
        "categorical_features": artifact["categorical_features"],
        "target_column": artifact["target_column"],
        "positive_class": artifact["positive_class"],
        "label_mapping": artifact["label_mapping"],
        "optimal_threshold": artifact["optimal_threshold"],
        "test_metrics_default": artifact.get("test_metrics_default_threshold", {}),
        "test_metrics_tuned": artifact.get("test_metrics_optimal_threshold", {}),
        "training_records": artifact.get("training_records", None),
        "top_features": artifact.get("top_features_shap", []),
        "calibrated": artifact.get("calibrated", False),
    }


if __name__ == "__main__":
    # Quick functional test — uses a single synthetic-but-valid feature dict.
    print("Loading artifact ...")
    info = get_model_info()
    print(f"Model version : {info['model_version']}")
    print(f"Best model    : {info['best_model_name']}")
    print(f"Threshold     : {info['optimal_threshold']:.2f}")

    sample_patient = {
        "Age": 58, "Gender": "Female", "BMI": 31.2,
        "Systolic_BP": 145.0, "Diastolic_BP": 92.0, "Heart_Rate": 88,
        "SpO2": 95.0, "Daily_Steps": 3500, "Sleep_Duration_hr": 5.8,
        "Activity_Minutes": 18, "Resting_Heart_Rate": 78, "HRV_ms": 28.0,
        "HbA1c": 7.8,
    }
    result = predict_patient(sample_patient)
    print("\nSample inference result:")
    for k, v in result.items():
        if k != "disclaimer":
            print(f"  {k}: {v}")
    print(f"  disclaimer: (see output)")
