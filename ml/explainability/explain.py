# NephroMind AI — Explainability Interface
"""
Explainability interface for NephroMind AI.
Provides SHAP-based individual and global explanations using the saved model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_PATH = ROOT / "ml" / "artifacts" / "nephromind_model_v1.joblib"

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def explain_patient(
    patient_data: dict[str, Any],
    artifact_path: Optional[str | Path] = None,
    top_n: int = 6,
) -> dict[str, Any]:
    """
    Generate SHAP-based explanation for a single patient's prediction.

    Returns
    -------
    dict with:
        top_features    : list of (feature, shap_value) tuples sorted by magnitude
        prediction_prob : float
        method          : str (explainer type used)
        disclaimer      : str
    """
    if not SHAP_AVAILABLE:
        return {
            "top_features": [],
            "prediction_prob": None,
            "method": "SHAP not available",
            "disclaimer": "Install shap package to enable explainability.",
        }

    path = Path(artifact_path) if artifact_path else DEFAULT_ARTIFACT_PATH
    artifact = joblib.load(path)
    pipeline = artifact["pipeline"]
    features = artifact["features"]
    numeric_features = artifact["numeric_features"]

    df = pd.DataFrame([{f: patient_data.get(f) for f in features}])
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Get the base pipeline (may be CalibratedClassifierCV wrapping a Pipeline)
    base_pipeline = pipeline
    if hasattr(pipeline, "calibrated_classifiers_"):
        base_pipeline = pipeline.calibrated_classifiers_[0].estimator

    preprocessor = base_pipeline.named_steps["preprocessor"]
    model = base_pipeline.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())

    X_transformed = preprocessor.transform(df)
    prob = float(pipeline.predict_proba(df)[0, 1])

    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_vals_raw = explainer.shap_values(X_transformed)
            if isinstance(shap_vals_raw, list) and len(shap_vals_raw) == 2:
                shap_vals = shap_vals_raw[1][0]
            else:
                shap_vals = shap_vals_raw[0] if shap_vals_raw.ndim > 1 else shap_vals_raw
            method = "TreeExplainer"
        else:
            # KernelExplainer requires background data — load from artifact if saved
            shap_path = ROOT / "ml" / "artifacts" / "shap_values_train_sample.npy"
            background = np.zeros((1, X_transformed.shape[1]))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_vals_raw = explainer.shap_values(X_transformed)
            if isinstance(shap_vals_raw, list) and len(shap_vals_raw) == 2:
                shap_vals = shap_vals_raw[1][0]
            else:
                shap_vals = shap_vals_raw[0]
            method = "KernelExplainer"

        shap_items = sorted(
            zip(feature_names, shap_vals.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return {
            "top_features": [
                {"feature": f, "shap_value": round(v, 5),
                 "direction": "↑ disease" if v > 0 else "↓ disease"}
                for f, v in shap_items[:top_n]
            ],
            "all_features": [
                {"feature": f, "shap_value": round(v, 5)}
                for f, v in shap_items
            ],
            "prediction_prob": round(prob, 4),
            "method": method,
            "disclaimer": (
                "SHAP values represent the contribution of each feature to this "
                "specific model prediction. They do NOT imply causation. "
                "Feature contributions are model-specific and dataset-specific."
            ),
        }
    except Exception as exc:
        return {
            "top_features": [],
            "prediction_prob": round(prob, 4),
            "method": "failed",
            "error": str(exc),
            "disclaimer": "SHAP explanation failed. Model prediction is still valid.",
        }


def get_global_importance(artifact_path: Optional[str | Path] = None) -> dict[str, Any]:
    """
    Return the global feature importance computed during training.
    Reads the saved CSV if available; otherwise returns model feature importances.
    """
    path = Path(artifact_path) if artifact_path else DEFAULT_ARTIFACT_PATH
    artifact = joblib.load(path)

    shap_csv = ROOT / "ml" / "artifacts" / "shap_global_importance.csv"
    fi_csv = ROOT / "ml" / "artifacts" / "feature_importance.csv"

    if shap_csv.exists():
        import pandas as pd
        df = pd.read_csv(shap_csv)
        return {
            "source": "SHAP",
            "features": df["feature"].tolist(),
            "importances": df["mean_abs_shap"].tolist(),
        }
    elif fi_csv.exists():
        import pandas as pd
        df = pd.read_csv(fi_csv)
        return {
            "source": "model_tree_importance",
            "features": df["feature"].tolist(),
            "importances": df["importance"].tolist(),
        }
    else:
        return {
            "source": "artifact_top_features",
            "features": artifact.get("top_features_shap", []),
            "importances": [],
        }
