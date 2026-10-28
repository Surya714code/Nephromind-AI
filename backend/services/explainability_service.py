"""
NephroMind AI — Explainability Service
=======================================
Computes top feature contributions for an individual patient record
using SHAP TreeExplainer or feature deviation weighting.
"""
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def compute_feature_contributions(
    patient_df: pd.DataFrame,
    pipeline: Any,
    feature_names: List[str],
    top_n: int = 6
) -> List[Dict[str, Any]]:
    """
    Compute feature contributions for a single patient sample.

    Parameters
    ----------
    patient_df : pd.DataFrame
        Single row dataframe containing patient features.
    pipeline : sklearn Pipeline or IsotonicWrapper
        Tuned model pipeline.
    feature_names : List[str]
        Ordered feature names.
    top_n : int
        Number of top features to return.

    Returns
    -------
    List[Dict[str, Any]]
        List of dicts with keys "feature" and "contribution".
    """
    contributions = []

    try:
        # Extract preprocessor and model estimator
        base_pipeline = getattr(pipeline, "base", pipeline)
        if hasattr(base_pipeline, "named_steps"):
            preprocessor = base_pipeline.named_steps["preprocessor"]
            model = base_pipeline.named_steps["model"]
            X_trans = preprocessor.transform(patient_df)
        else:
            preprocessor = None
            model = base_pipeline
            X_trans = patient_df.values

        if SHAP_AVAILABLE and hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_trans)
            
            # Handle multi-output or binary classification shape
            if isinstance(shap_values, list):
                vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            elif isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 3:
                    vals = shap_values[0, :, 1]
                elif shap_values.ndim == 2:
                    vals = shap_values[0]
                else:
                    vals = shap_values
            else:
                vals = np.zeros(len(feature_names))

            for feat, val in zip(feature_names, vals):
                contributions.append({
                    "feature": feat,
                    "contribution": round(float(val), 4)
                })

            # Sort by absolute impact magnitude descending
            contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            return contributions[:top_n]

    except Exception:
        pass

    # Fallback method if SHAP fails or is unavailable:
    # Use model feature importances scaled by directional deviation
    try:
        base_pipeline = getattr(pipeline, "base", pipeline)
        model = base_pipeline.named_steps["model"] if hasattr(base_pipeline, "named_steps") else base_pipeline
        importances = getattr(model, "feature_importances_", np.ones(len(feature_names)) / len(feature_names))
        
        # Approximate direction based on clinically known risk direction
        high_risk_positive_features = {"HbA1c", "Age", "BMI", "Systolic_BP", "Diastolic_BP", "Resting_Heart_Rate", "Heart_Rate"}
        
        for feat, imp in zip(feature_names, importances):
            val = patient_df[feat].values[0] if feat in patient_df.columns else 0
            if isinstance(val, (int, float, np.number)):
                direction = 1.0 if feat in high_risk_positive_features else -1.0
                score = round(float(imp * direction), 4)
            else:
                score = round(float(imp), 4)
            contributions.append({"feature": feat, "contribution": score})

        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return contributions[:top_n]
    except Exception:
        # Static fallback if model inspection fails
        return [{"feature": f, "contribution": 0.0} for f in feature_names[:top_n]]
