"""
NephroMind AI — DNN Prediction Service
=======================================
Loads the PyTorch DNN model, scaler, and feature metadata at startup.
Executes inference for the 8-feature Renal Sentinel feature vector.

Model: NephroMind DNN (PyTorch MLP)
Features: Age, BMI, Systolic_BP, Diastolic_BP, Heart_Rate, SpO2, HbA1c, Gender_encoded
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn

from backend.schemas import PredictionRequest, PredictionResponse, FeatureContribution

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"

# ── DNN Architecture (must match training) ────────────────────────────────────
class NephroMindDNN(nn.Module):
    """
    Tabular MLP for kidney disease risk classification.
    Input: 8 clinical/physiological features from Renal Sentinel hardware.
    Output: logit (sigmoid applied for probability).
    """
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.30),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.25),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

# ── Module-level model state (loaded once at startup) ─────────────────────────
_model: Optional[NephroMindDNN] = None
_scaler = None
_features_meta: Optional[Dict[str, Any]] = None
_locked_threshold: float = 0.5
_model_version: str = "nephromind_dnn_v1"

# ── Human-readable feature names for display ──────────────────────────────────
FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "Age": "Age",
    "BMI": "BMI",
    "Systolic_BP": "Systolic BP",
    "Diastolic_BP": "Diastolic BP",
    "Heart_Rate": "Heart Rate",
    "SpO2": "SpO₂",
    "HbA1c": "HbA1c",
    "Gender_encoded": "Gender (encoded)",
}


def load_model() -> None:
    """Load DNN model, scaler, and feature metadata into memory at startup."""
    global _model, _scaler, _features_meta, _locked_threshold, _model_version

    # Load feature metadata
    features_path = MODELS_DIR / "nephromind_features.json"
    if not features_path.exists():
        raise FileNotFoundError(f"Feature metadata not found: {features_path}")
    with open(features_path) as f:
        _features_meta = json.load(f)

    # Load scaler
    scaler_path = MODELS_DIR / "nephromind_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    _scaler = joblib.load(scaler_path)

    # Load DNN checkpoint
    model_path = MODELS_DIR / "nephromind_dnn.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"DNN checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"),
                             weights_only=False)
    n_features = checkpoint["n_features"]
    _model = NephroMindDNN(n_features=n_features)
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()
    _model_version = checkpoint.get("model_version", "nephromind_dnn_v1")

    # Load locked threshold
    threshold_path = MODELS_DIR / "dnn_threshold.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            t_data = json.load(f)
        _locked_threshold = float(t_data.get("locked_threshold", 0.5))
    else:
        _locked_threshold = 0.5

    print(f"[INFO] DNN loaded: {_model_version}, "
          f"features={_features_meta['feature_names']}, "
          f"threshold={_locked_threshold}")


def is_model_loaded() -> bool:
    return _model is not None and _scaler is not None and _features_meta is not None


def get_model_version() -> str:
    return _model_version


def get_locked_threshold() -> float:
    return _locked_threshold


def _build_feature_vector(request: PredictionRequest) -> np.ndarray:
    """
    Convert API request to ordered feature array matching training schema.
    Feature order: Age, BMI, Systolic_BP, Diastolic_BP, Heart_Rate, SpO2, HbA1c, Gender_encoded
    """
    gender_encoded = 1.0 if request.Gender.strip().capitalize() == "Male" else 0.0
    feature_names = _features_meta["feature_names"]

    feature_map = {
        "Age":            float(request.Age),
        "BMI":            float(request.BMI),
        "Systolic_BP":    float(request.Systolic_BP),
        "Diastolic_BP":   float(request.Diastolic_BP),
        "Heart_Rate":     float(request.Heart_Rate),
        "SpO2":           float(request.SpO2),
        "HbA1c":          float(request.HbA1c),
        "Gender_encoded": gender_encoded,
    }

    return np.array([[feature_map[f] for f in feature_names]], dtype=np.float32)


def _get_raw_input_values(request: PredictionRequest) -> Dict[str, float]:
    """Return the actual (unscaled) input values for each feature."""
    gender_encoded = 1.0 if request.Gender.strip().capitalize() == "Male" else 0.0
    return {
        "Age": float(request.Age),
        "BMI": float(request.BMI),
        "Systolic_BP": float(request.Systolic_BP),
        "Diastolic_BP": float(request.Diastolic_BP),
        "Heart_Rate": float(request.Heart_Rate),
        "SpO2": float(request.SpO2),
        "HbA1c": float(request.HbA1c),
        "Gender_encoded": gender_encoded,
    }


def _compute_contributions(
    X_scaled: np.ndarray, feature_names: List[str]
) -> Optional[List[Dict[str, Any]]]:
    """
    Compute feature contributions using Gradient × Input attribution
    on the SCALED feature vector.

    The gradient is computed with respect to the sigmoid probability output.
    Each contribution = (scaled_input_value) × (gradient of sigmoid probability
    w.r.t. that scaled input).

    Returns None if attribution fails, so the caller can display an honest
    "unavailable" state instead of fabricated results.
    """
    try:
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True)
        _model.eval()
        logit = _model(X_tensor)
        prob = torch.sigmoid(logit)
        prob.backward()
        grads = X_tensor.grad.detach().numpy()[0]
        # Input × gradient attribution (on scaled features)
        attrs = X_scaled[0] * grads

        contributions = [
            {
                "feature": feature_names[i],
                "display_name": FEATURE_DISPLAY_NAMES.get(feature_names[i], feature_names[i]),
                "contribution": round(float(attrs[i]), 6),
            }
            for i in range(len(feature_names))
        ]
        # Sort by absolute contribution magnitude (descending)
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return contributions
    except Exception as e:
        print(f"[WARN] Attribution computation failed: {e}")
        return None


def run_prediction(request: PredictionRequest) -> PredictionResponse:
    """Run DNN inference for a validated patient request."""
    if not is_model_loaded():
        load_model()

    feature_names = _features_meta["feature_names"]

    # 1. Build raw feature array
    X_raw = _build_feature_vector(request)

    # 2. Scale using training scaler (transform only)
    X_scaled = _scaler.transform(X_raw).astype(np.float32)

    # 3. DNN inference
    _model.eval()
    with torch.no_grad():
        logit = _model(torch.tensor(X_scaled))
        probability = float(torch.sigmoid(logit).item())

    probability = round(probability, 6)
    risk_score  = round(probability * 100.0, 1)

    # 4. Classification using locked threshold
    prediction = "Yes" if probability >= _locked_threshold else "No"

    # 5. Risk category
    if probability < _locked_threshold:
        risk_category = "Lower Risk"
    elif probability < _locked_threshold + 0.09:
        risk_category = "Moderate Risk"
    else:
        risk_category = "Higher Risk"

    # 6. Explainability — gradient × input attribution on scaled features
    #    w.r.t. sigmoid probability output (all 8 features returned)
    contributions_raw = _compute_contributions(X_scaled, feature_names)

    attribution_available = contributions_raw is not None

    if attribution_available:
        top_features = [
            FeatureContribution(
                feature=c["feature"],
                display_name=c["display_name"],
                contribution=c["contribution"],
            )
            for c in contributions_raw
        ]
    else:
        top_features = []

    # 7. Collect actual input values for transparency
    raw_inputs = _get_raw_input_values(request)

    # 8. Assessment timestamp
    assessment_ts = datetime.now(timezone.utc).isoformat()

    disclaimer = (
        "PROTOTYPE ONLY — NephroMind DNN calculates a prototype kidney disease risk score "
        "from physiological biomarkers (Renal Sentinel Station). "
        "It is NOT a clinical diagnosis and has NOT been clinically validated. "
        "Results must be reviewed by a qualified physician."
    )

    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 4),
        risk_score=risk_score,
        risk_category=risk_category,
        threshold=round(_locked_threshold, 4),
        model_version=_model_version,
        top_features=top_features,
        attribution_method="Gradient × Input (scaled features, w.r.t. sigmoid probability)"
            if attribution_available
            else "unavailable",
        attribution_available=attribution_available,
        input_values=raw_inputs,
        assessment_timestamp=assessment_ts,
        disclaimer=disclaimer,
    )
