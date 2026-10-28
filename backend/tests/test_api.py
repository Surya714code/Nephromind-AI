"""
NephroMind AI — Backend API Test Suite
======================================
Tests for health check, valid inference, input validation, and model loading.

Feature vector: Age, Gender, BMI, Systolic_BP, Diastolic_BP, Heart_Rate, SpO2, HbA1c
Locked threshold: 0.56
"""
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.main import app

# Correct 8-feature sample matching the Renal Sentinel DNN input vector
VALID_SAMPLE = {
    "Age": 58.0,
    "Gender": "Female",
    "BMI": 31.2,
    "Systolic_BP": 145.0,
    "Diastolic_BP": 92.0,
    "Heart_Rate": 88.0,
    "SpO2": 95.0,
    "HbA1c": 7.8,
}

LOCKED_THRESHOLD = 0.56


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """Test GET /health returns status healthy and model_loaded=True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_version" in data
    assert data["model_type"] == "Deep Neural Network (MLP)"


def test_valid_prediction(client):
    """Test valid POST /predict request returns structured prediction."""
    response = client.post("/predict", json=VALID_SAMPLE)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in ("Yes", "No")
    assert 0.0 <= data["probability"] <= 1.0
    assert 0.0 <= data["risk_score"] <= 100.0
    assert data["risk_category"] in ("Lower Risk", "Moderate Risk", "Higher Risk")
    assert data["threshold"] == pytest.approx(LOCKED_THRESHOLD, abs=0.01)
    assert isinstance(data["top_features"], list)
    assert "disclaimer" in data

    # Verify attribution metadata
    assert "attribution_method" in data
    assert "attribution_available" in data
    assert "input_values" in data
    assert "assessment_timestamp" in data

    # Verify input values are echoed back
    assert data["input_values"]["Age"] == 58.0
    assert data["input_values"]["HbA1c"] == 7.8


def test_attribution_completeness(client):
    """Test that all 8 features are returned in attribution when available."""
    response = client.post("/predict", json=VALID_SAMPLE)
    assert response.status_code == 200
    data = response.json()

    if data["attribution_available"]:
        assert len(data["top_features"]) == 8
        feature_names = {f["feature"] for f in data["top_features"]}
        expected = {"Age", "BMI", "Systolic_BP", "Diastolic_BP",
                    "Heart_Rate", "SpO2", "HbA1c", "Gender_encoded"}
        assert feature_names == expected

        # Verify each contribution has display_name and signed value
        for f in data["top_features"]:
            assert "display_name" in f
            assert isinstance(f["contribution"], (int, float))

        # Verify sorted by absolute contribution (descending)
        abs_values = [abs(f["contribution"]) for f in data["top_features"]]
        assert abs_values == sorted(abs_values, reverse=True)


def test_prediction_consistency(client):
    """Test that the same input produces the same output."""
    resp1 = client.post("/predict", json=VALID_SAMPLE)
    resp2 = client.post("/predict", json=VALID_SAMPLE)
    d1 = resp1.json()
    d2 = resp2.json()

    assert d1["probability"] == d2["probability"]
    assert d1["prediction"] == d2["prediction"]
    assert d1["risk_score"] == d2["risk_score"]
    assert d1["risk_category"] == d2["risk_category"]


def test_clinical_markers_separate(client):
    """Test that clinical markers are only populated when provided and stay separate from DNN."""
    # Without clinical markers
    resp_no_markers = client.post("/predict", json=VALID_SAMPLE)
    d1 = resp_no_markers.json()
    assert d1.get("clinical_markers") is None

    # With clinical markers
    sample_with_markers = {
        **VALID_SAMPLE,
        "creatinine": 1.8,
        "egfr": 38.0,
    }
    resp_with_markers = client.post("/predict", json=sample_with_markers)
    d2 = resp_with_markers.json()
    assert d2["clinical_markers"] is not None
    assert d2["clinical_markers"]["creatinine_mgdL"] == 1.8
    assert d2["clinical_markers"]["egfr_ml_min"] == 38.0
    assert "G3b" in d2["clinical_markers"]["ckd_stage"]

    # DNN probability should be the same regardless of clinical markers
    assert d1["probability"] == d2["probability"]


def test_missing_feature(client):
    """Test POST /predict with missing required feature yields 422 error."""
    invalid_sample = VALID_SAMPLE.copy()
    del invalid_sample["HbA1c"]

    response = client.post("/predict", json=invalid_sample)
    assert response.status_code == 422
    data = response.json()
    assert "error" in data


def test_invalid_data_type(client):
    """Test POST /predict with invalid data type yields 422 error."""
    invalid_sample = VALID_SAMPLE.copy()
    invalid_sample["Age"] = "invalid_age_string"

    response = client.post("/predict", json=invalid_sample)
    assert response.status_code == 422
    data = response.json()
    assert "error" in data


def test_invalid_gender_value(client):
    """Test POST /predict with invalid gender string yields 422 error."""
    invalid_sample = VALID_SAMPLE.copy()
    invalid_sample["Gender"] = "Other"

    response = client.post("/predict", json=invalid_sample)
    assert response.status_code == 422
