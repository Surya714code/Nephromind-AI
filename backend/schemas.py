"""
NephroMind AI — Backend Data Schemas
=====================================
Pydantic schemas for the DNN inference endpoint.
Feature vector: Age, Gender, BMI, Systolic_BP, Diastolic_BP, Heart_Rate, SpO2, HbA1c
(8 clinical/physiological features from the Renal Sentinel Station hardware).

Intentionally excluded (not available from Renal Sentinel hardware):
- Daily_Steps, Sleep_Duration_hr, Activity_Minutes, Resting_Heart_Rate, HRV_ms

Intentionally excluded (leakage / non-clinical):
- Creatinine_mg_dL, eGFR, UACR_mg_g, Diabetes, Diabetes_Pedigree, Diabetes_Duration_Years
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    # Core Renal Sentinel physiological inputs
    Age: float = Field(..., description="Patient age in years", ge=1, le=120)
    Gender: str = Field(..., description="Gender: 'Male' or 'Female'")
    BMI: float = Field(..., description="Body Mass Index (kg/m²)", ge=10.0, le=70.0)
    Systolic_BP: float = Field(..., description="Systolic Blood Pressure in mmHg", ge=50.0, le=250.0)
    Diastolic_BP: float = Field(..., description="Diastolic Blood Pressure in mmHg", ge=30.0, le=150.0)
    Heart_Rate: float = Field(..., description="Heart Rate in bpm (from MAX30102)", ge=30.0, le=220.0)
    SpO2: float = Field(..., description="Blood Oxygen Saturation % (from MAX30102)", ge=70.0, le=100.0)
    HbA1c: float = Field(..., description="Glycated Hemoglobin %", ge=3.0, le=20.0)

    # Optional clinical marker inputs (NOT used in DNN; used in clinical-marker layer only)
    creatinine: Optional[float] = Field(None, description="Serum Creatinine mg/dL (clinical layer only)", ge=0.1, le=30.0)
    egfr: Optional[float] = Field(None, description="eGFR mL/min/1.73m² (clinical layer only)", ge=0.0, le=200.0)
    uacr: Optional[float] = Field(None, description="UACR mg/g (clinical layer only)", ge=0.0, le=10000.0)
    urine_dipstick: Optional[str] = Field(None, description="Urine dipstick screening result (e.g., 'Negative', 'Trace', '1+', '2+', '3+')")

    # Optional patient context
    patient_id: Optional[str] = Field(None, description="Patient identifier")

    @field_validator("Gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        formatted = str(v).strip().capitalize()
        if formatted not in ("Male", "Female"):
            raise ValueError("Gender must be 'Male' or 'Female'")
        return formatted


class FeatureContribution(BaseModel):
    feature: str = Field(..., description="Internal feature name")
    display_name: str = Field(default="", description="Human-readable feature name")
    contribution: float = Field(..., description="Signed attribution score (Gradient × Input on scaled features)")


class ClinicalMarkerAssessment(BaseModel):
    """
    Rule-based interpretation of kidney markers — separate from DNN prediction.
    These markers directly encode kidney status and are NOT fed into the DNN
    to avoid target leakage.
    """
    creatinine_mgdL: Optional[float] = None
    creatinine_status: Optional[str] = None
    egfr_ml_min: Optional[float] = None
    ckd_stage: Optional[str] = None
    uacr_mg_g: Optional[float] = None
    albuminuria_category: Optional[str] = None
    urine_dipstick: Optional[str] = None
    dipstick_note: Optional[str] = None
    marker_layer_note: str = (
        "Clinical marker assessment is rule-based and independent of the DNN prediction. "
        "These values directly reflect kidney function status and require physician interpretation."
    )


class PredictionResponse(BaseModel):
    # DNN AI prediction
    prediction: str = Field(..., description="'Yes' or 'No'")
    probability: float = Field(..., description="DNN sigmoid probability (0.0 to 1.0)")
    risk_score: float = Field(..., description="Risk Score (0 to 100)")
    risk_category: str = Field(..., description="'Lower Risk', 'Moderate Risk', or 'Higher Risk'")
    threshold: float = Field(..., description="Locked classification threshold")
    model_version: str = Field(..., description="DNN model artifact version")
    model_type: str = Field(default="Deep Neural Network (MLP)", description="AI model type")

    # Explainability
    top_features: List[FeatureContribution] = Field(
        default_factory=list, description="Feature attributions (Gradient × Input on scaled features)"
    )
    attribution_method: str = Field(
        default="unavailable",
        description="Description of the attribution method used"
    )
    attribution_available: bool = Field(
        default=False,
        description="Whether attribution was successfully computed"
    )

    # Actual input values for transparency
    input_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Actual (unscaled) input values submitted to the DNN"
    )

    # Assessment metadata
    assessment_timestamp: Optional[str] = Field(
        None, description="ISO-8601 timestamp of when this assessment was computed"
    )

    # Clinical marker layer (optional, rule-based, separate from DNN)
    clinical_markers: Optional[ClinicalMarkerAssessment] = Field(
        None, description="Rule-based clinical kidney marker interpretation (separate from DNN)"
    )

    # Metadata
    disclaimer: str = Field(..., description="Non-clinical prototype disclaimer")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    model_type: str = "Deep Neural Network (MLP)"
