"""
NephroMind AI — FastAPI Backend Application
============================================
Production-ready REST API serving the DNN inference engine and hardware ingestion.

Model: Deep Neural Network (MLP) — NephroMind DNN v1
Features: Age, Gender, BMI, Systolic_BP, Diastolic_BP, Heart_Rate, SpO2, HbA1c
Hardware: Renal Sentinel Station (ESP32 + BP module + MAX30102)
"""
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.schemas import PredictionRequest, PredictionResponse, HealthResponse, ClinicalMarkerAssessment
from backend.services.prediction_service import (
    load_model,
    is_model_loaded,
    get_model_version,
    get_locked_threshold,
    run_prediction,
)
from backend.services.clinical_marker_service import interpret_clinical_markers


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifact at backend startup."""
    try:
        load_model()
        print("[INFO] Model artifact successfully loaded at startup.")
    except Exception as e:
        print(f"[ERROR] Failed to load model artifact at startup: {e}")
    yield


app = FastAPI(
    title="NephroMind AI — Backend API",
    description="Clinical decision support inference service for Kidney Disease risk prediction.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Configuration for local frontend development (all Vite ports supported)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Clean validation error handler without exposing technical stack traces."""
    errors = exc.errors()
    details = []
    for err in errors:
        loc = " -> ".join([str(x) for x in err.get("loc", []) if str(x) != "body"])
        msg = err.get("msg", "Invalid value")
        details.append(f"{loc}: {msg}" if loc else msg)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Invalid patient data submitted",
            "message": "Please review the input fields. Required: Age, Gender, BMI, Systolic_BP, Diastolic_BP, Heart_Rate, SpO2, HbA1c (8 clinical features from Renal Sentinel Station).",
            "details": details,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler preventing stack trace exposure."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Prediction service error",
            "message": "An internal error occurred while processing the prediction request. Please try again or contact support.",
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Return backend operational status and model readiness."""
    if not is_model_loaded():
        try:
            load_model()
        except Exception:
            pass
    loaded = is_model_loaded()
    version = get_model_version() if loaded else "unavailable"
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        model_version=version,
        model_type="Deep Neural Network (MLP)",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_endpoint(request: PredictionRequest) -> PredictionResponse:
    """
    Run kidney disease risk prediction using the NephroMind Deep Neural Network.

    Accepts 8 physiological features from the Renal Sentinel Station:
    - Age, Gender, BMI, Systolic_BP, Diastolic_BP, Heart_Rate, SpO2, HbA1c

    Optional clinical marker fields (creatinine, egfr, uacr, urine_dipstick)
    are passed through to the rule-based clinical marker layer ONLY.
    They are NOT fed into the DNN to prevent target leakage.
    """
    if not is_model_loaded():
        try:
            load_model()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model artifact unavailable.",
            )

    try:
        response = run_prediction(request)

        # Add clinical marker layer (rule-based, separate from DNN)
        if any(v is not None for v in [request.creatinine, request.egfr, request.uacr, request.urine_dipstick]):
            marker_data = interpret_clinical_markers(
                creatinine=request.creatinine,
                egfr=request.egfr,
                uacr=request.uacr,
                urine_dipstick=request.urine_dipstick,
            )
            from backend.schemas import ClinicalMarkerAssessment
            response.clinical_markers = ClinicalMarkerAssessment(**marker_data)

        return response
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error executing prediction model.",
        )



@app.post("/ingest", tags=["Hardware Ingestion"])
async def hardware_ingest(payload: dict):
    """
    Renal Sentinel Station hardware ingestion endpoint.

    Accepts raw sensor data from the ESP32 microcontroller and immediately
    runs DNN prediction + clinical marker interpretation.

    Expected payload from ESP32:
    {
        "patient_id": "P12345",
        "age": 58,
        "gender": "Male",
        "bmi": 28.5,
        "systolic_bp": 135,
        "diastolic_bp": 85,
        "heart_rate": 76,
        "spo2": 97,
        "hba1c": 7.2,
        "urine_dipstick": "Negative"  (optional)
    }
    """
    if not is_model_loaded():
        try:
            load_model()
        except Exception:
            raise HTTPException(status_code=503, detail="Model not ready")

    try:
        # Map lowercase hardware keys to schema keys
        mapped = {
            "Age":         float(payload.get("age", payload.get("Age", 0))),
            "Gender":      str(payload.get("gender", payload.get("Gender", "Male"))),
            "BMI":         float(payload.get("bmi", payload.get("BMI", 0))),
            "Systolic_BP": float(payload.get("systolic_bp", payload.get("Systolic_BP", 0))),
            "Diastolic_BP": float(payload.get("diastolic_bp", payload.get("Diastolic_BP", 0))),
            "Heart_Rate":  float(payload.get("heart_rate", payload.get("Heart_Rate", 0))),
            "SpO2":        float(payload.get("spo2", payload.get("SpO2", 0))),
            "HbA1c":       float(payload.get("hba1c", payload.get("HbA1c", 0))),
            "urine_dipstick": payload.get("urine_dipstick"),
        }
        request = PredictionRequest(**mapped)
        response = run_prediction(request)

        # Clinical marker layer
        marker_data = interpret_clinical_markers(
            urine_dipstick=mapped.get("urine_dipstick")
        )
        from backend.schemas import ClinicalMarkerAssessment
        response.clinical_markers = ClinicalMarkerAssessment(**marker_data)

        return {
            "patient_id": payload.get("patient_id"),
            "hardware_source": "Renal Sentinel Station",
            "prediction": response.model_dump(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
