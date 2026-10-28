# NephroMind AI — Backend API Documentation & Integration Guide

**Version**: `nephromind_dnn_v1`  
**Model**: Deep Neural Network (PyTorch MLP) — `models/nephromind_dnn.pt`  
**Backend Framework**: FastAPI + Uvicorn + Pydantic v2  
**Locked Decision Threshold**: `0.56` (Optimized on Validation via Youden's J)  
**Input Features**: 8 physiological & clinical features from the Renal Sentinel Station

---

## 1. Architecture Overview

NephroMind AI connects directly to the **Renal Sentinel Station** hardware (MAX30102 PPG sensor, BP cuff, and MCU):

```
Renal Sentinel Station (Hardware)
   [BP + Heart Rate + SpO2]
             ↓
Backend REST API (FastAPI)
             ↓
Preprocessing & Feature Scaling (StandardScaler)
             ↓
Primary Model: PyTorch Deep Neural Network (MLP)
   [8 Features → 128 → 64 → 32 → 16 → 1]
             ↓
Sigmoid Activation & Locked Threshold (0.56)
             ↓
Explainability: Gradient × Input Attributions
             +
Clinical Layer: Rule-Based Kidney Marker Evaluation (KDIGO / Albuminuria)
             ↓
React + Vite Clinician Dashboard
```

---

## 2. API Endpoints Specification

### 2.1 Health Check
- **Endpoint**: `GET /health`
- **Description**: Returns operational status and model readiness.
- **Response (`HTTP 200 OK`)**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "nephromind_dnn_v1",
  "model_type": "Deep Neural Network (MLP)"
}
```

---

### 2.2 Model Risk Prediction (DNN)
- **Endpoint**: `POST /predict`
- **Description**: Executes current kidney disease risk inference using the 8 Renal Sentinel features, with optional clinical markers evaluated in a separate rule-based layer.
- **Request Headers**: `Content-Type: application/json`

#### Request Body Schema (8 Renal Sentinel Features + Optional Clinical Markers)
```json
{
  "Age": 58.0,
  "Gender": "Male",
  "BMI": 28.5,
  "Systolic_BP": 142.0,
  "Diastolic_BP": 88.0,
  "Heart_Rate": 76.0,
  "SpO2": 96.0,
  "HbA1c": 7.8,
  "creatinine": 1.4,
  "egfr": 52.0,
  "uacr": 85.0,
  "urine_dipstick": "1+"
}
```

#### Successful Response (`HTTP 200 OK`)
```json
{
  "prediction": "Yes",
  "probability": 0.7762,
  "risk_score": 77.6,
  "risk_category": "Higher Risk",
  "threshold": 0.56,
  "model_version": "nephromind_dnn_v1",
  "model_type": "Deep Neural Network (MLP)",
  "top_features": [
    {
      "feature": "HbA1c",
      "contribution": 0.2845
    },
    {
      "feature": "Systolic_BP",
      "contribution": 0.2211
    },
    {
      "feature": "Age",
      "contribution": 0.1834
    }
  ],
  "clinical_markers": {
    "creatinine_mgdL": 1.4,
    "creatinine_status": "Elevated (> 1.2 mg/dL)",
    "egfr_ml_min": 52.0,
    "ckd_stage": "Stage 3a (Mild-Moderate decrease: 45–59 mL/min)",
    "uacr_mg_g": 85.0,
    "albuminuria_category": "A2: Microalbuminuria (30–300 mg/g)",
    "urine_dipstick": "1+ (Protein detected: ~30 mg/dL)"
  },
  "disclaimer": "PROTOTYPE ONLY. This model calculates a risk score from physiological biomarkers. It is NOT a clinical diagnostic tool and has NOT been clinically validated."
}
```

---

### 2.3 Renal Sentinel Hardware Ingestion
- **Endpoint**: `POST /ingest`
- **Description**: Direct hardware telemetry ingestion endpoint for ESP32 / Arduino microcontrollers deployed in the Renal Sentinel Station.
- **Request Body**:
```json
{
  "patient_id": "P12345",
  "age": 62,
  "gender": "Female",
  "bmi": 31.2,
  "systolic_bp": 155,
  "diastolic_bp": 95,
  "heart_rate": 82,
  "spo2": 94,
  "hba1c": 8.1,
  "urine_dipstick": "2+"
}
```

---

## 3. Running the Server

```bash
# Start FastAPI backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Start React Frontend
npm run dev
```
