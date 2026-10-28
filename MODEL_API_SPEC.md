# NephroMind AI — Model & API Specification

**Model Name**: NephroMind Deep Neural Network (`NephroMindDNN`)  
**Framework**: PyTorch 2.x  
**Model Type**: Multilayer Perceptron (MLP) with Batch Normalization & Dropout  
**Baseline Model**: Gradient Boosting Classifier (for benchmarking only)  
**Hardware Source**: Renal Sentinel Station (MAX30102 PPG + BP Cuff + Clinical Intake)  

---

## 1. Feature Architecture

### 1.1 Primary DNN Feature Vector (8 Features)
| Index | Feature | Source | Type | Range / Description |
|---|---|---|---|---|
| 0 | `Age` | Clinical Intake | Float | Patient age in years |
| 1 | `BMI` | Clinical Intake | Float | Body mass index (kg/m²) |
| 2 | `Systolic_BP` | Renal Sentinel Hardware | Float | Systolic blood pressure (mmHg) |
| 3 | `Diastolic_BP` | Renal Sentinel Hardware | Float | Diastolic blood pressure (mmHg) |
| 4 | `Heart_Rate` | MAX30102 PPG Sensor | Float | Pulse rate in bpm |
| 5 | `SpO2` | MAX30102 PPG Sensor | Float | Blood oxygen saturation % |
| 6 | `HbA1c` | Clinical Lab / POC | Float | Glycated hemoglobin % |
| 7 | `Gender_encoded` | Clinical Intake | Int/Float | 1 for Male, 0 for Female |

### 1.2 Target Leakage Prevention & Explicit Exclusions
The following variables are **strictly excluded** from the neural network input:
- `Creatinine_mg_dL`, `eGFR`, `UACR_mg_g`, `Kidney_Disease_Stage` (Direct definitions/proxies of kidney damage)
- `Diabetes`, `Diabetes_Duration_Years`, `Diabetes_Pedigree` (Direct diagnostic labels)
- `Daily_Steps`, `Sleep_Duration_hr`, `Activity_Minutes`, `Resting_Heart_Rate`, `HRV_ms` (Not captured by Renal Sentinel hardware)

### 1.3 Separate Clinical Marker Evaluation Layer (Rule-Based)
Kidney-specific lab markers are evaluated through a **rule-based clinical decision support layer** rather than fed to the ML model:
- **eGFR**: Categorized into KDIGO CKD Stages 1 through 5
- **Creatinine**: Flagged as Normal vs Elevated (> 1.2 mg/dL)
- **UACR**: Categorized into A1 (Normal), A2 (Microalbuminuria), A3 (Macroalbuminuria)
- **Urine Dipstick**: Categorized by protein trace levels

---

## 2. Model Architecture & Hyperparameters

```
Input: [Batch, 8]
  ↓
Linear(8, 128) → ReLU → BatchNorm1d(128) → Dropout(0.30)
  ↓
Linear(128, 64) → ReLU → BatchNorm1d(64) → Dropout(0.25)
  ↓
Linear(64, 32)  → ReLU → Dropout(0.20)
  ↓
Linear(32, 16)  → ReLU
  ↓
Linear(16, 1)   → Logit Output
  ↓
Sigmoid Activation → Risk Probability p ∈ [0, 1]
```

- **Loss Function**: `BCEWithLogitsLoss(pos_weight=0.23)` (Accounting for 81.1% class imbalance)
- **Optimizer**: AdamW (`lr=0.001`, `weight_decay=1e-4`)
- **LR Scheduler**: `ReduceLROnPlateau(factor=0.5, patience=5)`
- **Explainability**: Gradient × Input ($x_i \cdot \frac{\partial \hat{y}}{\partial x_i}$) computed dynamically in PyTorch during forward pass.

---

## 3. Verified Performance Metrics

| Metric | Primary DNN (PyTorch) | Baseline Gradient Boosting |
|---|---|---|
| **ROC-AUC** | **0.749** | 0.762 |
| **PR-AUC** | **0.923** | 0.921 |
| **Precision** | **0.931** | 0.812 |
| **Specificity** | **0.799 (Discriminative)** | 0.006 (Collapses to predicting all positive) |
| **Sensitivity / Recall** | **0.629** | 0.997 |
| **Locked Threshold** | **0.56** (Youden's J Optimized) | 0.36 |
