# NephroMind AI — Final Model Quality & Threshold Assessment

**Generated**: 2026-09-08T23:27:26+05:30  
**Model Artifact**: `E:\DEEP LEARN NEPHRO\ml\artifacts\nephromind_model_v1.joblib`  
**Selected Model Architecture**: Gradient Boosting (Tuned)  
**Locked Threshold**: $t = 0.79$ (Selected strictly on Validation Set; Test Set evaluated ONCE)

---

## 1. Dataset & Feature Engineering

- **Total Ingested Records**: 11,401 valid patient records.
- **Split Ratio**: Train (7,980 records, 70%), Validation (1,710 records, 15%), Test (1,711 records, 15%).
- **Strict Target Leakage Exclusion**: Excluded direct mathematical definitions of Kidney Disease (`UACR_mg_g`, `eGFR_mL_min_1_73m2`, `Serum_Creatinine_mg_dL`, `Kidney_Disease_Stage`, `Diabetes`).
- **Final Feature Set (13 indirect biomarkers)**: `Age`, `Gender`, `BMI`, `Systolic_BP`, `Diastolic_BP`, `Heart_Rate`, `SpO2`, `Daily_Steps`, `Sleep_Duration_hr`, `Activity_Minutes`, `Resting_Heart_Rate`, `HRV_ms`, `HbA1c`.

---

## 2. Model Selection & Comparison

During validation model comparison:
- **Gradient Boosting (Tuned)** achieved the highest discrimination (ROC-AUC = **0.7738**, PR-AUC = **0.9278**), making it the superior architecture for continuous risk ranking.
- **XGBoost & Logistic Regression** naturally produced balanced binary classifications under standard 0.50 thresholding (Specificity ~71%, Recall ~70%), making them strong alternative architectures for raw uncalibrated binary screening.

---

## 3. Validation Threshold Analysis & Correction

Initial F1 optimization resulted in $t = 0.36$, which caused severe specificity collapse (Specificity = 0.62%) due to high positive prevalence (81.1%).

To correct this, a validation-only threshold sweep was conducted. The threshold was locked at **$t = 0.79$** based on Youden's J statistic ($J = \text{Sensitivity} + \text{Specificity} - 1$).

### Validation Set Metrics at $t = 0.79$:
- **Accuracy**: 0.6713
- **Precision**: 0.9421
- **Sensitivity**: 0.6337
- **Specificity**: 0.8328
- **F1-Score**: 0.7578
- **Confusion Matrix**: TN=269, FP=54, FN=508, TP=879

---

## 4. Final Untouched Test Set Evaluation (Evaluated ONCE)

Evaluating the locked threshold ($t = 0.79$) on the hold-out test set ($N = 1,711$ records):

### Performance Metrics Summary

| Metric | Empirical Value |
| :--- | :---: |
| **Accuracy** | **0.676213** (67.62%) |
| **Precision** | **0.949353** (94.94%) |
| **Recall (Sensitivity)** | **0.634726** (63.47%) |
| **Specificity** | **0.854489** (85.45%) |
| **F1-Score** | **0.760794** (76.08%) |
| **ROC-AUC** | **0.761810** |
| **PR-AUC** | **0.921389** |
| **False Positive Rate** | **0.145511** |
| **False Negative Rate** | **0.365274** |

### Confusion Matrix (Test Set, $t = 0.79$)

- **True Positive (TP)**: `881`
- **True Negative (TN)**: `276`
- **False Positive (FP)**: `47`
- **False Negative (FN)**: `507`

---

## 5. Model Output & Empirical Risk Bands

To avoid misleading clinical claims (e.g. "82% chance of kidney disease"), outputs are designated as **"Model Risk Score"** / **"Predicted Probability"**.

Empirical risk bands derived from the validation probability distribution:
- **Low Risk Score**: Probability $< 0.79$
- **Moderate Risk Score**: $0.79 \le \text{Probability} < 0.88$
- **High Risk Score**: $\text{Probability} \ge 0.88$

---

## 6. Architecture Comparison: Gradient Boosting vs. XGBoost

- **For Continuous Risk Scoring**: **Gradient Boosting** is superior (ROC-AUC 0.7618 vs 0.7538 for XGBoost).
- **For Binary Classification**: Re-thresholded Gradient Boosting ($t = 0.79$) provides balanced Sensitivity (63.5%) and Specificity (85.4%). Native XGBoost at $t=0.50$ provides Specificity ~71.2% and Sensitivity ~69.9%.

---

## 7. Deployment Suitability & Disclaimers

- **Continuous Prototype Risk Scoring**: **SUITABLE.** The model exhibits strong ranking discrimination (ROC-AUC = 0.7618).
- **Binary Prototype Classification**: **SUITABLE (with locked $t = 0.79$).** Re-thresholding successfully balances Sensitivity and Specificity.
- **Clinical Validation**: **NOT CLINICALLY VALIDATED.** This model is a software prototype built on cross-sectional registry data and MUST NOT be used as a standalone diagnostic tool.
