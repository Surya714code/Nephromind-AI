# NephroMind AI — Dataset Summary
Generated: 2026-09-08T23:07:57.921509

## Dataset Overview
| Item | Value |
|------|-------|
| Source file | dataset_nephro.csv |
| Total rows in CSV | 11,419 |
| Valid patient records | 11,401 |
| Excluded audit-log rows | 18 |
| Invalid BP records (Sys==Dia) | 8 (converted to NaN) |

## Target Variable
- Column: `Kidney_Disease`
- Task: **Current kidney-disease status classification** (retrospective, cross-sectional)
- Positive class: `Yes`
- Negative class: `No`

| Class | Count | Percentage |
|-------|-------|------------|
| Yes (Kidney Disease) | 9,249 | 81.12% |
| No | 2,152 | 18.88% |

> ⚠️ **Scientific Limitation**: This dataset does NOT contain longitudinal follow-up data.
> This model classifies current status only. It MUST NOT be described as progression
> prediction, clinical diagnosis, or clinically validated prediction.

## Feature Set (Final — 13 features)
### Numeric Features (12)
- `Age`
- `BMI`
- `Systolic_BP`
- `Diastolic_BP`
- `Heart_Rate`
- `SpO2`
- `Daily_Steps`
- `Sleep_Duration_hr`
- `Activity_Minutes`
- `Resting_Heart_Rate`
- `HRV_ms`
- `HbA1c`

### Categorical Features (1)
- `Gender`

## Excluded Features
| Feature | Reason |
|---------|--------|
| Patient_ID | Identifier only |
| eGFR | **Direct target leakage** (Kidney_Disease = Yes if eGFR < 60) |
| UACR_mg_g | **Direct target leakage** (Kidney_Disease = Yes if UACR ≥ 30) |
| Creatinine_mg_dL | Provenance unresolved |
| Diabetes | Deterministically derived from HbA1c ≥ 6.5 |
| Diabetes_Pedigree | Synthetic sequential placeholder |
| Diabetes_Duration_Years | Documented inconsistent/unreliable |

## Train / Validation / Test Split
| Split | Records | Positive (Yes) | Positive % |
|-------|---------|----------------|------------|
| Train | 7,980 | 6,474 | 81.13% |
| Validation | 1,710 | 1,387 | 81.11% |
| Test | 1,711 | 1,388 | 81.12% |

- Stratified splits preserving class ratio
- Fixed random seed: 42
- No test-set information entered preprocessing, tuning, or threshold selection
