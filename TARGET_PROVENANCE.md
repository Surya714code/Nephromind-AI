# Target provenance and leakage investigation

## Investigation scope

Searched:

- Every repository source/configuration/documentation file, excluding generated dependency and build directories.
- The complete raw CSV, including its appended `DATA QUALITY AUDIT LOG`.
- The valid 11,401-row patient subset identified by `Patient_ID` matching `^P[0-9]{5}$`.

No backend, ML code, data-generation script, README, or external source documentation exists in the repository. The embedded audit log documents only `Diabetes_Pedigree`, `Diabetes_Duration_Years`, blood-pressure errors, and `Diabetes` label logic. It does **not** state how `Kidney_Disease` was generated.

## 1. Target definition

- Target column: `Kidney_Disease`
- Observed values: `Yes`, `No`
- Approved role: retrospective binary prototype target
- Valid-patient class distribution: **9,249 Yes (81.12%)** and **2,152 No (18.88%)**

This target is a contemporaneous kidney-disease label, not a longitudinal DKD progression outcome. The source contains no outcome date, follow-up interval, progression event, or future laboratory measurement.

## 2. Recovered target-generation evidence

Although no original generation code or written rule was found, exhaustive evaluation of every valid patient row yields this exact mapping:

```text
Kidney_Disease = "Yes" if (eGFR < 60) OR (UACR_mg_g >= 30)
Kidney_Disease = "No" otherwise
```

Evidence:

| Candidate rule | Exact matches | False positives | False negatives |
|---|---:|---:|---:|
| `eGFR < 60 OR UACR_mg_g >= 30` | 11,401 / 11,401 | 0 | 0 |
| `eGFR < 60` alone | 7,298 / 11,401 | 0 | 4,103 |
| `UACR_mg_g >= 30` alone | 10,035 / 11,401 | 0 | 1,366 |
| `eGFR < 60 OR UACR_mg_g >= 30 OR Creatinine_mg_dL >= 1.3` | 11,381 / 11,401 | 20 | 0 |

Boundary evidence supports the UACR criterion: three rows have `UACR_mg_g = 30` and all have `Kidney_Disease = Yes`. No row has `eGFR = 60`, so strict versus inclusive treatment at exactly 60 cannot be distinguished from this file. The exact equality of the two-variable rule across all rows is strong evidence that the label was deterministically constructed from these threshold criteria, rather than manually assigned record by record.

This is a recovered rule from data, not an original generator artifact. The exact source process, author, and whether the data were synthetic or copied from another system remain undocumented.

## 3. Provenance of kidney-related variables

| Variable | Directly used in recovered rule? | Leakage status | Evidence and decision |
|---|---|---|---|
| `eGFR` | Yes | **Confirmed direct target leakage** | Its `< 60` threshold is part of a rule that exactly reconstructs every target label. Exclude permanently from any model predicting this target. |
| `UACR_mg_g` | Yes | **Confirmed direct target leakage** | Its `>= 30` threshold is part of a rule that exactly reconstructs every target label. Exclude permanently from any model predicting this target. |
| `Creatinine_mg_dL` | No evidence of direct use | **Unknown indirect provenance** | The target is already reconstructed perfectly without creatinine. Adding `Creatinine_mg_dL >= 1.3` creates 20 false positives. No code/log states whether eGFR was derived from creatinine or whether both were generated from a common source. Do not classify creatinine as directly leaked, but keep its provenance conditional. |

No other kidney-related field exists in the dataset. There is no diagnosis date, CKD stage, dialysis indicator, renal medication, referral status, prior kidney-disease label, future laboratory result, or explicit progression outcome field.

## 4. Potential leakage and suspicious variables

### Confirmed exclusions

| Variable | Reason |
|---|---|
| `Patient_ID` | Identifier only. |
| `Diabetes_Pedigree` | Embedded audit log states it is a synthetic arithmetic sequence unrelated to patients. |
| `Diabetes` | Deterministic from `HbA1c >= 6.5`; generated/redundant label rather than an independent deployment feature. |
| `Diabetes_Duration_Years` | Embedded audit log states it was independently assigned and is inconsistent with the generated Diabetes label. |
| `eGFR` | Confirmed direct target construction input. |
| `UACR_mg_g` | Confirmed direct target construction input. |

### Conditional / unresolved variables

| Variable(s) | Status | Reason |
|---|---|---|
| `Creatinine_mg_dL` | Conditional, provenance unknown | Not needed for and not supported by the recovered rule. It may nevertheless be upstream of eGFR in an undisclosed source process. Require acceptance of this residual risk before use. |
| `Systolic_BP`, `Diastolic_BP` | Allowed after data-quality handling | Eight records intentionally have zero pulse pressure. Set only those BP values to missing during training-only preprocessing, then impute from training data. |
| Age, Gender, BMI, HbA1c, Heart_Rate, SpO2, Daily_Steps, Sleep_Duration_hr, Activity_Minutes, Resting_Heart_Rate, HRV_ms | No target-construction evidence found | They are candidate features only. Their actual collection time relative to the contemporaneous label is undocumented, so inference-time availability is not proven. |

The supplied wearable fields have no timestamp, device provenance, or encounter relationship. They do not directly reconstruct the target in this investigation, but should be treated as available-at-inference **unknown**, rather than assumed to be pre-diagnosis values.

## 5. Recommended feature configurations

Because direct target construction was recovered, a configuration that includes eGFR or UACR is invalid for model development against `Kidney_Disease` and must not be trained.

### Model A: conservative

Exclude `Patient_ID`, `Diabetes_Pedigree`, `Diabetes`, `Diabetes_Duration_Years`, `eGFR`, `UACR_mg_g`, and `Creatinine_mg_dL`.

Candidate features:

```text
Age, Gender, BMI, Systolic_BP, Diastolic_BP, Heart_Rate, SpO2,
Daily_Steps, Sleep_Duration_hr, Activity_Minutes,
Resting_Heart_Rate, HRV_ms, HbA1c
```

This is the safest configuration while creatinine provenance is unresolved.

### Model B: extended, conditional

Use the Model A features plus `Creatinine_mg_dL` **only after the project accepts the unresolved possibility that creatinine was upstream of eGFR or generated from the same undisclosed rule source**.

`eGFR` and `UACR_mg_g` remain excluded in Model B. Including either would reproduce target construction, not provide a valid prediction experiment.

## 6. Final recommendation

1. Treat `Kidney_Disease` as a deterministic clinical-criteria label in this dataset, not as an independently observed future outcome.
2. Permanently exclude `eGFR` and `UACR_mg_g` from all models evaluated against this target.
3. Start future training with Model A. Model B can be a clearly labeled sensitivity experiment only if creatinine's unresolved indirect-provenance risk is accepted.
4. Do not report either model as a DKD progression predictor or as clinically validated. A true progression model requires a temporally defined cohort and an outcome occurring after feature collection.
5. Preserve the approved BP handling: convert values in both BP fields to missing only where systolic equals diastolic for the eight documented patient IDs; fit any imputer solely on the training partition.

No model was trained, no feature was removed from the raw CSV, and no frontend file was modified during this investigation.
