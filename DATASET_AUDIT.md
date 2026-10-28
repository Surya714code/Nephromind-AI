# Dataset audit: dataset nephro.csv

## Scope and source

- Source file: `E:\dataset nephro.csv`
- Raw CSV dimensions when loaded literally: **11,419 rows × 21 columns**
- Valid patient records identified by `Patient_ID` matching `P` followed by five digits: **11,401 rows**
- Non-patient rows: **18 appended audit-log/metadata rows**
- Full-row duplicates: **0**
- Duplicate `Patient_ID` values in the literal CSV: **3**, all caused by repeated `Description` labels in the appended metadata; valid patient IDs are unique (**11,401 / 11,401**).

The raw CSV is not a clean rectangular patient table. Its final 18 rows are a human-readable `DATA QUALITY AUDIT LOG` appended using the same 21 columns. These rows cause `Age` and `BMI` to load as text and create apparent missing values. They must be retained in the raw source but explicitly excluded by a future ingestion rule, not silently dropped.

## Columns and observed types

| Column | Literal CSV type | Valid patient type | Unique values in valid patient rows | Missing in valid patient rows |
|---|---:|---:|---:|---:|
| Patient_ID | text | identifier | 11,401 | 0 |
| Age | text due to appended log | integer | 68 | 0 |
| Gender | text | categorical | 2 | 0 |
| BMI | text due to appended log | numeric | 352 | 0 |
| Diabetes_Pedigree | numeric | numeric | 11,401 | 0 |
| Diabetes_Duration_Years | numeric | numeric | 31 | 0 |
| Creatinine_mg_dL | numeric | numeric | 220 | 0 |
| eGFR | numeric | numeric | 6,295 | 0 |
| UACR_mg_g | numeric | numeric | 5,885 | 0 |
| Systolic_BP | numeric | numeric | 117 | 0 |
| Diastolic_BP | numeric | numeric | 90 | 0 |
| Heart_Rate | numeric | numeric | 79 | 0 |
| SpO2 | numeric | numeric | 10 | 0 |
| Daily_Steps | numeric | numeric | 7,360 | 0 |
| Sleep_Duration_hr | numeric | numeric | 81 | 0 |
| Activity_Minutes | numeric | numeric | 229 | 0 |
| Resting_Heart_Rate | numeric | numeric | 71 | 0 |
| HRV_ms | numeric | numeric | 111 | 0 |
| HbA1c | numeric | numeric | 77 | 0 |
| Diabetes | text | binary categorical | 2 | 0 |
| Kidney_Disease | text | binary categorical | 2 | 0 |

There are no constant columns in the 11,401 valid patient rows. The literal file has 18 missing values in most clinical fields only because of the appended metadata rows; patient rows themselves have no missing values.

## Candidate targets and problem definition

Two binary label-like columns are present:

- `Diabetes`: **6,052 Yes (53.08%)**, **5,349 No (46.92%)**.
- `Kidney_Disease`: **9,249 Yes (81.12%)**, **2,152 No (18.88%)**.

`Diabetes` must **not** be a model target. The appended dataset log states that it was defined exactly as `HbA1c >= 6.5`; the audit confirmed **0 mismatches**. Training this target with HbA1c would be direct label leakage.

`Kidney_Disease` is the only plausible target for a binary kidney-disease classification prototype. It is materially imbalanced, with `Yes` as the majority class. However, it represents an observed **current disease label** in this cross-sectional dataset. The file contains no longitudinal outcome, progression time, index date, or future follow-up label. Therefore it does **not** support a valid claim that a model predicts DKD progression or progression time. It could only be evaluated as a retrospective binary classification prototype unless a properly defined future outcome is supplied.

The dataset also contains patients without the deterministic Diabetes label who have `Kidney_Disease=Yes` (3,550 rows). A DKD-only cohort or DKD-specific target definition cannot be inferred from this file.

## Class distribution and association checks

| Diabetes | Kidney_Disease=No | Kidney_Disease=Yes | Total |
|---|---:|---:|---:|
| No | 1,799 | 3,550 | 5,349 |
| Yes | 353 | 5,699 | 6,052 |
| Total | 2,152 | 9,249 | 11,401 |

The strongest point-biserial correlations with `Kidney_Disease=Yes` among numeric fields were UACR_mg_g (**0.439**), Creatinine_mg_dL (**0.382**), eGFR (**-0.407**), and HbA1c (**0.295**). These are associations in this dataset, not evidence of causality or clinical validity.

## Ranges and data-quality findings

Observed valid-patient ranges:

| Feature | Minimum | Maximum | Audit note |
|---|---:|---:|---|
| Age | 12 | 85 | Pediatric records are present. No age eligibility criterion is supplied. |
| BMI | 13.0 | 52.4 | Extremes require source review, not automatic deletion. |
| Diabetes_Pedigree | 0.150 | 11.550 | See confirmed placeholder leakage below. |
| Diabetes_Duration_Years | 0 | 30 | See confirmed consistency issue below. |
| Creatinine_mg_dL | 0.30 | 2.89 | High-end values should be retained unless source rules are supplied. |
| eGFR | 4.0 | 190.0 | 2,593 rows equal 4.0 and 994 equal 190.0; this suggests boundary clipping or synthetic generation and needs provenance review. |
| UACR_mg_g | 5.00 | 335.97 | Right-tail observations are numerous; do not remove by IQR alone. |
| Systolic_BP | 80 | 206 | Eight invalid zero-pulse-pressure readings are confirmed below. |
| Diastolic_BP | 45 | 152 | One value exceeds the usual IQR upper bound; no automatic deletion is justified. |
| Heart_Rate | 45 | 127 | Some IQR outliers; not automatically invalid. |
| SpO2 | 91 | 100 | 12 IQR outliers on the low side; not automatically invalid. |
| Daily_Steps | 200 | 20,000 | High-end values require no action without device context. |
| Sleep_Duration_hr | 3.0 | 11.0 | IQR outliers exist; retain pending protocol. |
| Activity_Minutes | 0 | 240 | Zero may be a valid observation. |
| Resting_Heart_Rate | 40 | 110 | IQR outliers; retain pending protocol. |
| HRV_ms | 10 | 120 | IQR outliers; retain pending protocol. |
| HbA1c | 4.5 | 12.5 | High-end values are clinically plausible enough to require review, not deletion. |

## Confirmed leakage and integrity risks

1. **Appended metadata rows**. The raw file includes 18 non-patient rows. Future ingestion must validate `Patient_ID` before type conversion. Recommended rule: include only IDs matching `^P[0-9]{5}$`, and record that 18 metadata rows were excluded.
2. **`Diabetes_Pedigree` is unusable.** The embedded audit log states it is a fixed arithmetic sequence beginning at 0.150 with +0.001 per row and has no patient meaning. The audit confirmed near-perfect correlation with sequential patient ID (**r ≈ 1.0**). It must be excluded from features as an identifier-derived synthetic placeholder.
3. **`Diabetes` is deterministically derived from HbA1c.** It cannot be a target when HbA1c is present. As a potential feature for `Kidney_Disease`, it is redundant with HbA1c and has generated-label provenance. The future feature specification should exclude it by default unless a deployment-time source and clinical rationale are established.
4. **`Diabetes_Duration_Years` conflicts with the generated Diabetes label.** The embedded log reports 5,153 `Diabetes=No` records with duration >0 and 205 `Diabetes=Yes` records with duration =0. It is not coherent with the supplied diabetes status and should be excluded or explicitly treated as unreliable until data provenance is corrected.
5. **Eight intentionally invalid blood-pressure pairs.** The embedded audit log names P01227, P02527, P02943, P03014, P03987, P06324, P08836, and P10495 as rows where systolic equals diastolic (zero pulse pressure). The audit confirmed eight such records. Do not delete their complete records; the later cleaning policy should set only the affected BP values to missing and impute within the training pipeline, if approved.
6. **Potential target-definition leakage.** eGFR, creatinine, and UACR are strongly associated with the existing disease label and may be part of how the label was assigned. No label-generation process for `Kidney_Disease` is included. These fields are clinically relevant but may yield optimistic retrospective performance if the label itself was derived from them. This must be resolved with dataset provenance before selecting a deployment model.
7. **No temporal validation is possible.** No dates, encounters, or follow-up outcomes exist. The dataset cannot establish feature availability before a future progression event.

## Cleaning decisions proposed for approval (not performed)

- Preserve `E:\dataset nephro.csv` unchanged.
- In a future ingestion script, explicitly retain 11,401 rows with valid patient IDs and log exclusion of 18 appended metadata rows.
- Use `Patient_ID` for uniqueness validation only, never as a model input.
- Exclude `Diabetes_Pedigree` from modeling.
- Exclude `Diabetes` by default because it is deterministically generated from HbA1c.
- Exclude `Diabetes_Duration_Years` pending correction or explicit acceptance of the known inconsistency.
- For the eight zero-pulse-pressure records, retain patient rows but replace only BP fields with missing values before a training-only imputer. This is a proposed policy, not executed.
- Do not remove statistical outliers solely by IQR. Validate any clinical bounds with a source owner first.

## Audit conclusion: training gate

**Training should not begin yet.** The file documents that at least one feature is synthetic and one label is deterministic, while the intended DKD progression target is absent. Before training, the project needs an explicit decision on:

1. Whether `Kidney_Disease` is an acceptable retrospective prototype target, despite not measuring DKD progression.
2. Whether the feature exclusions and BP missing-value policy above are approved.
3. Whether eGFR, creatinine, and UACR were used to define `Kidney_Disease`; if so, whether their inclusion is permitted for the intended use case.
4. The allowed patient age range and whether pediatric rows should be retained.
