"""
NephroMind AI — Step 1: Dataset Inspection
==========================================
Inspects the raw dataset. Reports column types, value distributions,
class balance, and flags the audit rows without modifying anything.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
import pandas as pd

DATASET_PATH = Path("E:/dataset nephro.csv")
PATIENT_ID_PATTERN = re.compile(r"^P[0-9]{5}$")

DNN_FEATURES = ["Age", "Gender", "BMI", "Systolic_BP", "Diastolic_BP",
                 "Heart_Rate", "SpO2", "HbA1c"]
TARGET = "Kidney_Disease"

# ── Load raw ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("NephroMind AI — Dataset Inspection")
print("=" * 60)

raw = pd.read_csv(DATASET_PATH, dtype=str)
print(f"\nTotal rows (raw): {len(raw)}")
print(f"Total columns:    {len(raw.columns)}")
print(f"\nAll columns:\n  {list(raw.columns)}")

# ── Split patients vs audit ───────────────────────────────────────────────────
patient_mask = raw["Patient_ID"].str.fullmatch(r"^P[0-9]{5}$", na=False)
patients = raw.loc[patient_mask].copy()
audit    = raw.loc[~patient_mask].copy()

print(f"\nValid patient rows : {len(patients)}")
print(f"Audit / other rows : {len(audit)}")

if len(patients) == 11_401:
    print("  ✓ 11,401 patient records confirmed")
else:
    print(f"  ✗ Expected 11,401, found {len(patients)}")

if len(audit) == 18:
    print("  ✓ 18 audit-log rows confirmed")
else:
    print(f"  ✗ Expected 18, found {len(audit)}")

print(f"\nAudit row Patient_IDs:\n  {audit['Patient_ID'].tolist()}")

# ── Target distribution ───────────────────────────────────────────────────────
print(f"\n[{TARGET}] value counts:")
print(patients[TARGET].value_counts(dropna=False).to_string())
pos = (patients[TARGET] == "Yes").sum()
neg = (patients[TARGET] == "No").sum()
print(f"\n  Positive (Yes): {pos}  ({100*pos/len(patients):.1f}%)")
print(f"  Negative (No):  {neg}  ({100*neg/len(patients):.1f}%)")
print(f"  Imbalance ratio: {pos/neg:.1f}:1")

# ── Gender distribution ───────────────────────────────────────────────────────
print(f"\n[Gender] value counts:")
print(patients["Gender"].value_counts(dropna=False).to_string())

# ── DNN feature ranges ────────────────────────────────────────────────────────
print(f"\nDNN feature ranges (numeric, patient rows only):")
for col in [c for c in DNN_FEATURES if c != "Gender"]:
    s = pd.to_numeric(patients[col], errors="coerce")
    print(f"  {col:<20}: min={s.min():.1f}  max={s.max():.1f}  nulls={s.isna().sum()}")

# ── Leakage-excluded columns ──────────────────────────────────────────────────
EXCLUDED = ["Creatinine_mg_dL", "eGFR", "UACR_mg_g", "Diabetes",
            "Diabetes_Pedigree", "Diabetes_Duration_Years",
            "Daily_Steps", "Sleep_Duration_hr", "Activity_Minutes",
            "Resting_Heart_Rate", "HRV_ms"]
print(f"\nColumns intentionally excluded from DNN feature vector:")
for col in EXCLUDED:
    reason = (
        "clinical kidney marker (leakage risk)" if col in ["Creatinine_mg_dL","eGFR","UACR_mg_g"]
        else "deterministic from HbA1c" if col == "Diabetes"
        else "synthetic / non-clinical" if col == "Diabetes_Pedigree"
        else "unreliable synthetic field" if col == "Diabetes_Duration_Years"
        else "wearable — not available from Renal Sentinel hardware"
    )
    print(f"  {col:<30}: {reason}")

print(f"\n✓ INSPECTION COMPLETE")
print(f"  DNN feature set: {DNN_FEATURES}")
