"""
NephroMind AI — Step 2: Data Preparation
=========================================
Loads, cleans, and splits the dataset for DNN training.
Saves:
  models/X_train.npy, X_val.npy, X_test.npy
  models/y_train.npy, models/y_val.npy, models/y_test.npy
  models/nephromind_scaler.pkl
  models/nephromind_features.json

Rules:
- Scaler fitted ONLY on training data
- Stratified 70/15/15 split
- Gender encoded: Male=1, Female=0
- No leakage columns included
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
DATASET_PATH = Path("E:/dataset nephro.csv")
MODELS_DIR   = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# ── Feature definitions ───────────────────────────────────────────────────────
NUMERIC_FEATURES = ["Age", "BMI", "Systolic_BP", "Diastolic_BP",
                    "Heart_Rate", "SpO2", "HbA1c"]
CATEGORICAL_FEATURES = ["Gender"]
DNN_FEATURES_ORDERED = ["Age", "BMI", "Systolic_BP", "Diastolic_BP",
                         "Heart_Rate", "SpO2", "HbA1c", "Gender_encoded"]
TARGET = "Kidney_Disease"

print("=" * 60)
print("NephroMind AI — Data Preparation")
print("=" * 60)

# ── 1. Load raw dataset ───────────────────────────────────────────────────────
raw = pd.read_csv(DATASET_PATH, dtype=str)
print(f"Raw rows loaded: {len(raw)}")

# ── 2. Filter valid patient records (remove 18 audit rows) ────────────────────
patient_mask = raw["Patient_ID"].str.fullmatch(r"^P[0-9]{5}$", na=False)
df = raw.loc[patient_mask].copy()
audit_count = (~patient_mask).sum()
print(f"Audit rows removed: {audit_count}")
print(f"Valid patient rows: {len(df)}")
assert len(df) == 11_401, f"Expected 11401, got {len(df)}"
assert audit_count == 18, f"Expected 18 audit rows, got {audit_count}"

# ── 3. Filter target to valid values ─────────────────────────────────────────
valid_target = df[TARGET].isin(["Yes", "No"])
invalid_count = (~valid_target).sum()
if invalid_count:
    print(f"  Dropping {invalid_count} rows with invalid target values")
df = df.loc[valid_target].copy()
print(f"Rows after target filter: {len(df)}")

# ── 4. Filter valid Gender values ────────────────────────────────────────────
valid_gender = df["Gender"].isin(["Male", "Female"])
invalid_gender = (~valid_gender).sum()
if invalid_gender:
    print(f"  Dropping {invalid_gender} rows with invalid Gender values")
df = df.loc[valid_gender].copy()
print(f"Rows after gender filter: {len(df)}")

# ── 5. Convert numeric features ──────────────────────────────────────────────
for col in NUMERIC_FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Check for nulls
null_counts = df[NUMERIC_FEATURES].isna().sum()
if null_counts.any():
    print(f"\nNull values found (will be median-imputed from training set):")
    print(null_counts[null_counts > 0].to_string())

# ── 6. Encode Gender: Male=1, Female=0 ───────────────────────────────────────
df["Gender_encoded"] = (df["Gender"] == "Male").astype(float)

# ── 7. Encode target: Yes=1, No=0 ────────────────────────────────────────────
df["target"] = (df[TARGET] == "Yes").astype(int)

print(f"\nClass distribution:")
print(f"  Positive (Yes=1): {df['target'].sum()}  ({100*df['target'].mean():.1f}%)")
print(f"  Negative (No=0):  {(df['target']==0).sum()}  ({100*(1-df['target'].mean()):.1f}%)")

# ── 8. Build feature matrix ───────────────────────────────────────────────────
feature_cols = ["Age", "BMI", "Systolic_BP", "Diastolic_BP",
                "Heart_Rate", "SpO2", "HbA1c", "Gender_encoded"]
X = df[feature_cols].values.astype(np.float32)
y = df["target"].values.astype(np.int64)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Feature order:        {feature_cols}")

# ── 9. Check for impossible values ────────────────────────────────────────────
print("\nFeature range check:")
for i, col in enumerate(feature_cols):
    if col == "Gender_encoded":
        continue
    col_vals = X[:, i]
    print(f"  {col:<20}: min={col_vals[~np.isnan(col_vals)].min():.1f}  "
          f"max={col_vals[~np.isnan(col_vals)].max():.1f}  "
          f"nulls={np.isnan(col_vals).sum()}")

# ── 10. Stratified 70/15/15 split ─────────────────────────────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
)
val_fraction_of_remaining = 0.15 / 0.85
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_fraction_of_remaining,
    random_state=RANDOM_SEED, stratify=y_temp
)

print(f"\nData splits:")
print(f"  Train:      {len(X_train)} rows  (pos={y_train.sum()}, {100*y_train.mean():.1f}%)")
print(f"  Validation: {len(X_val)} rows  (pos={y_val.sum()}, {100*y_val.mean():.1f}%)")
print(f"  Test:       {len(X_test)} rows  (pos={y_test.sum()}, {100*y_test.mean():.1f}%)")

# ── 11. Fit scaler on TRAINING data only ─────────────────────────────────────
scaler = StandardScaler()

# Handle NaNs before scaling — impute with training-set medians
train_medians = np.nanmedian(X_train, axis=0)
for split_name, X_split in [("train", X_train), ("val", X_val), ("test", X_test)]:
    for j in range(X_split.shape[1]):
        mask = np.isnan(X_split[:, j])
        if mask.any():
            X_split[mask, j] = train_medians[j]
            print(f"  Imputed {mask.sum()} NaNs in {feature_cols[j]} ({split_name})")

X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
X_val_sc   = scaler.transform(X_val).astype(np.float32)
X_test_sc  = scaler.transform(X_test).astype(np.float32)

print(f"\nScaler fitted on training data only ✓")
print(f"  Means: {[f'{m:.3f}' for m in scaler.mean_]}")
print(f"  Stds:  {[f'{s:.3f}' for s in scaler.scale_]}")

# ── 12. Save all artifacts ────────────────────────────────────────────────────
np.save(MODELS_DIR / "X_train.npy", X_train_sc)
np.save(MODELS_DIR / "X_val.npy",   X_val_sc)
np.save(MODELS_DIR / "X_test.npy",  X_test_sc)
np.save(MODELS_DIR / "y_train.npy", y_train)
np.save(MODELS_DIR / "y_val.npy",   y_val)
np.save(MODELS_DIR / "y_test.npy",  y_test)

joblib.dump(scaler, MODELS_DIR / "nephromind_scaler.pkl")
print(f"  Saved: nephromind_scaler.pkl")

features_meta = {
    "feature_names": feature_cols,
    "n_features": len(feature_cols),
    "numeric_features": ["Age", "BMI", "Systolic_BP", "Diastolic_BP",
                         "Heart_Rate", "SpO2", "HbA1c"],
    "categorical_encoded": {"Gender_encoded": {"Male": 1, "Female": 0}},
    "scaler_type": "StandardScaler",
    "training_medians": {col: float(train_medians[i]) for i, col in enumerate(feature_cols)},
    "random_seed": RANDOM_SEED,
    "split": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
    "target": "Kidney_Disease",
    "positive_class": "Yes",
    "excluded_features": ["Creatinine_mg_dL","eGFR","UACR_mg_g","Diabetes",
                          "Diabetes_Pedigree","Diabetes_Duration_Years",
                          "Daily_Steps","Sleep_Duration_hr","Activity_Minutes",
                          "Resting_Heart_Rate","HRV_ms"],
    "exclusion_reason": {
        "Creatinine_mg_dL": "clinical kidney marker — target leakage risk",
        "eGFR": "clinical kidney marker — target leakage risk",
        "UACR_mg_g": "clinical kidney marker — target leakage risk",
        "Diabetes": "deterministic from HbA1c",
        "Diabetes_Pedigree": "synthetic/non-clinical",
        "Diabetes_Duration_Years": "unreliable synthetic field",
        "Daily_Steps": "wearable — not available from Renal Sentinel hardware",
        "Sleep_Duration_hr": "wearable — not available from Renal Sentinel hardware",
        "Activity_Minutes": "wearable — not available from Renal Sentinel hardware",
        "Resting_Heart_Rate": "wearable — not available from Renal Sentinel hardware",
        "HRV_ms": "wearable — not available from Renal Sentinel hardware",
    }
}
with open(MODELS_DIR / "nephromind_features.json", "w") as f:
    json.dump(features_meta, f, indent=2)
print(f"  Saved: nephromind_features.json")

print(f"\n✓ DATA PREPARATION COMPLETE")
print(f"  Feature vector ({len(feature_cols)} features): {feature_cols}")
