"""Validated ingestion for the immutable raw NephroMind CSV."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


PATIENT_ID_PATTERN = re.compile(r"^P[0-9]{5}$")


def load_patient_records(dataset_path: str | Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load valid patient records while preserving the raw audit-log rows on disk.

    The source CSV intentionally appends 18 non-patient audit-log rows. This function
    filters only records whose patient ID matches the configured patient-ID pattern.
    """
    raw = pd.read_csv(dataset_path, dtype=str)
    patient_mask = raw["Patient_ID"].str.fullmatch(config["valid_patient_id_pattern"], na=False)
    patients = raw.loc[patient_mask].copy()
    audit_rows = int((~patient_mask).sum())

    if len(patients) != 11_401:
        raise ValueError(f"Expected 11,401 valid patient records; found {len(patients)}.")
    if audit_rows != 18:
        raise ValueError(f"Expected 18 appended audit-log rows; found {audit_rows}.")
    if patients["Patient_ID"].duplicated().any():
        raise ValueError("Patient_ID must be unique after audit-log filtering.")

    numeric_features = config["numeric_features"]
    patients[numeric_features] = patients[numeric_features].apply(pd.to_numeric, errors="raise")
    patients["Gender"] = patients["Gender"].astype("string")
    patients[config["target_column"]] = patients[config["target_column"]].astype("string")

    target_values = set(patients[config["target_column"]].dropna().unique())
    if target_values != {"Yes", "No"}:
        raise ValueError(f"Unexpected target values: {target_values}")
    if patients[numeric_features + config["categorical_features"] + [config["target_column"]]].isna().any().any():
        raise ValueError("Unexpected missing values in valid source records before documented BP handling.")

    return patients, {"raw_rows": len(raw), "patient_rows": len(patients), "audit_log_rows": audit_rows}


def mark_documented_invalid_blood_pressure(
    patients: pd.DataFrame, invalid_patient_ids: list[str]
) -> tuple[pd.DataFrame, int]:
    """Set both BP values to missing only for the eight documented invalid records."""
    cleaned = patients.copy()
    flagged = cleaned["Patient_ID"].isin(invalid_patient_ids)
    if int(flagged.sum()) != len(invalid_patient_ids):
        raise ValueError("One or more documented invalid BP patient IDs were not found.")
    if not (cleaned.loc[flagged, "Systolic_BP"] == cleaned.loc[flagged, "Diastolic_BP"]).all():
        raise ValueError("Documented invalid BP records no longer have equal systolic and diastolic values.")
    cleaned.loc[flagged, ["Systolic_BP", "Diastolic_BP"]] = float("nan")
    return cleaned, int(flagged.sum())
