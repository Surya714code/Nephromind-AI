"""
NephroMind AI — Clinical Marker Service
=========================================
Rule-based interpretation of kidney-specific clinical markers.

This is SEPARATE from the DNN prediction.
These markers (Creatinine, eGFR, UACR) directly encode kidney function
status and are intentionally excluded from the DNN to prevent target leakage.

Sources:
- CKD stages: KDIGO 2012 CKD classification (eGFR categories)
- Albuminuria: KDIGO ACR categories
- Creatinine: general clinical reference ranges
"""
from __future__ import annotations
from typing import Optional


def interpret_clinical_markers(
    creatinine: Optional[float] = None,
    egfr: Optional[float] = None,
    uacr: Optional[float] = None,
    urine_dipstick: Optional[str] = None,
) -> dict:
    """
    Rule-based interpretation of kidney clinical markers.
    Returns a dict compatible with ClinicalMarkerAssessment schema.

    Note: eGFR alone does not diagnose CKD — it requires persistence
    over 3+ months and clinical context. This is a screening interpretation only.
    """
    result = {
        "marker_layer_note": (
            "Clinical marker assessment is rule-based and independent of the DNN AI prediction. "
            "These values directly reflect kidney function status. "
            "A physician must interpret all results in clinical context. "
            "eGFR reduction alone does not confirm CKD (requires 3+ months persistence)."
        )
    }

    # ── Creatinine ────────────────────────────────────────────────────────────
    if creatinine is not None:
        result["creatinine_mgdL"] = round(float(creatinine), 2)
        # General reference ranges (sex-specific not available here)
        if creatinine < 0.6:
            result["creatinine_status"] = "Below normal range"
        elif creatinine <= 1.2:
            result["creatinine_status"] = "Normal range"
        elif creatinine <= 2.0:
            result["creatinine_status"] = "Mildly elevated"
        elif creatinine <= 5.0:
            result["creatinine_status"] = "Moderately elevated"
        else:
            result["creatinine_status"] = "Severely elevated"

    # ── eGFR / CKD stage (KDIGO 2012) ────────────────────────────────────────
    if egfr is not None:
        result["egfr_ml_min"] = round(float(egfr), 1)
        if egfr >= 90:
            stage = "G1 (Normal or high, ≥90)"
        elif egfr >= 60:
            stage = "G2 (Mildly decreased, 60-89)"
        elif egfr >= 45:
            stage = "G3a (Mild-moderate decrease, 45-59)"
        elif egfr >= 30:
            stage = "G3b (Moderate-severe decrease, 30-44)"
        elif egfr >= 15:
            stage = "G4 (Severely decreased, 15-29)"
        else:
            stage = "G5 (Kidney failure, <15)"
        result["ckd_stage"] = stage

    # ── UACR / Albuminuria (KDIGO 2012) ──────────────────────────────────────
    if uacr is not None:
        result["uacr_mg_g"] = round(float(uacr), 1)
        if uacr < 30:
            cat = "A1 — Normal to mildly increased (<30 mg/g)"
        elif uacr < 300:
            cat = "A2 — Moderately increased (30-299 mg/g)"
        else:
            cat = "A3 — Severely increased (≥300 mg/g)"
        result["albuminuria_category"] = cat

    # ── Urine dipstick (qualitative screening) ────────────────────────────────
    if urine_dipstick is not None:
        result["urine_dipstick"] = str(urine_dipstick)
        result["dipstick_note"] = (
            "Urine dipstick is a qualitative screening result. "
            "It does NOT directly measure UACR. "
            "Positive dipstick findings require confirmation by laboratory UACR quantification."
        )

    # If no markers provided
    if len(result) <= 1:
        result["no_markers_provided"] = True
        result["note"] = "No clinical kidney markers were provided in this request."

    return result
