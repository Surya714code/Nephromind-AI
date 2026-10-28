# NephroMind AI — Explainability Report
Generated: 2026-09-08T23:07:57.921509

## Important Disclaimer
Feature contributions described in this report indicate association with the model
prediction, NOT causation. The phrase "contributed to the model prediction" does
NOT mean the feature caused kidney disease.

## Explainability Method
SHAP (SHapley Additive exPlanations) — TreeExplainer for tree-based models

## Final Model
Gradient Boosting (Tuned)

## Global Feature Importance (Top Features)
Features listed contributed most to the model's predictions across the training set.

1. `HbA1c`
2. `Age`
3. `Daily_Steps`
4. `Activity_Minutes`
5. `BMI`
6. `HRV_ms`
7. `Sleep_Duration_hr`
8. `Systolic_BP`
9. `Diastolic_BP`
10. `Heart_Rate`

## Plot Locations
- `ml/reports/figures/shap_global_importance.png` — Mean absolute SHAP values
- `ml/reports/figures/shap_summary_dot.png` — SHAP beeswarm plot
- `ml/reports/figures/shap_individual_patient_0.png` — Individual patient explanation
- `ml/reports/figures/feature_importance_model.png` — Model tree importance

## Prototype Risk Level Mapping
**PROTOTYPE ONLY — NOT clinically validated**

The following prototype mapping of model probability to risk level is provided
for dashboard display purposes only. It is NOT a clinically validated threshold.

| Probability Range | Prototype Risk Category |
|------------------|------------------------|
| 0.00 – 0.40 | Low (further assessment recommended) |
| 0.40 – 0.65 | Medium (medical consultation advised) |
| 0.65 – 1.00 | High (urgent nephrology referral suggested) |

This mapping is illustrative and must be reviewed by a qualified nephrologist
before any clinical deployment. It does not constitute a clinical decision.

## Limitations
- SHAP values explain model predictions, not clinical ground truth
- Correlation ≠ causation
- Results are specific to this dataset and model version
- Not validated on external clinical data
