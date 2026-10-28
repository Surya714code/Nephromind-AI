# NephroMind AI — Model Comparison
Generated: 2026-09-08T23:07:57.921509

## Validation Set Comparison (All Models)

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision | Specificity | Accuracy |
|-------|---------|--------|-----|--------|-----------|-------------|----------|
| Gradient Boosting (Tuned) ← **SELECTED** | 0.7655 | 0.9307 | 0.8953 | 0.9986 | 0.8114 | 0.0031 | 0.8105 |
| Gradient Boosting | 0.7578 | 0.9254 | 0.8937 | 0.9913 | 0.8136 | 0.0248 | 0.8088 |
| Random Forest | 0.7564 | 0.9238 | 0.8789 | 0.9337 | 0.8301 | 0.1796 | 0.7912 |
| XGBoost | 0.7538 | 0.9223 | 0.7918 | 0.6994 | 0.9125 | 0.7121 | 0.7018 |
| MLP | 0.7511 | 0.9153 | 0.8927 | 0.9870 | 0.8149 | 0.0372 | 0.8076 |
| Logistic Regression | 0.7457 | 0.9157 | 0.7801 | 0.6828 | 0.9097 | 0.7090 | 0.6877 |
| MLP (Tuned) | 0.7457 | 0.9151 | 0.8957 | 1.0000 | 0.8111 | 0.0000 | 0.8111 |
| Majority Baseline | 0.5000 | 0.8111 | 0.8957 | 1.0000 | 0.8111 | 0.0000 | 0.8111 |

> **Note**: All metrics computed at threshold = 0.50 unless noted.
> ROC-AUC and PR-AUC are threshold-independent.
> Model selection based on weighted combination of ROC-AUC (30%), PR-AUC (30%),
> F1 (20%), Recall (20%).

## Selection Rationale
- Primary criterion: ROC-AUC + PR-AUC (threshold-independent discrimination)
- Secondary criteria: F1, Recall (clinically relevant — missing true positives
  has higher cost than false alarms in disease screening context)
- Accuracy deliberately de-emphasised due to class imbalance (81%/19% split)
- Majority baseline included to validate that models genuinely learn beyond prevalence

## Majority Baseline Comparison
The majority baseline always predicts the positive class (kidney disease).
Any useful model must substantially exceed baseline PR-AUC = ~0.811 (class prevalence).
