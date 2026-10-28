# NephroMind AI — Model Training Report
Generated: 2026-09-08T23:07:57.921509

## Objective
Current kidney-disease status classification (binary: Yes / No).
This is a retrospective, cross-sectional classification task.

## Selected Final Model
**Gradient Boosting (Tuned)**

## Preprocessing Pipeline
- Numeric features: Median imputation → StandardScaler
- Categorical features: Mode imputation → OneHotEncoder
- Pipeline fitted on training data only; applied to validation and test sets

## Class Imbalance Handling
- Class distribution: Yes (81.1%) / No (18.9%) — imbalanced dataset
- All applicable tree/linear models trained with `class_weight="balanced"`
- XGBoost trained with `scale_pos_weight` proportional to class imbalance
- No SMOTE applied (class weights preferred for this dataset size)

## Hyperparameter Tuning
- Strategy: RandomizedSearchCV on training set only (5-fold stratified CV)
- Scoring: ROC-AUC
- n_iter: 30
- Validation set untouched during tuning

## Threshold Optimisation
- Method: F1-maximising threshold search on validation set
- Optimal threshold: **0.36**

## Test Set Evaluation (First and Only Use of Test Set)

### Default Threshold (0.50)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8095 |
| Precision | 0.8116 |
| Recall (Sensitivity) | 0.9964 |
| F1-Score | 0.8946 |
| ROC-AUC | 0.7618 |
| PR-AUC | 0.9214 |
| Specificity | 0.0062 |
| Brier Score | 0.1320 |

Confusion Matrix (t=0.50):
- TN=2, FP=321, FN=5, TP=1383

### Optimal Threshold (0.36)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8101 |
| Precision | 0.8117 |
| Recall (Sensitivity) | 0.9971 |
| F1-Score | 0.8949 |
| ROC-AUC | 0.7618 |
| PR-AUC | 0.9214 |
| Specificity | 0.0062 |
| Brier Score | 0.1320 |

Confusion Matrix (t=0.36):
- TN=2, FP=321, FN=4, TP=1384

## Validation Comparison (All Models, ROC-AUC Ranked)
| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision | Specificity |
|-------|---------|--------|-----|--------|-----------|-------------|
| Gradient Boosting (Tuned) | 0.7655 | 0.9307 | 0.8953 | 0.9986 | 0.8114 | 0.0031 |
| Gradient Boosting | 0.7578 | 0.9254 | 0.8937 | 0.9913 | 0.8136 | 0.0248 |
| Random Forest | 0.7564 | 0.9238 | 0.8789 | 0.9337 | 0.8301 | 0.1796 |
| XGBoost | 0.7538 | 0.9223 | 0.7918 | 0.6994 | 0.9125 | 0.7121 |
| MLP | 0.7511 | 0.9153 | 0.8927 | 0.9870 | 0.8149 | 0.0372 |
| Logistic Regression | 0.7457 | 0.9157 | 0.7801 | 0.6828 | 0.9097 | 0.7090 |
| MLP (Tuned) | 0.7457 | 0.9151 | 0.8957 | 1.0000 | 0.8111 | 0.0000 |
| Majority Baseline | 0.5000 | 0.8111 | 0.8957 | 1.0000 | 0.8111 | 0.0000 |

## Artifact
- Location: `E:\DEEP LEARN NEPHRO\ml\artifacts\nephromind_model_v1.joblib`
- Format: joblib (compressed, includes preprocessing + model + metadata)
- Model version: v1.0.0

## Limitations
1. Cross-sectional data only — no progression prediction possible
2. Synthetic/registry dataset — not clinically validated
3. Target leakage features (eGFR, UACR) excluded — model uses indirect biomarkers
4. NOT a replacement for clinical diagnosis or nephrology consultation
5. Results should not be interpreted causally
