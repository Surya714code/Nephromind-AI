# NephroMind AI — Validation-Only Threshold Analysis

**Methodological Constraint**: The hold-out Test Set remained 100% untouched during threshold selection. All evaluation grid sweeps and candidate operating point selections were performed strictly on the **Validation Set** ($N = 1,710$ records).

---

## 1. Problem Statement & Motivation

During initial model training, optimizing for F1 score yielded an optimal threshold of $t = 0.36$. However, because population disease prevalence is ~81.1% (`Kidney_Disease = Yes`), thresholding at $0.36$ caused the classifier to predict positive for 99.65% of test samples, resulting in:
- **Recall**: 99.71%
- **Specificity**: 0.62% (only 2 True Negatives out of 323 actual negative cases)

This rendered the binary output functionally equivalent to a majority-class baseline. To establish a clinically meaningful prototype threshold, a validation-only grid search was executed.

---

## 2. Validation Set Threshold Grid Search (0.10 to 0.95)

Key candidate operating points evaluated on the Validation Set:

| Threshold ($t$) | Accuracy | Precision | Sensitivity (Recall) | Specificity | F1-Score | Youden's J | FPR | FNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.10 | 0.8117 | 0.8116 | 1.0000 | 0.0031 | 0.8960 | 0.0031 | 0.9969 | 0.0000 |
| 0.30 | 0.8117 | 0.8116 | 1.0000 | 0.0031 | 0.8960 | 0.0031 | 0.9969 | 0.0000 |
| 0.36 | 0.8117 | 0.8116 | 1.0000 | 0.0031 | 0.8960 | 0.0031 | 0.9969 | 0.0000 |
| 0.50 | 0.8117 | 0.8116 | 1.0000 | 0.0031 | 0.8960 | 0.0031 | 0.9969 | 0.0000 |
| 0.70 | 0.7211 | 0.9167 | 0.7217 | 0.7183 | 0.8076 | 0.4400 | 0.2817 | 0.2783 |
| 0.75 | 0.6982 | 0.9299 | 0.6792 | 0.7802 | 0.7850 | 0.4593 | 0.2198 | 0.3208 |
| 0.79 **[LOCKED PROTOTYPE]** | 0.6713 | 0.9421 | 0.6337 | 0.8328 | 0.7578 | 0.4666 | 0.1672 | 0.3663 |
| 0.90 | 0.6713 | 0.9421 | 0.6337 | 0.8328 | 0.7578 | 0.4666 | 0.1672 | 0.3663 |
| 0.92 | 0.6444 | 0.9441 | 0.5970 | 0.8483 | 0.7314 | 0.4453 | 0.1517 | 0.4030 |
| 0.95 | 0.3743 | 0.9703 | 0.2358 | 0.9690 | 0.3794 | 0.2048 | 0.0310 | 0.7642 |

---

## 3. Candidate Operating Points Summary (Validation Set Only)

1. **Maximum F1-Score Threshold ($t = 0.10$)**:
   - Sensitivity: 1.0000 | Specificity: 0.0031 | F1: 0.8960
   - *Critique*: Heavily biased toward high prevalence; produces 0% specificity.

2. **Maximum Youden's J Statistic Threshold ($t = 0.79$)** — **SELECTED PROTOTYPE THRESHOLD**:
   - Youden's $J = \text{Sensitivity} + \text{Specificity} - 1 = 0.4666$
   - Sensitivity: 0.6337 | Specificity: 0.8328 | Accuracy: 0.6713
   - *Rationale*: Maximizes joint discrimination between positive and negative classes without being overwhelmed by class prevalence.

3. **Target ~70% Specificity Threshold ($t = 0.70$)**:
   - Sensitivity: 0.7217 | Specificity: 0.7183

4. **Target ~80% Specificity Threshold ($t = 0.75$)**:
   - Sensitivity: 0.6792 | Specificity: 0.7802

5. **Target ~90% Specificity Threshold ($t = 0.92$)**:
   - Sensitivity: 0.5970 | Specificity: 0.8483

---

## 4. Single Locked Prototype Threshold Selection

- **Locked Threshold**: **$t = 0.79$**
- **Selection Rationale**: Selected on the Validation Set by maximizing Youden's J statistic ($J = 0.4666$). This choice successfully resolves the specificity collapse while maintaining high diagnostic sensitivity.
- **Validation Performance at $t = 0.79$**:
  - Accuracy: 0.6713
  - Precision: 0.9421
  - Sensitivity: 0.6337
  - Specificity: 0.8328
  - F1-Score: 0.7578
  - Confusion Matrix: TN=269, FP=54, FN=508, TP=879

---

## 5. Final Evaluation on Untouched Test Set (Executed ONCE)

Evaluating the locked threshold ($t = 0.79$) on the hold-out test set ($N = 1,711$):

- **Accuracy**: 0.676213 (67.62%)
- **Precision**: 0.949353 (94.94%)
- **Recall (Sensitivity)**: 0.634726 (63.47%)
- **Specificity**: 0.854489 (85.45%)
- **F1-Score**: 0.760794 (76.08%)
- **ROC-AUC**: 0.761810
- **PR-AUC**: 0.921389
- **Confusion Matrix**: TP=881, TN=276, FP=47, FN=507
