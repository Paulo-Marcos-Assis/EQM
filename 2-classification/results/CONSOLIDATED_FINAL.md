# Final Consolidation — 16 Combinations (Vectorization × Classifier)
## Executive Summary
This document consolidates the results of **16 combinations** of vectorizations and classifiers for fraud detection in news (EQM), evaluated on the development set (1,643 examples, 148 positives, 9.01%). The isolated test set (2,046 examples) was evaluated only once in Task 7.
**Best model identified:** tfidf/base + linear_svc (dev F1: 0.9764)
---
## 1. Complete Results — All 16 Combinations
### 1.1 Overall Table Sorted by Dev F1
| Rank | Vectorization | Variant | Classifier | F1 | Precision | Recall | CV F1 | PR-AUC | ROC-AUC | Accuracy |
|------|---------------|---------|------------|-----|-----------|--------|-------|--------|---------|----------|
| 1 | tfidf | base | linear_svc | 0.9764 | 0.9732 | 0.9797 | 0.9622 | 0.9973 | 0.9997 | 0.9957 |
| 2 | tfidf | base | logistic_regression | 0.9737 | 0.9487 | 1.0000 | 0.9486 | 0.9960 | 0.9996 | 0.9951 |
| 3 | bertimbau_large | base | logistic_regression | 0.9699 | 0.9603 | 0.9797 | 0.9549 | 0.9899 | 0.9992 | 0.9945 |
| 4 | tfidf_stem | stem | logistic_regression | 0.9673 | 0.9367 | 1.0000 | 0.9487 | 0.9957 | 0.9996 | 0.9939 |
| 5 | bertimbau_large | base | linear_svc | 0.9630 | 0.9597 | 0.9662 | 0.9526 | 0.9863 | 0.9988 | 0.9933 |
| 6 | tfidf_stem | stem | linear_svc | 0.9592 | 0.9658 | 0.9527 | 0.9547 | 0.9969 | 0.9997 | 0.9927 |
| 7 | bertimbau_base | base | logistic_regression | 0.9579 | 0.9193 | 1.0000 | 0.9488 | 0.9931 | 0.9993 | 0.9921 |
| 8 | bertimbau_base | base | linear_svc | 0.9539 | 0.9295 | 0.9797 | 0.9556 | 0.9905 | 0.9991 | 0.9915 |
| 9 | tfidf | base | xgboost | 0.9492 | 0.9524 | 0.9459 | 0.9545 | 0.9920 | 0.9992 | 0.9909 |
| 10 | bertimbau_base | base | xgboost | 0.9481 | 0.9125 | 0.9865 | 0.9386 | 0.9910 | 0.9991 | 0.9903 |
| 11 | tfidf_stem | stem | xgboost | 0.9467 | 0.9342 | 0.9595 | 0.9606 | 0.9939 | 0.9994 | 0.9903 |
| 12 | bertimbau_large | base | xgboost | 0.9446 | 0.9119 | 0.9797 | 0.9392 | 0.9833 | 0.9986 | 0.9897 |
| 13 | bertimbau_large | base | random_forest | 0.9396 | 0.9333 | 0.9459 | 0.9276 | 0.9637 | 0.9973 | 0.9890 |
| 14 | tfidf_stem | stem | random_forest | 0.9389 | 0.8957 | 0.9865 | 0.9332 | 0.9807 | 0.9982 | 0.9884 |
| 15 | tfidf | base | random_forest | 0.9299 | 0.8795 | 0.9865 | 0.9285 | 0.9741 | 0.9978 | 0.9866 |
| 16 | bertimbau_base | base | random_forest | 0.9246 | 0.8981 | 0.9527 | 0.9190 | 0.9470 | 0.9969 | 0.9860 |

### 1.2 Top 5 Models
1. **tfidf/base + linear_svc: 0.9764**
   - Precision: 0.9732
   - Recall: 0.9797
   - CV F1: 0.9622
   - PR-AUC: 0.9973
   - Best params: {'C': 10}
2. **tfidf/base + logistic_regression: 0.9737**
   - Precision: 0.9487
   - Recall: 1.0000
   - CV F1: 0.9486
   - PR-AUC: 0.9960
   - Best params: {'C': 10}
3. **bertimbau_large/base + logistic_regression: 0.9699**
   - Precision: 0.9603
   - Recall: 0.9797
   - CV F1: 0.9549
   - PR-AUC: 0.9899
   - Best params: {'C': 10}
4. **tfidf_stem/stem + logistic_regression: 0.9673**
   - Precision: 0.9367
   - Recall: 1.0000
   - CV F1: 0.9487
   - PR-AUC: 0.9957
   - Best params: {'C': 10}
5. **bertimbau_large/base + linear_svc: 0.9630**
   - Precision: 0.9597
   - Recall: 0.9662
   - CV F1: 0.9526
   - PR-AUC: 0.9863
   - Best params: {'C': 10}
---
## 2. Analysis by Classifier
### 2.1 LinearSVC
**Best:** tfidf/base (F1=0.9764)
**Worst:** bertimbau_base/base (F1=0.9539)
**Mean F1:** 0.9631
| Vectorization | Variant | F1 | Precision | Recall | PR-AUC |
|---------------|---------|-----|-----------|--------|--------|
| tfidf | base | 0.9764 | 0.9732 | 0.9797 | 0.9973 |
| bertimbau_large | base | 0.9630 | 0.9597 | 0.9662 | 0.9863 |
| tfidf_stem | stem | 0.9592 | 0.9658 | 0.9527 | 0.9969 |
| bertimbau_base | base | 0.9539 | 0.9295 | 0.9797 | 0.9905 |

### 2.2 LogisticRegression
**Best:** tfidf/base (F1=0.9737)
**Worst:** bertimbau_base/base (F1=0.9579)
**Mean F1:** 0.9672
| Vectorization | Variant | F1 | Precision | Recall | PR-AUC |
|---------------|---------|-----|-----------|--------|--------|
| tfidf | base | 0.9737 | 0.9487 | 1.0000 | 0.9960 |
| bertimbau_large | base | 0.9699 | 0.9603 | 0.9797 | 0.9899 |
| tfidf_stem | stem | 0.9673 | 0.9367 | 1.0000 | 0.9957 |
| bertimbau_base | base | 0.9579 | 0.9193 | 1.0000 | 0.9931 |

### 2.3 RandomForest
**Best:** bertimbau_large/base (F1=0.9396)
**Worst:** bertimbau_base/base (F1=0.9246)
**Mean F1:** 0.9333
| Vectorization | Variant | F1 | Precision | Recall | PR-AUC |
|---------------|---------|-----|-----------|--------|--------|
| bertimbau_large | base | 0.9396 | 0.9333 | 0.9459 | 0.9637 |
| tfidf_stem | stem | 0.9389 | 0.8957 | 0.9865 | 0.9807 |
| tfidf | base | 0.9299 | 0.8795 | 0.9865 | 0.9741 |
| bertimbau_base | base | 0.9246 | 0.8981 | 0.9527 | 0.9470 |

### 2.4 XGBoost
**Best:** tfidf/base (F1=0.9492)
**Worst:** bertimbau_large/base (F1=0.9446)
**Mean F1:** 0.9471
| Vectorization | Variant | F1 | Precision | Recall | PR-AUC |
|---------------|---------|-----|-----------|--------|--------|
| tfidf | base | 0.9492 | 0.9524 | 0.9459 | 0.9920 |
| bertimbau_base | base | 0.9481 | 0.9125 | 0.9865 | 0.9910 |
| tfidf_stem | stem | 0.9467 | 0.9342 | 0.9595 | 0.9939 |
| bertimbau_large | base | 0.9446 | 0.9119 | 0.9797 | 0.9833 |

---
## 3. Analysis by Vectorization
### 3.1 bertimbau_base
**Best classifier:** logistic_regression (F1=0.9579, variant=base)
**Worst classifier:** random_forest (F1=0.9246, variant=base)
| Classifier | Variant | F1 | Precision | Recall | CV F1 |
|------------|---------|-----|-----------|--------|-------|
| logistic_regression | base | 0.9579 | 0.9193 | 1.0000 | 0.9488 |
| linear_svc | base | 0.9539 | 0.9295 | 0.9797 | 0.9556 |
| xgboost | base | 0.9481 | 0.9125 | 0.9865 | 0.9386 |
| random_forest | base | 0.9246 | 0.8981 | 0.9527 | 0.9190 |

### 3.2 bertimbau_large
**Best classifier:** logistic_regression (F1=0.9699, variant=base)
**Worst classifier:** random_forest (F1=0.9396, variant=base)
| Classifier | Variant | F1 | Precision | Recall | CV F1 |
|------------|---------|-----|-----------|--------|-------|
| logistic_regression | base | 0.9699 | 0.9603 | 0.9797 | 0.9549 |
| linear_svc | base | 0.9630 | 0.9597 | 0.9662 | 0.9526 |
| xgboost | base | 0.9446 | 0.9119 | 0.9797 | 0.9392 |
| random_forest | base | 0.9396 | 0.9333 | 0.9459 | 0.9276 |

### 3.3 tfidf
**Best classifier:** linear_svc (F1=0.9764, variant=base)
**Worst classifier:** random_forest (F1=0.9299, variant=base)
| Classifier | Variant | F1 | Precision | Recall | CV F1 |
|------------|---------|-----|-----------|--------|-------|
| linear_svc | base | 0.9764 | 0.9732 | 0.9797 | 0.9622 |
| logistic_regression | base | 0.9737 | 0.9487 | 1.0000 | 0.9486 |
| xgboost | base | 0.9492 | 0.9524 | 0.9459 | 0.9545 |
| random_forest | base | 0.9299 | 0.8795 | 0.9865 | 0.9285 |

### 3.4 tfidf_stem
**Best classifier:** logistic_regression (F1=0.9673, variant=stem)
**Worst classifier:** random_forest (F1=0.9389, variant=stem)
| Classifier | Variant | F1 | Precision | Recall | CV F1 |
|------------|---------|-----|-----------|--------|-------|
| logistic_regression | stem | 0.9673 | 0.9367 | 1.0000 | 0.9487 |
| linear_svc | stem | 0.9592 | 0.9658 | 0.9527 | 0.9547 |
| xgboost | stem | 0.9467 | 0.9342 | 0.9595 | 0.9606 |
| random_forest | stem | 0.9389 | 0.8957 | 0.9865 | 0.9332 |

---
## 4. Conclusions and Recommendations
### 4.1 Best Model
**tfidf/base + linear_svc** was carried to Task 7 (final evaluation on the isolated test set).
- **Dev F1:** 0.9764
- **Precision:** 0.9732
- **Recall:** 0.9797
- **CV F1 (estimated generalization):** 0.9622
- **PR-AUC:** 0.9973
- **ROC-AUC:** 0.9997
- **Best hyperparameters:** {'C': 10}

### 4.2 General Observations
1. **TF-IDF dominates:** the best combination is TF-IDF + LinearSVC (0.9764); TF-IDF + LogisticRegression (0.9737) is second
2. **Linear models win:** LinearSVC/LogisticRegression outperform ensemble methods (RF/XGBoost) on both sparse and dense representations
3. **Stemming does NOT help:** tfidf_stem best (LR 0.9673) loses to no-stem (SVC 0.9764 / LR 0.9737); stem helps RF (+0.009) only
4. **BERTimbau competitive but behind:** large + LR reaches 0.9699; embeddings add little over TF-IDF for this task
5. **All 16 combos ≥ 0.92 dev F1** — task is highly separable (leakage checks A–G PASS, Task 5b)
6. **Test generalization healthy:** dev→test gap −0.0201 F1 (Task 7)

### 4.3 Next Steps
- **Task 7 done:** best model evaluated on isolated test once (F1 0.9563)
- **Task 8 done:** domain bias LOW (semantic features dominate; official minority sources F1 1.0000)
- **Future work:** diversify portals further; threshold tuning on a fresh split if needed; monitor out-of-distribution performance

---
**Generated on:** 2026-08-19 16:21:15
