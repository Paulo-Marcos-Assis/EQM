#!/usr/bin/env python3
"""
Task 9 — Consolidation and Documentation
- training/results/CONSOLIDATED_FINAL.md: ranking of 16 combos, analysis by
  vectorization and by algorithm, best model + hyperparameters.
- Section obs2 (IR topics) coverage map in Markdown.
- FLOW_DIAGRAM.md update (EQM, English).
"""

import os
import json
import glob
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "training/results")
OUT = os.path.join(RESULTS_DIR, "CONSOLIDATED_FINAL.md")

ALGO_NAMES = {
    'linear_svc': 'LinearSVC',
    'logistic_regression': 'LogisticRegression',
    'random_forest': 'RandomForest',
    'xgboost': 'XGBoost',
}


def load_all_results():
    rows = []
    for j in glob.glob(os.path.join(RESULTS_DIR, "*", "*_results.json")):
        with open(j) as f:
            rows.extend(json.load(f))
    return rows


def fmt(rows):
    def key(r):
        return r.get('f1', 0)
    return sorted(rows, key=key, reverse=True)


def main():
    rows = load_all_results()
    rows = fmt(rows)
    assert len(rows) == 16, f"Expected 16 combos, got {len(rows)}"

    best = rows[0]
    best_label = f"{best['vectorization']}/{best['variant']} + {best['classifier']}"

    lines = []
    lines.append("# Final Consolidation — 16 Combinations (Vectorization × Classifier)\n")
    lines.append("## Executive Summary\n")
    lines.append("This document consolidates the results of **16 combinations** of vectorizations "
                 "and classifiers for fraud detection in news (EQM), evaluated on the development "
                 "set (1,643 examples, 148 positives, 9.01%). The isolated test set (2,046 examples) "
                 "was evaluated only once in Task 7.\n")
    lines.append(f"**Best model identified:** {best_label} "
                 f"(dev F1: {best['f1']:.4f})\n")
    lines.append("---\n")
    lines.append("## 1. Complete Results — All 16 Combinations\n")
    lines.append("### 1.1 Overall Table Sorted by Dev F1\n")
    lines.append("| Rank | Vectorization | Variant | Classifier | F1 | Precision | Recall | CV F1 | PR-AUC | ROC-AUC | Accuracy |\n")
    lines.append("|------|---------------|---------|------------|-----|-----------|--------|-------|--------|---------|----------|\n")
    for i, r in enumerate(rows, 1):
        label = r['classifier']
        lines.append(
            f"| {i} | {r['vectorization']} | {r['variant']} | {label} | "
            f"{r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['cv_f1']:.4f} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | "
            f"{r['accuracy']:.4f} |\n")
    lines.append("\n### 1.2 Top 5 Models\n")
    for i, r in enumerate(rows[:5], 1):
        label = f"{r['vectorization']}/{r['variant']} + {r['classifier']}"
        lines.append(f"{i}. **{label}: {r['f1']:.4f}**\n")
        lines.append(f"   - Precision: {r['precision']:.4f}\n")
        lines.append(f"   - Recall: {r['recall']:.4f}\n")
        lines.append(f"   - CV F1: {r['cv_f1']:.4f}\n")
        lines.append(f"   - PR-AUC: {r['pr_auc']:.4f}\n")
        lines.append(f"   - Best params: {r['best_params']}\n")
    lines.append("---\n")

    lines.append("## 2. Analysis by Classifier\n")
    for slug, name in ALGO_NAMES.items():
        sub = [r for r in rows if r['classifier'] == slug]
        if not sub:
            continue
        best_c = max(sub, key=lambda r: r['f1'])
        worst_c = min(sub, key=lambda r: r['f1'])
        mean_f1 = sum(r['f1'] for r in sub) / len(sub)
        lines.append(f"### 2.{list(ALGO_NAMES).index(slug)+1} {name}\n")
        lines.append(f"**Best:** {best_c['vectorization']}/{best_c['variant']} "
                     f"(F1={best_c['f1']:.4f})\n")
        lines.append(f"**Worst:** {worst_c['vectorization']}/{worst_c['variant']} "
                     f"(F1={worst_c['f1']:.4f})\n")
        lines.append(f"**Mean F1:** {mean_f1:.4f}\n")
        lines.append("| Vectorization | Variant | F1 | Precision | Recall | PR-AUC |\n")
        lines.append("|---------------|---------|-----|-----------|--------|--------|\n")
        for r in sorted(sub, key=lambda x: x['f1'], reverse=True):
            lines.append(f"| {r['vectorization']} | {r['variant']} | {r['f1']:.4f} | "
                         f"{r['precision']:.4f} | {r['recall']:.4f} | {r['pr_auc']:.4f} |\n")
        lines.append("\n")
    lines.append("---\n")

    lines.append("## 3. Analysis by Vectorization\n")
    vecs = sorted({r['vectorization'] for r in rows})
    for vi, v in enumerate(vecs, 1):
        sub = [r for r in rows if r['vectorization'] == v]
        best_v = max(sub, key=lambda r: r['f1'])
        worst_v = min(sub, key=lambda r: r['f1'])
        lines.append(f"### 3.{vi} {v}\n")
        lines.append(f"**Best classifier:** {best_v['classifier']} "
                     f"(F1={best_v['f1']:.4f}, variant={best_v['variant']})\n")
        lines.append(f"**Worst classifier:** {worst_v['classifier']} "
                     f"(F1={worst_v['f1']:.4f}, variant={worst_v['variant']})\n")
        lines.append("| Classifier | Variant | F1 | Precision | Recall | CV F1 |\n")
        lines.append("|------------|---------|-----|-----------|--------|-------|\n")
        for r in sorted(sub, key=lambda x: x['f1'], reverse=True):
            lines.append(f"| {r['classifier']} | {r['variant']} | {r['f1']:.4f} | "
                         f"{r['precision']:.4f} | {r['recall']:.4f} | {r['cv_f1']:.4f} |\n")
        lines.append("\n")
    lines.append("---\n")

    lines.append("## 4. Conclusions and Recommendations\n")
    lines.append("### 4.1 Best Model\n")
    lines.append(f"**{best_label}** was carried to Task 7 (final evaluation on the isolated test set).\n")
    lines.append(f"- **Dev F1:** {best['f1']:.4f}\n")
    lines.append(f"- **Precision:** {best['precision']:.4f}\n")
    lines.append(f"- **Recall:** {best['recall']:.4f}\n")
    lines.append(f"- **CV F1 (estimated generalization):** {best['cv_f1']:.4f}\n")
    lines.append(f"- **PR-AUC:** {best['pr_auc']:.4f}\n")
    lines.append(f"- **ROC-AUC:** {best['roc_auc']:.4f}\n")
    lines.append(f"- **Best hyperparameters:** {best['best_params']}\n")
    lines.append("\n### 4.2 General Observations\n")
    lines.append("1. **TF-IDF dominates:** the best combination is TF-IDF + LinearSVC (0.9764); "
                 "TF-IDF + LogisticRegression (0.9737) is second\n")
    lines.append("2. **Linear models win:** LinearSVC/LogisticRegression outperform ensemble "
                 "methods (RF/XGBoost) on both sparse and dense representations\n")
    lines.append("3. **Stemming does NOT help:** tfidf_stem best (LR 0.9673) loses to no-stem "
                 "(SVC 0.9764 / LR 0.9737); stem helps RF (+0.009) only\n")
    lines.append("4. **BERTimbau competitive but behind:** large + LR reaches 0.9699; embeddings "
                 "add little over TF-IDF for this task\n")
    lines.append("5. **All 16 combos ≥ 0.92 dev F1** — task is highly separable (leakage checks "
                 "A–G PASS, Task 5b)\n")
    lines.append("6. **Test generalization healthy:** dev→test gap −0.0201 F1 (Task 7)\n")
    lines.append("\n### 4.3 Next Steps\n")
    lines.append("- **Task 7 done:** best model evaluated on isolated test once (F1 0.9563)\n")
    lines.append("- **Task 8 done:** domain bias LOW (semantic features dominate; official "
                 "minority sources F1 1.0000)\n")
    lines.append("- **Future work:** diversify portals further; threshold tuning on a fresh split "
                 "if needed; monitor out-of-distribution performance\n")
    lines.append("\n---\n")
    lines.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    with open(OUT, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"Saved: {OUT}")
    print(f"Combos: {len(rows)}")


if __name__ == "__main__":
    main()