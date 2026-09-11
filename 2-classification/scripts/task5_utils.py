#!/usr/bin/env python3
"""
Shared functions for Task 5 — consolidated report generation.
Mirrors NDMAIS_BIAS/scripts/task5_utils.py, adapted to EQM (English, EQM paths/data).
"""

import os
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save_comparison_csv(all_results, out_path):
    cols = ['vectorization', 'variant', 'cv_f1', 'f1', 'precision', 'recall',
            'pr_auc', 'roc_auc', 'accuracy']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()
        for r in all_results:
            row = {k: r.get(k, '') for k in cols}
            for k in cols:
                if isinstance(row[k], float):
                    row[k] = f"{row[k]:.4f}"
            writer.writerow(row)


def save_comparison_png(all_results, out_path, classifier_name):
    labels = [f"{r['vectorization']}{('/'+r['variant']) if r.get('variant') else ''}"
              for r in all_results]
    f1_vals = [r.get('f1', 0) for r in all_results]
    prec_vals = [r.get('precision', 0) for r in all_results]
    rec_vals = [r.get('recall', 0) for r in all_results]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, f1_vals, width, label='F1', color='#2196F3')
    ax.bar(x, prec_vals, width, label='Precision', color='#4CAF50')
    ax.bar(x + width, rec_vals, width, label='Recall', color='#FF9800')

    ax.set_ylabel('Score')
    ax.set_title(f'{classifier_name} — Comparison across Vectorizations (Dev Set)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)

    for i, v in enumerate(f1_vals):
        ax.text(i - width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def save_explanation_md(all_results, classifier_name, classifier_desc,
                        config_str, out_path):
    best = max(all_results, key=lambda r: r.get('f1', 0))

    lines = []
    lines.append(f"# Experiment Report: {classifier_name}\n")
    lines.append("## 1. Objective\n")
    lines.append(
        "Evaluate the impact of different vectorization techniques on the performance of the "
        f"{classifier_name} classifier for fraud detection in news "
        "(EQM). Imbalanced dataset (8.88% positives).\n")
    lines.append("## 2. Methodology\n")
    lines.append("### 2.1 Data Split\n")
    lines.append("- **Training:** 6,642 examples (590 positive, 6,052 negative)\n")
    lines.append("- **Development:** 1,643 examples (148 positive, 1,495 negative)\n")
    lines.append("- **Test:** 2,046 examples (isolated, not used in this phase)\n")
    lines.append("### 2.2 Classifier\n")
    lines.append(f"{classifier_desc}\n")
    lines.append("**Configuration:**\n")
    lines.append(f"```python\n{config_str}\n```\n")
    lines.append("### 2.3 Protocol\n")
    lines.append("1. GridSearchCV with stratified 5-fold on training (F1 as scoring)\n")
    lines.append("2. Retrain with best hyperparameters on full training set\n")
    lines.append("3. Evaluate on dev set\n")
    lines.append("4. Test remains isolated (Task 7)\n")
    lines.append("## 3. Results\n")
    lines.append("### 3.1 Comparative Table (Dev Set)\n")
    lines.append("| Vectorization | Variant | CV F1 | F1 | Precision | Recall | PR-AUC | ROC-AUC | Accuracy |\n")
    lines.append("|---------------|---------|-------|----|-----------|--------|--------|---------|----------|\n")
    for r in all_results:
        variant = r.get('variant', '')
        lines.append(
            f"| {r['vectorization']} | {variant} | {r.get('cv_f1', 0):.4f} | "
            f"{r.get('f1', 0):.4f} | {r.get('precision', 0):.4f} | "
            f"{r.get('recall', 0):.4f} | {r.get('pr_auc', 0):.4f} | "
            f"{r.get('roc_auc', 0):.4f} | {r.get('accuracy', 0):.4f} |\n"
        )
    best_label = best['vectorization']
    if best.get('variant'):
        best_label += f"/{best['variant']}"
    lines.append(f"\n**Best Vectorization: {best_label} (F1: {best.get('f1', 0):.4f})**\n")
    lines.append("### 3.2 Best Hyperparameters per Vectorization\n")
    lines.append("| Vectorization | Variant | Best Params |\n")
    lines.append("|---------------|---------|-------------|\n")
    for r in all_results:
        params_str = json.dumps(r.get('best_params', {}))
        lines.append(f"| {r['vectorization']} | {r.get('variant', '')} | {params_str} |\n")
    lines.append("\n## 4. Analysis\n")
    lines.append(f"- **Best combination:** {best_label} + {classifier_name}\n")
    lines.append(f"- **Dev F1:** {best.get('f1', 0):.4f}\n")
    lines.append(f"- **Precision:** {best.get('precision', 0):.4f}\n")
    lines.append(f"- **Recall:** {best.get('recall', 0):.4f}\n")
    lines.append(f"- **CV F1 (generalization estimate):** {best.get('cv_f1', 0):.4f}\n")
    lines.append("\n## 5. Algorithm Justification\n")
    lines.append(
        "- **LinearSVC:** linear kernel, efficient in high dimensionality (TF-IDF); "
        "robust with balanced class_weight.\n"
        "- **LogisticRegression:** probabilistic baseline, interpretable; good calibration.\n"
        "- **RandomForest:** captures non-linearities and interactions; relevant for BERTimbau "
        "(dense representations).\n"
        "- **XGBoost:** gradient boosting with tree_method='hist' — fast/stable on sparse "
        "data; dynamic scale_pos_weight for imbalance.\n")

    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def generate_consolidated_reports(all_results, results_dir, classifier_name,
                                  classifier_desc, config_str,
                                  consolidated_json_name):
    classifier_slug = classifier_name.lower().replace(' ', '_')
    classifier_dir = os.path.join(results_dir, classifier_slug)
    os.makedirs(classifier_dir, exist_ok=True)

    consolidated_path = os.path.join(classifier_dir, consolidated_json_name)
    with open(consolidated_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    save_comparison_csv(
        all_results,
        os.path.join(classifier_dir, f"{classifier_slug}_comparison.csv")
    )
    save_comparison_png(
        all_results,
        os.path.join(classifier_dir, f"{classifier_slug}_comparison.png"),
        classifier_name
    )
    save_explanation_md(
        all_results, classifier_name, classifier_desc, config_str,
        os.path.join(classifier_dir, "EXPLANATION.md")
    )