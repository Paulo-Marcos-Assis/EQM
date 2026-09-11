#!/usr/bin/env python3
"""Task 7 — Final Test Evaluation.
Load best model (Task 5) + vectorization; evaluate on isolated test (used only once).
Leakage: link overlap + near-text duplicates (train/test, dev/test).
Reporting: dev->test gap.
"""

import os
import sys
import csv
import json
import pickle
import numpy as np
from scipy import sparse

from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score, accuracy_score,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEC_DIR = os.path.join(BASE_DIR, "vectorization")
RESULTS_DIR = os.path.join(BASE_DIR, "training/results")
TRAIN_DIR = os.path.join(BASE_DIR, "FOR_TRAINING")
TEST_DIR = os.path.join(BASE_DIR, "FOR_TEST")
OUT_DIR = os.path.join(RESULTS_DIR, "_logs", "task7_final_test")

VEC_KEY = "tfidf"
VARIANT = "base"
ALGO_SLUG = "linear_svc"


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def load_raw_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def leakage_check():
    print("\n" + "=" * 72)
    print("LEAKAGE CHECK (train/test and dev/test)")
    print("=" * 72)

    train_rows = load_raw_csv(os.path.join(TRAIN_DIR, "train.csv"))
    dev_rows = load_raw_csv(os.path.join(TRAIN_DIR, "dev.csv"))
    test_rows = load_raw_csv(os.path.join(TEST_DIR, "test.csv"))

    train_links = set(r["link_noticia"].strip() for r in train_rows)
    dev_links = set(r["link_noticia"].strip() for r in dev_rows)
    test_links = set(r["link_noticia"].strip() for r in test_rows)
    train_test_overlap = len(train_links & test_links)
    dev_test_overlap = len(dev_links & test_links)
    print(f"Link overlap train∩test: {train_test_overlap}")
    print(f"Link overlap dev∩test:   {dev_test_overlap}")

    train_texts = set(r["texto_completo"].strip() for r in train_rows)
    dev_texts = set(r["texto_completo"].strip() for r in dev_rows)
    test_texts = set(r["texto_completo"].strip() for r in test_rows)
    print(f"Exact text overlap train∩test: {len(train_texts & test_texts)}")
    print(f"Exact text overlap dev∩test:   {len(dev_texts & test_texts)}")

    vec_path = os.path.join(VEC_DIR, VEC_KEY)
    X_test = sparse.load_npz(os.path.join(vec_path, "test_sparse.npz"))
    X_train = sparse.load_npz(os.path.join(vec_path, "train_sparse.npz"))
    X_dev = sparse.load_npz(os.path.join(vec_path, "dev_sparse.npz"))
    y_test = np.load(os.path.join(vec_path, "labels_test.npy"))
    y_train = np.load(os.path.join(vec_path, "labels_train.npy"))
    y_dev = np.load(os.path.join(vec_path, "labels_dev.npy"))

    def top_cos(q, pool):
        qn = q.multiply(1.0 / np.sqrt(q.multiply(q).sum(axis=1)))
        pn = pool.multiply(1.0 / np.sqrt(pool.multiply(pool).sum(axis=1)))
        return qn.dot(pn.T).toarray()

    for name, pool, y_pool in [("train", X_train, y_train), ("dev", X_dev, y_dev)]:
        sims = top_cos(X_test, pool)
        best = sims.max(axis=1)
        for thr in [0.999, 0.99, 0.95]:
            idx = np.where(best >= thr)[0]
            n = len(idx)
            same = sum(1 for i in idx if y_test[i] == y_pool[sims[i].argmax()])
            print(f"Near-dup test→{name} (cos≥{thr}): {n} rows "
                  f"({same} same-label / {n - same} diff-label)")

    return train_test_overlap, dev_test_overlap


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, "task7_execution_log.txt")
    log_file = open(log_path, 'w')
    tee = Tee(sys.stdout, log_file)
    sys.stdout = tee

    print("=" * 72)
    print("TASK 7 — Final Test Evaluation (best model on isolated test)")
    print("=" * 72)
    print(f"Best model: {VEC_KEY}/{VARIANT} + {ALGO_SLUG} (from Task 5)")

    with open(os.path.join(RESULTS_DIR, VEC_KEY, ALGO_SLUG, VARIANT, "model.pkl"), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(VEC_DIR, VEC_KEY, "vectorizer.pkl"), 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"Loaded model + vectorizer ({VEC_KEY})")

    X_test = sparse.load_npz(os.path.join(VEC_DIR, VEC_KEY, "test_sparse.npz"))
    y_test = np.load(os.path.join(VEC_DIR, VEC_KEY, "labels_test.npy"))
    print(f"Test matrix: {X_test.shape} | positives: {int((y_test == 1).sum())} / {len(y_test)}")

    y_pred = model.predict(X_test)
    y_scores = model.decision_function(X_test)

    metrics = {
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_test, y_pred),
        'pr_auc': average_precision_score(y_test, y_scores),
        'roc_auc': roc_auc_score(y_test, y_scores),
    }
    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    with open(os.path.join(OUT_DIR, "test_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(y_test, y_pred, zero_division=0)
    with open(os.path.join(OUT_DIR, "classification_report.txt"), 'w') as f:
        f.write(f"Task 7 — Final Test Evaluation\n")
        f.write(f"Best model: {VEC_KEY}/{VARIANT} + {ALGO_SLUG}\n\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write(f"\n{report}\n")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"Test — {VEC_KEY}/{VARIANT} + {ALGO_SLUG}")
    fig.colorbar(im)
    classes = ['Negative', 'Positive']
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes)
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {OUT_DIR}/confusion_matrix.png")

    tt_overlap, dt_overlap = leakage_check()

    print("\n" + "=" * 72)
    print("DEV → TEST GAP (generalization)")
    print("=" * 72)
    dev_metrics_path = os.path.join(RESULTS_DIR, VEC_KEY, ALGO_SLUG, VARIANT,
                                    "classification_report.txt")
    dev_metrics = {}
    with open(dev_metrics_path) as f:
        for line in f:
            line = line.strip()
            for k in ['f1', 'precision', 'recall', 'accuracy', 'pr_auc', 'roc_auc']:
                if line.startswith(f"{k}:"):
                    dev_metrics[k] = float(line.split(":")[1].strip())

    print(f"{'Metric':<12} {'Dev':>8} {'Test':>8} {'Delta':>8}")
    for k in ['f1', 'precision', 'recall', 'accuracy', 'pr_auc', 'roc_auc']:
        dv = dev_metrics.get(k, 0)
        tv = metrics[k]
        print(f"{k:<12} {dv:.4f} {tv:.4f} {tv - dv:+.4f}")

    summary = {
        'best_model': f"{VEC_KEY}/{VARIANT} + {ALGO_SLUG}",
        'test_metrics': metrics,
        'dev_metrics': dev_metrics,
        'link_overlap_train_test': tt_overlap,
        'link_overlap_dev_test': dt_overlap,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("TASK 7 COMPLETE")
    print("=" * 72)

    sys.stdout = sys.__stdout__
    log_file.close()
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()