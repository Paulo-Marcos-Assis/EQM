#!/usr/bin/env python3
"""
Task 7b — Test-set cluster analysis (added post-Task 7, per user request)

Answers: "can the clusters also be generated regarding the labels considering
the test set, now that the results exist?"

Three analyses (all 3D when possible), outputs in:
    training/results/visualizations/test_set_clusters/

  1. k-means cluster purity on the TEST set (BERTimbau Large embeddings):
       clusters vs TRUE labels and vs PREDICTED labels (Task 7 best model);
       purity / homogeneity / completeness / ARI per k (2, 3, 5).
  2. t-SNE colored by PREDICTED label (all 10,331 rows, same shared space as
       Task 5b) -> where do predictions sit relative to the clusters?
       Plus a misclassification overlay (correct vs error) on the test subset.
  3. t-SNE RESTRICTED to the test set only (2,046 embeddings), fresh projection:
       colored by TRUE label and by PREDICTED label (held-out separability).
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import (
    f1_score, confusion_matrix, classification_report,
    homogeneity_score, completeness_score, v_measure_score,
    adjusted_rand_score, adjusted_mutual_info_score,
)
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEC = os.path.join(BASE, "vectorization")
RESULTS = os.path.join(BASE, "training/results")
VIS_DIR = os.path.join(RESULTS, "visualizations")
OUT_DIR = os.path.join(VIS_DIR, "test_set_clusters")
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "test_set_clusters_report.txt")

VEC_KEY = "tfidf"
ALGO_SLUG = "linear_svc"
VARIANT = "base"

log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


def section(title):
    log("\n" + "=" * 72)
    log(f"### {title}")
    log("=" * 72)


def load_predictions():
    """Best model (tfidf + LinearSVC) predictions on all splits."""
    with open(os.path.join(RESULTS, VEC_KEY, ALGO_SLUG, VARIANT, "model.pkl"), 'rb') as f:
        model = pickle.load(f)
    preds = {}
    labels = {}
    for split in ["train", "dev", "test"]:
        X = sparse.load_npz(os.path.join(VEC, VEC_KEY, f"{split}_sparse.npz"))
        y = np.load(os.path.join(VEC, VEC_KEY, f"labels_{split}.npy"))
        preds[split] = model.predict(X)
        labels[split] = y
    return model, preds, labels


def cluster_purity(cluster_labels, true_labels):
    """Purity: fraction of rows in each cluster belonging to its majority true class."""
    n = len(true_labels)
    clusters = np.unique(cluster_labels)
    correct = 0
    for c in clusters:
        m = cluster_labels == c
        if m.sum() == 0:
            continue
        labels_c = true_labels[m]
        correct += np.bincount(labels_c).max()
    return correct / n


def kmeans_analysis(Xte, yte, y_pred_test):
    section("1. k-means cluster purity on TEST set (BERTimbau Large embeddings)")
    log(f"  test embeddings: {Xte.shape} | positives: {int(yte.sum())} / {len(yte)}")

    rows = []
    for k in [2, 3, 5]:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        cl = km.fit_predict(Xte)
        pur_true = cluster_purity(cl, yte)
        pur_pred = cluster_purity(cl, y_pred_test)
        homo = homogeneity_score(yte, cl)
        comp = completeness_score(yte, cl)
        v = v_measure_score(yte, cl)
        ari = adjusted_rand_score(yte, cl)
        ami = adjusted_mutual_info_score(yte, cl, average_method='arithmetic')
        log(f"\n  k={k}:")
        log(f"    purity vs TRUE labels:      {pur_true:.4f}")
        log(f"    purity vs PREDICTED labels: {pur_pred:.4f}")
        log(f"    homogeneity={homo:.4f} completeness={comp:.4f} "
            f"V-measure={v:.4f}")
        log(f"    ARI={ari:.4f} AMI={ami:.4f}")
        log(f"    cluster sizes: {np.bincount(cl).tolist()}")

        ct_true = pd.crosstab(pd.Series(cl, name='cluster'),
                              pd.Series(yte, name='true_label'))
        ct_pred = pd.crosstab(pd.Series(cl, name='cluster'),
                              pd.Series(y_pred_test, name='pred_label'))
        log(f"\n  Cross-tab k={k} x TRUE label:\n{ct_true.to_string()}")
        log(f"  Cross-tab k={k} x PREDICTED label:\n{ct_pred.to_string()}")

        # 3D: t-SNE on TEST embeddings colored by cluster
        tsne_k = TSNE(n_components=3, perplexity=30, max_iter=1000,
                      init='pca', random_state=42, verbose=0)
        Xk = tsne_k.fit_transform(Xte)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#FF5722']
        for c in np.unique(cl):
            m = cl == c
            ax.scatter(Xk[m, 0], Xk[m, 1], Xk[m, 2], c=colors[c % len(colors)],
                       s=8, alpha=0.6, label=f'cluster {c}')
        ax.set_title(f'k-means k={k} (test embeddings) — clusters')
        ax.legend()
        fig.tight_layout()
        p = os.path.join(OUT_DIR, f"kmeans_k{k}_clusters_3d.png")
        fig.savefig(p, dpi=150)
        plt.close()
        log(f"  Saved: {p}")

        rows.append({
            'k': k,
            'purity_true': round(pur_true, 4),
            'purity_predicted': round(pur_pred, 4),
            'homogeneity': round(homo, 4),
            'completeness': round(comp, 4),
            'v_measure': round(v, 4),
            'ari': round(ari, 4),
            'ami': round(ami, 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "kmeans_purity.csv"), index=False)
    log(f"\nSaved: {os.path.join(OUT_DIR, 'kmeans_purity.csv')}")


def tsne_global_predicted(preds, labels):
    section("2. t-SNE colored by PREDICTED label (all splits, shared space)")
    X_all, y_all, pred_all, split_all = [], [], [], []
    for i, split in enumerate(["train", "dev", "test"]):
        X = np.load(os.path.join(VEC, "bertimbau_large", f"{split}_embeddings.npy"))
        X_all.append(X)
        y_all.append(labels[split])
        pred_all.append(preds[split])
        split_all.append(np.full(len(X), i))
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    pred_all = np.concatenate(pred_all)
    split_all = np.concatenate(split_all)
    log(f"  total rows: {X_all.shape[0]} | predicted pos: {int(pred_all.sum())}")

    tsne = TSNE(n_components=3, perplexity=30, max_iter=1000, init='pca',
                random_state=42, verbose=0)
    Xt = tsne.fit_transform(X_all)
    log("  t-SNE (3D, all 10,331 rows) fitted")

    # colored by predicted label
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for p in [0, 1]:
        m = pred_all == p
        c = '#B0BEC5' if p == 0 else '#E53935'
        ax.scatter(Xt[m, 0], Xt[m, 1], Xt[m, 2], c=c, s=8, alpha=0.6,
                   label=f'predicted={p}' if p == 1 else 'predicted non-fraud=0')
    ax.set_title('t-SNE 3D (BERTimbau Large) — colored by PREDICTED label')
    ax.legend()
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "tsne_3d_by_predicted_label.png")
    fig.savefig(p1, dpi=150)
    plt.close()
    log(f"  Saved: {p1}")

    # misclassification overlay (errors highlighted on TEST points)
    err = pred_all != y_all
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ok = ~err
    ax.scatter(Xt[ok, 0], Xt[ok, 1], Xt[ok, 2], c='#B0BEC5', s=6, alpha=0.5,
               label='correct')
    ax.scatter(Xt[err, 0], Xt[err, 1], Xt[err, 2], c='#E53935', s=40,
               marker='X', alpha=0.9, label=f'ERROR ({int(err.sum())})')
    ax.set_title('t-SNE 3D — misclassifications (all splits)')
    ax.legend()
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "tsne_3d_errors.png")
    fig.savefig(p2, dpi=150)
    plt.close()
    log(f"  Saved: {p2} (total errors across splits: {int(err.sum())})")


def tsne_test_only(labels, preds):
    section("3. t-SNE RESTRICTED to the test set (fresh projection)")
    Xte = np.load(os.path.join(VEC, "bertimbau_large", "test_embeddings.npy"))
    yte = labels['test']
    pte = preds['test']
    log(f"  test-only embeddings: {Xte.shape} | pos: {int(yte.sum())} | "
        f"predicted pos: {int(pte.sum())}")

    tsne = TSNE(n_components=3, perplexity=30, max_iter=1000, init='pca',
                random_state=42, verbose=0)
    Xt = tsne.fit_transform(Xte)
    log("  t-SNE (3D, 2,046 test rows) fitted")

    label_colors = {0: '#B0BEC5', 1: '#E53935'}

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for lab in [0, 1]:
        m = yte == lab
        ax.scatter(Xt[m, 0], Xt[m, 1], Xt[m, 2], c=label_colors[lab], s=10,
                   alpha=0.6, label=f'fraud={lab}' if lab == 1 else 'non-fraud=0')
    ax.set_title('t-SNE 3D TEST-ONLY — colored by TRUE label')
    ax.legend()
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "tsne_test_only_by_true_label.png")
    fig.savefig(p1, dpi=150)
    plt.close()
    log(f"  Saved: {p1}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for p in [0, 1]:
        m = pte == p
        c = '#B0BEC5' if p == 0 else '#E53935'
        ax.scatter(Xt[m, 0], Xt[m, 1], Xt[m, 2], c=c, s=10, alpha=0.6,
                   label=f'predicted={p}' if p == 1 else 'predicted non-fraud=0')
    ax.set_title('t-SNE 3D TEST-ONLY — colored by PREDICTED label')
    ax.legend()
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "tsne_test_only_by_predicted_label.png")
    fig.savefig(p2, dpi=150)
    plt.close()
    log(f"  Saved: {p2}")

    err = pte != yte
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ok = ~err
    ax.scatter(Xt[ok, 0], Xt[ok, 1], Xt[ok, 2], c='#B0BEC5', s=8, alpha=0.5,
               label='correct')
    ax.scatter(Xt[err, 0], Xt[err, 1], Xt[err, 2], c='#E53935', s=50,
               marker='X', alpha=0.95, label=f'ERROR ({int(err.sum())})')
    ax.set_title('t-SNE 3D TEST-ONLY — errors highlighted')
    ax.legend()
    fig.tight_layout()
    p3 = os.path.join(OUT_DIR, "tsne_test_only_errors.png")
    fig.savefig(p3, dpi=150)
    plt.close()
    log(f"  Saved: {p3} (test errors: {int(err.sum())})")


def main():
    section("Task 7b — Test-set cluster analysis")
    model, preds, labels = load_predictions()
    log("  Best model loaded: tfidf/base + LinearSVC (Task 5/7)")

    Xte = np.load(os.path.join(VEC, "bertimbau_large", "test_embeddings.npy"))
    yte = labels['test']
    y_pred_test = preds['test']
    log(f"  test F1: {f1_score(yte, y_pred_test, zero_division=0):.4f}")

    kmeans_analysis(Xte, yte, y_pred_test)
    tsne_global_predicted(preds, labels)
    tsne_test_only(labels, preds)

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    log(f"\nSaved report: {LOG_PATH}")
    print("TASK 7b COMPLETE")


if __name__ == "__main__":
    main()