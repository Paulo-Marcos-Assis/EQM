#!/usr/bin/env python3
"""
Task 5 — Training (12 combinations)
- 4 classifiers × 3 base vectorizations (TF-IDF, BERTimbau Base, BERTimbau Large) = 12 combos
  saved in training/results/<vectorization>/<algorithm>/base/
- GridSearchCV StratifiedKFold(5) on training (scoring f1); retrains with best params;
  evaluates on dev. Test remains isolated (Task 7).
- TF-IDF stem variant is NOT trained here — it belongs to Task 6 (stem ablation).
- Resume: combos already trained (classification_report.txt present) are skipped.
"""

import os
import sys
import json
import pickle
import numpy as np
from scipy import sparse

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score, accuracy_score,
)
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task5_utils import generate_consolidated_reports

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEC_DIR = os.path.join(BASE_DIR, "vectorization")
RESULTS_DIR = os.path.join(BASE_DIR, "training/results")

# (vec_key, vec_type, variant) — base runs + TF-IDF RSLP stem variant (mirrors NDMAIS_BIAS)
VECTORIZATIONS = [
    ("tfidf", "sparse", "base"),
    ("tfidf_stem", "sparse", "stem"),
    ("bertimbau_base", "dense", "base"),
    ("bertimbau_large", "dense", "base"),
]

N_SPLITS = 5


def make_estimators():
    ests = []
    ests.append({
        'name': 'LinearSVC',
        'slug': 'linear_svc',
        'clf': LinearSVC(class_weight='balanced', dual='auto',
                         max_iter=10000, random_state=42),
        'grid': {'C': [0.1, 1, 10]},
        'score': 'decision_function',
        'desc': (
            "SVM (Support Vector Machine) finds the optimal hyperplane that best separates the "
            "classes.\nLinearSVC: linear kernel, more efficient in high dimensionality.\n"
            "class_weight='balanced': adjusts weights inversely proportional to class frequency.\n"
            "decision_function provides scores for PR-AUC and ROC-AUC."
        ),
        'config': (
            "LinearSVC(class_weight='balanced', dual='auto', max_iter=10000, random_state=42)\n"
            "Grid: C=[0.1, 1, 10]\n"
            "CV: StratifiedKFold(5, shuffle=True, random_state=42)\n"
            "Scoring: f1"
        ),
    })
    ests.append({
        'name': 'LogisticRegression',
        'slug': 'logistic_regression',
        'clf': LogisticRegression(class_weight='balanced', max_iter=2000,
                                  random_state=42),
        'grid': {'C': [0.1, 1, 10]},
        'score': 'predict_proba',
        'desc': (
            "Logistic Regression: linear, interpretable probabilistic baseline.\n"
            "class_weight='balanced': adjusts weights inversely proportional to class frequency.\n"
            "predict_proba provides scores for PR-AUC and ROC-AUC."
        ),
        'config': (
            "LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)\n"
            "Grid: C=[0.1, 1, 10]\n"
            "CV: StratifiedKFold(5, shuffle=True, random_state=42)\n"
            "Scoring: f1"
        ),
    })
    ests.append({
        'name': 'RandomForest',
        'slug': 'random_forest',
        'clf': RandomForestClassifier(class_weight='balanced', random_state=42,
                                      n_jobs=-1),
        'grid': {'n_estimators': [100, 200], 'max_depth': [20, 50, None]},
        'score': 'predict_proba',
        'desc': (
            "Random Forest: ensemble of trees that captures non-linearities and interactions.\n"
            "class_weight='balanced': adjusts class weights.\n"
            "predict_proba provides scores for PR-AUC and ROC-AUC."
        ),
        'config': (
            "RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)\n"
            "Grid: n_estimators=[100, 200] × max_depth=[20, 50, None]\n"
            "CV: StratifiedKFold(5, shuffle=True, random_state=42)\n"
            "Scoring: f1"
        ),
    })
    return ests


def make_xgb_estimator(y_train):
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos
    return {
        'name': 'XGBoost',
        'slug': 'xgboost',
        'clf': XGBClassifier(random_state=42, eval_metric='logloss',
                             tree_method='hist', scale_pos_weight=scale_pos_weight,
                             n_jobs=-1),
        'grid': {'max_depth': [4, 6, 8], 'learning_rate': [0.05, 0.1]},
        'score': 'predict_proba',
        'desc': (
            "XGBoost: gradient boosting with tree_method='hist' (histogram-based) — fast and "
            "stable on sparse data.\n"
            f"Dynamic scale_pos_weight = n_neg/n_pos = {scale_pos_weight:.2f} "
            f"(n_neg={n_neg}, n_pos={n_pos}) — computed in script, never hardcoded.\n"
            "predict_proba provides scores for PR-AUC and ROC-AUC."
        ),
        'config': (
            "XGBClassifier(random_state=42, eval_metric='logloss', tree_method='hist',\n"
            f"             scale_pos_weight={scale_pos_weight:.2f}, n_jobs=-1)\n"
            "Grid: max_depth=[4, 6, 8] × learning_rate=[0.05, 0.1]\n"
            "CV: StratifiedKFold(5, shuffle=True, random_state=42)\n"
            "Scoring: f1"
        ),
    }


def load_data(vec_key, vec_type):
    vec_path = os.path.join(VEC_DIR, vec_key)
    if vec_type == "sparse":
        X_train = sparse.load_npz(os.path.join(vec_path, "train_sparse.npz"))
        X_dev = sparse.load_npz(os.path.join(vec_path, "dev_sparse.npz"))
    else:
        X_train = np.load(os.path.join(vec_path, "train_embeddings.npy"))
        X_dev = np.load(os.path.join(vec_path, "dev_embeddings.npy"))
    y_train = np.load(os.path.join(vec_path, "labels_train.npy"))
    y_dev = np.load(os.path.join(vec_path, "labels_dev.npy"))
    return X_train, X_dev, y_train, y_dev


def get_scores(model, X_dev, score_type):
    if score_type == 'decision_function':
        return model.decision_function(X_dev)
    return model.predict_proba(X_dev)[:, 1]


def evaluate_model(y_true, y_pred, y_scores=None):
    metrics = {
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
    }
    if y_scores is not None:
        metrics['pr_auc'] = average_precision_score(y_true, y_scores)
        metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
    return metrics


def format_cv_table(grid):
    cv_res = grid.cv_results_
    lines = []
    lines.append("Full CV table (mean ± std per candidate):")
    lines.append("  | Params | CV F1 mean ± std |")
    lines.append("  |--------|-------------------|")
    for mean, std, params in zip(cv_res['mean_test_score'],
                                 cv_res['std_test_score'],
                                 cv_res['params']):
        lines.append(f"  | {params} | {mean:.4f} ± {std:.4f} |")
    return "\n".join(lines)


def save_results(out_dir, vec_key, variant, best_params, cv_f1, cv_table,
                 dev_metrics, y_dev, y_pred, model):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "best_params.json"), 'w') as f:
        json.dump(best_params, f, indent=2)

    with open(os.path.join(out_dir, "model.pkl"), 'wb') as f:
        pickle.dump(model, f)

    report = classification_report(y_dev, y_pred, zero_division=0)
    label = f"{vec_key}/{variant}"
    with open(os.path.join(out_dir, "classification_report.txt"), 'w') as f:
        f.write(f"Task 5 — {label}\n")
        f.write(f"Best params: {best_params}\n")
        f.write(f"CV F1 (mean): {cv_f1:.4f}\n")
        f.write(f"\n{cv_table}\n")
        f.write(f"\nDev metrics:\n")
        for k, v in dev_metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write(f"\n{report}\n")

    cm = confusion_matrix(y_dev, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(label)
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
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


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


def load_existing_result(out_dir, vec_key, variant, slug):
    """Reload metrics of an already-trained combo from its saved files,
    so consolidated reports include ALL combos, including skipped ones."""
    with open(os.path.join(out_dir, "best_params.json")) as f:
        best_params = json.load(f)
    metrics = {}
    with open(os.path.join(out_dir, "classification_report.txt")) as f:
        for line in f:
            line = line.strip()
            if line.startswith("CV F1 (mean):"):
                metrics['cv_f1'] = float(line.split(":")[1].strip())
            elif line.startswith("f1:") or line.startswith("precision:") or \
                 line.startswith("recall:") or line.startswith("pr_auc:") or \
                 line.startswith("roc_auc:") or line.startswith("accuracy:"):
                k, v = line.split(":")
                metrics[k.strip()] = float(v.strip())
    return {
        'vectorization': vec_key,
        'variant': variant,
        'classifier': slug,
        'best_params': best_params,
        'cv_f1': metrics.get('cv_f1', 0),
        'f1': metrics.get('f1', 0),
        'precision': metrics.get('precision', 0),
        'recall': metrics.get('recall', 0),
        'pr_auc': metrics.get('pr_auc', 0),
        'roc_auc': metrics.get('roc_auc', 0),
        'accuracy': metrics.get('accuracy', 0),
    }


def run():
    main_log_dir = os.path.join(RESULTS_DIR, "_logs")
    os.makedirs(main_log_dir, exist_ok=True)
    log_path = os.path.join(main_log_dir, "task5_execution_log.txt")
    log_file = open(log_path, 'w')
    tee = Tee(sys.stdout, log_file)
    sys.stdout = tee

    print("=" * 72)
    print("TASK 5 — Training (16 combinations)")
    print("=" * 72)

    # Estimate y_train once for XGBoost scale_pos_weight
    _, _, y_train_ref, _ = load_data("tfidf", "sparse")

    classifiers = make_estimators()
    classifiers.append(make_xgb_estimator(y_train_ref))

    total_combos = len(VECTORIZATIONS) * len(classifiers)
    done = skipped = 0

    all_by_classifier = {c['slug']: [] for c in classifiers}

    for vec_key, vec_type, variant in VECTORIZATIONS:
        X_train, X_dev, y_train, y_dev = load_data(vec_key, vec_type)
        for est in classifiers:
            done += 1
            label = f"{vec_key}/{variant} + {est['slug']}"
            out_dir = os.path.join(RESULTS_DIR, vec_key, est['slug'], variant)
            marker = os.path.join(out_dir, "classification_report.txt")

            if os.path.exists(marker):
                print(f"[{done}/{total_combos}] SKIP (already trained): {label}")
                skipped += 1
                all_by_classifier[est['slug']].append(
                    load_existing_result(out_dir, vec_key, variant, est['slug'])
                )
                continue

            print(f"\n{'=' * 64}")
            print(f"[{done}/{total_combos}] {label}")
            print(f"{'=' * 64}")
            print(f"  Train: {X_train.shape} | Dev: {X_dev.shape} "
                  f"| pos={int((y_train == 1).sum())} neg={int((y_train == 0).sum())}")

            clf = est['clf']
            grid = est['grid']
            n_candidates = np.prod([len(v) for v in grid.values()])
            print(f"  GridSearchCV: {n_candidates} candidates × {N_SPLITS} folds "
                  f"= {int(n_candidates * N_SPLITS)} fits")

            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
            gs = GridSearchCV(clf, grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
            gs.fit(X_train, y_train)

            print(f"  Best params: {gs.best_params_}")
            print(f"  CV F1 (mean): {gs.best_score_:.4f}")
            cv_table = format_cv_table(gs)

            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_dev)
            y_scores = get_scores(best_model, X_dev, est['score'])
            dev_metrics = evaluate_model(y_dev, y_pred, y_scores)
            for k, v in dev_metrics.items():
                print(f"  Dev {k}: {v:.4f}")

            save_results(out_dir, vec_key, variant, gs.best_params_,
                         gs.best_score_, cv_table, dev_metrics,
                         y_dev, y_pred, best_model)

            all_by_classifier[est['slug']].append({
                'vectorization': vec_key,
                'variant': variant,
                'classifier': est['slug'],
                'best_params': gs.best_params_,
                'cv_f1': gs.best_score_,
                **dev_metrics,
            })
            print(f"  DONE — F1={dev_metrics['f1']:.4f}")

    print(f"\nSummary: {done} combos processed, {skipped} skipped (already existing)")

    print("\nGenerating consolidated reports per classifier...")
    for est in classifiers:
        slug = est['slug']
        if all_by_classifier[slug]:
            generate_consolidated_reports(
                all_by_classifier[slug], RESULTS_DIR, est['name'],
                est['desc'], est['config'], f"{slug}_results.json"
            )
            print(f"  {est['name']}: {len(all_by_classifier[slug])} consolidated results")

    print(f"\n{'=' * 72}")
    print("TASK 5 COMPLETE")
    print(f"{'=' * 72}")

    sys.stdout = sys.__stdout__
    log_file.close()
    print(f"Log: {log_path}")


if __name__ == "__main__":
    run()