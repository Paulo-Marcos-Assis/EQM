#!/usr/bin/env python3
"""
Task 8 — Domain Bias Analysis (EQM, multi-portal)
- Metrics by portal: F1/PR-AUC of best model grouped by news host (dev+test, held-out only)
- Cross-domain robustness: ndmais bulk vs NSC volume vs official minority sources
  (mpsc.mp.br, tjsc.jus.br, pc.sc.gov.br) vs other portals
- Feature attribution: top-20 TF-IDF features (LinearSVC + LogisticRegression weights),
  classified as "fraud semantics" vs "portal style"
- Domain bias level + recommendations
Mirrors NDMAIS_BIAS/scripts/task8_domain_bias.py, adapted for multi-portal EQM.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from urllib.parse import urlparse
from sklearn.metrics import (
    f1_score, precision_score, recall_score, average_precision_score,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEC_DIR = os.path.join(BASE_DIR, "vectorization")
RESULTS_DIR = os.path.join(BASE_DIR, "training/results")
DATA_DIR = os.path.join(BASE_DIR, "FOR_TRAINING")
TEST_DIR = os.path.join(BASE_DIR, "FOR_TEST")
OUT_DIR = os.path.join(RESULTS_DIR, "task8_domain_bias")
os.makedirs(OUT_DIR, exist_ok=True)

VEC_KEY = "tfidf"
VARIANT = "base"
BEST_SLUG = "linear_svc"


def extract_portal(link):
    if pd.isna(link) or not link:
        return '(sem link)'
    try:
        host = urlparse(link).netloc.replace('www.', '')
        return host if host else '(sem dominio)'
    except Exception:
        return '(sem dominio)'


def load_best_predictions():
    """Best model predictions + scores on dev and test (held-out only)."""
    model_path = os.path.join(RESULTS_DIR, VEC_KEY, BEST_SLUG, VARIANT, "model.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    preds, scores, labels = {}, {}, {}
    for split, folder in [("dev", DATA_DIR), ("test", TEST_DIR)]:
        X = sparse.load_npz(os.path.join(VEC_DIR, VEC_KEY, f"{split}_sparse.npz"))
        y = np.load(os.path.join(VEC_DIR, VEC_KEY, f"labels_{split}.npy"))
        preds[split] = model.predict(X)
        scores[split] = model.decision_function(X)
        labels[split] = y
    return preds, scores, labels


def group_portal(portal):
    """Assign a portal to a domain group for cross-domain robustness."""
    p = str(portal)
    if p == 'ndmais.com.br':
        return 'ndmais_bulk'
    if p == 'nsctotal.com.br':
        return 'nsc_volume'
    if p in ('mpsc.mp.br', 'tjsc.jus.br', 'pc.sc.gov.br'):
        return 'official_minority'
    return 'other'


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


def get_feature_attribution():
    results = {}
    tfidf_path = os.path.join(VEC_DIR, VEC_KEY, "vectorizer.pkl")
    with open(tfidf_path, 'rb') as f:
        tfidf = pickle.load(f)
    feature_names = tfidf.get_feature_names_out()

    for slug in ['linear_svc', 'logistic_regression']:
        model_path = os.path.join(RESULTS_DIR, VEC_KEY, slug, VARIANT, "model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        coefs = model.coef_[0]
        top_pos = np.argsort(coefs)[-20:][::-1]
        top_neg = np.argsort(coefs)[:20]
        results[f'tfidf_{slug}'] = {
            'top_fraud': [(feature_names[i], float(coefs[i])) for i in top_pos],
            'top_nonfraud': [(feature_names[i], float(coefs[i])) for i in top_neg],
        }
    return results


def classify_features(fraud_features):
    semantic_fraud = [
        'licitac', 'irregularidade', 'fraude', 'fraudes', 'operacao', 'empresa',
        'contrato', 'ex-prefeito', 'compra', 'respirador', 'cpi', 'prefeito',
        'tce', 'corrupcao', 'investigacao', 'processo', 'publico', 'desvio',
        'propina', 'superfaturamento', 'criminoso', 'tributario', 'sonegacao',
        'peculato', 'prevaricacao', 'corrupto', 'golpe', 'estelionato',
        'malversacao', 'advogado', 'mpsc', 'justica', 'stf', 'stj', 'ministerio',
        'denuncia', 'indiciamento', 'inquerito', 'mandado', 'prisao',
        'condenacao', 'pena', 'furto', 'roubo', 'recurso', 'pagamento',
        'prestador', 'servico', 'superfaturamento', 'licitantes', 'processual',
        'municipal', 'estadual', 'politica', 'político', 'vereador',
    ]
    stylized_portal = [
        'assinatura', 'redacao', 'fale conosco', 'editoria', 'repórter',
        'publicidade', 'publicadas', 'copyright', 'todos os direitos',
        'leia mais', 'veja tambem', 'compartilhe', 'comente', 'curta',
        'siga', 'newsletter', 'assine', 'edicoes', 'acesso', 'facebook',
        'twitter', 'instagram', 'youtube', 'whatsapp', 'contato',
    ]
    classified = {'semantic_fraud': [], 'stylized_portal': [], 'other': []}
    for feat, weight in fraud_features:
        feat_lower = feat.lower()
        if any(s in feat_lower for s in semantic_fraud):
            classified['semantic_fraud'].append((feat, weight))
        elif any(s in feat_lower for s in stylized_portal):
            classified['stylized_portal'].append((feat, weight))
        else:
            classified['other'].append((feat, weight))
    return classified


def run():
    log_path = os.path.join(OUT_DIR, '_logs', 'task8_stdout.txt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, 'w')
    tee = Tee(sys.stdout, log_file)
    sys.stdout = tee

    print("=" * 72)
    print("TASK 8 — Domain Bias Analysis (EQM, multi-portal)")
    print("=" * 72)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    dev_df = pd.read_csv(os.path.join(DATA_DIR, "dev.csv"))
    test_df = pd.read_csv(os.path.join(TEST_DIR, "test.csv"))

    # === 1. Portal distribution ===
    print("\n--- 1. Portal distribution ---")
    all_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    all_df['portal'] = all_df['link_noticia'].apply(extract_portal)
    unique_portals = all_df['portal'].nunique()
    pos_by_portal = all_df[all_df.label == 1]['portal'].value_counts()
    main_volume = all_df['portal'].value_counts().idxmax()
    main_pos = pos_by_portal.idxmax()
    print(f"  Total unique portals: {unique_portals}")
    print(f"  Portals with positives: {len(pos_by_portal)}")
    print(f"  Largest portal by volume: {main_volume} "
          f"({all_df['portal'].value_counts().max()} rows)")
    print(f"  Largest portal by fraud positives: {main_pos} ({pos_by_portal.max()} pos)")
    for name, df in [('Train', train_df), ('Dev', dev_df), ('Test', test_df)]:
        portals = df['link_noticia'].apply(extract_portal)
        print(f"  {name}: {len(df)} rows ({int((df.label == 1).sum())} pos), "
              f"{portals.nunique()} portals")

    # === 2. Metrics by portal (best model on dev+test held-out) ===
    print("\n--- 2. Metrics by portal (best model: tfidf/base + LinearSVC, dev+test) ---")
    preds, scores, labels = load_best_predictions()

    devp = dev_df['link_noticia'].apply(extract_portal)
    testp = test_df['link_noticia'].apply(extract_portal)
    all_portal = pd.concat([devp, testp], ignore_index=True)
    all_label = np.concatenate([labels['dev'], labels['test']])
    all_pred = np.concatenate([preds['dev'], preds['test']])
    all_score = np.concatenate([scores['dev'], scores['test']])

    portal_metrics_rows = []
    for portal in all_portal.unique():
        m = (all_portal == portal).values
        n = int(m.sum())
        n_pos = int(all_label[m].sum())
        if n_pos == 0:
            continue
        f1 = f1_score(all_label[m], all_pred[m], zero_division=0)
        prec = precision_score(all_label[m], all_pred[m], zero_division=0)
        rec = recall_score(all_label[m], all_pred[m], zero_division=0)
        try:
            prauc = average_precision_score(all_label[m], all_score[m])
        except Exception:
            prauc = float('nan')
        portal_metrics_rows.append({
            'portal': portal, 'n': n, 'pos': n_pos,
            'f1': f1, 'precision': prec, 'recall': rec, 'pr_auc': prauc,
        })

    pm = pd.DataFrame(portal_metrics_rows)
    pm = pm.sort_values('pos', ascending=False)
    pd.set_option('display.width', 150)
    pd.set_option('display.max_rows', None)
    print(pm.to_string(index=False, formatters={
        'f1': '{:.4f}'.format, 'precision': '{:.4f}'.format,
        'recall': '{:.4f}'.format, 'pr_auc': '{:.4f}'.format}))
    pm.to_csv(os.path.join(OUT_DIR, 'portal_metrics.csv'), index=False)

    # === 3. Cross-domain robustness by group ===
    print("\n--- 3. Cross-domain robustness (grouped sources, dev+test) ---")
    groups = all_portal.apply(group_portal)
    group_rows = []
    for g in groups.unique():
        m = (groups == g).values
        n = int(m.sum())
        n_pos = int(all_label[m].sum())
        f1 = f1_score(all_label[m], all_pred[m], zero_division=0)
        prec = precision_score(all_label[m], all_pred[m], zero_division=0)
        rec = recall_score(all_label[m], all_pred[m], zero_division=0)
        try:
            prauc = average_precision_score(all_label[m], all_score[m])
        except Exception:
            prauc = float('nan')
        group_rows.append({
            'group': g, 'n': n, 'pos': n_pos,
            'f1': f1, 'precision': prec, 'recall': rec, 'pr_auc': prauc,
        })
        print(f"  {g:<16} n={n:5d} pos={n_pos:4d} F1={f1:.4f} "
              f"Prec={prec:.4f} Rec={rec:.4f} PR-AUC={prauc:.4f}")
    gdf = pd.DataFrame(group_rows)
    gdf.to_csv(os.path.join(OUT_DIR, 'cross_domain_robustness.csv'), index=False)

    official = gdf[gdf.group == 'official_minority']
    ndmais = gdf[gdf.group == 'ndmais_bulk']
    if len(official) and len(ndmais):
        print(f"\n  Official-minority F1 ({official['f1'].iloc[0]:.4f}) vs "
              f"ndmais bulk F1 ({ndmais['f1'].iloc[0]:.4f})")

    # === 4. Feature attribution ===
    print("\n--- 4. Feature Attribution (TF-IDF top-20) ---")
    feature_attribution = get_feature_attribution()
    classified_all = {}
    for model_name, feats in feature_attribution.items():
        c = classify_features(feats['top_fraud'])
        classified_all[model_name] = c
        n_sem = len(c['semantic_fraud'])
        n_sty = len(c['stylized_portal'])
        n_oth = len(c['other'])
        print(f"\n  === {model_name} ===")
        print(f"  Top-20 fraud features: semantic={n_sem}/20 stylized={n_sty}/20 other={n_oth}/20")
        for i, (feat, weight) in enumerate(feats['top_fraud']):
            tag = '[SEMANTIC]' if feat in [f for f, _ in c['semantic_fraud']] else \
                  '[STYLIZED]' if feat in [f for f, _ in c['stylized_portal']] else '[OTHER]'
            print(f"    {i+1:2d}. {feat:<32s} w={weight:+.4f} {tag}")

    # === 5. Domain bias assessment ===
    print("\n--- 5. Domain bias assessment ---")
    svc_ratio = len(classified_all['tfidf_linear_svc']['semantic_fraud']) / 20
    lr_ratio = len(classified_all['tfidf_logistic_regression']['semantic_fraud']) / 20
    print(f"  LinearSVC semantic ratio: {svc_ratio:.0%} ({int(svc_ratio*20)}/20)")
    print(f"  LR semantic ratio:        {lr_ratio:.0%} ({int(lr_ratio*20)}/20)")

    if svc_ratio >= 0.7 and lr_ratio >= 0.7:
        bias_level = "LOW"
        explanation = ("Fraud-semantic features dominate (>70% of top-20 for both models). "
                       "Model learns the fraud CONCEPT, not portal style.")
    elif svc_ratio >= 0.5 or lr_ratio >= 0.5:
        bias_level = "MEDIUM"
        explanation = ("Mixed semantic/stylistic. Partial portal-bias risk.")
    else:
        bias_level = "HIGH"
        explanation = ("Stylistic features dominate. HIGH portal-bias risk.")
    print(f"  Domain bias level: {bias_level}")
    print(f"  {explanation}")

    # === Save ===
    summary = {
        'portal_distribution': {
            'total_portals': int(unique_portals),
            'portals_with_positives': int(len(pos_by_portal)),
            'largest_by_volume': main_volume,
            'largest_by_fraud_positives': main_pos,
            'splits': {
                'train': {'total': int(len(train_df)), 'pos': int((train_df.label == 1).sum()),
                          'portals': int(train_df['link_noticia'].apply(extract_portal).nunique())},
                'dev': {'total': int(len(dev_df)), 'pos': int((dev_df.label == 1).sum()),
                        'portals': int(dev_df['link_noticia'].apply(extract_portal).nunique())},
                'test': {'total': int(len(test_df)), 'pos': int((test_df.label == 1).sum()),
                         'portals': int(test_df['link_noticia'].apply(extract_portal).nunique())},
            },
        },
        'feature_attribution': {
            k: {kk: [(f, w) for f, w in vv] for kk, vv in v.items()}
            for k, v in feature_attribution.items()
        },
        'domain_bias_assessment': {
            'level': bias_level,
            'explanation': explanation,
            'svc_semantic_ratio': svc_ratio,
            'lr_semantic_ratio': lr_ratio,
        },
    }
    with open(os.path.join(OUT_DIR, 'task8_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # DOMAIN_BIAS_REPORT.md
    with open(os.path.join(OUT_DIR, 'DOMAIN_BIAS_REPORT.md'), 'w') as f:
        f.write("# Task 8 — Domain Bias Analysis Report (EQM)\n\n")
        f.write(f"**Date:** 2026-08-19\n\n")
        f.write("## 1. Portal Distribution\n\n")
        f.write(f"- **Total unique portals:** {unique_portals} (multi-portal dataset)\n")
        f.write(f"- **Portals with fraud positives:** {len(pos_by_portal)}\n")
        f.write(f"- **Largest by volume:** {main_volume}\n")
        f.write(f"- **Largest by fraud positives:** {main_pos} ({pos_by_portal.max()})\n\n")
        f.write("| Split | Total | Pos | Neg | Portals |\n")
        f.write("|-------|-------|-----|-----|----------|\n")
        for split_name, sd in summary['portal_distribution']['splits'].items():
            f.write(f"| {split_name.title()} | {sd['total']} | {sd['pos']} | "
                    f"{sd['total']-sd['pos']} | {sd['portals']} |\n")

        f.write("\n## 2. Metrics by Portal (best model, dev+test held-out)\n\n")
        f.write("| Portal | n | Pos | F1 | Precision | Recall | PR-AUC |\n")
        f.write("|--------|---|-----|----|-----------|--------|--------|\n")
        for _, r in pm.head(25).iterrows():
            f.write(f"| {r['portal']} | {r['n']} | {r['pos']} | {r['f1']:.4f} | "
                    f"{r['precision']:.4f} | {r['recall']:.4f} | {r['pr_auc']:.4f} |\n")

        f.write("\n## 3. Cross-Domain Robustness\n\n")
        f.write("| Group | n | Pos | F1 | Precision | Recall | PR-AUC |\n")
        f.write("|-------|---|-----|----|-----------|--------|--------|\n")
        for _, r in gdf.iterrows():
            f.write(f"| {r['group']} | {r['n']} | {r['pos']} | {r['f1']:.4f} | "
                    f"{r['precision']:.4f} | {r['recall']:.4f} | {r['pr_auc']:.4f} |\n")

        f.write("\n## 4. Feature Attribution (top-20 fraud features)\n\n")
        for model_name, feats in feature_attribution.items():
            c = classified_all[model_name]
            f.write(f"### {model_name}\n\n")
            f.write(f"- Semantic (fraud): {len(c['semantic_fraud'])}/20\n")
            f.write(f"- Stylized (portal): {len(c['stylized_portal'])}/20\n")
            f.write(f"- Other: {len(c['other'])}/20\n\n")
            f.write("| Rank | Feature | Weight | Type |\n")
            f.write("|------|---------|--------|------|\n")
            for i, (feat, weight) in enumerate(feats['top_fraud']):
                ftype = 'Semantic' if feat in [f for f, _ in c['semantic_fraud']] else \
                        'Stylized' if feat in [f for f, _ in c['stylized_portal']] else 'Other'
                f.write(f"| {i+1} | {feat} | {weight:+.4f} | {ftype} |\n")
            f.write("\n")

        f.write("## 5. Domain Bias Assessment\n\n")
        f.write(f"- **Bias level:** {bias_level}\n")
        f.write(f"- **LinearSVC semantic ratio:** {svc_ratio:.0%}\n")
        f.write(f"- **LR semantic ratio:** {lr_ratio:.0%}\n")
        f.write(f"- **Explanation:** {explanation}\n\n")

        f.write("## 6. Recommendations\n\n")
        f.write("1. Multi-portal dataset (74 domains) — cross-domain evaluation IS possible (unlike NDMAIS single-portal)\n")
        f.write("2. Official minority sources (mpsc.mp.br, tjsc.jus.br, pc.sc.gov.br) evaluated separately\n")
        f.write("3. Feature attribution determines whether the model learns fraud semantics or portal style\n")
        f.write("4. **Future work:** diversify further (more portals per source type), monitor F1 on out-of-distribution portals\n")

    print(f"\nResults saved: {OUT_DIR}")
    print(f"  - task8_summary.json")
    print(f"  - DOMAIN_BIAS_REPORT.md")
    print(f"  - portal_metrics.csv")
    print(f"  - cross_domain_robustness.csv")

    print(f"\n{'=' * 72}")
    print("TASK 8 COMPLETE")
    print(f"{'=' * 72}")

    sys.stdout = sys.__stdout__
    log_file.close()
    print(f"Log: {log_path}")


if __name__ == "__main__":
    run()