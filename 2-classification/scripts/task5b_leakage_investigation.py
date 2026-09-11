#!/usr/bin/env python3
"""
Task 5b — Leakage Investigation & Pipeline Integrity (mirrors NDMAIS_BIAS)

Confirms the pipeline is methodologically sound before Task 6:
- A. Split integrity       — row identity/link/exact-text overlap across train/dev/test
- B. Label file cross-check — labels_*.npy vs raw CSV labels (count, order, values)
- C. Textual near-duplicates— cross-split exact overlap on processed text (Embeddings branch)
- D. Embedding duplicates   — dev/test rows with near-identical train neighbor (cosine; BERTimbau Large)
- E. Label-permutation test — train with shuffled labels; sound pipeline must collapse to ~random
- F. Memorization check    — LR on half the train; F1 must NOT stay high
- G. Leak-dropped retrain  — retrain LR after removing cross-split near-dups (cos>=0.99); F1 delta

t-SNE visualizations (BERTimbau Large embeddings):
- training/results/visualizations/tsne_3d_by_label.png (color by label — genuine separability?)
- training/results/visualizations/tsne_3d_by_split.png (color by split — no cross-split clustering)
- training/results/_logs/tsne_bertimbau_large.png     (diagnostic 2D)
"""

import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, recall_score
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FT = os.path.join(BASE, "FOR_TRAINING")
FTE = os.path.join(BASE, "FOR_TEST")
VEC = os.path.join(BASE, "vectorization")
OUT_DIR = os.path.join(BASE, "training/results/_logs")
VIS_DIR = os.path.join(BASE, "training/results/visualizations")
OUT = os.path.join(OUT_DIR, "LEAKAGE_INVESTIGATION.txt")

log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


def section(title):
    log("\n" + "=" * 72)
    log(f"### {title}")
    log("=" * 72)


def load_splits():
    train = pd.read_csv(os.path.join(FT, "train.csv"))
    dev = pd.read_csv(os.path.join(FT, "dev.csv"))
    test = pd.read_csv(os.path.join(FTE, "test.csv"))
    return train, dev, test


def set_overlap(a, b, name):
    n = len(a.intersection(b))
    log(f"  overlap {name}: {n}")
    return n


# ---------- A. Split integrity ----------
section("A. Split integrity (raw CSVs)")
train, dev, test = load_splits()
log(f"  counts: train={len(train)} dev={len(dev)} test={len(test)} | "
    f"sum={len(train)+len(dev)+len(test)}")

idx_col = 'original_index' if 'original_index' in train.columns else None
log(f"  original_index col present: {idx_col}")
if idx_col:
    ti, di, te_i = set(train[idx_col]), set(dev[idx_col]), set(test[idx_col])
    log(f"  original_index overlap train/dev: {len(ti.intersection(di))}")
    log(f"  original_index overlap train/test: {len(ti.intersection(te_i))}")
    log(f"  original_index overlap dev/test: {len(di.intersection(te_i))}")


def link_overlap(a, b, an):
    sa = set(a['link_noticia'].dropna())
    sb = set(b['link_noticia'].dropna())
    sa.discard(''); sb.discard('')
    n = len(sa.intersection(sb))
    log(f"  link overlap {an}: {n}")
    return n


link_overlap(train, dev, "train/dev")
link_overlap(train, test, "train/test")
link_overlap(dev, test, "dev/test")

for a, b, an in [(train, dev, "train/dev"), (train, test, "train/test"),
                 (dev, test, "dev/test")]:
    if 'texto_completo' in a.columns and 'texto_completo' in b.columns:
        sa, sb = set(a['texto_completo']), set(b['texto_completo'])
        log(f"  exact texto_completo overlap {an}: {len(sa.intersection(sb))}")

for df, name in [(train, "train"), (dev, "dev"), (test, "test")]:
    dups = df['texto_completo'].duplicated().sum()
    log(f"  texto_completo duplicated within {name}: {dups}")

# ---------- B. Label file cross-check ----------
section("B. Label files vs raw CSVs")
raw_labels = {"train": train['label'].values, "dev": dev['label'].values,
              "test": test['label'].values}
for vec in ["tfidf", "bertimbau_base", "bertimbau_large"]:
    ok = True
    for split in ["train", "dev", "test"]:
        path = os.path.join(VEC, vec, f"labels_{split}.npy")
        if not os.path.exists(path):
            ok = False
            log(f"  {vec}/{split}: MISSING {path}")
            continue
        arr = np.load(path)
        if arr.shape != raw_labels[split].shape or not np.array_equal(arr, raw_labels[split]):
            ok = False
            log(f"  {vec}/{split}: MISMATCH shape={arr.shape} raw={raw_labels[split].shape} "
                f"equal={np.array_equal(arr, raw_labels[split])}")
    if ok:
        log(f"  {vec}: labels ok (identical to raw CSVs, same order)")

# ---------- C. Textual near-duplicates cross-split (processed) ----------
section("C. Cross-split near-duplicates on PROCESSED text (Embeddings branch)")
bert_df = {
    "train": pd.read_csv(os.path.join(FT, "Pre_processed_for_Embeddings/train_bert.csv")),
    "dev": pd.read_csv(os.path.join(FT, "Pre_processed_for_Embeddings/dev_bert.csv")),
    "test": pd.read_csv(os.path.join(FTE, "Pre_processed_for_Embeddings/test_bert.csv")),
}
for a, b, an in [("train", "dev", "train/dev"), ("train", "test", "train/test"),
                 ("dev", "test", "dev/test")]:
    sa, sb = set(bert_df[a]['processed_text']), set(bert_df[b]['processed_text'])
    log(f"  exact processed_text overlap {an}: {len(sa.intersection(sb))}")

# ---------- D. Embedding nearest-neighbor cross-split ----------
section("D. Embedding near-identical pairs across splits (bertimbau_large)")

def load_emb(vec, split):
    X = np.load(os.path.join(VEC, vec, f"{split}_embeddings.npy"))
    y = np.load(os.path.join(VEC, vec, f"labels_{split}.npy"))
    return X, y


Xtr, ytr = load_emb("bertimbau_large", "train")
Xde, yde = load_emb("bertimbau_large", "dev")
Xte, yte = load_emb("bertimbau_large", "test")
log(f"  shapes: train={Xtr.shape} dev={Xde.shape} test={Xte.shape}")

tr_n = normalize(Xtr, norm='l2')


def max_cos_to_train(Xq):
    q_n = normalize(Xq, norm='l2')
    out = np.zeros(len(q_n), dtype=np.float32)
    chunk = 128
    for i in range(0, len(q_n), chunk):
        sim = q_n[i:i + chunk] @ tr_n.T
        out[i:i + chunk] = sim.max(axis=1)
    return out


def report_cos(name, X, y):
    log(f"\n  --- {name}: max cosine to train ---")
    mcos = max_cos_to_train(X)
    for thr in [0.9999, 0.999, 0.995, 0.99, 0.95]:
        mask = mcos >= thr
        log(f"    sim>= {thr}: {mask.sum()} rows "
            f"(pos={int((y[mask]==1).sum())}, neg={int((y[mask]==0).sum())})")
    q_n = normalize(X, norm='l2')
    sim = q_n @ tr_n.T
    top = sim.argmax(axis=1)
    top_sim = sim.max(axis=1)
    order = np.argsort(-top_sim)[:10]
    log(f"    Top-10 nearest train neighbors:")
    for i in order:
        j = top[i]
        log(f"      {name.lower()} row {i} (label={y[i]}) ~ train row {j} (label={ytr[j]}) "
            f"cos={top_sim[i]:.6f} {'SAME-LABEL' if y[i]==ytr[j] else 'DIFF-LABEL'}")
    return mcos


dev_cos = report_cos("DEV", Xde, yde)
test_cos = report_cos("TEST", Xte, yte)

# ---------- E. Label-permutation sanity test ----------
section("E. Label-permutation test (bertimbau_large, LR C=10)")
rng = np.random.RandomState(42)
y_shuff = ytr.copy()
rng.shuffle(y_shuff)
lr_perm = LogisticRegression(C=10, max_iter=2000).fit(Xtr, y_shuff)
y_perm_pred = lr_perm.predict(Xde)
log(f"  LR trained on shuffled labels -> dev F1 = {f1_score(yde, y_perm_pred):.4f} "
    f"(recall={recall_score(yde, y_perm_pred):.4f}). "
    f"If ~random (~0.09), plumbing is sound; if high, embedding/label misaligned.")

# ---------- F. Memorization check ----------
section("F. Memorization check — LR on half the train (bertimbau_large)")
Xtr_h, ytr_h = Xtr[:len(Xtr)//2], ytr[:len(ytr)//2]
lr_half = LogisticRegression(C=10, max_iter=2000).fit(Xtr_h, ytr_h)
log(f"  LR on {len(Xtr_h)} rows -> dev F1 = {f1_score(yde, lr_half.predict(Xde)):.4f}")

# ---------- G. Leak-dropped retrain ----------
section("G. Retrain after dropping cross-split near-duplicates (bertimbau_large, LR C=10)")
q_n_de = normalize(Xde, norm='l2')
sim_de = q_n_de @ tr_n.T
anchor_dev = set()
for i in np.where(dev_cos >= 0.99)[0]:
    anchor_dev.add(int(sim_de[i].argmax()))
q_n_te = normalize(Xte, norm='l2')
sim_te = q_n_te @ tr_n.T
anchor_test = set()
for i in np.where(test_cos >= 0.99)[0]:
    anchor_test.add(int(sim_te[i].argmax()))
anchors = anchor_dev | anchor_test
log(f"  train anchors involved in near-dup (cos>=0.99) with dev/test: {len(anchors)}")

lr_full = LogisticRegression(C=10, max_iter=2000).fit(Xtr, ytr)
f1_full = f1_score(yde, lr_full.predict(Xde))

if anchors:
    keep_mask = np.ones(len(ytr), dtype=bool)
    keep_mask[list(anchors)] = False
    lr_drop = LogisticRegression(C=10, max_iter=2000).fit(Xtr[keep_mask], ytr[keep_mask])
    f1_drop = f1_score(yde, lr_drop.predict(Xde))
    log(f"  LR full train : dev F1 = {f1_full:.4f}")
    log(f"  LR minus {len(anchors)} near-dup anchors : dev F1 = {f1_drop:.4f}")
    log(f"  => drop delta = {f1_full - f1_drop:+.4f}")
else:
    log(f"  LR full train : dev F1 = {f1_full:.4f}")
    log(f"  No near-dup anchors found at 0.99 -> no drop test possible")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines) + "\n")
log(f"\nSaved: {OUT}")

# ---------- t-SNE visualizations (BERTimbau Large) ----------
section("t-SNE visualizations (bertimbau_large embeddings)")
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Concatenate all splits with split labels
X_all = np.vstack([Xtr, Xde, Xte])
y_all = np.concatenate([ytr, yde, yte])
split_all = np.concatenate([
    np.full(len(Xtr), 0),
    np.full(len(Xde), 1),
    np.full(len(Xte), 2),
])

log(f"  Running TSNE on {X_all.shape[0]} x {X_all.shape[1]} (perplexity=30, "
    f"n_iter=1000, init=pca)... this may take a few minutes")
tsne = TSNE(n_components=3, perplexity=30, max_iter=1000, init='pca',
            random_state=42, verbose=0)
X_tsne = tsne.fit_transform(X_all)

split_names = {0: 'train', 1: 'dev', 2: 'test'}
split_colors = {0: '#2196F3', 1: '#FF9800', 2: '#4CAF50'}
label_colors = {0: '#B0BEC5', 1: '#E53935'}

# tsne_3d_by_label.png
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for lab in [0, 1]:
    m = y_all == lab
    ax.scatter(X_tsne[m, 0], X_tsne[m, 1], X_tsne[m, 2],
               c=label_colors[lab], s=8, alpha=0.6,
               label=f'fraud={lab}' if lab == 1 else 'non-fraud=0')
ax.set_title('t-SNE 3D (BERTimbau Large) — colored by LABEL')
ax.legend()
fig.tight_layout()
p1 = os.path.join(VIS_DIR, "tsne_3d_by_label.png")
fig.savefig(p1, dpi=150)
plt.close()
log(f"  Saved: {p1}")

# tsne_3d_by_split.png
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for sp in [0, 1, 2]:
    m = split_all == sp
    ax.scatter(X_tsne[m, 0], X_tsne[m, 1], X_tsne[m, 2],
               c=split_colors[sp], s=8, alpha=0.6,
               label=split_names[sp])
ax.set_title('t-SNE 3D (BERTimbau Large) — colored by SPLIT')
ax.legend()
fig.tight_layout()
p2 = os.path.join(VIS_DIR, "tsne_3d_by_split.png")
fig.savefig(p2, dpi=150)
plt.close()
log(f"  Saved: {p2}")

# Diagnostic 2D (tsne_bertimbau_large.png) in _logs
tsne2 = TSNE(n_components=2, perplexity=30, max_iter=1000, init='pca',
             random_state=42, verbose=0)
X_tsne2 = tsne2.fit_transform(X_all)
fig, ax = plt.subplots(figsize=(9, 7))
for lab in [0, 1]:
    m = y_all == lab
    ax.scatter(X_tsne2[m, 0], X_tsne2[m, 1], c=label_colors[lab], s=6, alpha=0.6,
               label=f'fraud={lab}' if lab == 1 else 'non-fraud=0')
ax.set_title('t-SNE 2D (BERTimbau Large) — colored by LABEL (diagnostic)')
ax.legend()
fig.tight_layout()
p3 = os.path.join(OUT_DIR, "tsne_bertimbau_large.png")
fig.savefig(p3, dpi=150)
plt.close()
log(f"  Saved: {p3}")

log("\nTASK 5b COMPLETE — audit + visualizations generated")
print(f"\nFull log saved: {OUT}")