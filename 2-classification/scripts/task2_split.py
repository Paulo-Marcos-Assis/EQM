#!/usr/bin/env python3
"""
Task 2 — Stratified Split (NEW_EQM_Classifier) — re-run on deduped dataset.
Mirrors NDMAIS_BIAS/scripts/task2_split.py with methodological corrections:
- Input: FOR_TRAINING/CONSOLIDATED_EQM_DEDUPED.csv (post rigorous 1a-1d)
- Preserves auxiliary columns (portal, texto_completo, texto_len, is_text_duplicate, original_index)
- Excludes empty links from dedup/overlap sets
- Text dedup cross-split hash-based (only when non-dup copy exists in train)
- Embedding cross-split dedup: BERTimbau Base, cos >= 0.99 (leakage audit)
"""

import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(REPO_ROOT, "data", "classification", "CONSOLIDATED_EQM_DEDUPED.csv")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "FOR_TRAINING")
TEST_DIR = os.path.join(PROJECT_ROOT, "FOR_TEST")
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

SEED = 42
EMB_THRESHOLD = 0.99
MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
URL_RE = re.compile(r"https?://[^\s]+")


def normalize_text_for_emb(s):
    s = str(s)
    s = URL_RE.sub(" ", s)
    return " ".join(s.split())


def embed_batch(texts, tokenizer, model, device, batch_size=32):
    embs = []
    with torch.no_grad():
        for b_start in range(0, len(texts), batch_size):
            batch = texts[b_start:b_start + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=512,
                            return_tensors='pt').to(device)
            out = model(**enc)
            mask = enc['attention_mask'].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            summed = torch.sum(out.last_hidden_state * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            embs.append((summed / counts).cpu().numpy().astype(np.float32))
    return np.vstack(embs)


print(f"[Task 2] Loading consolidated deduped dataset: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)
print(f"[Task 2] Initial shape: {df.shape}")
print(f"[Task 2] Label dist: {df['label'].value_counts().to_dict()}")

df['original_index'] = df.index

# Step 1: First split — traindev / test (80/20 stratified by label)
print("\n[Step 1] Split 1: traindev / test (80/20)")
traindev, test = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=SEED
)
print(f"  traindev: {len(traindev)} rows ({traindev['label'].value_counts().to_dict()})")
print(f"  test: {len(test)} rows ({test['label'].value_counts().to_dict()})")

# Step 2: Second split — train / dev from traindev (80/20)
print("\n[Step 2] Split 2: train / dev from traindev (80/20)")
train, dev = train_test_split(
    traindev, test_size=0.2, stratify=traindev['label'], random_state=SEED
)
print(f"  train: {len(train)} rows ({train['label'].value_counts().to_dict()})")
print(f"  dev: {len(dev)} rows ({dev['label'].value_counts().to_dict()})")

# Step 3: Link dedup between splits (exclude empty links)
print("\n[Step 3] Link dedup between splits")
train_links = set(train['link_noticia'].unique()) - {''}
dev_links = set(dev['link_noticia'].unique()) - {''}
test_links = set(test['link_noticia'].unique()) - {''}

n_dev_before = len(dev)
dev = dev[~((dev['link_noticia'] != '') & dev['link_noticia'].isin(train_links))]
link_dedup_dev = n_dev_before - len(dev)
print(f"  dev: {n_dev_before} → {len(dev)} (removed {link_dedup_dev} link duplicates)")

n_test_before = len(test)
test = test[~((test['link_noticia'] != '') & test['link_noticia'].isin(train_links))]
test = test[~((test['link_noticia'] != '') & test['link_noticia'].isin(dev_links))]
link_dedup_test = n_test_before - len(test)
print(f"  test: {n_test_before} → {len(test)} (removed {link_dedup_test} link duplicates)")

# Step 4: Text similarity dedup (cross-split, hash-based, only when non-dup copy in train)
print("\n[Step 4] Text similarity dedup")
train['text_hash'] = train['texto_completo'].str[:200]
dev['text_hash'] = dev['texto_completo'].str[:200]
test['text_hash'] = test['texto_completo'].str[:200]

dup_indices = set(df[df['is_text_duplicate'] == True].index)
print(f"  Total text duplicates in consolidated: {len(dup_indices)}")

text_dedup_dev = 0
text_dedup_test = 0
for split_name, split_df in [('dev', dev), ('test', test)]:
    dups_in_split = split_df[split_df['is_text_duplicate'] == True]
    remove_indices = []
    for idx in dups_in_split.index:
        text_hash = split_df.loc[idx, 'text_hash']
        same_hash_in_train = train[(train['text_hash'] == text_hash) & (train['is_text_duplicate'] == False)]
        if len(same_hash_in_train) > 0:
            remove_indices.append(idx)
    if split_name == 'dev':
        n_before = len(dev)
        dev = dev.drop(remove_indices)
        text_dedup_dev = n_before - len(dev)
        print(f"  dev: {n_before} → {len(dev)} (removed {text_dedup_dev} cross-split text duplicates)")
    else:
        n_before = len(test)
        test = test.drop(remove_indices)
        text_dedup_test = n_before - len(test)
        print(f"  test: {n_before} → {len(test)} (removed {text_dedup_test} cross-split text duplicates)")

# Step 4b: Embedding-based cross-split leakage detection (BERT Base, cos >= 0.99)
print(f"\n[Step 4b] Embedding-based cross-split leakage detection ({MODEL_NAME}, cos >= {EMB_THRESHOLD})")

train_texts = train['texto_noticia'].apply(normalize_text_for_emb).tolist()
dev_texts = dev['texto_noticia'].apply(normalize_text_for_emb).tolist()
test_texts = test['texto_noticia'].apply(normalize_text_for_emb).tolist()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Loading {MODEL_NAME} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

print("  Embedding train...")
emb_train = embed_batch(train_texts, tokenizer, model, device)
print("  Embedding dev...")
emb_dev = embed_batch(dev_texts, tokenizer, model, device)
print("  Embedding test...")
emb_test = embed_batch(test_texts, tokenizer, model, device)

emb_train = normalize(emb_train, norm='l2')
emb_dev = normalize(emb_dev, norm='l2')
emb_test = normalize(emb_test, norm='l2')


def find_duplicates_cross(emb_src, emb_tgt, threshold=EMB_THRESHOLD):
    sim = emb_tgt @ emb_src.T
    dups = np.where(sim >= threshold)
    return set(map(tuple, zip(*dups)))


dev_train_dups = find_duplicates_cross(emb_train, emb_dev, EMB_THRESHOLD)
dev_remove_idx = set([i for i, _ in dev_train_dups])
n_before = len(dev)
dev = dev.drop(index=[dev.index[i] for i in dev_remove_idx])
emb_dedup_dev = n_before - len(dev)
print(f"  dev: {n_before} → {len(dev)} (removed {emb_dedup_dev} embedding duplicates vs train)")

test_train_dups = find_duplicates_cross(emb_train, emb_test, EMB_THRESHOLD)
test_dev_dups = find_duplicates_cross(emb_dev, emb_test, EMB_THRESHOLD)
test_remove_idx = set([i for i, _ in test_train_dups]) | set([i for i, _ in test_dev_dups])
n_before = len(test)
test = test.drop(index=[test.index[i] for i in test_remove_idx])
emb_dedup_test = n_before - len(test)
print(f"  test: {n_before} → {len(test)} (removed {emb_dedup_test} embedding duplicates vs train/dev)")

# Drop temporary hash column (keep original_index for Task 8 traceability)
train = train.drop(columns=['text_hash'], errors='ignore')
dev = dev.drop(columns=['text_hash'], errors='ignore')
test = test.drop(columns=['text_hash'], errors='ignore')

# Step 5: Save splits
print("\n[Step 5] Saving splits")
train.to_csv(os.path.join(TRAIN_DIR, "train.csv"), index=False, encoding='utf-8')
dev.to_csv(os.path.join(TRAIN_DIR, "dev.csv"), index=False, encoding='utf-8')
test.to_csv(os.path.join(TEST_DIR, "test.csv"), index=False, encoding='utf-8')
print(f"  Saved: {os.path.join(TRAIN_DIR, 'train.csv')}")
print(f"  Saved: {os.path.join(TRAIN_DIR, 'dev.csv')}")
print(f"  Saved: {os.path.join(TEST_DIR, 'test.csv')}")

# Step 6: Generate SPLIT_REPORT.txt
print("\n[Step 6] Generating SPLIT_REPORT.txt")
report_path = os.path.join(TRAIN_DIR, "SPLIT_REPORT.txt")


def split_stats(name, split_df):
    total = len(split_df)
    pos = int((split_df['label'] == 1).sum())
    neg = total - pos
    pos_pct = pos / total * 100 if total > 0 else 0
    return f"{name}: {total} rows | pos={pos} ({pos_pct:.2f}%) | neg={neg}"


with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("SPLIT REPORT — NEW_EQM_Classifier (Task 2, post-1a-1d re-run)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Input: CONSOLIDATED_EQM_DEDUPED.csv ({len(df)} rows, post rigorous 1a-1d)\n")
    f.write(f"Random seed: {SEED}\n")
    f.write(f"Split ratio: 80/20 (traindev/test) → 80/20 (train/dev)\n\n")
    f.write("--- DEDUP REMOVALS (cross-split) ---\n")
    f.write(f"Link dedup: dev={link_dedup_dev}, test={link_dedup_test}\n")
    f.write(f"Text dedup (hash, non-dup-in-train only): dev={text_dedup_dev}, test={text_dedup_test}\n")
    f.write(f"Embedding dedup (BERT Base cos>={EMB_THRESHOLD}): dev={emb_dedup_dev}, test={emb_dedup_test}\n\n")
    f.write("--- FINAL SPLIT DISTRIBUTION ---\n")
    f.write(split_stats("Train (64%)", train) + "\n")
    f.write(split_stats("Dev (16%)", dev) + "\n")
    f.write(split_stats("Test (20%)", test) + "\n")
    f.write(f"\nTotal after splits: {len(train) + len(dev) + len(test)} rows\n")
    f.write(f"Removed by dedup: {len(df) - len(train) - len(dev) - len(test)} rows\n")

    train_links_final = set(train['link_noticia'].unique()) - {''}
    dev_links_final = set(dev['link_noticia'].unique()) - {''}
    test_links_final = set(test['link_noticia'].unique()) - {''}

    train_dev_overlap = len(train_links_final & dev_links_final)
    train_test_overlap = len(train_links_final & test_links_final)
    dev_test_overlap = len(dev_links_final & test_links_final)

    f.write(f"\n--- LINK OVERLAP (zero leakage expected) ---\n")
    f.write(f"Train ∩ Dev: {train_dev_overlap} links\n")
    f.write(f"Train ∩ Test: {train_test_overlap} links\n")
    f.write(f"Dev ∩ Test: {dev_test_overlap} links\n")

    f.write(f"\n--- PORTAL DISTRIBUTION (top 5 per split) ---\n")
    for split_name, split_df in [("Train", train), ("Dev", dev), ("Test", test)]:
        f.write(f"\n{split_name}:\n")
        if 'portal' in split_df.columns:
            portal_table = split_df.groupby('portal')['label'].agg(['count']).sort_values('count', ascending=False).head(5)
            f.write(portal_table.to_string() + "\n")
        else:
            f.write("(portal column not present)\n")

    f.write(f"\n--- METHODOLOGY NOTE ---\n")
    f.write(f"Tripartite holdout (train/dev/test) + internal CV is the ML standard.\n")
    f.write(f"Raschka, 'Model Selection vs Assessment':\n")
    f.write(f"  - Train (64%): fitting + HPO\n")
    f.write(f"  - Dev (16%): selection validation (compare 12 combos + threshold)\n")
    f.write(f"  - Test (20%): isolated, used only once in Task 7\n")
    f.write(f"Auxiliary columns preserved: portal, texto_completo, texto_len, is_text_duplicate, original_index\n")

print(f"[Task 2] SPLIT_REPORT.txt saved: {report_path}")
print(f"\n[Task 2] Task 2 completed successfully!")
print(f"  Train: {len(train)} rows | Dev: {len(dev)} rows | Test: {len(test)} rows")