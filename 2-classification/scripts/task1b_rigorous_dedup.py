#!/usr/bin/env python3
"""
Task 1b — Rigorous pre-split global dedup (unsupervised, label-free) on
FOR_TRAINING/CONSOLIDATED_EQM.csv. Mirrors NDMAIS_BIAS task1c_final_dedup_global.py.

Sequential (label-free, keep='first'):
  1a. URL (link_noticia)          — exact
  1b. Title (titulo)              — normalized (lowercase, collapse spaces)
  1c. Text (texto_noticia)        — normalized (exact)
  1d. TF-IDF (ngram 1-2)          — cosine >= 0.90 near-dup

NOT applied globally: embedding dedup (1e) — only cross-split in Task 2 (cos>=0.99).
See documentation/DEDUP_METHOD_NOTE.md.

Output: FOR_TRAINING/CONSOLIDATED_EQM_DEDUPED.csv + FOR_TRAINING/DEDUP_FINAL_REPORT.txt
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "FOR_TRAINING/CONSOLIDATED_EQM.csv")
OUT_CSV = os.path.join(BASE, "FOR_TRAINING/CONSOLIDATED_EQM_DEDUPED.csv")
OUT_REPORT = os.path.join(BASE, "FOR_TRAINING/DEDUP_FINAL_REPORT.txt")

URL_RE = re.compile(r"https?://[^\s]+")


def normalize_text(s):
    s = str(s)
    s = URL_RE.sub(" ", s)
    return " ".join(s.split()).strip()


def normalize_title(s):
    return normalize_text(s).lower()


def main():
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(str(msg))

    df = pd.read_csv(SRC)
    log(f"=== SOURCE === {len(df)} rows | pos={(df['label']==1).sum()} | neg={(df['label']==0).sum()}")

    # 1a URL exact
    before = len(df)
    df['link_noticia'] = df['link_noticia'].fillna('').astype(str).str.strip()
    df = df.drop_duplicates(subset=['link_noticia'], keep='first')
    log(f"1a URL exact: removed {before - len(df)} -> {len(df)}")

    # 1b Title exact (normalized)
    before = len(df)
    df['titulo_norm'] = df['titulo'].apply(normalize_title)
    df = df.drop_duplicates(subset=['titulo_norm'], keep='first').drop(columns=['titulo_norm'])
    log(f"1b Title exact (norm): removed {before - len(df)} -> {len(df)}")

    # 1c texto_noticia exact (normalized)
    before = len(df)
    df['texto_noticia_norm'] = df['texto_noticia'].apply(normalize_text)
    df = df.drop_duplicates(subset=['texto_noticia_norm'], keep='first').drop(columns=['texto_noticia_norm'])
    log(f"1c texto_noticia exact (norm): removed {before - len(df)} -> {len(df)}")

    # 1d TF-IDF near-dup cos>=0.90
    before = len(df)
    texts = df['texto_noticia'].apply(normalize_text).tolist()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, max_features=10000,
                          lowercase=False, norm='l2', token_pattern=r'\b\w+\b')
    Xt = vec.fit_transform(texts)
    n = Xt.shape[0]
    remove_set = set()
    for q_start in range(0, n, 512):
        q_end = min(q_start + 512, n)
        sim_block = cosine_similarity(Xt[q_start:q_end], Xt)
        for r in range(q_start, q_end):
            sim_row = sim_block[r - q_start]
            dups = np.where(sim_row >= 0.90)[0]
            for j in dups:
                if j > r and j not in remove_set:
                    remove_set.add(j)
    df_1d = df.iloc[[i for i in range(n) if i not in remove_set]].reset_index(drop=True)
    log(f"1d TF-IDF near-dup (cos>=0.90): removed {len(remove_set)} -> {len(df_1d)}")
    df = df_1d

    # Class-ratio check per step already implicit; final ratio:
    final_df = df.copy()
    final_df.to_csv(OUT_CSV, index=False)
    pos = int((final_df['label'] == 1).sum())
    neg = int((final_df['label'] == 0).sum())
    log(f"=== FINAL CLEAN (1a-1d) === {len(final_df)} rows | pos={pos} | neg={neg} | ratio={neg/pos:.2f}")
    log(f"Saved: {OUT_CSV}")

    with open(OUT_REPORT, "w", encoding="utf-8") as fp:
        fp.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()