#!/usr/bin/env python3
"""
Task 3 — Dual Preprocessing + IR Normalization (NEW_EQM_Classifier)
Mirrors NEW_training/scripts/task3_preprocess.py with additions:
- Heavy (TF-IDF) branch in TWO variants:
    * No stem: remove HTML, URLs, punctuation; lowercase; remove accents;
      remove PT stopwords (curated list, normalized to accent-free form).
    * With stem: same as above + stemming via nltk.stem.RSLPStemmer.
- Light (BERTimbau) branch: remove URLs only; normalize spaces;
  preserve accents, uppercase, punctuation. NEVER stemming.
- Number/compound policy: numeric sequences (monetary values such as
  `R$ 4.443,36` / `247,675.85`, years, etc.) are PROTECTED before
  punctuation removal so they survive as single tokens; hyphenated
  compounds are NOT split.
- Training content = titulo + ' ' + texto_noticia.
- Applied separately to train, dev and test.
- Vectorization (fit/transform) remains for Task 4 — vectorizer fitted
  ONLY on training (anti data-leak).
"""

import os
import re
import unicodedata

import pandas as pd
from nltk.stem import RSLPStemmer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(PROJECT_ROOT, "FOR_TRAINING")
TEST_DIR = os.path.join(PROJECT_ROOT, "FOR_TEST")

SPARSE_DIR = "Pre_processed_for_Sparse"
SPARSE_STEM_DIR = "Pre_processed_for_Sparse_Stem"
EMB_DIR = "Pre_processed_for_Embeddings"

# --- Curated PT stopword list (source: NEW_training/scripts/task3_preprocess.py) ---
# Normalized to accent-free lowercase so it matches after remove_accents().
_CURATED_STOPWORDS_PT = [
    'a', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as', 'ate',
    'ate', 'com', 'como', 'da', 'das', 'de', 'dela', 'delas', 'dele', 'deles', 'depois',
    'do', 'dos', 'e', 'ela', 'elas', 'ele', 'eles', 'em', 'entre', 'era', 'eram', 'essa',
    'essas', 'esse', 'esses', 'esta', 'estamos', 'estao', 'estas', 'estava', 'estavam',
    'este', 'esteja', 'estejam', 'estes', 'esteve', 'estive', 'estivemos', 'estiver',
    'estiveram', 'estivesse', 'estivessem', 'estou', 'eu', 'foi', 'fomos', 'for', 'fora',
    'foram', 'fosse', 'fossem', 'fui', 'ha', 'isso', 'isto', 'ja', 'la', 'lhe', 'lhes',
    'lo', 'mais', 'mas', 'me', 'mesmo', 'meu', 'meus', 'minha', 'minhas', 'muito', 'na',
    'nao', 'nas', 'nem', 'no', 'nos', 'nossa', 'nossas', 'nosso', 'nossos', 'num', 'numa',
    'o', 'os', 'ou', 'para', 'pela', 'pelas', 'pelo', 'pelos', 'por', 'qual', 'quando',
    'que', 'quem', 'sao', 'se', 'seja', 'sejam', 'sem', 'sera', 'serao', 'seu', 'seus',
    'so', 'sua', 'suas', 'tambem', 'te', 'tem', 'temos', 'tenho', 'tera', 'terao', 'teu',
    'teus', 'tinha', 'tinham', 'tive', 'tivemos', 'tu', 'tua', 'tuas', 'um', 'uma', 'umas',
    'uns', 'voce', 'voces', 'vos', 'a', 'as',
]


def _accent_free(text):
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


STOPWORDS_PT = {_accent_free(w).lower() for w in _CURATED_STOPWORDS_PT if _accent_free(w).lower()}

_NUMBER_RE = re.compile(r'\d+(?:[.,]\d+)+|\d+')
_TOKEN_NUM_RE = re.compile(r'^[\d.,]+$')


def remove_html(text):
    return re.sub(r'<[^>]+>', ' ', text)


def remove_urls(text):
    return re.sub(r'http[s]?://\S+|www\.\S+', ' ', text)


def protect_numbers(text, registry):
    """Replace numeric sequences (monetary values, years) with placeholders
    so punctuation removal does not split them. Restored later as single tokens."""
    def _repl(m):
        registry.append(m.group(0))
        return f' _NUM{len(registry) - 1}_ '
    return _NUMBER_RE.sub(_repl, text)


def remove_punctuation_keep_hyphens(text):
    """Remove punctuation but keep hyphens inside compounds (not split)."""
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\s-+|-+\s', ' ', text)
    return text


def normalize_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()


def heavy_preprocess(text, stem=False, stemmer=None):
    if pd.isna(text):
        text = ''
    registry = []
    text = remove_html(text)
    text = remove_urls(text)
    text = text.lower()
    text = protect_numbers(text, registry)
    text = _accent_free(text)
    text = remove_punctuation_keep_hyphens(text)
    text = normalize_spaces(text)

    tokens = [t for t in text.split() if t not in STOPWORDS_PT]

    restored = []
    for t in tokens:
        m = re.fullmatch(r'_NUM(\d+)_', t)
        if m:
            restored.append(registry[int(m.group(1))])
        else:
            restored.append(t)

    if stem and stemmer is not None:
        restored = [
            t if _TOKEN_NUM_RE.fullmatch(t) else stemmer.stem(t)
            for t in restored
        ]

    return ' '.join(restored)


def light_preprocess(text):
    if pd.isna(text):
        return ''
    text = remove_urls(text)
    text = normalize_spaces(text)
    return text


def process_file(input_path, output_path, preprocess_fn, label, stem_label=''):
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  [{label}{stem_label}] Loaded: {len(df):,} rows from {input_path}")

    df['processed_text'] = (
        df['titulo'].fillna('') + ' ' + df['texto_noticia'].fillna('')
    ).apply(preprocess_fn)

    cols_keep = ['link_noticia', 'portal', 'label', 'processed_text']
    for c in cols_keep:
        if c not in df.columns:
            df[c] = ''
    df = df[cols_keep]

    df.to_csv(output_path, index=False, encoding='utf-8')

    n_pos = int((df['label'] == 1).sum())
    n_neg = int((df['label'] == 0).sum())
    empty = int((df['processed_text'].str.strip() == '').sum())
    print(f"  [{label}{stem_label}] Saved: {output_path} ({len(df):,} rows)")
    print(f"  [{label}{stem_label}] pos={n_pos} neg={n_neg} empty_processed={empty}")
    return {'rows': len(df), 'pos': n_pos, 'neg': n_neg, 'empty': empty}


def main():
    print("=" * 70)
    print("TASK 3 — DUAL PREPROCESSING + IR NORMALIZATION")
    print("=" * 70)

    for split_dir in (TRAIN_DIR, TEST_DIR):
        for sub in (SPARSE_DIR, SPARSE_STEM_DIR, EMB_DIR):
            os.makedirs(os.path.join(split_dir, sub), exist_ok=True)

    stemmer = RSLPStemmer()

    report = {}

    print("\n--- HEAVY — NO STEM (TF-IDF) ---")
    for name in ['train', 'dev', 'test']:
        src = os.path.join(TRAIN_DIR if name != 'test' else TEST_DIR, f"{name}.csv")
        out = os.path.join(TRAIN_DIR, SPARSE_DIR, f"{name}_preprocessed.csv")
        report[f'heavy_{name}'] = process_file(
            src, out, lambda t, s=stemmer: heavy_preprocess(t, stem=False, stemmer=s), name.upper()
        )

    print("\n--- HEAVY — WITH STEM (RSLP, TF-IDF ablation) ---")
    for name in ['train', 'dev', 'test']:
        src = os.path.join(TRAIN_DIR if name != 'test' else TEST_DIR, f"{name}.csv")
        out = os.path.join(TRAIN_DIR, SPARSE_STEM_DIR, f"{name}_preprocessed_stem.csv")
        report[f'stem_{name}'] = process_file(
            src, out, lambda t, s=stemmer: heavy_preprocess(t, stem=True, stemmer=s), name.upper(), " [STEM]"
        )

    print("\n--- LIGHT (BERTimbau) ---")
    for name in ['train', 'dev', 'test']:
        src = os.path.join(TRAIN_DIR if name != 'test' else TEST_DIR, f"{name}.csv")
        out = os.path.join(TRAIN_DIR, EMB_DIR, f"{name}_bert.csv")
        report[f'light_{name}'] = process_file(src, out, light_preprocess, name.upper())

    print("\n--- MIRROR TEST TO FOR_TEST ---")
    for sub, fname in [
        (SPARSE_DIR, "test_preprocessed.csv"),
        (SPARSE_STEM_DIR, "test_preprocessed_stem.csv"),
        (EMB_DIR, "test_bert.csv"),
    ]:
        src = os.path.join(TRAIN_DIR, sub, fname)
        out = os.path.join(TEST_DIR, sub, fname)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        pd.read_csv(src, low_memory=False).to_csv(out, index=False, encoding='utf-8')
        print(f"  Mirrored: {out}")

    # --- Verification report ---
    report_path = os.path.join(TRAIN_DIR, "PREPROCESS_REPORT.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("PREPROCESS REPORT — NEW_EQM_Classifier (Task 3)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Branches:\n")
        f.write("  Heavy no-stem  : Pre_processed_for_Sparse/*_preprocessed.csv\n")
        f.write("  Heavy RSLP-stem: Pre_processed_for_Sparse_Stem/*_preprocessed_stem.csv\n")
        f.write("  Light (BERTimbau): Pre_processed_for_Embeddings/*_bert.csv\n\n")
        for key, r in report.items():
            pct = r['pos'] / r['rows'] * 100 if r['rows'] else 0
            f.write(f"{key:14s}: rows={r['rows']:>6,} pos={r['pos']:>4} ({pct:.2f}%) "
                    f"neg={r['neg']:>6,} empty_processed={r['empty']}\n")
        f.write("\nPolicy notes:\n")
        f.write("  - Numbers/monetary values protected as single tokens (Task 4 tokenizer)\n")
        f.write("  - Hyphenated compounds NOT split\n")
        f.write("  - Stopwords: curated list (source NEW_training), accent-normalized\n")
        f.write("  - Lemmatization NOT implemented (documented gap); RSLP = stemming only\n")

    print("\n" + "=" * 70)
    print("TASK 3 COMPLETED")
    print("=" * 70)
    print("\nNOTE: The TF-IDF vectorizer will be fitted ONLY on training in Task 4.")
    print("      Dev and test receive only transform() — no data leakage.")


if __name__ == "__main__":
    main()
