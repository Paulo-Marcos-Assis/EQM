#!/usr/bin/env python3
"""
Task 4 — Vectorization (NEW_EQM_Classifier)
Mirrors NEW_training/scripts/task4a_tfidf_fasttext.py (TF-IDF part) + task4b_bert.py,
with EQM-specific changes:
- TF-IDF with CUSTOM tokenizer that keeps monetary/numeric values as single tokens
  (PT monetary regex + plain numbers + words incl. hyphenated compounds).
- BERTimbau base/large: MEAN POOLING of last layer EXCLUDING padding tokens
  (guided by attention_mask) — NOT [CLS] as in NEW_training.
- Anti-leak: TF-IDF fitted ONLY on train; dev/test get transform() only.
  Embeddings from pre-trained frozen models (no fit on data).
- Saves float32 embeddings + int labels per split.
"""

import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(PROJECT_ROOT, "FOR_TRAINING")
TEST_DIR = os.path.join(PROJECT_ROOT, "FOR_TEST")
VEC_DIR = os.path.join(PROJECT_ROOT, "vectorization")

SPARSE_TRAIN = os.path.join(TRAIN_DIR, "Pre_processed_for_Sparse/train_preprocessed.csv")
SPARSE_DEV = os.path.join(TRAIN_DIR, "Pre_processed_for_Sparse/dev_preprocessed.csv")
SPARSE_TEST = os.path.join(TEST_DIR, "Pre_processed_for_Sparse/test_preprocessed.csv")

SPARSE_STEM_TRAIN = os.path.join(TRAIN_DIR, "Pre_processed_for_Sparse_Stem/train_preprocessed_stem.csv")
SPARSE_STEM_DEV = os.path.join(TRAIN_DIR, "Pre_processed_for_Sparse_Stem/dev_preprocessed_stem.csv")
SPARSE_STEM_TEST = os.path.join(TEST_DIR, "Pre_processed_for_Sparse_Stem/test_preprocessed_stem.csv")

EMB_TRAIN = os.path.join(TRAIN_DIR, "Pre_processed_for_Embeddings/train_bert.csv")
EMB_DEV = os.path.join(TRAIN_DIR, "Pre_processed_for_Embeddings/dev_bert.csv")
EMB_TEST = os.path.join(TEST_DIR, "Pre_processed_for_Embeddings/test_bert.csv")

MODELS = {
    "bertimbau_base": ("neuralmind/bert-base-portuguese-cased", 768, 32),
    "bertimbau_large": ("neuralmind/bert-large-portuguese-cased", 1024, 16),
}

# Custom PT token pattern: monetary/numeric values as single token, then plain
# numbers/years, then words (keeps hyphenated compounds intact, never splits).
# Examples: '4,443.36' -> one token; '2025' -> one token; 'quarta-feira' -> one token.
# Used as token_pattern (string) so the pickled vectorizer has NO importable-callable
# dependency — pickle-safe across scripts.
PT_TOKEN_PATTERN = r'\d+(?:[.,]\d+)+|\d+|[^\W\d_][\w-]*'


def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    texts = df['processed_text'].fillna('').values
    labels = df['label'].values.astype(int)
    return texts, labels


def vectorize_tfidf(train_path, dev_path, test_path, out_dir, variant_name):
    print("\n" + "=" * 70)
    print(f"TF-IDF VETORIZAÇÃO ({variant_name})")
    print("=" * 70)

    os.makedirs(out_dir, exist_ok=True)

    train_texts, train_labels = load_data(train_path)
    dev_texts, dev_labels = load_data(dev_path)
    test_texts, test_labels = load_data(test_path)
    print(f"Train: {len(train_texts)} | Dev: {len(dev_texts)} | Test: {len(test_texts)}")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=10000,
        norm='l2',
        lowercase=False,
        token_pattern=PT_TOKEN_PATTERN,
    )

    print("Fitando TF-IDF no TREINO (apenas)...")
    train_matrix = vectorizer.fit_transform(train_texts)
    print(f"Vocabulário: {len(vectorizer.vocabulary_):,} termos")
    print(f"Matriz treino: {train_matrix.shape}")

    print("Transform dev e test...")
    dev_matrix = vectorizer.transform(dev_texts)
    test_matrix = vectorizer.transform(test_texts)
    print(f"Matriz dev: {dev_matrix.shape} | Matriz test: {test_matrix.shape}")

    sparse.save_npz(os.path.join(out_dir, "train_sparse.npz"), train_matrix)
    sparse.save_npz(os.path.join(out_dir, "dev_sparse.npz"), dev_matrix)
    sparse.save_npz(os.path.join(out_dir, "test_sparse.npz"), test_matrix)

    np.save(os.path.join(out_dir, "labels_train.npy"), train_labels)
    np.save(os.path.join(out_dir, "labels_dev.npy"), dev_labels)
    np.save(os.path.join(out_dir, "labels_test.npy"), test_labels)

    with open(os.path.join(out_dir, "vectorizer.pkl"), 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f"Arquivos salvos em: {out_dir}")

    feat = vectorizer.get_feature_names_out()
    monetary = [t for t in feat if re.fullmatch(r'\d+(?:[.,]\d+)+', t)]
    print(f"\nExemplos de features monetárias únicas no vocabulário: {monetary[:10]}")
    return train_matrix.shape, dev_matrix.shape, test_matrix.shape


def mean_pool(last_hidden, attention_mask):
    """Mean pooling over real tokens (exclude padding via attention_mask)."""
    mask = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
    summed = (last_hidden * mask).sum(dim=1)             # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-9)             # (B, 1)
    return summed / counts


def extract_embeddings(texts, tokenizer, model, device, batch_size):
    all_embeddings = []
    model.eval()
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = list(texts[i:i + batch_size])
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt',
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            emb = mean_pool(outputs.last_hidden_state, attention_mask).float().cpu().numpy()

            all_embeddings.append(emb)
            if (i // batch_size) % 10 == 0:
                print(f"    Batch {i // batch_size + 1}/{n_batches}")

    return np.vstack(all_embeddings)


def vectorize_model(model_key, cfg):
    model_name, dim, batch_size = cfg
    print(f"\n{'=' * 70}")
    print(f"VETORIZAÇÃO: {model_key} ({model_name}, dim={dim})")
    print(f"{'=' * 70}")

    out_dir = os.path.join(VEC_DIR, model_key)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"Carregando modelo: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    print(f"Modelo carregado. Parâmetros: {sum(p.numel() for p in model.parameters()):,}")

    for split_name, file_path in [("train", EMB_TRAIN), ("dev", EMB_DEV), ("test", EMB_TEST)]:
        print(f"\n  Processando {split_name}...")
        texts, labels = load_data(file_path)
        print(f"  {split_name}: {len(texts)} textos")

        embeddings = extract_embeddings(texts, tokenizer, model, device, batch_size)
        print(f"  Embeddings shape: {embeddings.shape} dtype={embeddings.dtype}")

        np.save(os.path.join(out_dir, f"{split_name}_embeddings.npy"), embeddings)
        np.save(os.path.join(out_dir, f"labels_{split_name}.npy"), labels)

    del model
    torch.cuda.empty_cache()
    print(f"\n  Arquivos salvos em: {out_dir}")
    return dim


def main():
    print("=" * 70)
    print("TASK 4 — VETORIZAÇÃO (TF-IDF + BERTimbau base/large)")
    print("=" * 70)

    shapes = {}

    tfidf_shapes = vectorize_tfidf(
        SPARSE_TRAIN, SPARSE_DEV, SPARSE_TEST,
        os.path.join(VEC_DIR, "tfidf"), "custom monetary tokenizer"
    )
    shapes['tfidf'] = {'train': tfidf_shapes[0], 'dev': tfidf_shapes[1], 'test': tfidf_shapes[2]}

    tfidf_stem_shapes = vectorize_tfidf(
        SPARSE_STEM_TRAIN, SPARSE_STEM_DEV, SPARSE_STEM_TEST,
        os.path.join(VEC_DIR, "tfidf_stem"), "RSLP stemming variant"
    )
    shapes['tfidf_stem'] = {'train': tfidf_stem_shapes[0], 'dev': tfidf_stem_shapes[1], 'test': tfidf_stem_shapes[2]}

    for model_key, cfg in MODELS.items():
        dim = vectorize_model(model_key, cfg)
        for split in ['train', 'dev', 'test']:
            n = np.load(os.path.join(VEC_DIR, model_key, f"{split}_embeddings.npy")).shape[0]
            shapes[model_key] = shapes.get(model_key, {})
            shapes[model_key][split] = (n, dim)

    # Summary report
    report_path = os.path.join(VEC_DIR, "VECTORIZATION_REPORT.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("VECTORIZATION REPORT — NEW_EQM_Classifier (Task 4)\n")
        f.write("=" * 60 + "\n\n")
        for vec, split_shapes in shapes.items():
            f.write(f"{vec}:\n")
            for split, s in split_shapes.items():
                f.write(f"  {split}: {s[0]} x {s[1]}\n")
        f.write("\nTF-IDF (no stem + RSLP stem) fitted on TRAIN only; dev/test transform() only (no data leak).\n")
        f.write("BERTimbau: mean pooling of last layer excluding padding tokens.\n")

    print(f"\nRelatório salvo: {report_path}")
    print("\n" + "=" * 70)
    print("TASK 4 CONCLUÍDA")
    print("=" * 70)


if __name__ == "__main__":
    main()
