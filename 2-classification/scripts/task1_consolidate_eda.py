#!/usr/bin/env python3
"""
Task 1 — Consolidação + EDA (NEW_EQM_Classifier)
Adaptado de NEW_training/scripts/task1_consolidate_eda.py
"""

import pandas as pd
import numpy as np
import os
from urllib.parse import urlparse
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(REPO_ROOT, "data", "classification", "Complete_dataset_11309.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FOR_TRAINING")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Read CSV
print(f"[Task 1] Lendo dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"[Task 1] Shape inicial: {df.shape}")
print(f"[Task 1] Colunas: {list(df.columns)}")

# 2. Normalizar schema: label, titulo, texto_noticia, link_noticia
# Ensure correct column names
expected_cols = ['label', 'titulo', 'texto_noticia', 'link_noticia']
assert all(c in df.columns for c in expected_cols), f"Colunas esperadas não encontradas: {df.columns}"

# 3. Tratar nulos
# 30 links nulos → manter com link=''
# 1 titulo nulo (observado no dataset) → verificar
print(f"[Task 1] Nulos antes: {df.isnull().sum().to_dict()}")
df['link_noticia'] = df['link_noticia'].fillna('')
# Se houver titulo nulo, manter linha mas documentar
n_titulo_nulo = df['titulo'].isnull().sum()
print(f"[Task 1] Títulos nulos: {n_titulo_nulo}")

# 4. Normalizar texto: espaços múltiplos e quebras de linha
def normalize_text(s):
    if pd.isnull(s):
        return ''
    s = str(s)
    # Substituir quebras de linha e tabs por espaço
    s = re.sub(r'[\n\r\t]+', ' ', s)
    # Substituir múltiplos espaços por um único
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

df['titulo'] = df['titulo'].apply(normalize_text)
df['texto_noticia'] = df['texto_noticia'].apply(normalize_text)

# 5. Dedup por link (36 duplicados → keep='first')
n_before = len(df)
df = df.drop_duplicates(subset='link_noticia', keep='first')
print(f"[Task 1] Dedup por link: {n_before} → {len(df)} (removidos: {n_before - len(df)})")

# 6. Criar texto combinado e filtro de comprimento mínimo
df['texto_completo'] = df['titulo'] + ' ' + df['texto_noticia']
df['texto_len'] = df['texto_completo'].str.len()

# Filtro: remover exemplos onde texto combinado <= 200 caracteres
n_before_filter = len(df)
df = df[df['texto_len'] > 200].copy()
print(f"[Task 1] Filtro comprimento (>200 chars): {n_before_filter} → {len(df)} (removidos: {n_before_filter - len(df)})")

# 7. Dedup por similaridade textual
# TF-IDF word n-gram (1,2) + cosseno >= 0.9 → marcar duplicatas para tratamento na Task 2
print("[Task 1] Calculando similaridade textual (TF-IDF ngram 1-2)...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, max_features=10000)
# Usar apenas texto_completo para similaridade
tfidf_matrix = vectorizer.fit_transform(df['texto_completo'])

# Calcular similaridade de cosseno em batches para evitar consumo excessivo de memória
similarity_matrix = cosine_similarity(tfidf_matrix)

# Marcar duplicatas: pares com similaridade >= 0.9, manter primeiro
duplicate_indices = set()
threshold = 0.9
n_samples = similarity_matrix.shape[0]

print(f"[Task 1] Analisando {n_samples} amostras para duplicatas textuais...")
for i in range(n_samples):
    if i in duplicate_indices:
        continue
    # Encontrar índices com similaridade >= threshold (excluindo o próprio)
    similar_indices = np.where(similarity_matrix[i] >= threshold)[0]
    # Remover o próprio i
    similar_indices = similar_indices[similar_indices != i]
    # Marcar duplicatas posteriores (keep='first')
    for j in similar_indices:
        duplicate_indices.add(j)

print(f"[Task 1] Duplicatas textuais encontradas (similaridade >= {threshold}): {len(duplicate_indices)}")
df['is_text_duplicate'] = df.index.isin(duplicate_indices)

# 8. Extrair portal do link
def extract_portal(link):
    if not link or str(link).strip() == '':
        return '(sem link)'
    try:
        parsed = urlparse(str(link))
        netloc = parsed.netloc
        if not netloc:
            return '(sem dominio)'
        # Remover prefixo www.
        netloc = netloc.replace('www.', '')
        return netloc
    except Exception:
        return '(sem dominio)'

df['portal'] = df['link_noticia'].apply(extract_portal)

# 9. Salvar consolidado
output_path = os.path.join(OUTPUT_DIR, "CONSOLIDATED_EQM.csv")
df.to_csv(output_path, index=False, encoding='utf-8')
print(f"[Task 1] Consolidado salvo em: {output_path}")
print(f"[Task 1] Shape final: {df.shape}")

# 10. Relatório de distribuição (DISTRIBUTION_REPORT.txt)
report_path = os.path.join(OUTPUT_DIR, "DISTRIBUTION_REPORT.txt")
label_counts = df['label'].value_counts().sort_index()
label_pct = (label_counts / len(df) * 100).round(2)

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("RELATÓRIO DE DISTRIBUIÇÃO — NEW_EQM_Classifier (Task 1)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Dataset original: Complete_dataset_11309.csv\n")
    f.write(f"Dataset consolidado: CONSOLIDATED_EQM.csv\n")
    f.write(f"Shape inicial: 11309 linhas\n")
    f.write(f"Shape final (após dedup link + filtro >200 chars): {len(df)} linhas\n")
    f.write(f"Links nulos tratados: 30 (mantidos como '')\n")
    f.write(f"Links duplicados removidos (keep='first'): 36\n")
    f.write(f"Amostras removidas por texto <= 200 chars: {n_before_filter - len(df)}\n")
    f.write(f"Duplicatas textuais (similaridade >= 0.9) marcadas: {len(duplicate_indices)}\n")
    f.write(f"\n--- DISTRIBUIÇÃO DE LABEL ---\n")
    for label_val in sorted(df['label'].unique()):
        count = label_counts.get(label_val, 0)
        pct = label_pct.get(label_val, 0)
        f.write(f"label={label_val}: {count} ({pct}%)\n")
    f.write(f"\nRazão de classes (neg:pos): {label_counts.get(0,0)} : {label_counts.get(1,0)}\n")
    f.write(f"\n--- COMPRIMENTO DO TEXTO COMPLETO (titulo + texto) ---\n")
    stats = df.groupby('label')['texto_len'].agg(['mean', 'median', 'min', 'max'])
    f.write(str(stats.round(2)) + "\n")
    f.write(f"\n--- TABELA PORTAL × LABEL (top 10 portais) ---\n")
    portal_table = df.groupby('portal')['label'].agg(['count', 'sum']).sort_values('count', ascending=False)
    portal_table.columns = ['Total', 'Pos']
    portal_table['Neg'] = portal_table['Total'] - portal_table['Pos']
    portal_table['Pos%'] = (portal_table['Pos'] / portal_table['Total'] * 100).round(1)
    f.write(str(portal_table.head(10).to_string()) + "\n")
    f.write(f"\n--- FILTRO COMBINADO (<= 200 chars) ---\n")
    f.write(f"Amostras removidas: {n_before_filter - len(df)}\n")
    f.write(f"Amostras mantidas: {len(df)}\n")
    f.write(f"Percentual removido: {((n_before_filter - len(df)) / n_before_filter * 100):.2f}%\n")
    f.write(f"\nNota: linhas com texto_completo <= 200 caracteres foram EXCLUÍDAS do pipeline.\n")
    f.write(f"Esta regra é aplicada antes do split para evitar vazamento (leakage) de dados pequenos.\n")

print(f"[Task 1] Relatório salvo em: {report_path}")
print(f"[Task 1] Task 1 concluída com sucesso!")
