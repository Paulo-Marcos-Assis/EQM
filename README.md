# EQM — Extração, Questionamento e Monitoramento

Pipeline completo para **detecção de fraudes em licitações públicas** a partir de
notícias jornalísticas de Santa Catarina: coleta (crawler), classificação binária,
extração de informações e vinculação (linkage) com processos licitatórios oficiais.

Este repositório reúne os conjuntos de dados, prompts, configs e scripts necessários
para reproduzir todos os experimentos da dissertação e dos artigos relacionados,
promovendo a reprodutibilidade e o reuso pela comunidade científica.

## Citações

> Disponibilizar publicamente, em repositório no GitHub (github.com/Paulo-Marcos-Assis/EQM),
> os conjuntos de dados construídos neste trabalho para classificação e extração de
> informações, bem como os prompts, configs e scripts necessários para a regeneração
> do benchmark de vinculação, promovendo a reprodutibilidade e o reuso pela
> comunidade científica. — *Objetivo específico 5 da dissertação.*

Artigos:
- **SBBD 2026** — *Information Extraction from Brazilian News Articles on Public
  Procurement Fraud* → `3-extraction/` e
  [comparative-information-extraction](https://github.com/Paulo-Marcos-Assis/comparative-information-extraction).
- **ENIAC 2026** — *A Hybrid RAG Pipeline for Linking Fraud News to Public Procurement
  Records* → `4-linkage/` e
  [rag-linkage-paper](https://github.com/Paulo-Marcos-Assis/rag-linkage-paper).

## Estrutura

```
EQM/
├── data/                 # DATASETS (classificação e extração)
│   ├── classification/   # 11.309 notícias rotuladas + splits train/dev/test
│   └── extraction/       # gold standard de 796 notícias anotadas
├── 1-crawler/            # coleta de notícias (scraping de 11+ portais de SC)
├── 2-classification/     # TF-IDF + LinearSVC (vencedor) — F1 0,956 no teste
├── 3-extraction/         # IE: heurística, RAG e LLM (município, modalidade, edital, objeto)
├── 4-linkage/            # benchmark de vinculação RAG (TF-IDF + rerank LLM)
└── .env.example          # variáveis de ambiente do crawler
```

## Pipeline (4 fases)

1. **Coleta** (`1-crawler/`) — `main.py` + `crawler_configs.json`, 11+ portais.
2. **Classificação** (`2-classification/`) — vencedor: TF-IDF + LinearSVC
   (dev F1 0,9764; test F1 0,9563; robusto cross-portal).
3. **Extração** (`3-extraction/`) — 3 abordagens; melhor: LLM estruturado
   (acurácia média 91,45%; F1 89,46%).
4. **Vinculação** (`4-linkage/`) — híbrido RAG: extração LLM + pré-filtro SQL +
   recuperação TF-IDF + rerank LLM (single-call `gpt-oss:20b` 76,8% dev / 71,5% test).

## Datasets

- **Classificação**: `data/classification/Complete_dataset_11309.csv` (11.309 notícias
  rotuladas, 968 positivas) + `CONSOLIDATED_EQM_DEDUPED.csv` (10.379) + splits.
- **Extração**: `data/extraction/Gabarito.csv` (796 notícias anotadas; 600 dev + 196 test).
- **Vinculação**: o benchmark sintético (714 pares) **não é publicado** por ética;
  regeneração determinística via prompts/configs/scripts de `4-linkage/` (seed 42).

## Reprodução

Cada etapa tem seu próprio README com instruções detalhadas. Visão geral:

```bash
# Classificação (vencedor TF-IDF + LinearSVC)
cd 2-classification && pip install -r requirements.txt
python scripts/task3_preprocess.py
python scripts/task4_vectorize.py
python scripts/task5_train.py
python scripts/task7_final_test.py

# Linkage (regeneração do benchmark de vinculação)
cd 4-linkage && pip install -r requirements.txt && make bench-both && make dev && make test
```

Hardware usado: AMD EPYC 9654 (96 núcleos, 1,5 TB RAM) + NVIDIA RTX A6000 (48 GB VRAM,
CUDA 12.8); modelos via Ollama (`gpt-oss:20b`, `qwen2.5:7b`).

## Licença

Código sob MIT (ver `LICENSE`). Dados para fins de pesquisa (ver `data/README.md`).