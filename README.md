# EQM — Exame de Qualificação de Mestrado

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
- **ERAMIA-RS 2025** (publicado) — ASSIS, Paulo Marcos de; CASTRO, Márcio; CARVALHO,
  Jônata Tyska. *Machine Learning-Based Classification of Portuguese News Articles on
  Public Procurement Fraud.* In: Anais da I Escola Regional de Aprendizado de Máquina e
  Inteligência Artificial da Região Sul (ERAMIA-RS 2025). Porto Alegre: Sociedade
  Brasileira de Computação, 2025. p. 88–91. DOI: [10.5753/eramiars.2025.16668](https://doi.org/10.5753/eramiars.2025.16668).
  → estágio de classificação deste repositório (`2-classification/`, `data/classification/`).
- **SBBD 2026** (aceito para publicação) — ASSIS, Paulo Marcos de; CASTRO, Márcio;
  CARVALHO, Jônata Tyska. *Information Extraction from Brazilian News Articles on Public
  Procurement Fraud.* In: Anais do 41º Simpósio Brasileiro de Bancos de Dados (SBBD).
  2026. No prelo. → `3-extraction/` e
  [comparative-information-extraction](https://github.com/Paulo-Marcos-Assis/comparative-information-extraction).
- **ENIAC 2026** (aceito para publicação) — ASSIS, Paulo Marcos de; CASTRO, Márcio;
  CARVALHO, Jônata Tyska. *A Hybrid RAG Pipeline for Linking Fraud News to Public
  Procurement Records.* In: Anais do 23º Encontro Nacional de Inteligência Artificial e
  Computacional (ENIAC). Porto Alegre: Sociedade Brasileira de Computação, 2026. No prelo.
  → `4-linkage/` e
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
   (acurácia média strict match 91,45% e F1 médio 89,86%, média dos 4 atributos —
   Tabela 3 do SBBD; com a validação manual do atributo *objeto* documentada em
   `data/extraction/DOCUMENTACAO.md`, o teste recalcula para **Ac 90,82% /**
   **F1 94,44%**).
4. **Vinculação** (`4-linkage/`) — híbrido RAG: extração LLM + pré-filtro SQL +
   recuperação TF-IDF + rerank LLM (single-call `gpt-oss:20b` 76,8% dev /
   76,2% test, 163/214; melhor no teste entre os cinco modelos avaliados:
   `gemma4:31b`, 78,5% — ver `4-linkage/results/README.md`).

## Datasets

- **Classificação**: `data/classification/Complete_dataset_11309.csv` (11.309 notícias
  rotuladas, 968 positivas) + `CONSOLIDATED_EQM.csv` (11.133, pré-deduplicação) +
  `CONSOLIDATED_EQM_DEDUPED.csv` (10.379) + splits.
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
CUDA 12.8). Modelos via Ollama: `gpt-oss:20b`, `gemma4:31b`, `qwen3.8:27b`,
`qwen3.5:9b`, `qwen2.5:7b` (avaliados no rerank; variantes também exploradas
`qwen2.5:14b` e `qwen3:8b` nos scripts, e `gemma3:12b` na extração auxiliar de
município/modalidade do benchmark de teste). Embeddings de recuperação:
`bert-base-portuguese-cased`, `neuralmind/bert-large-portuguese-cased` e
`tcepi/helbert-base` (HelBERT).