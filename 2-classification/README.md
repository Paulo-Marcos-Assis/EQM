# 2. Classificação Binária de Notícias

Detecção de notícias sobre fraudes ou irregularidades em licitações públicas
(classificação binária: `1` = fraude/irregularidade, `0` = demais notícias).

**Modelo final (vencedor): TF-IDF + LinearSVC** — escolhido por melhor desempenho e,
principalmente, por robustez cross-portal (não captura padrões estilísticos do portal).

## Resultados do modelo vencedor

| Métrica | Dev (1.643) | Teste (2.046) |
|---|---|---|
| F1 | **0,9764** | **0,9563** |
| Precisão | 0,9732 | 0,9409 |
| Recall | 0,9797 | 0,9722 |
| Acurácia | 0,9957 | 0,9922 |
| PR-AUC | 0,9973 | 0,9905 |
| ROC-AUC | 0,9997 | 0,9991 |

Hyperparâmetros: `C=10`, `class_weight='balanced'`, `max_iter=10000`,
TF-IDF de 10.000 features. Erros no teste: 16 (11 FP / 5 FN). Detalhes em
`results/CONSOLIDATED_FINAL.md` e `results/`.

**Por que TF-IDF + SVM (e não BERT):** no experimento de viés de domínio
(`results/cross_portal_REPORT_FINAL.md`), os 4 melhores modelos NDMAIS_BIAS
(BERTimbau treinado **somente** em `ndmais.com.br`, dev F1 = 1,0 no domínio)
foram avaliados em 5 subsets disjuntos de 438 notícias cada (2.190 no total),
sorteados do pool **não-ndmais (9.555 linhas, 73 portais no pool; os subsets
abrangem 66 portais distintos — 25–29 portais por subset)**. Os 3 modelos
lineares colapsaram para F1 0,36–0,40 (Large+SVC 0,364; Large+LR 0,368; Base+SVC
0,384; Base+LR 0,401); os RandomForest seguraram melhor (Base 0,869; Large 0,699 —
relatório dos 12 combos). Já o TF-IDF + LinearSVC (multi-portal) **manteve** o
desempenho: F1 0,961 (ndmais) → 0,965 cross-portal (Δ +0,004). O modelo aprende o
*conceito* de fraude (lexical), não o estilo editorial.

## Pipeline (scripts)

| Script | Etapa |
|---|---|
| `task1_consolidate_eda.py` | Consolida e faz EDA do dataset completo |
| `task1b_rigorous_dedup.py` | Deduplicação rigorosa (TF-IDF + similaridade) |
| `task2_split.py` | Divisão train/dev/test estratificada (porta por domínio) |
| `task3_preprocess.py` | Pré-processamento (stemming RSLP, etc.) |
| `task4_vectorize.py` | TF-IDF + BERTimbau (média dos tokens) |
| `task5_train.py` | Treino das 12–16 combinações (GridSearch C, StratifiedKFold) |
| `task7_final_test.py` | Avaliação isolada no teste (Task 7) |
| `task8_domain_bias.py` | Viés de domínio (NDMAIS→cross-portal) + atribuição de features |
| `task9_consolidate.py` | Consolida resultados |

Os scripts resolvem o caminho raiz do projeto a partir da própria localização do
arquivo (não há caminhos absolutos fixos). O dataset completo e os splits estão em
`../data/classification/`.

## Reprodução

```bash
pip install -r requirements.txt
# (opcional) reconstruir splits a partir do dataset completo
python scripts/task1_consolidate_eda.py
python scripts/task1b_rigorous_dedup.py
python scripts/task2_split.py
# pipeline completo
python scripts/task3_preprocess.py
python scripts/task4_vectorize.py
python scripts/task5_train.py
python scripts/task7_final_test.py
python scripts/task8_domain_bias.py
```

Para reproduzir somente o modelo vencedor consulte `results/` (artefatos do
modelo final: `linear_svc_modelo_final.pkl`, `best_params.json`,
`classificacao_teste.txt`, `resumo_teste.json`).

## Hardware / ambiente de execução

AMD EPYC 9654 (96 núcleos, 1,5 TB RAM), GPU NVIDIA RTX A6000 (48 GB VRAM, CUDA 12.8),
Python 3.12, scikit-learn 1.8, PyTorch 2.12, Transformers 5.8 (para os modelos BERT).