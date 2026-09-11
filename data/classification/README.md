# Dataset de Classificação Binária de Notícias

## Arquivos

| Arquivo | Registros | Descrição |
|---|---|---|
| `Complete_dataset_11309.csv` | **11.309** | Dataset completo rotulado (pré-deduplicação) |
| `CONSOLIDATED_EQM.csv` | **11.133** | Consolidado (dedup de links + filtro >200 chars) — Task 1 |
| `CONSOLIDATED_EQM_DEDUPED.csv` | **10.379** | Consolidado após deduplicação rigorosa (Task 1b) |
| `train.csv` | 6.642 | Split de treino (8,88% positivos) |
| `dev.csv` | 1.643 | Split de desenvolvimento (9,01% positivos) |
| `test.csv` | 2.046 | Split de teste (acessado uma única vez) |
| `SPLIT_REPORT.txt` | — | Relatório da divisão (leakage check: 0 links sobrepostos) |
| `DISTRIBUTION_REPORT.txt` | — | Distribuição por portal/domínio |

## Colunas

- `label` — `1` = notícia sobre fraude/irregularidade em licitação; `0` = demais.
- `titulo` — título normalizado da notícia.
- `texto_noticia` — texto completo normalizado.
- `link_noticia` — URL de origem.

## Distribuição de rótulos (dataset completo)

- Positivos (`1`): **968** (8,56%)
- Negativos (`0`): **10.341** (91,44%)

## Origem

Notícias coletadas por web scraping de 11+ portais de Santa Catarina (`1-crawler/`)
e rotuladas via heurísticas + revisão manual. Os splits foram gerados por
`2-classification/scripts/task2_split.py` com estratificação por domínio do portal
para evitar vazamento.

## Reprodução / pipeline

Consulte `2-classification/README.md`. Para reconstruir `CONSOLIDATED_EQM.csv`,
`CONSOLIDATED_EQM_DEDUPED.csv` e os splits a partir de `Complete_dataset_11309.csv`,
execute `task1_consolidate_eda.py` → `task1b_rigorous_dedup.py` → `task2_split.py`.