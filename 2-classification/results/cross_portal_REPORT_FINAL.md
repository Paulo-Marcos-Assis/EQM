# RELATÓRIO FINAL — Teste Cross-Portal

Modelos treinados em `ndmais.com.br` avaliados em portais **diferentes** de ndmais
(dados do pipeline NEW_EQM_Classifier que nunca participaram do treino NDMAIS).

Fonte: `NEW_EQM_Classifier/FOR_TRAINING/CONSOLIDATED_EQM_DEDUPED.csv`
Pool: 9.555 não-ndmais (199 pos / 9.356 neg) | 5 subsets disjuntos de 438 linhas
(39 pos + 399 neg, razão neg/pos = 10,23 ≈ mesma do teste NDMAIS 10,22).

Todos os **12 combos** do PROGRESS.md do NDMAIS (3 vetorizações × 4 classificadores) foram avaliados.
Relatório completo (incl. por-subset e por-portal): `../results_all12/REPORT_ALL12.md`.

## 1. Resultados Agregados (5 subsets = 2.190 linhas, 195 pos / 1.995 neg)

### 1.1 Ordenado por F1 cross-portal (default threshold)

| Modelo | F1 | Prec | Rec | PR-AUC | ROC-AUC | Acc | pred+ | F1 dev (in-domain) | ΔF1 vs dev |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LinearSVC | **0.9650** | 0.9415 | 0.9897 | 0.9975 | 0.9997 | 0.9936 | 0.094 | 0.9610 | +0.0040 |
| TF-IDF + LogisticRegression | <u>0.9463</u> | 0.9023 | 0.9949 | 0.9968 | 0.9996 | 0.9900 | 0.098 | 0.9661 | -0.0198 |
| TF-IDF + RandomForest | 0.9205 | 0.8682 | 0.9795 | 0.9806 | 0.9975 | 0.9849 | 0.100 | 0.9593 | -0.0388 |
| BERTimbau Base + RandomForest | 0.8688 | 0.7773 | 0.9846 | 0.9699 | 0.9957 | 0.9735 | 0.113 | <u>0.9956</u> | -0.1268 |
| TF-IDF + XGBoost | 0.8674 | 0.7720 | 0.9897 | 0.9919 | 0.9987 | 0.9731 | 0.114 | 0.9205 | -0.0531 |
| BERTimbau Large + RandomForest | 0.6993 | 0.5406 | 0.9897 | 0.9715 | 0.9950 | 0.9242 | 0.163 | **1.0000** | -0.3007 |
| BERTimbau Base + XGBoost | 0.6380 | 0.4707 | 0.9897 | 0.9330 | 0.9914 | 0.9000 | 0.187 | 0.9912 | -0.3532 |
| BERTimbau Large + XGBoost | 0.5893 | 0.4196 | 0.9897 | 0.9316 | 0.9888 | 0.8772 | 0.210 | <u>0.9956</u> | -0.4063 |
| BERTimbau Base + LogisticRegression | 0.4012 | 0.2516 | 0.9897 | 0.8813 | 0.9863 | 0.7370 | 0.350 | <u>0.9956</u> | -0.5944 |
| BERTimbau Base + LinearSVC | 0.3841 | 0.2383 | 0.9897 | 0.8388 | 0.9816 | 0.7174 | 0.370 | **1.0000** | -0.6159 |
| BERTimbau Large + LogisticRegression | 0.3680 | 0.2260 | 0.9897 | 0.9277 | 0.9904 | 0.6973 | 0.390 | **1.0000** | -0.6320 |
| BERTimbau Large + LinearSVC | 0.3642 | 0.2231 | 0.9897 | 0.9107 | 0.9883 | 0.6922 | 0.395 | **1.0000** | -0.6358 |

### 1.2 Ordenado por vetorização (comparação por família)

| Vetorização | Melhor F1 cross-portal | Melhor F1 dev (in-domain) | ΔF1 |
|---|---|---|---|
| **TF-IDF** | **0.9650** (SVC) | 0.9661 (LR) | **-0.001** |
| BERTimbau Base | <u>0.8688</u> (RF) | 1.0000 (SVC) | -0.131 |
| BERTimbau Large | 0.6993 (RF) | 1.0000 (SVC/LR/RF) | -0.301 |

## 2. Matrizes de Confusão Agregadas (destaque)

| Modelo | TN | FP | FN | TP | Taxa pred. positiva | Taxa real positiva |
|---|---|---|---|---|---|---|
| TF-IDF + LinearSVC | 1895 | 100 | 2 | 193 | 0.094 | 0.089 |
| TF-IDF + LogisticRegression | 1865 | 130 | 1 | 194 | 0.098 | 0.089 |
| TF-IDF + RandomForest | 1868 | 127 | 4 | 191 | 0.100 | 0.089 |
| BERTimbau Base  + LinearSVC | 1378 | 617 | 2 | 193 | 0.370 | 0.089 |
| BERTimbau Large + LinearSVC | 1323 | 672 | 2 | 193 | 0.395 | 0.089 |
| BERTimbau Large + RandomForest | 1831 | 164 | 2 | 193 | 0.163 | 0.089 |

## 3. Por Subset

| Subset | Modelo | F1 | Prec | Rec |
|---|---|---|---|---|
| 01 | TF-IDF + LinearSVC | 0.9744 | 0.9744 | 0.9744 |
| 02 | TF-IDF + LinearSVC | 0.9750 | 0.9512 | 1.0000 |
| 03 | TF-IDF + LinearSVC | 0.9398 | 0.8864 | 1.0000 |
| 04 | TF-IDF + LinearSVC | 0.9620 | 0.9500 | 0.9744 |
| 05 | TF-IDF + LinearSVC | 0.9750 | 0.9512 | 1.0000 |
| 01 | TF-IDF + LogisticRegression | 0.9630 | 0.9286 | 1.0000 |
| 02 | TF-IDF + LogisticRegression | 0.9512 | 0.9070 | 1.0000 |
| 03 | TF-IDF + LogisticRegression | 0.9176 | 0.8478 | 1.0000 |
| 04 | TF-IDF + LogisticRegression | 0.9500 | 0.9268 | 0.9744 |
| 05 | TF-IDF + LogisticRegression | 0.9512 | 0.9070 | 1.0000 |
| 01 | TF-IDF + RandomForest | 0.9500 | 0.9268 | 0.9744 |
| 02 | TF-IDF + RandomForest | 0.9383 | 0.9048 | 0.9744 |
| 03 | TF-IDF + RandomForest | 0.9176 | 0.8478 | 1.0000 |
| 04 | TF-IDF + RandomForest | 0.8941 | 0.8261 | 0.9744 |
| 05 | TF-IDF + RandomForest | 0.9048 | 0.8444 | 0.9744 |
| 01 | TF-IDF + XGBoost | 0.8864 | 0.7959 | 1.0000 |
| 02 | TF-IDF + XGBoost | 0.8837 | 0.8085 | 0.9744 |
| 03 | TF-IDF + XGBoost | 0.8478 | 0.7358 | 1.0000 |
| 04 | TF-IDF + XGBoost | 0.8352 | 0.7308 | 0.9744 |
| 05 | TF-IDF + XGBoost | 0.8864 | 0.7959 | 1.0000 |
| 01 | BERTimbau Base + LinearSVC | 0.3900 | 0.2422 | 1.0000 |
| 02 | BERTimbau Base + LinearSVC | 0.3744 | 0.2317 | 0.9744 |
| 03 | BERTimbau Base + LinearSVC | 0.3800 | 0.2360 | 0.9744 |
| 04 | BERTimbau Base + LinearSVC | 0.3861 | 0.2393 | 1.0000 |
| 05 | BERTimbau Base + LinearSVC | 0.3900 | 0.2422 | 1.0000 |
| 01 | BERTimbau Base + LogisticRegression | 0.4084 | 0.2566 | 1.0000 |
| 02 | BERTimbau Base + LogisticRegression | 0.4000 | 0.2517 | 0.9744 |
| 03 | BERTimbau Base + LogisticRegression | 0.3918 | 0.2452 | 0.9744 |
| 04 | BERTimbau Base + LogisticRegression | 0.4000 | 0.2500 | 1.0000 |
| 05 | BERTimbau Base + LogisticRegression | 0.4062 | 0.2549 | 1.0000 |
| 01 | BERTimbau Base + RandomForest | 0.8966 | 0.8125 | 1.0000 |
| 02 | BERTimbau Base + RandomForest | 0.8736 | 0.7917 | 0.9744 |
| 03 | BERTimbau Base + RandomForest | 0.8261 | 0.7170 | 0.9744 |
| 04 | BERTimbau Base + RandomForest | 0.8837 | 0.8085 | 0.9744 |
| 05 | BERTimbau Base + RandomForest | 0.8667 | 0.7647 | 1.0000 |
| 01 | BERTimbau Base + XGBoost | 0.6393 | 0.4699 | 1.0000 |
| 02 | BERTimbau Base + XGBoost | 0.6179 | 0.4524 | 0.9744 |
| 03 | BERTimbau Base + XGBoost | 0.6230 | 0.4578 | 0.9744 |
| 04 | BERTimbau Base + XGBoost | 0.6724 | 0.5065 | 1.0000 |
| 05 | BERTimbau Base + XGBoost | 0.6393 | 0.4699 | 1.0000 |
| 01 | BERTimbau Large + LinearSVC | 0.3662 | 0.2241 | 1.0000 |
| 02 | BERTimbau Large + LinearSVC | 0.3585 | 0.2197 | 0.9744 |
| 03 | BERTimbau Large + LinearSVC | 0.3636 | 0.2235 | 0.9744 |
| 04 | BERTimbau Large + LinearSVC | 0.3645 | 0.2229 | 1.0000 |
| 05 | BERTimbau Large + LinearSVC | 0.3679 | 0.2254 | 1.0000 |
| 01 | BERTimbau Large + LogisticRegression | 0.3732 | 0.2294 | 1.0000 |
| 02 | BERTimbau Large + LogisticRegression | 0.3619 | 0.2222 | 0.9744 |
| 03 | BERTimbau Large + LogisticRegression | 0.3619 | 0.2222 | 0.9744 |
| 04 | BERTimbau Large + LogisticRegression | 0.3679 | 0.2254 | 1.0000 |
| 05 | BERTimbau Large + LogisticRegression | 0.3750 | 0.2308 | 1.0000 |
| 01 | BERTimbau Large + RandomForest | 0.7091 | 0.5493 | 1.0000 |
| 02 | BERTimbau Large + RandomForest | 0.6847 | 0.5278 | 0.9744 |
| 03 | BERTimbau Large + RandomForest | 0.6726 | 0.5135 | 0.9744 |
| 04 | BERTimbau Large + RandomForest | 0.7156 | 0.5571 | 1.0000 |
| 05 | BERTimbau Large + RandomForest | 0.7156 | 0.5571 | 1.0000 |
| 01 | BERTimbau Large + XGBoost | 0.6240 | 0.4535 | 1.0000 |
| 02 | BERTimbau Large + XGBoost | 0.5507 | 0.3838 | 0.9744 |
| 03 | BERTimbau Large + XGBoost | 0.5714 | 0.4043 | 0.9744 |
| 04 | BERTimbau Large + XGBoost | 0.6240 | 0.4535 | 1.0000 |
| 05 | BERTimbau Large + XGBoost | 0.5821 | 0.4105 | 1.0000 |

## 4. Interpretação (Achado Principal)

1. **TF-IDF é o verdadeiro generalizador cross-portal.** F1 = 0.9650 (SVC), 0.9463 (LR),
   0.9205 (RF), 0.8674 (XGB) — desempenho praticamente igual ao in-domain (dev 0.96), com taxa
   de predição positiva corretamente calibrada (~9,4% vs 8,9% real). O vocabulário léxico de fraude
   (licitação, contrato, TCE, superfaturamento) transfere entre portais.

2. **Os modelos "melhores" in-domain (BERTimbau, F1 dev = 1,0) são os PIORES na transferência.**
   Colapsam para 0.36–0.40 (SVC/LR) no default threshold: recall altíssimo (0.99) mas precisão
   colapsada (0.22–0.39) — superprevêem positivos (37–40% previstos vs 8,9% reais). Eles memorizam
   o estilo do portal junto com o conceito.

3. **O ranking generaliza bem mesmo nos BERTimbau.** PR-AUC 0.84–0.97 e ROC-AUC 0.98–0.99
   indicam que até os modelos densos discriminam fraude de não-fraude fora do domínio; o problema
   é a **calibração do limiar**. Com limiar ótimo por modelo, o F1 recupera para 0.80–0.94:

   | Modelo | F1 limiar default | F1 limiar ótimo | Limiar ótimo |
   |---|---|---|---|
   | BERTimbau Base + LinearSVC | 0.3841 | **0.8028** | score ≥ 1.1273 |
   | BERTimbau Large + LinearSVC | 0.3642 | **0.8585** | score ≥ 1.1569 |
   | BERTimbau Large + LogisticRegression | 0.3680 | **0.8804** | score ≥ 4.5366 |
   | BERTimbau Large + RandomForest | 0.6993 | **0.9430** | prob ≥ 0.8400 |

4. **Portais 100% positivos trivialmente "perfeitos".** mpsc/pc-sc/scc10 (fontes oficiais) e
   g1/portalmenina aparecem com F1=1,0 nos subsets porque **todas** as suas linhas são positivas
   (predizer tudo positivo = recall 1,0 sem negativos para errar). Portais sem positivos
   (bbc, cartacapital) mostram F1=0 porque tudo é previsto positivo (precisão 0).

## 5. Conclusão para o Artigo

O teste cross-portal demonstra empiricamente que **a escolha da vetorização muda com o objetivo**:

- **In-domain (portal único):** BERTimbau domina (F1 = 1,0) — domínio trivialmente separável,
  mas a vitória é parcialmente **overfitting de domínio** (estilo editorial + conceito).
- **Cross-portal (produção multi-fonte):** TF-IDF generaliza (F1 = 0.9650, Δ ≈ 0) porque aprende
  o **léxico do conceito de fraude**, não o estilo de um portal. BERTimbau exige recalibração de
  limiar para transferir (F1 sobe de ~0.36 para 0.80–0.94).

Isto explica a diferença entre os pipelines espelhados: o NDMAIS (single-portal) elegeu BERTimbau
porque seu domínio é trivial; o EQM (multi-portal) elegeu TF-IDF porque seu domínio é heterogêneo.
Para implantação em ambientes multi-fonte, recomenda-se **TF-IDF (ou BERTimbau com limiar recalibrado
em dados multi-portal)** — o que o EQM já faz.