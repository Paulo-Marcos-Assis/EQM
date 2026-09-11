# Task 8 — Domain Bias Analysis Report (EQM)

**Date:** 2026-08-19

## 1. Portal Distribution

- **Total unique portals:** 74 (multi-portal dataset)
- **Portals with fraud positives:** 60
- **Largest by volume:** nsctotal.com.br
- **Largest by fraud positives:** ndmais.com.br (720)

| Split | Total | Pos | Neg | Portals |
|-------|-------|-----|-----|----------|
| Train | 6642 | 590 | 6052 | 52 |
| Dev | 1643 | 148 | 1495 | 28 |
| Test | 2046 | 180 | 1866 | 31 |

## 2. Metrics by Portal (best model, dev+test held-out)

| Portal | n | Pos | F1 | Precision | Recall | PR-AUC |
|--------|---|-----|----|-----------|--------|--------|
| ndmais.com.br | 289 | 254 | 0.9609 | 0.9535 | 0.9685 | 0.9916 |
| mpsc.mp.br | 19 | 19 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| nsctotal.com.br | 2795 | 9 | 0.9474 | 0.9000 | 1.0000 | 1.0000 |
| scc10.com.br | 6 | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| g1.globo.com | 4 | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| portalmenina.com.br | 4 | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| scempauta.com.br | 2 | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| jornalconexao.com.br | 62 | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| jornalaw.com.br | 2 | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| notisul.com.br | 2 | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| noticias.r7.com | 2 | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pc.sc.gov.br | 2 | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| garopaba.sc | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tnsul.com | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ocp.news | 39 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| agenciabrasil.ebc.com.br | 6 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gauchazh.clicrbs.com.br | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| jornalmetas.com.br | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| mpc.sc.gov.br | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| jatv.com.br | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| omunicipio.com.br | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| clickcamboriu.com.br | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gazetasbs.com.br | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tjsc.jus.br | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| atualfm.com.br | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## 3. Cross-Domain Robustness

| Group | n | Pos | F1 | Precision | Recall | PR-AUC |
|-------|---|-----|----|-----------|--------|--------|
| other | 583 | 43 | 0.9773 | 0.9556 | 1.0000 | 0.9984 |
| nsc_volume | 2795 | 9 | 0.9474 | 0.9000 | 1.0000 | 1.0000 |
| ndmais_bulk | 289 | 254 | 0.9609 | 0.9535 | 0.9685 | 0.9916 |
| official_minority | 22 | 22 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## 4. Feature Attribution (top-20 fraud features)

### tfidf_linear_svc

- Semantic (fraud): 16/20
- Stylized (portal): 0/20
- Other: 4/20

| Rank | Feature | Weight | Type |
|------|---------|--------|------|
| 1 | licitacao | +3.1129 | Semantic |
| 2 | licitacoes | +2.8981 | Semantic |
| 3 | respiradores | +1.7403 | Semantic |
| 4 | empresa | +1.7251 | Semantic |
| 5 | fraude | +1.6797 | Semantic |
| 6 | contratos | +1.6653 | Semantic |
| 7 | contrato | +1.6479 | Semantic |
| 8 | operacao | +1.5084 | Semantic |
| 9 | compra | +1.5061 | Semantic |
| 10 | superfaturamento | +1.5034 | Semantic |
| 11 | corrupcao | +1.4366 | Semantic |
| 12 | mascaras | +1.3632 | Other |
| 13 | fraudes | +1.2720 | Semantic |
| 14 | irregularidades | +1.2528 | Semantic |
| 15 | publicos | +1.2204 | Semantic |
| 16 | mensageiro | +1.1658 | Other |
| 17 | ex-prefeito | +1.1591 | Semantic |
| 18 | concorrencia | +1.1155 | Other |
| 19 | tce | +1.0807 | Semantic |
| 20 | licitatorios | +1.0728 | Other |

### tfidf_logistic_regression

- Semantic (fraud): 18/20
- Stylized (portal): 0/20
- Other: 2/20

| Rank | Feature | Weight | Type |
|------|---------|--------|------|
| 1 | licitacao | +10.6189 | Semantic |
| 2 | licitacoes | +8.4230 | Semantic |
| 3 | operacao | +7.2930 | Semantic |
| 4 | empresa | +6.8760 | Semantic |
| 5 | respiradores | +6.4074 | Semantic |
| 6 | contrato | +5.6835 | Semantic |
| 7 | fraude | +5.6360 | Semantic |
| 8 | contratos | +5.6074 | Semantic |
| 9 | compra | +5.5847 | Semantic |
| 10 | corrupcao | +4.6779 | Semantic |
| 11 | publicos | +4.6117 | Semantic |
| 12 | prefeito | +4.1696 | Semantic |
| 13 | fraudes | +4.0980 | Semantic |
| 14 | mensageiro | +4.0548 | Other |
| 15 | prefeitura | +4.0142 | Other |
| 16 | ex-prefeito | +3.9530 | Semantic |
| 17 | irregularidades | +3.8967 | Semantic |
| 18 | empresas | +3.8189 | Semantic |
| 19 | superfaturamento | +3.7465 | Semantic |
| 20 | tce | +3.4861 | Semantic |

## 5. Domain Bias Assessment

- **Bias level:** LOW
- **LinearSVC semantic ratio:** 80%
- **LR semantic ratio:** 90%
- **Explanation:** Fraud-semantic features dominate (>70% of top-20 for both models). Model learns the fraud CONCEPT, not portal style.

## 6. Recommendations

1. Multi-portal dataset (74 domains) — cross-domain evaluation IS possible (unlike NDMAIS single-portal)
2. Official minority sources (mpsc.mp.br, tjsc.jus.br, pc.sc.gov.br) evaluated separately
3. Feature attribution determines whether the model learns fraud semantics or portal style
4. **Future work:** diversify further (more portals per source type), monitor F1 on out-of-distribution portals
