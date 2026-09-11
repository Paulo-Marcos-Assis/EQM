# Documentação do Dataset — Extração de Informação (Objeto)

## Visão Geral

Esta documentação consolida o estado **válido** dos arquivos de dataset do projeto de
Extração de Informação (IE) para fraudes em licitações públicas de Santa Catarina (SC).
Contém apenas as correções e métricas efetivamente aplicadas e validadas, excluindo
tentativas intermediárias incorretas.

## Arquivos

| Arquivo | Linhas | Papel |
|---------|--------|-------|
| `Gabarito.csv` | 796 | Dataset de ouro completo (ground truth) |
| `dev.csv` | 600 | Conjunto de desenvolvimento (~80%) — refinamento de heurísticas/prompts |
| `teste.csv` | 196 | Conjunto de teste (hold-out, ~20%) — avaliação final |

Todos os arquivos possuem a mesma estrutura de colunas (21 colunas).

## Colunas

```
link_noticia, titulo, texto_noticia, municipio_ente, municipio_presente,
municipios_extraidos, (coluna vazia), edital, edital_presente, editais_extraidos,
validacao_edital, modalidade_licitacao, modalidade_presente, modalidades_extraidas,
validacao_modalidade, unnamed:_13, objeto, objeto_presente, objeto_extraido,
validacao_objeto, validacao_municipio
```

### Convenções implementadas
- Nomes de colunas: **minúsculas, sem acentos, sem caracteres especiais**.
- Colunas de validação (`validacao_*`) usam a notação: `tp*` (verdadeiro positivo),
  `vn*` (verdadeiro negativo), `fp*` (falso positivo), `fn*` (falso negativo).

## Correções Aplicadas (estado válido)

### 1. Padronização dos nomes das colunas
Conforme solicitação, todos os nomes de colunas foram convertidos para minúsculas,
sem acentos e sem caracteres especiais (ex.: `validaçao_objeto` → `validacao_objeto`).

### 2. Realinhamento da coluna `link_noticia`
**Diagnóstico:** no dataset original, o bloco de conteúdo (`titulo`, `texto_noticia`,
`validacao_*`, `objeto`, etc.) era coerente dentro de cada linha, mas a coluna
`link_noticia` estava **deslocada de uma linha para cima** — a URL da linha `i`
pertencia ao título da linha `i+1`.

**Correção aplicada (Regra A):** `link_noticia[i]` recebe o valor da linha `i-1`
(a URL desce uma linha), preservando título/texto/validações como um bloco coerente.

| Arquivo | antes (titulo==url) | depois (titulo==url) |
|---------|---------------------|----------------------|
| `Gabarito.csv` | 51/796 | **436/796** |
| `dev.csv` | — | 269/600 |
| `teste.csv` | — | 167/196 |

As divergências restantes são naturais: slugs de URL que diferem do título final
publicado (títulos editados, URLs com query string, formato `.ghtml`, etc.), e não
deslocamentos de linha.

### 3. Validação manual do atributo Objeto
A coluna `validacao_objeto` foi **revalidada manualmente** (autor) usando o
`Gabarito.csv` como base. A associação foi preservada por **posição da linha
(conteúdo)**, e **não** foi sobrescrita por outras fontes (ex.: `18_11`).

Distribuição de `validacao_objeto`:

| Dataset | tp* | vn* | fp* | fn* | Total |
|---------|-----|-----|-----|-----|-------|
| `Gabarito.csv` | 612 | 105 | 56 | 23 | 796 |
| `dev.csv` | 459 | 80 | 41 | 20 | 600 |
| `teste.csv` | 153 | 25 | 15 | 3 | 196 |

Observação: `dev = Gabarito − teste` (consistente).

### 4. Validações de município, modalidade e edital (conjunto de teste)
Para o `teste.csv`, as validações das colunas `validacao_municipio`,
`validacao_modalidade` e `validacao_edital` foram completadas (196/196) a partir da
fonte canônica `18_11_analise_geral_strict_structured.csv` (colunas
`classif_strict_*`), usada originalmente para gerar a Tabela 3 do paper.

**Importante:** isso **não** afetou `validacao_objeto`, que permaneceu com a
validação manual do autor.

## Métricas — Atributo Objeto (validação manual)

Métricas padrão para classificação, calculadas sobre `validacao_objeto`:

```
Acurácia  = (TP + TN) / N
Precisão  = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1-Score  = 2 * (Precisão * Recall) / (Precisão + Recall)
```

### Conjunto de teste (N = 196)
| Classe | % |
|--------|-----|
| TP | 153 |
| TN | 25 |
| FP | 15 |
| FN | 3 |

| Métrica | Valor |
|---------|-------|
| Acurácia | **90.82%** |
| Precisão | **91.07%** |
| Recall | **98.08%** |
| F1-Score | **94.44%** |

### Conjunto de desenvolvimento (N = 600)
| Classe | % |
|--------|-----|
| TP | 459 |
| TN | 80 |
| FP | 41 |
| FN | 20 |

| Métrica | Valor |
|---------|-------|
| Acurácia | **89.83%** |
| Precisão | **91.80%** |
| Recall | **95.82%** |
| F1-Score | **93.77%** |

### Dataset completo / Gabarito (N = 796)
| Classe | % |
|--------|-----|
| TP | 612 |
| TN | 105 |
| FP | 56 |
| FN | 23 |

| Métrica | Valor |
|---------|-------|
| Acurácia | **90.08%** |
| Precisão | **91.62%** |
| Recall | **96.38%** |
| F1-Score | **93.94%** |

## Nota sobre o paper (iephase.md)

A Tabela 3 do paper reporta, para o atributo Objeto (LLM Estruturado): Acurácia
93.33%, Precisão 94.05%, Recall 98.14%, F1 96.05%.

Os dados recalculados neste dataset (com a validação manual do autor e a linha de
Jaguaruna incluída como `vn*`) resultam em métricas divergentes — ver valores do
conjunto de teste acima. **O paper ainda está desatualizado** e deve ser revisado
com estes números.

## Fontes

- Dataset base original: `split/2 (novembro)/Gabarito.csv`
- Fonte das validações de município/modalidade/edital (teste):
  `PIPELINE/test/test 2 (validation)/LLM_ALL/18_11_analise_geral_strict_structured.csv`
- Validação manual do objeto: realizada pelo autor sobre o `Gabarito.csv`
