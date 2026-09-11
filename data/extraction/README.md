# Dataset Gold Standard de Extração de Informação

Gold standard de **796 notícias anotadas** com os quatro atributos de contratação
pública exigidos para a vinculação: *município*, *modalidade*, *nº de edital* e *objeto*.

## Arquivos

| Arquivo | Registros | Descrição |
|---|---|---|
| `Gabarito.csv` | **796** | Gold standard completo (município, modalidade, nº edital, objeto) |
| `dev.csv` | 600 | Divisão de desenvolvimento |
| `teste.csv` | 196 | Divisão de teste |
| `DOCUMENTACAO.md` | — | Documentação da construção e cálculo das métricas |

## Origem

Notícias selecionadas a partir do corpus de fraude (983 notícias reais) e anotadas
para os estudos comparativos de Extração de Informação (heurística, RAG e LLM).
Detalhes e métricas atualizadas em `DOCUMENTACAO.md` (a recálculo ali documentado
substitui os números antigos da Tabela 3 do artigo do SBBD).

## Referência

Artigo *"Information Extraction from Brazilian News Articles on Public Procurement
Fraud"* (SBBD 2026) — código e abordagens em `3-extraction/` do repositório e em
`github.com/Paulo-Marcos-Assis/comparative-information-extraction`.