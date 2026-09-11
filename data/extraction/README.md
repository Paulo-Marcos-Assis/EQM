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
substitui os números antigos da Tabela 3 do artigo do SBBD): para o atributo
*objeto*, test (N=196) recalcula para **Ac 90,82% / F1 94,44%**.

> **Sobre o `DOCUMENTACAO.md`:** trata-se do registro original da curadoria do gold
> standard, mantido verbatim. A tabela "antes/depois" do realinhamento de
> `link_noticia` descreve o estado intermediário computado **no momento da correção**
> (o deslocamento de uma linha foi aplicado antes da publicação das planilhas);
> ela não é reproduzível como igualdade exata `titulo == url` nos CSVs finais. O
> conjunto commitado é o gold standard validado (com colunas `validacao_*`).

## Referência

Artigo *"Information Extraction from Brazilian News Articles on Public Procurement
Fraud"* (SBBD 2026) — código e abordagens em `3-extraction/` do repositório e em
`github.com/Paulo-Marcos-Assis/comparative-information-extraction`.