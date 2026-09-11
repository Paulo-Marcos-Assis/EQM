# Datasets

Conjuntos de dados construídos neste trabalho, organizados por etapa do pipeline:

| Pasta | Etapa | Conteúdo |
|---|---|---|
| `classification/` | Classificação binária | Dataset completo (11.309 notícias rotuladas) + splits train/dev/test e consolidated deduplicado |
| `extraction/` | Extração de Informação | Gold standard de 796 notícias anotadas (4 atributos) — 600 dev + 196 test |

## Ética

- Os textos das notícias foram coletados de portais públicos de Santa Catarina e
  rotulados manualmente/automaticamente neste trabalho para fins de pesquisa.
- **O benchmark sintético de vinculação NÃO é publicado** (714 notícias fictícias de
  fraude ligadas a processos reais) por questão ética. Sua regeneração é possível via
  `4-linkage/` (prompts + configs + scripts com seed fixo 42).

## Formato

Todos os arquivos são CSV UTF-8 com cabeçalho. Endpoint de origem e estrutura de
colunas documentados nos READMEs de cada subpasta.