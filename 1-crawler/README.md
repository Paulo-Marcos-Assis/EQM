# 1. Coleta de Notícias (Crawler)

Coletor de notícias publicadas em portais de Santa Catarina sobre fraudes e irregularidades
em licitações públicas. Faz scraping de 11+ portais, com seletores configuráveis por portal.

## Portais configurados

`ndmais`, `nsc`, `jornalconexao`, `olharsc`, `tvbv`, `portaldelicitacao`, `ocpnews`,
`iclnoticias`, `g1sc`, `bbcportuguese`, `scempauta`.

## Arquivos

| Arquivo | Descrição |
|---|---|
| `main.py` | Coletor (consumidor de fila) com busca por data nas páginas dos portais |
| `tools.py` | Parsers de data (ISO, formato português, DD/MM/YYYY, G1) |
| `crawler_configs.json` | Seletores HTML, limites de paginação e parser de data por portal |
| `schemas/v1/schema.json` | Schema do JSON de trigger (validação da mensagem de entrada) |
| `requirements.txt` | Dependências Python |

## Como funciona

1. Uma mensagem de trigger com `portal`, template de URL de listagem (`url` com `{}`)
   e intervalo de datas chega pela fila (`INPUT_QUEUE`).
2. O coletor encontra a página onde está a data-alvo (busca binária + varredura de paginação).
3. Extrai título, chamada, texto, URL e data de cada artigo usando os seletores do
   `crawler_configs.json`.
4. Publica os registros no MongoDB e os encaminha à fila de saída (`OUTPUT_QUEUE`).

## Dependências de infraestrutura (produção)

Este é o código de produção do ambiente CEOS. Para executá-lo é necessário:

- **Pacote `service_essentials`** — biblioteca interna da UFSC (repositório privado)
  com a classe base `CachedCollectorService`, gerenciadores de fila (RabbitMQ),
  MongoDB, MinIO e logging. No ambiente original é instalado via imagem-base `ceos-base`
  (Dockerfile).
- **RabbitMQ** — fila de mensagens (`INPUT_QUEUE`/`OUTPUT_QUEUE`).
- **MongoDB** — persistência dos artigos e do cache de configuração.
- **MinIO** — armazenamento de objetos (modo de ingestão por pasta).

## Variáveis de ambiente

Vide `.env.example` na raiz do repositório e `schemas/v1/schema.json` para o contrato
da mensagem de entrada.

## Execução

```bash
pip install -r requirements.txt
# configurar variáveis de ambiente (ver .env.example)
python main.py
```