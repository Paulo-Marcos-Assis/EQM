===================================================================================
HYBRID HEURISTIC APPROACH - EXTRAÇÃO DE INFORMAÇÕES DE LICITAÇÕES
===================================================================================

VISÃO GERAL:
Este projeto implementa uma abordagem híbrida para extração automática de 
informações estruturadas de notícias sobre fraudes em licitações públicas em 
Santa Catarina. Combina técnicas de busca (Apache Solr), expressões regulares 
e modelos de linguagem (LLM) para extrair atributos específicos.

===================================================================================
ARQUIVOS E SUAS FUNÇÕES:
===================================================================================

1. APACHE_SOLAR_3_ATRIBUTOS.ipynb
   - Notebook principal de extração híbrida
   - Extrai 3 atributos: município, modalidade de licitação e número de edital
   - Tecnologias usadas:
     * Apache Solr: extração de municípios (295 municípios de SC indexados)
     * Regex: extração de números de editais (padrões como "36/2018", "CC184/2021")
     * LLM não utilizado neste notebook
   - Processa 983 notícias do arquivo base
   - Gera arquivo de saída com resultados da extração

2. Avaliacao_abordagem_hibrida_modalidade_objeto.ipynb
   - Notebook de avaliação e validação do pipeline híbrido
   - Avalia 4 atributos: município, modalidade, edital e objeto
   - Tecnologias usadas:
     * Apache Solr: extração de municípios
     * Regex: extração de editais
     * LLM (Ollama - modelo gpt-oss:20b): extração de modalidade e objeto
   - Calcula métricas de desempenho (acurácia, precisão, recall, F1-score)
   - Duas métricas: "Contém" (menos restritiva) e "Restrita" (acerto exato)
   - Processa em lotes com salvamento incremental
   - Gera relatórios detalhados de performance

3. 983_noticias_fraude.csv
   - Dataset completo com 983 notícias sobre fraudes em licitações
   - Fonte: Ministério Público de Santa Catarina (MPSC)
   - Colunas principais:
     * link_noticia: URL da notícia original
     * titulo: título da notícia
     * texto_noticia: texto completo
     * municipio_ente, edital, modalidade_licitacao, objeto: gabarito manual
     * colunas de validação (*_presente, *_extraidos, validaçao_*)

4. 600_gabarito.csv
   - Subset de 600 notícias com gabarito validado manualmente
   - Usado para treinamento e validação intermediária
   - Mesma estrutura do arquivo de 983 notícias
   - Contém validações manuais para todos os atributos

5. 196_validation.csv
   - Conjunto de validação com 196 notícias
   - Usado para avaliação final do modelo
   - Contém gabarito completo e validações manuais
   - Estrutura similar aos outros CSVs

===================================================================================
DEPENDÊNCIAS ENTRE ARQUIVOS:
===================================================================================

APACHE_SOLAR_3_ATRIBUTOS.ipynb
  ├─ Lê: 983_noticias_fraude.csv (ou arquivo configurado)
  ├─ Requer: Apache Solr rodando (localhost:8983)
  └─ Gera: extractionv0_Solr_3_atributos.csv

Avaliacao_abordagem_hibrida_modalidade_objeto.ipynb
  ├─ Lê: 196_validation.csv (configurável via CAMINHO_CSV)
  ├─ Requer: Apache Solr (localhost:8983) + Ollama LLM
  ├─ Gera: analise_parcial_hibrido.csv (salvamento incremental)
  └─ Gera: analise_geral_hibrido.csv (resultado final)

===================================================================================
FLUXO DE TRABALHO:
===================================================================================

1. PREPARAÇÃO DO AMBIENTE:
   - Instalar Apache Solr 9.6.1
   - Configurar Ollama com modelo gpt-oss:20b
   - Instalar dependências Python: pysolr, pandas, langchain-ollama, tqdm

2. INDEXAÇÃO (APACHE_SOLAR_3_ATRIBUTOS.ipynb):
   - Indexar 295 municípios de SC no Solr (core: municipios)
   - Indexar 12 modalidades de licitação no Solr (core: modalidades)

3. EXTRAÇÃO (APACHE_SOLAR_3_ATRIBUTOS.ipynb):
   - Processar notícias extraindo município, modalidade e edital
   - Salvar resultados com predições

4. AVALIAÇÃO (Avaliacao_abordagem_hibrida_modalidade_objeto.ipynb):
   - Carregar dataset de validação (196_validation.csv)
   - Executar pipeline híbrido completo (Solr + Regex + LLM)
   - Calcular métricas de performance
   - Gerar relatório com acurácias por atributo

===================================================================================
RESULTADOS OBTIDOS (196 notícias de validação):
===================================================================================

MÉTRICA "CONTÉM" (menos restritiva):
  - Município:   66.84% acurácia
  - Modalidade:  89.29% acurácia
  - Edital:      95.41% acurácia
  - Objeto:      93.33% acurácia (validação manual)
  - MÉDIA GERAL: 86.22%

MÉTRICA "RESTRITA" (acerto exato):
  - Município:   36.22% acurácia
  - Modalidade:  89.29% acurácia
  - Edital:      91.33% acurácia
  - Objeto:      93.33% acurácia (validação manual)
  - MÉDIA GERAL: 77.54%

===================================================================================
OBSERVAÇÕES TÉCNICAS:
===================================================================================

- Apache Solr é usado para busca eficiente de entidades (municípios)
- Regex captura padrões estruturados (números de editais)
- LLM extrai informações complexas e contextuais (modalidade, objeto)
- Abordagem híbrida combina velocidade (Solr/Regex) com precisão (LLM)
- Sistema robusto com salvamento incremental e capacidade de retomada
- Métricas duplas permitem avaliar diferentes níveis de rigor

===================================================================================
AUTOR/CONTATO:
Projeto desenvolvido para análise de fraudes em licitações públicas
Dataset: Ministério Público de Santa Catarina (MPSC)
===================================================================================
