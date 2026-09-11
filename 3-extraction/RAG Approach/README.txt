===================================================================================
RAG APPROACH - EXTRAÇÃO DE INFORMAÇÕES COM RETRIEVAL-AUGMENTED GENERATION
===================================================================================

VISÃO GERAL:
Este projeto implementa uma abordagem baseada em RAG (Retrieval-Augmented Generation)
para extração automática de informações estruturadas de notícias sobre fraudes em 
licitações públicas. Combina busca semântica vetorial com modelos de linguagem (LLM)
para extrair atributos específicos de forma mais precisa e contextual.

===================================================================================
ARQUIVOS E SUAS FUNÇÕES:
===================================================================================

1. similarity_pipeline.ipynb
   - Notebook principal do pipeline RAG completo
   - Implementa extração e avaliação integradas
   - Componentes principais:
     
     A. EXTRAÇÃO RAG (Células 0-5):
        * Chunking: Divide texto em pedaços de 600 caracteres com overlap de 100
        * Embedding: Usa BERTimbau (neuralmind/bert-base-portuguese-cased)
        * Indexação: Cria banco vetorial FAISS temporário por notícia
        * Retrieval: Busca top-2 trechos mais relevantes por atributo
        * Generation: LLM (Ollama gpt-oss:20b) extrai JSON dos trechos
        
     B. AVALIAÇÃO COM MÉTRICAS HÍBRIDAS (Células 14-17):
        * Exact Match: Para edital, modalidade, município
        * Similaridade Semântica: Para objeto (threshold 0.83)
        * Calcula TP, FP, FN, TN para cada atributo
        * Gera métricas: Acurácia, Precisão, Recall, F1-Score
   
   - Tecnologias usadas:
     * LangChain: Framework RAG (text splitters, vectorstores, retrievers)
     * FAISS: Banco vetorial para busca de similaridade
     * HuggingFace Embeddings: Modelo BERTimbau para vetorização
     * Ollama: Servidor LLM para geração de respostas
     * Scikit-learn: Cálculo de similaridade de cosseno
   
   - Fluxo de processamento:
     1. Carrega notícia do CSV
     2. Segmenta texto em chunks
     3. Cria índice vetorial (FAISS)
     4. Faz queries especializadas por atributo
     5. Recupera trechos relevantes
     6. Envia trechos ao LLM para extração
     7. Compara com ground truth usando métrica apropriada
     8. Salva resultados e métricas

2. 196_validation.csv
   - Dataset de validação com 196 notícias
   - Contém gabarito completo para todos os atributos
   - Colunas principais:
     * link_noticia: URL da notícia original
     * titulo: título da notícia
     * texto_noticia: texto completo
     * titulo_texto: concatenação de título + texto (usado no RAG)
     * municipio_ente, edital, modalidade_licitacao, objeto: gabarito
     * *_presente: flags indicando presença do atributo (1/0)
   - Usado para avaliação final do pipeline RAG

3. 600_gabarito.csv
   - Dataset maior com 600 notícias validadas manualmente
   - Mesma estrutura do arquivo de 196 notícias
   - Pode ser usado para experimentos adicionais
   - Contém validações manuais completas

4. just_test.csv
   - Subset de teste com 10 notícias
   - Criado para testes rápidos antes de processar dataset completo
   - Usado durante desenvolvimento do pipeline
   - Mesma estrutura dos outros CSVs

===================================================================================
DEPENDÊNCIAS ENTRE ARQUIVOS:
===================================================================================

similarity_pipeline.ipynb
  ├─ Lê: 196_validation.csv (ou just_test.csv para testes)
  ├─ Requer: 
  │   ├─ Ollama LLM (https://ollama-dev.ceos.ufsc.br)
  │   ├─ Modelo: gpt-oss:20b
  │   └─ Embedding: neuralmind/bert-base-portuguese-cased
  ├─ Gera: resultado_rag_parcial.csv (salvamento incremental)
  └─ Gera: resultado_rag_final.csv (resultado completo com métricas)

===================================================================================
FLUXO DE TRABALHO:
===================================================================================

1. PREPARAÇÃO DO AMBIENTE:
   - Instalar dependências Python:
     * sentence-transformers
     * faiss-cpu
     * langchain-community
     * langchain-huggingface
     * langchain-ollama
     * scikit-learn
   - Configurar acesso ao servidor Ollama
   - Baixar modelo de embedding BERTimbau (automático na 1ª execução)

2. TESTE RÁPIDO (Opcional):
   - Usar just_test.csv (10 notícias)
   - Validar funcionamento do pipeline
   - Ajustar parâmetros se necessário

3. PROCESSAMENTO COMPLETO:
   - Executar células do notebook em sequência
   - Pipeline processa 196 notícias
   - Salvamento automático a cada 20 notícias
   - Tempo estimado: ~20 minutos (6 segundos/notícia)

4. ANÁLISE DE RESULTADOS:
   - Revisar métricas por atributo
   - Analisar casos de FP/FN
   - Ajustar threshold de similaridade se necessário
   - Comparar com abordagem híbrida (Hybrid Heuristic Approach)

===================================================================================
RESULTADOS OBTIDOS (196 notícias de validação):
===================================================================================

MÉTRICAS POR ATRIBUTO:
  - Município:   58.16% acurácia | 65.55% precisão | 65.55% recall (Exact Match)
  - Modalidade:  80.61% acurácia | 51.02% precisão | 64.10% recall (Exact Match)
  - Edital:      96.43% acurácia | 75.00% precisão | 69.23% recall (Exact Match)
  - Objeto:      37.24% acurácia | 38.24% precisão | 57.14% recall (Similaridade ≥0.83)

ACURÁCIA MÉDIA GERAL: 68.11%

OBSERVAÇÕES:
  - Edital tem melhor performance (96.43%)
  - Objeto tem desafios devido à variabilidade textual
  - Modalidade sofre com confusão entre "fraude na concorrência" vs "concorrência"
  - Município tem boa recall mas precisão pode melhorar

===================================================================================
VANTAGENS DA ABORDAGEM RAG:
===================================================================================

1. CONTEXTUALIZAÇÃO INTELIGENTE:
   - Envia apenas trechos relevantes ao LLM (não texto completo)
   - Reduz ruído e melhora foco do modelo
   - Queries especializadas por atributo

2. ESCALABILIDADE:
   - Processa textos longos sem limitação de tokens
   - Busca vetorial é eficiente (FAISS)
   - Pode processar milhares de documentos

3. FLEXIBILIDADE:
   - Fácil adicionar novos atributos (criar nova query)
   - Pode ajustar chunk_size e overlap
   - Threshold de similaridade configurável

4. AVALIAÇÃO ROBUSTA:
   - Métricas híbridas (exact + semantic)
   - Similaridade semântica para textos descritivos
   - Captura nuances que match exato perde

===================================================================================
COMPARAÇÃO COM HYBRID HEURISTIC APPROACH:
===================================================================================

RAG APPROACH:
  ✓ Melhor para textos descritivos (objeto)
  ✓ Mais flexível e escalável
  ✓ Não requer indexação prévia (Solr)
  ✗ Mais lento (6s/notícia vs 1-2s)
  ✗ Requer mais recursos computacionais

HYBRID HEURISTIC:
  ✓ Mais rápido para entidades estruturadas
  ✓ Excelente para editais (regex) e municípios (Solr)
  ✓ Menor uso de recursos
  ✗ Menos flexível para novos atributos
  ✗ Requer manutenção de índices (Solr)

RECOMENDAÇÃO:
  - Use RAG para: objeto, descrições complexas, novos atributos
  - Use Hybrid para: edital, município, modalidade (se bem definida)
  - Considere abordagem híbrida combinando ambos

===================================================================================
PARÂMETROS CONFIGURÁVEIS:
===================================================================================

CHUNKING:
  - chunk_size: 600 caracteres (ajustar para textos mais/menos densos)
  - chunk_overlap: 100 caracteres (evita perda de contexto nas bordas)

RETRIEVAL:
  - k: 2 (número de chunks recuperados por query)
  - Aumentar k se contexto insuficiente

SIMILARIDADE:
  - SIMILARITY_THRESHOLD: 0.83 (para classificar TP vs FP em objeto)
  - Ajustar baseado em análise de falsos positivos/negativos

LLM:
  - temperature: 0 (determinístico)
  - model: gpt-oss:20b (pode testar outros modelos Ollama)

PROCESSAMENTO:
  - SAVE_EVERY: 20 (frequência de salvamento parcial)
  - START_INDEX/END_INDEX: controla slice de processamento
  - RESUME_PROCESSING: True (retoma de onde parou)

===================================================================================
LIMITAÇÕES E TRABALHOS FUTUROS:
===================================================================================

LIMITAÇÕES ATUAIS:
  - Performance de objeto ainda baixa (37.24%)
  - Confusão entre modalidade e fraude em modalidade
  - Dependência de servidor Ollama externo
  - Tempo de processamento relativamente alto

MELHORIAS FUTURAS:
  1. Fine-tuning do modelo de embedding em domínio jurídico
  2. Prompt engineering mais sofisticado (few-shot examples)
  3. Ensemble com Hybrid Approach para melhor performance
  4. Cache de embeddings para acelerar reprocessamento
  5. Ajuste dinâmico de threshold por tipo de objeto
  6. Análise de erros para identificar padrões de falha

===================================================================================
AUTOR/CONTATO:
Projeto desenvolvido para análise de fraudes em licitações públicas
Dataset: Ministério Público de Santa Catarina (MPSC)
Abordagem: RAG (Retrieval-Augmented Generation)
===================================================================================
