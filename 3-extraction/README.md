# ***Information Extraction from Brazilian News Articles on Public Procurement Fraud***

paulo.marcos@grad.ufsc.br, {marcio.castro, jonata.tyska}@ufsc.br

*Abstract. Combating fraud in public procurement is a critical task for oversight agencies. While journalistic coverage can offers a rich source of early warning signals, its unstructured nature hinders automated auditing. To bridge this gap, we propose an Information Extraction (IE) pipeline to extract four key entities required for administrative linkage: municipality, procurement modality, notice number, and contract object. We evaluate three extraction approaches: (1) a Hybrid-Heuristic baseline; (2) a Retrieval-Augmented Generation (RAG) framework; and (3) an End-to-End Structured Large Language Model (LLM). Results demonstrate that the structured LLM significantly outperforms modular baselines, achieving an average accuracy of 91.45% across the four attributes. Furthermore, we introduce a gold-standard dataset of 796 annotated articles. All experiments are reproducible.*

---

*Notas de consistência com os dados publicados em `../data/extraction/`:*
- Os 91,45% acima são a média strict match do paper (SBBD 2026, Tabela 3). Com a
  validação **manual** do atributo *objeto* (`data/extraction/DOCUMENTACAO.md`), o
  teste (N=196) recalcula para **Acurácia 90,82% · Precisão 91,07% · Recall 98,08% ·
  F1 94,44%** — esses são os números da dissertação.
- O `classifier/` (baseline SVM sobre embeddings BERT) exige os pesos do modelo
  `bert_embeddings/model.safetensors`, que **não são versionados** (arquivo git-LFS
  de 1,3 GB; acima da cota LFS gratuita do GitHub). Para re-executar o classifier,
  baixe os pesos de `neuralmind/bert-large-portuguese-cased` (Hugging Face) e salve
  em `classifier/bert_embeddings/model.safetensors` — a configuração de pooling já
  está versionada (dim 1024, mean pooling) e confere com o `auto_svm_best_model_bert.pkl`.
