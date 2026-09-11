# Benchmark de Vinculação

O benchmark de vinculação (714 pares: 500 dev + 214 test) relaciona notícias de fraude
aos seus respectivos processos licitatórios nas bases oficiais (PNCP/e-Sfinge SC).

**Os 714 pares sintéticos NÃO são publicados** por razões éticas (notícias fictícias
de fraude ligadas a processos e municípios reais). A regeneração é totalmente
determinística e documentada em `../4-linkage/benchmarks/`:

- `regenerate_benchmark.py` — gera os pares (`--split dev|test|both`).
- `prompts/` — 4 prompts exatos usados no paper (geração, extração, rerank).
- `configs/` — 5 YAMLs (modelos, seeds, temperatura, hiperparâmetros).
- Seed fixa **42** garante regeneração bit-a-bit.

Execução: `cd ../4-linkage && make bench-both`.

Ver `4-linkage/README.md` para o pipeline completo (indexes → retrieval → rerank).