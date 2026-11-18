# Decisões de arquitetura (ADRs)

Cada ADR registra uma decisão, o contexto em que foi tomada, os números que a justificaram e o que ela custa. Para uma decisão nova, copie o [modelo](0000-template.md) e use o próximo número.

| ADR | Decisão | Status | Evidência |
|---|---|---|---|
| [0001](0001-corpus-langgraph-docs.md) | Corpus: `src/oss` do `langchain-ai/docs`, commit fixado | Aceito | 743 documentos, 10.800 chunks |
| [0002](0002-pgvector.md) | pgvector (HNSW) em vez de Qdrant | Aceito | `ef_search=200` empata com a busca exata (`eval/results/retrieval_bench.json`) |
| [0003](0003-embeddings.md) | Embeddings e LLMs pela API da OpenAI | Aceito | custo da indexação: ~US$ 0,04 |
| [0004](0004-langfuse.md) | Langfuse self-hosted para tracing | Aceito | trace por request com o mesmo id do stream SSE |
| [0005](0005-busca-lexical-bm25.md) | BM25 em memória, fundido à busca densa por RRF | Aceito | recall@5: híbrido 0,745 × denso 0,687; FTS do Postgres 0,212 |
| [0006](0006-reranker.md) | Reranker local jina-turbo | Aceito com ressalva | +1,1 ponto de recall@5 sobre o híbrido, +2,3 s em CPU |
| [0007](0007-cache-semantico-theta.md) | Cache exato sempre; semântico só com θ = 0,98 | Aceito | vizinhas chegam a 0,979 de similaridade |
| [0008](0008-mapa-umap.md) | Mapa UMAP na ingestão; pergunta posicionada pelos vizinhos | Aceito | UMAP em 80–140 s, uma vez por ingestão |

## O que ainda falta medir

- **ADR-0006**: o ganho do reranker foi medido só no retrieval. Falta o eval de ponta a ponta da v1 (correção e fidelidade da resposta), que parou quando os créditos da API acabaram (issue #35).
- **ADR-0007**: a taxa de acerto do cache no eval da v3 (issue #54).
