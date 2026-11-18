# RAG X-Ray

Um agente de RAG "de vidro": você faz uma pergunta e vê, em tempo real, cada etapa que o agente executa (cache, roteamento, retrieval, rerank, geração e checagem de groundedness), com tracing completo e um eval com números medidos de qualidade, custo e latência.

> 🚧 Em construção. O projeto é desenvolvido em fases; veja o [roadmap](docs/ROADMAP.md) e as [milestones](https://github.com/hOtoch/RAG-X-Ray/milestones).

- **Corpus:** documentação open source da LangChain (LangGraph, LangChain, Deep Agents e integrações): 743 páginas, 10.800 chunks ([ADR-0001](docs/adr/0001-corpus-langgraph-docs.md))
- **Stack:** Python, FastAPI, LangGraph, Postgres + pgvector, Redis, Langfuse, Next.js + React Flow
- **Briefing completo:** [context.md](context.md)

## O agente

```
cache exato ─ hit ─► resposta
  └ miss ─► cache semântico (θ = 0,98) ─ hit ─► resposta
              └ miss ─► router (modelo small)
                          ├ conversa ───────► resposta direta (small)
                          ├ fora de escopo ─► recusa por template (sem LLM)
                          ├ RAG simples ────► retrieve híbrido ─► rerank ─► comprimir ─► gerar (medium)
                          └ multi-hop ──────► decompor ─► retrieve ×N em paralelo ─► juntar ─► rerank
                                               ─► avaliar trechos ─(insuficiente)─► nova busca (máx. 2)
                                               ─► comprimir ─► gerar (large)
gerar ─► groundedness ─ ok ─► grava no cache ─► resposta com citações
                      ├ falhou ─► sobe um nível de modelo e gera de novo (uma vez)
                      └ falhou de novo ─► "não sei"
```

O grafo é montado a partir de uma config (`configs/pipelines.yaml`), e cada versão do pipeline liga ou desliga features. A interface desenha a topologia real do grafo compilado.

## Resultados

Todos os números são medidos, contra o mesmo [golden dataset](eval/README.md) de 160 perguntas. Os arquivos de cada execução estão em [`eval/results/`](eval/results).

### Ablação de ponta a ponta

| Versão | Faithfulness | Correção | Recall@5 (chunk) | Recall@5 (página) | Custo/pergunta | p95 |
|---|---|---|---|---|---|---|
| v0 RAG ingênuo | 0,984 | 0,869 | 0,687 | 0,889 | US$ 0,00081 | 5,2 s |
| v1-hybrid (parcial, 58 perguntas simples) | 0,998 | 0,948 | 0,828 | — | US$ 0,00090 | — |
| v1, v2, v3 | não medido | | | | | |

- A linha da v1-hybrid é parcial: a conta da OpenAI ficou sem créditos na pergunta 58. Nas mesmas 58 perguntas, a v0 teve faithfulness 0,996, correção 0,888 e recall@5 0,793. A comparação é justa, mas cobre só perguntas simples.
- As versões v1 (com rerank), v2 (router, cascata, groundedness) e v3 (cache) estão implementadas e testadas, mas o eval delas depende de créditos na API.
- **Faithfulness** e **Correção** vêm de um juiz LLM (`gpt-4.1`) ainda não calibrado contra rótulos humanos.
- **Recall@5 (chunk)** subestima o retrieval, porque o gabarito marca só os chunks que originaram a pergunta.

### Retrieval (sem LLM, 130 perguntas com gabarito)

| Estratégia | Recall@5 | Recall@5 (página) | MRR | Latência p50 |
|---|---|---|---|---|
| Densa (HNSW) | 0,687 | 0,889 | 0,607 | 135 ms |
| BM25 | 0,684 | 0,923 | 0,605 | 33 ms |
| Full-text do Postgres | 0,212 | 0,481 | 0,209 | 99 ms |
| **Híbrida (densa + BM25, RRF)** | **0,745** | 0,942 | 0,686 | 168 ms |
| Híbrida + rerank jina-turbo | 0,756 | 0,954 | 0,689 | 2,5 s |
| Híbrida + rerank MiniLM-L-12 | 0,717 | 0,942 | 0,704 | 3,9 s |

Detalhes nos ADRs [0005](docs/adr/0005-busca-lexical-bm25.md) e [0006](docs/adr/0006-reranker.md).

### Cache semântico

Com `text-embedding-3-small`, perguntas diferentes de propósito chegam a 0,979 de similaridade, e paráfrases ficam em média em 0,89. Para falso positivo ≤ 2%, θ precisa ser 0,98, e aí o cache semântico quase não acerta paráfrases. A economia do cache vem do cache exato. Curva completa no [ADR-0007](docs/adr/0007-cache-semantico-theta.md).

### Onde falha

[docs/failures.md](docs/failures.md) lista os modos de falha encontrados, com exemplos e causas.

## Rodando localmente

Pré-requisitos: Docker, [uv](https://docs.astral.sh/uv/), Node 20+ e uma chave de API da OpenAI.

```bash
cp .env.example .env          # preencha OPENAI_API_KEY
docker compose up -d          # Postgres + pgvector, Redis e Langfuse
cd backend && uv sync
uv run python -m ingest       # baixa, divide e indexa o corpus (~US$ 0,05)
uv run uvicorn app.main:app --reload
```

Em outro terminal:

```bash
cd frontend && npm install && npm run dev -- --port 3001
```

- Interface: `http://localhost:3001` (modo replay funciona sem backend; modo ao vivo usa a API)
- API: `http://localhost:8000/docs`
- Langfuse: `http://localhost:3000` (usuário `dev@ragxray.local`, senha `ragxray-local`, só para uso local)
- Testes: `cd backend && uv run pytest` (85 testes, sem rede e sem LLM real)
- Eval: `uv run --project backend python -m eval.run_eval --pipeline v3` (na raiz do repositório)
- Gravar demos do replay: `cd backend && uv run python scripts/record_demos.py --pipeline v3`

## Estrutura

```
backend/    API FastAPI (app/), grafo LangGraph (app/graph/), ingestão (ingest/) e scripts
eval/       golden dataset, juiz, run_eval, bench de retrieval, calibração do cache e do juiz, gate da CI
frontend/   interface Next.js: grafo ao vivo, linha do tempo, ranking, citações e replay
configs/    versões do pipeline e tabela de preços dos modelos
docs/       roadmap, ADRs, contrato de eventos e modos de falha
```

## Decisões (ADRs)

| ADR | Decisão |
|---|---|
| [0001](docs/adr/0001-corpus-langgraph-docs.md) | Corpus: `src/oss` do repositório `langchain-ai/docs`, commit fixado |
| [0002](docs/adr/0002-pgvector.md) | pgvector em vez de Qdrant |
| [0003](docs/adr/0003-embeddings.md) | Embeddings e LLMs pela OpenAI |
| [0004](docs/adr/0004-langfuse.md) | Langfuse self-hosted para tracing |
| [0005](docs/adr/0005-busca-lexical-bm25.md) | BM25 em memória e RRF |
| [0006](docs/adr/0006-reranker.md) | Reranker local jina-turbo, com ressalva |
| [0007](docs/adr/0007-cache-semantico-theta.md) | Cache exato sempre; semântico só com θ = 0,98 |

## Licença

[MIT](LICENSE)
