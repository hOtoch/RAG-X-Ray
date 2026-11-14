# RAG X-Ray

Um agente de RAG "de vidro": você faz uma pergunta e vê, em tempo real, cada etapa que o agente executa (cache, roteamento, retrieval, rerank, geração e checagem de groundedness), com tracing completo e um eval com números medidos de qualidade, custo e latência.

> 🚧 Em construção. O projeto é desenvolvido em fases; veja o [roadmap](docs/ROADMAP.md) e as [milestones](https://github.com/hOtoch/RAG-X-Ray/milestones).

- **Corpus:** documentação open source da LangChain (LangGraph, LangChain, Deep Agents e integrações): 743 páginas, 10.800 chunks ([ADR-0001](docs/adr/0001-corpus-langgraph-docs.md))
- **Stack:** Python, FastAPI, LangGraph, Postgres + pgvector, Redis, Langfuse, Next.js + React Flow
- **Briefing completo:** [context.md](context.md)

## Resultados (ablação)

Cada linha é uma versão do pipeline rodada contra o mesmo [golden dataset](eval/README.md) de 160 perguntas. Todos os números são medidos; o arquivo de cada execução está em [`eval/results/`](eval/results).

| Versão | Faithfulness | Correção | Recall@5 (chunk) | Recall@5 (página) | Custo/pergunta | p95 |
|---|---|---|---|---|---|---|
| v0 RAG ingênuo | 0,984 | 0,869 | 0,687 | 0,889 | US$ 0,00081 | 5,2 s |

Como ler:

- **Faithfulness** e **Correção** vêm de um juiz LLM (`gpt-4.1`) que ainda não foi calibrado contra rótulos humanos. A calibração está planejada na F3.
- **Recall@5 (chunk)** subestima o retrieval, porque o gabarito marca só os chunks que originaram a pergunta. **Recall@5 (página)** conta acerto quando a página certa foi recuperada.
- Na v0, 87% das perguntas armadilha e fora de escopo terminaram em recusa, que é o comportamento esperado; 4% das perguntas respondíveis foram recusadas por engano.
- Modelos da v0: geração com `gpt-4.1-mini`, embeddings com `text-embedding-3-small`.

## Rodando localmente

Pré-requisitos: Docker, [uv](https://docs.astral.sh/uv/) e uma chave de API da OpenAI.

```bash
cp .env.example .env          # preencha OPENAI_API_KEY
docker compose up -d          # Postgres + pgvector, Redis e Langfuse
cd backend && uv sync
uv run python -m ingest       # baixa, divide e indexa o corpus (~US$ 0,05)
uv run uvicorn app.main:app --reload
```

Depois:

```bash
curl -N -X POST localhost:8000/query/stream -H "Content-Type: application/json" -d '{"question": "How do I pause a LangGraph graph for human approval?"}'
```

- API: `http://localhost:8000/docs`
- Langfuse: `http://localhost:3000` (usuário `dev@ragxray.local`, senha `ragxray-local`, só para uso local)
- Testes: `cd backend && uv run pytest`
- Eval: `uv run --project backend python -m eval.run_eval --pipeline v0` (na raiz do repositório)

## Estrutura

```
backend/    API FastAPI (app/), grafo LangGraph (app/graph/) e ingestão (ingest/)
eval/       golden dataset, juiz, run_eval.py e resultados versionados
frontend/   interface Next.js (F6)
configs/    versões do pipeline e tabela de preços dos modelos
docs/       roadmap, ADRs e contrato de eventos
```

## Licença

[MIT](LICENSE)
