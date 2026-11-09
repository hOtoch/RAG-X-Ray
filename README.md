# RAG X-Ray

Um agente de RAG "de vidro": você faz uma pergunta e vê, em tempo real, cada etapa que o agente executa (cache, roteamento, retrieval, rerank, geração e checagem de groundedness), com tracing completo e um eval com números medidos de qualidade, custo e latência.

> 🚧 Em construção. O projeto está sendo desenvolvido em fases; veja o [roadmap](docs/ROADMAP.md) e as [milestones](https://github.com/hOtoch/RAG-X-Ray/milestones).

- **Corpus:** documentação do LangGraph
- **Stack:** Python, FastAPI, LangGraph, Postgres + pgvector, Redis, Langfuse, Next.js + React Flow
- **Briefing completo:** [context.md](context.md)

## Estrutura

```
backend/    API FastAPI, grafo LangGraph e pipeline de ingestão
eval/       golden dataset, scripts de eval e resultados versionados
frontend/   interface Next.js
docs/       roadmap e ADRs
scripts/    utilitários (ex.: geração do roadmap no GitHub)
```

## Licença

[MIT](LICENSE)
