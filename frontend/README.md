# Frontend do RAG X-Ray

Interface Next.js que consome o stream SSE do backend (`POST /query/stream`) e desenha cada etapa do agente.

- `components/GraphView.tsx`: topologia real do grafo (`GET /graph/{pipeline}`), com os nós acendendo conforme os eventos chegam.
- `components/TraceTimeline.tsx`: waterfall de latência, tokens e custo por nó, com o custo contrafactual (tudo no modelo large).
- `components/RankingPanel.tsx`: bump chart da posição de cada trecho em denso → BM25 → RRF → rerank.
- `components/RouterExplainer.tsx`: rota, modelo, motivo, cache e groundedness.
- `components/AnswerWithCitations.tsx`: resposta em streaming, citações clicáveis e feedback 👍/👎.

O modo **replay** lê arquivos estáticos em `public/demos/` (gerados por `backend/scripts/record_demos.py` ou `export_demos.py`) e funciona sem backend e sem LLM. É o modo padrão do demo público.

```bash
npm install
npm run dev -- --port 3001
```

`NEXT_PUBLIC_API_URL` aponta para o backend (padrão `http://localhost:8000`).
