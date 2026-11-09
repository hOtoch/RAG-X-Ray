# RAG X-Ray: briefing de implementação

> Documento de contexto para um agente de código implementar o projeto do zero.
> Leia tudo antes de começar. As seções marcadas como **DECISÃO PENDENTE** precisam ser confirmadas com o usuário antes de escrever código que dependa delas.

---

## 0. Objetivo do projeto

Um projeto de portfólio para vagas de **Engenheiro de IA**: um **agente de RAG "de vidro"**. O usuário digita uma pergunta e vê, em tempo real, cada etapa que o agente executa: cache, roteamento, retrieval, rerank, geração, checagem de groundedness e resposta final com citações. Junto disso, o projeto mostra o tracing completo e um sistema de eval com números reais.

**Princípio guia:** o visual atrai, mas o que convence o recrutador são **métricas** (qualidade × custo × latência) e a **narrativa de engenharia**: medir, mudar, medir de novo. Toda decisão de arquitetura precisa ter um número no eval que a justifique. Não crie nó no grafo só para enfeitar.

Nomes sugeridos: **RAG X-Ray** ou **Glassbox RAG**.

---

## 1. Domínio e corpus

**DECISÃO PENDENTE: confirmar com o usuário.** Não use "chat com PDF" genérico. Opções:

| Opção | Por que é boa |
|---|---|
| **Documentação técnica grande** (ex.: docs do LangGraph ou do Kubernetes) | Perguntas multi-hop naturais, respostas verificáveis, recrutador entende na hora |
| **Normas do Banco Central, da CVM ou da LGPD** | Domínio brasileiro pouco explorado em portfólio, texto denso, citação obrigatória |
| **Papers do arXiv sobre RAG e agentes** | O assunto do projeto vira o próprio dataset |

**Recomendação:** documentação técnica ou normas regulatórias. O critério decisivo é ser possível dizer se uma resposta está certa, porque é isso que torna o eval viável.

Requisitos do corpus:
- De centenas a poucos milhares de documentos. Grande o bastante para o retrieval importar e pequeno o bastante para indexar barato.
- Licença que permita redistribuir ou pelo menos indexar.
- Metadados úteis (título, seção, URL ou fonte) para as citações.

---

## 2. Stack

| Camada | Escolha | Observação |
|---|---|---|
| Backend | **Python 3.12+, FastAPI, LangGraph, LangChain** | Streaming via **SSE** |
| Vetores | **Postgres + pgvector** (padrão) ou Qdrant | Um banco só simplifica a operação |
| Busca lexical | BM25 (Postgres full-text ou `rank_bm25`) | Para o retrieval híbrido |
| Cache | **Redis** (RedisVL tem cache semântico pronto) | Cache exato e semântico |
| Reranker | Cross-encoder open source local (ex.: família bge-reranker) ou uma API | Configurável |
| LLMs | Três níveis configuráveis por env: `small`, `medium`, `large` | Provedor plugável e nomes de modelo **nunca** hardcoded |
| Tracing | **Langfuse auto-hospedado** (padrão) ou LangSmith | Bônus: spans com as convenções GenAI do OpenTelemetry |
| Eval | **RAGAS** e/ou **DeepEval** + scripts próprios | Roda local e na CI |
| Frontend | **Next.js + React Flow** + lib de gráficos | Consome o stream SSE |
| Infra local | `docker compose` (Postgres, Redis, Langfuse) | Um comando sobe tudo |
| Deploy | Backend no Fly, Railway ou Render; frontend na Vercel | Rate limit e teto de gasto diário obrigatórios |

> Antes de usar qualquer API de LangGraph ou LangChain, confira a documentação da versão instalada. As APIs mudam com frequência.

---

## 3. Estrutura sugerida do repositório

```
rag-xray/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI: rotas /query (SSE), /traces, /replay, /health
│   │   ├── config.py            # Settings via env (modelos por nível, limiares, budgets)
│   │   ├── graph/
│   │   │   ├── state.py         # Schema do estado do LangGraph
│   │   │   ├── build.py         # Montagem do grafo (nós + arestas condicionais)
│   │   │   └── nodes/           # Um arquivo por nó (cache, router, retrieve, rerank, ...)
│   │   ├── retrieval/           # BM25, denso, RRF, reranker
│   │   ├── cache/               # Cache exato + semântico
│   │   ├── llm/                 # Fábrica de modelos por nível, contagem de tokens e custo
│   │   ├── events/              # Contrato de eventos SSE (seção 6)
│   │   └── tracing/             # Integração com Langfuse/OTel
│   ├── ingest/                  # Pipeline de ingestão: carregar, chunkar, embeddar, indexar
│   ├── tests/
│   └── pyproject.toml
├── eval/
│   ├── datasets/                # golden.jsonl (seção 7)
│   ├── generate_synthetic.py    # Geração de perguntas sintéticas
│   ├── run_eval.py              # Roda o dataset contra uma config e salva resultados
│   ├── judge_calibration/       # Rótulos humanos + cálculo de concordância
│   └── results/                 # Um JSON por experimento (versionado)
├── frontend/                    # Next.js
│   ├── app/
│   └── components/
│       ├── GraphView/           # React Flow com o grafo ao vivo
│       ├── EmbeddingMap/        # Projeção UMAP 2D
│       ├── RankingPanel/        # Bump chart BM25 → denso → RRF → rerank
│       ├── TraceTimeline/       # Waterfall de latência, tokens e custo
│       ├── RouterExplainer/
│       ├── AnswerWithCitations/
│       └── CompareMode/
├── docs/
│   ├── adr/                     # Architecture Decision Records
│   └── failures.md              # Onde o sistema falha e por quê
├── .github/workflows/eval.yml   # Eval na CI com gate de regressão
├── docker-compose.yml
└── README.md
```

---

## 4. Arquitetura do agente (LangGraph)

### 4.1 Fluxo

```
query
  │
  ▼
[cache exato] ──hit──► resposta
  │ miss
  ▼
[cache semântico] ──hit (sim > θ)──► resposta
  │ miss
  ▼
[router / classificador]  ← modelo small
  ├─ trivial/conversa ──► modelo small responde direto
  ├─ fora de escopo   ──► recusa educada
  ├─ RAG simples      ──► retrieve ─► rerank ─► gerar (modelo medium)
  └─ multi-hop        ──► decompor query ─► retrieve em paralelo (Send API)
                            ─► avaliar docs ─► (reescrever e tentar de novo, máx. N)
                            ─► gerar (modelo large)
  ▼
[checagem de groundedness] ──falhou──► escalar de modelo ou responder "não sei"
  ▼
resposta com citações + grava no cache
```

### 4.2 Nós

| Nó | Responsabilidade | Modelo |
|---|---|---|
| `cache_exact` | Hash da query normalizada (minúsculas, espaços, pontuação) | nenhum |
| `cache_semantic` | Embedding da query, vizinho mais próximo no cache, hit se `sim > θ` | embedding |
| `router` | Classifica em `trivial`, `out_of_scope`, `simple_rag` ou `multi_hop` com saída estruturada (JSON) e motivo curto | small |
| `direct_answer` | Responde perguntas triviais sem retrieval | small |
| `refuse` | Recusa educada para perguntas fora de escopo | small ou template |
| `decompose` | Quebra a pergunta multi-hop em sub-perguntas | small/medium |
| `retrieve` | Retrieval híbrido: BM25 + denso combinados por **RRF** | nenhum |
| `rerank` | Cross-encoder e corte em top-k | reranker |
| `grade_docs` | Avalia se os docs são relevantes; se não forem, reescreve a query e repete (Corrective RAG), até `max_retries` | small |
| `compress_context` | Remove duplicatas, comprime e respeita o orçamento de tokens da rota | nenhum ou small |
| `generate` | Gera a resposta com citações `[n]` ligadas aos chunks | medium/large conforme a rota |
| `groundedness_check` | Verifica se cada afirmação está nos chunks; se falhar, escala de modelo uma vez ou responde "não sei" | small |
| `cache_write` | Grava a resposta nos dois caches (só se a groundedness passou) | nenhum |

### 4.3 Estado do grafo (esboço)

```python
class AgentState(TypedDict):
    request_id: str
    query: str
    normalized_query: str
    route: Literal["trivial", "out_of_scope", "simple_rag", "multi_hop"] | None
    route_reason: str | None
    sub_queries: list[str]
    candidates: list[ScoredChunk]       # com scores BM25, denso e RRF
    reranked: list[ScoredChunk]         # com score do reranker
    context: list[Chunk]                # após compressão
    answer: str | None
    citations: list[Citation]
    grounded: bool | None
    model_tier_used: Literal["small", "medium", "large"] | None
    retries: int
    cache_hit: Literal["exact", "semantic"] | None
    usage: list[NodeUsage]              # tokens in/out, custo e latência por nó
    budget_usd_remaining: float
```

### 4.4 Técnicas de redução de custo e tokens

Cada uma deve aparecer como um experimento no eval (seção 7.4):

- **Cache em camadas** (exato + semântico). **Meça a taxa de falso positivo do cache semântico** e ajuste θ com dados. Isso é um diferencial.
- **Prompt caching do provedor:** system prompt e definições de tools fixos no início do prompt, com a parte variável no fim.
- **Roteamento de modelo com cascata:** começa no modelo barato e escala só se a confiança ou a groundedness ficar baixa. Registre o custo contrafactual: quanto custaria usar só o modelo `large`.
- **Retrieval híbrido + rerank:** 4 chunks bons custam menos que 20 médios.
- **Orçamento de contexto por rota:** dedupe, compressão e teto de tokens de entrada.
- **Guardrails de laço:** `max_retries` e `budget_usd_remaining` no estado. Interrompa quando estourar.
- **Saída estruturada** em todos os nós internos (router, graders): JSON curto, sem texto livre.

---

## 5. Experiência visual (frontend)

Por ordem de prioridade:

1. **Grafo ao vivo (React Flow):** a topologia real do LangGraph. Os nós acendem conforme os eventos chegam, e as arestas mostram o caminho escolhido pelo router.
2. **Linha do tempo estilo trace:** waterfall com latência, tokens in/out e custo por nó, mais um contador de custo acumulado.
3. **Painel de ranking:** bump chart com a posição de cada chunk em BM25 → denso → RRF → rerank. Mostra por que o reranker importa.
4. **Explicação do router:** por exemplo, "Rota: multi-hop · Modelo: large · Motivo: 2 entidades + comparação · Cache: miss".
5. **Resposta com citações clicáveis** que abrem o documento de origem com o trecho destacado.
6. **Mapa de embeddings 2D (UMAP):** o corpus como nuvem de pontos, com a query aparecendo e os chunks recuperados acendendo. A projeção do corpus é **pré-calculada** na ingestão; a query é projetada na hora (`umap.transform`).
7. **Modo comparação:** a mesma pergunta em duas configs lado a lado (ex.: RAG ingênuo × pipeline completo).
8. **Modo replay:** traces gravados que o visitante reproduz passo a passo **sem chamar LLM**. É o modo padrão do demo público: nunca quebra e não gera custo.

---

## 6. Contrato de eventos SSE (backend → frontend)

Todo evento tem um envelope comum. O mesmo stream alimenta a interface e é persistido para o replay.

```jsonc
{
  "request_id": "uuid",
  "seq": 12,                    // ordem monotônica
  "ts": "2026-11-09T12:00:00.123Z",
  "node": "rerank",             // id do nó no grafo (igual ao id no React Flow)
  "type": "node_start" | "node_end" | "token" | "data" | "error" | "done",
  "payload": { ... }
}
```

Payloads por tipo:
- `node_start`: `{}`
- `node_end`: `{ "latency_ms", "tokens_in", "tokens_out", "cost_usd", "model" }`
- `token`: `{ "text" }` (streaming da resposta final)
- `data`: dados específicos do nó, por exemplo:
  - `router`: `{ "route", "reason", "confidence" }`
  - `retrieve`: `{ "chunks": [{ "id", "bm25_rank", "dense_rank", "rrf_rank", "score" }] }`
  - `rerank`: `{ "chunks": [{ "id", "rerank_rank", "score" }] }`
  - `cache_semantic`: `{ "hit", "similarity", "threshold" }`
  - `embedding_map`: `{ "query_xy": [x, y], "hit_ids": [...] }`
  - `groundedness_check`: `{ "grounded", "unsupported_claims": [...] }`
- `done`: `{ "answer", "citations", "total_cost_usd", "total_latency_ms", "counterfactual_cost_usd" }`

Implementação: `graph.astream_events(...)` ou `stream_mode=["updates", "messages", "custom"]`, com eventos `custom` emitidos pelos nós para os payloads de `data`. Confira a API na versão instalada.

---

## 7. Tracing e eval

### 7.1 Tracing
- Langfuse (auto-hospedado no `docker compose`) ou LangSmith, ligado **desde o primeiro dia**.
- Cada request vira um trace com spans por nó, incluindo tokens, custo, modelo e rota.
- `request_id` compartilhado entre o trace e o stream SSE.

### 7.2 Golden dataset

Formato `eval/datasets/golden.jsonl`:

```jsonc
{
  "id": "q-0001",
  "question": "...",
  "type": "simple" | "multi_hop" | "out_of_scope" | "trap",
  "reference_answer": "...",
  "relevant_doc_ids": ["doc-12#chunk-3", "doc-40#chunk-1"],
  "source": "synthetic" | "manual"
}
```

- De 100 a 200 perguntas, parte geradas sinteticamente (RAGAS ou DeepEval) e parte curadas à mão.
- Inclua perguntas **armadilha** (premissa falsa) e **fora de escopo**.
- Mantenha um conjunto separado para ajustar o limiar θ do cache semântico: pares de perguntas parecidas que devem ou não compartilhar a resposta.

### 7.3 Métricas
- **Retrieval:** recall@k, MRR, nDCG.
- **Geração:** faithfulness, relevância da resposta, precisão de citação.
- **Operacionais:** custo/query, latência p50/p95, taxa de acerto e de falso positivo do cache, distribuição de rotas, acurácia do router.
- **LLM-as-judge calibrado:** cerca de 50 respostas rotuladas à mão em `eval/judge_calibration/`, com a concordância juiz × humano (kappa de Cohen) reportada no README.

### 7.4 Experimentos (ablação)

Cada versão do pipeline é uma config nomeada. O `run_eval.py` recebe a config e salva o resultado em `eval/results/<versao>.json`. A tabela final do README:

| Versão | Faithfulness | Recall@5 | Custo/query | p95 |
|---|---|---|---|---|
| v0 RAG ingênuo | … | … | … | … |
| v1 + híbrido + rerank | … | … | … | … |
| v2 + router + cascata | … | … | … | … |
| v3 + cache | … | … | … | … |

> Preencha com números **medidos**. Nunca invente valores.

### 7.5 CI
- `.github/workflows/eval.yml` roda o eval num subconjunto a cada PR e **falha** se faithfulness ou recall cair além de uma tolerância configurada, ou se o custo/query subir além do limite.

### 7.6 Eval online
- Botões 👍/👎 na interface, ligados ao trace.
- Amostragem de traces reais para o juiz, com os casos de falha voltando para o golden dataset.

---

## 8. Fases de implementação

Cada fase termina com um commit ou release e, quando couber, uma nova linha na tabela de ablação.

| Fase | Entrega | Critério de pronto |
|---|---|---|
| **1** | Infra local + ingestão + RAG ingênuo + golden dataset + `run_eval.py` | `docker compose up` sobe tudo; eval v0 salvo em `eval/results/` |
| **2** | Tracing ligado | Todo request aparece no Langfuse/LangSmith com spans por nó |
| **3** | Retrieval híbrido (BM25 + denso + RRF) + rerank | Eval v1 mostra o ganho de recall e faithfulness |
| **4** | Router + cascata de modelos + groundedness check | Eval v2 mostra a redução de custo; acurácia do router medida |
| **5** | Cache exato + semântico com θ ajustado | Eval v3 com taxa de acerto e de falso positivo reportadas |
| **6** | Frontend: grafo ao vivo, linha do tempo, painel de ranking, explicação do router, citações | Uma pergunta real percorre a interface do início ao fim via SSE |
| **7** | Mapa UMAP, modo replay, modo comparação | Demo público roda em replay sem chamar LLM |
| **8** | Eval na CI, rate limit, teto de gasto, deploy | PR com regressão falha na CI; demo no ar com budget diário |
| **9** | README com tabela de ablação, ADRs, `docs/failures.md`, vídeo de 90s, post | Recrutador entende o projeto em 2 minutos |

---

## 9. Regras para o agente de código

- **Confirme o domínio** (seção 1) com o usuário antes da fase 1.
- **Nomes de modelo e provedor vêm de configuração**, nunca hardcoded. Use três níveis: `small`, `medium`, `large`.
- **Toda feature nova de arquitetura vem com um experimento no eval.** Sem número, sem merge.
- **Nunca invente métricas** no README ou na interface. Só valores medidos.
- **Custo controlado:** rate limit por IP, teto diário em USD e modo replay como padrão no demo público.
- **Segredos** só em `.env` (com um `.env.example` versionado). Nada de chave no código.
- **Testes:** unitários para retrieval, RRF, cache e router (com LLM mockado); um teste de integração do grafo inteiro.
- **ADRs curtos** em `docs/adr/` para cada escolha relevante (pgvector × Qdrant, Langfuse × LangSmith, limiar θ, política de cascata).
- **Documente as falhas** em `docs/failures.md`: onde o sistema erra e por quê. Isso conta pontos com quem avalia.
