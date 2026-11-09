# Roadmap do RAG X-Ray

Gerado por `scripts/roadmap.py`. Cada milestone é uma fase do `context.md`; cada item é uma issue no GitHub.

## F1 · Fundação + RAG ingênuo (v0)

Infra local, ingestão dos docs do LangGraph, RAG ingênuo, golden dataset e run_eval.py. Pronto: `docker compose up` sobe tudo e o eval v0 está salvo em eval/results/.

- [ ] Inicializar o repositório e a estrutura de pastas `infra`
- [ ] ADR-0001: domínio e corpus (docs do LangGraph) `docs` `adr`
- [ ] ADR-0002 e 0003: pgvector e modelo de embeddings `docs` `adr` `retrieval`
- [ ] Esqueleto do backend: pyproject, FastAPI e /health `infra`
- [ ] docker compose com Postgres + pgvector e Redis `infra`
- [ ] config.py com Settings via env `infra` `llm`
- [ ] Fábrica de LLM por nível com contagem de tokens e custo `llm`
- [ ] Ingestão 1/3: baixar e carregar os docs do LangGraph `ingest`
- [ ] Ingestão 2/3: chunking por estrutura do markdown `ingest`
- [ ] Ingestão 3/3: embeddings e indexação no pgvector `ingest` `retrieval`
- [ ] Configs nomeadas de pipeline (v0, v1, ...) `eval` `agent`
- [ ] Grafo v0: retrieve denso → generate com citações `agent`
- [ ] Rota POST /query (síncrona) `agent`
- [ ] Golden dataset 1/2: geração sintética de perguntas `eval`
- [ ] Golden dataset 2/2: perguntas manuais, armadilhas e fora de escopo `eval`
- [ ] Métricas de retrieval: recall@k, MRR, nDCG `eval` `retrieval`
- [ ] run_eval.py: rodar uma config e salvar o resultado `eval`
- [ ] CI básica: lint e testes do backend `ci`
- [ ] Release v0: primeiro eval e tabela de ablação `eval` `docs`

## F2 · Tracing e contrato de eventos

Langfuse ligado, todo request vira um trace com spans por nó, e o stream SSE segue o contrato da seção 6. Pronto: todo request aparece no Langfuse com spans por nó.

- [ ] ADR-0004: Langfuse self-hosted `docs` `adr` `tracing`
- [ ] Langfuse no docker compose `infra` `tracing`
- [ ] Trace por request com spans por nó `tracing`
- [ ] Tokens, custo, modelo e rota nos spans `tracing` `llm`
- [ ] Contrato de eventos SSE em Pydantic `tracing`
- [ ] POST /query em streaming SSE `tracing` `agent`
- [ ] Persistir eventos por request e GET /traces/{id} `tracing`
- [ ] Spans com as convenções GenAI do OpenTelemetry `tracing` `bonus`

## F3 · Retrieval híbrido + rerank (v1)

BM25 + denso + RRF, reranker cross-encoder e orçamento de contexto. Pronto: eval v1 mostra o ganho de recall e faithfulness.

- [ ] BM25 no Postgres `retrieval` `ingest`
- [ ] Reciprocal Rank Fusion (RRF) `retrieval`
- [ ] Nó retrieve híbrido `retrieval` `agent`
- [ ] Nó rerank com cross-encoder `retrieval` `agent`
- [ ] compress_context: dedupe e orçamento de tokens `agent` `llm`
- [ ] Testes unitários de retrieval `retrieval`
- [ ] Calibração do LLM-as-judge (kappa de Cohen) `eval`
- [ ] Release v1: eval do híbrido + rerank `eval` `docs`

## F4 · Router + cascata + groundedness (v2)

Classificação de rota, cascata de modelos, Corrective RAG, checagem de groundedness e guardrails. Pronto: eval v2 mostra a redução de custo e a acurácia do router.

- [ ] Rota esperada no golden dataset `eval`
- [ ] Nó router com saída estruturada `agent` `llm`
- [ ] Nós direct_answer e refuse `agent`
- [ ] Arestas condicionais por rota `agent`
- [ ] decompose + retrieve em paralelo (Send API) `agent` `retrieval`
- [ ] grade_docs + reescrita de query (Corrective RAG) `agent`
- [ ] groundedness_check com escalada ou "não sei" `agent` `llm`
- [ ] Guardrails: budget_usd_remaining no estado `agent` `llm`
- [ ] Custo contrafactual (tudo no modelo large) `llm` `eval`
- [ ] Prompt caching do provedor `llm`
- [ ] Teste de integração do grafo inteiro `agent`
- [ ] Release v2: eval do router + cascata `eval` `docs` `adr`

## F5 · Cache exato + semântico (v3)

Cache em camadas com θ ajustado por dados. Pronto: eval v3 com taxa de acerto e de falso positivo reportadas.

- [ ] Nó cache_exact `cache` `agent`
- [ ] Nó cache_semantic com RedisVL `cache` `agent`
- [ ] Nó cache_write (só se grounded) `cache` `agent`
- [ ] Dataset de pares para ajustar θ `cache` `eval`
- [ ] Calibração de θ: acerto × falso positivo `cache` `eval` `adr`
- [ ] Testes unitários de cache `cache`
- [ ] Release v3: eval com cache `eval` `docs`

## F6 · Frontend: execução ao vivo

Grafo ao vivo, linha do tempo, painel de ranking, explicação do router e citações. Pronto: uma pergunta real percorre a interface do início ao fim via SSE.

- [ ] Setup do frontend Next.js `frontend` `infra`
- [ ] Tipos do contrato de eventos e cliente SSE `frontend` `tracing`
- [ ] GET /graph: topologia do LangGraph `agent` `frontend`
- [ ] GraphView: grafo ao vivo com React Flow `frontend`
- [ ] TraceTimeline: waterfall de latência, tokens e custo `frontend`
- [ ] RankingPanel: bump chart BM25 → denso → RRF → rerank `frontend` `retrieval`
- [ ] RouterExplainer `frontend`
- [ ] AnswerWithCitations e visualização da fonte `frontend`
- [ ] Página principal integrando tudo `frontend`

## F7 · Mapa UMAP, replay e comparação

Mapa de embeddings, modo replay (padrão do demo público) e modo comparação. Pronto: o demo roda em replay sem chamar LLM.

- [ ] UMAP do corpus na ingestão `ingest`
- [ ] Evento embedding_map com a query projetada `agent` `retrieval`
- [ ] EmbeddingMap: nuvem de pontos interativa `frontend`
- [ ] GET /replay/{id}: reemitir eventos gravados `tracing`
- [ ] Curadoria dos traces de demo `eval` `tracing`
- [ ] Modo replay como padrão no frontend `frontend`
- [ ] CompareMode: duas configs lado a lado `frontend` `eval`

## F8 · Eval na CI, guardrails e deploy

Gate de regressão na CI, rate limit, teto de gasto, feedback online e deploy. Pronto: PR com regressão falha na CI; demo no ar com budget diário.

- [ ] Subconjunto estratificado do eval para CI `eval` `ci`
- [ ] Workflow eval.yml com gate de regressão `ci` `eval`
- [ ] Rate limit por IP `deploy`
- [ ] Teto de gasto diário em USD `deploy` `llm`
- [ ] Feedback 👍/👎 ligado ao trace `frontend` `tracing` `eval`
- [ ] Deploy do backend e dados `deploy` `adr`
- [ ] Deploy do frontend na Vercel `deploy` `frontend`

## F9 · Narrativa e divulgação

README com tabela de ablação, ADRs, failures.md, vídeo e post. Pronto: um recrutador entende o projeto em 2 minutos.

- [ ] README final `docs`
- [ ] docs/failures.md `docs` `eval`
- [ ] Revisão dos ADRs `docs` `adr`
- [ ] Vídeo de 90 segundos `docs`
- [ ] Post de divulgação `docs`
