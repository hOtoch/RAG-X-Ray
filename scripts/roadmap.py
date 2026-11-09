"""Roadmap do RAG X-Ray: milestones e issues.

Uso:
    python scripts/roadmap.py md        # gera docs/ROADMAP.md
    python scripts/roadmap.py gh        # cria labels, milestones e issues no GitHub (idempotente)

Requer `gh auth login` para o modo gh.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = "hOtoch/RAG-X-Ray"

LABELS = {
    "infra": ("0e8a16", "Docker, configuração, setup do repositório"),
    "ingest": ("1d76db", "Pipeline de ingestão do corpus"),
    "retrieval": ("5319e7", "BM25, denso, RRF, reranker"),
    "agent": ("b60205", "Nós e arestas do LangGraph"),
    "llm": ("d93f0b", "Fábrica de modelos, tokens, custo"),
    "cache": ("fbca04", "Cache exato e semântico"),
    "tracing": ("c5def5", "Langfuse, eventos SSE, persistência de traces"),
    "eval": ("006b75", "Dataset, métricas, experimentos"),
    "frontend": ("e99695", "Next.js, React Flow, visualizações"),
    "ci": ("bfd4f2", "GitHub Actions"),
    "deploy": ("f9d0c4", "Deploy, rate limit, teto de gasto"),
    "docs": ("0075ca", "README, ADRs, failures.md, divulgação"),
    "adr": ("7057ff", "Architecture Decision Record"),
    "bonus": ("ededed", "Opcional, se sobrar tempo"),
}


def issue(title: str, labels: list[str], contexto: str, tarefas: list[str], pronto: list[str]) -> dict:
    body = f"## Contexto\n{contexto}\n\n## Tarefas\n"
    body += "\n".join(f"- [ ] {t}" for t in tarefas)
    body += "\n\n## Pronto quando\n"
    body += "\n".join(f"- {p}" for p in pronto)
    return {"title": title, "labels": labels, "body": body}


MILESTONES: list[dict] = [
    # ------------------------------------------------------------------ F1
    {
        "title": "F1 · Fundação + RAG ingênuo (v0)",
        "description": "Infra local, ingestão dos docs do LangGraph, RAG ingênuo, golden dataset e run_eval.py. "
        "Pronto: `docker compose up` sobe tudo e o eval v0 está salvo em eval/results/.",
        "issues": [
            issue(
                "Inicializar o repositório e a estrutura de pastas",
                ["infra"],
                "A pasta ainda não é um repositório próprio. Criar o esqueleto da seção 3 do context.md.",
                [
                    "`git init`, remote `origin` apontando para hOtoch/RAG-X-Ray, branch `main`",
                    "Criar `backend/`, `eval/`, `frontend/`, `docs/adr/` com `.gitkeep` onde precisar",
                    "`.gitignore` (Python, Node, `.env`, caches de modelo, dados brutos)",
                    "`.env.example` vazio com os grupos de variáveis comentados",
                    "README provisório com o pitch de uma linha e o link para `context.md`",
                    "Escolher e adicionar LICENSE (MIT sugerida)",
                ],
                ["Primeiro commit na `main` do GitHub com a estrutura e sem segredos"],
            ),
            issue(
                "ADR-0001: domínio e corpus (docs do LangGraph)",
                ["docs", "adr"],
                "Registrar a decisão do domínio, que é pré-requisito da ingestão.",
                [
                    "Criar template de ADR em `docs/adr/0000-template.md` (contexto, decisão, alternativas, consequências)",
                    "Escrever `0001-corpus-langgraph-docs.md`: por que LangGraph docs, alternativas descartadas, licença (MIT), tamanho estimado",
                    "Definir a fonte exata (pasta `docs/` do repo langchain-ai/langgraph, commit fixo) para o corpus ser reprodutível",
                ],
                ["ADR mergeado com a fonte e o commit do corpus definidos"],
            ),
            issue(
                "ADR-0002 e 0003: pgvector e modelo de embeddings",
                ["docs", "adr", "retrieval"],
                "A Anthropic não oferece API de embeddings, então o modelo de embedding precisa ser escolhido à parte.",
                [
                    "`0002-pgvector.md`: pgvector × Qdrant (um banco só, BM25 no mesmo Postgres)",
                    "`0003-embeddings.md`: comparar opção local (ex.: família bge via sentence-transformers) × API (ex.: Voyage) em custo, latência, dimensão e reprodutibilidade",
                    "Deixar o modelo de embedding configurável por env",
                ],
                ["Dois ADRs mergeados e a variável de embedding listada no `.env.example`"],
            ),
            issue(
                "Esqueleto do backend: pyproject, FastAPI e /health",
                ["infra"],
                "Base do backend em Python 3.12 com ferramentas de qualidade desde o início.",
                [
                    "`backend/pyproject.toml` gerenciado com `uv` (FastAPI, uvicorn, pydantic-settings, pytest, ruff)",
                    "`app/main.py` com `GET /health` retornando versão e status",
                    "Configurar ruff (lint + format) e pytest",
                    "Teste de `/health` com `TestClient`",
                ],
                ["`uv run pytest` e `uv run ruff check` passam; `uvicorn app.main:app` responde em /health"],
            ),
            issue(
                "docker compose com Postgres + pgvector e Redis",
                ["infra"],
                "Um comando sobe a infra local. Langfuse entra na F2.",
                [
                    "`docker-compose.yml` com `pgvector/pgvector` e `redis` (ou `redis-stack` por causa do RedisVL)",
                    "Volumes nomeados e healthchecks",
                    "Script SQL de init habilitando a extensão `vector`",
                    "Variáveis de conexão no `.env.example`",
                    "Seção \"Rodando localmente\" no README",
                ],
                ["`docker compose up -d` deixa os dois serviços healthy e `SELECT * FROM pg_extension` mostra `vector`"],
            ),
            issue(
                "config.py com Settings via env",
                ["infra", "llm"],
                "Todas as escolhas de modelo, limiar e orçamento vêm de configuração. Nada hardcoded.",
                [
                    "`app/config.py` com pydantic-settings: provedor, modelos `small`/`medium`/`large`, modelo de embedding, top-k, limiares, budgets",
                    "Tabela de preço por modelo (USD por 1M tokens in/out) também configurável",
                    "Atualizar `.env.example` com todos os campos",
                    "Teste que carrega Settings a partir de um env de exemplo",
                ],
                ["Nenhum nome de modelo aparece fora de `.env`/`.env.example`"],
            ),
            issue(
                "Fábrica de LLM por nível com contagem de tokens e custo",
                ["llm"],
                "Camada única para pedir um modelo por nível e medir o uso de cada chamada.",
                [
                    "`app/llm/factory.py`: `get_llm(tier)` retornando o chat model do provedor configurado (Anthropic por padrão)",
                    "`app/llm/usage.py`: extrair tokens in/out (e tokens de cache, quando houver) da resposta e calcular custo com a tabela de preço",
                    "Dataclass `NodeUsage` (nó, modelo, tokens in/out, custo, latência)",
                    "Testes com resposta mockada",
                ],
                ["Uma chamada real ao nível `small` retorna texto e um `NodeUsage` com custo > 0"],
            ),
            issue(
                "Ingestão 1/3: baixar e carregar os docs do LangGraph",
                ["ingest"],
                "Transformar a pasta de docs (markdown/notebooks) em documentos com metadados para citação.",
                [
                    "`ingest/fetch.py`: baixar a pasta de docs no commit fixado no ADR-0001",
                    "`ingest/load.py`: ler `.md` (e `.ipynb`, se valer a pena) e extrair título, caminho, URL pública e seções",
                    "Gerar ids estáveis `doc-<slug>`",
                    "Salvar o corpus normalizado em `data/corpus.jsonl` (fora do git)",
                    "Contar documentos e tokens e anotar no ADR-0001",
                ],
                ["`corpus.jsonl` gerado de forma reprodutível, com URL e título em todo documento"],
            ),
            issue(
                "Ingestão 2/3: chunking por estrutura do markdown",
                ["ingest"],
                "Chunks que respeitam headers e blocos de código citam melhor que cortes por tamanho fixo.",
                [
                    "Dividir por headers e depois por tamanho máximo de tokens, com overlap configurável",
                    "Não quebrar blocos de código ao meio",
                    "Ids `doc-<slug>#chunk-<n>` e metadados (título, seção, URL com âncora)",
                    "Testes unitários com um markdown de exemplo",
                ],
                ["Estatística de tamanho dos chunks (min/mediana/max) impressa ao rodar"],
            ),
            issue(
                "Ingestão 3/3: embeddings e indexação no pgvector",
                ["ingest", "retrieval"],
                "Tabela de chunks com vetor e metadados, indexação idempotente.",
                [
                    "Schema/migração da tabela `chunks` (id, doc_id, texto, metadados JSONB, embedding `vector(d)`)",
                    "Índice HNSW para similaridade de cosseno",
                    "Embeddar em lote e fazer upsert (rodar duas vezes não duplica)",
                    "Comando único `uv run python -m ingest` rodando fetch → load → chunk → index",
                ],
                ["Uma busca por similaridade via SQL retorna chunks coerentes para uma pergunta de teste"],
            ),
            issue(
                "Configs nomeadas de pipeline (v0, v1, ...)",
                ["eval", "agent"],
                "Cada experimento de ablação é uma config nomeada. É ela que o run_eval.py recebe.",
                [
                    "Arquivo `configs/pipelines.yaml` com a config `v0` (denso top-k, sem rerank, sem router, sem cache)",
                    "Loader validado com Pydantic e flags por feature (hybrid, rerank, router, cache, ...)",
                    "O grafo é montado a partir da config",
                ],
                ["`load_pipeline(\"v0\")` retorna uma config válida e testada"],
            ),
            issue(
                "Grafo v0: retrieve denso → generate com citações",
                ["agent"],
                "O RAG ingênuo que serve de linha de base para todos os experimentos.",
                [
                    "`graph/state.py` com o subconjunto do `AgentState` necessário agora",
                    "Nó `retrieve` (denso, top-k) e nó `generate` (modelo `medium`) com citações `[n]` ligadas aos chunks",
                    "`graph/build.py` montando o grafo a partir da config",
                    "Registrar `NodeUsage` de cada nó no estado",
                    "Conferir a API do LangGraph na versão instalada antes de escrever",
                ],
                ["Rodar o grafo numa pergunta devolve resposta, citações válidas e usage por nó"],
            ),
            issue(
                "Rota POST /query (síncrona)",
                ["agent"],
                "Expor o grafo por HTTP. O streaming SSE entra na F2.",
                [
                    "`POST /query` recebendo `{question, pipeline}` e devolvendo resposta, citações, custo e latência",
                    "`request_id` (uuid) gerado por request",
                    "Teste de integração com LLM mockado",
                ],
                ["`curl` em /query devolve uma resposta com citações para uma pergunta real"],
            ),
            issue(
                "Golden dataset 1/2: geração sintética de perguntas",
                ["eval"],
                "Parte do dataset é gerada a partir do corpus com RAGAS ou DeepEval.",
                [
                    "`eval/generate_synthetic.py` gerando perguntas simples e multi-hop a partir dos chunks",
                    "Saída no formato da seção 7.2 (`id`, `question`, `type`, `reference_answer`, `relevant_doc_ids`, `source`)",
                    "Revisar por amostragem e descartar perguntas ruins",
                    "Registrar o custo da geração",
                ],
                ["~100 perguntas sintéticas revisadas em `eval/datasets/golden.jsonl`"],
            ),
            issue(
                "Golden dataset 2/2: perguntas manuais, armadilhas e fora de escopo",
                ["eval"],
                "O que separa um eval sério de um eval de fachada.",
                [
                    "~30 perguntas curadas à mão a partir de dúvidas reais sobre LangGraph",
                    "~15 perguntas armadilha (premissa falsa, API que não existe)",
                    "~15 perguntas fora de escopo",
                    "Script de validação do jsonl (schema, ids existentes no índice, sem duplicatas)",
                ],
                ["Dataset com 150–200 perguntas, validado, com a distribuição por `type` no README"],
            ),
            issue(
                "Métricas de retrieval: recall@k, MRR, nDCG",
                ["eval", "retrieval"],
                "Implementação própria e testada, para não depender de caixa-preta.",
                [
                    "`eval/metrics/retrieval.py` com recall@k, MRR e nDCG@k",
                    "Testes unitários com casos calculados à mão",
                ],
                ["Testes passando e funções usadas pelo run_eval.py"],
            ),
            issue(
                "run_eval.py: rodar uma config e salvar o resultado",
                ["eval"],
                "O coração da narrativa medir → mudar → medir.",
                [
                    "`eval/run_eval.py --pipeline v0 [--subset N]` rodando o dataset contra o grafo",
                    "Métricas de geração via RAGAS/DeepEval (faithfulness, relevância da resposta)",
                    "Métricas operacionais: custo/query, latência p50/p95, tokens",
                    "Salvar `eval/results/<pipeline>.json` com config, versão do dataset, commit e métricas por pergunta e agregadas",
                    "Concorrência limitada e retomada se cair no meio",
                ],
                ["Rodar duas vezes com `--subset 10` gera arquivos com o mesmo formato"],
            ),
            issue(
                "CI básica: lint e testes do backend",
                ["ci"],
                "Antes de crescer, garantir que todo PR passa por lint e testes.",
                [
                    "`.github/workflows/ci.yml` com `uv`, `ruff check`, `ruff format --check` e `pytest`",
                    "Testes sem chamar LLM real (mocks)",
                ],
                ["PR de exemplo mostra o check verde"],
            ),
            issue(
                "Release v0: primeiro eval e tabela de ablação",
                ["eval", "docs"],
                "Fechamento da F1.",
                [
                    "Rodar o eval completo da config `v0`",
                    "Commitar `eval/results/v0.json`",
                    "Criar a tabela de ablação no README com a linha v0 (só números medidos)",
                    "Tag `v0.1.0`",
                ],
                ["README mostra a linha v0 com faithfulness, recall@5, custo/query e p95 medidos"],
            ),
        ],
    },
    # ------------------------------------------------------------------ F2
    {
        "title": "F2 · Tracing e contrato de eventos",
        "description": "Langfuse ligado, todo request vira um trace com spans por nó, e o stream SSE segue o contrato da seção 6. "
        "Pronto: todo request aparece no Langfuse com spans por nó.",
        "issues": [
            issue(
                "ADR-0004: Langfuse self-hosted",
                ["docs", "adr", "tracing"],
                "Registrar a escolha de tracing.",
                ["Langfuse × LangSmith: custo, operação, integração com LangGraph, OTel"],
                ["ADR mergeado"],
            ),
            issue(
                "Langfuse no docker compose",
                ["infra", "tracing"],
                "O Langfuse v3 depende de Postgres, ClickHouse, Redis e armazenamento S3 (MinIO). Conferir o compose oficial.",
                [
                    "Adicionar os serviços do Langfuse ao `docker-compose.yml` (reaproveitar Postgres/Redis se fizer sentido)",
                    "Projeto e chaves iniciais via variáveis de ambiente",
                    "Documentar portas e o login no README",
                ],
                ["`docker compose up` sobe o Langfuse acessível em localhost"],
            ),
            issue(
                "Trace por request com spans por nó",
                ["tracing"],
                "Integrar o callback do Langfuse ao grafo.",
                [
                    "Módulo `app/tracing/` criando o handler por request",
                    "`request_id` usado como id (ou metadado) do trace",
                    "Cada nó do LangGraph aparece como span",
                    "Tracing desligável por env (para testes)",
                ],
                ["Uma chamada a /query aparece no Langfuse com um span por nó"],
            ),
            issue(
                "Tokens, custo, modelo e rota nos spans",
                ["tracing", "llm"],
                "O trace precisa mostrar os mesmos números da interface.",
                [
                    "Enviar o `NodeUsage` de cada nó como atributos/metadados do span",
                    "Custo total e pipeline usada no trace",
                    "Conferir que o custo do Langfuse bate com o calculado por nós",
                ],
                ["Custo e tokens por nó visíveis no Langfuse e iguais aos do `/query`"],
            ),
            issue(
                "Contrato de eventos SSE em Pydantic",
                ["tracing"],
                "Seção 6 do context.md: o envelope e os payloads que alimentam a interface e o replay.",
                [
                    "`app/events/schema.py` com o envelope (`request_id`, `seq`, `ts`, `node`, `type`, `payload`)",
                    "Modelos de payload por tipo (`node_start`, `node_end`, `token`, `data`, `error`, `done`)",
                    "Gerar JSON Schema para o frontend reaproveitar",
                    "Testes de serialização",
                ],
                ["JSON Schema exportado em `docs/events.schema.json`"],
            ),
            issue(
                "POST /query em streaming SSE",
                ["tracing", "agent"],
                "Traduzir o stream do LangGraph para o envelope de eventos.",
                [
                    "Usar `astream` com `stream_mode` updates/messages/custom (conferir a API da versão instalada)",
                    "Emitir `node_start`/`node_end` com latência e usage, `token` na geração e `done` no fim",
                    "Nós emitem eventos `custom` para os payloads `data`",
                    "`seq` monotônico e tratamento de erro com evento `error`",
                    "Manter a versão síncrona para o run_eval.py",
                ],
                ["`curl -N` mostra os eventos chegando na ordem certa até o `done`"],
            ),
            issue(
                "Persistir eventos por request e GET /traces/{id}",
                ["tracing"],
                "Base do modo replay da F7.",
                [
                    "Tabela `events` no Postgres (request_id, seq, payload JSONB)",
                    "Gravar o stream enquanto ele é emitido",
                    "`GET /traces/{request_id}` devolvendo os eventos em ordem",
                ],
                ["Eventos de um request podem ser lidos de volta, idênticos aos emitidos"],
            ),
            issue(
                "Spans com as convenções GenAI do OpenTelemetry",
                ["tracing", "bonus"],
                "Bônus: mostrar familiaridade com OTel.",
                ["Atributos `gen_ai.*` (sistema, modelo, tokens) nos spans de LLM"],
                ["Atributos visíveis no trace"],
            ),
        ],
    },
    # ------------------------------------------------------------------ F3
    {
        "title": "F3 · Retrieval híbrido + rerank (v1)",
        "description": "BM25 + denso + RRF, reranker cross-encoder e orçamento de contexto. "
        "Pronto: eval v1 mostra o ganho de recall e faithfulness.",
        "issues": [
            issue(
                "BM25 no Postgres",
                ["retrieval", "ingest"],
                "Busca lexical para pegar nomes de API e termos exatos que o denso erra.",
                [
                    "Comparar Postgres full-text (`tsvector` + `ts_rank`, que não é BM25 puro), extensão BM25 e `rank_bm25` em memória; registrar a escolha num ADR curto",
                    "Coluna/índice na ingestão",
                    "Função `bm25_search(query, k)` com testes",
                ],
                ["Busca por um nome de função exato (ex.: `add_conditional_edges`) traz o chunk certo no top-3"],
            ),
            issue(
                "Reciprocal Rank Fusion (RRF)",
                ["retrieval"],
                "Função pura e testada para combinar rankings.",
                [
                    "`retrieval/rrf.py` com `k` configurável",
                    "Testes unitários com rankings calculados à mão (empates, listas de tamanhos diferentes)",
                ],
                ["Testes passando"],
            ),
            issue(
                "Nó retrieve híbrido",
                ["retrieval", "agent"],
                "Retrieve passa a combinar as duas buscas e guarda os ranks de cada fonte.",
                [
                    "`ScoredChunk` com `bm25_rank`, `dense_rank`, `rrf_rank` e score",
                    "BM25 e denso em paralelo e fusão por RRF",
                    "Evento `data` do retrieve conforme o contrato",
                    "Flag `hybrid` na config de pipeline",
                ],
                ["O evento de retrieve traz os três ranks por chunk"],
            ),
            issue(
                "Nó rerank com cross-encoder",
                ["retrieval", "agent"],
                "4 chunks bons custam menos que 20 médios.",
                [
                    "`retrieval/reranker.py` com cross-encoder local configurável (família bge-reranker) e interface para usar uma API",
                    "Nó `rerank` com corte em top-k e evento `data`",
                    "Medir latência em CPU",
                    "ADR curto sobre o modelo escolhido",
                ],
                ["Flag `rerank` na config liga e desliga o nó"],
            ),
            issue(
                "compress_context: dedupe e orçamento de tokens",
                ["agent", "llm"],
                "Teto de tokens de entrada por rota.",
                [
                    "Remover chunks duplicados ou quase duplicados",
                    "Cortar no orçamento de tokens configurado",
                    "Registrar tokens antes/depois no evento",
                ],
                ["Contexto nunca passa do orçamento; testes cobrem o corte"],
            ),
            issue(
                "Testes unitários de retrieval",
                ["retrieval"],
                "Regra do projeto: testes unitários para retrieval, RRF, cache e router.",
                ["Fixtures com um mini-corpus no Postgres de teste", "Cobrir BM25, denso, híbrido e rerank (rerank mockado)"],
                ["Suíte roda na CI"],
            ),
            issue(
                "Calibração do LLM-as-judge (kappa de Cohen)",
                ["eval"],
                "Antes de confiar no faithfulness de v1 em diante, medir se o juiz concorda com um humano.",
                [
                    "Amostrar ~50 respostas de v0/v1 e rotular à mão em `eval/judge_calibration/`",
                    "Script que roda o juiz nas mesmas respostas e calcula o kappa de Cohen",
                    "Ajustar o prompt do juiz se a concordância for baixa",
                ],
                ["Kappa reportado e salvo em `eval/judge_calibration/results.json`"],
            ),
            issue(
                "Release v1: eval do híbrido + rerank",
                ["eval", "docs"],
                "Medir o ganho e explicar.",
                [
                    "Config `v1` e eval completo",
                    "Linha v1 na tabela de ablação",
                    "Comentário curto no README sobre o que mudou e por quê (inclusive se algo piorou)",
                ],
                ["`eval/results/v1.json` commitado e tabela atualizada"],
            ),
        ],
    },
    # ------------------------------------------------------------------ F4
    {
        "title": "F4 · Router + cascata + groundedness (v2)",
        "description": "Classificação de rota, cascata de modelos, Corrective RAG, checagem de groundedness e guardrails. "
        "Pronto: eval v2 mostra a redução de custo e a acurácia do router.",
        "issues": [
            issue(
                "Rota esperada no golden dataset",
                ["eval"],
                "Para medir a acurácia do router é preciso saber a rota certa de cada pergunta.",
                [
                    "Campo `expected_route` (`trivial`, `out_of_scope`, `simple_rag`, `multi_hop`)",
                    "Adicionar ~15 perguntas triviais/conversa",
                    "Métrica de acurácia e matriz de confusão do router no run_eval.py",
                ],
                ["Todo item do golden tem rota esperada"],
            ),
            issue(
                "Nó router com saída estruturada",
                ["agent", "llm"],
                "Modelo `small` decide a rota com JSON curto.",
                [
                    "Prompt com parte fixa no início (prompt caching) e a pergunta no fim",
                    "Saída estruturada `{route, reason, confidence}`",
                    "Evento `data` do router",
                    "Testes com LLM mockado",
                ],
                ["Router classifica as perguntas do golden e a acurácia é calculável"],
            ),
            issue(
                "Nós direct_answer e refuse",
                ["agent"],
                "Rotas baratas que não fazem retrieval.",
                ["`direct_answer` com modelo `small`", "`refuse` por template (sem LLM) ou modelo `small`"],
                ["Perguntas triviais e fora de escopo terminam sem retrieval"],
            ),
            issue(
                "Arestas condicionais por rota",
                ["agent"],
                "Montar o grafo da seção 4.1.",
                [
                    "Arestas condicionais saindo do router",
                    "Modelo de geração por rota (`medium` para simple_rag, `large` para multi_hop)",
                    "Exportar o desenho do grafo (mermaid) para o README",
                ],
                ["Cada rota percorre o caminho esperado num teste de integração"],
            ),
            issue(
                "decompose + retrieve em paralelo (Send API)",
                ["agent", "retrieval"],
                "Perguntas multi-hop viram sub-perguntas buscadas em paralelo.",
                [
                    "Nó `decompose` com saída estruturada (lista de sub-perguntas)",
                    "Fan-out com `Send` para o retrieve e junção dos resultados",
                    "Conferir a API de Send na versão instalada",
                ],
                ["Uma pergunta multi-hop gera N retrieves paralelos visíveis no trace"],
            ),
            issue(
                "grade_docs + reescrita de query (Corrective RAG)",
                ["agent"],
                "Se os docs não forem relevantes, reescrever a query e tentar de novo até `max_retries`.",
                [
                    "Nó `grade_docs` com modelo `small` e saída estruturada",
                    "Nó de reescrita e aresta de volta para o retrieve",
                    "Contador `retries` no estado",
                ],
                ["O laço nunca passa de `max_retries`; teste cobre o caso"],
            ),
            issue(
                "groundedness_check com escalada ou \"não sei\"",
                ["agent", "llm"],
                "Verifica se cada afirmação está nos chunks.",
                [
                    "Saída `{grounded, unsupported_claims}`",
                    "Se falhar: escalar o modelo uma vez; se falhar de novo, responder \"não sei\"",
                    "Evento `data` conforme o contrato",
                ],
                ["Perguntas armadilha do golden terminam em \"não sei\" com mais frequência que no v1"],
            ),
            issue(
                "Guardrails: budget_usd_remaining no estado",
                ["agent", "llm"],
                "Interromper a execução quando o orçamento do request estourar.",
                [
                    "Debitar o custo de cada nó",
                    "Aresta de saída antecipada com resposta de fallback",
                    "Teste com orçamento minúsculo",
                ],
                ["Request com orçamento estourado termina de forma limpa com evento explicando"],
            ),
            issue(
                "Custo contrafactual (tudo no modelo large)",
                ["llm", "eval"],
                "Mostrar quanto a cascata economiza.",
                [
                    "Calcular o custo se cada chamada usasse o modelo `large` com os mesmos tokens",
                    "Incluir `counterfactual_cost_usd` no evento `done` e no resultado do eval",
                ],
                ["Eval reporta economia percentual da cascata"],
            ),
            issue(
                "Prompt caching do provedor",
                ["llm"],
                "Experimento de custo: prefixo fixo cacheado.",
                [
                    "Revisar todos os prompts: system e instruções fixas no início, variável no fim",
                    "Marcar os pontos de cache (`cache_control` na Anthropic)",
                    "Registrar tokens lidos do cache no `NodeUsage` e no custo",
                ],
                ["Eval mostra a fração de tokens de entrada servida do cache"],
            ),
            issue(
                "Teste de integração do grafo inteiro",
                ["agent"],
                "Regra do projeto: um teste de integração do grafo inteiro com LLM mockado.",
                ["LLM falso determinístico por nó", "Um caso por rota, verificando eventos e estado final"],
                ["Teste roda na CI sem rede"],
            ),
            issue(
                "Release v2: eval do router + cascata",
                ["eval", "docs", "adr"],
                "Medir custo, acurácia do router e distribuição de rotas.",
                [
                    "Config `v2` e eval completo",
                    "Linha v2 na tabela de ablação + matriz de confusão do router",
                    "ADR da política de cascata",
                ],
                ["`eval/results/v2.json` commitado e tabela atualizada"],
            ),
        ],
    },
    # ------------------------------------------------------------------ F5
    {
        "title": "F5 · Cache exato + semântico (v3)",
        "description": "Cache em camadas com θ ajustado por dados. "
        "Pronto: eval v3 com taxa de acerto e de falso positivo reportadas.",
        "issues": [
            issue(
                "Nó cache_exact",
                ["cache", "agent"],
                "Hash da query normalizada no Redis.",
                [
                    "Normalização (minúsculas, espaços, pontuação) com testes",
                    "Chave inclui a versão da pipeline/corpus para não servir resposta velha",
                    "TTL configurável",
                ],
                ["Segunda chamada idêntica volta do cache sem chamar LLM"],
            ),
            issue(
                "Nó cache_semantic com RedisVL",
                ["cache", "agent"],
                "Vizinho mais próximo no cache; hit se `sim > θ`.",
                [
                    "Índice vetorial no Redis via RedisVL (conferir a API da versão)",
                    "θ configurável",
                    "Evento `data` com `{hit, similarity, threshold}`",
                ],
                ["Paráfrase próxima de uma pergunta cacheada vira hit"],
            ),
            issue(
                "Nó cache_write (só se grounded)",
                ["cache", "agent"],
                "Não cachear resposta ruim.",
                ["Gravar nos dois caches só quando `grounded == True`", "Teste para o caso não-grounded"],
                ["Respostas \"não sei\" e não-grounded nunca entram no cache"],
            ),
            issue(
                "Dataset de pares para ajustar θ",
                ["cache", "eval"],
                "Conjunto separado: pares de perguntas que devem ou não compartilhar a resposta.",
                [
                    "`eval/datasets/cache_pairs.jsonl` com ~100 pares rotulados",
                    "Incluir pares difíceis (mesma API, pergunta diferente)",
                ],
                ["Pares validados e balanceados"],
            ),
            issue(
                "Calibração de θ: acerto × falso positivo",
                ["cache", "eval", "adr"],
                "O diferencial: escolher θ com dados.",
                [
                    "Script que varre θ e calcula taxa de acerto e de falso positivo",
                    "Gráfico da curva salvo em `eval/results/`",
                    "ADR com o θ escolhido e o raciocínio",
                ],
                ["θ no `.env.example` justificado por um número"],
            ),
            issue(
                "Testes unitários de cache",
                ["cache"],
                "Regra do projeto.",
                ["Redis de teste (fakeredis ou container)", "Cobrir normalização, hit/miss, TTL e write condicional"],
                ["Suíte roda na CI"],
            ),
            issue(
                "Release v3: eval com cache",
                ["eval", "docs"],
                "Simular tráfego repetido para medir o cache.",
                [
                    "Modo do run_eval.py que inclui paráfrases e repetições",
                    "Config `v3`, linha na tabela com taxa de acerto e falso positivo",
                ],
                ["`eval/results/v3.json` commitado e tabela atualizada"],
            ),
        ],
    },
    # ------------------------------------------------------------------ F6
    {
        "title": "F6 · Frontend: execução ao vivo",
        "description": "Grafo ao vivo, linha do tempo, painel de ranking, explicação do router e citações. "
        "Pronto: uma pergunta real percorre a interface do início ao fim via SSE.",
        "issues": [
            issue(
                "Setup do frontend Next.js",
                ["frontend", "infra"],
                "Base do app.",
                [
                    "Next.js (App Router) + TypeScript + lint/format",
                    "Variável de URL do backend e CORS no FastAPI",
                    "Job de lint/build do frontend na CI",
                ],
                ["`npm run dev` mostra a página inicial e a CI builda o frontend"],
            ),
            issue(
                "Tipos do contrato de eventos e cliente SSE",
                ["frontend", "tracing"],
                "Mesmo contrato do backend, tipado.",
                [
                    "Gerar tipos TS a partir de `docs/events.schema.json`",
                    "Cliente SSE para POST (fetch + ReadableStream) com cancelamento",
                    "Store da execução (ex.: Zustand) agregando eventos por nó",
                ],
                ["Eventos de um request real aparecem no store em ordem"],
            ),
            issue(
                "GET /graph: topologia do LangGraph",
                ["agent", "frontend"],
                "A interface desenha a topologia real, não uma cópia à mão.",
                ["Endpoint que exporta nós e arestas do grafo compilado", "Ids iguais aos usados no campo `node` dos eventos"],
                ["Mudou o grafo no backend, mudou o desenho no front"],
            ),
            issue(
                "GraphView: grafo ao vivo com React Flow",
                ["frontend"],
                "Prioridade 1 da experiência visual.",
                [
                    "Layout automático (dagre/elk) a partir de /graph",
                    "Estados por nó: ocioso, rodando, concluído, erro, pulado",
                    "Destacar as arestas do caminho escolhido pelo router",
                ],
                ["Os nós acendem na ordem real durante uma pergunta"],
            ),
            issue(
                "TraceTimeline: waterfall de latência, tokens e custo",
                ["frontend"],
                "Prioridade 2.",
                ["Barras por nó no tempo", "Tooltip com tokens in/out, modelo e custo", "Contador de custo acumulado"],
                ["Números batem com o evento `done`"],
            ),
            issue(
                "RankingPanel: bump chart BM25 → denso → RRF → rerank",
                ["frontend", "retrieval"],
                "Prioridade 3: mostra por que o reranker importa.",
                ["Bump chart com a posição de cada chunk nas quatro etapas", "Hover mostra o trecho do chunk"],
                ["Painel renderiza a partir dos eventos de retrieve e rerank"],
            ),
            issue(
                "RouterExplainer",
                ["frontend"],
                "Prioridade 4: \"Rota: multi-hop · Modelo: large · Motivo: ... · Cache: miss\".",
                ["Componente a partir dos eventos do router e dos caches"],
                ["Mostra rota, modelo, motivo, confiança e status de cache"],
            ),
            issue(
                "AnswerWithCitations e visualização da fonte",
                ["frontend"],
                "Prioridade 5: resposta em streaming com citações clicáveis.",
                [
                    "Streaming dos eventos `token`",
                    "Citações `[n]` clicáveis",
                    "Endpoint `GET /chunks/{id}` e painel com o trecho destacado e link para a doc original",
                ],
                ["Clicar numa citação mostra o trecho certo"],
            ),
            issue(
                "Página principal integrando tudo",
                ["frontend"],
                "Fechamento da F6.",
                [
                    "Layout responsivo com input de pergunta e seletor de pipeline",
                    "Estados de vazio, carregando e erro",
                    "Teste ponta a ponta manual documentado",
                ],
                ["Uma pergunta real percorre a interface do início ao fim via SSE"],
            ),
        ],
    },
    # ------------------------------------------------------------------ F7
    {
        "title": "F7 · Mapa UMAP, replay e comparação",
        "description": "Mapa de embeddings, modo replay (padrão do demo público) e modo comparação. "
        "Pronto: o demo roda em replay sem chamar LLM.",
        "issues": [
            issue(
                "UMAP do corpus na ingestão",
                ["ingest"],
                "A projeção do corpus é pré-calculada.",
                [
                    "Ajustar UMAP 2D nos embeddings dos chunks",
                    "Salvar coordenadas por chunk e o modelo UMAP serializado",
                    "`GET /embedding-map` com os pontos (id, x, y, doc)",
                ],
                ["Endpoint devolve todos os pontos em < 1s"],
            ),
            issue(
                "Evento embedding_map com a query projetada",
                ["agent", "retrieval"],
                "A query é projetada na hora com `umap.transform`.",
                ["Projetar a query e emitir `{query_xy, hit_ids}`", "Medir a latência do transform"],
                ["Evento emitido em toda pergunta com retrieval"],
            ),
            issue(
                "EmbeddingMap: nuvem de pontos interativa",
                ["frontend"],
                "Prioridade 6.",
                [
                    "Scatter performático (canvas/WebGL) com cor por documento",
                    "Query aparece e os chunks recuperados acendem",
                    "Zoom/pan e hover com título do chunk",
                ],
                ["Fluido com o corpus inteiro"],
            ),
            issue(
                "GET /replay/{id}: reemitir eventos gravados",
                ["tracing"],
                "Replay sem chamar LLM.",
                ["Reemitir eventos via SSE respeitando os intervalos originais (com fator de velocidade)", "Mesmo envelope do /query"],
                ["Replay idêntico ao request original na interface"],
            ),
            issue(
                "Curadoria dos traces de demo",
                ["eval", "tracing"],
                "As perguntas que o visitante vai ver.",
                [
                    "Gravar ~10 perguntas que cobrem todas as rotas, um hit de cache e um \"não sei\"",
                    "Exportar os eventos para um seed versionado",
                ],
                ["Seed carregado no deploy"],
            ),
            issue(
                "Modo replay como padrão no frontend",
                ["frontend"],
                "O demo público nunca quebra e não gera custo.",
                ["Seletor de perguntas gravadas", "Controles play/pause/passo", "Modo ao vivo atrás de flag"],
                ["Interface funciona com o backend sem chave de LLM"],
            ),
            issue(
                "CompareMode: duas configs lado a lado",
                ["frontend", "eval"],
                "Prioridade 7: ex. RAG ingênuo × pipeline completo.",
                ["Backend aceita a pipeline por request (já existe) e o front roda duas em paralelo", "Comparativo de custo, latência e resposta", "Versão em replay"],
                ["v0 × v3 lado a lado numa mesma pergunta"],
            ),
        ],
    },
    # ------------------------------------------------------------------ F8
    {
        "title": "F8 · Eval na CI, guardrails e deploy",
        "description": "Gate de regressão na CI, rate limit, teto de gasto, feedback online e deploy. "
        "Pronto: PR com regressão falha na CI; demo no ar com budget diário.",
        "issues": [
            issue(
                "Subconjunto estratificado do eval para CI",
                ["eval", "ci"],
                "Barato o bastante para rodar a cada PR.",
                ["~20–30 perguntas cobrindo todos os `type`", "Estimar custo e tempo por execução"],
                ["Subset roda em < 5 min com custo conhecido"],
            ),
            issue(
                "Workflow eval.yml com gate de regressão",
                ["ci", "eval"],
                "Falha se faithfulness ou recall caírem, ou se o custo subir além do limite.",
                [
                    "Workflow com serviços Postgres/Redis e índice de teste",
                    "Comparar com `eval/results/` da main e tolerâncias configuráveis",
                    "Comentário no PR com a tabela de diferença",
                    "Segredos de API no GitHub Actions",
                ],
                ["Um PR que piora o retrieval de propósito falha"],
            ),
            issue(
                "Rate limit por IP",
                ["deploy"],
                "Proteção do demo.",
                ["Limite por IP no /query (Redis)", "Resposta 429 clara e evento de erro na interface"],
                ["Teste do limite"],
            ),
            issue(
                "Teto de gasto diário em USD",
                ["deploy", "llm"],
                "Quando estoura, o demo cai para replay.",
                ["Contador diário no Redis alimentado pelo custo real", "Bloquear o modo ao vivo e avisar na interface"],
                ["Simulação de teto estourado leva para o replay"],
            ),
            issue(
                "Feedback 👍/👎 ligado ao trace",
                ["frontend", "tracing", "eval"],
                "Eval online (seção 7.6).",
                [
                    "Botões na resposta enviando score ao Langfuse pelo `request_id`",
                    "Script que amostra traces reais, roda o juiz e propõe casos de falha para o golden",
                ],
                ["Feedback aparece no Langfuse"],
            ),
            issue(
                "Deploy do backend e dados",
                ["deploy", "adr"],
                "Fly, Railway ou Render + Postgres/Redis gerenciados.",
                [
                    "ADR: provedor e onde fica o Langfuse em produção",
                    "Dockerfile do backend (modelo de rerank/embedding no build ou volume)",
                    "Rodar a ingestão no banco de produção",
                    "Segredos no provedor",
                ],
                ["/health público respondendo"],
            ),
            issue(
                "Deploy do frontend na Vercel",
                ["deploy", "frontend"],
                "Demo público.",
                ["Projeto Vercel com variáveis de ambiente", "Domínio/URL no README"],
                ["Demo no ar em modo replay"],
            ),
        ],
    },
    # ------------------------------------------------------------------ F9
    {
        "title": "F9 · Narrativa e divulgação",
        "description": "README com tabela de ablação, ADRs, failures.md, vídeo e post. "
        "Pronto: um recrutador entende o projeto em 2 minutos.",
        "issues": [
            issue(
                "README final",
                ["docs"],
                "A peça mais lida do projeto.",
                [
                    "Pitch, GIF do demo e link",
                    "Diagrama de arquitetura e do grafo",
                    "Tabela de ablação completa (v0–v3) e kappa do juiz, só números medidos",
                    "Como rodar localmente em 3 comandos",
                ],
                ["Alguém de fora lê e entende em 2 minutos"],
            ),
            issue(
                "docs/failures.md",
                ["docs", "eval"],
                "Onde o sistema falha e por quê.",
                ["Categorizar erros do eval v3 (retrieval, geração, router, cache)", "Exemplos concretos e hipóteses de melhoria"],
                ["Pelo menos 5 modos de falha documentados com exemplos"],
            ),
            issue(
                "Revisão dos ADRs",
                ["docs", "adr"],
                "Consistência e índice.",
                ["Índice em `docs/adr/README.md`", "Atualizar ADRs que mudaram com os números finais"],
                ["Todos os ADRs com status atualizado"],
            ),
            issue(
                "Vídeo de 90 segundos",
                ["docs"],
                "Demo narrado.",
                ["Roteiro: problema, demo ao vivo/replay, tabela de ablação, aprendizado", "Gravar e publicar; link no README"],
                ["Vídeo no ar"],
            ),
            issue(
                "Post de divulgação",
                ["docs"],
                "LinkedIn ou blog contando a narrativa medir → mudar → medir.",
                ["Rascunho com 2–3 números do eval e um caso de falha", "Publicar"],
                ["Post publicado com link para o repo"],
            ),
        ],
    },
]


def write_markdown(path: Path) -> None:
    lines = [
        "# Roadmap do RAG X-Ray",
        "",
        "Gerado por `scripts/roadmap.py`. Cada milestone é uma fase do `context.md`; cada item é uma issue no GitHub.",
        "",
    ]
    for m in MILESTONES:
        lines += [f"## {m['title']}", "", m["description"], ""]
        for i in m["issues"]:
            lines.append(f"- [ ] {i['title']} `{'` `'.join(i['labels'])}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"escrito {path}")


def gh(*args: str, stdin: str | None = None) -> str:
    r = subprocess.run(["gh", *args], input=stdin, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)}\n{r.stderr}")
    return r.stdout


def create_on_github() -> None:
    existing_labels = {l["name"] for l in json.loads(gh("label", "list", "-R", REPO, "--limit", "200", "--json", "name"))}
    for name, (color, desc) in LABELS.items():
        if name not in existing_labels:
            gh("label", "create", name, "-R", REPO, "--color", color, "--description", desc)
            print(f"label {name}")

    milestones = {m["title"]: m["number"] for m in json.loads(gh("api", f"repos/{REPO}/milestones?state=all&per_page=100"))}
    existing_issues = {
        i["title"] for i in json.loads(gh("issue", "list", "-R", REPO, "--state", "all", "--limit", "500", "--json", "title"))
    }

    for m in MILESTONES:
        if m["title"] not in milestones:
            out = gh("api", f"repos/{REPO}/milestones", "-f", f"title={m['title']}", "-f", f"description={m['description']}")
            milestones[m["title"]] = json.loads(out)["number"]
            print(f"milestone {m['title']}")
        for i in m["issues"]:
            if i["title"] in existing_issues:
                continue
            args = ["issue", "create", "-R", REPO, "--title", i["title"], "--milestone", m["title"], "--body-file", "-"]
            for label in i["labels"]:
                args += ["--label", label]
            url = gh(*args, stdin=i["body"]).strip()
            print(f"  {url}  {i['title']}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "md"
    root = Path(__file__).resolve().parent.parent
    if mode == "md":
        write_markdown(root / "docs" / "ROADMAP.md")
    elif mode == "gh":
        create_on_github()
    else:
        sys.exit("uso: roadmap.py [md|gh]")
