# ADR-0004: Langfuse self-hosted para tracing

- **Status:** Aceito
- **Data:** 2026-11-14

## Contexto

O projeto precisa mostrar, para cada pergunta, o trace completo: um span por nó do grafo, com tokens, custo, modelo e rota. O `context.md` (seção 7.1) pede tracing ligado desde o início e o mesmo `request_id` no trace e no stream SSE.

## Decisão

Usar **Langfuse self-hosted**, no mesmo `docker compose` do projeto.

- O grafo recebe o `CallbackHandler` do Langfuse, que cria um span por nó e uma observação por chamada de LLM.
- O id do trace é derivado do `request_id` (`create_trace_id(seed=request_id)`), e o `request_id` também vai como sessão. Com isso, qualquer parte do sistema chega ao trace só com o `request_id`.
- Cada nó grava o próprio `NodeUsage` (modelo, tokens, custo, latência) nos metadados do seu span.
- O custo e a latência totais medidos pelo backend vão como scores do trace (`total_cost_usd`, `total_latency_ms`), pela API REST.
- O tracing desliga sozinho se `LANGFUSE_ENABLED=false` ou se as chaves não estiverem no ambiente. Os testes rodam com ele desligado.

## Alternativas consideradas

- **LangSmith:** integração nativa com LangGraph e menos setup, mas é um serviço externo com limites no plano gratuito, e o demo público dependeria dele.
- **Só OpenTelemetry com um coletor genérico:** mais portátil, mas sem interface pronta para custo, tokens e scores.

## Consequências

- O Langfuse v3 traz ClickHouse e MinIO junto, além de dois containers próprios. O `docker compose up` fica mais pesado (cerca de 2 GB de imagens a mais).
- O Langfuse reaproveita o Postgres (banco `langfuse`) e o Redis (db 1) do projeto, para não duplicar serviços.
- A imagem `minio/minio` saiu do Docker Hub; o compose usa `cgr.dev/chainguard/minio`.
- O custo que aparece no trace é o calculado pelo backend, a partir de `configs/prices.yaml`. O custo que o próprio Langfuse calcula depende da tabela de preços dele e apareceu zerado para os modelos usados aqui.
- O SDK do Langfuse 5.2 recusou `create_score` contra o servidor 3.227 ("Bad request"); por isso os scores vão pela API REST.
- A chamada de embedding não vira uma observação própria no Langfuse (o callback do LangChain não cobre embeddings); o custo dela fica nos metadados do span `retrieve`.
- Em produção, o Langfuse precisa de segredos próprios (`NEXTAUTH_SECRET`, `SALT`, `ENCRYPTION_KEY`) e de chaves de projeto diferentes das locais. Isso fica para o ADR de deploy (F8).
