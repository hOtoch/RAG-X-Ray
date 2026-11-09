# ADR-0002: pgvector como banco vetorial

- **Status:** Proposto
- **Data:** 2026-11-09

## Contexto

O projeto precisa guardar embeddings dos chunks e fazer busca por similaridade, além de busca lexical (BM25) para o retrieval híbrido da F3. O `context.md` (seção 2) indica Postgres + pgvector como padrão, com Qdrant como alternativa.

Restrições:

- Operação simples: a infra local deve subir com um `docker compose` (issue #5).
- Deploy barato na fase 8 (Fly, Railway ou Render).
- O corpus é pequeno para padrões de produção: cerca de 773 páginas, dezenas de milhares de chunks no máximo.
- A mesma base precisa guardar metadados dos chunks, eventos de trace (`events`, issue da F2) e o contador de custo diário.

## Decisão

Usar **Postgres com a extensão pgvector** para os vetores, com índice HNSW para similaridade de cosseno. A busca BM25 também fica no Postgres, e os metadados e eventos ficam no mesmo banco.

## Alternativas consideradas

- **Qdrant:** busca vetorial mais madura e com filtros mais ricos, mas é um serviço a mais para operar, monitorar e pagar no deploy. Além disso, a busca lexical ficaria fora do banco vetorial, e o RRF precisaria juntar duas fontes separadas.
- **Chroma ou LanceDB (embutidos):** simples para protótipo, mas sem a mesma história de produção e sem um banco relacional para metadados e traces.

## Consequências

- Um banco só: menos infra, backup e deploy mais simples.
- BM25 no Postgres não é BM25 puro. O `ts_rank` do full-text é aproximado; a escolha definitiva fica no ADR da issue de BM25 (F3).
- Com o corpus atual, HNSW tem latência de sobra. Se o corpus crescer muito, o índice precisa de ajuste de `m` e `ef_search`, medidos no eval.
- Trocar para Qdrant depois fica restrito à camada `retrieval/`, desde que a interface de busca seja mantida.
