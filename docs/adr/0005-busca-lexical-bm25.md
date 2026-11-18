# ADR-0005: BM25 em memória para a busca lexical

- **Status:** Aceito
- **Data:** 2026-11-18

## Contexto

O retrieval híbrido da F3 precisa de uma busca lexical para acertar termos exatos (nomes de classe, de função, de erro) que a busca densa às vezes perde. O ADR-0002 deixou a escolha do BM25 em aberto: o full-text search do Postgres é aproximado, e não BM25 de verdade.

## Medição

`eval/retrieval_bench.py`, 130 perguntas com gabarito, mesmo índice, mesmos embeddings:

| Estratégia | Recall@5 | Recall@5 (página) | Recall@20 | MRR | Latência p50 |
|---|---|---|---|---|---|
| Densa (HNSW, `ef_search=200`) | 0,687 | 0,889 | 0,855 | 0,607 | 135 ms |
| BM25 (`rank_bm25`, em memória) | 0,684 | 0,923 | 0,868 | 0,605 | 33 ms |
| Full-text do Postgres (`ts_rank_cd`) | 0,212 | 0,481 | 0,430 | 0,209 | 99 ms |
| Híbrida densa + BM25 (RRF) | **0,745** | **0,942** | **0,903** | **0,686** | 168 ms |
| Híbrida densa + full-text (RRF) | 0,614 | 0,885 | 0,828 | 0,527 | 235 ms |

A latência é só do retrieval, sem a chamada de embedding da pergunta (os vetores das perguntas ficam em cache no bench).

## Decisão

- Busca lexical com **BM25 Okapi em memória** (`rank_bm25`), com o índice montado a partir da tabela de chunks na primeira busca.
- O tokenizador mantém identificadores inteiros (`add_conditional_edges`) e também as partes deles, e remove palavras muito comuns em perguntas, em inglês e português.
- Fusão com **RRF** (k = 60), 20 candidatos de cada fonte.
- `hnsw.ef_search = 200` na busca densa. Com o padrão do pgvector (40), a busca aproximada perdia 3 pontos de recall@5 contra a busca exata neste corpus; com 200 o resultado ficou igual ao exato.

## Alternativas consideradas

- **Full-text do Postgres:** fica no mesmo banco e não ocupa memória da API, mas teve um terço do recall do BM25. O `ts_rank_cd` não normaliza pela frequência do termo no corpus como o BM25, e o stemming em inglês atrapalha nomes de API.
- **Extensão BM25 no Postgres (ParadeDB `pg_search`):** BM25 real dentro do banco, mas exigiria trocar a imagem do Postgres. Fica como opção se o índice em memória pesar no deploy.

## Consequências

- O índice BM25 de 10.800 chunks ocupa memória do processo da API e leva alguns segundos para ser montado na primeira busca (`warmup` no início do servidor resolve).
- Cada réplica da API monta o próprio índice. Se o corpus crescer muito, migrar para `pg_search`.
- A busca densa ficou cerca de 90 ms mais lenta com `ef_search = 200`, o que é pouco perto da latência do LLM.
- As perguntas sintéticas tendem a repetir o vocabulário do trecho de origem, o que favorece o BM25 neste eval. A vantagem em perguntas reais pode ser menor (registrado em `docs/failures.md`).
