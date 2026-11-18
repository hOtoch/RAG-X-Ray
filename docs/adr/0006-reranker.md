# ADR-0006: Reranker local (jina-reranker-v1-turbo-en), com ganho pequeno e custo alto de latência

- **Status:** Aceito com ressalva (falta o eval de ponta a ponta da v1)
- **Data:** 2026-11-18

## Contexto

O briefing aposta no rerank com cross-encoder ("4 chunks bons custam menos que 20 médios"). A máquina de desenvolvimento só tem CPU, e o deploy também deve ser em CPU, então o modelo precisa ser pequeno.

## Medição

`eval/retrieval_bench.py`, 130 perguntas com gabarito. Todos os rerankers recebem os 20 primeiros candidatos do retrieval híbrido (denso + BM25 + RRF) e devolvem os 5 melhores. Modelos em ONNX via `fastembed`.

| Estratégia | Recall@5 | Recall@5 (página) | MRR | nDCG@5 | Latência p50 |
|---|---|---|---|---|---|
| Híbrida, sem rerank | 0,745 | 0,942 | 0,686 | 0,648 | 168 ms |
| + ms-marco-MiniLM-L-6-v2 | 0,710 | 0,942 | 0,699 | 0,630 | 2,4 s |
| + ms-marco-MiniLM-L-12-v2 | 0,717 | 0,942 | 0,704 | 0,645 | 3,9 s |
| + jina-reranker-v1-turbo-en | **0,756** | **0,954** | 0,689 | **0,654** | 2,5 s |

## Decisão

- O reranker padrão é o **jina-reranker-v1-turbo-en** (`RERANKER_MODEL`), usado nas pipelines v1, v2 e v3.
- A pipeline **v1-hybrid** (sem rerank) fica registrada para comparação de ponta a ponta.

## Por que "com ressalva"

- O ganho em retrieval é pequeno: +1,1 ponto de recall@5 e +1,2 ponto de recall por página. Os dois modelos MS MARCO **pioraram** o recall@5.
- O custo é alto: cerca de +2,3 s por pergunta em CPU, mais que a geração inteira da v0 em alguns casos.
- O eval completo da v1 (com rerank) não chegou a rodar: a conta da OpenAI ficou sem créditos durante o eval da v1-hybrid. Sem esse número, não dá para dizer se o rerank melhora a resposta final. Se o ganho em correção não aparecer, a decisão certa é tirar o rerank das pipelines v2 e v3 (`rerank: false`; o grafo com router já funciona sem ele, com o nó `select_context`).

## Consequências

- O modelo (~150 MB) é baixado na primeira execução para `data/models/` e precisa ir na imagem de deploy ou num volume.
- A inferência roda sob um lock (uma de cada vez), porque a sessão ONNX já usa todos os núcleos.
- Um reranker por API (Cohere, Voyage, Jina) reduziria a latência, mas adicionaria custo por pergunta e outra chave. Fica como experimento.
