# Eval

Tudo roda a partir da raiz do repositório, com a infra no ar (`docker compose up -d`) e o corpus indexado.

```bash
uv run --project backend python -m eval.run_eval --pipeline v0
```

O resultado vai para `eval/results/<pipeline>.json`, com a config usada, o commit, o hash do dataset, os modelos, as métricas agregadas (geral e por tipo de pergunta) e uma linha por pergunta. Use `--subset N` para um teste rápido e `--no-judge` para medir só retrieval, custo e latência.

## Golden dataset

`datasets/golden.jsonl` tem 160 perguntas:

| Tipo | Quantidade | Origem |
|---|---|---|
| `simple` | 94 | 70 sintéticas, 24 manuais |
| `multi_hop` | 36 | 30 sintéticas, 6 manuais |
| `trap` (premissa falsa) | 15 | manuais |
| `out_of_scope` | 15 | manuais |

Como cada parte foi feita:

- **Sintéticas** (`generate_synthetic.py`): o modelo `large` recebe um chunk (pergunta simples) ou dois chunks de seções diferentes da mesma página (multi-hop) e escreve a pergunta e a resposta de referência. Uma segunda chamada descarta perguntas que só fazem sentido com o trecho na frente. A amostragem dá mais peso às páginas de LangGraph e exclui trechos padronizados que se repetem em várias páginas. As 100 perguntas foram lidas uma a uma depois de geradas; a primeira rodada foi descartada inteira por dois defeitos do gerador (nenhuma pergunta de integrações e perguntas repetidas).
- **Manuais** (`datasets/manual_questions.yaml`): as perguntas foram escritas à mão, no estilo de quem consulta a documentação, parte delas em português. Para cada uma, `build_golden.py` busca os trechos mais próximos dentro das páginas indicadas, e o modelo `large` redige a resposta de referência só com esses trechos e marca quais a sustentam. As 30 respostas foram lidas depois de geradas.
- **Armadilhas e fora de escopo**: escritas à mão. A resposta de referência descreve o comportamento esperado (apontar a premissa falsa ou recusar).

Custo de geração do dataset: cerca de US$ 0,96 (US$ 0,39 da rodada descartada, US$ 0,40 da rodada usada e US$ 0,17 das referências manuais).

### Limitações conhecidas

- O gabarito de retrieval marca os chunks que originaram a pergunta. Outros chunks do corpus também podem responder, então `recall@5` por chunk subestima o retrieval. Por isso o eval também reporta `doc_recall@5`, que conta acerto no nível da página.
- As respostas de referência foram redigidas por um modelo a partir da documentação e depois lidas, mas não foram conferidas por uma segunda pessoa.
- As perguntas sintéticas tendem a repetir o vocabulário do trecho de origem, o que favorece a busca lexical.

## Métricas

- **Retrieval** (só perguntas com gabarito): `recall@5`, `doc_recall@5`, `mrr`, `ndcg@5`, com implementação própria em `metrics/retrieval.py`.
- **Geração** (`judge.py`, modelo `large` como juiz):
  - `faithfulness`: fração das afirmações da resposta que o contexto recuperado sustenta.
  - `citation_precision`: fração das afirmações com citação em que os trechos citados sustentam a afirmação.
  - `correctness`: comparação com a resposta de referência (1, 0,5 ou 0).
  - `refusal_rate_*`: recusar uma pergunta respondível é erro; recusar armadilha ou fora de escopo é o comportamento certo.
- **Operacionais**: custo por pergunta (sem o custo do juiz), latência p50 e p95, tokens.

O briefing sugere RAGAS ou DeepEval. As métricas de geração foram implementadas com um juiz próprio porque ele fica mais fácil de calibrar contra rótulos humanos (issue de calibração, F3) e não prende o projeto a uma versão dessas bibliotecas. A concordância do juiz com um humano ainda não foi medida; até lá, os números de `faithfulness` e `correctness` dependem de um juiz não calibrado.

A latência é medida com 4 perguntas em paralelo, na máquina de desenvolvimento.
