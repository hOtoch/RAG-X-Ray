# ADR-0003: Embeddings (e LLMs) pela API da OpenAI

- **Status:** Aceito
- **Data:** 2026-11-14

## Contexto

O retrieval denso precisa de um modelo de embedding. A máquina de desenvolvimento tem um Ryzen 5 3500X, 16 GB de RAM e uma GPU AMD (sem CUDA), então um modelo local roda só em CPU. O corpus tem 773 páginas em inglês, e as perguntas podem vir em português.

A única chave de API disponível no `.env` é a da OpenAI. Não há chave da Anthropic.

## Decisão

- **Embeddings:** `text-embedding-3-small` da OpenAI (1536 dimensões), definido em `EMBEDDING_MODEL`. O pgvector indexa até 2000 dimensões com HNSW, então o vetor cabe sem redução.
- **LLMs:** enquanto só existir a chave da OpenAI, os três níveis (`small`, `medium`, `large`) também usam modelos da OpenAI, definidos em `LLM_MODEL_*`. O provedor continua plugável por `LLM_PROVIDER`; trocar para Anthropic exige só mudar o `.env`.

Nenhum nome de modelo fica no código.

## Alternativas consideradas

- **Modelo local multilíngue (multilingual-e5-small, bge-m3):** sem custo por chamada e reprodutível offline, mas a indexação em CPU é lenta nesta máquina e o deploy precisaria carregar o modelo na memória.
- **Voyage:** boa qualidade para retrieval, mas exige mais uma conta e mais uma chave.

## Consequências

- Indexar o corpus custa centavos e leva poucos minutos.
- O eval depende de uma API externa: o modelo pode ser descontinuado, e isso obriga a reindexar. O nome do modelo de embedding precisa ficar gravado em cada `eval/results/*.json`.
- Trocar de modelo de embedding muda a dimensão do vetor e exige recriar a tabela de chunks.
- O experimento de prompt caching (F4) passa a medir o cache automático da OpenAI, e não o `cache_control` da Anthropic.
