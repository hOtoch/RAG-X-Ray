# ADR-0007: Cache exato sempre, cache semântico só com θ = 0,98

- **Status:** Aceito
- **Data:** 2026-11-18

## Contexto

O cache semântico promete responder paráfrases sem chamar o LLM. O risco é o falso positivo: servir a resposta de uma pergunta parecida, mas diferente. Para uma documentação técnica, isso é pior que não ter cache, porque a resposta errada vem com citações e parece confiável.

## Medição

`eval/cache_calibration.py` gerou 120 pares a partir do golden dataset: 60 paráfrases (mesma resposta, um terço em português) e 60 perguntas vizinhas (mesmo tema e vocabulário, resposta diferente). A similaridade é o cosseno entre os embeddings de `text-embedding-3-small`. Resultado em `eval/results/cache_threshold.json`.

| θ | Acerto em paráfrases | Falso positivo |
|---|---|---|
| 0,85 | 73% | 48% |
| 0,90 | 60% | 28% |
| 0,93 | 37% | 10% |
| 0,95 | 15% | 8% |
| 0,97 | 0% | 3% |
| 0,98 | 0% | 0% |

- Similaridade média das paráfrases: 0,89 (mínima 0,71).
- Similaridade média das vizinhas: 0,83 (máxima 0,979). Exemplo: "stream de tokens **com** metadados" e "stream de tokens **sem** metadados" ficaram com 0,979.

As duas distribuições se sobrepõem quase inteiras. Não existe θ que acerte uma fração útil das paráfrases sem servir respostas erradas.

## Decisão

- **Cache exato** sempre ligado na v3: pergunta normalizada (minúsculas, sem acento, sem pontuação) com hash no Redis.
- **Cache semântico** com **θ = 0,98**, o menor limiar com falso positivo ≤ 2% nos pares. Na prática ele só acerta perguntas quase idênticas, e fica como rede de segurança para variações mínimas de digitação que a normalização não cobre.
- Só respostas que passaram na checagem de groundedness entram no cache (e as recusas por template, que não dependem de modelo). "Não sei" nunca entra.
- As chaves incluem a pipeline, o modelo de embedding e o commit do corpus.
- A busca do vizinho mais próximo é feita em numpy sobre os vetores guardados num hash do Redis, em vez do índice vetorial do RedisVL. Com poucos milhares de entradas, leva menos de 1 ms e não depende do módulo de busca do Redis.

## Alternativas consideradas

- **θ ≈ 0,90**, valor comum em exemplos de cache semântico: acertaria 60% das paráfrases, mas serviria a resposta errada em 28% das perguntas vizinhas.
- **Reranker ou LLM confirmando o acerto do cache:** reduziria o falso positivo, mas cada confirmação custa uma chamada, e parte da economia do cache some. Fica como experimento futuro.
- **Modelo de embedding maior** (`text-embedding-3-large`): pode separar melhor as distribuições; precisa ser medido com os mesmos pares antes de mudar o θ.

## Consequências

- A economia do cache vem quase toda do cache exato, ou seja, de perguntas repetidas. No demo público, com perguntas sugeridas, isso é a maior parte do tráfego.
- Os pares "vizinhos" foram escritos para serem difíceis. Em tráfego real o falso positivo seria menor, mas sem dados reais não há como dizer quanto. A curva deve ser recalculada com perguntas reais quando houver tráfego (eval online, F8).
- O resultado vale para `text-embedding-3-small`; trocar de modelo exige rodar a calibração de novo.
