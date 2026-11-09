# ADR-0001: Corpus da documentação do LangGraph

- **Status:** Aceito
- **Data:** 2026-11-09

## Contexto

O projeto precisa de um corpus que permita dizer se uma resposta está certa. Sem isso o eval não é viável, e o `context.md` (seção 1) exige esse critério. Os requisitos são: de centenas a poucos milhares de documentos, licença que permita indexar, e metadados de título e URL para as citações.

A documentação do LangGraph é publicada no repositório [langchain-ai/docs](https://github.com/langchain-ai/docs), que é MIT. Os arquivos de conteúdo são `.mdx`, com blocos de código em Python e JavaScript na mesma página (abas de linguagem), e cada página tem uma URL pública estável.

Fonte fixada para reprodutibilidade:

- Repositório: `langchain-ai/docs`
- Commit: `4db8ed56a4c7c8cbdca47053d9eee4fc57124ffd` (2026-11-09)
- Licença: MIT

Tamanho medido nesse commit:

| Escopo | Páginas `.mdx`/`.md` | Tamanho | Tokens (aprox., bytes/4) |
|---|---|---|---|
| `src/oss/langgraph/**` | 45 | ~891 KB | ~223 mil |
| `src/oss/langgraph/**` + `src/langsmith/**` | 627 | ~8,1 MB | — |
| `src/oss/**` (todo o OSS) | 773 | ~8,5 MB | — |

## Decisão

Usar todo o `src/oss/**` do commit fixado acima: 773 páginas `.md`/`.mdx`, cerca de 8,5 MB. Isso inclui LangGraph, LangChain, DeepAgents, conceitos, referência e os guias de integração em Python e JavaScript (529 páginas).

O escopo completo atende o critério de "centenas a poucos milhares de documentos" do briefing. Os guias de integração competem com as páginas de LangGraph no retrieval, o que torna o eval mais exigente e mais realista.

O número de documentos muda o comportamento do retrieval, então ele precisa aparecer em todo `eval/results/*.json`.

## Alternativas consideradas

- **Só `src/oss/langgraph/**` (45 páginas):** mais focado, mas pequeno demais para o critério do briefing e para um eval com perguntas de retrieval difíceis.
- **Recorte sem integrações (~210 páginas):** LangGraph, LangChain, DeepAgents, conceitos e referência. Evita o ruído dos guias de integração, mas descarta justamente as páginas que competem no retrieval.
- **Documentação do Kubernetes:** corpus grande e muito multi-hop, mas indexação mais cara e menos alinhada ao tema de agentes. Licença CC BY 4.0, exigiria atribuição.
- **Normas do Banco Central, CVM ou LGPD:** domínio brasileiro pouco explorado, com citação obrigatória, mas os textos são PDFs e exigem parsing mais difícil antes de qualquer RAG. Fica como projeto futuro.
- **Papers do arXiv sobre RAG:** o tema vira o próprio dataset, mas o parsing de PDF científico é trabalhoso e é mais difícil dizer se uma resposta está certa.

## Consequências

- O eval tem uma referência verificável: a própria documentação, com URL por página.
- A reprodutibilidade depende do commit fixado. Atualizar a fonte exige novo ADR ou nota neste.
- Como a documentação muda com frequência, o eval só é comparável entre versões que usam o mesmo commit. Isso precisa ser registrado em cada `eval/results/*.json`.
- A licença MIT permite indexar e citar com atribuição ao repositório de origem.
