# ADR-0001: Corpus da documentação do LangGraph

- **Status:** Proposto (escopo exato a confirmar antes da issue de ingestão)
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

Usar a documentação do LangGraph do commit fixado acima, na pasta `src/oss/langgraph/**`, com as 45 páginas `.mdx` como ponto de partida. Essa pasta é a mais relevante para perguntas de RAG, agentes e grafos, e já cobre o tema central do projeto.

**Ponto em aberto:** o escopo de 45 páginas fica abaixo da faixa de "centenas de documentos" do briefing. Antes da issue de ingestão, decidir entre:

1. Manter só `src/oss/langgraph/**` (mais simples, perguntas mais focadas).
2. Ampliar para `src/oss/**` (773 páginas, inclui LangChain, conceitos e Python/JS), que atende melhor o briefing e aumenta o desafio do retrieval.

Esta decisão deve ser revista quando a ingestão estiver pronta, porque o número de documentos muda o comportamento do retrieval e precisa aparecer nos eval.

## Alternativas consideradas

- **Documentação do Kubernetes:** corpus grande e muito multi-hop, mas indexação mais cara e menos alinhada ao tema de agentes. Licença CC BY 4.0, exigiria atribuição.
- **Normas do Banco Central, CVM ou LGPD:** domínio brasileiro pouco explorado, com citação obrigatória, mas os textos são PDFs e exigem parsing mais difícil antes de qualquer RAG. Fica como projeto futuro.
- **Papers do arXiv sobre RAG:** o tema vira o próprio dataset, mas o parsing de PDF científico é trabalhoso e é mais difícil dizer se uma resposta está certa.

## Consequências

- O eval tem uma referência verificável: a própria documentação, com URL por página.
- A reprodutibilidade depende do commit fixado. Atualizar a fonte exige novo ADR ou nota neste.
- Como a documentação muda com frequência, o eval só é comparável entre versões que usam o mesmo commit. Isso precisa ser registrado em cada `eval/results/*.json`.
- A licença MIT permite indexar e citar com atribuição ao repositório de origem.
