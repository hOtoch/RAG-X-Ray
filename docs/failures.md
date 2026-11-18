# Onde o sistema falha e por quê

Registro honesto dos modos de falha encontrados até agora, com exemplos reais do eval. Cada item diz o que acontece, a causa provável e o que mudaria.

## 1. O cache semântico não separa paráfrases de perguntas vizinhas

**O que acontece.** Com `text-embedding-3-small`, a pergunta "como fazer stream de tokens **com** metadados" e a pergunta "como fazer stream de tokens **sem** metadados" têm similaridade 0,979. Paráfrases verdadeiras ficam, em média, em 0,89. Qualquer θ que acerte uma parte útil das paráfrases também serve a resposta errada para perguntas vizinhas (28% de falso positivo com θ = 0,90).

**Causa.** Embeddings de uso geral medem proximidade de assunto, não equivalência de pedido. Uma palavra de negação ou um nome de parâmetro diferente muda a resposta, mas quase não muda o vetor.

**O que foi feito.** θ = 0,98 (falso positivo ≤ 2% nos pares de teste). Na prática, o cache semântico só acerta perguntas quase idênticas, e a economia vem do cache exato. Detalhes no [ADR-0007](adr/0007-cache-semantico-theta.md).

**O que mudaria.** Confirmar o acerto com o reranker ou com um modelo pequeno antes de servir a resposta, ou testar um modelo de embedding maior com os mesmos pares.

## 2. O reranker quase não ajuda e custa caro em CPU

**O que acontece.** Dois de três cross-encoders testados pioraram o recall@5 em relação ao retrieval híbrido sem rerank. O melhor (jina-turbo) ganhou 1,1 ponto e acrescentou cerca de 2,3 s por pergunta.

**Causa provável.** Os modelos MS MARCO foram treinados em perguntas de busca web, curtas e em inglês, e não em documentação técnica com blocos de código. Além disso, o gabarito do eval marca o chunk que originou a pergunta; quando o reranker escolhe outro chunk igualmente correto, conta como erro.

**O que mudaria.** Medir o efeito na resposta final (o eval completo da v1 ficou pendente) e testar um reranker multilíngue maior ou via API. Ver [ADR-0006](adr/0006-reranker.md).

## 3. O gabarito de retrieval subestima o sistema

**O que acontece.** O recall@5 por chunk da v0 é 0,687, mas o recall por página é 0,889. Em muitos "erros", o sistema trouxe outro trecho da mesma página, ou de outra página, que também responde à pergunta.

**Causa.** Cada pergunta sintética tem como gabarito apenas os chunks de onde foi gerada. A documentação repete informação entre páginas (guias, referências, exemplos).

**O que mudaria.** Rotular à mão o conjunto de trechos aceitáveis para um subconjunto de perguntas, ou usar um juiz para decidir se o trecho recuperado responde.

## 4. Perguntas sintéticas favorecem a busca lexical

**O que acontece.** Sozinho, o BM25 empata com a busca densa (recall@5 0,684 contra 0,687) e ganha no recall por página.

**Causa provável.** As perguntas sintéticas são escritas a partir do trecho e repetem o vocabulário dele, incluindo nomes de classe e de parâmetro. Usuários reais descrevem o problema com outras palavras, muitas vezes em português.

**O que mudaria.** Aumentar a fração de perguntas manuais e trazer perguntas reais do eval online (botões 👍/👎) para o golden dataset.

## 5. Armadilhas: o modelo às vezes aceita a premissa falsa

**O que acontece.** Na v0, a correção nas perguntas armadilha foi 0,767, a menor entre os tipos. Duas das 15 armadilhas não foram recusadas: o modelo respondeu como se o parâmetro inventado existisse, usando trechos parecidos.

**Causa.** Sem verificação, o modelo tende a "ajudar" com o que encontrou de mais próximo.

**O que foi feito.** A v2 tem checagem de groundedness: afirmações sem apoio nos trechos levam a uma nova tentativa com um modelo maior e, se falhar de novo, a "não sei". O efeito dessa mudança ainda não foi medido (ver item 7).

## 6. Recusa por engano em perguntas respondíveis

**O que acontece.** Na v0, 4% das perguntas respondíveis (5 de 130) terminaram em "não encontrei na documentação".

**Causa provável.** O trecho certo não estava entre os 5 recuperados: na v0, o recall@5 por página é 0,889, ou seja, em cerca de 1 de cada 9 perguntas nenhuma página certa chega à geração.

**O que foi feito.** O retrieval híbrido subiu o recall@5 por página de 0,889 para 0,942 e o recall@20 de 0,855 para 0,903, e a rota multi-hop passa 8 trechos para a geração em vez de 5. O efeito na taxa de recusa ainda não foi medido.

## 7. Números que faltam

- O eval de ponta a ponta das versões v1, v2 e v3 não foi concluído: a conta da OpenAI ficou sem créditos no meio do eval da v1-hybrid (pergunta 58 de 160). Só há um resultado parcial da v1-hybrid, nas 58 primeiras perguntas (todas do tipo simples).
- A concordância do juiz com rótulos humanos (kappa de Cohen) não foi medida. O script está pronto em `eval/judge_calibration/`, mas depende de alguém rotular 50 respostas.
- A latência foi medida numa máquina de desenvolvimento com 4 perguntas em paralelo, e não reflete o deploy.
