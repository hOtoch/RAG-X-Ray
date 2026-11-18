# ADR-0008: Mapa UMAP gerado na ingestão; a pergunta é posicionada pelos vizinhos

- **Status:** Aceito
- **Data:** 2026-11-18

## Contexto

A interface mostra um mapa 2D do corpus com a pergunta e os trechos recuperados (issues #64 a #66). O mapa responde a duas perguntas que o ranking sozinho não responde: em que área da documentação a pergunta caiu, e se os trechos vieram de um lugar só ou de áreas diferentes (um sinal de pergunta ambígua ou multi-hop).

O UMAP de 10.800 vetores de 1536 dimensões leva de 80 a 140 s em CPU e depende de `umap-learn`, que puxa `numba` e `scikit-learn`. Para posicionar uma pergunta nova há duas opções:

1. **`UMAP.transform`**: carregar o modelo ajustado no servidor e projetar o embedding da pergunta.
2. **Média dos vizinhos**: colocar a pergunta na média das posições dos k vizinhos mais próximos da busca densa, que o retrieve já calculou.

## Decisão

O UMAP roda **uma vez, na ingestão** (`backend/ingest/corpus_map.py`), num grupo opcional de dependências (`uv sync --group map`). As coordenadas vão para a tabela `chunk_map` e para o arquivo estático `frontend/public/corpus_map.json` (0,8 MB), que a interface desenha também no modo replay.

A pergunta é posicionada pela **opção 2**: a média das posições dos 10 primeiros vizinhos densos, com peso 1/rank (`backend/app/retrieval/embedding_map.py`). O evento `embedding_map` sai logo depois do `retrieve`.

As coordenadas são normalizadas pelos quantis 0,5% e 99,5%, e não pelo mínimo e pelo máximo. Na primeira versão, meia dúzia de trechos isolados encolhia o resto do mapa no centro do painel.

## Consequências

- O servidor não carrega `numba` nem o modelo UMAP, e o custo por request é um dicionário em memória e uma média de 10 pontos.
- O ponto da pergunta fica sempre no meio dos trechos recuperados. É isso que o mapa quer mostrar, mas não é a projeção "verdadeira": uma pergunta cujos vizinhos estão em duas regiões distantes aparece entre elas, num lugar onde talvez não haja nenhum trecho. As linhas da pergunta até cada trecho deixam isso visível.
- O mapa precisa ser gerado de novo depois de cada ingestão (os ids dos chunks mudam). Se a tabela não existir, o evento simplesmente não é emitido. O servidor guarda as coordenadas em memória, então precisa ser reiniciado depois de gerar o mapa.
- O UMAP é estocástico; `random_state=42` deixa o mapa reprodutível na mesma máquina e na mesma versão da biblioteca.

## Alternativas consideradas

- **`UMAP.transform` no servidor**: a projeção correta, mas soma ~200 MB de dependências e alguns segundos de JIT do `numba` na primeira chamada, para um ganho só visual.
- **PCA**: é linear e roda em milissegundos, inclusive para a pergunta, mas não separa as áreas da documentação (a variância de embeddings de texto se espalha por muitas dimensões).
- **t-SNE**: não tem `transform` e preserva pior a estrutura global.
