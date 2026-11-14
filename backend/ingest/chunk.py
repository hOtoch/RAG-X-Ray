"""Divide os documentos em chunks que respeitam headers e blocos de código."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from ingest.load import Document

MAX_TOKENS = 400
OVERLAP_TOKENS = 60
MIN_TOKENS = 12
HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$")


@dataclass
class Chunk:
    id: str
    doc_id: str
    ord: int
    title: str
    section: str
    url: str
    text: str
    token_count: int

    @property
    def embedding_text(self) -> str:
        """Texto enviado ao modelo de embedding: o contexto do título ajuda a busca."""
        header = self.title if not self.section else f"{self.title} > {self.section}"
        return f"{header}\n\n{self.text}"


@lru_cache
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text, disallowed_special=()))


def anchor(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def split_blocks(text: str) -> list[str]:
    """Separa em parágrafos, mantendo cada bloco de código inteiro."""
    blocks: list[str] = []
    current: list[str] = []
    in_code = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            current.append(line)
            continue
        if not in_code and not line.strip():
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def split_sections(text: str) -> list[tuple[list[str], str]]:
    """Retorna (caminho de headers, corpo) para cada seção, ignorando '#' dentro de código."""
    sections: list[tuple[list[str], str]] = []
    path: list[tuple[int, str]] = []
    body: list[str] = []
    in_code = False

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append(([h for _, h in path], content))
        body.clear()

    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        match = None if in_code else HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            path[:] = [(lv, h) for lv, h in path if lv < level] + [(level, match.group(2))]
        else:
            body.append(line)
    flush()
    return sections


def hard_wrap(lines: list[str], max_tokens: int) -> list[str]:
    """Linha única maior que o limite (tabela ou JSON em uma linha): corta por tokens."""
    enc = _encoder()
    out: list[str] = []
    for line in lines:
        tokens = enc.encode(line, disallowed_special=())
        if len(tokens) <= max_tokens:
            out.append(line)
        else:
            step = max_tokens - 20
            out.extend(enc.decode(tokens[i : i + step]) for i in range(0, len(tokens), step))
    return out


def split_oversized(block: str, max_tokens: int) -> list[str]:
    """Bloco maior que o limite (em geral código longo): corta por linhas, reabrindo a cerca de código."""
    lines = block.split("\n")
    fence = lines[0] if lines[0].lstrip().startswith("```") else None
    parts: list[str] = []
    current: list[str] = []
    for line in hard_wrap(lines, max_tokens):
        if current and count_tokens("\n".join([*current, line])) > max_tokens:
            parts.append("\n".join(current) + ("\n```" if fence else ""))
            current = [fence] if fence else []
        current.append(line)
    if current:
        parts.append("\n".join(current))
    return parts


def pack(blocks: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    pieces: list[str] = []
    for block in blocks:
        pieces.extend(split_oversized(block, max_tokens) if count_tokens(block) > max_tokens else [block])

    chunks: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if current and count_tokens("\n\n".join([*current, piece])) > max_tokens:
            chunks.append("\n\n".join(current))
            last = current[-1]
            # Sobreposição: repete o último parágrafo se ele for curto e não for código.
            keep = count_tokens(last) <= overlap_tokens and not last.lstrip().startswith("```")
            current = [last] if keep else []
        current.append(piece)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_document(
    doc: Document, max_tokens: int = MAX_TOKENS, overlap_tokens: int = OVERLAP_TOKENS
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for headers, body in split_sections(doc.text):
        section = " > ".join(headers)
        url = f"{doc.url}#{anchor(headers[-1])}" if headers else doc.url
        for text in pack(split_blocks(body), max_tokens, overlap_tokens):
            n = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{doc.id}#chunk-{n}",
                    doc_id=doc.id,
                    ord=n,
                    title=doc.title,
                    section=section,
                    url=url,
                    text=text,
                    token_count=count_tokens(text),
                )
            )
    return chunks


def chunk_corpus(docs: list[Document]) -> list[Chunk]:
    # Chunks com menos de MIN_TOKENS (um link solto, uma linha de tabela) só adicionam ruído ao índice.
    chunks = [c for d in docs for c in chunk_document(d) if c.token_count >= MIN_TOKENS]
    sizes = [c.token_count for c in chunks]
    print(
        f"chunk: {len(chunks)} chunks de {len(docs)} documentos | tokens min={min(sizes)} "
        f"mediana={int(statistics.median(sizes))} max={max(sizes)} total={sum(sizes)}"
    )
    return chunks
