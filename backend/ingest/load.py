"""Transforma os arquivos .mdx em documentos limpos com metadados (data/corpus.jsonl)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from app.config import ROOT
from ingest.fetch import RAW_DIR

CORPUS_FILE = ROOT / "data" / "corpus.jsonl"
SITE = "https://docs.langchain.com"

FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
IMPORT = re.compile(r"^import\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"];?\s*$", re.MULTILINE)
COMPONENT = re.compile(r"^[ \t]*<(\w+)\s*/>[ \t]*$", re.MULTILINE)
LANG_BLOCK = re.compile(r"^:::(python|js)[ \t]*\n(.*?)^:::[ \t]*$", re.DOTALL | re.MULTILINE)
MERMAID = re.compile(r"```mermaid\n.*?```", re.DOTALL)
REF_LINK = re.compile(r"@\[(`[^`\]]+`|[^\]]+)\](?:\[[^\]]*\])?")


@dataclass
class Document:
    id: str
    path: str
    url: str
    title: str
    description: str
    text: str


def doc_id(rel: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", rel.lower().removesuffix(".mdx").removesuffix(".md")).strip("-")
    return f"doc-{slug}"


def doc_url(rel: str) -> str:
    """rel é o caminho dentro de src/oss. Páginas sem linguagem no caminho são servidas em /oss/python/."""
    page = rel.removesuffix(".mdx").removesuffix(".md").removesuffix("/index")
    first = page.split("/")[0]
    if first not in ("python", "javascript"):
        page = f"python/{page}"
    return f"{SITE}/oss/{page}"


def keep_language(text: str, language: str) -> str:
    """Mantém o conteúdo dos blocos :::python ou :::js da linguagem escolhida e remove os da outra."""
    return LANG_BLOCK.sub(lambda m: m.group(2) if m.group(1) == language else "", text)


def resolve_snippets(text: str, raw_dir: Path, language: str, depth: int = 0) -> str:
    imports = dict(IMPORT.findall(text))
    text = IMPORT.sub("", text)

    def replace(match: re.Match[str]) -> str:
        source = imports.get(match.group(1))
        if not source or depth >= 2:
            return ""
        file = raw_dir / "src" / source.lstrip("/")
        if not file.exists():
            return ""
        snippet = FRONTMATTER.sub("", file.read_text(encoding="utf-8"))
        return resolve_snippets(keep_language(snippet, language), raw_dir, language, depth + 1).strip()

    return COMPONENT.sub(replace, text)


def clean(text: str) -> str:
    text = MERMAID.sub("", text)
    text = REF_LINK.sub(lambda m: m.group(1), text)  # @[`Send`] -> `Send`
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_document(path: Path, raw_dir: Path = RAW_DIR) -> Document | None:
    rel = path.relative_to(raw_dir / "src" / "oss").as_posix()
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    meta: dict = {}
    match = FRONTMATTER.match(raw)
    if match:
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        raw = raw[match.end() :]
    language = "js" if rel.startswith("javascript/") else "python"
    text = clean(resolve_snippets(keep_language(raw, language), raw_dir, language))
    if len(text) < 80:
        return None  # página só de índice ou redirecionamento
    title = str(meta.get("title") or path.stem.replace("-", " ").title())
    return Document(
        id=doc_id(rel),
        path=f"src/oss/{rel}",
        url=doc_url(rel),
        title=title,
        description=str(meta.get("description") or ""),
        text=text,
    )


def load() -> list[Document]:
    root = RAW_DIR / "src" / "oss"
    files = sorted(p for p in root.rglob("*") if p.suffix in (".md", ".mdx"))
    docs = [d for d in (load_document(p) for p in files) if d]
    ids = [d.id for d in docs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("ids de documento duplicados")
    CORPUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS_FILE.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
    print(f"load: {len(docs)} documentos de {len(files)} arquivos -> {CORPUS_FILE}")
    return docs


def read_corpus() -> list[Document]:
    with CORPUS_FILE.open(encoding="utf-8") as f:
        return [Document(**json.loads(line)) for line in f]


if __name__ == "__main__":
    load()
