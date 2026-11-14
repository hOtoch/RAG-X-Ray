"""Baixa a documentação do commit fixado no ADR-0001 para data/raw/."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import httpx

from app.config import ROOT

REPO = "langchain-ai/docs"
COMMIT = "4db8ed56a4c7c8cbdca47053d9eee4fc57124ffd"
# Páginas do corpus e snippets de código que elas importam.
PREFIXES = ("src/oss/", "src/snippets/")
EXTENSIONS = (".md", ".mdx")

RAW_DIR = ROOT / "data" / "raw"


def list_files(client: httpx.Client) -> list[str]:
    r = client.get(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}", params={"recursive": "1"})
    r.raise_for_status()
    tree = r.json()
    if tree.get("truncated"):
        raise RuntimeError("Árvore do GitHub veio truncada; o corpus ficaria incompleto")
    return sorted(
        item["path"]
        for item in tree["tree"]
        if item["type"] == "blob" and item["path"].startswith(PREFIXES) and item["path"].endswith(EXTENSIONS)
    )


def download(client: httpx.Client, path: str) -> bool:
    target = RAW_DIR / path
    if target.exists():
        return False
    r = client.get(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}")
    r.raise_for_status()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(r.content)
    return True


def fetch() -> list[str]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        paths = list_files(client)
        with ThreadPoolExecutor(max_workers=16) as pool:
            downloaded = sum(pool.map(lambda p: download(client, p), paths))
    (RAW_DIR / "manifest.json").write_text(
        json.dumps({"repo": REPO, "commit": COMMIT, "files": paths}, indent=2), encoding="utf-8"
    )
    print(f"fetch: {len(paths)} arquivos no commit {COMMIT[:8]} ({downloaded} baixados agora)")
    return paths


if __name__ == "__main__":
    fetch()
