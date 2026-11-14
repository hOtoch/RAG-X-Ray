from ingest.chunk import chunk_document, count_tokens, split_blocks, split_sections
from ingest.load import Document, doc_id, doc_url, keep_language, load_document

SAMPLE = """Intro paragraph.

## Install

Run this:

```python
# not a heading
x = 1

y = 2
```

### Details

More text.

## Usage

Final text.
"""


def make_doc(text: str) -> Document:
    return Document(
        id="doc-a", path="src/oss/a.mdx", url="https://x/oss/python/a", title="A", description="", text=text
    )


def test_doc_id_and_url():
    assert doc_id("langgraph/graph-api.mdx") == "doc-langgraph-graph-api"
    assert doc_url("langgraph/graph-api.mdx").endswith("/oss/python/langgraph/graph-api")
    assert doc_url("javascript/integrations/x.mdx").endswith("/oss/javascript/integrations/x")


def test_keep_language():
    text = "a\n:::python\npy\n:::\n:::js\njs\n:::\nb"
    assert "py" in keep_language(text, "python") and "js" not in keep_language(text, "python")
    assert "js" in keep_language(text, "js") and "py" not in keep_language(text, "js")


def test_sections_ignore_headings_inside_code():
    sections = split_sections(SAMPLE)
    assert [h for h, _ in sections] == [[], ["Install"], ["Install", "Details"], ["Usage"]]
    assert "# not a heading" in sections[1][1]


def test_code_block_is_one_block():
    blocks = split_blocks(split_sections(SAMPLE)[1][1])
    assert len(blocks) == 2
    assert blocks[1].startswith("```python") and blocks[1].endswith("```")


def test_chunk_ids_and_metadata():
    chunks = chunk_document(make_doc(SAMPLE))
    assert [c.id for c in chunks] == [f"doc-a#chunk-{i}" for i in range(len(chunks))]
    install = next(c for c in chunks if c.section == "Install")
    assert install.url.endswith("#install")
    assert next(c for c in chunks if c.section == "Install > Details").url.endswith("#details")
    assert install.embedding_text.startswith("A > Install")


def test_long_text_is_split_within_budget():
    text = "## S\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(30))
    chunks = chunk_document(make_doc(text), max_tokens=200, overlap_tokens=0)
    assert len(chunks) > 5
    assert all(c.token_count <= 200 for c in chunks)


def test_oversized_code_is_split_and_fenced():
    code = "```python\n" + "\n".join(f"value_{i} = {i}" for i in range(400)) + "\n```"
    chunks = chunk_document(make_doc("## C\n\n" + code), max_tokens=200, overlap_tokens=0)
    assert len(chunks) > 1
    assert all(c.text.startswith("```python") and c.text.rstrip().endswith("```") for c in chunks)
    assert all(count_tokens(c.text) <= 210 for c in chunks)


def test_load_document_resolves_snippets(tmp_path):
    raw = tmp_path
    (raw / "src/oss/langgraph").mkdir(parents=True)
    (raw / "src/snippets").mkdir(parents=True)
    (raw / "src/snippets/a-py.mdx").write_text("```python\nprint('hi')\n```\n", encoding="utf-8")
    page = raw / "src/oss/langgraph/page.mdx"
    page.write_text(
        "---\ntitle: My Page\ndescription: Desc\n---\n\n"
        "import APy from '/snippets/a-py.mdx';\n\n"
        "Some intro text that is long enough to be kept as a real page of documentation.\n\n"
        ":::python\n<APy />\n:::\n:::js\nconsole.log(1)\n:::\n",
        encoding="utf-8",
    )
    doc = load_document(page, raw)
    assert doc is not None
    assert doc.title == "My Page" and doc.id == "doc-langgraph-page"
    assert "print('hi')" in doc.text
    assert "console.log" not in doc.text and "import APy" not in doc.text
