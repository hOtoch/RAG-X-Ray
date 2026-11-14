"""Nó generate: responde usando só o contexto, com citações [n] ligadas aos chunks."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import Tier, get_settings
from app.graph.state import AgentState, Citation, ScoredChunk
from app.llm.factory import get_llm
from app.llm.usage import usage_from_message
from app.pipeline import PipelineConfig

# A parte fixa vem primeiro e a variável (contexto e pergunta) no fim: isso favorece o prompt caching.
SYSTEM_PROMPT = """You are a documentation assistant for LangChain, LangGraph and related open-source libraries.

Rules:
- Answer using ONLY the numbered context passages provided by the user message.
- After each claim, cite the passages that support it with their numbers in square brackets, like [1] or [2][3].
- If the context does not contain the answer, say that you don't know based on the documentation. Do not guess.
- If the question assumes something the context contradicts or does not support, say so.
- Be concise and technical. Include short code only when it is present in the context.
- Answer in the same language as the question."""

CITATION = re.compile(r"\[(\d+)\]")


def format_context(chunks: list[ScoredChunk]) -> str:
    parts = []
    for n, c in enumerate(chunks, start=1):
        header = c.title if not c.section else f"{c.title} > {c.section}"
        parts.append(f"[{n}] {header}\n{c.text}")
    return "\n\n---\n\n".join(parts)


def extract_citations(answer: str, chunks: list[ScoredChunk]) -> list[Citation]:
    """Mantém só as citações que apontam para um chunk existente, sem repetir."""
    seen: list[int] = []
    for match in CITATION.finditer(answer):
        n = int(match.group(1))
        if 1 <= n <= len(chunks) and n not in seen:
            seen.append(n)
    return [
        Citation(
            n=n,
            chunk_id=chunks[n - 1].id,
            title=chunks[n - 1].title,
            section=chunks[n - 1].section,
            url=chunks[n - 1].url,
        )
        for n in seen
    ]


FINAL_ANSWER_TAG = "final_answer"


def generate_answer(query: str, chunks: list[ScoredChunk], tier: Tier):
    """Chama o modelo do nível pedido. Retorna (resposta, citações, uso)."""
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(f"Context passages:\n\n{format_context(chunks)}\n\nQuestion: {query}"),
    ]
    # A tag marca esta chamada como a resposta final: só os tokens dela vão para o stream da interface.
    message = get_llm(tier).invoke(messages, config={"tags": [FINAL_ANSWER_TAG]})
    answer = message.text if isinstance(getattr(message, "text", None), str) else str(message.content)
    usage = usage_from_message("generate", get_settings().model_for(tier), message)
    return answer, extract_citations(answer, chunks), usage


def make_generate(pipeline: PipelineConfig):
    def generate(state: AgentState) -> dict:
        tier = pipeline.generate_tier
        answer, citations, usage = generate_answer(state["query"], state.get("context", []), tier)
        return {"answer": answer, "citations": citations, "model_tier_used": tier, "usage": [usage]}

    return generate
