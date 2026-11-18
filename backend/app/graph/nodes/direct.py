"""Rotas baratas, sem retrieval: resposta direta (trivial) e recusa (fora de escopo)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.graph.nodes.generate import FINAL_ANSWER_TAG
from app.graph.state import AgentState
from app.llm.factory import get_llm
from app.llm.usage import usage_from_message
from app.pipeline import PipelineConfig

DIRECT_PROMPT = """You are a documentation assistant for the open-source LangChain ecosystem (LangGraph, \
LangChain, Deep Agents and their integrations).

The user sent a greeting, small talk or a question about you. Reply briefly and in a friendly tone, in the \
user's language, and say that you can answer questions about that documentation. Do not answer technical \
questions here."""

# Recusa por template: não gasta tokens e não depende de modelo.
REFUSAL = {
    "pt": (
        "Só consigo responder perguntas sobre a documentação do ecossistema LangChain "
        "(LangGraph, LangChain, Deep Agents e integrações). Essa pergunta está fora desse escopo."
    ),
    "en": (
        "I can only answer questions about the LangChain ecosystem documentation "
        "(LangGraph, LangChain, Deep Agents and integrations). This question is outside that scope."
    ),
}
PORTUGUESE_HINTS = frozenset(
    "qual quais como quando onde quem por que porque não nao é e o a os as um uma de do da em no na para pra "
    "com que se meu minha você voce melhor fazer faço faco".split()
)


def detect_language(text: str) -> str:
    """Heurística barata: português se houver acento típico ou várias palavras comuns do idioma."""
    lowered = text.lower()
    if any(ch in lowered for ch in "ãõçáéíóúâêô"):
        return "pt"
    words = [w.strip("?!.,;:") for w in lowered.split()]
    return "pt" if sum(w in PORTUGUESE_HINTS for w in words) >= 2 else "en"


def make_direct_answer(pipeline: PipelineConfig):
    def direct_answer(state: AgentState) -> dict:
        message = get_llm("small").invoke(
            [SystemMessage(DIRECT_PROMPT), HumanMessage(state["query"])], config={"tags": [FINAL_ANSWER_TAG]}
        )
        answer = message.text if isinstance(getattr(message, "text", None), str) else str(message.content)
        usage = usage_from_message("direct_answer", get_settings().model_for("small"), message)
        return {"answer": answer, "citations": [], "model_tier_used": "small", "usage": [usage]}

    return direct_answer


def make_refuse(pipeline: PipelineConfig):
    def refuse(state: AgentState) -> dict:
        return {"answer": REFUSAL[detect_language(state["query"])], "citations": [], "model_tier_used": None}

    return refuse
