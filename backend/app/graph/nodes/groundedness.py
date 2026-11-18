"""Checagem de groundedness: verifica se a resposta se apoia nos trechos. Se não, escala o modelo
uma vez; se ainda falhar, responde que não sabe."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.config import Tier
from app.graph.instrument import emit_data
from app.graph.nodes.direct import detect_language
from app.graph.nodes.generate import format_context
from app.graph.state import AgentState, spent_usd
from app.llm.structured import call_structured
from app.pipeline import PipelineConfig, budget_usd

GROUNDEDNESS_PROMPT = """You verify the answer of a documentation assistant against the passages it was given.

List in `unsupported_claims` every factual claim in the answer that the passages do not support (quote or \
paraphrase each briefly). Ignore statements that the documentation does not cover something, and ignore \
"I don't know" statements: those are not claims.

grounded is true only when unsupported_claims is empty."""

DONT_KNOW = {
    "pt": "Não encontrei na documentação informação suficiente para responder a essa pergunta com segurança.",
    "en": "I could not find enough information in the documentation to answer this question reliably.",
}


class GroundednessVerdict(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)


TIER_UP: dict[str, Tier] = {"small": "medium", "medium": "large"}


def decide(grounded: bool, state: AgentState, pipeline: PipelineConfig) -> str:
    """finish se a resposta está sustentada; escalate uma vez se ainda há nível acima e orçamento; senão give_up."""
    if grounded:
        return "finish"
    tier = state.get("model_tier_used") or "large"
    can_escalate = pipeline.escalate and state.get("escalations", 0) < 1 and tier in TIER_UP
    if can_escalate and spent_usd(state) < budget_usd(pipeline):
        return "escalate"
    return "give_up"


def make_groundedness_check(pipeline: PipelineConfig):
    def groundedness_check(state: AgentState) -> dict:
        context = format_context(state.get("context", []))
        human = f"Passages:\n\n{context}\n\nAnswer to verify:\n{state.get('answer', '')}"
        verdict, usage = call_structured("small", GroundednessVerdict, GROUNDEDNESS_PROMPT, human, "groundedness_check")
        grounded = verdict.grounded and not verdict.unsupported_claims
        action = decide(grounded, state, pipeline)
        emit_data(
            "groundedness_check",
            {"grounded": grounded, "unsupported_claims": verdict.unsupported_claims, "next_action": action},
        )
        return {
            "grounded": grounded,
            "unsupported_claims": verdict.unsupported_claims,
            "next_action": action,
            "usage": [usage],
        }

    return groundedness_check


def make_escalate(pipeline: PipelineConfig):
    """Sobe um nível de modelo para a próxima tentativa de geração."""

    def escalate(state: AgentState) -> dict:
        tier = TIER_UP[state.get("model_tier_used") or "medium"]
        emit_data("escalate", {"from_tier": state.get("model_tier_used"), "to_tier": tier})
        return {"next_tier": tier, "escalations": state.get("escalations", 0) + 1}

    return escalate


def make_give_up(pipeline: PipelineConfig):
    """Última saída: nem o modelo mais forte produziu uma resposta sustentada pelos trechos."""

    def give_up(state: AgentState) -> dict:
        return {"answer": DONT_KNOW[detect_language(state["query"])], "citations": [], "gave_up": True}

    return give_up
