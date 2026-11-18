"""Nó router: classifica a pergunta e decide o caminho no grafo. Usa o modelo small."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.graph.instrument import emit_data
from app.graph.state import AgentState
from app.llm.structured import call_structured
from app.pipeline import PipelineConfig

Route = Literal["trivial", "out_of_scope", "simple_rag", "multi_hop"]

ROUTER_PROMPT = """You are the router of a documentation assistant. The assistant answers questions about the \
open-source LangChain ecosystem: LangGraph, LangChain, Deep Agents and their integrations (models, vector \
stores, tools, middleware, MCP, streaming, persistence, agents, deployment of these libraries).

Classify the user message into exactly one route:

- trivial: greetings, thanks, small talk, or questions about the assistant itself ("hi", "what can you do?"). \
No documentation lookup is needed.
- out_of_scope: anything not about the LangChain ecosystem (general knowledge, other frameworks or tools, \
recipes, news, math, coding help unrelated to these libraries).
- simple_rag: a question about the LangChain ecosystem that one focused documentation passage can answer \
(one concept, one API, one error, one how-to).
- multi_hop: a question about the LangChain ecosystem that needs several pieces of documentation combined: \
comparisons ("difference between X and Y"), questions with two or more distinct parts, or questions that \
chain concepts ("how do I use X together with Y").

Rules:
- A question that mentions a LangChain-ecosystem name but assumes something odd or possibly false is still \
simple_rag or multi_hop: the documentation lookup is what reveals the false premise. Never send it to \
out_of_scope.
- When unsure between simple_rag and multi_hop, choose simple_rag.
- The message can be in any language.

Give a short reason (at most 12 words) and your confidence from 0 to 1."""


class RouteDecision(BaseModel):
    route: Route
    reason: str = Field(description="At most 12 words")
    confidence: float = Field(ge=0, le=1)


def make_router(pipeline: PipelineConfig):
    def router(state: AgentState) -> dict:
        decision, usage = call_structured("small", RouteDecision, ROUTER_PROMPT, state["query"], "router")
        emit_data("router", decision.model_dump())
        return {
            "route": decision.route,
            "route_reason": decision.reason,
            "route_confidence": decision.confidence,
            "usage": [usage],
        }

    return router
