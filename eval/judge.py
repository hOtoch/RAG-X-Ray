"""LLM-as-judge: faithfulness, precisão de citação e correção da resposta.

São duas chamadas separadas de propósito. A de faithfulness vê o contexto e a resposta, mas não a
referência: mede se a resposta se apoia no que foi recuperado. A de correção vê a referência e a
resposta, mas não o contexto: mede se a resposta está certa.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.config import Tier, get_settings
from app.llm.factory import get_llm
from app.llm.usage import usage_from_message

JUDGE_TIER: Tier = "large"

FAITHFULNESS_PROMPT = """You are a strict evaluator of a documentation QA system.

You receive numbered context passages and an answer that cites passages like [1] or [2][3].

1. Split the answer into its factual claims (atomic statements about the libraries, APIs or behavior). \
Ignore meta statements such as "based on the documentation" or "I don't know".
2. For each claim:
   - supported: true only if the claim can be inferred from the context passages as a whole.
   - cited: the passage numbers the answer attaches to that claim (empty list if none).
   - supported_by_cited: true only if the cited passages themselves support the claim. False if the claim \
has citations that do not support it. False if cited is empty.
3. is_refusal: true if the answer as a whole declines to answer, says it does not know, or says the \
documentation does not cover the question or does not support its premise."""

CORRECTNESS_PROMPT = """You are a strict evaluator of a documentation QA system.

Compare the answer with the reference answer for the question.

- correct: the answer conveys the essential content of the reference and says nothing that contradicts it. \
Extra correct detail is fine.
- partially_correct: the answer gets part of the reference right but misses an essential point, or mixes \
correct content with a minor error.
- incorrect: the answer contradicts the reference, misses its main point, or fails to answer.

When the reference says the assistant should decline or should point out a false premise, an answer is \
correct only if it does that. An answer that goes along with a false premise and invents details is incorrect.
The answer may be in a different language than the reference; judge the content, not the language."""


class Claim(BaseModel):
    claim: str
    supported: bool
    cited: list[int] = Field(default_factory=list)
    supported_by_cited: bool = False


class FaithfulnessVerdict(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    is_refusal: bool = False


class CorrectnessVerdict(BaseModel):
    verdict: Literal["correct", "partially_correct", "incorrect"]
    reason: str = Field(description="One short sentence")


CORRECTNESS_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


class JudgeResult(BaseModel):
    faithfulness: float | None  # None quando a resposta não faz afirmações (ex.: recusa)
    citation_precision: float | None  # None quando nenhuma afirmação tem citação
    correctness: float
    correctness_verdict: str
    correctness_reason: str
    is_refusal: bool
    n_claims: int
    unsupported_claims: list[str]
    judge_cost_usd: float


def _invoke(schema: type[BaseModel], system: str, human: str) -> tuple[BaseModel, float]:
    llm = get_llm(JUDGE_TIER).with_structured_output(schema, include_raw=True)
    out = llm.invoke([("system", system), ("human", human)])
    cost = usage_from_message("judge", get_settings().model_for(JUDGE_TIER), out["raw"]).cost_usd
    if out["parsed"] is None:
        raise ValueError(f"Juiz devolveu saída inválida: {out.get('parsing_error')}")
    return out["parsed"], cost


def score_faithfulness(verdict: FaithfulnessVerdict) -> tuple[float | None, float | None]:
    """(faithfulness, precisão de citação) a partir do veredito por afirmação."""
    claims = verdict.claims
    faithfulness = sum(c.supported for c in claims) / len(claims) if claims else None
    cited = [c for c in claims if c.cited]
    citation_precision = sum(c.supported_by_cited for c in cited) / len(cited) if cited else None
    return faithfulness, citation_precision


def judge(question: str, answer: str, context: str, reference: str) -> JudgeResult:
    faith, cost_a = _invoke(
        FaithfulnessVerdict, FAITHFULNESS_PROMPT, f"Context passages:\n\n{context}\n\nAnswer to evaluate:\n{answer}"
    )
    correct, cost_b = _invoke(
        CorrectnessVerdict,
        CORRECTNESS_PROMPT,
        f"Question: {question}\n\nReference answer: {reference}\n\nAnswer to evaluate:\n{answer}",
    )
    faithfulness, citation_precision = score_faithfulness(faith)
    return JudgeResult(
        faithfulness=faithfulness,
        citation_precision=citation_precision,
        correctness=CORRECTNESS_SCORE[correct.verdict],
        correctness_verdict=correct.verdict,
        correctness_reason=correct.reason,
        is_refusal=faith.is_refusal,
        n_claims=len(faith.claims),
        unsupported_claims=[c.claim for c in faith.claims if not c.supported],
        judge_cost_usd=cost_a + cost_b,
    )
