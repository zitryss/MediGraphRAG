"""
Regression tests for runtime answer guardrail behavior.

Pure unit tests — no live Neo4j or remote LLM calls.
"""

import os
import sys
from typing import Protocol, Required, TypeVar

import pytest
from typing_extensions import TypedDict, Unpack

pytest.importorskip(
    "langchain_neo4j",
    reason="langchain_neo4j not installed — skipping guardrail behavior tests",
)
pytest.importorskip(
    "langchain",
    reason="langchain not installed — skipping guardrail behavior tests",
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ontographrag.rag.answer_guardrails import RUNTIME_GUARDRAIL_ABSTENTION
from ontographrag.rag.systems._records import (
    CompleteRetrievalContext,
    RetrievalContext,
)
from ontographrag.rag.systems._response_records import GenerationResponse, UsedEvidence
from ontographrag.rag.systems.enhanced_rag_system import EnhancedRAGSystem
from ontographrag.rag.systems.vanilla_rag_system import VanillaRAGSystem
from tests.conftest import unused_runnable

CallbackResult_co = TypeVar("CallbackResult_co", covariant=True)


class KeywordCallback(Protocol[CallbackResult_co]):
    def __call__(self, **kwargs: object) -> CallbackResult_co: ...


class ContextCallback(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> RetrievalContext: ...


class MockVerdict(TypedDict):
    decision: str
    answers_question: bool
    supported_by_context: bool
    reason: str


class MockMetadata(TypedDict):
    enabled: bool
    final_decision: str


class GuardrailArguments(TypedDict, total=False):
    question: str
    llm: object
    response: Required[str]
    context: Required[RetrievalContext]
    runtime_guardrail: bool
    runtime_guardrail_mode: str
    retry_factory: object


def _base_context() -> CompleteRetrievalContext:
    return {
        "chunks": [{"document": "doc.txt", "text": "supporting evidence"}],
        "documents": ["doc.txt"],
        "total_score": 0.9,
    }


def test_enhanced_retry_then_abstain_sets_final_decision_to_abstain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system: EnhancedRAGSystem = EnhancedRAGSystem.__new__(EnhancedRAGSystem)
    verdicts: list[MockVerdict] = [
        {
            "decision": "retry",
            "answers_question": False,
            "supported_by_context": True,
            "reason": "off target",
        },
        {
            "decision": "retry",
            "answers_question": False,
            "supported_by_context": True,
            "reason": "still off target",
        },
    ]

    next_verdict: KeywordCallback[MockVerdict] = lambda **_: verdicts.pop(0)
    monkeypatch.setattr(
        "ontographrag.rag.systems.enhanced_rag_system.evaluate_runtime_answer_guardrail",
        next_verdict,
    )

    response, _, metadata = system._apply_runtime_answer_guardrail(
        question="Who is the father-in-law of Helena Palaiologina?",
        llm=unused_runnable(),
        response="Lazar Brankovic",
        context=_base_context(),
        runtime_guardrail=True,
        runtime_guardrail_mode="retry_then_abstain",
        retry_factory=lambda: ("Still wrong", _base_context()),
    )

    assert response == RUNTIME_GUARDRAIL_ABSTENTION
    assert "retried" in metadata
    assert metadata["retried"] is True
    assert "final_decision" in metadata
    assert metadata["final_decision"] == "abstain"
    assert "retry_verdict" in metadata
    assert "decision" in metadata["retry_verdict"]
    assert metadata["retry_verdict"]["decision"] == "retry"


def test_vanilla_abstention_zeroes_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    system: VanillaRAGSystem = VanillaRAGSystem.__new__(VanillaRAGSystem)
    context: CompleteRetrievalContext = _base_context()

    context_callback: ContextCallback = lambda *_, **__: context
    monkeypatch.setattr(system, "get_vanilla_rag_context", context_callback)
    answer_callback: KeywordCallback[str] = lambda **_: "Wrong answer"
    monkeypatch.setattr(system, "_invoke_answer_chain", answer_callback)

    def guardrail_callback(**_: object) -> tuple[str, MockMetadata]:
        return (
            RUNTIME_GUARDRAIL_ABSTENTION,
            {"enabled": True, "final_decision": "abstain"},
        )

    monkeypatch.setattr(system, "_apply_runtime_answer_guardrail", guardrail_callback)

    result: GenerationResponse = system.generate_response(
        "Question?",
        llm=unused_runnable(),
        runtime_guardrail=True,
    )

    assert result["response"] == RUNTIME_GUARDRAIL_ABSTENTION
    assert result["confidence"] == 0.0


def test_enhanced_abstention_zeroes_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    system: EnhancedRAGSystem = EnhancedRAGSystem.__new__(EnhancedRAGSystem)
    context: CompleteRetrievalContext = {
        **_base_context(),
        "entities": {},
        "relationships": [],
        "entity_count": 1,
        "relationship_count": 0,
        "search_method": "vector_similarity",
    }

    context_callback: ContextCallback = lambda *_, **__: context
    monkeypatch.setattr(system, "get_rag_context", context_callback)
    answer_callback: KeywordCallback[str] = lambda **_: "Wrong answer"
    monkeypatch.setattr(system, "_invoke_answer_chain", answer_callback)

    def guardrail_callback(
        **kwargs: Unpack[GuardrailArguments],
    ) -> tuple[str, RetrievalContext, MockMetadata]:
        return (
            RUNTIME_GUARDRAIL_ABSTENTION,
            kwargs["context"],
            {"enabled": True, "final_decision": "abstain"},
        )

    monkeypatch.setattr(system, "_apply_runtime_answer_guardrail", guardrail_callback)

    def evidence_callback(response: str, context: RetrievalContext) -> UsedEvidence:
        return {
            "used_entities": [],
            "used_chunks": [],
            "reasoning_edges": [],
        }

    monkeypatch.setattr(system, "_extract_used_entities_and_chunks", evidence_callback)

    result: GenerationResponse = system.generate_response(
        "Question?",
        llm=unused_runnable(),
        runtime_guardrail=True,
        max_hops=1,
    )

    assert result["response"] == RUNTIME_GUARDRAIL_ABSTENTION
    assert result["confidence"] == 0.0


def test_enhanced_retrieval_span_context_does_not_retry_vector_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system: EnhancedRAGSystem = EnhancedRAGSystem.__new__(EnhancedRAGSystem)
    context: CompleteRetrievalContext = {
        **_base_context(),
        "entities": {},
        "relationships": [],
        "entity_count": 0,
        "relationship_count": 0,
        "search_method": "retrieval_span_similarity",
    }

    context_callback: ContextCallback = lambda *_, **__: context
    monkeypatch.setattr(system, "get_rag_context", context_callback)
    answer_callback: KeywordCallback[str] = lambda **_: "Insufficient Information"
    monkeypatch.setattr(system, "_invoke_answer_chain", answer_callback)

    def guardrail_callback(
        **kwargs: Unpack[GuardrailArguments],
    ) -> tuple[str, RetrievalContext, MockMetadata]:
        return (
            kwargs["response"],
            kwargs["context"],
            {"enabled": False, "final_decision": "keep"},
        )

    monkeypatch.setattr(system, "_apply_runtime_answer_guardrail", guardrail_callback)
    fail_vector_search: ContextCallback = lambda *_, **__: pytest.fail(
        "pure vector context should not retry vector-only",
    )
    monkeypatch.setattr(system, "_vector_similarity_search", fail_vector_search)

    def evidence_callback(response: str, context: RetrievalContext) -> UsedEvidence:
        return {
            "used_entities": [],
            "used_chunks": [],
            "reasoning_edges": [],
        }

    monkeypatch.setattr(system, "_extract_used_entities_and_chunks", evidence_callback)

    result: GenerationResponse = system.generate_response(
        "Question?",
        llm=unused_runnable(),
        max_hops=1,
    )

    assert result["response"] == "Insufficient Information"
