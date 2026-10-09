"""
Shared fixtures for the ontographrag regression suite.

All fixtures here are pure-Python — no live Neo4j or LLM calls.
Tests that need graph behaviour use the stub_graph fixture which
records Cypher calls without executing them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Never, Protocol, cast

import pytest

from ontographrag.rag.systems._records import (
    RetrievedChunk,
    RetrievedEntity,
    RetrievedRelationship,
)

if TYPE_CHECKING:
    from langchain_core.prompt_values import PromptValue
    from langchain_core.runnables import Runnable

    from ontographrag.providers.model_providers import MessageContent, ModelProvider


def unused_runnable() -> Runnable[PromptValue, object]:
    """A real Runnable boundary that fails if an intercepted test calls it."""
    from langchain_core.runnables import RunnableLambda

    def reject_invoke(value: PromptValue) -> Never:
        pytest.fail("The isolated test must not invoke its model dependency")

    return RunnableLambda(reject_invoke)


def required_text_field(record: Mapping[str, object], key: str) -> str:
    """Assert the string shape expected by a test before comparing its content."""
    assert key in record
    value: object = record[key]
    assert isinstance(value, str)
    return value


def required_int_field(record: Mapping[str, object], key: str) -> int:
    """Assert the integer shape expected by a test without coercing the value."""
    assert key in record
    value: object = record[key]
    assert isinstance(value, int)
    return value


class UnusedGenerationProvider:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        /,
    ) -> Never:
        pytest.fail("The isolated test must not invoke its generation dependency")


class UnusedGraph:
    def query(
        self,
        query: str,
        params: dict[str, object] | None = None,
    ) -> Never:
        pytest.fail("The isolated test must not query its graph dependency")


def unused_model_provider() -> ModelProvider:
    """Create the actual nominal extension boundary, without an SDK client."""
    from ontographrag.providers.model_providers import ModelProvider

    class UnusedProvider(ModelProvider):
        def generate(
            self,
            system_prompt: MessageContent,
            user_prompt: MessageContent,
            model: str,
            **kwargs: object,
        ) -> Never:
            pytest.fail("The isolated test must not invoke its provider dependency")

    return UnusedProvider()


# ---------------------------------------------------------------------------
# Minimal chunk / entity / relationship factories
# ---------------------------------------------------------------------------


def make_chunk(
    text: str,
    kg_name: str = "test_kg",
    file_name: str = "doc.txt",
    score: float = 1.0,
) -> RetrievedChunk:
    chunk_id: str = hashlib.sha1(f"{kg_name}:{file_name}:{text}".encode()).hexdigest()
    return {
        "text": text,
        "chunk_id": chunk_id,
        "score": score,
        "document": file_name,
        "kg_name": kg_name,
        "entities": [],
        "linked_entity_ids": [],
    }


def make_entity(
    eid: str,
    source: str = "entity_lookup",
    min_hops: int = 0,
    etype: str = "Disease",
) -> RetrievedEntity:
    return {
        "id": eid,
        "name": eid,
        "type": etype,
        "description": eid,
        "source": source,
        "min_hops": min_hops,
    }


def make_relationship(
    src: str,
    tgt: str,
    rel_type: str = "TREATS",
    negated: bool = False,
    condition: str | None = None,
    quantitative: str | None = None,
) -> RetrievedRelationship:
    key: str = f"{src}-{rel_type}-{tgt}"
    return {
        "key": key,
        "source": src,
        "target": tgt,
        "type": rel_type,
        "negated": negated,
        "condition": condition,
        "quantitative": quantitative,
    }


# ---------------------------------------------------------------------------
# Neo4j graph stub — records calls, returns configurable results
# ---------------------------------------------------------------------------


class StubGraph:
    """Minimal stand-in for langchain_neo4j.Neo4jGraph.

    Tests configure `query_results` as a list of return values.
    Each call to .query() pops the first entry from the list.
    """

    def __init__(
        self,
        query_results: Sequence[list[dict[str, object]]] | None = None,
    ) -> None:
        self._results: list[list[dict[str, object]]] = list(query_results or [])
        self.queries: list[
            tuple[str, dict[str, object]]
        ] = []  # recorded (query_str, params) pairs

    def query(
        self,
        query: str,
        params: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        self.queries.append((query, params or {}))
        if self._results:
            return self._results.pop(0)
        return []

    def last_query(self) -> tuple[str, dict[str, object]] | None:
        return self.queries[-1] if self.queries else None


@pytest.fixture
def stub_graph() -> StubGraph:
    return StubGraph()


# ---------------------------------------------------------------------------
# Minimal LLM stub
# ---------------------------------------------------------------------------


class StubLLM:
    """Minimal stand-in for a LangChain Runnable that returns a fixed string."""

    def __init__(self, response: str = "[]") -> None:
        self._response: str = response

    def invoke(self, inputs: object) -> str:
        return self._response

    def __or__(self, other: object) -> _Chain:
        # Support `prompt | llm | parser` chaining in tests
        return _Chain(self, other)


class _Invokable(Protocol):
    def invoke(self, inputs: object) -> object: ...


class _Chain:
    def __init__(self, *parts: object) -> None:
        self._parts: tuple[object, ...] = parts

    def __or__(self, other: object) -> _Chain:
        return _Chain(*self._parts, other)

    def invoke(self, inputs: object) -> object:
        result: object = inputs
        for part in self._parts:
            if hasattr(part, "invoke"):
                # The chain intentionally uses duck typing after the existing
                # attribute guard; non-callable invoke attributes still fail.
                result = cast(_Invokable, part).invoke(result)
        return result


@pytest.fixture
def stub_llm() -> StubLLM:
    return StubLLM()


pytest_plugins: list[str] = ["tests.offline_support"]
