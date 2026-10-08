"""
Shared fixtures for the ontographrag regression suite.

All fixtures here are pure-Python — no live Neo4j or LLM calls.
Tests that need graph behaviour use the stub_graph fixture which
records Cypher calls without executing them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import pytest

from ontographrag.rag.systems._records import (
    RetrievedChunk,
    RetrievedEntity,
    RetrievedRelationship,
)

# ---------------------------------------------------------------------------
# Minimal chunk / entity / relationship factories
# ---------------------------------------------------------------------------


def make_chunk(
    text: str, kg_name: str = "test_kg", file_name: str = "doc.txt", score: float = 1.0
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
    eid: str, source: str = "entity_lookup", min_hops: int = 0, etype: str = "Disease"
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
        self, query_results: Sequence[list[dict[str, object]]] | None = None
    ) -> None:
        self._results: list[list[dict[str, object]]] = list(query_results or [])
        self.queries: list[
            tuple[str, dict[str, object]]
        ] = []  # recorded (query_str, params) pairs

    def query(
        self, query: str, params: dict[str, object] | None = None
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

    def __or__(self, other):
        # Support `prompt | llm | parser` chaining in tests
        return _Chain(self, other)


class _Chain:
    def __init__(self, *parts) -> None:
        self._parts = parts

    def __or__(self, other):
        return _Chain(*self._parts, other)

    def invoke(self, inputs):
        result = inputs
        for part in self._parts:
            if hasattr(part, "invoke"):
                result = part.invoke(result)
        return result


@pytest.fixture
def stub_llm() -> StubLLM:
    return StubLLM()


pytest_plugins = ["tests.offline_support"]
