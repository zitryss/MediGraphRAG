"""Stateful retrieval adapter tests using real system constructors."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from ontographrag.rag.systems import _vector_search as vector
from ontographrag.rag.systems import enhanced_rag_system as enhanced
from ontographrag.rag.systems import vanilla_rag_system as vanilla


@pytest.fixture
def system(monkeypatch: pytest.MonkeyPatch) -> enhanced.EnhancedRAGSystem:
    model = Mock(spec=["embed_query"])
    model.embed_query.return_value = [1.0, 0.0, 0.0]
    monkeypatch.setattr(enhanced, "load_embedding_model", Mock(return_value=(model, 3)))
    return enhanced.EnhancedRAGSystem(
        neo4j_uri="bolt://unit.invalid", neo4j_user="unit", neo4j_password="dummy"
    )


def test_real_constructor_cache_and_connection_lifecycle(
    system: enhanced.EnhancedRAGSystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = Mock()
    factory = Mock(return_value=graph)
    monkeypatch.setattr(enhanced, "Neo4jGraph", factory)
    assert system._create_neo4j_connection() is graph
    assert system._create_neo4j_connection() is graph
    assert factory.call_count == 1
    assert system._generate_query_embedding("q") == [1, 0, 0]
    assert system._generate_query_embedding("q") == [1, 0, 0]
    assert isinstance(system.embedding_model, Mock)
    assert system.embedding_model.embed_query.call_count == 1
    system.clear_retrieval_caches()
    assert system._graph is None
    assert system._embedding_cache == {}
    assert system._vector_index_available is None


def test_embedding_encode_interface_and_fifo_eviction(
    system: enhanced.EnhancedRAGSystem,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = Mock(return_value=np.array([0.5, 0.5]))
    monkeypatch.setattr(system, "embedding_model", SimpleNamespace(encode=encoder))
    system._embedding_cache = {str(i): [0] for i in range(2048)}
    assert system._generate_query_embedding("new") == [0.5, 0.5]
    assert "0" not in system._embedding_cache
    assert len(system._embedding_cache) == 2048
    monkeypatch.setattr(system, "embedding_model", object())
    with pytest.raises(ValueError, match="Unsupported embedding"):
        system._generate_query_embedding("unsupported")


@pytest.mark.parametrize("available", [True, False])
def test_vector_index_probe_and_cache(
    system: enhanced.EnhancedRAGSystem, available: bool
) -> None:
    graph = Mock()
    graph.query.side_effect = (
        [[{"name": "retrieval_vector", "state": "ONLINE"}], []] if available else [[]]
    )
    system._graph = graph
    assert system.check_vector_index() is available
    count = graph.query.call_count
    assert system.check_vector_index() is available
    assert graph.query.call_count == count
    if available:
        assert system._active_vector_index_name == "retrieval_vector"
        assert graph.query.call_args.args[1] == {"query_vector": [1.0, 0.0, 0.0]}


@pytest.mark.parametrize("entities", [True, False])
def test_late_interaction_corpus_cache_scope_and_fallback(
    system: enhanced.EnhancedRAGSystem, entities: bool
) -> None:
    graph = Mock()
    graph.query.side_effect = [[], [{"text": "passage", "chunk_id": "c"}]]
    opts: dict[str, Any] = {
        "kg_name": "unit",
        "document_names": ["b", "a"],
        "question_id": "q",
        "include_entities": entities,
    }
    rows = system._late_interaction_corpus_rows(graph, **opts)
    assert rows == [{"text": "passage", "chunk_id": "c"}]
    rows[0]["text"] = "mutated"
    assert system._late_interaction_corpus_rows(graph, **opts)[0]["text"] == "passage"
    assert graph.query.call_count == 2
    assert graph.query.call_args.args[1]["question_id"] == "q"
    assert graph.query.call_args.args[1]["kg_name"] == "unit"
    assert ("HAS_ENTITY" in graph.query.call_args.args[0]) is entities


@pytest.mark.parametrize("index", ["vector", "retrieval_vector"])
def test_vector_search_maps_scoped_result(
    system: enhanced.EnhancedRAGSystem, index: str
) -> None:
    system._vector_index_available = True
    system._active_vector_index_name = index
    row = {
        "text": "evidence",
        "chunk_id": "c",
        "chunk_element_id": "ce",
        "document": "doc",
        "score": 0.8,
        "kg_name": "unit",
        "question_id": "q",
        "entities": [],
    }
    graph = Mock()
    graph.query.return_value = [row]
    result = system._vector_similarity_search(
        graph,
        "query",
        document_names=["doc"],
        kg_name="unit",
        question_id="q",
        max_chunks=2,
    )
    assert result["chunks"][0]["text"] == "evidence"
    assert result["total_score"] == 0.8
    query, params = graph.query.call_args.args
    assert params["kg_name"] == "unit"
    assert params["question_id"] == "q"
    assert params["document_names"] == ["doc"]
    assert "40" in query


def test_text_search_maps_result_and_error(system: enhanced.EnhancedRAGSystem) -> None:
    graph = Mock()
    graph.query.return_value = [
        {
            "text": "evidence",
            "chunk_id": "c",
            "chunk_element_id": "ce",
            "document": "doc",
            "score": 0.5,
            "entities": [],
        }
    ]
    result = system._text_similarity_search(
        graph, "a b", kg_name="unit", question_id="q", document_names=["doc"]
    )
    assert result["chunks"][0]["text"] == "evidence"
    assert result["search_method"] == "text_similarity_fallback"
    assert graph.query.call_args.args[1]["search_terms"] == ["a b"]
    graph.query.side_effect = RuntimeError("offline")
    result = system._text_similarity_search(graph, "query")
    assert result["chunks"] == []
    assert "offline" in result["error"]


@pytest.mark.parametrize(
    "applied,score,method",
    [
        (True, 0.8, "late_interaction_similarity"),
        (True, 0, "late_interaction_unavailable"),
        (False, 0.8, "late_interaction_unavailable"),
    ],
)
def test_late_interaction_score_signal(
    system: enhanced.EnhancedRAGSystem,
    monkeypatch: pytest.MonkeyPatch,
    applied: bool,
    score: float,
    method: str,
) -> None:
    row = {
        "text": "passage",
        "chunk_id": "c",
        "chunk_element_id": "ce",
        "document": "doc",
        "entities": [],
    }
    graph = Mock()
    graph.query.return_value = [row]

    def rescore(
        query: str, rows: list[dict], **kwargs: object
    ) -> tuple[list[dict], dict]:
        return [{**r, "score": score} for r in rows], {"applied": applied}

    monkeypatch.setattr(vector, "late_interaction_rescore_chunks_for_query", rescore)
    result = system._late_interaction_search(graph, "q", kg_name="unit")
    assert result["search_method"] == method
    if applied and score:
        assert result["total_score"] == score
        assert result["documents"] == ["doc"]
    else:
        assert result["chunks"] == []


def test_ppr_conserves_mass_and_prefers_weighted_neighbors(
    system: enhanced.EnhancedRAGSystem,
) -> None:
    assert system._ppr_entity_scores([], [], []) == {}
    result = system._ppr_entity_scores(
        ["a"],
        ["a", "b", "c", "d"],
        [("a", "b", 1.0), ("a", "c", 0.4), ("a", "a", 1.0), ("missing", "b", 1.0)],
    )
    assert sum(result.values()) == pytest.approx(1)
    assert result["b"] > result["c"] > result["d"]
    assert system._ppr_entity_scores(["a"], ["a"], []) == {"a": pytest.approx(1)}


def test_subgraph_edges_scope_and_error(system: enhanced.EnhancedRAGSystem) -> None:
    graph = Mock()
    graph.query.return_value = [
        {"src": "a", "tgt": "b", "conf": 0.8},
        {"src": None, "tgt": "b", "conf": 1},
    ]
    assert system._fetch_subgraph_edges(
        graph, ["a", "b"], kg_name="unit", question_id="q", document_names=["doc"]
    ) == [("a", "b", 0.8)]
    assert graph.query.call_args.args[1]["question_id"] == "q"
    assert "r.questionIds" in graph.query.call_args.args[0]
    graph.query.side_effect = RuntimeError("offline")
    assert system._fetch_subgraph_edges(graph, ["a"]) == []
    assert system._fetch_subgraph_edges(graph, []) == []


def test_vanilla_real_constructor_and_invalid_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Mock()
    monkeypatch.setattr(vanilla, "load_embedding_model", Mock(return_value=(model, 3)))
    system = vanilla.VanillaRAGSystem(
        neo4j_uri="bolt://unit.invalid",
        neo4j_user="unit",
        neo4j_password="dummy",
        embedding_model="local",
    )
    assert system.embedding_model is model
    assert system.embedding_dimension == 3
    with pytest.raises(ValueError, match="connection parameters"):
        vanilla.VanillaRAGSystem(neo4j_uri="", neo4j_user="", neo4j_password="")
