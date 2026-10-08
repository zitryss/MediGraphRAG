"""Stateful retrieval adapter tests using real system constructors."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import SimpleNamespace
from typing import TYPE_CHECKING, TypedDict, Unpack, cast
from unittest.mock import Mock

import numpy as np
import pytest
from langchain_core.prompt_values import PromptValue

from ontographrag.rag.systems import _vector_search as vector
from ontographrag.rag.systems import enhanced_rag_system as enhanced
from ontographrag.rag.systems import vanilla_rag_system as vanilla
from ontographrag.rag.systems._records import RetrievedChunk, TraversalProjection

if TYPE_CHECKING:
    from ontographrag.rag.systems._records import (
        CompleteRetrievedChunk,
        CountedGraphRetrievalContext,
        GraphExpansion,
        GraphRetrievalContext,
        ProducedGraphRetrievalContext,
        RetrievedEntity,
        ScoredCompleteRetrievalContext,
        ScoredRetrievedChunk,
    )
    from ontographrag.rag.systems._response_records import GenerationResponse


class EntityExtras(TypedDict, total=False):
    ann_score: float
    weighted_score: float


class EntityFixture(EntityExtras):
    id: str
    name: str
    type: str
    element_id: str
    exact_in_query: int
    token_overlap: int


class RelationshipFixture(TypedDict):
    source: str
    target: str
    source_element_id: str
    target_element_id: str
    relationship_type: str
    relationship_element_id: str
    confidence: float


class AppliedMetadata(TypedDict):
    applied: bool


class CorpusOptions(TypedDict):
    kg_name: str
    document_names: list[str]
    question_id: str
    include_entities: bool


@pytest.fixture
def system(monkeypatch: pytest.MonkeyPatch) -> enhanced.EnhancedRAGSystem:
    model: Mock = Mock(spec=["embed_query"])
    model.embed_query.return_value = [1.0, 0.0, 0.0]
    monkeypatch.setattr(enhanced, "load_embedding_model", Mock(return_value=(model, 3)))
    return enhanced.EnhancedRAGSystem(
        neo4j_uri="bolt://unit.invalid", neo4j_user="unit", neo4j_password="dummy"
    )


def test_real_constructor_cache_and_connection_lifecycle(
    system: enhanced.EnhancedRAGSystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph: Mock = Mock()
    factory: Mock = Mock(return_value=graph)
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
    encoder: Mock = Mock(return_value=np.array([0.5, 0.5]))
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
    graph: Mock = Mock()
    graph.query.side_effect = (
        [[{"name": "retrieval_vector", "state": "ONLINE"}], []] if available else [[]]
    )
    system._graph = graph
    assert system.check_vector_index() is available
    count: int = graph.query.call_count
    assert system.check_vector_index() is available
    assert graph.query.call_count == count
    if available:
        assert system._active_vector_index_name == "retrieval_vector"
        assert graph.query.call_args.args[1] == {"query_vector": [1.0, 0.0, 0.0]}


@pytest.mark.parametrize("entities", [True, False])
def test_late_interaction_corpus_cache_scope_and_fallback(
    system: enhanced.EnhancedRAGSystem, entities: bool
) -> None:
    graph: Mock = Mock()
    graph.query.side_effect = [[], [{"text": "passage", "chunk_id": "c"}]]
    opts: CorpusOptions = {
        "kg_name": "unit",
        "document_names": ["b", "a"],
        "question_id": "q",
        "include_entities": entities,
    }
    rows: list[CompleteRetrievedChunk] = system._late_interaction_corpus_rows(
        graph, **opts
    )
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
    row: RetrievedChunk = {
        "text": "evidence",
        "chunk_id": "c",
        "chunk_element_id": "ce",
        "document": "doc",
        "score": 0.8,
        "kg_name": "unit",
        "question_id": "q",
        "entities": [],
    }
    graph: Mock = Mock()
    graph.query.return_value = [row]
    result: GraphRetrievalContext = system._vector_similarity_search(
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
    graph: Mock = Mock()
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
    result: GraphRetrievalContext = system._text_similarity_search(
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
    row: RetrievedChunk = {
        "text": "passage",
        "chunk_id": "c",
        "chunk_element_id": "ce",
        "document": "doc",
        "entities": [],
    }
    graph: Mock = Mock()
    graph.query.return_value = [row]

    def rescore(
        query: str, rows: list[RetrievedChunk], **kwargs: object
    ) -> tuple[list[RetrievedChunk], AppliedMetadata]:
        # Each copied row retains its input shape; only the declared float score changes.
        scored: list[RetrievedChunk] = [
            cast(RetrievedChunk, {**r, "score": score}) for r in rows
        ]
        return scored, {"applied": applied}

    monkeypatch.setattr(vector, "late_interaction_rescore_chunks_for_query", rescore)
    result: GraphRetrievalContext = system._late_interaction_search(
        graph, "q", kg_name="unit"
    )
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
    result: dict[str, float] = system._ppr_entity_scores(
        ["a"],
        ["a", "b", "c", "d"],
        [("a", "b", 1.0), ("a", "c", 0.4), ("a", "a", 1.0), ("missing", "b", 1.0)],
    )
    assert sum(result.values()) == pytest.approx(1)
    assert result["b"] > result["c"] > result["d"]
    assert system._ppr_entity_scores(["a"], ["a"], []) == {"a": pytest.approx(1)}


def test_subgraph_edges_scope_and_error(system: enhanced.EnhancedRAGSystem) -> None:
    graph: Mock = Mock()
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
    model: Mock = Mock()
    monkeypatch.setattr(vanilla, "load_embedding_model", Mock(return_value=(model, 3)))
    system: vanilla.VanillaRAGSystem = vanilla.VanillaRAGSystem(
        neo4j_uri="bolt://unit.invalid",
        neo4j_user="unit",
        neo4j_password="dummy",
        embedding_model="local",
    )
    assert system.embedding_model is model
    assert system.embedding_dimension == 3
    with pytest.raises(ValueError, match="connection parameters"):
        vanilla.VanillaRAGSystem(neo4j_uri="", neo4j_user="", neo4j_password="")


def _entity(identifier: str, **extra: Unpack[EntityExtras]) -> EntityFixture:
    return {
        "id": identifier,
        "name": identifier,
        "type": "Drug",
        "element_id": f"e-{identifier}",
        "exact_in_query": 1,
        "token_overlap": 1,
        **extra,
    }


def _passage(identifier: str, entity: str) -> ScoredRetrievedChunk:
    entities: list[RetrievedEntity] = [{"id": entity}]
    return {
        "chunk_id": identifier,
        "chunk_element_id": f"ce-{identifier}",
        "text": f"{entity} treats illness",
        "document": "doc",
        "kg_name": "unit",
        "question_id": "q",
        "position": 1,
        "passage_index": int(identifier[-1]) if identifier[-1].isdigit() else 0,
        "linked_entity_ids": [entity],
        "linked_entity_count": 1,
        "score": 0.8,
        "entities": entities,
    }


def _neighbor() -> TraversalProjection:
    return {
        "id": "illness",
        "name": "illness",
        "type": "Disease",
        "element_id": "e-illness",
        "min_hops": 1,
        "seed_connections": 1,
        "sample_paths": [
            {
                "nodes": ["aspirin", "illness"],
                "node_ids": ["aspirin", "illness"],
                "rels": [
                    {
                        "type": "TREATS",
                        "confidence": 0.9,
                        "question_ids": ["q"],
                        "passage_keys": ["p"],
                        "provenance_positions": [1],
                    }
                ],
            }
        ],
    }


def _relation() -> RelationshipFixture:
    return {
        "source": "aspirin",
        "target": "illness",
        "source_element_id": "e-aspirin",
        "target_element_id": "e-illness",
        "relationship_type": "TREATS",
        "relationship_element_id": "r1",
        "confidence": 0.9,
    }


@pytest.mark.parametrize("ppr", [False, True])
def test_entity_first_real_traversal_assembles_scoped_evidence(
    system: enhanced.EnhancedRAGSystem, monkeypatch: pytest.MonkeyPatch, ppr: bool
) -> None:
    system.use_ppr_scoring = ppr
    system.use_rog_path_planning = True
    monkeypatch.setattr(system, "_TRAVERSAL_CHUNK_MIN_SIM", 0.1)
    calls: list[tuple[str, dict[str, object]]] = []

    def query(
        statement: str, params: dict[str, object] | None = None
    ) -> list[Mapping[str, object]]:
        params = params or {}
        calls.append((statement, params))
        if "SHOW INDEXES" in statement:
            return [{"options": {"indexConfig": {"vector.dimensions": 3}}}]
        if "queryNodes" in statement:
            return [_entity("aspirin", ann_score=0.9, weighted_score=0.9)]
        if "AS rel_type" in statement:
            return [{"rel_type": "TREATS"}]
        if "sample_paths" in statement:
            return [_neighbor()]
        if "candidate_chunk_limit" in params:
            return [_passage("c1", "aspirin"), _passage("c2", "illness")]
        if "AS eid, c.embedding" in statement:
            return [{"eid": "ce-c2", "emb": [1, 0, 0]}]
        if "AS relationship_type" in statement:
            return [_relation()]
        if "a.id AS src" in statement:
            return [{"src": "aspirin", "tgt": "illness", "conf": 1.0}]
        return [_entity("aspirin")]

    graph: Mock = Mock(query=Mock(side_effect=query))
    context: ProducedGraphRetrievalContext | None = system._entity_first_search(
        graph, "aspirin", kg_name="unit", question_id="q", document_names=["doc"]
    )
    assert context is not None
    assert context["retrieval_route"] == "entity_first"
    assert {c["chunk_id"] for c in context["chunks"]} == {"c1", "c2"}
    assert context["entities"]["illness"]["source"] == "graph_traversal"
    assert context["relationships"][0]["type"] == "TREATS"
    assert context["traversal_paths"][0]["question_ids"] == ["q"]
    chunk_params: dict[str, object] = next(
        p for _, p in calls if "candidate_chunk_limit" in p
    )
    assert chunk_params["question_id"] == "q"
    assert chunk_params["document_names"] == ["doc"]
    assert context["total_score"] == pytest.approx(
        sum(c["score"] for c in context["chunks"])
    )


@pytest.mark.parametrize("failure", ["empty", "masked", "thin", "database"])
def test_entity_first_declines_unsupported_context(
    system: enhanced.EnhancedRAGSystem, failure: str
) -> None:
    def query(
        statement: str, params: dict[str, object] | None = None
    ) -> list[Mapping[str, object]]:
        if failure == "database":
            raise RuntimeError("database unavailable")
        if "SHOW INDEXES" in statement:
            return [{"options": {"indexConfig": {"vector.dimensions": 99}}}]
        if "query_text" in (params or {}):
            return [] if failure == "empty" else [_entity("aspirin")]
        if "candidate_chunk_limit" in (params or {}):
            return [_passage("c1", "aspirin")]
        return []

    graph: Mock = Mock(query=Mock(side_effect=query))
    result: ProducedGraphRetrievalContext | None = system._entity_first_search(
        graph,
        "aspirin",
        anchor_mask_entity_ids=["aspirin"] if failure == "masked" else [],
    )
    assert result is None


def test_graph_expansion_filters_seeds_and_preserves_path_provenance(
    system: enhanced.EnhancedRAGSystem,
) -> None:
    row: TraversalProjection = _neighbor()
    row["sample_paths"] *= 2
    graph: Mock = Mock(query=Mock(return_value=[row, {"id": None}]))
    result: GraphExpansion = system._expand_entities_via_graph(
        graph,
        ["aspirin", "blocked"],
        kg_name="unit",
        question_id="q",
        document_names=["doc"],
        allowed_rel_types=["TREATS"],
        excluded_entity_ids={"blocked"},
    )
    assert len(result["paths"]) == 1
    assert result["paths"][0]["provenance_positions"] == [1]
    assert result["paths"][0]["node_ids"] == ["aspirin", "illness"]
    params: dict[str, object] = graph.query.call_args.args[1]
    assert params["seed_ids"] == ["aspirin"]
    assert params["rog_allowed_rel_types"] == ["TREATS"]
    graph.query.side_effect = RuntimeError("traversal unavailable")
    assert system._expand_entities_via_graph(graph, ["aspirin"]) == {
        "neighbors": {},
        "paths": [],
    }
    assert system._expand_entities_via_graph(
        graph, ["blocked"], excluded_entity_ids={"blocked"}
    ) == {"neighbors": {}, "paths": []}


def test_relation_planning_uses_cosine_and_skips_unusable_labels(
    system: enhanced.EnhancedRAGSystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    embeddings: dict[str, list[float]] = {
        "query": [1, 0],
        "treats": [1, 0],
        "unrelated": [0, 1],
        "zero": [0, 0],
    }

    def fake_generate_query_embedding(text: str) -> list[float]:
        return embeddings[text]

    monkeypatch.setattr(
        system, "_generate_query_embedding", fake_generate_query_embedding
    )
    graph: Mock = Mock(
        query=Mock(
            return_value=[
                {"rel_type": key} for key in ["TREATS", "UNRELATED", "ZERO", "MISSING"]
            ]
        )
    )
    assert system._plan_relation_paths(
        graph, ["aspirin"], "query", kg_name="unit", question_id="q"
    ) == ["TREATS"]
    assert system._plan_relation_paths(graph, [], "query") is None
    assert system._plan_relation_paths(graph, ["aspirin"], "zero") is None
    assert system._plan_relation_paths(graph, ["aspirin"], "query", min_sim=1.1) is None
    graph.query.side_effect = RuntimeError("offline graph")
    assert system._plan_relation_paths(graph, ["aspirin"], "query") is None


@pytest.mark.parametrize("with_edges", [False, True])
def test_rfge_merges_seed_and_neighbor_passages_with_actual_ppr(
    system: enhanced.EnhancedRAGSystem,
    monkeypatch: pytest.MonkeyPatch,
    with_edges: bool,
) -> None:
    seed: RetrievedChunk = _passage("c1", "aspirin")
    extra_entities: list[RetrievedEntity] = [
        {"id": "xx"},
        {"id": "masked"},
        {"id": "named", "name": "Masked Name"},
        {},
    ]
    seed["entities"] += extra_entities
    monkeypatch.setattr(
        system, "_semantic_similarity_search", Mock(return_value={"chunks": [seed]})
    )

    def query(
        statement: str, params: dict[str, object] | None = None
    ) -> list[Mapping[str, object]]:
        if "sample_paths" in statement:
            return [_neighbor()]
        if "a.id AS src" in statement:
            return (
                [{"src": "aspirin", "tgt": "illness", "conf": 1.0}]
                if with_edges
                else []
            )
        if "AS seed_entity_id" in statement:
            return [
                {**_passage("c2", "illness"), "seed_entity_id": "illness"},
                {**seed, "seed_entity_id": "illness"},
            ]
        if "AS relationship_type" in statement:
            return [_relation()]
        return []

    graph: Mock = Mock(query=Mock(side_effect=query))
    context: ProducedGraphRetrievalContext | None = (
        system._retriever_first_graph_expansion(
            graph,
            "aspirin",
            kg_name="unit",
            question_id="q",
            document_names=["doc"],
            anchor_mask_entity_ids=["masked"],
            anchor_mask_entity_names=["Masked Name"],
        )
    )
    assert context is not None
    assert context["seed_entity_count"] == 1
    assert context["diagnostics"]["rfge_fired"]
    assert len(context["chunks"]) == (2 if with_edges else 1)
    assert context["grounding_quality"] == (1 if with_edges else 0)
    assert seed["score"] == 0.8  # result scoring never mutates the retrieval input
    assert context["relationships"][0]["key"] == "r1"


def test_rfge_declines_empty_unanchored_or_failed_retrieval(
    system: enhanced.EnhancedRAGSystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch: Mock = Mock(
        side_effect=[
            None,
            {"chunks": [{"chunk_id": "c", "entities": []}]},
            RuntimeError("vector unavailable"),
        ]
    )
    monkeypatch.setattr(system, "_semantic_similarity_search", fetch)
    for _ in range(3):
        assert system._retriever_first_graph_expansion(Mock(), "query") is None


def _context(route: str = "entity_first") -> CountedGraphRetrievalContext:
    return {
        "query": "aspirin",
        "chunks": [_passage("c1", "aspirin")],
        "entities": {
            "aspirin": {
                "id": "aspirin",
                "description": "aspirin",
                "type": "Drug",
                "mentioned_in_chunks": ["c1"],
            }
        },
        "relationships": [],
        "graph_neighbors": {},
        "traversal_paths": [],
        "documents": ["doc"],
        "total_score": 0.8,
        "entity_count": 1,
        "relationship_count": 0,
        "search_method": route,
        "retrieval_route": route,
        "grounding_quality": 1.0,
    }


@pytest.mark.parametrize(
    "route", ["entity", "rfge", "vector", "text", "strict", "empty", "error"]
)
def test_context_orchestrator_selects_fallback_without_crossing_scope(
    system: enhanced.EnhancedRAGSystem, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    graph: Mock = Mock(
        query=Mock(return_value=[{"chunk_count": 0 if route == "empty" else 2}])
    )
    system._graph = graph
    if route == "error":
        graph.query.side_effect = RuntimeError("storage unavailable")
    system.allow_vector_augmentation = False
    system.use_rfge = True
    system.retrieval_mode = "entity_first" if route == "strict" else "hybrid_auto"
    system.allow_vector_fallback = route != "strict"
    entity: Mock = Mock(return_value=_context() if route == "entity" else None)
    rfge: Mock = Mock(return_value=_context("rfge") if route == "rfge" else None)
    vector_fetch: Mock = Mock(return_value=_context("vector"))
    text_fetch: Mock = Mock(return_value=_context("text"))
    for name, value in [
        ("_entity_first_search", entity),
        ("_retriever_first_graph_expansion", rfge),
        ("_semantic_similarity_search", vector_fetch),
        ("_text_similarity_search", text_fetch),
        ("check_vector_index", Mock(return_value=route != "text")),
        ("_first_stage_late_interaction_enabled", Mock(return_value=False)),
    ]:
        monkeypatch.setattr(system, name, value)
    result: ProducedGraphRetrievalContext = system.get_rag_context(
        "aspirin", kg_name="unit", question_id="q", document_names=["doc"], max_chunks=2
    )
    if route in {"empty", "error"}:
        assert not result["chunks"] and result.get("error")
        entity.assert_not_called()
    elif route == "strict":
        assert result["retrieval_route"] == "entity_first_empty"
        vector_fetch.assert_not_called()
        rfge.assert_not_called()
    else:
        assert result["chunks"][0]["chunk_id"] == "c1"
        selected: Mock = {
            "entity": entity,
            "rfge": rfge,
            "vector": vector_fetch,
            "text": text_fetch,
        }[route]
        assert selected.call_args.kwargs["question_id"] == "q"
        if route in {"entity", "rfge"}:
            vector_fetch.assert_not_called()


@pytest.mark.parametrize(
    "mode", ["normal", "insufficient_retry", "abstain", "llm_error", "empty"]
)
def test_generate_response_runs_actual_prompt_and_guardrail_orchestration(
    system: enhanced.EnhancedRAGSystem, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    from langchain_core.runnables import RunnableLambda

    monkeypatch.setenv("ONTOGRAPHRAG_RERANKER", "0")
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION", "0")
    system._graph = Mock(query=Mock(return_value=[]))
    ctx: CountedGraphRetrievalContext = _context()
    if mode == "empty":
        ctx["chunks"] = []
    fetch: Mock = Mock(return_value=ctx)
    monkeypatch.setattr(system, "get_rag_context", fetch)
    monkeypatch.setattr(system, "_should_run_query_fusion", Mock(return_value=False))
    retry: Mock = Mock(return_value=_context("vector"))
    monkeypatch.setattr(system, "_vector_similarity_search", retry)
    prompts: list[str] = []
    responses: Iterator[str] = iter(
        ["Insufficient information", "Aspirin treats illness."]
        if mode == "insufficient_retry"
        else ["Aspirin treats illness."]
    )

    def answer(prompt: PromptValue) -> str:
        prompts.append(prompt.to_string())
        if mode == "llm_error":
            raise RuntimeError("provider unavailable")
        return next(responses)

    if mode == "abstain":
        monkeypatch.setattr(
            enhanced,
            "evaluate_runtime_answer_guardrail",
            Mock(return_value={"decision": "abstain"}),
        )
    result: GenerationResponse = system.generate_response(
        "Does aspirin treat illness?",
        RunnableLambda(answer),
        kg_name="unit",
        question_id="q",
        max_hops=1,
        runtime_guardrail=mode == "abstain",
    )
    if mode == "llm_error":
        assert result["error"] == "provider unavailable"
    elif mode == "empty":
        assert not prompts and result["confidence"] == 0
    elif mode == "abstain":
        assert result["guardrail"]["final_decision"] == "abstain"
        assert result["confidence"] == 0
    else:
        assert result["response"] == "Aspirin treats illness."
        assert 0 < result["confidence"] <= 1
        assert result["sources"] == ["doc"]
        assert "aspirin treats illness" in prompts[0].lower()
        assert result["context"]["graph_state"]
        if mode == "insufficient_retry":
            assert result["guardrail"]["retried_insufficient_info"]
            retry.assert_called_once()
    assert fetch.call_args.kwargs["question_id"] == "q"


@pytest.mark.parametrize(
    "index", ["vector", "retrieval_vector", "missing", "empty", "error"]
)
def test_vanilla_full_context_contract(
    monkeypatch: pytest.MonkeyPatch, index: str
) -> None:
    monkeypatch.setattr(
        vanilla,
        "load_embedding_model",
        Mock(return_value=(Mock(embed_query=Mock(return_value=[1, 0, 0])), 3)),
    )
    retriever: vanilla.VanillaRAGSystem = vanilla.VanillaRAGSystem()
    monkeypatch.setattr(
        retriever, "_first_stage_late_interaction_enabled", Mock(return_value=False)
    )
    monkeypatch.setattr(
        retriever,
        "_resolve_vector_index_name",
        Mock(return_value=None if index == "missing" else index),
    )

    def query(
        statement: str, params: dict[str, object] | None = None
    ) -> list[Mapping[str, object]]:
        if index == "error":
            raise RuntimeError("database gone")
        if "chunk_count" in statement:
            return [{"chunk_count": 0 if index == "empty" else 2}]
        if "queryNodes" in statement:
            return [_passage("c1", "aspirin"), _passage("c2", "illness")]
        return []

    graph: Mock = Mock(query=Mock(side_effect=query))
    monkeypatch.setattr(vanilla, "Neo4jGraph", Mock(return_value=graph))
    result: ScoredCompleteRetrievalContext = retriever.get_vanilla_rag_context(
        "aspirin", document_names=["doc"], kg_name="unit", question_id="q", max_chunks=2
    )
    if index in {"missing", "empty", "error"}:
        assert result["error"] and not result["chunks"]
    else:
        assert len(result["chunks"]) == 2
        assert result["total_score"] == pytest.approx(1.6)
        statement, params = next(
            (c.args[0], c.args[1])
            for c in graph.query.call_args_list
            if "queryNodes" in c.args[0]
        )
        assert params["question_id"] == "q" and params["document_names"] == ["doc"]
        assert f"'{index}'" in statement


@pytest.mark.parametrize("outcome", ["success", "empty", "error", "fusion"])
def test_vanilla_answer_chain_and_query_fusion(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    from langchain_core.runnables import RunnableLambda

    monkeypatch.setattr(vanilla, "load_embedding_model", Mock(return_value=(Mock(), 3)))
    monkeypatch.setenv("ONTOGRAPHRAG_RERANKER", "0")
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION", "0")
    retriever: vanilla.VanillaRAGSystem = vanilla.VanillaRAGSystem()
    context: CountedGraphRetrievalContext = _context("vector")
    if outcome == "empty":
        context["chunks"] = []
    fetch: Mock = Mock(return_value=context)
    if outcome == "fusion":
        alternate: CountedGraphRetrievalContext = _context("vector")
        alternate["chunks"] = [_passage("c2", "illness")]
        fetch.side_effect = [context, RuntimeError("variant unavailable"), alternate]
    monkeypatch.setattr(retriever, "get_vanilla_rag_context", fetch)
    monkeypatch.setattr(
        retriever, "_should_run_query_fusion", Mock(return_value=outcome == "fusion")
    )
    monkeypatch.setattr(
        retriever,
        "_generate_query_variants",
        Mock(return_value=["variant one", "variant two"]),
    )
    prompts: list[str] = []

    def answer(prompt: PromptValue) -> str:
        prompts.append(prompt.to_string())
        if outcome == "error":
            raise RuntimeError("generation failed")
        return "Aspirin treats illness."

    result: GenerationResponse = retriever.generate_response(
        "aspirin",
        RunnableLambda(answer),
        kg_name="unit",
        question_id="q",
        runtime_guardrail=False,
    )
    if outcome == "empty":
        assert not prompts and result["confidence"] == 0
    elif outcome == "error":
        assert result["error"] == "generation failed"
    else:
        assert result["response"] == "Aspirin treats illness."
        assert "aspirin" in prompts[0].lower()
        if outcome == "fusion":
            assert {c["chunk_id"] for c in result["context"]["chunks"]} == {"c1", "c2"}
            assert fetch.call_count == 3


def test_entity_browser_and_statistics_handle_results_and_errors(
    system: enhanced.EnhancedRAGSystem,
) -> None:
    graph: Mock = Mock()
    system._graph = graph
    row: dict[str, str | list[str]] = {
        "id": "aspirin",
        "type": "Drug",
        "description": "medicine",
        "element_id": "e1",
        "relationships": [],
        "mentions": ["doc"],
    }
    graph.query.return_value = [row]
    assert system.get_entity_details("aspirin") == row
    assert graph.query.call_args.args[1] == {"entity_id": "aspirin"}
    graph.query.return_value = [
        {
            "documents": 1,
            "chunks": 2,
            "entities": 3,
            "relationships": 4,
            "document_names": ["doc"],
        }
    ]
    assert system.get_knowledge_graph_stats()["chunks"] == 2
    graph.query.return_value = [
        {
            "id": "aspirin",
            "element_id": "e1",
            "type": "Drug",
            "name": "Aspirin",
            "chunk_mentions": 3,
            "documents": ["doc"],
        }
    ]
    assert system.search_entities("asp", top_k=2)[0]["score"] == pytest.approx(0.3)
    assert graph.query.call_args.args[1] == {"query": "asp", "top_k": 2}
    graph.query.return_value = []
    assert system.get_entity_details("missing")["error"]
    assert system.get_knowledge_graph_stats()["documents"] == 0
    graph.query.side_effect = RuntimeError("database unavailable")
    assert system.get_entity_details("aspirin")["error"] == "database unavailable"
    assert system.get_knowledge_graph_stats()["error"] == "database unavailable"
    assert system.search_entities("asp") == []


@pytest.mark.parametrize("augmentation", ["success", "failure"])
def test_entity_primary_hybrid_keeps_graph_evidence_when_vector_boundary_fails(
    system: enhanced.EnhancedRAGSystem,
    monkeypatch: pytest.MonkeyPatch,
    augmentation: str,
) -> None:
    system._graph = Mock(query=Mock(return_value=[{"chunk_count": 3}]))
    system.allow_vector_augmentation = True
    monkeypatch.setattr(system, "_entity_first_search", Mock(return_value=_context()))
    monkeypatch.setattr(system, "check_vector_index", Mock(return_value=True))
    monkeypatch.setattr(system, "_graph_context_is_meaningful", Mock(return_value=True))
    vector_ctx: CountedGraphRetrievalContext = _context("vector")
    vector_ctx["chunks"] = [_passage("c2", "illness")]
    vector_fetch: Mock = Mock(
        return_value=vector_ctx,
        side_effect=RuntimeError("vector unavailable")
        if augmentation == "failure"
        else None,
    )
    monkeypatch.setattr(system, "_semantic_similarity_search", vector_fetch)
    result: ProducedGraphRetrievalContext = system.get_rag_context(
        "aspirin", kg_name="unit", question_id="q", max_chunks=3
    )
    assert {c["chunk_id"] for c in result["chunks"]} == (
        {"c1", "c2"} if augmentation == "success" else {"c1"}
    )
    assert "aspirin" in result["entities"]
    assert vector_fetch.call_args.kwargs["question_id"] == "q"


def test_enhanced_query_fusion_survives_failed_variant(
    system: enhanced.EnhancedRAGSystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain_core.runnables import RunnableLambda

    monkeypatch.setenv("ONTOGRAPHRAG_RERANKER", "0")
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION", "0")
    system._graph = Mock(query=Mock(return_value=[]))
    variant: CountedGraphRetrievalContext = _context("vector")
    variant["chunks"] = [_passage("c2", "illness")]
    fetch: Mock = Mock(side_effect=[_context(), RuntimeError("bad variant"), variant])
    monkeypatch.setattr(system, "get_rag_context", fetch)
    monkeypatch.setattr(system, "_should_run_query_fusion", Mock(return_value=True))
    monkeypatch.setattr(
        system,
        "_generate_query_variants",
        Mock(return_value=["bad variant", "useful variant"]),
    )

    def fake_response(prompt: object) -> str:
        return "Aspirin treats illness."

    result: GenerationResponse = system.generate_response(
        "aspirin",
        RunnableLambda(fake_response),
        max_hops=1,
        runtime_guardrail=False,
        question_id="q",
        kg_name="unit",
    )
    assert result["response"] == "Aspirin treats illness."
    assert {c["chunk_id"] for c in result["context"]["chunks"]} == {"c1", "c2"}
    assert fetch.call_count == 3
    assert all(c.kwargs["question_id"] == "q" for c in fetch.call_args_list)
