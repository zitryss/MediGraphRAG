"""Graph adapter parameterization and symbolic retrieval diversity."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import Mock

import pytest
from neo4j.exceptions import TransientError

from ontographrag.kg.loaders import graph_db_access
from ontographrag.kg.loaders.graph_db_access import graphDBdataAccess
from ontographrag.kg.utils.source_node import sourceNode
from ontographrag.rag.graph_state import (
    dominant_anchor_id,
    graph_state_diversity,
    mean_pairwise_jaccard,
    normalized_entropy,
    shannon_entropy,
    summarize_context_graph_state,
)
from ontographrag.rag.retrieval_sampling import (
    compute_candidate_limit,
    select_ranked_subset,
    stable_sample_seed,
)


@pytest.mark.parametrize(
    "max_chunks,temp,factor,cap,expected",
    [
        (0, 0, 5, 200, 1),
        (-2, 1, 0, 200, 1),
        (10, 1, 4, 25, 25),
        (30, 1, 4, 25, 30),
        (3, -1, 4, 200, 3),
    ],
)
def test_candidate_limits(
    max_chunks: int, temp: float, factor: int, cap: int, expected: int
) -> None:
    assert compute_candidate_limit(max_chunks, temp, factor, hard_cap=cap) == expected


def test_ranked_sampling_is_stable_bounded_and_order_preserving() -> None:
    items = [{"id": i, "score": float("nan") if i == 1 else 1.0} for i in range(20)]
    opts: dict[str, Any] = {
        "max_items": 3,
        "retrieval_temperature": 2,
        "shortlist_factor": 2,
        "seed_parts": ["kg", "question"],
        "sample_id": 1,
    }
    selected = select_ranked_subset(items, **opts)
    assert selected == select_ranked_subset(items, **opts)
    assert len(selected) == 3
    assert [i["id"] for i in selected] == sorted(i["id"] for i in selected)
    assert all(i["id"] < 6 for i in selected)
    assert select_ranked_subset([], **opts) == []
    assert (
        select_ranked_subset(
            items, max_items=2, retrieval_temperature=0, shortlist_factor=2
        )
        == items[:2]
    )
    assert (
        select_ranked_subset(
            items, max_items=2, retrieval_temperature=1, shortlist_factor=1
        )
        == items[:2]
    )
    assert select_ranked_subset(
        [1, 2, 3],
        max_items=1,
        retrieval_temperature=1,
        shortlist_factor=3,
        score_getter=float,
    )[0] in [1, 2, 3]
    assert stable_sample_seed("a", None) == stable_sample_seed("a", "")
    assert stable_sample_seed("a", "q1") != stable_sample_seed("a", "q2")


@pytest.mark.parametrize(
    "labels,entropy,normalized",
    [
        ([], 0, 0),
        (["a"], 0, 0),
        (["a", "a"], 0, 0),
        (["a", "b"], 1, 1),
        (["a", "a", "b", "b"], 1, 0.5),
    ],
)
def test_entropy_known_distributions(
    labels: list[str], entropy: float, normalized: float
) -> None:
    assert shannon_entropy(labels) == pytest.approx(entropy)
    assert normalized_entropy(labels) == pytest.approx(normalized)


@pytest.mark.parametrize(
    "sets,expected",
    [
        ([], 0),
        ([[]], 1),
        ([[], []], 1),
        ([["a"], ["b"]], 0),
        ([["a", "b"], ["b", "c"]], 1 / 3),
    ],
)
def test_jaccard_boundaries(sets: list[list[str]], expected: float) -> None:
    assert mean_pairwise_jaccard(sets) == pytest.approx(expected)


def test_symbolic_state_normalizes_deduplicates_and_measures_diversity() -> None:
    state = summarize_context_graph_state(
        {
            "retrieval_route": " graph ",
            "diagnostics": {
                "seed_entity_ids": [" a ", "a", ""],
                "seed_entities": ["Alpha"],
            },
            "chunks": [
                {
                    "chunk_id": "c",
                    "linked_entity_ids": ["a"],
                    "entities": [{"id": "b"}],
                },
                {"text": "fallback"},
                None,
            ],
            "relationships": [
                {"key": "r"},
                {"element_id": "r2"},
                {"source": "a", "target": "b", "type": "T"},
                None,
            ],
            "traversal_paths": [{"node_ids": ["a", "b"]}, {"path": "fallback"}, None],
        }
    )
    assert state["entity_ids"] == ["a", "b"]
    assert state["seed_entity_ids"] == ["a"]
    assert state["path_ids"] == ["a->b", "fallback"]
    assert state["relationship_count"] == 3
    assert state["chunk_ids"] == ["c", "fallback"]
    result = graph_state_diversity([state, state])
    assert result["chunk_entropy"] == 0
    assert result["subgraph_jaccard"] == 1
    assert result["dominant_seed_entity_fraction"] == 1
    assert dominant_anchor_id([state]) == "a"
    assert dominant_anchor_id([]) == ""
    assert graph_state_diversity([])["sample_count"] == 0
    malformed_states: list[Any] = [None]
    assert graph_state_diversity(malformed_states)["empty_state_count"] == 1


def test_source_create_update_parameters_and_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = Mock(_database="unit")
    access = graphDBdataAccess(graph)
    source = sourceNode()
    source.file_name = "a.pdf"
    access.create_source_node(source, "kg1")
    assert graph.query.call_args.args[1]["kg_name"] == "kg1"
    assert graph.query.call_args.args[1]["st"] == "New"
    source.status = "Complete"
    # Legacy annotation says float, but the adapter consumes total_seconds().
    monkeypatch.setattr(source, "processing_time", timedelta(seconds=1.234))
    source.node_count = 0
    source.is_cancelled = False
    access.update_source_node(source)
    props = graph.query.call_args.args[1]["props"]
    assert props == {
        "fileName": "a.pdf",
        "status": "Complete",
        "processingTime": 1.23,
        "nodeCount": 0,
        "is_cancelled": False,
    }
    assert graph.query.call_args.kwargs["session_params"] == {"database": "unit"}
    graph.query.side_effect = [[{"is_cancelled": True}], []]
    access.update_exception_db("a.pdf", "failed", retry_condition=True)
    assert graph.query.call_args.args[1]["status"] == "Cancelled"
    assert graph.query.call_args.args[1]["retry_condition"] is None


@pytest.mark.parametrize(
    "method,args,params",
    [
        ("get_kg", ("a'quoted",), {"kg_name": "a'quoted"}),
        ("get_kg_entities", ("a'quoted", -3), {"kg_name": "a'quoted", "limit": 1}),
        ("get_kg_stats", ("a'quoted",), {"kg_name": "a'quoted"}),
        ("get_websource_url", ("a'quoted",), {"file_name": "a'quoted"}),
    ],
)
def test_scoped_graph_reads_parameterize_untrusted_names(
    method: str, args: tuple[object, ...], params: dict[str, object]
) -> None:
    graph = Mock(_database="unit")
    graph.query.return_value = []
    access = graphDBdataAccess(graph)
    assert getattr(access, method)(*args) in (None, [])
    assert graph.query.call_args.args[1] == params
    assert "a'quoted" not in graph.query.call_args.args[0]


def test_graph_metadata_coercion_and_empty_delete() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    graph = Mock(_database="unit")
    graph.query.return_value = [{"kg": {"created": now, "nested": ({1: now},)}}]
    access = graphDBdataAccess(graph)
    result = access.create_kg("kg", "description", "csv")
    assert result == {"created": now.isoformat(), "nested": [{"1": now.isoformat()}]}
    assert graph.query.call_args.args[1] == {
        "kg_name": "kg",
        "description": "description",
        "data_source": "csv",
    }
    graph.query.return_value = []
    assert access.delete_kg_by_name("kg") == 0
    assert graph.query.call_args.args[1] == {"kg_name": "kg"}
    assert "KnowledgeGraph" in graph.query.call_args.args[0]


def test_graph_deadlock_retry_and_other_error(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = Mock(_database="unit")
    access = graphDBdataAccess(graph)
    sleep = Mock()
    monkeypatch.setattr(graph_db_access.time, "sleep", sleep)
    graph.query.side_effect = [TransientError("DeadlockDetected"), [{"n": 1}]]
    assert access.execute_query("RETURN $n", {"n": 1}, delay=2) == [{"n": 1}]
    sleep.assert_called_once_with(2)
    graph.query.side_effect = ValueError("invalid query")
    with pytest.raises(ValueError, match="invalid query"):
        access.execute_query("bad")


@pytest.mark.parametrize(
    "rows,expected",
    [
        ([], False),
        ([{"totalGdsProcedures": 0}], False),
        ([{"totalGdsProcedures": 1}], True),
    ],
)
def test_gds_detection(rows: list[dict[str, int]], expected: bool) -> None:
    graph = Mock(_database="unit")
    graph.query.return_value = rows
    access = graphDBdataAccess(graph)
    assert access.check_gds_version() is expected
    graph.query.side_effect = RuntimeError("offline")
    assert not access.check_gds_version()
