"""Graph adapter parameterization and symbolic retrieval diversity."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Never, TypedDict
from unittest.mock import Mock

import pytest
from neo4j.exceptions import TransientError

from ontographrag.kg.loaders import graph_db_access
from ontographrag.kg.loaders.graph_db_access import (
    NodeRelationshipCounts,
    RawGraphRow,
    graphDBdataAccess,
)
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

if TYPE_CHECKING:
    from ontographrag.kg.loaders.graph_db_access import ConnectionInfo
    from ontographrag.rag.graph_state import (
        EmptyGraphStateDiversity,
        GraphStateDiversity,
        GraphStateSummary,
    )


class SamplingOptions(TypedDict):
    max_items: int
    retrieval_temperature: float
    shortlist_factor: int
    seed_parts: list[str]
    sample_id: int


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
    items: list[dict[str, int | float]] = [
        {"id": i, "score": float("nan") if i == 1 else 1.0} for i in range(20)
    ]
    opts: SamplingOptions = {
        "max_items": 3,
        "retrieval_temperature": 2,
        "shortlist_factor": 2,
        "seed_parts": ["kg", "question"],
        "sample_id": 1,
    }
    selected: list[dict[str, int | float]] = select_ranked_subset(items, **opts)
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


def _jaccard_cases() -> list[tuple[list[list[str]], float]]:
    return [
        ([], 0),
        ([[]], 1),
        ([[], []], 1),
        ([["a"], ["b"]], 0),
        ([["a", "b"], ["b", "c"]], 1 / 3),
    ]


@pytest.mark.parametrize(
    "sets,expected",
    _jaccard_cases(),
)
def test_jaccard_boundaries(sets: list[list[str]], expected: float) -> None:
    assert mean_pairwise_jaccard(sets) == pytest.approx(expected)


def test_symbolic_state_normalizes_deduplicates_and_measures_diversity() -> None:
    state: GraphStateSummary = summarize_context_graph_state(
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
    result: EmptyGraphStateDiversity | GraphStateDiversity = graph_state_diversity(
        [state, state]
    )
    assert result["chunk_entropy"] == 0
    assert result["subgraph_jaccard"] == 1
    assert result["dominant_seed_entity_fraction"] == 1
    assert dominant_anchor_id([state]) == "a"
    assert dominant_anchor_id([]) == ""
    assert graph_state_diversity([])["sample_count"] == 0
    malformed_states: list[None] = [None]
    assert graph_state_diversity(malformed_states)["empty_state_count"] == 1


def test_source_create_update_parameters_and_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph: Mock = Mock(_database="unit")
    access: graphDBdataAccess = graphDBdataAccess(graph)
    source: sourceNode = sourceNode()
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
    props: dict[str, str | float | int | bool] = graph.query.call_args.args[1]["props"]
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
    graph: Mock = Mock(_database="unit")
    graph.query.return_value = []
    access: graphDBdataAccess = graphDBdataAccess(graph)
    assert getattr(access, method)(*args) in (None, [])
    assert graph.query.call_args.args[1] == params
    assert "a'quoted" not in graph.query.call_args.args[0]


def test_graph_metadata_coercion_and_empty_delete() -> None:
    now: datetime = datetime(2026, 1, 1, tzinfo=UTC)
    graph: Mock = Mock(_database="unit")
    graph.query.return_value = [{"kg": {"created": now, "nested": ({1: now},)}}]
    access: graphDBdataAccess = graphDBdataAccess(graph)
    result: RawGraphRow | None = access.create_kg("kg", "description", "csv")
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
    graph: Mock = Mock(_database="unit")
    access: graphDBdataAccess = graphDBdataAccess(graph)
    sleep: Mock = Mock()
    monkeypatch.setattr(time, "sleep", sleep)
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
    graph: Mock = Mock(_database="unit")
    graph.query.return_value = rows
    access: graphDBdataAccess = graphDBdataAccess(graph)
    assert access.check_gds_version() is expected
    graph.query.side_effect = RuntimeError("offline")
    assert not access.check_gds_version()


@pytest.mark.parametrize("state", ["indexed", "empty", "unembedded", "embedded"])
def test_connection_reports_index_and_chunk_readiness(
    monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    monkeypatch.setattr(
        graph_db_access, "load_embedding_model", Mock(return_value=(Mock(), 3))
    )
    index: list[dict[str, int]] = (
        [{"vector_dimensions": 3}] if state == "indexed" else []
    )
    chunks: list[dict[str, int]] = (
        []
        if state == "empty"
        else [{"hasEmbedding": 0 if state == "unembedded" else 2, "chunks": 2}]
    )
    graph: Mock = Mock(
        _database="unit",
        query=Mock(
            side_effect=[
                index,
                chunks,
                [{"totalGdsProcedures": 1}],
                [{"edition": "community"}],
            ]
        ),
    )
    result: ConnectionInfo | None = graphDBdataAccess(
        graph
    ).connection_check_and_get_vector_dimensions("unit")
    assert result is not None
    assert result["message"] == "Connection Successful"
    assert result["gds_status"] and result["write_access"]
    if state == "indexed":
        assert result["db_vector_dimension"] == 3
    elif state != "embedded":
        assert result["chunks_exists"] is (state == "unembedded")
    assert all(
        c.kwargs["session_params"] == {"database": "unit"}
        for c in graph.query.call_args_list
    )


def test_knn_and_index_rebuild_pass_configured_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KNN_MIN_SCORE", "0.82")
    monkeypatch.setattr(
        graph_db_access, "load_embedding_model", Mock(return_value=(Mock(), 3))
    )
    graph: Mock = Mock(_database="unit", query=Mock(return_value=[{"name": "vector"}]))
    db: graphDBdataAccess = graphDBdataAccess(graph)
    db.update_KNN_graph()
    assert graph.query.call_args.args[1] == {"score": 0.82}
    db.drop_create_vector_index("true")
    assert graph.query.call_args.args[1] == {"dimensions": 3}
    assert "drop index vector" in graph.query.call_args_list[-2].args[0]
    graph.query.reset_mock(return_value=True)
    graph.query.return_value = []
    db.update_KNN_graph()
    assert graph.query.call_count == 1


@pytest.mark.parametrize("delete_entities", ["true", "false"])
def test_document_deletion_preserves_shared_entity_filter(
    delete_entities: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph: Mock = Mock(_database="unit", query=Mock(return_value=[]))
    db: graphDBdataAccess = graphDBdataAccess(graph)
    monkeypatch.setenv("GCS_FILE_CACHE", "False")
    for name in ["a.pdf", "b.pdf"]:
        (tmp_path / name).write_text("test")
    count: int = db.delete_file_from_graph(
        '["a.pdf", "b.pdf"]',
        '["local", "local"]',
        delete_entities,
        str(tmp_path),
        "bolt://unit.invalid",
    )
    assert not list(tmp_path.glob("*.pdf"))
    assert count == 2
    statement, params = graph.query.call_args_list[0].args[:2]
    assert params["filename_list"] == ["a.pdf", "b.pdf"]
    assert params["source_types_list"] == ["local", "local"]
    if delete_entities == "true":
        assert "WHERE NOT d2 IN documents" in statement
        assert graph.query.call_count == 2
    else:
        assert graph.query.call_count == 1
        assert "DETACH DELETE e" not in statement


def test_orphan_and_duplicate_maintenance_records_explicit_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DUPLICATE_SCORE_VALUE", "0.9")
    monkeypatch.setenv("DUPLICATE_TEXT_DISTANCE", "3")
    graph: Mock = Mock(
        _database="unit", query=Mock(side_effect=[[{"id": "a"}], [{"total": 1}]])
    )
    db: graphDBdataAccess = graphDBdataAccess(graph)
    assert db.list_unconnected_nodes() == ([{"id": "a"}], {"total": 1})
    graph.query.side_effect = [[{"id": "a"}], [{"total": 1}]]
    assert db.get_duplicate_nodes_list()[1] == {"total": 1}
    assert graph.query.call_args.args[1] == {
        "duplicate_score_value": 0.9,
        "duplicate_text_distance": 3,
    }
    graph.query.side_effect = None
    graph.query.return_value = [{"totalMerged": 2}]
    assert db.merge_duplicate_nodes(
        '[{"firstElementId":"a","similarElementIds":["b"]}]'
    ) == [{"totalMerged": 2}]
    assert graph.query.call_args.args[1]["rows"][0]["similarElementIds"] == ["b"]
    db.delete_unconnected_nodes('[" a "," b "]')
    assert graph.query.call_args.args[1] == {"elementIds": ["a", "b"]}
    with pytest.raises(ValueError):
        db.delete_unconnected_nodes("not json")


@pytest.mark.parametrize("community", [False, True])
def test_count_update_sums_real_record_components(community: bool) -> None:
    graph: Mock = Mock(
        _database="unit",
        query=Mock(
            side_effect=[
                [{"label": "__Community__"}] if community else [],
                [
                    {
                        "filename": "doc",
                        "chunkNodeCount": "2",
                        "chunkRelCount": "1",
                        "entityNodeCount": 3,
                        "entityEntityRelCount": 4,
                        "communityNodeCount": 5,
                        "communityRelCount": 6,
                    }
                ],
                [],
            ]
        ),
    )
    result: dict[str | None, NodeRelationshipCounts] | list[Never] = graphDBdataAccess(
        graph
    ).update_node_relationship_count(None if community else "doc")
    assert isinstance(result, dict)
    assert result["doc"]["nodeCount"] == (10 if community else 5)
    assert result["doc"]["relationshipCount"] == (11 if community else 5)
    assert graph.query.call_args.args[1]["filename"] == "doc"
