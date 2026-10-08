"""Local-file ingestion and graph adapter contracts (no live database)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from ontographrag.kg.csv_processor import (
    DocumentCSVProcessor,
    MedicalReportCSVProcessor,
)
from ontographrag.kg.loaders import kg_loader
from ontographrag.kg.loaders.kg_loader import KGLoader
from ontographrag.kg.utils import graph_query
from shared import common_fn


@pytest.mark.parametrize("id_column", [None, "id"])
def test_document_batch_preserves_absolute_rows(
    tmp_path: Path, id_column: str | None
) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("id,text,kind\na,alpha,A\nb,beta,B\nc,gamma,C\n")
    processor = (
        DocumentCSVProcessor("text", id_column)
        if id_column
        else DocumentCSVProcessor("text")
    )
    assert processor.validate_csv_format(str(path))["is_valid"]
    result = processor.load_documents_bulk(str(path), start_row=1, max_rows=1)
    assert result["documents"] == [
        {
            "text": "beta",
            "doc_id": "b" if id_column else "row_1",
            "row_index": 1,
            "metadata": {"kind": "B", **({} if id_column else {"id": "b"})},
        }
    ]
    assert result["metadata"]["total_documents"] == 1


def test_document_validation_and_blank_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text('other,text\na,"   "\nb,valid\n')
    processor = DocumentCSVProcessor("missing", "id")
    assert len(processor.validate_csv_format(str(path))["validation_errors"]) == 2
    with pytest.raises(FileNotFoundError):
        processor.validate_csv_format(str(tmp_path / "missing"))
    assert (
        DocumentCSVProcessor("text").load_documents_bulk(str(path))["documents"][0][
            "text"
        ]
        == "valid"
    )
    path.write_text("")
    with pytest.raises(ValueError, match="No columns to parse"):
        processor.validate_csv_format(str(path))


@pytest.mark.xfail(
    strict=True, reason="CSV missing text is currently ingested as literal 'nan'"
)
def test_missing_csv_text_is_not_a_document(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("id,text\na,\nb,valid\n")
    assert (
        len(DocumentCSVProcessor("text").load_documents_bulk(str(path))["documents"])
        == 1
    )


def test_medical_template_roundtrip_and_section_filter(tmp_path: Path) -> None:
    processor = MedicalReportCSVProcessor()
    path = tmp_path / "reports.csv"
    processor.create_csv_template(str(path), num_sample_rows=6)
    result = processor.load_reports_bulk(str(path), start_row=1, max_rows=2)
    assert result["metadata"]["total_reports"] == 2
    assert result["reports"][0]["metadata"]["patient_id"] == "PAT002"
    assert result["reports"][0]["row_index"] == 1
    assert "chief_complaint" in result["reports"][0]["sections"]
    path.write_text(
        "patient_id|chief_complaint|discharge_diagnosis\nP1|short|   Long enough diagnosis  \n"
    )
    report = processor.load_reports_bulk(str(path))["reports"][0]
    assert report["sections"] == {"discharge_diagnosis": "Long enough diagnosis"}
    assert "chief_complaint" in report["field_names"]


@pytest.mark.parametrize("content", ["a,b\n", "a,b\nx,y\n", ""])
def test_medical_invalid_input(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(content)
    processor = MedicalReportCSVProcessor()
    assert not processor.validate_csv_format(str(path))["is_valid"]
    with pytest.raises(ValueError, match="CSV validation failed"):
        MedicalReportCSVProcessor().load_reports_bulk(str(path))
    assert not processor.validate_csv_format(str(tmp_path / "absent"))["is_valid"]


@pytest.mark.parametrize(
    "docker,expected",
    [("true", "bolt://neo4j:7687"), ("false", "bolt://localhost:7687")],
)
def test_loader_constructor(
    monkeypatch: pytest.MonkeyPatch, docker: str, expected: str
) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://neo4j:7687")
    monkeypatch.setenv("DOCKER_ENV", docker)
    monkeypatch.setenv("NEO4J_USERNAME", "unit")
    assert KGLoader().neo4j_uri == expected
    assert KGLoader().neo4j_user == "unit"


def test_loader_ontology_and_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = KGLoader()
    ontology = {
        "categories": [{"name": "Disease"}],
        "mappings": [{"category": "Disease", "terms": ["fever"]}],
    }
    path = tmp_path / "ontology.json"
    path.write_text(json.dumps(ontology))
    assert loader._load_ontology(str(path)) == ontology
    nodes, groups, edges = loader._create_graph_with_ontology(
        "fever fever pain a", ontology
    )
    assert {n["label"] for n in nodes} == {"Fever", "Pain"}
    assert (
        next(n for n in nodes if n["label"] == "Fever")["properties"]["frequency"] == 2
    )
    assert groups[0]["label"] == "Disease"
    assert sum(e["type"] == "MEMBER_OF" for e in edges) == 1
    assert loader._create_graph_with_ontology("a", {}) == ([], [], [])
    with pytest.raises(ValueError, match="Unsupported"):
        loader._load_ontology("bad.txt")
    onto = Mock()
    onto.load.return_value = onto
    onto.classes.return_value = [SimpleNamespace(name="Disease")]
    onto.object_properties.return_value = [SimpleNamespace(name="TREATS")]
    monkeypatch.setattr(kg_loader, "get_ontology", Mock(return_value=onto))
    assert loader._load_ontology("x.owl") == {
        "node_labels": ["Disease"],
        "relationship_types": ["TREATS"],
    }


@pytest.mark.parametrize("filename", ["graph", "nested/graph.json"])
def test_loader_file_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    monkeypatch.chdir(tmp_path)
    data = {"nodes": [{"id": "a"}], "relationships": []}
    result = KGLoader().save_to_file(data, filename)
    assert result["status"] == "success"
    assert json.loads(Path(result["file_path"]).read_text()) == data
    assert (
        KGLoader().save_to_file({"bad": object()}, str(tmp_path / "bad.json"))["status"]
        == "error"
    )


def test_loader_pdf_success_and_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: "fever pain")])
    monkeypatch.setattr(kg_loader, "PdfReader", Mock(return_value=reader))
    result = KGLoader().load_from_pdf(
        str(tmp_path / "doc.pdf"), str(tmp_path / "missing.json")
    )
    assert result["status"] == "success"
    assert result["filename"] == "doc.pdf"
    assert {node["label"] for node in result["nodes"]} == {"Fever", "Pain"}
    monkeypatch.setattr(kg_loader, "PdfReader", Mock(side_effect=OSError("unreadable")))
    result = KGLoader().load_from_pdf("bad.pdf")
    assert result["details"]["error_type"] == "OSError"


@pytest.mark.parametrize("authenticated,agent", [(False, False), (True, True)])
def test_graph_driver_environment_and_connectivity(
    monkeypatch: pytest.MonkeyPatch, authenticated: bool, agent: bool
) -> None:
    for name, value in {
        "NEO4J_URI": "bolt://unit.invalid",
        "NEO4J_USERNAME": "unit" if authenticated else "",
        "NEO4J_USER": "",
        "NEO4J_PASSWORD": "dummy" if authenticated else "",
        "NEO4J_DATABASE": "unit",
        "ENABLE_USER_AGENT": str(agent),
        "NEO4J_USER_AGENT": "tests",
    }.items():
        monkeypatch.setenv(name, value)
    session = Mock()
    driver = Mock()
    driver.session.return_value.__enter__ = Mock(return_value=session)
    driver.session.return_value.__exit__ = Mock(return_value=False)
    factory = Mock(return_value=driver)
    monkeypatch.setattr(graph_query.GraphDatabase, "driver", factory)
    assert graph_query.get_graphDB_driver(None, None, None, "") is driver
    session.run.assert_called_once_with("RETURN 1")
    assert (factory.call_args.kwargs["auth"] is not None) == authenticated
    assert ("user_agent" in factory.call_args.kwargs) == agent
    factory.side_effect = RuntimeError("unavailable")
    assert graph_query.get_graphDB_driver("bolt://unit.invalid", "u", "p") is None
    monkeypatch.setenv("NEO4J_URI", "")
    assert graph_query.get_graphDB_driver(None, None, None) is None


@pytest.mark.parametrize(
    "attrs,expected",
    [
        ({}, "unknown_model"),
        ({"model": 7}, "7"),
        ({"model_name": "kept", "model": "other"}, "kept"),
    ],
)
def test_shared_model_name(attrs: dict[str, object], expected: str) -> None:
    model = SimpleNamespace(**attrs)
    assert common_fn.wrap_llm_with_model_name(model) is model
    assert model.model_name == expected


def test_shared_embedding_delegation(monkeypatch: pytest.MonkeyPatch) -> None:
    from ontographrag.kg.utils import common_functions

    factory = Mock(return_value=("embedding", 3))
    monkeypatch.setattr(common_functions, "load_embedding_model", factory)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    assert common_fn.load_embedding_model() == ("embedding", 3)
    factory.assert_called_with("local")
    common_fn.load_embedding_model("explicit")
    factory.assert_called_with("explicit")


def test_shared_merge_maps_identity_and_relationship_properties() -> None:
    graph = Mock()
    a = SimpleNamespace(id="a", type="Disease", properties={"id": "wrong", "name": "A"})
    b = SimpleNamespace(id="b", type="Drug", properties={})
    rel = SimpleNamespace(
        source=b, target=a, type="may-treat condition", properties={"confidence": 0.8}
    )
    common_fn._add_graph_documents_with_merge(
        graph, [SimpleNamespace(nodes=[a, b], relationships=[rel])]
    )
    calls = graph.query.call_args_list
    assert calls[0].args[1]["properties"] == {"id": "a", "name": "A"}
    assert "MAY_TREAT_CONDITION" in calls[2].args[0]
    assert calls[2].args[1] == {
        "source_id": "b",
        "target_id": "a",
        "properties": {"confidence": 0.8},
    }


class GraphNode(dict):
    """A minimal Neo4j value, not a database emulator."""

    def __init__(self, identity: int, **props: object) -> None:
        super().__init__(props)
        self.id = identity
        self.labels = {"Disease"}
        self.type = "REL"


class QueryResult(list):
    def single(self) -> dict | None:
        return self[0] if self else None


@pytest.mark.parametrize(
    "kg_label,query,limit",
    [(None, None, None), ("unit", None, 2), (None, "MATCH (n) RETURN n AS node", 2)],
)
def test_loader_database_read_serializes_graph_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kg_label: str | None,
    query: str | None,
    limit: int | None,
) -> None:
    a = GraphNode(1, name="A")
    b = GraphNode(2, name="B", custom=Path("value"))
    missing = GraphNode(3)
    rel = GraphNode(4, weight=0.5)
    session = Mock()
    session.run.side_effect = [
        QueryResult([{"node_count": 3}]),
        QueryResult([{"rel_count": 2}]),
        QueryResult([{"node": a}, {"node": b}, {}]),
        QueryResult(
            [{"r": rel, "start": a, "end": b}, {"r": rel, "start": b, "end": missing}]
        ),
    ]
    driver = Mock()
    driver.session.return_value = session
    monkeypatch.setattr(kg_loader.GraphDatabase, "driver", Mock(return_value=driver))
    # The loader hardcodes a package-local storage directory; redirect that boundary.
    monkeypatch.setattr(kg_loader, "__file__", str(tmp_path / "kg_loader.py"))
    options: dict[str, Any] = {
        key: value
        for key, value in {"kg_label": kg_label, "query": query, "limit": limit}.items()
        if value is not None
    }
    result = KGLoader().load_from_neo4j("bolt://unit.invalid", "u", "p", **options)
    assert result["status"] == "success"
    assert result["loaded_nodes"] == 2
    assert result["loaded_relationships"] == 1
    assert result["nodes"][1]["properties"]["custom"] == "value"
    assert result["relationships"][0]["from"] == 1
    assert result["complete_import"] == (
        kg_label is None and query is None and limit is None
    )
    session.close.assert_called_once()
    driver.close.assert_called_once()


def test_loader_database_failure_still_closes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    driver = Mock()
    driver.verify_connectivity.side_effect = RuntimeError("offline")
    monkeypatch.setattr(kg_loader.GraphDatabase, "driver", Mock(return_value=driver))
    monkeypatch.setattr(kg_loader, "__file__", str(tmp_path / "kg_loader.py"))
    result = KGLoader().load_from_neo4j("bolt://unit.invalid", "u", "p")
    assert result["details"]["error_type"] == "RuntimeError"
    driver.close.assert_called_once()


@pytest.mark.parametrize("label", [None, "Disease"])
def test_loader_sample_filters_dangling_relationships(
    monkeypatch: pytest.MonkeyPatch, label: str | None
) -> None:
    a = GraphNode(1, name="A", custom=Path("value"))
    rel = GraphNode(2, custom=Path("value"))
    session = Mock()
    session.run.side_effect = [
        QueryResult([{"n": a}]),
        QueryResult(
            [
                {"r": rel, "start": a, "end": a},
                {"r": rel, "start": a, "end": GraphNode(99)},
            ]
        ),
    ]
    result = KGLoader()._load_neo4j_sample(session, label or "", 2000, 4000)
    assert result["loaded_nodes"] == 1
    assert result["loaded_relationships"] == 1
    assert result["sample_strategy"] == "high_degree_nodes"
    assert "LIMIT 200" in session.run.call_args_list[0].args[0]
    session.run.side_effect = RuntimeError("offline")
    assert (
        KGLoader()._load_neo4j_sample(session, label or "", 2000, 4000)["status"]
        == "error"
    )


@pytest.mark.parametrize("clear", [True, False])
def test_loader_save_merge_keys_relationships_and_explicit_clear(
    monkeypatch: pytest.MonkeyPatch, clear: bool
) -> None:
    from unittest.mock import MagicMock

    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    session.run.return_value.single.return_value = {"node_id": "stored"}
    monkeypatch.setattr(kg_loader.GraphDatabase, "driver", Mock(return_value=driver))
    data = {
        "nodes": [
            {"id": "a", "label": "Drug Name", "properties": {"id": "external"}},
            {"id": "b", "properties": {"name": "B"}},
            {"id": "c", "properties": {}},
        ],
        "relationships": [
            {"from": "a", "to": "b", "type": "TREATS", "id": "r"},
            {"from": "b", "to": "c", "type": "REL"},
            {"from": "missing", "to": "c", "type": "REL"},
        ],
    }
    result = KGLoader().save_to_neo4j(
        "bolt://unit.invalid", "u", "p", data, clear_database=clear
    )
    assert result["status"] == "success"
    calls = session.run.call_args_list
    assert any("DETACH DELETE" in c.args[0] for c in calls) is clear
    merges = [c for c in calls if "MERGE (n:" in c.args[0]]
    assert [c.kwargs["unique_value"] for c in merges[:2]] == ["external", "B"]
    assert "Drug_Name" in merges[0].args[0]
    assert "unique_hash" in merges[2].kwargs["properties"]
    assert sum("MERGE (a)-" in c.args[0] for c in calls) == 2
    session.run.side_effect = RuntimeError("write failed")
    assert (
        KGLoader().save_to_neo4j("bolt://unit.invalid", "u", "p", data)["status"]
        == "error"
    )


def test_legacy_chunking_page_metadata_and_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_core.documents import Document

    from ontographrag.kg.utils import create_chunks

    splitter = Mock()
    splitter.split_documents.side_effect = lambda docs: [
        Document(page_content=d.page_content) for d in docs
    ]
    monkeypatch.setattr(
        create_chunks.RecursiveCharacterTextSplitter,
        "from_tiktoken_encoder",
        Mock(return_value=splitter),
    )
    pages = [Document(page_content=f"page {i}", metadata={"page": i}) for i in range(4)]
    result = create_chunks.CreateChunksofDocument(pages, Mock()).split_file_into_chunks(
        4000, 0
    )
    assert [d.page_content for d in result] == ["page 0", "page 1"]
    assert [d.metadata["page_number"] for d in result] == [1, 2]
    docs = [Document(page_content="plain")]
    result = create_chunks.CreateChunksofDocument(docs, Mock()).split_file_into_chunks(
        100, 0
    )
    assert result[0].page_content == "plain"
