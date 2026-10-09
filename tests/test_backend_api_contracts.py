"""HTTP and state-management behavior with explicit external boundaries."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import sys
import tempfile
from collections.abc import Callable
from importlib.machinery import ModuleSpec
from inspect import Signature
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, cast
from unittest.mock import AsyncMock, MagicMock, Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.testclient import TestClient
from starlette.responses import StreamingResponse
from typing_extensions import TypedDict

from tests.known_defects import (
    AuthMiddlewareDefect,
    CreatorSignatureDefect,
    ReadbackDriverLeak,
    ReadbackIdentityDefect,
)

if TYPE_CHECKING:
    from unittest.mock import _Call

    from ontographrag.api.runtime_helpers import (
        DoctorCheck,
        DoctorReport,
        JSONValue,
        Neo4jHealthReport,
    )
    from ontographrag.rag.systems.enhanced_rag_system import EnhancedRAGSystem


class AppExports(Protocol):
    """Exports of the exact app.py path loaded below, without importing it eagerly."""

    app: FastAPI
    neo4j_ready: bool
    _rag_system: EnhancedRAGSystem | None
    _rag_system_embedding: str | None
    _kg_building: bool
    _REQUEST_TIMEOUT_SECONDS: int
    _cors_origins: list[str]
    _DEBUG_ERRORS: bool
    _APP_API_KEY: str

    def get_rag_system(
        self,
        embedding_model: str | None = None,
    ) -> EnhancedRAGSystem: ...
    def invalidate_rag_system(self, reason: str = "") -> None: ...
    def validate_ontology_schema(
        self,
        ontology_bytes: bytes,
        filename: str,
    ) -> list[str]: ...
    def validate_file_upload(
        self,
        file: UploadFile,
        max_size_bytes: int = 52428800,
        allowed_extensions: set[str] | None = None,
    ) -> None: ...
    def _extract_text_from_bytes(
        self,
        data: bytes,
        filename: str,
    ) -> tuple[str, str]: ...
    def check_neo4j_connection(self) -> None: ...
    def neo4j_health(self) -> Neo4jHealthReport: ...
    async def kg_progress_stream(self, request: Request) -> StreamingResponse: ...
    def _log_progress(self, msg: str) -> None: ...
    def _probe_openai_models(self) -> str: ...
    def _probe_openrouter_models(self) -> str: ...
    def _probe_deepseek_models(self) -> str: ...


class OntologyProperty(TypedDict, total=False):
    name: str
    identifier: bool
    type: str


class OntologyClass(TypedDict, total=False):
    id: str
    properties: list[OntologyProperty]


OntologyRelationship = TypedDict(
    "OntologyRelationship",
    {"id": str, "from": str, "to": str, "cardinality": str},
    total=False,
)


class OntologyFixture(TypedDict, total=False):
    classes: list[OntologyClass | int]
    relationships: list[OntologyRelationship | int]


class BuiltGraph(TypedDict):
    nodes: list[dict[str, str]]
    metadata: dict[str, int | bool]


class LoadedGraph(TypedDict, total=False):
    status: str
    nodes: list[dict[str, str]]
    relationships: list[dict[str, str]]


class BuildResponse(TypedDict):
    kg_name: str
    deduplicated: bool
    method: str
    graph_data: BuiltGraph | LoadedGraph


class ReadbackNode(TypedDict):
    id: int
    properties: dict[str, str]


class ReadbackRelationship(TypedDict):
    start: int
    type: str


class ReadbackGraph(TypedDict):
    nodes: list[ReadbackNode]
    relationships: list[ReadbackRelationship]


class ReadbackResponse(TypedDict):
    stats: dict[str, int]
    graph_data: ReadbackGraph
    kg_settings: dict[str, str]


@pytest.fixture
def app_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppExports:
    monkeypatch.setenv("APP_API_KEY", "")
    monkeypatch.setenv("ONTOGRAPHRAG_DEBUG_ERRORS", "0")
    path: Path = Path(__file__).parents[1] / "ontographrag/api/app.py"
    spec: ModuleSpec | None = importlib.util.spec_from_file_location(
        "backend_contract_app",
        path,
    )
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_READY_WRITE_PROBE_DIR", tmp_path)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    # The executed source path above is fixed; this cast only describes its exports.
    return cast(AppExports, module)


@pytest.fixture
def client(app_module: AppExports) -> TestClient:
    return TestClient(app_module.app, raise_server_exceptions=False)


def test_authentication_public_route_remains_public(
    app_module: AppExports,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "_APP_API_KEY", "unit-secret")
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize(
    "headers",
    [{}, {"X-API-Key": "wrong"}],
    ids=["missing", "wrong"],
)
@pytest.mark.xfail(
    strict=True,
    raises=AuthMiddlewareDefect,
    reason="The precise intended 401 escapes auth middleware and becomes HTTP 500",
)
def test_unauthorized_request_returns_401(
    app_module: AppExports,
    monkeypatch: pytest.MonkeyPatch,
    headers: dict[str, str],
) -> None:
    monkeypatch.setattr(app_module, "_APP_API_KEY", "unit-secret")
    # Propagate the server exception to distinguish this known 500 from an unrelated
    # 500; constructing without a context manager does not run live startup hooks.
    client: TestClient = TestClient(app_module.app, raise_server_exceptions=True)
    try:
        response: httpx.Response = client.get("/kg/list", headers=headers)
    except HTTPException as exc:
        if exc.status_code == 401 and exc.detail == "Invalid or missing API key":
            raise AuthMiddlewareDefect(
                "Intended unauthorized response escaped middleware",
            ) from exc
        raise
    assert response.status_code == 401


def test_cached_rag_model_switch_and_invalidation(
    app_module: AppExports,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_fake: Mock = Mock()
    second_fake: Mock = Mock()
    factory: Mock = Mock(side_effect=[first_fake, second_fake])
    monkeypatch.setattr(app_module, "EnhancedRAGSystem", factory)
    first: EnhancedRAGSystem = app_module.get_rag_system("LOCAL")
    assert app_module.get_rag_system() is first
    assert app_module.get_rag_system("local") is first
    second: EnhancedRAGSystem = app_module.get_rag_system("other")
    assert second is not first
    assert factory.call_count == 2
    second_fake.clear_retrieval_caches.side_effect = RuntimeError("cache failure")
    app_module.invalidate_rag_system("mutation")
    assert app_module._rag_system is None
    assert app_module._rag_system_embedding is None
    second_fake.clear_retrieval_caches.assert_called_once()


INVALID_ONTOLOGIES: list[tuple[OntologyFixture, str]] = [
    ({"classes": [1]}, "expected object"),
    ({"classes": [{}]}, "missing 'id'"),
    ({"classes": [{"id": "A"}]}, "missing identifier"),
    ({"relationships": [{}]}, "missing 'id'"),
    ({"relationships": [1]}, "expected object"),
    (
        {
            "classes": [
                {
                    "id": "A",
                    "properties": [{"name": "x", "identifier": True, "type": "enum"}],
                },
            ],
        },
        "no enum_values",
    ),
]


@pytest.mark.parametrize(
    "payload,fragment",
    INVALID_ONTOLOGIES,
)
def test_ontology_rejects_malformed_entries(
    app_module: AppExports,
    payload: OntologyFixture,
    fragment: str,
) -> None:
    assert any(
        fragment in e
        for e in app_module.validate_ontology_schema(
            json.dumps(payload).encode(),
            "schema.json",
        )
    )


def test_ontology_relationship_validation_and_formats(app_module: AppExports) -> None:
    schema: OntologyFixture = {
        "classes": [
            {
                "id": "A",
                "properties": [
                    {"name": "id", "identifier": True},
                    {"name": "id", "type": "integer"},
                    {},
                ],
            },
            {"id": "A"},
        ],
        "relationships": [
            {"id": "r", "from": "missing", "to": "other", "cardinality": "bad"},
            {"id": "r", "from": "A", "to": "A"},
        ],
    }
    errors: list[str] = app_module.validate_ontology_schema(
        json.dumps(schema).encode(),
        "ontology.json",
    )
    for fragment in [
        "type conflict",
        "Duplicate entity",
        "Duplicate relationship",
        "domain",
        "range",
        "cardinality",
        "missing 'name'",
    ]:
        assert any(fragment in e for e in errors)
    assert app_module.validate_ontology_schema(b"broken", "x.owl") == []
    assert "Invalid JSON" in app_module.validate_ontology_schema(b"\xff", "x.json")[0]
    assert app_module.validate_ontology_schema(b"{}", "x.json") == []


def test_upload_extension_and_text_decoding(app_module: AppExports) -> None:
    file: UploadFile = UploadFile(filename="bad.exe", file=io.BytesIO(b"bad"))
    with pytest.raises(HTTPException) as error:
        app_module.validate_file_upload(file, allowed_extensions={".txt"})
    assert error.value.status_code == 400
    assert app_module._extract_text_from_bytes("café".encode(), "text.txt") == (
        "café",
        "plaintext",
    )
    assert app_module._extract_text_from_bytes(b"caf\xe9", "text.csv") == (
        "café",
        "plaintext",
    )
    with pytest.raises(HTTPException) as error:
        app_module._extract_text_from_bytes(b"not pdf", "text.pdf")
    assert error.value.status_code == 500


def test_readiness_success_and_runtime_failure(
    app_module: AppExports,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "neo4j_ready", True)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr(app_module, "get_rag_system", Mock(return_value=object()))
    assert client.get("/ready").status_code == 200
    monkeypatch.setattr(
        app_module,
        "get_rag_system",
        Mock(side_effect=RuntimeError("runtime unavailable")),
    )
    response: httpx.Response = client.get("/ready")
    assert response.status_code == 503
    assert not response.json()["ready"]
    assert "runtime unavailable" in response.text


def test_startup_and_database_health(
    app_module: AppExports,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver: MagicMock = MagicMock()
    session: MagicMock = driver.session.return_value.__enter__.return_value
    session.run.return_value.single.return_value = {"c": 7}
    monkeypatch.setattr(app_module, "get_graphDB_driver", Mock(return_value=driver))
    app_module.check_neo4j_connection()
    assert app_module.neo4j_ready
    assert client.get("/health/neo4j").json() == {"status": "ok", "nodeCount": 7}
    monkeypatch.setattr(
        app_module,
        "get_graphDB_driver",
        Mock(side_effect=RuntimeError("unavailable")),
    )
    assert client.get("/health/neo4j").status_code == 500


def test_named_kg_crud_and_failure(
    app_module: AppExports,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langchain_neo4j

    from ontographrag.kg.loaders import graph_db_access

    db: Mock = Mock()
    db.create_kg.return_value = {"name": "unit"}
    db.get_kg.return_value = {"name": "unit"}
    db.get_kg_stats.return_value = [{"documentCount": 2}]
    db.get_kg_entities.return_value = [{"id": "e"}]
    db.delete_kg_by_name.return_value = 2
    monkeypatch.setattr(langchain_neo4j, "Neo4jGraph", Mock())
    monkeypatch.setattr(graph_db_access, "graphDBdataAccess", Mock(return_value=db))
    monkeypatch.setattr(app_module, "neo4j_ready", True)
    assert client.post("/kg/create", data={"kg_name": "unit"}).json()["kg"] == {
        "name": "unit",
    }
    assert client.get("/kg/unit").json()["stats"] == {"documentCount": 2}
    assert client.get("/kg/unit/entities?limit=2").json()["count"] == 1
    db.get_kg_entities.assert_called_once_with("unit", 2)
    assert (
        client.delete("/kg/unit?delete_entities=false").json()["deleted_documents"] == 2
    )
    db.delete_kg_by_name.assert_called_once_with("unit", False)
    db.get_kg.return_value = None
    assert client.get("/kg/missing").status_code == 404
    db.get_kg.side_effect = RuntimeError("query failed")
    assert client.get("/kg/unit").status_code == 500
    db.create_kg.side_effect = RuntimeError("write failed")
    assert client.post("/kg/create", data={"kg_name": "unit"}).status_code == 500


def test_csv_validation_cleans_temporary_file(
    app_module: AppExports,
    client: TestClient,
    tmp_path: Path,
) -> None:
    response: httpx.Response = client.post(
        "/validate_csv",
        files={"csv_file": ("bad.csv", b"a|b\nx|y\n")},
    )
    assert response.status_code == 200
    assert not response.json()["is_valid"]
    assert len(response.json()["errors"]) == 3
    assert list(tmp_path.glob("validate_*")) == []


def test_save_graph_filters_properties_and_handles_missing_state(
    app_module: AppExports,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "neo4j_ready", True)
    monkeypatch.setattr(app_module, "current_graph_data", None)
    form: dict[str, str] = {
        "kg_id": "unit",
        "uri": "bolt://unit.invalid",
        "user": "unit",
        "password": "dummy",
    }
    assert client.post("/save_kg_to_neo4j", data=form).status_code == 400
    driver: MagicMock = MagicMock()
    monkeypatch.setattr(app_module, "get_graphDB_driver", Mock(return_value=driver))
    monkeypatch.setattr(
        app_module,
        "current_graph_data",
        {
            "nodes": [
                {
                    "id": "a",
                    "properties": {"embedding": [1], "name": "A", "none": None},
                },
                {"id": "b", "labels": ["Drug"]},
            ],
            "relationships": [{"from": "a", "to": "b", "type": "REL"}, {"from": "a"}],
        },
    )
    result: httpx.Response = client.post("/save_kg_to_neo4j", data=form)
    assert result.status_code == 200
    assert result.json()["nodes_saved"] == 2
    assert result.json()["relationships_saved"] == 1
    calls: list[_Call] = (
        driver.session.return_value.__enter__.return_value.run.call_args_list
    )
    assert calls[1].args[1] == {"name": "A", "id": "a"}


def test_progress_stream_completion_and_disconnect(
    app_module: AppExports,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def consume(disconnected: bool) -> list[str | bytes | memoryview]:
        request: Request = Request({"type": "http"})
        monkeypatch.setattr(
            request,
            "is_disconnected",
            AsyncMock(return_value=disconnected),
        )
        response: StreamingResponse = await app_module.kg_progress_stream(request)
        return [event async for event in response.body_iterator]

    app_module._log_progress("done")
    assert asyncio.run(consume(False)) == [
        'data: {"line": "done"}\n\n',
        'data: {"done": true}\n\n',
    ]
    assert asyncio.run(consume(True)) == []


@pytest.mark.parametrize(
    "provider,key",
    [
        ("openai", "OPENAI_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
    ],
)
def test_provider_probe_transport(
    app_module: AppExports,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    key: str,
) -> None:
    probes: dict[str, Callable[[], str]] = {
        "openai": app_module._probe_openai_models,
        "openrouter": app_module._probe_openrouter_models,
        "deepseek": app_module._probe_deepseek_models,
    }
    probe: Callable[[], str] = probes[provider]
    monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="not set"):
        probe()
    monkeypatch.setenv(key, "unit")
    response: Mock = Mock()
    response.json.return_value = {"data": [{"id": "unit"}]}
    get: Mock = Mock(return_value=response)
    monkeypatch.setattr(httpx, "get", get)
    assert probe() == "reachable (1 models visible)"
    assert get.call_args.kwargs["headers"] == {"Authorization": "Bearer unit"}
    response.raise_for_status.assert_called_once()


@pytest.mark.parametrize(
    "provider,key",
    [
        ("openai", "OPENAI_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
    ],
)
@pytest.mark.parametrize(
    "payload,expected_count,expected_error",
    [
        ({"data": [{"id": "a"}, {"id": "b"}]}, 2, None),
        ({"data": {"a": 1, "b": 2}}, 2, None),
        ({"data": "abc"}, 3, None),
        ({}, 0, None),
        ({"data": 1}, None, TypeError),
        ([], None, AttributeError),
    ],
)
def test_provider_probe_preserves_json_shape_behavior(
    app_module: AppExports,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    key: str,
    payload: JSONValue,
    expected_count: int | None,
    expected_error: type[TypeError | AttributeError] | None,
) -> None:
    probes: dict[str, Callable[[], str]] = {
        "openai": app_module._probe_openai_models,
        "openrouter": app_module._probe_openrouter_models,
        "deepseek": app_module._probe_deepseek_models,
    }
    probe: Callable[[], str] = probes[provider]
    monkeypatch.setenv(key, "unit")
    response: httpx.Response = httpx.Response(
        200,
        json=payload,
        request=httpx.Request("GET", "https://unit.invalid/models"),
    )
    get: Mock = Mock(return_value=response)
    monkeypatch.setattr(httpx, "get", get)
    if expected_error is not None:
        with pytest.raises(expected_error):
            probe()
    else:
        assert probe() == f"reachable ({expected_count} models visible)"
    get.assert_called_once()
    assert get.call_args.kwargs["headers"] == {"Authorization": "Bearer unit"}


@pytest.mark.parametrize("healthy", [True, False])
def test_doctor_local_success_and_dependency_failure(
    app_module: AppExports,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    healthy: bool,
) -> None:
    import ollama

    from ontographrag.kg.utils import common_functions

    ui: Path = tmp_path / "ui"
    ui.mkdir()
    if healthy:
        (ui / "index.html").write_text("<html>unit</html>")
    monkeypatch.setattr(app_module, "_UI_DIST_DIR", ui)
    driver: MagicMock = MagicMock()
    driver.session.return_value.__enter__.return_value.run.return_value.single.return_value = {
        "c": 2,
    }
    monkeypatch.setattr(
        app_module,
        "get_graphDB_driver",
        Mock(
            return_value=driver,
            side_effect=None if healthy else RuntimeError("offline database"),
        ),
    )
    embedder: Mock = Mock()
    embedder.embed_query.return_value = [1.0, 0.0]
    monkeypatch.setattr(
        common_functions,
        "load_embedding_model",
        Mock(
            return_value=(embedder, 2),
            side_effect=None if healthy else RuntimeError("missing embedder"),
        ),
    )
    monkeypatch.setattr(
        ollama,
        "list",
        Mock(side_effect=RuntimeError("no local service")),
    )
    for key in [
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)
    if healthy:
        monkeypatch.setenv("OPENAI_API_KEY", "unit")
    monkeypatch.setattr(
        app_module,
        "_probe_openai_models",
        Mock(return_value="unit reachable"),
    )
    result: DoctorReport = client.get(
        "/doctor",
        params={"write_probe_dir": str(tmp_path), "probe_models": "true"},
    ).json()
    checks: dict[str, DoctorCheck] = {c["check"]: c for c in result["checks"]}
    assert checks["write_permissions"]["status"] == "ok"
    assert result["summary"]["fail"] == (0 if healthy else 4)
    assert checks["neo4j"]["status"] == ("ok" if healthy else "fail")
    if healthy:
        assert checks["openai_connectivity"]["detail"] == "unit reachable"
    assert not list(tmp_path.glob(".ontographrag_write_probe*"))


def test_authenticated_request_reaches_readiness_guard(
    app_module: AppExports,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "_APP_API_KEY", "unit")
    assert client.get("/kg/list", headers={"X-API-Key": "unit"}).status_code == 503


@pytest.mark.parametrize(
    "outcome",
    ["built", "reload_failed", "deduplicated", "builder_failed", "invalid_ontology"],
)
def test_kg_build_lifecycle_and_cleanup(
    app_module: AppExports,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: str,
) -> None:
    from ontographrag.kg.loaders import kg_loader

    monkeypatch.setattr(app_module, "require_neo4j", lambda: None)
    driver: MagicMock = MagicMock()
    session: MagicMock = driver.session.return_value.__enter__.return_value
    session.run.return_value.single.return_value = (
        {"kgName": "unit", "fileName": "doc.txt"} if outcome == "deduplicated" else None
    )
    monkeypatch.setattr(app_module, "get_graphDB_driver", Mock(return_value=driver))
    builder: Mock = Mock(
        ontology_path=None,
        ontology_classes=[],
        ontology_relationships=[],
    )
    built: BuiltGraph = {
        "nodes": [{"id": "new"}],
        "metadata": {
            "total_entities": 1,
            "stored_relationships": 0,
            "stored_in_neo4j": True,
        },
    }
    builder.generate_knowledge_graph.return_value = built
    if outcome == "builder_failed":
        builder.generate_knowledge_graph.side_effect = RuntimeError(
            "write transaction failed",
        )
    factory: Mock = Mock(return_value=builder)
    monkeypatch.setattr(app_module, "UnifiedOntologyGuidedKGCreator", factory)
    monkeypatch.setattr(app_module, "get_llm_provider", Mock(return_value=Mock()))
    invalidated: Mock = Mock()
    monkeypatch.setattr(app_module, "invalidate_rag_system", invalidated)
    loaded: LoadedGraph = (
        {"status": "error"}
        if outcome == "reload_failed"
        else {"status": "success", "nodes": [{"id": "persisted"}], "relationships": []}
    )
    loader: Mock = Mock(load_from_neo4j=Mock(return_value=loaded))
    monkeypatch.setattr(kg_loader, "KGLoader", Mock(return_value=loader))
    ontology: OntologyFixture = {
        "classes": [{"id": "Drug", "properties": [{"name": "id", "identifier": True}]}],
    }
    if outcome == "invalid_ontology":
        ontology = {"classes": [{}]}
    response: httpx.Response = client.post(
        "/create_ontology_guided_kg",
        data={"kg_name": "unit", "max_chunks": "2"},
        files={
            "file": ("doc.txt", b"Aspirin treats illness"),
            "ontology_file": ("schema.json", json.dumps(ontology).encode()),
        },
    )
    assert app_module._kg_building is False
    assert list(tmp_path.glob("ontology_*")) == []
    driver.close.assert_called_once()
    if outcome in {"builder_failed", "invalid_ontology"}:
        assert response.status_code == (500 if outcome == "builder_failed" else 422)
        invalidated.assert_not_called()
    else:
        assert response.status_code == 200, response.text
        data: BuildResponse = response.json()
        assert data["kg_name"] == "unit"
        if outcome == "deduplicated":
            assert data["deduplicated"] is True
            factory.assert_not_called()
        else:
            assert data["method"] == "ontology_guided"
            assert data["graph_data"] == (
                built if outcome == "reload_failed" else loaded
            )
            assert factory.call_args.kwargs["strict_ontology"] is True
            invalidated.assert_called_once()
            assert builder.generate_knowledge_graph.call_args.args[5] == "unit"


def test_kg_empty_input_and_chunk_limits_fail_before_build(
    app_module: AppExports,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "require_neo4j", lambda: None)
    monkeypatch.setattr(
        app_module,
        "get_graphDB_driver",
        Mock(side_effect=RuntimeError("dedup unavailable")),
    )
    factory: Mock = Mock()
    monkeypatch.setattr(app_module, "UnifiedOntologyGuidedKGCreator", factory)
    for limit, expected in [(0, 422), (501, 422), (1, 400)]:
        response: httpx.Response = client.post(
            "/create_ontology_guided_kg",
            data={"max_chunks": str(limit)},
            files={"file": ("empty.txt", b" ")},
        )
        assert response.status_code == expected
    factory.assert_not_called()
    assert not app_module._kg_building


@pytest.mark.parametrize("fails", [False, True])
def test_file_build_readback_and_worker_failure(
    app_module: AppExports,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fails: bool,
) -> None:
    monkeypatch.setattr(app_module, "require_neo4j", lambda: None)
    monkeypatch.setattr(app_module, "get_llm_provider", Mock(return_value=Mock()))
    builder: Mock = Mock()
    if fails:
        builder.generate_knowledge_graph.side_effect = RuntimeError("worker failed")
    monkeypatch.setattr(
        app_module,
        "UnifiedOntologyGuidedKGCreator",
        Mock(return_value=builder),
    )
    driver: MagicMock = MagicMock()
    driver.session.return_value.__enter__.return_value.run.return_value = []
    monkeypatch.setattr(app_module, "get_graphDB_driver", Mock(return_value=driver))
    response: httpx.Response = client.post(
        "/load_kg_from_file",
        data={"kg_name": "unit"},
        files={"file": ("doc.txt", b"Aspirin treats illness")},
    )
    assert response.status_code == (500 if fails else 200), response.text
    if not fails:
        assert response.json()["graph_data"] == {"nodes": [], "relationships": []}
        driver.close.assert_called_once()
        assert all(
            call.args[1]["kg_name"] == "unit"
            for call in driver.session.return_value.__enter__.return_value.run.call_args_list
        )


@pytest.mark.parametrize("outcome", ["success", "missing_password", "worker_failure"])
def test_bulk_job_cleans_csv_and_invalidates_only_success(
    app_module: AppExports,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: str,
) -> None:
    monkeypatch.setattr(app_module, "require_neo4j", lambda: None)
    monkeypatch.setenv("NEO4J_PASSWORD", "dummy")
    if outcome == "missing_password":
        monkeypatch.delenv("NEO4J_PASSWORD")
    monkeypatch.setattr(app_module, "get_llm_provider", Mock(return_value=Mock()))
    builder: Mock = Mock()
    builder.bulk_process_documents.return_value = {
        "metadata": {"total_documents_processed": 1, "total_knowledge_graphs": 1},
    }
    if outcome == "worker_failure":
        builder.bulk_process_documents.side_effect = RuntimeError("worker unavailable")
    monkeypatch.setattr(
        app_module,
        "UnifiedOntologyGuidedKGCreator",
        Mock(return_value=builder),
    )
    invalidate: Mock = Mock()
    monkeypatch.setattr(app_module, "invalidate_rag_system", invalidate)
    response: httpx.Response = client.post(
        "/bulk_process_csv",
        data={"kg_name": "unit", "text_column": "text", "batch_size": "2"},
        files={"csv_file": ("doc.csv", b"text\nevidence\n")},
    )
    assert list(tmp_path.glob("bulk_*")) == []
    if outcome == "success":
        assert (
            response.status_code == 200
            and response.json()["total_documents_processed"] == 1
        )
        assert builder.bulk_process_documents.call_args.kwargs["kg_name"] == "unit"
        invalidate.assert_called_once()
    else:
        assert response.status_code == 500
        invalidate.assert_not_called()


def test_preview_never_requests_persistence(
    app_module: AppExports,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "require_neo4j", lambda: None)
    monkeypatch.setattr(app_module, "get_llm_provider", Mock(return_value=Mock()))
    builder: Mock = Mock(
        generate_knowledge_graph=Mock(
            return_value={
                "nodes": [],
                "relationships": [],
                "metadata": {"total_entities": 0},
            },
        ),
    )
    monkeypatch.setattr(
        app_module,
        "UnifiedOntologyGuidedKGCreator",
        Mock(return_value=builder),
    )
    invalidate: Mock = Mock()
    monkeypatch.setattr(app_module, "invalidate_rag_system", invalidate)
    response: httpx.Response = client.post(
        "/extract_graph",
        files={"file": ("doc.txt", b"evidence")},
    )
    assert response.status_code == 200
    assert builder.generate_knowledge_graph.call_args.args[2] is None
    invalidate.assert_not_called()


def _install_readback_driver(
    app_module: AppExports,
    monkeypatch: pytest.MonkeyPatch,
    *,
    named: bool,
    fail_query: bool = False,
) -> MagicMock:
    from datetime import date

    class Node(dict[str, str | date]):
        def __init__(self, identity: int, application_id: str, label: str) -> None:
            super().__init__(
                id=application_id,
                name=f"node-{identity}",
                created=date(2026, 1, 1),
            )
            self.id: int = identity
            self.labels: set[str] = {label}

    class Relation(dict[str, float]):
        id: int = 3
        type: str = "TREATS"

    drug: Node = Node(1, "drug-aspirin", "Drug")
    disease: Node = Node(2, "disease-pain", "Disease")
    doc: Node = Node(0, "document-unit", "Document")
    relation: Relation = Relation(confidence=0.8)
    monkeypatch.setattr(app_module, "require_neo4j", lambda: None)
    driver: MagicMock = MagicMock()
    session: MagicMock = driver.session.return_value.__enter__.return_value

    def run(
        statement: str,
        params: dict[str, list[int] | list[str]] | None = None,
    ) -> Mock | list[dict[str, Node | Relation | None]]:
        if fail_query:
            raise RuntimeError("readback query failed")
        if "shared_count" in statement:
            return Mock(single=Mock(return_value={"shared_count": 1}))
        if "count(d) AS c" in statement:
            return Mock(single=Mock(return_value={"c": int(named)}))
        if "count(" in statement:
            return Mock(single=Mock(return_value={"c": 4}))
        if "AS provider" in statement:
            return Mock(
                single=Mock(
                    return_value={
                        "provider": "unit",
                        "model": "fake",
                        "embeddingModel": "fake",
                        "maxChunks": 2,
                    },
                ),
            )
        if "RETURN d" in statement:
            return [{"d": doc}]
        if "RETURN DISTINCT e" in statement:
            return [{"e": drug}, {"e": drug}, {"e": disease}, {"e": None}]
        if "RETURN n " in statement:
            return [{"n": drug}, {"n": disease}]
        if "RETURN r," in statement:
            assert params is not None
            selected: set[int | str]
            match: bool
            if (
                "id(a) IN $entity_ids" in statement
                and "id(b) IN $entity_ids" in statement
            ):
                selected = set(params["entity_ids"])
                match = {drug.id, disease.id}.issubset(selected)
            elif "n.id IN $node_ids" in statement and "m.id IN $node_ids" in statement:
                selected = set(params["node_ids"])
                match = {drug["id"], disease["id"]}.issubset(selected)
            elif (
                "id(n) IN $node_ids" in statement and "id(m) IN $node_ids" in statement
            ):
                selected = set(params["node_ids"])
                match = {drug.id, disease.id}.issubset(selected)
            else:
                raise AssertionError(
                    f"Unrecognized relationship identity contract: {statement}",
                )
            # Match actual identity semantics: internal numeric IDs never equal
            # application IDs. A mismatched query returns no relationships.
            return (
                [
                    {"r": relation, "start": drug, "end": disease},
                    {"r": None, "start": drug, "end": None},
                ]
                if match
                else []
            )
        raise AssertionError(statement)

    session.run.side_effect = run
    monkeypatch.setattr(app_module, "get_graphDB_driver", Mock(return_value=driver))
    return driver


@pytest.mark.parametrize(
    "named,sample,complete",
    [(True, False, False), (True, True, True), (False, False, False)],
)
def test_database_visualization_load_serializes_and_deduplicates(
    app_module: AppExports,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    named: bool,
    sample: bool,
    complete: bool,
) -> None:
    driver: MagicMock = _install_readback_driver(app_module, monkeypatch, named=named)
    response: httpx.Response = client.post(
        "/load_kg_from_neo4j",
        data={
            "kg_label": "unit" if named else "",
            "sample_mode": str(sample).lower(),
            "load_complete": str(complete).lower(),
            "limit": "2",
        },
    )
    assert response.status_code == 200, response.text
    data: ReadbackResponse = response.json()
    assert data["stats"]["loaded_nodes"] == (3 if named else 2)
    nodes: list[ReadbackNode] = data["graph_data"]["nodes"]
    assert len({node["id"] for node in nodes}) == len(nodes)
    assert nodes[0]["properties"]["created"] == "2026-01-01"
    assert {node["properties"]["id"] for node in nodes} >= {
        "drug-aspirin",
        "disease-pain",
    }
    driver.session.return_value.__exit__.assert_called_once()
    if named:
        assert data["stats"]["loaded_relationships"] == 1
        assert data["graph_data"]["relationships"][0]["start"] == 1
        assert data["kg_settings"]["provider"] == "unit"
        calls: list[_Call] = (
            driver.session.return_value.__enter__.return_value.run.call_args_list
        )
        entity_query: str = next(
            c.args[0] for c in calls if "RETURN DISTINCT e" in c.args[0]
        )
        assert ("LIMIT 2" in entity_query) is not complete
        assert ("ORDER BY rand()" in entity_query) is sample
    # Fallback relationship correctness is a separate strict regression below.


@pytest.mark.xfail(
    strict=True,
    raises=ReadbackIdentityDefect,
    reason="Fallback compares numeric internal IDs against string node.id properties",
)
def test_readback_fallback_preserves_relationship_identity(
    app_module: AppExports,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver: MagicMock = _install_readback_driver(app_module, monkeypatch, named=False)
    response: httpx.Response = client.post("/load_kg_from_neo4j")
    assert response.status_code == 200, response.text
    data: ReadbackResponse = response.json()
    assert {node["properties"]["id"] for node in data["graph_data"]["nodes"]} == {
        "drug-aspirin",
        "disease-pain",
    }
    calls: list[_Call] = (
        driver.session.return_value.__enter__.return_value.run.call_args_list
    )
    relationship_calls: list[_Call] = [c for c in calls if "RETURN r," in c.args[0]]
    assert len(relationship_calls) == 1
    query: str
    params: dict[str, list[int] | list[str]]
    query, params = relationship_calls[0].args
    if (
        "n.id IN $node_ids" in query
        and "m.id IN $node_ids" in query
        and params == {"node_ids": [1, 2]}
        and data["graph_data"]["relationships"] == []
    ):
        raise ReadbackIdentityDefect(
            "Known internal/application ID mismatch removed the existing edge",
        )
    if "n.id IN $node_ids" in query and "m.id IN $node_ids" in query:
        assert set(params["node_ids"]) == {"drug-aspirin", "disease-pain"}
    else:
        assert "id(n) IN $node_ids" in query and "id(m) IN $node_ids" in query
        assert set(params["node_ids"]) == {1, 2}
    assert data["stats"]["loaded_relationships"] == 1
    assert data["graph_data"]["relationships"][0]["type"] == "TREATS"


@pytest.mark.parametrize(
    "named,fail_query",
    [(True, False), (False, False), (False, True)],
    ids=["named_success", "fallback_success", "query_exception"],
)
@pytest.mark.xfail(
    strict=True,
    raises=ReadbackDriverLeak,
    reason="Request-owned readback driver is not closed on success or exception",
)
def test_readback_always_closes_owned_driver(
    app_module: AppExports,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    named: bool,
    fail_query: bool,
) -> None:
    driver: MagicMock = _install_readback_driver(
        app_module,
        monkeypatch,
        named=named,
        fail_query=fail_query,
    )
    response: httpx.Response = client.post(
        "/load_kg_from_neo4j",
        data={"kg_label": "unit" if named else ""},
    )
    assert response.status_code == (500 if fail_query else 200), response.text
    driver.session.return_value.__exit__.assert_called_once()
    if fail_query:
        session: MagicMock = driver.session.return_value.__enter__.return_value
        assert session.run.call_count == 1
        exc: BaseException = driver.session.return_value.__exit__.call_args.args[1]
        assert isinstance(exc, RuntimeError) and str(exc) == "readback query failed"
    else:
        assert response.json()["stats"]["loaded_nodes"] == (3 if named else 2)
    if driver.close.call_count == 0:
        raise ReadbackDriverLeak(
            "Completed owned session but driver.close was never called",
        )
    driver.close.assert_called_once_with()


@pytest.mark.xfail(
    strict=True,
    raises=CreatorSignatureDefect,
    reason="KG build endpoint passes provider positional argument absent from real creator signature",
)
def test_kg_build_call_matches_actual_creator_signature(
    app_module: AppExports,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect

    from ontographrag.kg.builders.enhanced_kg_creator import (
        UnifiedOntologyGuidedKGCreator,
    )

    monkeypatch.setattr(app_module, "require_neo4j", lambda: None)
    monkeypatch.setattr(
        app_module,
        "get_graphDB_driver",
        Mock(side_effect=RuntimeError("dedup unavailable")),
    )
    monkeypatch.setattr(app_module, "get_llm_provider", Mock(return_value=Mock()))
    builder: Mock = Mock(
        ontology_path=None,
        ontology_classes=[],
        ontology_relationships=[],
        generate_knowledge_graph=Mock(return_value={"metadata": {}}),
    )
    monkeypatch.setattr(
        app_module,
        "UnifiedOntologyGuidedKGCreator",
        Mock(return_value=builder),
    )
    response: httpx.Response = client.post(
        "/create_ontology_guided_kg",
        files={"file": ("doc.txt", b"Aspirin treats pain.")},
    )
    assert response.status_code == 200
    call: _Call = builder.generate_knowledge_graph.call_args
    signature: Signature = inspect.signature(
        UnifiedOntologyGuidedKGCreator.generate_knowledge_graph,
    )
    try:
        signature.bind(builder, *call.args, **call.kwargs)
    except TypeError as exc:
        if (
            str(exc) == "too many positional arguments"
            and len(call.args) == 9
            and call.args[-1] == "openai"
        ):
            # The remaining argument sequence must bind successfully: this sentinel
            # must not cover a different missing/extra/invalid argument contract.
            signature.bind(builder, *call.args[:-1], **call.kwargs)
            raise CreatorSignatureDefect(
                "Only final provider argument exceeds real creator signature",
            ) from exc
        raise
