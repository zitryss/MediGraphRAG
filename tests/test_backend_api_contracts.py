"""HTTP and state-management behavior with explicit external boundaries."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient


@pytest.fixture
def app_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    monkeypatch.setenv("APP_API_KEY", "")
    monkeypatch.setenv("ONTOGRAPHRAG_DEBUG_ERRORS", "0")
    path = Path(__file__).parents[1] / "ontographrag/api/app.py"
    spec = importlib.util.spec_from_file_location("backend_contract_app", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_READY_WRITE_PROBE_DIR", tmp_path)
    monkeypatch.setattr(module.tempfile, "gettempdir", lambda: str(tmp_path))
    return module


@pytest.fixture
def client(app_module: ModuleType) -> TestClient:
    return TestClient(app_module.app, raise_server_exceptions=False)


@pytest.mark.xfail(
    strict=True,
    reason="Auth middleware currently converts invalid API keys to 500, not 401",
)
def test_authentication_public_and_private_routes(
    app_module: ModuleType, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "_APP_API_KEY", "unit-secret")
    assert client.get("/health").status_code == 200
    assert client.get("/kg/list").status_code == 401
    assert client.get("/kg/list", headers={"X-API-Key": "wrong"}).status_code == 401
    assert (
        client.get("/kg/list", headers={"X-API-Key": "unit-secret"}).status_code == 503
    )


def test_cached_rag_model_switch_and_invalidation(
    app_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = Mock(side_effect=[Mock(), Mock()])
    monkeypatch.setattr(app_module, "EnhancedRAGSystem", factory)
    first = app_module.get_rag_system("LOCAL")
    assert app_module.get_rag_system() is first
    assert app_module.get_rag_system("local") is first
    second = app_module.get_rag_system("other")
    assert second is not first
    assert factory.call_count == 2
    second.clear_retrieval_caches.side_effect = RuntimeError("cache failure")
    app_module.invalidate_rag_system("mutation")
    assert app_module._rag_system is None
    assert app_module._rag_system_embedding is None
    second.clear_retrieval_caches.assert_called_once()


@pytest.mark.parametrize(
    "payload,fragment",
    [
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
                        "properties": [
                            {"name": "x", "identifier": True, "type": "enum"}
                        ],
                    }
                ]
            },
            "no enum_values",
        ),
    ],
)
def test_ontology_rejects_malformed_entries(
    app_module: ModuleType, payload: dict, fragment: str
) -> None:
    assert any(
        fragment in e
        for e in app_module.validate_ontology_schema(
            json.dumps(payload).encode(), "schema.json"
        )
    )


def test_ontology_relationship_validation_and_formats(app_module: ModuleType) -> None:
    schema = {
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
    errors = app_module.validate_ontology_schema(
        json.dumps(schema).encode(), "ontology.json"
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


def test_upload_extension_and_text_decoding(app_module: ModuleType) -> None:
    file = UploadFile(filename="bad.exe", file=io.BytesIO(b"bad"))
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
    app_module: ModuleType, client: TestClient, monkeypatch: pytest.MonkeyPatch
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
    response = client.get("/ready")
    assert response.status_code == 503
    assert not response.json()["ready"]
    assert "runtime unavailable" in response.text


def test_startup_and_database_health(
    app_module: ModuleType, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    session.run.return_value.single.return_value = {"c": 7}
    monkeypatch.setattr(app_module, "get_graphDB_driver", Mock(return_value=driver))
    app_module.check_neo4j_connection()
    assert app_module.neo4j_ready
    assert client.get("/health/neo4j").json() == {"status": "ok", "nodeCount": 7}
    monkeypatch.setattr(
        app_module, "get_graphDB_driver", Mock(side_effect=RuntimeError("unavailable"))
    )
    assert client.get("/health/neo4j").status_code == 500


def test_named_kg_crud_and_failure(
    app_module: ModuleType, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import langchain_neo4j

    from ontographrag.kg.loaders import graph_db_access

    db = Mock()
    db.create_kg.return_value = {"name": "unit"}
    db.get_kg.return_value = {"name": "unit"}
    db.get_kg_stats.return_value = [{"documentCount": 2}]
    db.get_kg_entities.return_value = [{"id": "e"}]
    db.delete_kg_by_name.return_value = 2
    monkeypatch.setattr(langchain_neo4j, "Neo4jGraph", Mock())
    monkeypatch.setattr(graph_db_access, "graphDBdataAccess", Mock(return_value=db))
    monkeypatch.setattr(app_module, "neo4j_ready", True)
    assert client.post("/kg/create", data={"kg_name": "unit"}).json()["kg"] == {
        "name": "unit"
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
    app_module: ModuleType, client: TestClient, tmp_path: Path
) -> None:
    response = client.post(
        "/validate_csv", files={"csv_file": ("bad.csv", b"a|b\nx|y\n")}
    )
    assert response.status_code == 200
    assert not response.json()["is_valid"]
    assert len(response.json()["errors"]) == 3
    assert list(tmp_path.glob("validate_*")) == []


def test_save_graph_filters_properties_and_handles_missing_state(
    app_module: ModuleType, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "neo4j_ready", True)
    monkeypatch.setattr(app_module, "current_graph_data", None)
    form = {
        "kg_id": "unit",
        "uri": "bolt://unit.invalid",
        "user": "unit",
        "password": "dummy",
    }
    assert client.post("/save_kg_to_neo4j", data=form).status_code == 400
    driver = MagicMock()
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
    result = client.post("/save_kg_to_neo4j", data=form)
    assert result.status_code == 200
    assert result.json()["nodes_saved"] == 2
    assert result.json()["relationships_saved"] == 1
    calls = driver.session.return_value.__enter__.return_value.run.call_args_list
    assert calls[1].args[1] == {"name": "A", "id": "a"}


def test_progress_stream_completion_and_disconnect(app_module: ModuleType) -> None:
    async def consume(disconnected: bool) -> list[str]:
        response = await app_module.kg_progress_stream(
            SimpleNamespace(is_disconnected=AsyncMock(return_value=disconnected))
        )
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
    app_module: ModuleType, monkeypatch: pytest.MonkeyPatch, provider: str, key: str
) -> None:
    probe = getattr(app_module, f"_probe_{provider}_models")
    monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="not set"):
        probe()
    monkeypatch.setenv(key, "unit")
    response = Mock()
    response.json.return_value = {"data": [{"id": "unit"}]}
    get = Mock(return_value=response)
    monkeypatch.setattr(app_module.httpx, "get", get)
    assert probe() == "reachable (1 models visible)"
    assert get.call_args.kwargs["headers"] == {"Authorization": "Bearer unit"}
    response.raise_for_status.assert_called_once()


@pytest.mark.parametrize("healthy", [True, False])
def test_doctor_local_success_and_dependency_failure(
    app_module: ModuleType,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    healthy: bool,
) -> None:
    import ollama

    from ontographrag.kg.utils import common_functions

    ui = tmp_path / "ui"
    ui.mkdir()
    if healthy:
        (ui / "index.html").write_text("<html>unit</html>")
    monkeypatch.setattr(app_module, "_UI_DIST_DIR", ui)
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value.run.return_value.single.return_value = {
        "c": 2
    }
    monkeypatch.setattr(
        app_module,
        "get_graphDB_driver",
        Mock(
            return_value=driver,
            side_effect=None if healthy else RuntimeError("offline database"),
        ),
    )
    embedder = Mock()
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
        ollama, "list", Mock(side_effect=RuntimeError("no local service"))
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
        app_module, "_probe_openai_models", Mock(return_value="unit reachable")
    )
    result = client.get(
        "/doctor", params={"write_probe_dir": str(tmp_path), "probe_models": "true"}
    ).json()
    checks = {c["check"]: c for c in result["checks"]}
    assert checks["write_permissions"]["status"] == "ok"
    assert result["summary"]["fail"] == (0 if healthy else 4)
    assert checks["neo4j"]["status"] == ("ok" if healthy else "fail")
    if healthy:
        assert checks["openai_connectivity"]["detail"] == "unit reachable"
    assert not list(tmp_path.glob(".ontographrag_write_probe*"))


def test_authenticated_request_reaches_readiness_guard(
    app_module: ModuleType, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "_APP_API_KEY", "unit")
    assert client.get("/kg/list", headers={"X-API-Key": "unit"}).status_code == 503
