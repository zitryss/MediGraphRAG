"""Untouched-source HTTP contracts; no application lifespan or live services."""

import importlib.util
import json
import socket
import sys
from collections import deque
from collections.abc import Callable, Iterator
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import dotenv
import httpx
import pytest
from fastapi.testclient import TestClient

from tests.test_backend_api_contracts import AppExports

if TYPE_CHECKING:
    from ontographrag.providers.model_providers import LangChainRunnableAdapter


@pytest.fixture
def isolated_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[AppExports]:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unexpected external connection")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    env_file: Path = tmp_path / ".env"
    env_file.write_text(
        "ONTOGRAPHRAG_REQUEST_TIMEOUT_SECONDS=240\nALLOWED_ORIGINS=https://baseline.invalid\n"
    )
    real_load: Callable[[Path], bool] = dotenv.load_dotenv

    def controlled_dotenv(*args: object, **kwargs: object) -> bool:
        return real_load(env_file)

    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", controlled_dotenv)
    monkeypatch.setenv("APP_API_KEY", "")
    monkeypatch.setenv("ONTOGRAPHRAG_DEBUG_ERRORS", "0")
    monkeypatch.setenv("ONTOGRAPHRAG_REQUEST_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv(
        "ONTOGRAPHRAG_RUNTIME_ANSWER_GUARDRAIL_MODE", "retry_then_abstain"
    )
    path: Path = Path(__file__).parents[1] / "ontographrag/api/app.py"
    spec: ModuleSpec | None = importlib.util.spec_from_file_location(
        "quality_baseline_app", path
    )
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_graphDB_driver", blocked)
    monkeypatch.setattr(module, "get_rag_system", blocked)
    monkeypatch.setattr(module, "get_llm_provider", blocked)
    # The fixed app.py loader executes the same exports as AppExports.
    yield cast(AppExports, module)


def test_openapi_contract(isolated_app: AppExports) -> None:
    reference: Path = Path(__file__).parent / "fixtures/quality_baseline/openapi.json"
    assert isolated_app.app.openapi() == json.loads(reference.read_text())


def test_import_configuration(
    isolated_app: AppExports, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert isolated_app._REQUEST_TIMEOUT_SECONDS == 15
    assert isolated_app._cors_origins == ["https://baseline.invalid"]
    assert isolated_app._DEBUG_ERRORS is False
    assert isolated_app._APP_API_KEY == ""
    monkeypatch.setenv("ONTOGRAPHRAG_REQUEST_TIMEOUT_SECONDS", "250")
    assert isolated_app._REQUEST_TIMEOUT_SECONDS == 15
    assert isolated_app.neo4j_ready is False
    assert isolated_app._rag_system is None


def test_http_health_error_and_validation(isolated_app: AppExports) -> None:
    client: TestClient = TestClient(isolated_app.app, raise_server_exceptions=False)
    headers: dict[str, str] = {"X-Request-ID": "baseline-request"}
    response: httpx.Response = client.get("/health", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "ontographrag"}
    assert response.headers["x-request-id"] == "baseline-request"
    assert int(response.headers["x-process-time-ms"]) >= 0
    response = client.post("/chat", json={"question": "test"}, headers=headers)
    assert response.status_code == 503
    assert response.json() == {
        "status": "error",
        "detail": "Neo4j database is unavailable. Check connection settings and restart the server.",
        "request_id": "baseline-request",
    }
    response = client.post("/chat", headers=headers)
    assert response.status_code == 422
    assert response.json() == {
        "status": "error",
        "detail": "Request validation failed",
        "errors": [
            {"type": "missing", "loc": ["body"], "msg": "Field required", "input": None}
        ],
        "request_id": "baseline-request",
    }
    client.close()


def test_completed_progress_stream(
    isolated_app: AppExports, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(isolated_app, "_kg_progress", deque(["first", "second"]))
    client: TestClient = TestClient(isolated_app.app)
    response: httpx.Response = client.get("/kg_progress_stream")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert (
        response.text
        == 'data: {"line": "first"}\n\ndata: {"line": "second"}\n\ndata: {"done": true}\n\n'
    )
    client.close()


def test_startup_and_health_graph_calls(
    isolated_app: AppExports, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver: MagicMock = MagicMock()
    factory: MagicMock = MagicMock(return_value=driver)
    session: MagicMock = driver.session.return_value.__enter__.return_value
    session.run.return_value.single.return_value = {"c": 1}
    monkeypatch.setattr(isolated_app, "get_graphDB_driver", factory)
    for key, value in {
        "URI": "bolt://baseline.invalid",
        "USERNAME": "baseline",
        "PASSWORD": "fake",
        "DATABASE": "baseline",
    }.items():
        monkeypatch.setenv(f"NEO4J_{key}", value)
    isolated_app.check_neo4j_connection()
    assert isolated_app.neo4j_ready is True
    factory.assert_called_once_with(
        "bolt://baseline.invalid", "baseline", "fake", "baseline"
    )
    session.run.assert_called_once_with("RETURN 1")
    session.run.reset_mock()
    assert isolated_app.neo4j_health() == {"status": "ok", "nodeCount": 1}
    session.run.assert_called_once_with("RETURN count(*) AS c")
    factory.side_effect = RuntimeError("offline")
    isolated_app.check_neo4j_connection()
    # Existing behavior: a failed re-check does not clear a prior success.
    assert isolated_app.neo4j_ready is True


@pytest.mark.parametrize(
    "body, detail",
    [
        ({}, "Missing question"),
        ({"question": 7}, "Missing question"),
        ({"question": "x" * 4097}, "Question too long (max 4096 chars)"),
        (
            {"question": "test", "document_names": "doc"},
            "document_names must be a list of strings",
        ),
    ],
)
def test_chat_validation(
    isolated_app: AppExports,
    monkeypatch: pytest.MonkeyPatch,
    body: dict[str, object],
    detail: str,
) -> None:
    monkeypatch.setattr(isolated_app, "neo4j_ready", True)
    client: TestClient = TestClient(isolated_app.app)
    response: httpx.Response = client.post(
        "/chat", json=body, headers={"X-Request-ID": "baseline"}
    )
    assert response.status_code == 422
    assert response.json() == {
        "status": "error",
        "detail": detail,
        "request_id": "baseline",
    }
    client.close()


def test_chat_success_contract(
    isolated_app: AppExports, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(isolated_app, "neo4j_ready", True)
    rag: MagicMock = MagicMock()
    rag.generate_response.return_value = {
        "response": "answer",
        "sources": ["doc"],
        "confidence": 0.8,
        "context": {"chunks": [{"text": "evidence"}], "grounding_quality": 0.9},
    }
    rag_factory: MagicMock = MagicMock(return_value=rag)
    provider_factory: MagicMock = MagicMock()
    monkeypatch.setattr(isolated_app, "get_rag_system", rag_factory)
    monkeypatch.setattr(isolated_app, "get_llm_provider", provider_factory)
    client: TestClient = TestClient(isolated_app.app)
    response: httpx.Response = client.post(
        "/chat",
        json={"question": "test", "runtime_guardrail": False},
        headers={"X-Request-ID": "baseline"},
    )
    assert response.status_code == 200
    reference: Path = Path(__file__).parent / "fixtures/quality_baseline/chat.json"
    assert response.json() == json.loads(reference.read_text())
    rag_factory.assert_called_once_with(embedding_model=None)
    provider_factory.assert_called_once_with("openrouter", "openai/gpt-oss-120b:free")
    args: tuple[str, LangChainRunnableAdapter, list[str]] = (
        rag.generate_response.call_args.args
    )
    assert args[0] == "test" and args[2] == []
    assert args[1].provider is provider_factory.return_value
    assert args[1].model == "openai/gpt-oss-120b:free"
    assert args[1].temperature == 1.0
    assert rag.generate_response.call_args.kwargs == {
        "kg_name": None,
        "answer_instructions": "",
        "runtime_guardrail": False,
        "runtime_guardrail_mode": "retry_then_abstain",
    }
    client.close()
