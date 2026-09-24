"""Exercise the real HTTP stack, stubbing only unrelated RAG import boundaries."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import logging
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from ontographrag.api import teapot as files


@pytest.fixture
def api(monkeypatch, tmp_path):
    boundaries = {
        "ontographrag.providers.model_providers": [
            "get_provider", "LangChainRunnableAdapter", "TemperatureLockedProvider",
        ],
        "ontographrag.kg.builders.enhanced_kg_creator": ["UnifiedOntologyGuidedKGCreator"],
        "ontographrag.rag.systems.enhanced_rag_system": ["EnhancedRAGSystem"],
        "ontographrag.rag.answer_guardrails": ["RUNTIME_GUARDRAIL_ABSTENTION"],
        "ontographrag.kg.csv_processor": ["MedicalReportCSVProcessor"],
        "ontographrag.kg.utils.graph_query": ["get_graphDB_driver"],
    }
    for name, attributes in boundaries.items():
        stub = ModuleType(name)
        for attribute in attributes:
            setattr(stub, attribute, Mock(side_effect=AssertionError("Unexpected RAG call")))
        monkeypatch.setitem(sys.modules, name, stub)
    app_path = Path(__file__).resolve().parents[1] / "ontographrag/api/app.py"
    spec = importlib.util.spec_from_file_location("_teapot_test_app", app_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._PROJECT_ROOT == app_path.parents[2]
    monkeypatch.setattr(module, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "_APP_API_KEY", None)
    return module


@pytest.fixture
def client(api):
    with TestClient(api.app, raise_server_exceptions=False) as client:
        yield client


def assert_artifact(root, response):
    assert response.status_code == 418
    assert response.headers["content-type"] == "application/json"
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body == {
        "status": "teapot", "detail": "I'm a teapot",
        "filename": body["filename"], "request_id": response.headers["x-request-id"],
    }
    assert re.fullmatch(r"teapot-[0-9a-f]{32}\.txt", body["filename"])
    assert re.fullmatch(rb"[0-9]{32}\n", (root / body["filename"]).read_bytes())
    return body["filename"]


def test_repeated_requests_root_and_correlated_logs(client, api, tmp_path, monkeypatch, caplog):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    caplog.set_level(logging.INFO, logger=api.__name__)
    names = [assert_artifact(tmp_path, client.get(
        "/teapot?api_key=do-not-log", headers={"X-Request-ID": "same-id"},
    )) for _ in range(2)]
    assert len(set(names)) == 2
    assert list(elsewhere.iterdir()) == []
    assert len(list(tmp_path.glob("teapot-*.txt"))) == 2
    assert "do-not-log" not in caplog.text
    records = [r.getMessage() for r in caplog.records if r.name == api.__name__]
    assert len(records) == 4
    assert all("request_id='same-id' method=GET route=/teapot" in r for r in records)
    assert sum("outcome=received" in r for r in records) == 2
    assert all(any(f"outcome=completed filename={name}" in r for r in records) for name in names)


def test_concurrent_requests(client, tmp_path):
    existing = tmp_path / "teapot-existing.txt"
    existing.write_text("preserve me")
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: client.get("/teapot"), range(24)))
    names = [assert_artifact(tmp_path, response) for response in responses]
    assert len(set(names)) == 24
    assert len(list(tmp_path.iterdir())) == 25
    assert existing.read_text() == "preserve me"


@pytest.mark.parametrize("method", ["HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
def test_unsupported_methods_have_no_side_effects(client, tmp_path, method):
    assert client.request(method, "/teapot").status_code == 405
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "incorrect"}])
def test_auth_denial_has_no_side_effects(client, api, tmp_path, monkeypatch, headers):
    monkeypatch.setattr(api, "_APP_API_KEY", "test-key")
    response = client.get("/teapot", headers=headers)
    assert response.status_code == 401
    assert response.json()["status"] == "error"
    assert response.json()["request_id"] == response.headers["x-request-id"]
    assert list(tmp_path.iterdir()) == []


def test_valid_auth_and_health_exemption(client, api, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "_APP_API_KEY", "test-key")
    assert client.get("/health").status_code == 200
    assert_artifact(tmp_path, client.get("/teapot", headers={"X-API-Key": "test-key"}))
    assert_artifact(tmp_path, client.get("/teapot?api_key=test-key"))


@pytest.mark.parametrize("failure", ["create", "write", "publish"])
def test_filesystem_failure_is_sanitized_and_cleaned(
    client, api, tmp_path, monkeypatch, caplog, failure,
):
    def fail(*args, **kwargs):
        raise OSError("private filesystem path /secret/root")

    if failure == "create":
        monkeypatch.setattr(files.tempfile, "NamedTemporaryFile", fail)
    elif failure == "publish":
        monkeypatch.setattr(files.os, "link", fail)
    else:
        original = files.tempfile.NamedTemporaryFile

        def partial_write(*args, **kwargs):
            staging = original(*args, **kwargs)

            def write(contents):
                staging.file.write(contents[:4])
                staging.file.flush()
                fail()

            staging.write = write
            return staging

        monkeypatch.setattr(files.tempfile, "NamedTemporaryFile", partial_write)
    caplog.set_level(logging.INFO, logger=api.__name__)
    response = client.get("/teapot", headers={"X-Request-ID": "failed-id"})
    assert response.status_code == 500
    assert response.json() == {
        "status": "error", "detail": "Could not create teapot file", "request_id": "failed-id",
    }
    assert response.headers["x-request-id"] == "failed-id"
    assert list(tmp_path.iterdir()) == []
    assert "request_id='failed-id' method=GET route=/teapot outcome=failed" in caplog.text
    assert "outcome=completed" not in caplog.text
    assert "/secret/root" not in response.text


def test_collision_does_not_overwrite_and_leading_zeroes_survive(tmp_path, monkeypatch):
    existing = tmp_path / "teapot-collision.txt"
    existing.write_text("preserve me")
    ids = iter(["collision", "unique"])
    monkeypatch.setattr(files.uuid, "uuid4", lambda: SimpleNamespace(hex=next(ids)))
    monkeypatch.setattr(files.secrets, "choice", lambda digits: "0")
    filename = files.create_teapot_file(tmp_path)
    assert filename == "teapot-unique.txt"
    assert (tmp_path / filename).read_bytes() == b"0" * 32 + b"\n"
    assert existing.read_text() == "preserve me"
    assert len(list(tmp_path.iterdir())) == 2


def test_exhausted_collisions_are_bounded_and_cleaned(tmp_path, monkeypatch):
    existing = tmp_path / "teapot-collision.txt"
    existing.write_text("preserve me")
    monkeypatch.setattr(files.uuid, "uuid4", lambda: SimpleNamespace(hex="collision"))
    with pytest.raises(FileExistsError):
        files.create_teapot_file(tmp_path)
    assert list(tmp_path.iterdir()) == [existing]
    assert existing.read_text() == "preserve me"


def test_openapi_documents_418(client):
    operation = client.get("/openapi.json").json()["paths"]["/teapot"]["get"]
    assert "418" in operation["responses"]
    assert "application/json" in operation["responses"]["418"]["content"]
    assert "200" not in operation["responses"]
