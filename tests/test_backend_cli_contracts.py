"""CLI transport and local artifact commands using fake HTTP/subprocess boundaries."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock

import httpx
import pytest
from click.testing import Result
from typer.testing import CliRunner
from typing_extensions import TypedDict

from ontographrag import cli


class BenchmarkDataset(TypedDict, total=False):
    path: str
    source: str
    download: str
    prep: str


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    client: MagicMock = MagicMock()
    factory: MagicMock = MagicMock()
    factory.return_value.__enter__.return_value = client
    monkeypatch.setattr(httpx, "Client", factory)
    return client


@pytest.mark.parametrize("as_json", [False, True])
def test_ask_payload_and_output(transport: MagicMock, as_json: bool) -> None:
    transport.post.return_value = httpx.Response(
        200,
        json={
            "response": "answer",
            "info": {
                "kg_confidence": 0.8,
                "entities": {"used_entities": [{"description": "source"}]},
            },
        },
    )
    result: Result = CliRunner().invoke(
        cli.app,
        [
            "ask",
            "question",
            "--kg-name",
            "unit",
            "--api-key",
            "unit",
            *(["--json"] if as_json else []),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "answer" in result.output
    assert transport.post.call_args.kwargs["json"]["kg_name"] == "unit"
    assert transport.post.call_args.kwargs["headers"] == {"X-API-Key": "unit"}
    if not as_json:
        assert "KG confidence: 0.800" in result.output
        assert "Sources: source" in result.output


def test_ingest_uploads_document_and_ontology(
    transport: MagicMock,
    tmp_path: Path,
) -> None:
    path: Path = tmp_path / "doc.txt"
    path.write_text("synthetic document")
    ontology: Path = tmp_path / "ontology.json"
    ontology.write_text("{}")
    transport.post.return_value = httpx.Response(
        200,
        json={
            "kg_name": "unit",
            "graph_data": {"nodes": [{"id": "a"}], "relationships": []},
        },
    )
    result: Result = CliRunner().invoke(
        cli.app,
        [
            "ingest",
            str(path),
            "--ontology",
            str(ontology),
            "--kg-name",
            "unit",
            "--enable-coreference",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Nodes: 1" in result.output
    assert transport.post.call_args.kwargs["files"]["file"][1] == b"synthetic document"
    assert transport.post.call_args.kwargs["files"]["ontology_file"][1] == b"{}"
    assert (
        transport.post.call_args.kwargs["data"]["enable_coreference_resolution"]
        == "true"
    )


@pytest.mark.parametrize(
    "body",
    ["not json", '{"detail":"denied"}', '{"message":"denied"}'],
)
def test_http_error_exit(transport: MagicMock, body: str) -> None:
    transport.post.return_value = httpx.Response(403, text=body)
    result: Result = CliRunner().invoke(cli.app, ["ask", "q"])
    assert result.exit_code == 2
    assert "403 Forbidden" in result.output


def test_download_stream_and_prepare_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    response: Mock = Mock(is_success=True)
    response.iter_bytes.return_value = iter([b"first", b"second"])
    stream: MagicMock = MagicMock()
    stream.return_value.__enter__.return_value = response
    monkeypatch.setattr(httpx, "stream", stream)
    path: Path = tmp_path / "data/raw.json"
    cli._download_to("https://unit.invalid/data", path)
    assert path.read_bytes() == b"firstsecond"
    specs: dict[str, BenchmarkDataset] = {
        "unit": {"path": str(path), "source": "https://unit.invalid"},
        "missing": {
            "path": str(tmp_path / "missing.json"),
            "source": "https://unit.invalid",
            "download": "https://unit.invalid/data",
        },
        "fullwiki": {"path": str(path), "source": "unit", "prep": "fullwiki"},
    }
    monkeypatch.setattr(cli, "_BENCHMARK_DATASETS", specs)
    runner: CliRunner = CliRunner()
    assert runner.invoke(cli.app, ["prepare", "unit"]).exit_code == 0
    assert runner.invoke(cli.app, ["prepare", "missing"]).exit_code == 1
    assert runner.invoke(cli.app, ["prepare", "unit", "--download"]).exit_code == 2
    assert runner.invoke(cli.app, ["prepare", "unknown"]).exit_code == 2
    assert "unit" in runner.invoke(cli.app, ["datasets"]).output
    call: Mock = Mock(return_value=7)
    monkeypatch.setattr(subprocess, "call", call)
    assert (
        runner.invoke(cli.app, ["prepare", "fullwiki", "--num-samples", "2"]).exit_code
        == 7
    )
    assert call.call_args.args[0][-2:] == ["--num-samples", "2"]
