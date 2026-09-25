"""CLI and provider constructor/request characterizations with mocked transports."""

import socket
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import typer
from typer.testing import CliRunner

from ontographrag import cli
from ontographrag.providers import model_providers


@pytest.fixture(autouse=True)
def block_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unexpected external connection")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


def test_cli_ask_defaults_and_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONTOGRAPHRAG_API_KEY", "fake-env-key")
    response: httpx.Response = httpx.Response(
        200,
        json={
            "response": "Grounded answer",
            "info": {
                "kg_confidence": 0.1256,
                "entities": {
                    "used_entities": [{"id": "A"}, {"id": "B", "description": "Beta"}]
                },
            },
        },
    )
    factory: MagicMock = MagicMock()
    client: MagicMock = factory.return_value.__enter__.return_value
    client.post.return_value = response
    monkeypatch.setattr(cli.httpx, "Client", factory)
    result = CliRunner().invoke(
        cli.app, ["ask", "Why?", "--server", "http://baseline.invalid///"]
    )
    assert result.exit_code == 0, result.output
    assert (
        result.output == "Grounded answer\n\nKG confidence: 0.126\nSources: A, Beta\n"
    )
    factory.assert_called_once_with(timeout=180.0)
    client.post.assert_called_once_with(
        "http://baseline.invalid/chat",
        json={"question": "Why?", "provider_rag": "openai", "model_rag": "gpt-4o-mini"},
        headers={"X-API-Key": "fake-env-key"},
    )
    assert cli._request_kwargs("explicit-key") == {
        "headers": {"X-API-Key": "explicit-key"}
    }
    assert cli._request_kwargs("") == {"headers": {"X-API-Key": "fake-env-key"}}
    monkeypatch.delenv("ONTOGRAPHRAG_API_KEY")
    assert cli._request_kwargs(None) == {}


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"detail": "detail", "message": "message"}, "detail"),
        ({"message": "message"}, "message"),
    ],
)
def test_cli_error_detail_precedence(payload: dict[str, str], expected: str) -> None:
    with pytest.raises(
        typer.BadParameter, match=f"503 Service Unavailable: {expected}"
    ):
        cli._raise_for_status(httpx.Response(503, json=payload))


def test_cli_serve_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    run: MagicMock = MagicMock()
    monkeypatch.setattr(uvicorn, "run", run)
    result = CliRunner().invoke(cli.app, ["serve"])
    assert result.exit_code == 0, result.output
    run.assert_called_once_with(
        "ontographrag.api.app:app", host="0.0.0.0", port=8004, reload=False
    )


def test_openai_constructor_and_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setattr(model_providers, "LLM_TIMEOUT_SECONDS", 17.0)
    factory: MagicMock = MagicMock()
    factory.return_value.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
    )
    monkeypatch.setattr(model_providers, "OpenAIClient", factory)
    provider = model_providers.get_provider("OpenAI", model="ignored", temperature=0.25)
    factory.assert_called_once_with(api_key="fake-key", timeout=17.0)
    assert provider.generate("system", "user", "model") == "answer"
    factory.return_value.chat.completions.create.assert_called_once_with(
        model="model",
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        temperature=1.0,
    )
    factory.return_value.chat.completions.create.reset_mock()
    provider.generate("system", "user", "model", temperature=0.0, max_tokens=8)
    assert (
        factory.return_value.chat.completions.create.call_args.kwargs["temperature"]
        == 0.0
    )
    assert (
        factory.return_value.chat.completions.create.call_args.kwargs["max_tokens"] == 8
    )
    assert model_providers.get_llm_provider is model_providers.get_provider
    with pytest.raises(ValueError, match="Unsupported provider: unknown"):
        model_providers.get_provider("unknown")


def test_gemini_constructor_and_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini")
    configure: MagicMock = MagicMock()
    model: MagicMock = MagicMock()
    model.return_value.generate_content.return_value.text = "answer"
    monkeypatch.setattr(model_providers.genai, "configure", configure)
    monkeypatch.setattr(model_providers.genai, "GenerativeModel", model)
    provider = model_providers.get_provider("gemini")
    configure.assert_called_once_with(api_key="fake-gemini")
    assert (
        provider.generate("system", "user", "gemini-model", temperature=0.9) == "answer"
    )
    assert model.call_args.args == ("gemini-model",)
    assert model.call_args.kwargs["generation_config"].temperature == 0
    model.return_value.generate_content.assert_called_once_with(
        contents=["system", "user"]
    )


def test_embedding_fallback_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    sentinel: object = object()
    loader: MagicMock = MagicMock(side_effect=[ImportError("missing"), sentinel])
    monkeypatch.setattr(model_providers, "get_embedding_model", loader)
    assert model_providers.get_embedding_method() == ("openai", sentinel)
    assert [call.args for call in loader.call_args_list] == [
        ("huggingface",),
        ("openai",),
    ]
    loader.side_effect = [ValueError("first"), ValueError("second")]
    with pytest.raises(
        RuntimeError,
        match="Failed to initialize any embedding model. OpenAI error: second",
    ):
        model_providers.get_embedding_method("custom")
