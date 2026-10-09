"""Provider transport contracts: constructors are real; clients are explicit fakes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import google.generativeai as genai
import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from ontographrag.providers import model_providers as providers


@pytest.mark.parametrize(
    "name",
    ["openai", "gemini", "huggingface", "ollama", "deepseek", "openrouter"],
)
def test_real_provider_factories(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers, "OpenAIClient", Mock())
    monkeypatch.setattr(genai, "configure", Mock())
    provider: providers.ModelProvider = providers.get_provider(name.upper())
    assert isinstance(provider, providers.ModelProvider)
    with pytest.raises(ValueError, match="Unsupported"):
        providers.get_provider("unsupported")


@pytest.mark.parametrize(
    "name,key,url",
    [
        (
            "deepseek",
            "DEEPSEEK_API_KEY",
            "https://api.deepseek.com/v1/chat/completions",
        ),
        (
            "openrouter",
            "OPENROUTER_API_KEY",
            "https://openrouter.ai/api/v1/chat/completions",
        ),
    ],
)
def test_http_provider_payload_and_http_failure(
    name: str,
    key: str,
    url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(key, "unit-token")
    response: Mock = Mock()
    response.json.return_value = {"choices": [{"message": {"content": "answer"}}]}
    post: Mock = Mock(return_value=response)
    monkeypatch.setattr(httpx, "post", post)
    provider: providers.ModelProvider = providers.get_provider(name)
    assert (
        provider.generate("system", "question", "unit-model", temperature=0.7)
        == "answer"
    )
    assert post.call_args.args == (url,)
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer unit-token"
    assert post.call_args.kwargs["json"] == {
        "model": "unit-model",
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ],
    }
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "denied",
        request=httpx.Request("POST", url),
        response=httpx.Response(401),
    )
    with pytest.raises(httpx.HTTPStatusError):
        provider.generate("s", "q", "m")


def test_openai_constructor_and_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    client: Mock = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
    )
    factory: Mock = Mock(return_value=client)
    monkeypatch.setattr(providers, "OpenAIClient", factory)
    monkeypatch.setenv("OPENAI_API_KEY", "unit")
    provider: providers.OpenAIProvider = providers.OpenAIProvider()
    assert factory.call_args.kwargs["api_key"] == "unit"
    assert provider.generate("s", "u", "m") == "answer"
    assert client.chat.completions.create.call_args.kwargs["temperature"] == 1


def test_huggingface_and_gemini_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    client: Mock = Mock()
    client.text_generation.return_value = "answer"
    monkeypatch.setattr(providers, "InferenceClient", Mock(return_value=client))
    assert providers.HuggingFaceProvider().generate("s", "u", "m") == "answer"
    client.text_generation.assert_called_once_with(
        prompt="s\n\nu",
        model="m",
        temperature=0,
    )
    model: Mock = Mock()
    model.generate_content.return_value = SimpleNamespace(text="gemini answer")
    monkeypatch.setattr(genai, "configure", Mock())
    monkeypatch.setattr(genai, "GenerativeModel", Mock(return_value=model))
    assert (
        providers.GeminiProvider().generate("s", "u", "m", temperature=0.9)
        == "gemini answer"
    )
    model.generate_content.assert_called_once_with(contents=["s", "u"])


@pytest.mark.parametrize(
    "content,force,expected",
    [
        ("answer", False, "answer"),
        ('{"ok":true}', True, '{"ok":true}'),
        ("bad", True, "Ollama error: Invalid JSON response from model: bad"),
    ],
)
def test_ollama_json_contract(
    monkeypatch: pytest.MonkeyPatch,
    content: str,
    force: bool,
    expected: str,
) -> None:
    import ollama

    chat: Mock = Mock(return_value={"message": {"content": content}})
    monkeypatch.setattr(ollama, "chat", chat)
    assert (
        providers.OllamaProvider().generate("s", "u", "m", force_json=force) == expected
    )
    assert ("format" in chat.call_args.kwargs) == force
    chat.side_effect = RuntimeError("model not found")
    assert "ollama pull m" in providers.OllamaProvider().generate("s", "u", "m")


@pytest.mark.parametrize(
    "input_value,system,user",
    [
        ("hello", "", "hello"),
        ({"chunk_text": "chunk"}, "", "chunk"),
        (
            {"messages": [SystemMessage(content="s"), HumanMessage(content="q")]},
            "s",
            "q",
        ),
        (SimpleNamespace(content="content"), "", "content"),
        (42, "", "42"),
    ],
)
def test_adapter_invocation(input_value: object, system: str, user: str) -> None:
    provider: Mock = Mock()
    provider.generate.return_value = "answer"
    adapter: providers.LangChainRunnableAdapter = providers.LangChainRunnableAdapter(
        provider,
        "m",
        temperature=0.3,
    )
    assert adapter.invoke(input_value) == "answer"
    provider.generate.assert_called_once_with(system, user, "m", temperature=0.3)
    assert adapter.bind(x=1) is adapter
    assert adapter.with_structured_output(dict) is adapter


def test_embedding_fallback_and_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    model: object = object()
    factory: Mock = Mock(side_effect=[ImportError("unavailable"), model])
    monkeypatch.setattr(providers, "get_embedding_model", factory)
    assert providers.get_embedding_method("huggingface") == ("openai", model)
    assert [c.args[0] for c in factory.call_args_list] == ["huggingface", "openai"]
    factory.side_effect = RuntimeError("offline")
    with pytest.raises(RuntimeError, match="Failed to initialize any"):
        providers.get_embedding_method("huggingface")


def test_adapter_preserves_multimodal_content_and_rejects_extra_kwargs() -> None:
    system: providers.MessageContent = [{"type": "text", "text": "rules"}]
    user: providers.MessageContent = [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}},
    ]
    provider: Mock = Mock()
    provider.generate.return_value = "answer"
    adapter: providers.LangChainRunnableAdapter = providers.LangChainRunnableAdapter(
        provider,
        "model",
    )
    assert (
        adapter.invoke(
            {"messages": [SystemMessage(content=system), HumanMessage(content=user)]},
        )
        == "answer"
    )
    assert provider.generate.call_args.args[:2] == (system, user)
    assert adapter.invoke({"input": user}) == "answer"
    assert provider.generate.call_args.args[1] is user
    invoke: Mock = Mock(wraps=adapter.invoke)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        invoke("input", unsupported=True)
    assert adapter.bind(response_format={"type": "json_object"}) is adapter
    assert adapter.with_structured_output(dict, include_raw=True) is adapter


def test_huggingface_preserves_stream_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks: object = iter(["first", "second"])
    client: Mock = Mock()
    client.text_generation.return_value = chunks
    monkeypatch.setattr(providers, "InferenceClient", Mock(return_value=client))
    assert (
        providers.HuggingFaceProvider().generate("s", "u", "m", stream=True) is chunks
    )
    client.text_generation.assert_called_once_with(
        prompt="s\n\nu",
        model="m",
        stream=True,
        temperature=0.0,
    )


def test_openai_preserves_multimodal_payload_and_stream_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client: Mock = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
    )
    monkeypatch.setattr(providers, "OpenAIClient", Mock(return_value=client))
    provider: providers.OpenAIProvider = providers.OpenAIProvider()
    content: providers.MessageContent = [{"type": "text", "text": "describe"}]
    assert provider.generate("s", content, "m", max_tokens=4) == "answer"
    assert (
        client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        is content
    )
    client.chat.completions.create.return_value = iter(["chunk"])
    with pytest.raises(AttributeError, match="choices"):
        provider.generate("s", "u", "m", stream=True)


def test_ollama_preserves_stream_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import ollama

    chat: Mock = Mock(return_value=iter([{"message": {"content": "chunk"}}]))
    monkeypatch.setattr(ollama, "chat", chat)
    assert "not subscriptable" in providers.OllamaProvider().generate(
        "s",
        "u",
        "m",
        stream=True,
    )
    assert chat.call_args.kwargs["stream"] is True


def test_temperature_lock_preserves_extension_identity() -> None:
    provider: Mock = Mock()
    extension: object = object()
    provider.extension = extension
    provider.generate.return_value = "answer"
    locked: providers.TemperatureLockedProvider = providers.TemperatureLockedProvider(
        provider,
        temperature=0.2,
    )
    assert locked.extension is extension
    assert locked.generate("s", "u", "m", temperature=0.9, custom=extension) == "answer"
    provider.generate.assert_called_once_with(
        "s",
        "u",
        "m",
        temperature=0.2,
        custom=extension,
    )


@pytest.mark.parametrize("name", ["openai", "gemini", "huggingface"])
def test_real_sdk_rejects_unsupported_provider_options(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unit-fake-key")
    monkeypatch.setattr(genai, "configure", Mock())
    provider: providers.ModelProvider = providers.get_provider(name)
    # Deliberately exercise the permissive Python boundary with an invalid key.
    generate: Mock = Mock(wraps=provider.generate)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        generate("system", "user", "model", unsupported_option=True)


def test_real_ollama_rejection_preserves_error_translation() -> None:
    generate: Mock = Mock(wraps=providers.OllamaProvider().generate)
    result: str = generate("system", "user", "model", unsupported_option=True)
    assert result.startswith("Ollama error:")
    assert "unexpected keyword argument" in result
    assert "unsupported_option" in result


def test_gemini_discards_opaque_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    model: Mock = Mock()
    model.generate_content.return_value = SimpleNamespace(text="answer")
    monkeypatch.setattr(genai, "configure", Mock())
    monkeypatch.setattr(genai, "GenerativeModel", Mock(return_value=model))
    temperature: object = object()
    assert (
        providers.GeminiProvider().generate("s", "u", "m", temperature=temperature)
        == "answer"
    )
    model.generate_content.assert_called_once_with(contents=["s", "u"])


def test_ollama_force_json_retains_truthy_nonboolean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ollama

    chat: Mock = Mock(return_value={"message": {"content": "{}"}})
    monkeypatch.setattr(ollama, "chat", chat)
    assert providers.OllamaProvider().generate("s", "u", "m", force_json="yes") == "{}"
    assert chat.call_args.kwargs["format"] == "json"
