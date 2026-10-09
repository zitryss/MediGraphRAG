import sys
from types import ModuleType
from typing import get_type_hints

import pytest
from langchain_core.embeddings import Embeddings

from shared.common_fn import load_embedding_model


def test_shared_embedding_loader_public_return_annotation_resolves() -> None:
    annotations: dict[str, object] = get_type_hints(load_embedding_model)

    assert annotations["embedding_model_name"] == str | None
    assert annotations["return"] == tuple[Embeddings, int]


class _Embedding(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return []


class _LoaderModule(ModuleType):
    def __init__(self) -> None:
        super().__init__("ontographrag.kg.utils.common_functions")
        self.calls: list[str | None] = []
        self.embedding: Embeddings = _Embedding()

    def load_embedding_model(self, name: str | None) -> tuple[Embeddings, int]:
        self.calls.append(name)
        return self.embedding, 3


@pytest.mark.parametrize(
    "name,environment,expected",
    [
        ("explicit", None, "explicit"),
        ("explicit", "configured", "explicit"),
        (None, None, "huggingface"),
        ("", None, "huggingface"),
        (None, "configured", "configured"),
        ("", "configured", "configured"),
        (None, "", ""),
        ("", "", ""),
    ],
)
def test_embedding_selection_preserves_falsey_defaults_and_loader_identity(
    monkeypatch: pytest.MonkeyPatch,
    name: str | None,
    environment: str | None,
    expected: str,
) -> None:
    loader: _LoaderModule = _LoaderModule()
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    if environment is None:
        monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("EMBEDDING_PROVIDER", environment)

    embedding, dimension = load_embedding_model(name)

    assert loader.calls == [expected]
    assert embedding is loader.embedding
    assert dimension == 3
