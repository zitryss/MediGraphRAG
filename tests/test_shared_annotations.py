from typing import get_type_hints

from langchain_core.embeddings import Embeddings

from shared.common_fn import load_embedding_model


def test_shared_embedding_loader_public_return_annotation_resolves() -> None:
    annotations: dict[str, object] = get_type_hints(load_embedding_model)

    assert annotations["embedding_model_name"] == str | None
    assert annotations["return"] == tuple[Embeddings, int]
