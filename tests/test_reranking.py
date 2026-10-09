import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, TypedDict

import pytest

from ontographrag.rag.systems._records import RetrievedChunk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ontographrag.rag import reranking as rerank_mod
from ontographrag.rag.reranking import (
    EncodedQuery,
    IndexedScoringMetadata,
    LateInteractionIndex,
)

if TYPE_CHECKING:
    from torch import Tensor

    from ontographrag.rag.reranking import LateInteractionBundle, TensorBackend


@dataclass
class TensorOutput:
    last_hidden_state: "Tensor"


class TensorOnlyBundle(TypedDict):
    torch: "TensorBackend"


def test_resolve_reranker_model_name_uses_accuracy_profile_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ONTOGRAPHRAG_RERANKER_MODEL", raising=False)
    monkeypatch.setenv("ONTOGRAPHRAG_RETRIEVAL_PROFILE", "accuracy")

    assert rerank_mod.resolve_reranker_model_name() == "BAAI/bge-reranker-base"


def test_rerank_chunks_for_query_sorts_by_model_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def predict(
            self,
            pairs: list[tuple[str, str]],
            show_progress_bar: bool = False,
        ) -> list[float]:
            return [0.1, 0.9]

    monkeypatch.setenv("ONTOGRAPHRAG_RERANKER", "1")
    monkeypatch.setenv("ONTOGRAPHRAG_RERANKER_MODEL", "fake-reranker")

    def fake_load_cross_encoder(model_name: "object") -> "FakeModel":
        return FakeModel()

    monkeypatch.setattr(rerank_mod, "_load_cross_encoder", fake_load_cross_encoder)

    chunks: list[RetrievedChunk] = [
        {"text": "Generic background.", "chunk_id": "c1", "score": 0.9},
        {"text": "Direct answer evidence.", "chunk_id": "c2", "score": 0.5},
    ]

    reranked, meta = rerank_mod.rerank_chunks_for_query("Question?", chunks)

    assert reranked[0]["chunk_id"] == "c2"
    assert reranked[1]["chunk_id"] == "c1"
    assert meta["applied"] is True
    assert meta["model"] == "fake-reranker"


def test_rerank_chunks_for_query_fails_soft_when_model_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONTOGRAPHRAG_RERANKER", "1")
    monkeypatch.setenv("ONTOGRAPHRAG_RERANKER_MODEL", "fake-reranker")

    def fake_load_cross_encoder(model_name: "object") -> "None":
        return None

    monkeypatch.setattr(rerank_mod, "_load_cross_encoder", fake_load_cross_encoder)

    chunks: list[RetrievedChunk] = [
        {"text": "First.", "chunk_id": "c1", "score": 0.9},
        {"text": "Second.", "chunk_id": "c2", "score": 0.8},
    ]

    reranked, meta = rerank_mod.rerank_chunks_for_query("Question?", chunks)

    assert [chunk["chunk_id"] for chunk in reranked] == ["c1", "c2"]
    assert meta["applied"] is False
    assert meta["reason"] == "model_unavailable"


def test_resolve_late_interaction_model_name_uses_accuracy_profile_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ONTOGRAPHRAG_LATE_INTERACTION_MODEL", raising=False)
    monkeypatch.delenv("HUGGINGFACE_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("ONTOGRAPHRAG_RETRIEVAL_PROFILE", "accuracy")

    assert rerank_mod.resolve_late_interaction_model_name() == "BAAI/bge-base-en-v1.5"


def test_late_interaction_rescore_chunks_for_query_sorts_by_model_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION", "1")
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION_MODEL", "fake-late-interaction")

    def fake_load_late_interaction_model(model_name: "object") -> "dict[str, bool]":
        return {"fake": True}

    monkeypatch.setattr(
        rerank_mod,
        "_load_late_interaction_model",
        fake_load_late_interaction_model,
    )

    def fake_late_interaction_scores(
        bundle: "object",
        query: "object",
        chunks: "object",
    ) -> "list[float]":
        return [0.2, 0.8]

    monkeypatch.setattr(
        rerank_mod,
        "_late_interaction_scores",
        fake_late_interaction_scores,
    )

    chunks: list[RetrievedChunk] = [
        {"text": "Generic background.", "chunk_id": "c1", "score": 0.9},
        {"text": "Direct answer evidence.", "chunk_id": "c2", "score": 0.5},
    ]

    reranked, meta = rerank_mod.late_interaction_rescore_chunks_for_query(
        "Question?",
        chunks,
    )

    assert reranked[0]["chunk_id"] == "c2"
    assert reranked[1]["chunk_id"] == "c1"
    assert meta["applied"] is True
    assert meta["model"] == "fake-late-interaction"


def test_late_interaction_rescore_chunks_can_replace_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION", "1")
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION_MODEL", "fake-late-interaction")

    def fake_load_late_interaction_model(model_name: "object") -> "dict[str, bool]":
        return {"fake": True}

    monkeypatch.setattr(
        rerank_mod,
        "_load_late_interaction_model",
        fake_load_late_interaction_model,
    )

    def fake_late_interaction_scores(
        bundle: "object",
        query: "object",
        chunks: "object",
    ) -> "list[float]":
        return [0.2, 0.8]

    monkeypatch.setattr(
        rerank_mod,
        "_late_interaction_scores",
        fake_late_interaction_scores,
    )

    chunks: list[RetrievedChunk] = [
        {"text": "Generic background.", "chunk_id": "c1", "score": 0.9},
        {"text": "Direct answer evidence.", "chunk_id": "c2", "score": 0.5},
    ]

    reranked, meta = rerank_mod.late_interaction_rescore_chunks_for_query(
        "Question?",
        chunks,
        replace_score=True,
    )

    assert reranked[0]["score"] == 0.8
    assert reranked[1]["score"] == 0.2
    assert meta["replace_score"] is True


def test_late_interaction_rescore_chunks_for_query_fails_soft_when_model_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION", "1")
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION_MODEL", "fake-late-interaction")

    def fake_load_late_interaction_model(model_name: "object") -> "None":
        return None

    monkeypatch.setattr(
        rerank_mod,
        "_load_late_interaction_model",
        fake_load_late_interaction_model,
    )

    chunks: list[RetrievedChunk] = [
        {"text": "First.", "chunk_id": "c1", "score": 0.9},
        {"text": "Second.", "chunk_id": "c2", "score": 0.8},
    ]

    reranked, meta = rerank_mod.late_interaction_rescore_chunks_for_query(
        "Question?",
        chunks,
    )

    assert [chunk["chunk_id"] for chunk in reranked] == ["c1", "c2"]
    assert meta["applied"] is False
    assert meta["reason"] == "model_unavailable"


def test_late_interaction_rescore_chunks_for_query_uses_indexed_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION", "1")
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION_MODEL", "fake-late-interaction")

    def fake_load_late_interaction_model(model_name: "object") -> "dict[str, bool]":
        return {"fake": True}

    monkeypatch.setattr(
        rerank_mod,
        "_load_late_interaction_model",
        fake_load_late_interaction_model,
    )

    def fake_late_interaction_scores_indexed(
        bundle: "object",
        query: "object",
        chunks: "object",
        max_chunks: "object" = None,
        index_key: "object" = None,
        shortlist_size: "object" = None,
    ) -> "tuple[list[tuple[int, float]], IndexedScoringMetadata]":
        return (
            [(1, 0.8), (0, 0.2)],
            {
                "index_cached": True,
                "prefiltered": True,
                "shortlist_size": 2,
                "corpus_size": 12,
            },
        )

    monkeypatch.setattr(
        rerank_mod,
        "_late_interaction_scores_indexed",
        fake_late_interaction_scores_indexed,
    )

    chunks: list[RetrievedChunk] = [
        {"text": "Generic background.", "chunk_id": "c1", "score": 0.9},
        {"text": "Direct answer evidence.", "chunk_id": "c2", "score": 0.5},
    ]

    reranked, meta = rerank_mod.late_interaction_rescore_chunks_for_query(
        "Question?",
        chunks,
        max_chunks=1,
        replace_score=True,
        index_key=("scope", "q1"),
    )

    assert [chunk["chunk_id"] for chunk in reranked] == ["c2"]
    assert reranked[0]["score"] == 0.8
    assert meta["index_keyed"] is True
    assert meta["index_cached"] is True
    assert meta["prefiltered"] is True


def test_build_late_interaction_index_reuses_cached_doc_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    class FakeTokenizer:
        def __call__(
            self,
            texts: str | list[str],
            return_tensors: str = "pt",
            padding: bool = True,
            truncation: bool = True,
            max_length: int = 256,
            return_special_tokens_mask: bool = True,
        ) -> dict[str, torch.Tensor]:
            items: list[str] = [texts] if isinstance(texts, str) else list(texts)
            encoded: list[list[int]] = []
            for idx, text in enumerate(items, 1):
                token_count: int = max(2, min(4, len(str(text).split()) + 1))
                encoded.append([idx + offset for offset in range(token_count)])
            width: int = max(len(row) for row in encoded)
            input_ids: Tensor = torch.zeros((len(encoded), width), dtype=torch.long)
            attention_mask: Tensor = torch.zeros_like(input_ids)
            for row_idx, row in enumerate(encoded):
                input_ids[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long)
                attention_mask[row_idx, : len(row)] = 1
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "special_tokens_mask": torch.zeros_like(input_ids),
            }

    class FakeModel:
        def __init__(self) -> None:
            self.calls: int = 0
            self.config: SimpleNamespace = SimpleNamespace(hidden_size=2)

        def __call__(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> TensorOutput:
            self.calls += 1
            base: Tensor = input_ids.unsqueeze(-1).float()
            hidden: Tensor = torch.cat([base, base + 1.0], dim=-1)
            return TensorOutput(last_hidden_state=hidden)

    fake_model: FakeModel = FakeModel()
    bundle: LateInteractionBundle = {
        "tokenizer": FakeTokenizer(),
        "model": fake_model,
        "model_name": "fake-late-interaction",
        "device": "cpu",
        "query_max_length": 16,
        "doc_max_length": 16,
        "torch": torch,
    }
    chunks: list[RetrievedChunk] = [
        {"chunk_id": "c1", "document": "doc", "text": "first evidence"},
        {"chunk_id": "c2", "document": "doc", "text": "second evidence"},
    ]

    rerank_mod._LATE_INTERACTION_INDEX_CACHE.clear()
    index_one, cache_hit_one = rerank_mod._build_late_interaction_index(
        bundle,
        chunks,
        index_key=("scope", "q1"),
    )
    index_two, cache_hit_two = rerank_mod._build_late_interaction_index(
        bundle,
        chunks,
        index_key=("scope", "q1"),
    )

    assert cache_hit_one is False
    assert cache_hit_two is True
    assert fake_model.calls == 1
    assert index_one["count"] == 2
    assert index_two["count"] == 2


def test_late_interaction_scores_indexed_prefilters_large_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    doc_tokens: list[Tensor] = [
        torch.tensor([[0.1, 0.0]], dtype=torch.float32),
        torch.tensor([[0.9, 0.0]], dtype=torch.float32),
        torch.tensor([[0.8, 0.0]], dtype=torch.float32),
    ]
    doc_pooled: Tensor = torch.stack(
        [
            torch.tensor([0.1, 0.0], dtype=torch.float32),
            torch.tensor([0.9, 0.0], dtype=torch.float32),
            torch.tensor([0.8, 0.0], dtype=torch.float32),
        ],
    )
    bundle: TensorOnlyBundle = {"torch": torch}

    def fake_encode_late_interaction_query(
        bundle: "object",
        query: "object",
    ) -> "EncodedQuery":
        return {
            "tokens": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "pooled": torch.tensor([1.0, 0.0], dtype=torch.float32),
        }

    monkeypatch.setattr(
        rerank_mod,
        "_encode_late_interaction_query",
        fake_encode_late_interaction_query,
    )

    def fake_build_late_interaction_index(
        bundle: "object",
        chunks: "object",
        index_key: "object" = None,
    ) -> "tuple[LateInteractionIndex, bool]":
        return ({"doc_tokens": doc_tokens, "doc_pooled": doc_pooled, "count": 3}, False)

    monkeypatch.setattr(
        rerank_mod,
        "_build_late_interaction_index",
        fake_build_late_interaction_index,
    )
    monkeypatch.setattr(rerank_mod, "_late_interaction_prefilter_min_docs", lambda: 1)
    monkeypatch.setattr(rerank_mod, "_late_interaction_shortlist_cap", lambda: 2)
    monkeypatch.setattr(
        rerank_mod,
        "_late_interaction_default_shortlist_factor",
        lambda: 1,
    )

    scored, meta = rerank_mod._late_interaction_scores_indexed(
        bundle,
        "Question?",
        [{}, {}, {}],
        max_chunks=1,
        index_key=("scope", "q1"),
        shortlist_size=2,
    )

    assert [idx for idx, _score in scored] == [1, 2]
    assert meta["prefiltered"] is True
    assert meta["shortlist_size"] == 2
    assert meta["corpus_size"] == 3
