"""Model-boundary fakes with real tensor scoring, masking and cache behavior."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import sentence_transformers
import torch
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer

from ontographrag.rag import reranking as ranking
from ontographrag.rag.reranking import LateInteractionIndex
from ontographrag.rag.systems._records import RetrievedChunk

if TYPE_CHECKING:
    from torch import Tensor

    from ontographrag.rag.reranking import LateInteractionBundle


@dataclass
class TensorOutput:
    last_hidden_state: torch.Tensor


@pytest.fixture
def clean_model_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("_RERANKER_CACHE", "_LATE_INTERACTION_CACHE"):
        monkeypatch.setattr(ranking, name, {})
    for name in ("_RERANKER_FAILURES", "_LATE_INTERACTION_FAILURES"):
        monkeypatch.setattr(
            ranking,
            name,
            set[tuple[str, str, int] | tuple[str, str, int, int]](),
        )
    monkeypatch.setattr(
        ranking,
        "_LATE_INTERACTION_INDEX_CACHE",
        OrderedDict[tuple[str, str, int, str, str], LateInteractionIndex](),
    )


@pytest.mark.usefixtures("clean_model_caches")
def test_model_loaders_cache_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cross: Mock = Mock(return_value=Mock())
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", cross)
    assert ranking._load_cross_encoder("unit") is ranking._load_cross_encoder("unit")
    cross.assert_called_once()
    cross.side_effect = RuntimeError("local weights absent")
    assert ranking._load_cross_encoder("missing") is None
    assert ranking._load_cross_encoder("missing") is None
    assert cross.call_count == 2
    tokenizer, model = Mock(), Mock()
    tok_factory, model_factory = Mock(return_value=tokenizer), Mock(return_value=model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tok_factory)
    monkeypatch.setattr(AutoModel, "from_pretrained", model_factory)
    bundle: LateInteractionBundle | None = ranking._load_late_interaction_model("unit")
    assert bundle is not None and bundle["model"] is model
    assert ranking._load_late_interaction_model("unit") is bundle
    model.eval.assert_called_once()
    tok_factory.side_effect = RuntimeError("local tokenizer absent")
    assert ranking._load_late_interaction_model("missing") is None
    assert ranking._load_late_interaction_model("missing") is None
    assert tok_factory.call_count == 2


@pytest.mark.usefixtures("clean_model_caches")
@pytest.mark.parametrize("special_mode", ["normal", "absent", "all"])
def test_token_maxsim_scoring_and_cached_index_agree(
    monkeypatch: pytest.MonkeyPatch,
    special_mode: str,
) -> None:
    def tokenize(texts: str | list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        values: list[str] = [texts] if isinstance(texts, str) else texts
        ids: Tensor = torch.tensor(
            [[1, 1] if "relevant" in text else [2, 2] for text in values],
        )
        result: dict[str, Tensor] = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        }
        if special_mode != "absent":
            result["special_tokens_mask"] = (
                torch.ones_like(ids) if special_mode == "all" else torch.zeros_like(ids)
            )
        return result

    def forward(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> TensorOutput:
        hidden: Tensor = torch.nn.functional.one_hot(
            input_ids - 1,
            num_classes=2,
        ).float()
        return TensorOutput(last_hidden_state=hidden)

    model: Mock = Mock(side_effect=forward)
    bundle: LateInteractionBundle = {
        "tokenizer": tokenize,
        "model": model,
        "model_name": "unit",
        "device": "cpu",
        "query_max_length": 8,
        "doc_max_length": 8,
        "torch": torch,
    }
    chunks: list[RetrievedChunk] = [
        {"chunk_id": "a", "text": "other"},
        {"chunk_id": "b", "text": "relevant", "document": "doc"},
    ]
    assert ranking._late_interaction_scores(
        bundle,
        "relevant",
        chunks,
    ) == pytest.approx([0, 1])
    first, meta = ranking._late_interaction_scores_indexed(
        bundle,
        "relevant",
        chunks,
        index_key="unit",
    )
    assert first == [(1, 1.0), (0, 0.0)]
    assert not meta["index_cached"]
    count: int = model.call_count
    second, meta = ranking._late_interaction_scores_indexed(
        bundle,
        "relevant",
        chunks,
        index_key="unit",
    )
    assert second == first and meta["index_cached"]
    assert model.call_count == count + 1  # query re-encoded; documents reused
    chunks[0]["text"] = "relevant changed"
    updated, cached = ranking._build_late_interaction_index(
        bundle,
        chunks,
        index_key="unit",
    )
    assert updated is not None
    # Cache identity includes text, preventing stale embeddings under same corpus key.
    assert not cached


@pytest.mark.parametrize("kind", ["cross", "late"])
def test_reranking_failed_model_preserves_inputs(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    monkeypatch.setenv("ONTOGRAPHRAG_RERANKER", "1")
    monkeypatch.setenv("ONTOGRAPHRAG_RERANKER_MODEL", "unit")
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION", "1")
    monkeypatch.setenv("ONTOGRAPHRAG_LATE_INTERACTION_MODEL", "unit")
    chunks: list[RetrievedChunk] = [{"text": "evidence", "score": 0.4}]
    if kind == "cross":
        monkeypatch.setattr(
            ranking,
            "_load_cross_encoder",
            Mock(
                return_value=Mock(predict=Mock(side_effect=RuntimeError("bad tensor"))),
            ),
        )
        result, meta = ranking.rerank_chunks_for_query("q", chunks)
    else:
        monkeypatch.setattr(
            ranking,
            "_load_late_interaction_model",
            Mock(return_value={}),
        )
        result, meta = ranking.late_interaction_rescore_chunks_for_query("q", chunks)
    assert result == chunks and result is not chunks
    assert "reason" in meta
    assert meta["reason"] == "scoring_failed"
