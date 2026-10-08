import sys
import types

import pytest

from ontographrag.kg.chunking import TextChunk, chunk_text


def test_chunk_text_falls_back_when_tiktoken_splitter_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSplitter:
        def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
            self.chunk_size: int = chunk_size
            self.chunk_overlap: int = chunk_overlap

        @classmethod
        def from_tiktoken_encoder(cls, chunk_size: int, chunk_overlap: int) -> None:
            raise RuntimeError("offline tokenizer assets unavailable")

        def split_text(self, text: str) -> list[str]:
            return [text[:5], text[5:]]

    class FakeModule(types.ModuleType):
        RecursiveCharacterTextSplitter: type[FakeSplitter] = FakeSplitter

    fake_module: FakeModule = FakeModule("langchain_text_splitters")
    monkeypatch.setitem(sys.modules, "langchain_text_splitters", fake_module)

    chunks: list[TextChunk] = chunk_text(
        "abcdefghij",
        chunk_size=5,
        chunk_overlap=0,
        embedding_fn=None,
    )

    assert [chunk["text"] for chunk in chunks] == ["abcde", "fghij"]
    assert chunks[0]["start_pos"] == 0
    assert chunks[1]["start_pos"] == 5
