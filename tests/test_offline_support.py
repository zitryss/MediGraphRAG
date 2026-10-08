"""The manifest fixture preserves forwarding and failures outside its one override."""

import io
from pathlib import Path
from typing import Protocol

import pytest

from tests.offline_support import manifest_opener


class BinaryOpener(Protocol):
    def __call__(self, path: Path, /, mode: str = "r") -> io.BytesIO | io.StringIO: ...


class TextOpener(Protocol):
    def __call__(
        self,
        path: str,
        /,
        *,
        encoding: str | None = None,
    ) -> io.StringIO: ...


def testmanifest_opener_intercepts_only_the_manifest_suffix() -> None:
    calls: list[tuple[Path, str]] = []
    original: io.BytesIO = io.BytesIO(b"source")

    def opener(path: Path, mode: str = "r") -> io.BytesIO:
        calls.append((path, mode))
        return original

    replacement: BinaryOpener = manifest_opener(opener)
    manifest: io.BytesIO | io.StringIO = replacement(
        Path("latest_results_manifest.json"),
        "rb",
    )
    assert manifest.read() == "{}"
    assert calls == []
    source: Path = Path("other.json")
    assert replacement(source, "rb") is original
    assert calls == [(source, "rb")]


def testmanifest_opener_preserves_keywords_and_exception_identity() -> None:
    failure: OSError = OSError("source unavailable")
    calls: list[tuple[str, str | None]] = []

    def opener(path: str, *, encoding: str | None = None) -> io.StringIO:
        calls.append((path, encoding))
        raise failure

    replacement: TextOpener = manifest_opener(opener)
    with pytest.raises(OSError) as caught:
        replacement("other.json", encoding="utf-16")
    assert caught.value is failure
    assert calls == [("other.json", "utf-16")]
