"""Consumed PyMuPDF 1.24.14 surface, re-exported by its fitz compatibility module.

Document iteration uses its integer __getitem__; no __iter__ is provided upstream.
Page and Pixmap describe returned objects, not their unconsumed constructors.
Only default text extraction and the consumed rendering option are declared.
"""

from io import BytesIO
from typing import Protocol

version: tuple[str, str, str]

class Pixmap(Protocol):
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def samples(self) -> bytes: ...

class Page(Protocol):
    def get_text(self) -> str: ...
    def get_pixmap(self, *, dpi: int | None = ...) -> Pixmap: ...

class Document:
    def __init__(
        self,
        *,
        stream: bytes | bytearray | BytesIO | None = ...,
        filetype: str | None = ...,
    ) -> None: ...
    def __getitem__(self, i: int = ...) -> Page: ...
    def close(self) -> None: ...

open: type[Document] = Document
