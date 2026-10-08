"""Lexical coverage must not confuse neighbouring or nested binding scopes."""

import pytest

from scripts.check_typing_coverage import Binding, inspect_source


@pytest.mark.parametrize(
    "source,symbol,kind",
    [
        ("async def f(*args, **kwargs):\n    pass\n", "args", "parameter"),
        ("def f() -> None:\n    def nested(x):\n        pass\n", "x", "parameter"),
        ("def f() -> None:\n    value = 1\n", "value", "assignment"),
        (
            "class C:\n    def f(self) -> None:\n        self.x = 1\n",
            "self.x",
            "assignment",
        ),
        ("x: int = 1\ndef f() -> None:\n    x = 2\n", "x", "assignment"),
    ],
)
def test_missing_annotations_are_reported(source: str, symbol: str, kind: str) -> None:
    bindings: list[Binding] = inspect_source(source, "fixture.py")
    assert any(
        binding.symbol == symbol
        and binding.kind == kind
        and binding.category == "missing"
        for binding in bindings
    )


def test_declaration_in_alternate_branch_covers_lexical_binding() -> None:
    source: str = "def f(flag: bool) -> None:\n    if flag:\n        x = 1\n    else:\n        x: int = 2\n    x = 3\n"
    bindings: list[Binding] = inspect_source(source, "fixture.py")
    values: list[Binding] = [binding for binding in bindings if binding.symbol == "x"]
    assert len(values) == 1
    assert values[0].category == "explicit"
    assert values[0].declaration == "int"


def test_syntax_constrained_bindings_are_separate_from_explicit_annotations() -> None:
    source: str = "def f() -> None:\n    a, b = (1, 2)\n    for i in range(2):\n        pass\n    with open('x') as stream:\n        pass\n    values: list[int] = [j for j in range(2)]\n"
    bindings: list[Binding] = inspect_source(source, "fixture.py")
    inferred: set[str] = {
        binding.symbol for binding in bindings if binding.category == "inferred"
    }
    assert inferred == {"a", "b", "i", "stream", "j"}
    assert not any(binding.category == "missing" for binding in bindings)


def test_any_is_not_explicit_precision() -> None:
    bindings: list[Binding] = inspect_source(
        "from typing import Any\nx: Any = 1\n", "fixture.py"
    )
    assert any(binding.category == "dynamic" for binding in bindings)


def test_fingerprint_ignores_line_shift_but_tracks_binding_change() -> None:
    first: list[Binding] = inspect_source("x: int = 1\n", "fixture.py")
    shifted: list[Binding] = inspect_source("\n\nx: int = 1\n", "fixture.py")
    assert first[0].fingerprint == shifted[0].fingerprint
    changed: list[Binding] = inspect_source("y: int = 1\n", "fixture.py")
    assert first[0].fingerprint != changed[0].fingerprint
