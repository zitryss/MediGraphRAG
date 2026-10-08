"""Lexical coverage must not confuse neighbouring or nested binding scopes."""

from datetime import date

import pytest

from scripts.check_typing_coverage import (
    Binding,
    ExceptionEntry,
    inspect_source,
    parse_entries,
    reconcile,
)


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


@pytest.mark.parametrize(
    "source,scope,category",
    [
        (
            "x: int = 1\ndef f() -> None:\n    x: str = ''\n    def g() -> None:\n        global x\n        x = 2\n",
            "<module>.f.g",
            "explicit",
        ),
        (
            "x = 1\ndef f() -> None:\n    x: str = ''\n    def g() -> None:\n        global x\n        x = 2\n",
            "<module>.f.g",
            "missing",
        ),
        (
            "def f() -> None:\n    x: int = 1\n    def g() -> None:\n        nonlocal x\n        x = 2\n",
            "<module>.f.g",
            "explicit",
        ),
        (
            "x: int\nvalues: list[int] = [(x := i) for i in range(3)]\n",
            "<module>",
            "explicit",
        ),
    ],
)
def test_external_and_walrus_owners(source: str, scope: str, category: str) -> None:
    bindings: list[Binding] = inspect_source(source, "fixture.py")
    assert any(
        binding.scope == scope
        and binding.symbol == "x"
        and binding.category == category
        for binding in bindings
    )


@pytest.mark.parametrize(
    "source,kind",
    [
        ("from typing import Any as Dynamic\nx: Dynamic = 1\n", "dynamic-type"),
        ("import typing as t\nx: t.Any = 1\n", "dynamic-type"),
        ("from typing import cast as convert\nx: int = convert(int, 1)\n", "cast"),
        ("x: int = 1  # type: ignore[assignment]\n", "suppression"),
        ("# pyright: reportUnknownVariableType=false\nx = 1\n", "suppression"),
    ],
)
def test_dynamic_escape_sites_are_not_silently_accepted(source: str, kind: str) -> None:
    bindings: list[Binding] = inspect_source(source, "fixture.py")
    assert any(
        binding.kind == kind and binding.category == "dynamic" for binding in bindings
    )


def test_pattern_captures_are_syntax_constrained() -> None:
    source: str = "def f(value: list[int]) -> None:\n    match value:\n        case [first, *rest]:\n            pass\n"
    bindings: list[Binding] = inspect_source(source, "fixture.py")
    assert {binding.symbol for binding in bindings if binding.kind == "pattern"} == {
        "first",
        "rest",
    }


def test_default_walrus_belongs_to_enclosing_scope() -> None:
    bindings: list[Binding] = inspect_source(
        "def f(value: int = (default := 1)) -> None:\n    pass\n", "fixture.py"
    )
    assert any(
        binding.symbol == "default" and binding.scope == "<module>"
        for binding in bindings
    )


def test_fingerprint_tracks_changed_assignment_contract() -> None:
    first: list[Binding] = inspect_source("x: int = 1\n", "fixture.py")
    changed: list[Binding] = inspect_source("x: str = '1'\n", "fixture.py")
    assert first[0].fingerprint != changed[0].fingerprint


@pytest.mark.parametrize(
    "change", ["pending", "expired", "fingerprint", "unused", "duplicate"]
)
def test_exception_ledger_fails_closed(change: str) -> None:
    bindings: list[Binding] = inspect_source("x = 1\n", "fixture.py")
    raw: dict[str, str] = {
        "identity": "fixture-boundary",
        "path": "fixture.py",
        "scope": "<module>",
        "symbol": "x",
        "kind": "assignment",
        "fingerprint": bindings[0].fingerprint,
        "reason": "Runtime-sensitive fixture",
        "evidence": "fixture contract",
        "owner": "engineer",
        "reviewer": "fixture reviewer",
        "disposition": "approved",
        "expires": "2099-01-01",
    }
    if change == "pending":
        raw["disposition"] = "pending"
    elif change == "expired":
        raw["expires"] = "2000-01-01"
    elif change == "fingerprint":
        raw["fingerprint"] = "0" * 64
    elif change == "unused":
        raw["symbol"] = "other"
    entries: list[ExceptionEntry] = parse_entries([raw])
    if change == "duplicate":
        entries.append(entries[0])
    assert reconcile(bindings, entries, date(2026, 10, 8))


def test_approved_exact_exception_is_identified() -> None:
    bindings: list[Binding] = inspect_source("x = 1\n", "fixture.py")
    entry: ExceptionEntry = ExceptionEntry(
        "fixture",
        "fixture.py",
        "<module>",
        "x",
        "assignment",
        bindings[0].fingerprint,
        "fixture reason",
        "fixture evidence",
        "engineer",
        "fixture reviewer",
        "approved",
        date(2099, 1, 1),
    )
    assert not reconcile(bindings, [entry], date(2026, 10, 8))
    assert bindings[0].exception_id == "fixture"


@pytest.mark.parametrize("value", [None, {}, [{}], [{"identity": "incomplete"}]])
def test_malformed_exception_ledger_rejected(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_entries(value)


def test_type_variable_is_a_definition_not_an_untyped_value() -> None:
    bindings: list[Binding] = inspect_source(
        'from typing import TypeVar\nT = TypeVar("T")\n', "fixture.py"
    )
    assert any(
        binding.symbol == "T" and binding.kind == "type-definition"
        for binding in bindings
    )
    assert not any(binding.category == "missing" for binding in bindings)


def test_nonlocal_parameter_retains_its_declared_type() -> None:
    bindings: list[Binding] = inspect_source(
        "def outer(x: int) -> None:\n    def inner() -> None:\n        nonlocal x\n        x = 2\n",
        "fixture.py",
    )
    assert any(
        binding.scope == "<module>.outer.inner"
        and binding.symbol == "x"
        and binding.category == "explicit"
        and binding.declaration == "int"
        for binding in bindings
    )


def test_compact_ignore_directive_is_detected() -> None:
    bindings: list[Binding] = inspect_source("x: int = 1 # type:ignore\n", "fixture.py")
    assert any(binding.kind == "suppression" for binding in bindings)


@pytest.mark.parametrize(
    "source",
    [
        "from typing import Any\ndef f(Any: Any) -> None: pass\n",
        'from typing import Any as A\nx: "list[A]" = []\n',
        'import typing as t\nx: list["t.Any"] = []\n',
        'from typing import Any\ndef f(x: "Any") -> "Any": pass\n',
    ],
)
def test_signature_and_forward_reference_any_is_detected(source: str) -> None:
    bindings: list[Binding] = inspect_source(source, "fixture.py")
    assert any(binding.kind == "dynamic-type" for binding in bindings)


def test_annotation_value_strings_are_not_forward_type_references() -> None:
    source: str = (
        "from typing import Annotated, Literal\n"
        'x: Literal["Any"] = "Any"\n'
        'y: Annotated[str, "Any"] = "value"\n'
    )
    bindings: list[Binding] = inspect_source(source, "fixture.py")
    assert not any(binding.category == "dynamic" for binding in bindings)
