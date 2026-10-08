"""Lexical coverage must not confuse neighbouring or nested binding scopes."""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.check_typing_coverage import (
    Binding,
    ExceptionEntry,
    discover,
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


def test_functional_typed_dict_is_a_definition_and_checks_forward_field_types() -> None:
    source: str = (
        "from typing import Any, TypedDict as Record\n"
        'Edge = Record("Edge", {"from": str, "to": "Any"})\n'
    )
    bindings: list[Binding] = inspect_source(source, "fixture.py")
    assert any(
        binding.symbol == "Edge" and binding.kind == "type-definition"
        for binding in bindings
    )
    assert sum(binding.kind == "dynamic-type" for binding in bindings) == 1
    assert not any(binding.category == "missing" for binding in bindings)


@pytest.mark.parametrize(
    "source",
    [
        'from typing import Any, TypeAlias\nAlias: TypeAlias = "list[Any]"\n',
        'from typing import Any, TypeVar\nT = TypeVar("T", bound="Any")\n',
        'from typing import Any, TypeVar\nT = TypeVar("T", "Any", int)\n',
        'from typing import Any, NewType\nT = NewType("T", "Any")\n',
        'from typing import Any, cast\nx: int = cast("Any", 1)\n',
    ],
)
def test_forward_types_in_definitions_and_casts_are_audited(source: str) -> None:
    bindings: list[Binding] = inspect_source(source, "fixture.py")
    assert any(binding.kind == "dynamic-type" for binding in bindings)


def test_lambda_default_executes_in_enclosing_scope() -> None:
    bindings: list[Binding] = inspect_source(
        "f = lambda value=(default := 1): value\n",
        "fixture.py",
    )
    assert any(
        binding.symbol == "default" and binding.scope == "<module>"
        for binding in bindings
    )


def test_lambda_parameter_shadows_imported_any() -> None:
    bindings: list[Binding] = inspect_source(
        "from typing import Any\nf = lambda Any: Any\n",
        "fixture.py",
    )
    assert not any(binding.kind == "dynamic-type" for binding in bindings)


def test_enum_members_are_distinct_from_ordinary_state() -> None:
    bindings: list[Binding] = inspect_source(
        "from enum import Enum as Base\nclass Color(Base):\n    RED = 1\n",
        "fixture.py",
    )
    assert any(
        binding.symbol == "RED" and binding.category == "enum-member"
        for binding in bindings
    )


def test_augmented_external_assignment_cannot_hide_missing_annotation() -> None:
    bindings: list[Binding] = inspect_source(
        "x = 0\ndef f() -> None:\n    global x\n    x += 1\n",
        "fixture.py",
    )
    assert any(
        binding.symbol == "x"
        and binding.scope == "<module>.f"
        and binding.category == "missing"
        for binding in bindings
    )


@pytest.mark.parametrize(
    "source",
    [
        'setattr(target, "field", 1)\n',
        'monkeypatch.setattr(target, "field", 1)\n',
    ],
)
def test_dynamic_attribute_updates_require_review(source: str) -> None:
    bindings: list[Binding] = inspect_source(source, "fixture.py")
    assert any(binding.kind == "dynamic-attribute" for binding in bindings)


def test_signature_dynamic_site_has_function_scope() -> None:
    bindings: list[Binding] = inspect_source(
        "from typing import Any\ndef f(value: Any) -> None: pass\n",
        "fixture.py",
    )
    assert any(
        binding.kind == "dynamic-type" and binding.scope == "<module>.f"
        for binding in bindings
    )


def test_declared_exception_target_is_explicit() -> None:
    bindings: list[Binding] = inspect_source(
        "def f() -> None:\n    error: ValueError\n    try:\n        pass\n    except ValueError as error:\n        pass\n",
        "fixture.py",
    )
    assert any(
        binding.symbol == "error" and binding.category == "explicit"
        for binding in bindings
    )
    assert not any(
        binding.symbol == "error" and binding.category == "inferred"
        for binding in bindings
    )


def test_definitions_have_their_own_binding_category() -> None:
    bindings: list[Binding] = inspect_source(
        "class C:\n    def f(self) -> None: pass\n",
        "fixture.py",
    )
    assert {
        binding.symbol for binding in bindings if binding.category == "definition"
    } == {"C", "f"}


@pytest.mark.parametrize("tracked", [b"outside.py\0", b"scripts/missing.py\0"])
def test_discovery_rejects_unclassified_or_deleted_tracked_files(
    tmp_path: Path,
    tracked: bytes,
) -> None:
    with patch(
        "scripts.check_typing_coverage.subprocess.check_output", return_value=tracked
    ):
        with pytest.raises(ValueError):
            discover(tmp_path)


def test_discovery_includes_new_source_and_stubs(tmp_path: Path) -> None:
    source: Path = tmp_path / "scripts" / "new.py"
    stub: Path = tmp_path / "shared" / "new.pyi"
    source.parent.mkdir()
    stub.parent.mkdir()
    source.write_text("value: int = 1\n")
    stub.write_text("value: int\n")
    with patch(
        "scripts.check_typing_coverage.subprocess.check_output", return_value=b""
    ):
        assert discover(tmp_path) == sorted([source, stub])


def test_discovery_rejects_empty_scope(tmp_path: Path) -> None:
    with patch(
        "scripts.check_typing_coverage.subprocess.check_output", return_value=b""
    ):
        with pytest.raises(ValueError, match="No maintained Python files"):
            discover(tmp_path)


def test_parse_failure_cannot_produce_a_clean_inventory() -> None:
    with pytest.raises(SyntaxError):
        inspect_source("def incomplete(\n", "fixture.py")


def test_foreign_attribute_mutation_defers_to_owning_type_checker() -> None:
    bindings: list[Binding] = inspect_source(
        "class State:\n    value: int\ndef f(state: State) -> None:\n    state.value = 1\n",
        "fixture.py",
    )
    assert any(
        binding.symbol == "state.value"
        and binding.kind == "attribute-mutation"
        and binding.category == "inferred"
        for binding in bindings
    )


@pytest.mark.parametrize(
    "source",
    [
        "from builtins import staticmethod as sm\nclass C:\n    @sm\n    def f(self) -> None: pass\n",
        "class C:\n    def f(*, self) -> None: pass\n",
        "class C:\n    def f(*self) -> None: pass\n",
    ],
)
def test_only_a_normal_positional_receiver_is_exempt(source: str) -> None:
    bindings: list[Binding] = inspect_source(source, "fixture.py")
    assert any(
        binding.symbol == "self" and binding.category == "missing"
        for binding in bindings
    )
