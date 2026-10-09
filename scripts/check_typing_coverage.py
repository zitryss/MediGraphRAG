"""Inventory maintained Python and report lexical annotation coverage.

This complements (and never replaces) strict mypy/Pyright: syntax-constrained
bindings are reported as inferred, not certified as concretely typed here.

Dynamic-read detection is a conservative lexical inventory: direct ``getattr``,
import-resolved builtins calls, and references acquiring those callables require
site proof. Shadowed names can produce reviewable false positives. This does not
resolve arbitrary Python execution, computed imports, or every alias invocation;
approval of a callable reference must cover its producer and downstream uses.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import subprocess
import tokenize
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal, TypeAlias, TypeGuard

ROOTS: tuple[str, ...] = ("ontographrag", "shared", "experiments", "tests", "scripts")
Category: TypeAlias = Literal[
    "explicit",
    "inferred",
    "import",
    "receiver",
    "enum-member",
    "definition",
    "missing",
    "dynamic",
]


@dataclass
class Binding:
    path: str
    scope: str
    symbol: str
    kind: str
    line: int
    category: Category
    declaration: str
    fingerprint: str
    exception_id: str = ""

    def record(self) -> dict[str, str | int]:
        return {
            "path": self.path,
            "scope": self.scope,
            "symbol": self.symbol,
            "kind": self.kind,
            "line": self.line,
            "category": self.category,
            "declaration": self.declaration,
            "fingerprint": self.fingerprint,
            "exception_id": self.exception_id,
        }


@dataclass
class Scope:
    name: str
    kind: str
    declarations: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    seen: set[str] = field(default_factory=lambda: set[str]())
    globals: set[str] = field(default_factory=lambda: set[str]())
    nonlocals: set[str] = field(default_factory=lambda: set[str]())
    aliases: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    enum: bool = False


class Declarations(ast.NodeVisitor):
    """Collect declarations without borrowing them from a nested lexical scope."""

    def __init__(self, scope: Scope) -> None:
        self.scope: Scope = scope

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.scope.declarations[ast.unparse(node.target)] = ast.unparse(node.annotation)

    def visit_Global(self, node: ast.Global) -> None:
        self.scope.globals.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.scope.nonlocals.update(node.names)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return None

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return None


class Coverage(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path: str = path
        self.scopes: list[Scope] = []
        self.bindings: list[Binding] = []

    @property
    def scope(self) -> Scope:
        return self.scopes[-1]

    def emit(
        self,
        node: ast.AST,
        symbol: str,
        kind: str,
        category: Category,
        declaration: str = "",
    ) -> None:
        fingerprint: str = hashlib.sha256(
            (
                f"{getattr(node, 'lineno', 1)}:{getattr(node, 'col_offset', 0)}:"
                + ast.dump(node, include_attributes=False)
            ).encode(),
        ).hexdigest()
        self.bindings.append(
            Binding(
                self.path,
                self.scope.name,
                symbol,
                kind,
                getattr(node, "lineno", 1),
                category,
                declaration,
                fingerprint,
            ),
        )

    def enter(self, name: str, kind: str, body: Iterable[ast.stmt]) -> None:
        scope: Scope = Scope(name, kind)
        collector: Declarations = Declarations(scope)
        for statement in body:
            collector.visit(statement)
        self.scopes.append(scope)

    def visit_Module(self, node: ast.Module) -> None:
        self.enter("<module>", "module", node.body)
        self.generic_visit(node)
        self.scopes.pop()

    def function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Defaults and decorators execute in the enclosing scope, not the body.
        for expression in [*node.decorator_list, *node.args.defaults]:
            self.visit(expression)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        receiver: bool = (
            self.scope.kind == "class"
            and bool(node.args.posonlyargs or node.args.args)
            and not any(
                self.resolve(decorator) in {"staticmethod", "builtins.staticmethod"}
                for decorator in node.decorator_list
            )
        )
        parameters: list[ast.arg] = [*node.args.posonlyargs, *node.args.args]
        parameters.extend(node.args.kwonlyargs)
        if node.args.vararg is not None:
            parameters.append(node.args.vararg)
        if node.args.kwarg is not None:
            parameters.append(node.args.kwarg)
        annotation_start: int = len(self.bindings)
        for parameter in parameters:
            if parameter.annotation is not None:
                self.annotation(parameter.annotation)
        if node.returns is not None:
            self.annotation(node.returns)
        for binding in self.bindings[annotation_start:]:
            binding.scope = f"{self.scope.name}.{node.name}"
        self.emit(node, node.name, "function-definition", "definition")
        self.scope.seen.add(node.name)
        self.enter(f"{self.scope.name}.{node.name}", "function", node.body)
        for index, parameter in enumerate(parameters):
            self.scope.seen.add(parameter.arg)
            if parameter.annotation is not None:
                self.scope.declarations[parameter.arg] = ast.unparse(
                    parameter.annotation,
                )
                self.emit(
                    parameter,
                    parameter.arg,
                    "parameter",
                    "explicit",
                    ast.unparse(parameter.annotation),
                )
            elif receiver and index == 0 and parameter.arg in {"self", "cls"}:
                self.emit(parameter, parameter.arg, "parameter", "receiver")
            else:
                self.emit(parameter, parameter.arg, "parameter", "missing")
        self.emit(
            node,
            "return",
            "signature",
            "explicit" if node.returns is not None else "missing",
            ast.unparse(node.returns) if node.returns is not None else "",
        )
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in [*node.decorator_list, *node.bases]:
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)
        enum: bool = any(
            self.resolve(base)
            in {
                "enum.Enum",
                "enum.IntEnum",
                "enum.StrEnum",
                "enum.Flag",
                "enum.IntFlag",
            }
            for base in node.bases
        )
        self.emit(node, node.name, "class-definition", "definition")
        self.scope.seen.add(node.name)
        self.enter(f"{self.scope.name}.{node.name}", "class", node.body)
        self.scope.enum = enum
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_scope: Scope = Scope(statement.name, "function")
                collector: Declarations = Declarations(method_scope)
                for child in statement.body:
                    collector.visit(child)
                self.scope.declarations.update(
                    (symbol, annotation)
                    for symbol, annotation in method_scope.declarations.items()
                    if symbol.startswith(("self.", "cls."))
                )
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    def bind(
        self,
        target: ast.expr,
        kind: str,
        inferred: bool = False,
        source: ast.AST | None = None,
    ) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self.bind(element, "unpacking", True, source)
            return
        if isinstance(target, ast.Starred):
            self.bind(target.value, kind, inferred, source)
            return
        if not isinstance(target, (ast.Name, ast.Attribute)):
            return
        symbol: str = ast.unparse(target)
        if symbol in self.scope.seen:
            return
        self.scope.seen.add(symbol)
        declaration: str = self.scope.declarations.get(symbol, "")
        if isinstance(target, ast.Attribute):
            for owner in reversed(self.scopes[:-1]):
                if owner.kind == "class":
                    declaration = declaration or owner.declarations.get(symbol, "")
                    if symbol.startswith(("self.", "cls.")):
                        declaration = declaration or owner.declarations.get(
                            symbol.split(".", 1)[1],
                            "",
                        )
                    break
        if isinstance(target, ast.Attribute) and not declaration:
            direct_receiver: bool = isinstance(
                target.value,
                ast.Name,
            ) and target.value.id in {"self", "cls"}
            if not direct_receiver:
                kind = "attribute-mutation"
                inferred = True
        category: Category = "explicit" if declaration else "missing"
        if not declaration and inferred:
            category = "inferred"
        if self.scope.enum and isinstance(target, ast.Name) and not declaration:
            category = "enum-member"
        if symbol in self.scope.globals:
            declaration = self.scopes[0].declarations.get(symbol, "")
            category = "explicit" if declaration else "missing"
        elif symbol in self.scope.nonlocals:
            category = "missing"
            for owner in reversed(self.scopes[1:-1]):
                if owner.kind not in {"function", "lambda"}:
                    continue
                if symbol in owner.declarations:
                    declaration = owner.declarations[symbol]
                    category = "explicit"
                    break
                if symbol in owner.seen:
                    break
        self.emit(
            source if source is not None else target,
            symbol,
            kind,
            category,
            declaration,
        )

    def visit_Assign(self, node: ast.Assign) -> None:
        constructor: str = (
            self.resolve(node.value.func) if isinstance(node.value, ast.Call) else ""
        )
        if constructor in {
            "typing.TypeVar",
            "typing.ParamSpec",
            "typing.TypeVarTuple",
            "typing.NewType",
            "typing.TypedDict",
            "typing_extensions.TypeVar",
            "typing_extensions.ParamSpec",
            "typing_extensions.TypeVarTuple",
            "typing_extensions.NewType",
            "typing_extensions.TypedDict",
        }:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.scope.seen.add(target.id)
                    self.emit(
                        node,
                        target.id,
                        "type-definition",
                        "inferred",
                        constructor,
                    )
            if constructor in {
                "typing.TypedDict",
                "typing_extensions.TypedDict",
            } and isinstance(node.value, ast.Call):
                for index, argument in enumerate(node.value.args):
                    if index == 1 and isinstance(argument, ast.Dict):
                        for field_type in argument.values:
                            self.annotation(field_type)
                    else:
                        self.visit(argument)
                for keyword in node.value.keywords:
                    self.visit(keyword.value)
            elif isinstance(node.value, ast.Call):
                for index, argument in enumerate(node.value.args):
                    if index == 0:
                        self.visit(argument)
                    else:
                        self.annotation(argument)
                for keyword in node.value.keywords:
                    if keyword.arg in {"bound", "default"}:
                        self.annotation(keyword.value)
                    else:
                        self.visit(keyword.value)
            return
        for target in node.targets:
            self.bind(target, "assignment", source=node)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.bind(node.target, "annotation", source=node)
        self.annotation(node.annotation)
        if node.value is not None:
            if self.resolve(node.annotation) in {
                "typing.TypeAlias",
                "typing_extensions.TypeAlias",
            }:
                self.annotation(node.value)
            else:
                self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.bind(node.target, "augmented-assignment", source=node)
        self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        self.bind(node.target, "loop", True)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.bind(node.target, "async-loop", True)
        self.generic_visit(node)

    def visit_withitem(self, node: ast.withitem) -> None:
        if node.optional_vars is not None:
            self.bind(node.optional_vars, "context-manager", True)
        self.visit(node.context_expr)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self.bind(
                ast.Name(id=node.name, ctx=ast.Store(), lineno=node.lineno),
                "exception",
                True,
                node,
            )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            symbol: str = alias.asname or alias.name.split(".")[0]
            self.scope.seen.add(symbol)
            self.scope.aliases[symbol] = alias.name
            self.emit(node, symbol, "import", "import", alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            symbol: str = alias.asname or alias.name
            qualified: str = f"{node.module}.{alias.name}"
            self.scope.seen.add(symbol)
            self.scope.aliases[symbol] = qualified
            self.emit(node, symbol, "import", "import", qualified)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        self.enter(f"{self.scope.name}.<lambda:{node.lineno}>", "lambda", [])
        for parameter in [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]:
            self.scope.seen.add(parameter.arg)
            self.emit(parameter, parameter.arg, "lambda-parameter", "inferred")
        if node.args.vararg is not None:
            self.scope.seen.add(node.args.vararg.arg)
            self.emit(
                node.args.vararg,
                node.args.vararg.arg,
                "lambda-parameter",
                "inferred",
            )
        if node.args.kwarg is not None:
            self.scope.seen.add(node.args.kwarg.arg)
            self.emit(
                node.args.kwarg,
                node.args.kwarg.arg,
                "lambda-parameter",
                "inferred",
            )
        self.visit(node.body)
        self.scopes.pop()

    def comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
    ) -> None:
        self.enter(
            f"{self.scope.name}.<comprehension:{node.lineno}>",
            "comprehension",
            [],
        )
        for generator in node.generators:
            self.bind(generator.target, "comprehension", True)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.comprehension(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        # PEP 572 binds a comprehension walrus in the containing non-comprehension
        # scope. Temporarily expose that owner without changing expression order.
        nested: list[Scope] = []
        while self.scope.kind == "comprehension":
            nested.append(self.scopes.pop())
        self.bind(node.target, "named-expression", True)
        self.scopes.extend(reversed(nested))
        self.visit(node.value)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name is not None:
            self.bind(
                ast.Name(id=node.name, ctx=ast.Store(), lineno=node.lineno),
                "pattern",
                True,
            )
        if node.pattern is not None:
            self.visit(node.pattern)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name is not None:
            self.bind(
                ast.Name(id=node.name, ctx=ast.Store(), lineno=node.lineno),
                "pattern",
                True,
            )

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest is not None:
            self.bind(
                ast.Name(id=node.rest, ctx=ast.Store(), lineno=node.lineno),
                "pattern",
                True,
            )
        self.generic_visit(node)

    def annotation(self, node: ast.expr) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                parsed: ast.expr = ast.parse(node.value, mode="eval").body
            except SyntaxError:
                self.emit(node, node.value, "unparsed-annotation", "dynamic")
                return
            for child in ast.walk(parsed):
                ast.copy_location(child, node)
            self.annotation(parsed)
            return
        if isinstance(node, ast.Subscript):
            self.visit(node.value)
            if self.resolve(node.value) in {
                "typing.Literal",
                "typing_extensions.Literal",
            }:
                return
            if self.resolve(node.value) in {
                "typing.Annotated",
                "typing_extensions.Annotated",
            } and isinstance(node.slice, ast.Tuple):
                self.annotation(node.slice.elts[0])
                return
            self.annotation(node.slice)
            return
        if isinstance(node, (ast.Tuple, ast.List)):
            for element in node.elts:
                self.annotation(element)
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            self.annotation(node.left)
            self.annotation(node.right)
            return
        self.visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and self.resolve(node) in {
            "getattr",
            "builtins.getattr",
        }:
            self.emit(
                node,
                ast.unparse(node),
                "dynamic-attribute-read-reference",
                "dynamic",
            )
        if isinstance(node.ctx, ast.Load) and self.resolve(node) in {
            "typing.Any",
            "typing_extensions.Any",
        }:
            self.emit(node, ast.unparse(node), "dynamic-type", "dynamic")

    def resolve(self, node: ast.expr) -> str:
        if isinstance(node, ast.Attribute):
            return f"{self.resolve(node.value)}.{node.attr}"
        if isinstance(node, ast.Name):
            for scope in reversed(self.scopes):
                if node.id in scope.aliases:
                    return scope.aliases[node.id]
                if node.id in scope.seen or node.id in scope.declarations:
                    break
            return node.id
        return ""

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load) and self.resolve(node) == "builtins.getattr":
            self.emit(
                node,
                ast.unparse(node),
                "dynamic-attribute-read-reference",
                "dynamic",
            )
        if self.resolve(node) in {"typing.Any", "typing_extensions.Any"}:
            self.emit(node, ast.unparse(node), "dynamic-type", "dynamic")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self.resolve(node.func) in {"getattr", "builtins.getattr"}:
            # Even a literal attribute or an object destination can hide Any from
            # both strict checkers. Concrete reads need site-specific producer
            # proof, not a second type engine or an automatic AST exemption.
            self.emit(
                node,
                ast.unparse(node.func),
                "dynamic-attribute-read",
                "dynamic",
                ast.unparse(node),
            )
            for argument in node.args:
                self.visit(argument)
            for keyword in node.keywords:
                self.visit(keyword.value)
            return
        if self.resolve(node.func) in {"typing.cast", "typing_extensions.cast"}:
            self.emit(node, ast.unparse(node.func), "cast", "dynamic")
            if node.args:
                self.annotation(node.args[0])
            for argument in node.args[1:]:
                self.visit(argument)
            for keyword in node.keywords:
                self.visit(keyword.value)
            return
        if self.resolve(node.func) in {
            "setattr",
            "builtins.setattr",
            "unittest.mock.patch",
            "unittest.mock.patch.object",
        } or (isinstance(node.func, ast.Attribute) and node.func.attr == "setattr"):
            self.emit(node, ast.unparse(node.func), "dynamic-attribute", "dynamic")
        self.generic_visit(node)


def discover(root: Path) -> list[Path]:
    tracked: list[str] = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--", "*.py", "*.pyi"],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    unexpected: list[str] = [
        name
        for name in tracked
        if name
        and name.split("/")[0] not in ROOTS
        and not name.startswith("reproducibility/arxiv_v1/")
    ]
    if unexpected:
        raise ValueError(f"Unclassified tracked Python paths: {unexpected}")
    missing: list[str] = [
        name for name in tracked if name and not (root / name).is_file()
    ]
    if missing:
        raise ValueError(f"Tracked Python paths missing from checkout: {missing}")
    paths: list[Path] = sorted(
        path
        for directory in ROOTS
        for path in (root / directory).rglob("*")
        if path.suffix in {".py", ".pyi"} and "__pycache__" not in path.parts
    )
    if not paths:
        raise ValueError("No maintained Python files discovered")
    return paths


def inspect_source(source: str, path: str) -> list[Binding]:
    tree: ast.Module = ast.parse(source, filename=path, type_comments=True)
    visitor: Coverage = Coverage(path)
    visitor.visit(tree)
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and any(
            marker in "".join(token.string.lower().split())
            for marker in ("type:ignore", "pyright:", "noqa")
        ):
            visitor.bindings.append(
                Binding(
                    path,
                    "<module>",
                    token.string,
                    "suppression",
                    token.start[0],
                    "dynamic",
                    "",
                    hashlib.sha256(
                        f"{token.start[0]}:{token.start[1]}:{token.string}".encode(),
                    ).hexdigest(),
                ),
            )
    return visitor.bindings


@dataclass(frozen=True)
class ExceptionEntry:
    identity: str
    path: str
    scope: str
    symbol: str
    kind: str
    fingerprint: str
    reason: str
    evidence: str
    owner: str
    reviewer: str
    disposition: str
    expires: date


def is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def parse_entries(value: object) -> list[ExceptionEntry]:
    if not is_object_list(value):
        raise TypeError("Exception ledger must be a list")
    entries: list[ExceptionEntry] = []
    identities: set[str] = set()
    fields: set[str] = {
        "identity",
        "path",
        "scope",
        "symbol",
        "kind",
        "fingerprint",
        "reason",
        "evidence",
        "owner",
        "reviewer",
        "disposition",
        "expires",
    }
    for raw in value:
        if not is_object_dict(raw) or set(raw) != fields:
            raise ValueError("Exception entry has missing or unexpected fields")
        values: dict[str, str] = {}
        for key, item in raw.items():
            if (
                not isinstance(key, str)
                or not isinstance(item, str)
                or not item.strip()
            ):
                raise ValueError("Exception fields must be nonempty strings")
            values[key] = item
        identity: str = values["identity"]
        if identity in identities:
            raise ValueError(f"Duplicate exception identity: {identity}")
        identities.add(identity)
        if values["disposition"] not in {"pending", "approved"}:
            raise ValueError(f"Invalid disposition for {identity}")
        fingerprint: str = values["fingerprint"]
        if len(fingerprint) != 64 or any(
            char not in "0123456789abcdef" for char in fingerprint
        ):
            raise ValueError(f"Invalid fingerprint for {identity}")
        entries.append(
            ExceptionEntry(
                identity,
                values["path"],
                values["scope"],
                values["symbol"],
                values["kind"],
                fingerprint,
                values["reason"],
                values["evidence"],
                values["owner"],
                values["reviewer"],
                values["disposition"],
                date.fromisoformat(values["expires"]),
            ),
        )
    return entries


def reconcile(
    bindings: list[Binding],
    entries: list[ExceptionEntry],
    today: date,
) -> list[str]:
    errors: list[str] = []
    for entry in entries:
        matches: list[Binding] = [
            binding
            for binding in bindings
            if (binding.path, binding.scope, binding.symbol, binding.kind)
            == (entry.path, entry.scope, entry.symbol, entry.kind)
            and binding.fingerprint == entry.fingerprint
        ]
        if len(matches) != 1:
            errors.append(
                f"{entry.identity}: unused, changed, or ambiguous fingerprint",
            )
            continue
        binding: Binding = matches[0]
        if binding.exception_id:
            errors.append(f"{entry.identity}: duplicate disposition for binding")
            continue
        if binding.category not in {"missing", "dynamic"}:
            errors.append(f"{entry.identity}: exception no longer necessary")
            continue
        if entry.expires < today:
            errors.append(f"{entry.identity}: expired")
            continue
        if entry.disposition != "approved":
            errors.append(f"{entry.identity}: pending independent review")
            continue
        binding.exception_id = entry.identity
    return errors


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exceptions", type=Path)
    args: argparse.Namespace = parser.parse_args()
    root: Path = Path(__file__).resolve().parents[1]
    paths: list[Path] = discover(root)
    bindings: list[Binding] = []
    for path in paths:
        bindings.extend(
            inspect_source(path.read_text(), path.relative_to(root).as_posix()),
        )
    entries: list[ExceptionEntry] = (
        parse_entries(json.loads(args.exceptions.read_text()))
        if args.exceptions is not None
        else []
    )
    ledger_errors: list[str] = reconcile(bindings, entries, datetime.now(UTC).date())
    failures: int = len(ledger_errors) + sum(
        binding.category in {"missing", "dynamic"} and not binding.exception_id
        for binding in bindings
    )
    args.output.write_text(
        json.dumps(
            {
                "files": [path.relative_to(root).as_posix() for path in paths],
                "bindings": [binding.record() for binding in bindings],
                "category_counts": {
                    category: sum(binding.category == category for binding in bindings)
                    for category in (
                        "explicit",
                        "inferred",
                        "import",
                        "receiver",
                        "enum-member",
                        "definition",
                        "missing",
                        "dynamic",
                    )
                },
                "approved_exceptions": sum(
                    bool(binding.exception_id) for binding in bindings
                ),
                "unresolved": failures,
                "ledger_errors": ledger_errors,
            },
            indent=2,
        )
        + "\n",
    )
    print(f"{len(paths)} files; {len(bindings)} bindings; {failures} unresolved")
    return int(args.check and failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
