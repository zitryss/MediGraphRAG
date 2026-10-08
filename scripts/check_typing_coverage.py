"""Inventory maintained Python and report lexical annotation coverage.

This complements (and never replaces) strict mypy/Pyright: syntax-constrained
bindings are reported as inferred, not certified as concretely typed here.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ROOTS: tuple[str, ...] = ("ontographrag", "shared", "experiments", "tests", "scripts")
Category = Literal["explicit", "inferred", "import", "receiver", "missing", "dynamic"]


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
        }


@dataclass
class Scope:
    name: str
    kind: str
    declarations: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    seen: set[str] = field(default_factory=lambda: set[str]())
    external: set[str] = field(default_factory=lambda: set[str]())


class Declarations(ast.NodeVisitor):
    """Collect declarations without borrowing them from a nested lexical scope."""

    def __init__(self, scope: Scope) -> None:
        self.scope: Scope = scope

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.scope.declarations[ast.unparse(node.target)] = ast.unparse(node.annotation)

    def visit_Global(self, node: ast.Global) -> None:
        self.scope.external.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.scope.external.update(node.names)

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
            ast.dump(node, include_attributes=False).encode()
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
            )
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
        receiver: bool = self.scope.kind == "class" and not any(
            isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
            for decorator in node.decorator_list
        )
        self.scope.seen.add(node.name)
        self.enter(f"{self.scope.name}.{node.name}", "function", node.body)
        parameters: list[ast.arg] = [*node.args.posonlyargs, *node.args.args]
        parameters.extend(node.args.kwonlyargs)
        if node.args.vararg is not None:
            parameters.append(node.args.vararg)
        if node.args.kwarg is not None:
            parameters.append(node.args.kwarg)
        for index, parameter in enumerate(parameters):
            self.scope.seen.add(parameter.arg)
            if parameter.annotation is not None:
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
        for parameter in parameters:
            if parameter.annotation is not None:
                self.visit(parameter.annotation)
        if node.returns is not None:
            self.visit(node.returns)
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
        self.scope.seen.add(node.name)
        self.enter(f"{self.scope.name}.{node.name}", "class", node.body)
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
        self.generic_visit(node)
        self.scopes.pop()

    def bind(self, target: ast.expr, kind: str, inferred: bool = False) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self.bind(element, "unpacking", True)
            return
        if isinstance(target, ast.Starred):
            self.bind(target.value, kind, inferred)
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
                            symbol.split(".", 1)[1], ""
                        )
                    break
        category: Category = "explicit" if declaration else "missing"
        if not declaration and inferred:
            category = "inferred"
        if symbol in self.scope.external:
            for owner in reversed(self.scopes[:-1]):
                if symbol in owner.declarations:
                    declaration = owner.declarations[symbol]
                    category = "explicit"
                    break
                if symbol in owner.seen:
                    break
        self.emit(target, symbol, kind, category, declaration)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self.bind(target, "assignment")
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.bind(node.target, "annotation")
        self.visit(node.annotation)
        if node.value is not None:
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
            self.emit(node, node.name, "exception", "inferred")
            self.scope.seen.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            symbol: str = alias.asname or alias.name.split(".")[0]
            self.scope.seen.add(symbol)
            self.emit(node, symbol, "import", "import", alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.visit_Import(ast.Import(names=node.names, lineno=node.lineno))

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.enter(f"{self.scope.name}.<lambda:{node.lineno}>", "lambda", [])
        for parameter in [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]:
            self.emit(parameter, parameter.arg, "lambda-parameter", "inferred")
        if node.args.vararg is not None:
            self.emit(
                node.args.vararg, node.args.vararg.arg, "lambda-parameter", "inferred"
            )
        if node.args.kwarg is not None:
            self.emit(
                node.args.kwarg, node.args.kwarg.arg, "lambda-parameter", "inferred"
            )
        self.visit(node.body)
        self.scopes.pop()

    def comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        self.enter(
            f"{self.scope.name}.<comprehension:{node.lineno}>", "comprehension", []
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
        self.bind(node.target, "named-expression", True)
        self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "Any" and isinstance(node.ctx, ast.Load):
            self.emit(node, "Any", "dynamic-type", "dynamic")


def discover(root: Path) -> list[Path]:
    tracked: list[str] = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--", "*.py", "*.pyi"], cwd=root
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
    return visitor.bindings


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args: argparse.Namespace = parser.parse_args()
    root: Path = Path(__file__).resolve().parents[1]
    paths: list[Path] = discover(root)
    bindings: list[Binding] = []
    for path in paths:
        bindings.extend(
            inspect_source(path.read_text(), path.relative_to(root).as_posix())
        )
    failures: int = sum(
        binding.category in {"missing", "dynamic"} for binding in bindings
    )
    args.output.write_text(
        json.dumps(
            {
                "files": [path.relative_to(root).as_posix() for path in paths],
                "bindings": [binding.record() for binding in bindings],
                "unresolved": failures,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{len(paths)} files; {len(bindings)} bindings; {failures} unresolved")
    return int(args.check and failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
