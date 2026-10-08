"""Whole-scope gate receipts must fail closed without running real checkers."""

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.check_typing_baseline import Command, Receipt, commands, execute, run


@dataclass
class GateFixture:
    root: Path
    output: Path
    paths: list[Path]
    scenario: str = "clean"
    ledger: Path | None = None
    calls: list[list[str]] = field(default_factory=lambda: list[list[str]]())

    def discover(self, root: Path) -> list[Path]:
        assert root == self.root
        return list(self.paths)

    def process(
        self,
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        text: bool,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == self.root
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        assert text and capture_output and not check
        self.calls.append(list(argv))
        stdout: str = "tool 1.0\n"
        returncode: int = 0
        if self.scenario == "mutated-source":
            self.paths[0].write_text("value: int = 2\n")
        if self.scenario == "added-source":
            added: Path = self.root / "scripts" / "added.py"
            added.write_text("value: int = 3\n")
            self.paths.append(added)
        if self.scenario == "mutated-config":
            (self.root / "mypy.ini").write_text("[mypy]\nstrict = False\n")
        if self.scenario == "deleted-source":
            self.paths[0].unlink()
        if self.scenario == "mutated-ledger" and self.ledger is not None:
            self.ledger.write_text("[{}]")
        if self.scenario == "missing-tool":
            raise FileNotFoundError("fixture tool absent")
        if "scripts.check_typing_coverage" in argv:
            manifest: list[str] = [
                path.relative_to(self.root).as_posix() for path in self.paths
            ]
            if self.scenario == "partial-audit":
                manifest = []
            unresolved: int = int(self.scenario == "pending-ledger")
            if self.scenario != "missing-audit":
                (self.output / "bindings.json").write_text(
                    json.dumps(
                        {
                            "files": manifest,
                            "unresolved": unresolved,
                            "ledger_errors": [],
                        }
                    ),
                )
            returncode = unresolved
        elif argv[0].endswith("mypy") and "--version" not in argv:
            count: int = 0 if self.scenario == "partial-mypy" else len(self.paths)
            stdout = f"Success: no issues found in {count} source files\n"
            if self.scenario == "malformed-mypy":
                stdout = "looks fine\n"
            if self.scenario == "failed-mypy":
                returncode = 1
                stdout = f"Found 1 error in 1 file (checked {count} source files)\n"
        elif argv[0].endswith("pyright") and "--version" not in argv:
            count = 0 if self.scenario == "partial-pyright" else len(self.paths)
            stdout = json.dumps({"summary": {"filesAnalyzed": count}})
            if self.scenario == "malformed-pyright":
                stdout = "not json"
            if self.scenario == "failed-pyright":
                returncode = 1
        elif (
            "format" in argv
            and self.scenario == "failed-format"
            or "check" in argv
            and self.scenario == "failed-lint"
            or "--version" in argv
            and self.scenario == "failed-version"
        ):
            returncode = 1
        return subprocess.CompletedProcess(argv, returncode, stdout, "")


@pytest.fixture
def gate(tmp_path: Path) -> GateFixture:
    root: Path = tmp_path / "repo"
    source: Path = root / "scripts" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("value: int = 1\n")
    for name in ("mypy.ini", "ruff.toml", "pyproject.toml"):
        (root / name).write_text("")
    (root / "pyrightconfig.json").write_text('{"typeCheckingMode": "strict"}')
    return GateFixture(root, tmp_path / "evidence", [source])


def invoke(gate: GateFixture, exceptions: Path | None = None) -> int:
    with (
        patch("scripts.check_typing_baseline.discover", side_effect=gate.discover),
        patch("scripts.check_typing_baseline.subprocess.run", side_effect=gate.process),
    ):
        return run(gate.root, gate.output, exceptions)


def test_complete_gate_records_exact_commands_hashes_and_versions(
    gate: GateFixture,
) -> None:
    assert invoke(gate) == 0
    receipt: str = (gate.output / "receipt.json").read_text()
    assert '"passed": true' in receipt
    assert '"scripts/example.py"' in receipt
    assert '"source_hashes_before"' in receipt
    assert '"source_hashes_after"' in receipt
    assert len(gate.calls) == 9
    for name in ("python", "ruff", "mypy", "pyright"):
        assert (gate.output / f"{name}-version.stdout").read_text() == "tool 1.0\n"
    for argv in gate.calls:
        if "--any-exprs-report" in argv:
            assert argv[argv.index("--any-exprs-report") + 1] == str(
                gate.output / "mypy-any",
            )
        if any(
            item in argv for item in ("--strict", "--outputjson", "format", "check")
        ):
            assert "scripts/example.py" in argv


@pytest.mark.parametrize(
    "scenario",
    [
        "partial-audit",
        "missing-audit",
        "pending-ledger",
        "partial-mypy",
        "partial-pyright",
        "malformed-mypy",
        "malformed-pyright",
        "failed-mypy",
        "failed-pyright",
        "failed-format",
        "failed-lint",
        "failed-version",
        "missing-tool",
        "mutated-source",
        "mutated-config",
        "added-source",
        "deleted-source",
    ],
)
def test_failures_never_become_a_clean_receipt(
    gate: GateFixture, scenario: str
) -> None:
    gate.scenario = scenario
    assert invoke(gate) == 1
    assert '"passed": false' in (gate.output / "receipt.json").read_text()
    if scenario in {"mutated-source", "mutated-config", "added-source"}:
        assert len(gate.calls) == 1


def test_empty_scope_runs_no_checker(gate: GateFixture) -> None:
    gate.paths.clear()
    assert invoke(gate) == 1
    assert not gate.calls


def test_non_strict_pyright_configuration_runs_no_checker(gate: GateFixture) -> None:
    (gate.root / "pyrightconfig.json").write_text('{"typeCheckingMode": "standard"}')
    assert invoke(gate) == 1
    assert not gate.calls


def test_output_cannot_be_inside_checkout(gate: GateFixture) -> None:
    with pytest.raises(ValueError, match="outside the checkout"):
        run(gate.root, gate.root / "evidence")


def test_nonempty_evidence_cannot_supply_stale_success(gate: GateFixture) -> None:
    gate.output.mkdir()
    previous: Path = gate.output / "bindings.json"
    previous.write_text("old evidence")
    with pytest.raises(ValueError, match="must be empty"):
        run(gate.root, gate.output)
    assert previous.read_text() == "old evidence"


def test_pending_exceptions_are_forwarded_only_to_audit(gate: GateFixture) -> None:
    ledger: Path = gate.output.parent / "pending.json"
    ledger.write_text("[]")
    gate.scenario = "pending-ledger"
    assert invoke(gate, ledger) == 1
    ledger_commands: list[list[str]] = [
        argv for argv in gate.calls if "--exceptions" in argv
    ]
    assert len(ledger_commands) == 1
    assert "scripts.check_typing_coverage" in ledger_commands[0]
    assert str(ledger) in ledger_commands[0]


def test_command_plan_rejects_duplicate_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nonempty and unique"):
        commands(["scripts/example.py", "scripts/example.py"], tmp_path)


def test_missing_executable_is_captured_as_a_failed_receipt(tmp_path: Path) -> None:
    with patch(
        "scripts.check_typing_baseline.subprocess.run",
        side_effect=FileNotFoundError("missing"),
    ):
        receipt: Receipt = execute(Command("missing", ["missing"]), tmp_path, tmp_path)
    assert receipt.returncode == 127
    assert (tmp_path / "missing.stderr").read_text() == "missing"


def test_ledger_mutation_invalidates_even_successful_tool_output(
    gate: GateFixture,
) -> None:
    ledger: Path = gate.output.parent / "exceptions.json"
    ledger.write_text("[]")
    gate.ledger = ledger
    gate.scenario = "mutated-ledger"
    assert invoke(gate, ledger) == 1
    assert len(gate.calls) == 1
    assert "Exception ledger changed" in (gate.output / "receipt.json").read_text()
