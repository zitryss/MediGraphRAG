"""Run fail-closed whole-scope typing gates with reproducible local receipts.

Invoke as ``python -m scripts.check_typing_baseline --output-dir PATH``.
PATH must be empty and outside the checkout. Tests and builds are separate gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from scripts.check_typing_coverage import discover, is_object_dict, is_object_list


@dataclass(frozen=True)
class Command:
    name: str
    argv: list[str]


@dataclass(frozen=True)
class Receipt:
    name: str
    argv: list[str]
    returncode: int
    started_at: str
    finished_at: str
    stdout: str
    stderr: str


def execute(command: Command, root: Path, output: Path) -> Receipt:
    started_at: str = datetime.now(UTC).isoformat()
    environment: dict[str, str] = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    stdout: str
    stderr: str
    returncode: int
    try:
        process: subprocess.CompletedProcess[str] = subprocess.run(
            command.argv,
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        stdout, stderr, returncode = (
            process.stdout,
            process.stderr,
            process.returncode,
        )
    except OSError as error:
        stdout, stderr, returncode = "", str(error), 127
    receipt: Receipt = Receipt(
        command.name,
        command.argv,
        returncode,
        started_at,
        datetime.now(UTC).isoformat(),
        stdout,
        stderr,
    )
    (output / f"{command.name}.stdout").write_text(stdout, encoding="utf-8")
    (output / f"{command.name}.stderr").write_text(stderr, encoding="utf-8")
    return receipt


def source_hashes(root: Path, paths: list[Path]) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def mypy_count(stdout: str) -> int:
    match: re.Match[str] | None = re.search(
        r"(?:Success: no issues found in |\(checked )(\d+) source files?",
        stdout,
    )
    if match is None:
        raise ValueError("mypy did not report its checked source-file count")
    return int(match.group(1))


def pyright_count(stdout: str) -> int:
    payload: object = json.loads(stdout)
    if not is_object_dict(payload):
        raise ValueError("Pyright output is not an object")
    summary: object = payload.get("summary")
    if not is_object_dict(summary):
        raise ValueError("Pyright output has no summary")
    count: object = summary.get("filesAnalyzed")
    if type(count) is not int:
        raise ValueError("Pyright output has no integer analyzed-file count")
    return count


def audit_errors(path: Path, expected: list[str]) -> list[str]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not is_object_dict(payload):
        return ["Binding audit output is not an object"]
    paths: object = payload.get("files")
    if not is_object_list(paths) or any(not isinstance(item, str) for item in paths):
        return ["Binding audit did not provide a file manifest"]
    if len(paths) != len(expected) or set(paths) != set(expected):
        return ["Binding audit manifest differs from maintained files"]
    unresolved: object = payload.get("unresolved")
    if (
        type(unresolved) is not int
        or unresolved != 0
        or payload.get("ledger_errors") != []
    ):
        return ["Binding audit contains unresolved bindings or ledger errors"]
    return []


def commands(
    paths: list[str],
    output: Path,
    exceptions: Path | None = None,
) -> list[Command]:
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("The maintained manifest must be nonempty and unique")
    audit_argv: list[str] = [
        ".venv/bin/python",
        "-m",
        "scripts.check_typing_coverage",
        "--check",
        "--output",
        str(output / "bindings.json"),
    ]
    if exceptions is not None:
        audit_argv.extend(["--exceptions", str(exceptions)])
    return [
        Command("python-version", [".venv/bin/python", "--version"]),
        Command("ruff-version", [".venv/bin/ruff", "--version"]),
        Command("mypy-version", [".venv/bin/mypy", "--version"]),
        Command("pyright-version", [".venv/bin/pyright", "--version"]),
        Command("bindings", audit_argv),
        Command(
            "format",
            [".venv/bin/ruff", "format", "--check", "--no-cache", *paths],
        ),
        Command(
            "lint",
            [
                ".venv/bin/ruff",
                "check",
                "--no-cache",
                "--extend-select",
                "ANN",
                "--output-format",
                "json",
                *paths,
            ],
        ),
        Command(
            "mypy",
            [
                ".venv/bin/mypy",
                "--config-file",
                "mypy.ini",
                "--explicit-package-bases",
                "--strict",
                "--cache-dir=/dev/null",
                "--no-incremental",
                "--any-exprs-report",
                str(output / "mypy-any"),
                *paths,
            ],
        ),
        Command(
            "pyright",
            [
                ".venv/bin/pyright",
                "--project",
                "pyrightconfig.json",
                "--pythonpath",
                ".venv/bin/python",
                "--warnings",
                "--outputjson",
                *paths,
            ],
        ),
    ]


def run(root: Path, output: Path, exceptions: Path | None = None) -> int:
    root = root.resolve()
    output = output.resolve()
    if output.is_relative_to(root):
        raise ValueError("Evidence directory must be outside the checkout")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evidence directory must be empty to reject stale receipts")
    output.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    receipts: list[Receipt] = []
    paths: list[Path] = []
    manifest: list[str] = []
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    exception_hash: str | None = None
    tracked_inputs: list[Path] = []
    try:
        paths = discover(root)
        manifest = [path.relative_to(root).as_posix() for path in paths]
        tracked_inputs = [
            *paths,
            *(
                root / name
                for name in (
                    "mypy.ini",
                    "pyrightconfig.json",
                    "ruff.toml",
                    "pyproject.toml",
                )
            ),
        ]
        before = source_hashes(root, tracked_inputs)
        configuration: object = json.loads((root / "pyrightconfig.json").read_text())
        if (
            not is_object_dict(configuration)
            or configuration.get("typeCheckingMode") != "strict"
        ):
            raise ValueError("Pyright configuration must enforce strict mode")
        if exceptions is not None:
            exceptions = exceptions.resolve()
            exception_hash = hashlib.sha256(exceptions.read_bytes()).hexdigest()
        for command in commands(manifest, output, exceptions):
            receipt: Receipt = execute(command, root, output)
            receipts.append(receipt)
            if receipt.returncode != 0:
                errors.append(f"{command.name} exited {receipt.returncode}")
            try:
                if command.name == "mypy" and mypy_count(receipt.stdout) != len(paths):
                    errors.append(
                        "mypy checked-file count differs from maintained manifest",
                    )
                if command.name == "pyright" and pyright_count(receipt.stdout) != len(
                    paths,
                ):
                    errors.append(
                        "Pyright analyzed-file count differs from maintained manifest",
                    )
                if command.name == "bindings":
                    errors.extend(audit_errors(output / "bindings.json", manifest))
            except (OSError, ValueError) as error:
                errors.append(f"{command.name}: {error}")
            after = source_hashes(root, tracked_inputs)
            if discover(root) != paths or before != after:
                errors.append("Maintained source/configuration changed during checking")
                break
            if (
                exceptions is not None
                and hashlib.sha256(exceptions.read_bytes()).hexdigest()
                != exception_hash
            ):
                errors.append("Exception ledger changed during checking")
                break
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        errors.append(str(error))
    (output / "receipt.json").write_text(
        json.dumps(
            {
                "root": str(root),
                "files": manifest,
                "source_hashes_before": before,
                "source_hashes_after": after,
                "exceptions": str(exceptions) if exceptions is not None else None,
                "exception_sha256": exception_hash,
                "commands": [asdict(receipt) for receipt in receipts],
                "errors": errors,
                "passed": not errors,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return int(bool(errors))


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exceptions", type=Path)
    args: argparse.Namespace = parser.parse_args()
    try:
        return run(
            Path(__file__).resolve().parents[1],
            args.output_dir,
            args.exceptions,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
