"""Run 13 experiment characterizations in a fresh, fail-closed process.

Invoke with the project interpreter and no arguments. Only wandb is replaced;
its API must never be used. Python audit hooks are not OS/network isolation.
This runner does not establish constructor or whole-experiment coverage.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import socket
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import CodeType, FrameType, ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

EXPECTED_NODE_IDS: tuple[str, ...] = (
    "tests/test_experiment_generation_failure.py::test_exact_insufficient_information_is_failure_when_gold_is_specific",
    "tests/test_experiment_generation_failure.py::test_exact_insufficient_information_is_not_failure_when_gold_matches",
    "tests/test_experiment_generation_failure.py::test_small_retrieval_study_builds_expected_variants",
    "tests/test_experiment_generation_failure.py::test_final_pair_retrieval_study_builds_expected_variants",
    "tests/test_experiment_generation_failure.py::test_temporary_env_restores_original_values",
    "tests/test_experiment_retrieval_scope.py::test_validate_retrieval_scope_rejects_cross_question_chunks",
    "tests/test_experiment_retrieval_scope.py::test_validate_retrieval_scope_allows_matching_question_scope",
    "tests/test_experiment_retrieval_scope.py::test_collect_sample_responses_preserves_question_scope_and_sampling_pool",
    "tests/test_experiment_retrieval_scope.py::test_retrieval_study_configs_do_not_override_embedding_profile",
    "tests/test_experiment_retrieval_scope.py::test_final_pair_configs_do_not_override_embedding_profile",
    "tests/test_experiment_retrieval_scope.py::test_full_kg_builder_profile_enables_stronger_extraction",
    "tests/test_experiment_retrieval_scope.py::test_kg_routing_distribution_counts_entity_first_and_dense_fallback",
    "tests/test_experiment_retrieval_scope.py::test_question_scoped_multihop_kg_build_uses_lightweight_profile",
)

BLOCKED_EVENTS: frozenset[str] = frozenset(
    {
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
        "socket.getnameinfo",
        "socket.bind",
        "socket.sendmsg",
        "socket.sendto",
        "subprocess.Popen",
        "os.system",
        "os.posix_spawn",
        "os.fork",
        "os.forkpty",
        "os.exec",
        "os.spawn",
    },
)


@dataclass
class BoundaryAudit:
    external_attempts: list[str] = field(default_factory=lambda: list[str]())
    sentinel_attempts: list[str] = field(default_factory=lambda: list[str]())
    permitted_probes: list[str] = field(default_factory=lambda: list[str]())
    probe_sockets: list[socket.socket] = field(
        default_factory=lambda: list[socket.socket](),
    )
    probe_module_code: CodeType | None = None
    probe_code: CodeType | None = None

    def install(self) -> None:
        # Pin reviewed installed bytes: dependency upgrades require fresh review.
        source: Path = Path(
            str(
                importlib.metadata.distribution("urllib3").locate_file(
                    "urllib3/util/connection.py",
                ),
            ),
        ).resolve()
        content: bytes = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != (
            "41e51412e3668739e9b8a34f2fe0742153a431d302bbca095fad8e0b4569cee8"
        ):
            raise RuntimeError("Unreviewed urllib3 IPv6 probe source")
        if "urllib3.util.connection" in sys.modules:
            raise RuntimeError("Use a fresh process without preloaded urllib3")
        self.probe_module_code = compile(
            content,
            str(source),
            "exec",
            dont_inherit=True,
        )
        self.probe_code = next(
            code
            for code in self.probe_module_code.co_consts
            if isinstance(code, CodeType) and code.co_name == "_has_ipv6"
        )
        sys.addaudithook(self.guard)
        # CPython does not emit audit events for these socket methods.
        for name in ("listen", "send", "sendall"):
            setattr(socket.socket, name, self.block_socket_method(name))

    def block_socket_method(self, name: str) -> Callable[..., None]:
        def blocked(*args: object, **kwargs: object) -> None:
            self.reject(f"socket.{name}")

        return blocked

    def reject(self, event: str) -> None:
        self.external_attempts.append(event)
        raise RuntimeError(f"External operation blocked: {event}")

    def permits_probe(self, args: tuple[object, ...], caller: FrameType) -> bool:
        module: ModuleType | None = sys.modules.get("urllib3.util.connection")
        parent: FrameType | None = caller.f_back
        if (
            self.permitted_probes
            or self.probe_code is None
            or module is None
            or caller.f_globals is not vars(module)
            or caller.f_code != self.probe_code
            or parent is None
            or parent.f_globals is not vars(module)
            or parent.f_code != self.probe_module_code
            or not getattr(module.__spec__, "_initializing", False)
            or len(args) != 2
        ):
            return False
        sock: object = args[0]
        return (
            sock is caller.f_locals.get("sock")
            and isinstance(sock, socket.socket)
            and sock.family == socket.AF_INET6
            and sock.type == socket.SOCK_STREAM
            and sock.proto == 0
            and args[1] == ("::1", 0)
            and caller.f_locals.get("host") == "::1"
        )

    def guard(self, event: str, args: tuple[object, ...]) -> None:
        if event == "socket.bind" and self.permits_probe(args, sys._getframe(1)):
            sock: object = args[0]
            assert isinstance(sock, socket.socket)
            self.probe_sockets.append(sock)
            self.permitted_probes.append("urllib3.util.connection._has_ipv6(::1, 0)")
            return
        if event in BLOCKED_EVENTS:
            self.reject(event)


class UnavailableWandb(ModuleType):
    def __init__(self, audit: BoundaryAudit) -> None:
        super().__init__("wandb")
        self._audit: BoundaryAudit = audit

    def __getattr__(self, name: str) -> object:
        self._audit.sentinel_attempts.append(name)
        raise RuntimeError(f"Unexpected wandb attribute use: {name}")


@dataclass(frozen=True)
class TestOutcome:
    nodeid: str
    phase: str
    outcome: str
    xfail: bool


@dataclass
class OutcomeAudit:
    collected: list[str] = field(default_factory=lambda: list[str]())
    deselected: list[str] = field(default_factory=lambda: list[str]())
    collection_errors: list[str] = field(default_factory=lambda: list[str]())
    reports: list[TestOutcome] = field(default_factory=lambda: list[TestOutcome]())

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.collected = [item.nodeid for item in session.items]

    def pytest_deselected(self, items: list[pytest.Item]) -> None:
        self.deselected.extend(item.nodeid for item in items)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if not report.passed:
            self.collection_errors.append(f"{report.nodeid}: {report.outcome}")

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.reports.append(
            TestOutcome(
                nodeid=report.nodeid,
                phase=report.when,
                outcome=report.outcome,
                xfail=hasattr(report, "wasxfail"),
            ),
        )

    def violations(self, pytest_exit: int, audit: BoundaryAudit) -> list[str]:
        errors: list[str] = []
        if pytest_exit != 0:
            errors.append(f"pytest exited {pytest_exit}")
        if Counter(self.collected) != Counter(EXPECTED_NODE_IDS):
            errors.append("collection does not match the exact 13 expected node IDs")
        if self.deselected:
            errors.append("tests were deselected")
        if self.collection_errors:
            errors.append("collection contained errors or skips")
        expected: Counter[tuple[str, str, str, bool]] = Counter(
            (nodeid, phase, "passed", False)
            for nodeid in EXPECTED_NODE_IDS
            for phase in ("setup", "call", "teardown")
        )
        observed: Counter[tuple[str, str, str, bool]] = Counter(
            (report.nodeid, report.phase, report.outcome, report.xfail)
            for report in self.reports
        )
        if observed != expected:
            errors.append("expected exactly 13 passed setup/call/teardown outcomes")
        if any(sock.fileno() != -1 for sock in audit.probe_sockets):
            errors.append("permitted probe socket was not closed")
        if audit.external_attempts:
            errors.append("audited external operations were attempted")
        if audit.sentinel_attempts:
            errors.append("wandb sentinel attributes were used")
        return errors


def main() -> int:
    if len(sys.argv) != 1:
        print("This fixed-scope runner accepts no arguments.", file=sys.stderr)
        return 2
    if "experiments.experiment" in sys.modules or "wandb" in sys.modules:
        print(
            "Use a fresh Python process without preloaded experiment/wandb.",
            file=sys.stderr,
        )
        return 2

    root: Path = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    temp_root: Path = (
        root
        / ".scratch/python-quality-refactor/evidence/engineer-07-probe-exception-temp"
    )
    temp_root.mkdir(parents=True, exist_ok=True)

    audit: BoundaryAudit = BoundaryAudit()
    audit.install()
    sys.modules["wandb"] = UnavailableWandb(audit)
    outcomes: OutcomeAudit = OutcomeAudit()
    with tempfile.TemporaryDirectory(prefix="run-", dir=temp_root) as directory:
        temporary: Path = Path(directory)
        os.environ["TMPDIR"] = directory
        os.environ["XDG_CACHE_HOME"] = str(temporary / "cache")
        os.environ["MPLCONFIGDIR"] = str(temporary / "matplotlib")
        # These tests do not render plots; avoid font discovery subprocesses.
        os.environ["MPL_IGNORE_SYSTEM_FONTS"] = "1"
        import pytest

        argv: list[str] = [
            "-q",
            "-rs",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            f"--basetemp={temporary / 'pytest'}",
            "tests/test_experiment_generation_failure.py",
            "tests/test_experiment_retrieval_scope.py",
        ]
        pytest_exit: int = int(pytest.main(argv, plugins=[outcomes]))

    violations: list[str] = outcomes.violations(pytest_exit, audit)
    exit_code: int = pytest_exit or int(bool(violations))
    print(
        "EXPERIMENT_ISOLATION_RESULT="
        + json.dumps(
            {
                "python": sys.version,
                "executable": sys.executable,
                "pytest_argv": argv,
                "expected": EXPECTED_NODE_IDS,
                "collected": outcomes.collected,
                "deselected": outcomes.deselected,
                "collection_errors": outcomes.collection_errors,
                "reports": [asdict(report) for report in outcomes.reports],
                "permitted_probes": audit.permitted_probes,
                "probe_sockets_closed": all(
                    sock.fileno() == -1 for sock in audit.probe_sockets
                ),
                "boundary_attempts": audit.external_attempts,
                "sentinel_attempts": audit.sentinel_attempts,
                "violations": violations,
                "pytest_exit": pytest_exit,
                "exit_code": exit_code,
            },
            sort_keys=True,
        ),
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
