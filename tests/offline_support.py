"""Default offline boundaries and scoped experiment imports for unit tests."""

from __future__ import annotations

import importlib
import importlib.util
import io
import os
import socket
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Concatenate,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
)

import dotenv
import pytest

if TYPE_CHECKING:
    from importlib.machinery import ModuleSpec

    from experiments.experiment import MIRAGEEvaluationPipeline as EvaluationPipeline


class DotenvDisabled(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> bool: ...


def deny_network(*args: object, **kwargs: object) -> None:
    raise RuntimeError("Unit tests must not connect to external services")


_OFFLINE_PATCH: pytest.StashKey[pytest.MonkeyPatch] = pytest.StashKey()


class OfflineOperation(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> None: ...


class OfflineTelemetry(ModuleType):
    init: ClassVar[OfflineOperation] = staticmethod(deny_network)
    log: ClassVar[OfflineOperation] = staticmethod(deny_network)
    finish: ClassVar[OfflineOperation] = staticmethod(deny_network)


class ExperimentModule(Protocol):
    MIRAGEEvaluationPipeline: type[EvaluationPipeline]
    wandb: OfflineTelemetry

    @staticmethod
    def update_runs_index(run_dir: Path, manifest: Mapping[str, object]) -> None: ...


def pytest_configure(config: pytest.Config) -> None:
    patch: pytest.MonkeyPatch = pytest.MonkeyPatch()
    config.stash[_OFFLINE_PATCH] = patch
    patch.setattr(socket.socket, "connect", deny_network)
    patch.setattr(socket.socket, "connect_ex", deny_network)
    patch.setattr(socket, "getaddrinfo", deny_network)
    patch.setenv("HF_HUB_OFFLINE", "1")
    patch.setenv("TRANSFORMERS_OFFLINE", "1")
    patch.setenv("MPLBACKEND", "Agg")
    patch.setenv("PYTHON_DOTENV_DISABLED", "1")


def pytest_unconfigure(config: pytest.Config) -> None:
    config.stash[_OFFLINE_PATCH].undo()


@pytest.fixture
def experiment_module(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[ExperimentModule]:
    """Import the real pipeline, faking only unavailable telemetry; restore globals."""
    # Import optional plotting before introducing telemetry, so it cannot retain
    # the temporary wandb module after this fixture exits.
    importlib.import_module("experiments.visualize_results")
    telemetry: OfflineTelemetry = OfflineTelemetry("wandb")
    monkeypatch.setitem(sys.modules, "wandb", telemetry)
    disable_dotenv: DotenvDisabled = lambda *a, **kw: False
    monkeypatch.setattr(dotenv, "load_dotenv", disable_dotenv)
    monkeypatch.setenv(
        "EMBEDDING_PROVIDER",
        os.getenv("EMBEDDING_PROVIDER", "huggingface"),
    )
    old_path: list[str] = sys.path[:]
    path: Path = Path(__file__).parents[1] / "experiments/experiment.py"
    spec: ModuleSpec | None = importlib.util.spec_from_file_location(
        "isolated_experiment",
        path,
    )
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    try:
        spec.loader.exec_module(module)
        yield cast(ExperimentModule, module)
    finally:
        sys.path[:] = old_path


OpenFile_contra = TypeVar("OpenFile_contra", contravariant=True)
OpenResult_co = TypeVar("OpenResult_co", covariant=True)
OpenParameters = ParamSpec("OpenParameters")


class ManifestOpener(Protocol[OpenFile_contra, OpenParameters, OpenResult_co]):
    def __call__(
        self,
        file: OpenFile_contra,
        *args: OpenParameters.args,
        **kwargs: OpenParameters.kwargs,
    ) -> OpenResult_co | io.StringIO: ...


def manifest_opener(
    original_open: Callable[
        Concatenate[OpenFile_contra, OpenParameters],
        OpenResult_co,
    ],
) -> ManifestOpener[OpenFile_contra, OpenParameters, OpenResult_co]:
    def fixture_open(
        file: OpenFile_contra,
        *args: OpenParameters.args,
        **kwargs: OpenParameters.kwargs,
    ) -> OpenResult_co | io.StringIO:
        if str(file).endswith("latest_results_manifest.json"):
            return io.StringIO("{}")
        return original_open(file, *args, **kwargs)

    return fixture_open


@pytest.fixture
def analysis_import(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[[str], ModuleType]]:
    """Load legacy scripts without depending on the author's paper manifest."""
    import builtins

    old_path: list[str] = sys.path[:]
    original_modules: dict[str, ModuleType] = {
        k: v for k, v in sys.modules.items() if k.startswith("experiments.")
    }

    monkeypatch.setattr(builtins, "open", manifest_opener(builtins.open))

    def load(name: str) -> ModuleType:
        path: Path = Path(__file__).parents[1] / "experiments" / f"{name}.py"
        spec: ModuleSpec | None = importlib.util.spec_from_file_location(
            f"analysis_{name}",
            path,
        )
        assert spec is not None and spec.loader is not None
        module: ModuleType = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module

    try:
        yield load
    finally:
        sys.path[:] = old_path
        for key in list(sys.modules):
            if key.startswith("experiments.") and key not in original_modules:
                loaded: ModuleType = sys.modules.pop(key)
                parent_name, _, attribute = key.rpartition(".")
                parent: ModuleType | None = sys.modules.get(parent_name)
                if parent is not None and getattr(parent, attribute, None) is loaded:
                    delattr(parent, attribute)
