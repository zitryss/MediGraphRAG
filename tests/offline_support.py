"""Default offline boundaries and scoped experiment imports for unit tests."""

from __future__ import annotations

import importlib
import importlib.util
import os
import socket
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import dotenv
import pytest


def deny_network(*args: object, **kwargs: object) -> None:
    raise RuntimeError("Unit tests must not connect to external services")


_OFFLINE_PATCH: pytest.StashKey[pytest.MonkeyPatch] = pytest.StashKey()


class OfflineTelemetry(ModuleType):
    init = staticmethod(deny_network)
    log = staticmethod(deny_network)
    finish = staticmethod(deny_network)


def pytest_configure(config: pytest.Config) -> None:
    patch = pytest.MonkeyPatch()
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
def experiment_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """Import the real pipeline, faking only unavailable telemetry; restore globals."""
    # Import optional plotting before introducing telemetry, so it cannot retain
    # the temporary wandb module after this fixture exits.
    importlib.import_module("experiments.visualize_results")
    telemetry = OfflineTelemetry("wandb")
    monkeypatch.setitem(sys.modules, "wandb", telemetry)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: False)
    monkeypatch.setenv(
        "EMBEDDING_PROVIDER", os.getenv("EMBEDDING_PROVIDER", "huggingface")
    )
    old_path = sys.path[:]
    path = Path(__file__).parents[1] / "experiments/experiment.py"
    spec = importlib.util.spec_from_file_location("isolated_experiment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = old_path


@pytest.fixture
def analysis_import(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[[str], ModuleType]]:
    """Load legacy scripts without depending on the author's paper manifest."""
    import builtins
    import io

    original_open = builtins.open
    old_path = sys.path[:]
    original_modules = {
        k: v for k, v in sys.modules.items() if k.startswith("experiments.")
    }

    def fixture_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if str(file).endswith("latest_results_manifest.json"):
            return io.StringIO("{}")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fixture_open)

    def load(name: str) -> ModuleType:
        path = Path(__file__).parents[1] / "experiments" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"analysis_{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module

    try:
        yield load
    finally:
        sys.path[:] = old_path
        for key in list(sys.modules):
            if key.startswith("experiments.") and key not in original_modules:
                loaded = sys.modules.pop(key)
                parent_name, _, attribute = key.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None and getattr(parent, attribute, None) is loaded:
                    delattr(parent, attribute)
