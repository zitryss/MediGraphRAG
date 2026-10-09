"""Retrieval audit and development package checks using local synthetic inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from experiments import retrieval_audit as audit
from scripts import check_arxiv_package as package

MALFORMED_MANIFEST_CASES: tuple[
    tuple[package.JSONValue, type[AttributeError | SystemExit | TypeError]],
    ...,
] = (
    (7, AttributeError),
    ({"files": "not-a-list"}, SystemExit),
    ({"files": [7]}, AttributeError),
    ({"files": [{"role": []}]}, TypeError),
)


@pytest.mark.parametrize(
    "support,expected",
    [
        ({"title": ["A"], "sent_id": [1]}, [("A", 1)]),
        ([["A", 1], ["bad"]], [("A", 1)]),
        (None, []),
    ],
)
def test_supporting_fact_shapes(
    support: audit.SupportingFacts | list[list[str | int]] | None,
    expected: list[tuple[str, int]],
) -> None:
    assert audit._get_supporting_facts({"supporting_facts": support}) == expected


def test_graph_presence_adapter_preserves_scope() -> None:
    driver: MagicMock = MagicMock()
    session: MagicMock = driver.session.return_value.__enter__.return_value
    session.run.side_effect = [
        [{"chunk_count": 2, "has_embedding": True}],
        [{"entity_count": 3}],
        [],
        [],
    ]
    assert audit._check_chunks_in_graph(driver, "q", [0, 1], "unit") == {
        0: {"chunk_count": 2, "entity_count": 3, "has_embedding": True},
        1: {"chunk_count": 0, "entity_count": 0, "has_embedding": False},
    }
    for call in session.run.call_args_list:
        assert call.args[1]["kg_name"] == "unit"
        assert call.args[1]["qid"] == "q"
        assert "$kg_name" in call.args[0]


def test_vector_retrievability_and_failed_query() -> None:
    driver: MagicMock = MagicMock()
    session: MagicMock = driver.session.return_value.__enter__.return_value
    session.run.side_effect = [
        [{"eid": "target"}],
        [{"eid": "other", "score": 0.9}, {"eid": "target", "score": 0.8}],
    ]
    assert audit._check_chunk_retrievable_via_vector(
        driver,
        "q",
        0,
        [1.0, 0.0],
        "unit",
    ) == (True, 0.8)
    session.run.side_effect = [[{"eid": "target"}], RuntimeError("index unavailable")]
    assert audit._check_chunk_retrievable_via_vector(
        driver,
        "q",
        0,
        [1.0, 0.0],
        "unit",
    ) == (False, None)
    session.run.side_effect = [[]]
    assert audit._check_chunk_retrievable_via_vector(
        driver,
        "q",
        0,
        [1.0, 0.0],
        "unit",
    ) == (False, None)


@pytest.mark.parametrize(
    "vanilla,kg,label",
    [
        (True, True, "both_correct"),
        (True, False, "vanilla_only"),
        (False, True, "kg_only"),
        (False, False, "neither"),
    ],
)
def test_correctness_classes(vanilla: bool, kg: bool, label: str) -> None:
    assert (
        audit._correctness_class({"vanilla_correct": vanilla, "kg_correct": kg})
        == label
    )


def test_package_manifest_hashes_and_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(package, "REPO", tmp_path)
    payload: Path = tmp_path / "artifact.txt"
    payload.write_text("synthetic artifact")
    digest: str = hashlib.sha256(payload.read_bytes()).hexdigest()
    assert package.sha256(payload) == digest
    entries: list[package.ManifestFile] = [
        {
            "role": role,
            "path": "artifact.txt",
            "size": payload.stat().st_size,
            "sha256": digest,
        }
        for role in [
            "selection_ids",
            "headline_results",
            "analysis_output",
            "gps_replay",
        ]
    ]
    package.check_manifest_files({"files": entries})
    payload.write_text("corrupted artifact")
    with pytest.raises(SystemExit) as exc:
        package.check_manifest_files({"files": entries})
    assert exc.value.code == 1
    with pytest.raises(SystemExit):
        package.check_manifest_files({"files": []})
    assert (
        package.strip_latex_comments("text%comment\nprice \\% preserved")
        == "text\nprice \\% preserved"
    )


def test_package_manifest_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest: Path = tmp_path / "MANIFEST.json"
    monkeypatch.setattr(package, "REPO", tmp_path)
    monkeypatch.setattr(package, "MANIFEST", manifest)
    with pytest.raises(SystemExit):
        package.load_manifest()
    manifest.write_text("bad")
    with pytest.raises(SystemExit):
        package.load_manifest()
    manifest.write_text(json.dumps({"files": []}))
    assert package.load_manifest() == {"files": []}


@pytest.mark.parametrize(
    "payload",
    [None, True, 7, 1.5, "raw", [], [1, "entry"], {"files": "unvalidated"}],
)
def test_manifest_decoder_preserves_unvalidated_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: package.JSONValue,
) -> None:
    manifest: Path = tmp_path / "MANIFEST.json"
    monkeypatch.setattr(package, "REPO", tmp_path)
    monkeypatch.setattr(package, "MANIFEST", manifest)
    manifest.write_text(json.dumps(payload))

    decoded: package.JSONValue = package.load_manifest()

    assert decoded == payload
    assert type(decoded) is type(payload)


@pytest.mark.parametrize(
    "payload,expected_error",
    MALFORMED_MANIFEST_CASES,
)
def test_manifest_checker_preserves_malformed_shape_failures(
    payload: package.JSONValue,
    expected_error: type[AttributeError | SystemExit | TypeError],
) -> None:
    with pytest.raises(expected_error):
        package.check_manifest_files(payload)
