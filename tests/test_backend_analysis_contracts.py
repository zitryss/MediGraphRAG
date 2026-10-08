"""Numerical and local-artifact contracts for offline research scripts."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, TypedDict
from unittest.mock import Mock

import numpy as np
import pytest

if TYPE_CHECKING:
    from experiments.verbalized_confidence import ConfidenceDocument
    from experiments.gps_v2_replay import ReplayQuestion, ReplayStore
    from experiments.lockin_accounting_analysis import AccountingRow


class CorpusOptions(TypedDict):
    hotpot_file: Path
    output: Path
    selection_output: Path
    dataset_name: str
    num_samples: int | None
    subset_seed: int
    overwrite: bool


@pytest.mark.parametrize("name", ["gps_v2_paper_numbers", "gps_v3_depth_matched"])
def test_auroc_bootstrap_and_rejection_integral(
    analysis_import: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    module: ModuleType = analysis_import(name)
    monkeypatch.setattr(module, "B", 30)
    monkeypatch.setattr(module, "RNG", np.random.default_rng(9))
    assert module.auroc_ci([0, 1], [False, True]) == (1, 1, 1)
    assert module.auroc_ci([0, 1], [True, True]) == (None, None, None)
    assert module.aurec([True, False], [0, 1]) == pytest.approx(0.25)
    assert module.latest_artifact("nonexistent-unit-dataset") is None


@pytest.mark.parametrize(
    "store,expected,reason",
    [
        (None, 0.5, "missing_store"),
        ({"null_reason": "generation_failed"}, 0.5, "generation_failed"),
        ({"answers": []}, 0.5, "no_a_entities"),
        ({"answers": [{"source": "surface", "cos": 0, "L": 2}]}, 0.5, "no_a_entities"),
        ({"answers": [{"cos": 0.9, "L": 2}]}, 0, None),
        ({"answers": [{"cos": 0.9, "L": 3}]}, 0.6, None),
        ({"answers": [{"cos": 0.9, "L": None}]}, 1, None),
    ],
)
def test_depth_matched_score(
    analysis_import: Callable[[str], ModuleType],
    store: ReplayStore | None,
    expected: float,
    reason: str | None,
) -> None:
    module: ModuleType = analysis_import("gps_v3_depth_matched")
    result = module.score_from_store_depth_matched(store, 2)
    assert result["score"] == pytest.approx(expected)
    assert result["null_reason"] == reason


def test_depth_replay_files_and_null_denominators(
    analysis_import: Callable[[str], ModuleType],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module: ModuleType = analysis_import("gps_v3_depth_matched")
    monkeypatch.setattr(module, "REPO", str(tmp_path))
    monkeypatch.setattr(module, "RUNS", str(tmp_path))
    run: Path = tmp_path / "run"
    run.mkdir()
    (tmp_path / "result.json").write_text(
        json.dumps(
            {"config_results": [{"details": [{"question_id": "q1", "hop_count": 3}]}]}
        )
    )
    questions: list[ReplayQuestion] = [
        {
            "question_id": "q1",
            "kg_correct": True,
            "store": {"kg": {"answers": [{"cos": 1, "L": 3}]}},
        },
        {"question_id": "q2", "store": {"kg": {"null_reason": "no_a_entities"}}},
        {"question_id": "q3", "store": {"kg": {"null_reason": "generation_failed"}}},
        {"question_id": "q4"},
    ]
    (run / "gps_v2_replay_musique_2026.json").write_text(
        json.dumps({"source_result": "result.json", "questions": questions})
    )
    assert module.scores_for("musique") == {"q1": 0, "q2": None}
    assert module.scored_rows("musique") == [("q1", 0, True), ("q2", None, False)]
    with pytest.raises(FileNotFoundError, match="missing GPS"):
        module.load_artifact("absent")
    assert module.expected_hop("hotpotqa", None) == 2
    assert module.expected_hop("realmedqa", {"hop_count": 9}) == 1
    assert module.expected_hop("2wikimultihopqa", {"hop_count": 3}) == 3
    assert module.expected_hop("unknown", None) == 1


def test_trust_silent_failure_and_selective_risk(
    analysis_import: Callable[[str], ModuleType],
) -> None:
    module: ModuleType = analysis_import("trust_analysis")
    assert module.silent_rate([]) == (None, 0)
    rows = [
        {"correct": False, "dse": 0, "sd": 0},
        {"correct": False, "dse": None, "sd": 0},
        {"correct": True, "dse": 0, "sd": 0},
    ]
    assert module.silent_rate(rows) == (0.5, 2)
    assert module.selective_risk([0.9, 0.1], [False, True], (0.5, 1)) == {
        "50": 0,
        "100": 0.5,
    }
    assert module.auroc([0.1, 0.9], [False, True]) == 1
    assert module.auroc([0.1], [True]) is None


@pytest.mark.parametrize(
    "word,expected",
    [("correct", True), ("INCORRECT", False), ("unclear", None), (None, None)],
)
def test_independent_judge_parsing_and_retry(
    analysis_import: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    word: str | None,
    expected: bool | None,
) -> None:
    module: ModuleType = analysis_import("rejudge_independent")
    client: Mock = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=word))]
    )
    assert (
        module.judge_one(client, "question", "expected", "answer", retries=2)
        is expected
    )
    assert client.chat.completions.create.call_args.kwargs["temperature"] == 0
    client.chat.completions.create.side_effect = RuntimeError("timeout")
    sleep: Mock = Mock()
    monkeypatch.setattr(module.time, "sleep", sleep)
    assert module.judge_one(client, "q", "a", "b", retries=2) is None
    assert [c.args[0] for c in sleep.call_args_list] == [2, 4]


@pytest.mark.parametrize(
    "text,expected", [("0.75", 0.75), ("80%", 0.8), ("101", None), ("", None), ("0", 0)]
)
def test_verbal_confidence_parser(
    analysis_import: Callable[[str], ModuleType], text: str, expected: float | None
) -> None:
    module: ModuleType = analysis_import("verbalized_confidence")
    client: Mock = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )
    assert module.ask_conf(client, "question", "response", retries=1) == expected


def test_certificate_mask_and_artifact(
    analysis_import: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module: ModuleType = analysis_import("certificate_eval")
    rows = [
        (0.1, 0.2, 0.3, True),
        (0.2, 0.3, 0.4, True),
        (0.8, 0.9, 0.9, False),
        (0.9, 0.9, None, False),
    ]
    assert module.certificate_mask(rows, 0.5).tolist() == [True, True, False, False]
    monkeypatch.setattr(module, "load_all", lambda: {"unit": rows})
    monkeypatch.setattr(module, "REPO", str(tmp_path))
    monkeypatch.setattr(module, "B", 20)
    monkeypatch.setattr(module, "RNG", np.random.default_rng(4))
    (tmp_path / "results/analyses").mkdir(parents=True)
    module.main()
    result = json.loads(
        (tmp_path / "results/analyses/certificate_eval.json").read_text()
    )
    assert result["pooled_certificate"] == {
        "coverage": 0.5,
        "precision": 1,
        "selected": 2,
        "n": 4,
    }
    assert result["baseline_sd_only"]["precision"] == 1
    assert result["split_half"]["B"] == 20


def test_graph_diversity_analysis_artifacts(
    analysis_import: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module: ModuleType = analysis_import("graph_state_diversity_analysis")
    details = [
        {
            "question_id": str(i),
            "kg_correct": bool(i % 2),
            "kg_subgraph_entropy": float(i),
            "kg_discrete_semantic_entropy": i / 2,
            "kg_graph_state_sample_count": 2,
        }
        for i in range(6)
    ]
    details += [{"question_id": "absent"}]
    monkeypatch.setattr(
        module, "_iter_result_details", lambda: iter([("unit", "adaptive", details)])
    )
    monkeypatch.setattr(module, "FIG_DIR", str(tmp_path / "figures"))
    monkeypatch.setattr(module, "OUT_JSON", str(tmp_path / "summary.json"))
    assert module.plot([]) == []
    module.main()
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["row_count"] == 6
    assert result["rows_by_dataset_policy"] == {"unit:adaptive": 6}
    assert {Path(p).suffix for p in result["figures"]} == {".png", ".pdf"}
    assert all(Path(p).stat().st_size > 100 for p in result["figures"])


def test_fullwiki_corpus_deduplicates_and_persists_selection(
    analysis_import: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from experiments import subset_selection

    monkeypatch.setitem(sys.modules, "subset_selection", subset_selection)
    module: ModuleType = analysis_import("prepare_hotpotqa_fullwiki_corpus")
    path: Path = tmp_path / "raw.json"
    path.write_text(
        json.dumps(
            [
                {
                    "_id": "q1",
                    "context": [["Title", ["one", "two"]], ["", ""], ["bad"]],
                },
                {"id": "q2", "context": [["Title", ["one", "two"]]]},
                "ignored",
            ]
        )
    )
    opts: CorpusOptions = {
        "hotpot_file": path,
        "output": tmp_path / "corpus.jsonl",
        "selection_output": tmp_path / "selection.json",
        "dataset_name": "hotpotqa_fullwiki",
        "num_samples": None,
        "subset_seed": 42,
        "overwrite": False,
    }
    result = module.prepare_corpus(**opts)
    assert result["selected_questions"] == 2
    assert result["corpus_documents"] == 1
    doc = json.loads(opts["output"].read_text())
    assert doc["text"] == "Title. one two"
    assert doc["source_question_ids"] == ["q1", "q2"]
    assert json.loads(opts["selection_output"].read_text())["question_ids"] == [
        "q1",
        "q2",
    ]
    with pytest.raises(FileExistsError):
        module.prepare_corpus(**opts)
    path.write_text("{}")
    with pytest.raises(ValueError, match="JSON list"):
        module._load_hotpot(path)


@pytest.mark.parametrize(
    "changes,bucket",
    [
        ({"correct": True}, "correct"),
        ({"dse": 1}, "wrong_not_silent"),
        ({"route": "empty"}, "silent_empty_route"),
        ({"route": "", "seu": 1}, "silent_route_unknown_flagged"),
        ({"route": ""}, "silent_route_unknown_calm"),
        ({"seu": 1}, "silent_populated_seu_contradiction"),
        ({"gps": None}, "silent_populated_gps_weak_or_abstained"),
        ({}, "silent_all_families_calm"),
    ],
)
def test_lockin_partition(
    analysis_import: Callable[[str], ModuleType], changes: AccountingRow, bucket: str
) -> None:
    module: ModuleType = analysis_import("lockin_accounting_analysis")
    row = {
        "correct": False,
        "dse": 0,
        "sd": 0,
        "seu": 0,
        "gps": 0,
        "route": "entity_first",
        "policy": "kg",
        **changes,
    }
    assert module.silent_bucket(row) == bucket
    result = module.account([row])
    assert result["answered"] == 1
    assert result["buckets"] == {bucket: 1}
    assert module.account([])["silent_upper_bound_rate"] is None
    assert not module.finite(float("nan"))
    assert not module.gps_flags({"policy": "vanilla"})


def test_diversity_analysis_numeric_output(
    analysis_import: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module: ModuleType = analysis_import("diversity_run_analysis")
    path: Path = tmp_path / "results.json"
    rows = [
        {
            "kg_graph_state_sample_count": 2,
            "kg_correct": False,
            "kg_sd_uq": i / 10,
            "kg_chunk_jaccard": 1 - i / 10,
        }
        for i in range(6)
    ]
    rows += [{"kg_generation_failed": True}, {"kg_graph_state_sample_count": 1}]
    path.write_text(
        json.dumps(
            {
                "config_results": [
                    {"config": {"name": "kg_entity_first"}, "details": rows}
                ]
            }
        )
    )
    monkeypatch.setattr(module, "OUT", str(tmp_path / "analysis.json"))
    module.main(str(path))
    result = json.loads((tmp_path / "analysis.json").read_text())
    assert result["n_answered"] == 7
    assert result["n_with_traces"] == 6
    assert result["stability_mean_jaccard"]["chunk"]["mean"] == pytest.approx(0.75)
    assert result["coupling_overlap_vs_sduq_amongwrong"]["chunk"][
        "spearman_rho"
    ] == pytest.approx(-1)
    assert not module.fin(None)
    with pytest.raises(SystemExit, match="no run JSON"):
        module.main(str(tmp_path / "absent*.json"))


def test_concentration_points_and_bootstrap(
    analysis_import: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "make_figures", ModuleType("make_figures"))
    module: ModuleType = analysis_import("make_concentration_figure")
    rows = [
        {"kg_retrieval_overlap": 0.5, "kg_sd_uq": 0, "kg_correct": False},
        {"kg_retrieval_overlap": 0.8, "kg_sd_uq": 0.1, "kg_correct": True},
        {},
    ]
    ow, sw, oc, sc = module.points(rows, "kg")
    assert ow.tolist() == [0.5]
    assert sw.tolist() == [-12]
    assert oc.tolist() == [0.8]
    assert sc[0] == pytest.approx(-1)
    assert module.binned_means([], []) == ([], [], [])
    monkeypatch.setattr(module, "RNG", np.random.default_rng(1))
    mids, means, cis = module.binned_means(np.arange(16), np.ones(16), boot=10)
    assert len(mids) == 4
    assert means == [1] * 4
    assert cis == [(1, 1)] * 4


def test_anchor_mask_candidate_filters_and_scoping(
    analysis_import: Callable[[str], ModuleType],
    experiment_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "experiments.experiment", experiment_module)
    module: ModuleType = analysis_import("anchor_masking_intervention")
    base = {
        "question_id": "q",
        "kg_sd_uq": 0,
        "kg_support_entailment_uncertainty": 0.9,
        "kg_retrieval_route": "entity_first",
    }
    rows = [
        {**base, "kg_generation_failed": True},
        {},
        {**base, "kg_sd_uq": 1},
        {**base, "kg_correct": True},
        {**base, "kg_retrieval_route": "entity_first_empty"},
        base,
    ]
    monkeypatch.setattr(module, "_result_details", lambda *a: rows)
    selected = module.select_candidates(
        "hotpotqa",
        "kg",
        sd_threshold=0.1,
        seu_threshold=0.5,
        wrong_only=True,
        require_populated_route=True,
        limit=1,
    )
    assert [r["question_id"] for r in selected] == ["q"]
    assert module._question_id_for_scope("hotpotqa", "q") == "q"
    assert module._question_id_for_scope("realmedqa", "q") is None
    factory: Mock = Mock()
    monkeypatch.setattr(module, "EnhancedRAGSystem", factory)
    module._make_kg_system("strict")
    assert factory.call_args.kwargs["allow_vector_fallback"] is False


def test_sensitivity_replay_skips_unusable_and_scores(
    analysis_import: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    module: ModuleType = analysis_import("gps_sensitivity")
    questions: list[ReplayQuestion] = [
        {
            "question_id": "a",
            "kg_correct": True,
            "store": {"kg": {"answers": [{"cos": 1, "L": 2}]}},
        },
        {
            "question_id": "b",
            "kg_correct": False,
            "store": {"kg": {"answers": [{"cos": 1, "L": None}]}},
        },
        {"question_id": "c", "store": {"kg": {"null_reason": "generation_failed"}}},
        {"question_id": "d"},
    ]
    monkeypatch.setattr(
        module,
        "load_artifact",
        lambda *a, **kw: {"source_result": "unit", "questions": questions},
    )
    monkeypatch.setattr(module, "row_map", lambda *a: {})
    assert module.auroc_at("hotpotqa", False, 0.6, 0.4) == (1, 2)
    questions[1]["kg_correct"] = True
    assert module.auroc_at("hotpotqa", False, 0.6, 0.4) == (None, 2)


def test_wave1_wilson_and_silent_counts(
    analysis_import: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    module: ModuleType = analysis_import("wave1_analysis")
    assert module.wilson_ci(0, 0) is None
    lo, hi = module.wilson_ci(5, 10)
    assert lo == pytest.approx(1 - hi)
    assert 0.2 < lo < 0.3
    monkeypatch.setattr(module, "KEY", {"unit": "Unit"})
    monkeypatch.setattr(module, "STRICT_RUNS", {})
    monkeypatch.setattr(
        module,
        "rows_for",
        lambda *a: [
            {"correct": False, "dse": 0, "sd": 0},
            {"correct": False, "dse": 1, "sd": 0.5},
            {"correct": True, "dse": 0, "sd": 0},
        ],
    )
    result = module.silent_cis()
    assert result["unit"]["kg"]["rate"] == 0.5
    assert result["pooled"]["kg"]["n"] == 2


def test_reporting_odds_ratio_and_answered_denominators(
    analysis_import: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module: ModuleType = analysis_import("reporting_robustness")
    assert module.odds_ratio_mh({}) is None
    assert module.odds_ratio_mh(
        {
            "unit": {
                "cert_correct": 4,
                "cert_wrong": 1,
                "uncert_correct": 2,
                "uncert_wrong": 3,
            }
        }
    ) == pytest.approx(6)
    monkeypatch.setattr(module, "REPO", str(tmp_path))
    monkeypatch.setattr(module, "KEY", {"realmedqa": "Unit"})
    monkeypatch.setattr(module, "MANIFEST", {"Unit": {"result_path": "result.json"}})
    monkeypatch.setattr(module, "STRICT_RUNS", {"realmedqa": "result.json"})
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "config_results": [
                    {
                        "config": {"name": "kg_entity_first"},
                        "details": [
                            {"question_id": "q1"},
                            {"question_id": "q2", "kg_generation_failed": True},
                        ],
                    }
                ]
            }
        )
    )
    assert module.answered_counts()["realmedqa"]["kg"] == {
        "total": 2,
        "answered": 1,
        "failures": 1,
        "failure_rate": 0.5,
    }
    assert module.realmedqa_adaptive_strict_complete_case() == {
        "adaptive_answered": 1,
        "strict_answered": 1,
        "both_answered": 1,
    }


def test_rejudge_metric_denominators(
    analysis_import: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    module: ModuleType = analysis_import("rejudge_wave1")
    rows = [{"qid": str(i), "sd": i / 12, "seu": i / 12} for i in range(12)]
    monkeypatch.setattr(module, "rows_for", lambda *a: rows)
    monkeypatch.setattr(
        module, "gps_scores", lambda *a, **kw: {str(i): i / 12 for i in range(12)}
    )
    result = module.metric_aurocs("unit", "kg", {str(i): i < 6 for i in range(12)})
    assert result == {"sd_uq": (1, 12), "seu": (1, 12), "gps": (1, 12)}
    assert module.metric_aurocs("unit", "kg", {})["gps"] == (None, 0)


def test_residue_audit_filter_and_artifact(
    analysis_import: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module: ModuleType = analysis_import("residue_audit")
    monkeypatch.setattr(module, "REPO", str(tmp_path))
    monkeypatch.setattr(module, "KEY", {"unit": "Unit"})
    monkeypatch.setattr(module, "MANIFEST", {"Unit": {"result_path": "result.json"}})
    monkeypatch.setattr(module, "SLUGS", ["unit"])
    monkeypatch.setattr(module, "OUT_PATH", str(tmp_path / "out.json"))
    monkeypatch.setattr(module, "gps_cache", lambda *a: {"q1": 0, "q2": 0})
    base = {
        "question_id": "q1",
        "question": "q",
        "expected": "a",
        "kg_response": "b",
        "kg_discrete_semantic_entropy": 0,
        "kg_sd_uq": 0,
        "kg_support_entailment_uncertainty": 0.2,
        "kg_retrieval_route": "entity_first",
    }
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "config_results": [
                    {
                        "config": {"name": "kg_entity_first"},
                        "details": [
                            base,
                            {**base, "question_id": "q2", "kg_correct": True},
                            {**base, "kg_generation_failed": True},
                        ],
                    }
                ]
            }
        )
    )
    module.main()
    result = json.loads((tmp_path / "out.json").read_text())
    assert result["count"] == 1
    assert result["cases"][0]["response"] == "b"
    assert "cannot be confirmed" in result["caveat"]


def test_migration_figure_artifacts(
    analysis_import: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(sys.modules, "make_figures", ModuleType("make_figures"))
    module: ModuleType = analysis_import("make_migration_figure")
    monkeypatch.setattr(module, "OUT", str(tmp_path))
    monkeypatch.setattr(
        module,
        "rows_for",
        lambda *a: [
            {"qid": "q1", "correct": False, "sd": 0, "seu": 0.2},
            {"qid": "q2", "correct": False, "sd": 0.001, "seu": 0.8},
        ],
    )
    assert module.xy({"sd": 0, "seu": 0.2}) == (-12, 0.2)
    module.main()
    assert (tmp_path / "lockin_migration.pdf").read_bytes().startswith(b"%PDF")
    assert (tmp_path / "lockin_migration.png").read_bytes().startswith(b"\x89PNG")
    module.plt.close("all")


@pytest.mark.parametrize(
    "policy,expected",
    [
        ("kg", [("kept", "question", "kg answer", True)]),
        ("vanilla", [("kept", "question", "dense answer", False)]),
    ],
)
def test_verbal_confidence_load_details_selects_policy_and_excludes_failures(
    analysis_import: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    policy: str,
    expected: list[tuple[str, str, str, bool]],
) -> None:
    module: ModuleType = analysis_import("verbalized_confidence")
    payload: ConfidenceDocument = {
        "config_results": [
            {
                "config": {"name": "kg_entity_first_unit"},
                "details": [
                    {
                        "question_id": "kept",
                        "question": "question",
                        "kg_response": "kg answer",
                        "kg_correct": True,
                        "vanilla_generation_failed": True,
                    },
                    {
                        "question_id": "failed",
                        "question": "question",
                        "kg_generation_failed": True,
                    },
                ],
            },
            {
                "config": {"name": "dense_floor_unit"},
                "details": [
                    {
                        "question_id": "kept",
                        "question": "question",
                        "vanilla_response": "dense answer",
                        "vanilla_correct": False,
                        "kg_generation_failed": True,
                    },
                    {
                        "question_id": "failed",
                        "question": "question",
                        "vanilla_generation_failed": True,
                    },
                ],
            },
        ],
    }
    (tmp_path / "result.json").write_text(json.dumps(payload))
    monkeypatch.setattr(module, "REPO", str(tmp_path))
    monkeypatch.setattr(module, "KEY", {"unit": "Unit"})
    monkeypatch.setattr(module, "MANIFEST", {"Unit": {"result_path": "result.json"}})
    assert module.load_details("unit", policy) == expected
