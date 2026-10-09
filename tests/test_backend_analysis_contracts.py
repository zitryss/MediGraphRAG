"""Numerical and local-artifact contracts for offline research scripts."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypedDict, cast
from unittest.mock import Mock

import numpy as np
import pytest

if TYPE_CHECKING:
    from openai import OpenAI

    from experiments.gps_v2_replay import ReplayQuestion, ReplayStore
    from experiments.lockin_accounting_analysis import AccountingRow
    from experiments.verbalized_confidence import ConfidenceDocument


if TYPE_CHECKING:
    from collections.abc import Iterable as gps_v3_depth_matched_Iterable
    from collections.abc import Iterable as lockin_accounting_analysis_Iterable
    from collections.abc import Iterator as make_concentration_figure_Iterator
    from collections.abc import Mapping as rejudge_wave1_Mapping
    from collections.abc import Mapping as reporting_robustness_Mapping
    from collections.abc import Sequence as certificate_eval_Sequence
    from collections.abc import Sequence as make_concentration_figure_Sequence
    from collections.abc import Sequence as trust_analysis_Sequence
    from pathlib import Path as prepare_hotpotqa_fullwiki_corpus_Path
    from typing import TypeGuard as lockin_accounting_analysis_TypeGuard

    import numpy as certificate_eval_np
    import numpy as diversity_run_analysis_np
    import numpy as make_concentration_figure_np
    import numpy as make_migration_figure_np
    import numpy as trust_analysis_np
    from numpy.typing import ArrayLike as trust_analysis_ArrayLike
    from numpy.typing import NDArray as certificate_eval_NDArray
    from numpy.typing import NDArray as make_concentration_figure_NDArray
    from numpy.typing import NDArray as trust_analysis_NDArray

    from experiments.anchor_masking_intervention import (
        InterventionCandidate as anchor_masking_intervention_InterventionCandidate,
    )
    from experiments.certificate_eval import (
        CertificateRow as certificate_eval_CertificateRow,
    )
    from experiments.gps_v2_replay import (
        ReplayArtifact as gps_v3_depth_matched_ReplayArtifact,
    )
    from experiments.gps_v2_replay import (
        ReplayScore as gps_v3_depth_matched_ReplayScore,
    )
    from experiments.gps_v2_replay import (
        ReplayStore as gps_v3_depth_matched_ReplayStore,
    )
    from experiments.gps_v3_depth_matched import HopRow as gps_v3_depth_matched_HopRow
    from experiments.graph_state_diversity_analysis import (
        DiversityRow as graph_state_diversity_analysis_DiversityRow,
    )
    from experiments.lockin_accounting_analysis import (
        AccountingRow as lockin_accounting_analysis_AccountingRow,
    )
    from experiments.lockin_accounting_analysis import (
        AccountingSummary as lockin_accounting_analysis_AccountingSummary,
    )
    from experiments.make_concentration_figure import (
        BinnedMeans as make_concentration_figure_BinnedMeans,
    )
    from experiments.make_concentration_figure import (
        ConcentrationDetail as make_concentration_figure_ConcentrationDetail,
    )
    from experiments.make_migration_figure import MigrationPoint
    from experiments.reporting_robustness import (
        AnsweredCounts as reporting_robustness_AnsweredCounts,
    )
    from experiments.reporting_robustness import (
        CompleteCase as reporting_robustness_CompleteCase,
    )
    from experiments.reporting_robustness import (
        ContingencyTable as reporting_robustness_ContingencyTable,
    )
    from experiments.trust_analysis import SilentRateRow
    from experiments.verbalized_confidence import (
        ConfidenceRow as verbalized_confidence_ConfidenceRow,
    )
    from experiments.wave1_analysis import SilentCI as wave1_analysis_SilentCI
    from ontographrag.rag.systems.enhanced_rag_system import (
        EnhancedRAGSystem as anchor_masking_intervention_EnhancedRAGSystem,
    )


class BootstrapExports(Protocol):
    @staticmethod
    def aurec(
        correct: gps_v3_depth_matched_Iterable[bool],
        uncertainty: gps_v3_depth_matched_Iterable[float],
    ) -> float | None: ...

    @staticmethod
    def auroc_ci(
        scores: gps_v3_depth_matched_Iterable[float],
        wrong: gps_v3_depth_matched_Iterable[bool],
    ) -> tuple[float | None, float | None, float | None]: ...

    @staticmethod
    def latest_artifact(slug: str, strict: bool = ...) -> str | None: ...


class GpsV3DepthMatchedExports(Protocol):
    @staticmethod
    def aurec(
        correct: gps_v3_depth_matched_Iterable[bool],
        uncertainty: gps_v3_depth_matched_Iterable[float],
    ) -> float | None: ...

    @staticmethod
    def auroc_ci(
        scores: gps_v3_depth_matched_Iterable[float],
        wrong: gps_v3_depth_matched_Iterable[bool],
    ) -> tuple[float | None, float | None, float | None]: ...

    @staticmethod
    def expected_hop(slug: str, row: gps_v3_depth_matched_HopRow | None) -> int: ...

    @staticmethod
    def latest_artifact(slug: str, strict: bool = ...) -> str | None: ...

    @staticmethod
    def load_artifact(
        slug: str,
        strict: bool = ...,
    ) -> gps_v3_depth_matched_ReplayArtifact: ...

    @staticmethod
    def score_from_store_depth_matched(
        store_row: gps_v3_depth_matched_ReplayStore | None,
        expected_depth: int,
        tau: float = ...,
        gamma: float = ...,
    ) -> gps_v3_depth_matched_ReplayScore: ...

    @staticmethod
    def scored_rows(
        slug: str,
        strict: bool = ...,
        side: str = ...,
    ) -> list[tuple[str, float | None, bool]]: ...

    @staticmethod
    def scores_for(
        slug: str,
        strict: bool = ...,
        side: str = ...,
    ) -> dict[str, float | None]: ...


class TrustAnalysisExports(Protocol):
    @staticmethod
    def auroc(
        scores: trust_analysis_ArrayLike,
        wrong: trust_analysis_Sequence[bool]
        | trust_analysis_NDArray[trust_analysis_np.int64],
    ) -> float | None: ...

    @staticmethod
    def selective_risk(
        scores: trust_analysis_ArrayLike,
        correct: trust_analysis_ArrayLike,
        coverages: trust_analysis_Sequence[float] = ...,
    ) -> dict[str, float]: ...

    @staticmethod
    def silent_rate(
        rows: trust_analysis_Sequence[SilentRateRow],
    ) -> tuple[float | None, int]: ...


class RejudgeIndependentExports(Protocol):
    @staticmethod
    def judge_one(
        client: OpenAI,
        q: str,
        expected: str,
        response: str,
        retries: int = ...,
    ) -> bool | None: ...

    time: ModuleType


class VerbalizedConfidenceExports(Protocol):
    @staticmethod
    def ask_conf(
        client: OpenAI,
        question: str,
        response: str,
        retries: int = ...,
    ) -> float | None: ...

    @staticmethod
    def load_details(
        slug: str,
        policy: str,
    ) -> list[verbalized_confidence_ConfidenceRow]: ...


class CertificateEvalExports(Protocol):
    @staticmethod
    def certificate_mask(
        rows: certificate_eval_Sequence[certificate_eval_CertificateRow],
        sd_thresh: float,
    ) -> certificate_eval_NDArray[
        certificate_eval_np.bool_ | certificate_eval_np.float64
    ]: ...

    @staticmethod
    def main() -> None: ...


class GraphStateDiversityAnalysisExports(Protocol):
    @staticmethod
    def main() -> None: ...

    @staticmethod
    def plot(rows: list[graph_state_diversity_analysis_DiversityRow]) -> list[str]: ...


class PrepareHotpotqaFullwikiCorpusExports(Protocol):
    @staticmethod
    def _load_hotpot(
        path: prepare_hotpotqa_fullwiki_corpus_Path,
    ) -> list[dict[str, RawJSON]]: ...

    @staticmethod
    def prepare_corpus(
        *,
        hotpot_file: prepare_hotpotqa_fullwiki_corpus_Path,
        output: prepare_hotpotqa_fullwiki_corpus_Path,
        selection_output: prepare_hotpotqa_fullwiki_corpus_Path,
        dataset_name: str,
        num_samples: int | None,
        subset_seed: int,
        overwrite: bool,
    ) -> CorpusSummary: ...


class LockinAccountingAnalysisExports(Protocol):
    @staticmethod
    def account(
        rows: lockin_accounting_analysis_Iterable[
            lockin_accounting_analysis_AccountingRow
        ],
    ) -> lockin_accounting_analysis_AccountingSummary: ...

    @staticmethod
    def finite(value: object) -> lockin_accounting_analysis_TypeGuard[int | float]: ...

    @staticmethod
    def gps_flags(row: lockin_accounting_analysis_AccountingRow) -> bool: ...

    @staticmethod
    def silent_bucket(row: lockin_accounting_analysis_AccountingRow) -> str: ...


class DiversityRunAnalysisExports(Protocol):
    @staticmethod
    def fin(x: float | str | None) -> bool | diversity_run_analysis_np.bool_: ...

    @staticmethod
    def main(pattern: str) -> None: ...


class MakeConcentrationFigureExports(Protocol):
    @staticmethod
    def binned_means(
        x: make_concentration_figure_NDArray[
            make_concentration_figure_np.float64 | make_concentration_figure_np.int64
        ]
        | list[float],
        y: make_concentration_figure_NDArray[
            make_concentration_figure_np.float64 | make_concentration_figure_np.int64
        ]
        | list[float],
        n_bins: int = ...,
        boot: int = ...,
    ) -> make_concentration_figure_BinnedMeans: ...

    @staticmethod
    def points(
        rows: make_concentration_figure_Sequence[
            make_concentration_figure_ConcentrationDetail
        ],
        pre: str,
    ) -> make_concentration_figure_Iterator[
        make_concentration_figure_NDArray[make_concentration_figure_np.float64]
    ]: ...


class AnchorMaskingInterventionExports(Protocol):
    @staticmethod
    def _make_kg_system(
        policy: str,
    ) -> anchor_masking_intervention_EnhancedRAGSystem: ...

    @staticmethod
    def _question_id_for_scope(dataset: str, qid: str) -> str | None: ...

    @staticmethod
    def select_candidates(
        dataset: str,
        policy: str,
        *,
        sd_threshold: float,
        seu_threshold: float,
        wrong_only: bool,
        require_populated_route: bool,
        limit: int,
    ) -> list[anchor_masking_intervention_InterventionCandidate]: ...


class GpsSensitivityExports(Protocol):
    @staticmethod
    def auroc_at(
        slug: str,
        strict: bool,
        tau: float,
        gamma: float,
    ) -> tuple[float | None, int]: ...


class Wave1AnalysisExports(Protocol):
    @staticmethod
    def silent_cis() -> dict[str, dict[str, wave1_analysis_SilentCI]]: ...

    @staticmethod
    def wilson_ci(k: int, n: int, z: float = ...) -> list[float] | None: ...


class ReportingRobustnessExports(Protocol):
    @staticmethod
    def answered_counts() -> dict[
        str,
        dict[str, reporting_robustness_AnsweredCounts],
    ]: ...

    @staticmethod
    def odds_ratio_mh(
        tables: reporting_robustness_Mapping[
            str,
            reporting_robustness_ContingencyTable,
        ],
    ) -> float | None: ...

    @staticmethod
    def realmedqa_adaptive_strict_complete_case() -> (
        reporting_robustness_CompleteCase
    ): ...


class RejudgeWave1Exports(Protocol):
    @staticmethod
    def metric_aurocs(
        slug: str,
        policy: str,
        labels_by_qid: rejudge_wave1_Mapping[str, bool | None],
    ) -> dict[str, tuple[float | None, int]]: ...


class ResidueAuditExports(Protocol):
    @staticmethod
    def main() -> None: ...


class MakeMigrationFigureExports(Protocol):
    @staticmethod
    def main() -> None: ...

    plt: ModuleType

    @staticmethod
    def xy(
        r: MigrationPoint,
    ) -> tuple[make_migration_figure_np.float64, float]: ...


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from experiments.anchor_masking_intervention import (
        CandidateDetail,
        InterventionCandidate,
    )
    from experiments.certificate_eval import CertificateReport, CertificateRow
    from experiments.diversity_run_analysis import TraceDetail, TraceReport
    from experiments.gps_v2_replay import ReplayArtifact
    from experiments.graph_state_diversity_analysis import (
        DiversityDetail,
        DiversityReport,
    )
    from experiments.lockin_accounting_analysis import AccountingSummary
    from experiments.make_concentration_figure import ConcentrationDetail
    from experiments.residue_audit import ResidueDetail, ResidueReport
    from experiments.wave1_analysis import SilentCI


class SilentRow(TypedDict):
    correct: bool
    dse: float | None
    sd: float


class RejudgeRow(TypedDict):
    qid: str
    sd: float
    seu: float


class CorpusSummary(TypedDict):
    dataset: str
    hotpot_file: str
    output: str
    selection_output: str
    available_questions: int
    selected_questions: int
    corpus_documents: int


class CorpusDocument(TypedDict):
    text: str
    source_question_ids: list[str]


class MigrationRow(TypedDict):
    qid: str
    correct: bool
    sd: float
    seu: float


RawJSON: TypeAlias = (
    None | bool | int | float | str | list["RawJSON"] | dict[str, "RawJSON"]
)


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
    module: BootstrapExports = cast(BootstrapExports, analysis_import(name))
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
    module: GpsV3DepthMatchedExports = cast(
        GpsV3DepthMatchedExports,
        analysis_import("gps_v3_depth_matched"),
    )
    result: gps_v3_depth_matched_ReplayScore = module.score_from_store_depth_matched(
        store,
        2,
    )
    assert result["score"] == pytest.approx(expected)
    assert result["null_reason"] == reason


def test_depth_replay_files_and_null_denominators(
    analysis_import: Callable[[str], ModuleType],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module: GpsV3DepthMatchedExports = cast(
        GpsV3DepthMatchedExports,
        analysis_import("gps_v3_depth_matched"),
    )
    monkeypatch.setattr(module, "REPO", str(tmp_path))
    monkeypatch.setattr(module, "RUNS", str(tmp_path))
    run: Path = tmp_path / "run"
    run.mkdir()
    (tmp_path / "result.json").write_text(
        json.dumps(
            {"config_results": [{"details": [{"question_id": "q1", "hop_count": 3}]}]},
        ),
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
        json.dumps({"source_result": "result.json", "questions": questions}),
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
    module: TrustAnalysisExports = cast(
        TrustAnalysisExports,
        analysis_import("trust_analysis"),
    )
    assert module.silent_rate([]) == (None, 0)
    rows: list[SilentRow] = [
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
    module: RejudgeIndependentExports = cast(
        RejudgeIndependentExports,
        analysis_import("rejudge_independent"),
    )
    client: Mock = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=word))],
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
    "text,expected",
    [("0.75", 0.75), ("80%", 0.8), ("101", None), ("", None), ("0", 0)],
)
def test_verbal_confidence_parser(
    analysis_import: Callable[[str], ModuleType],
    text: str,
    expected: float | None,
) -> None:
    module: VerbalizedConfidenceExports = cast(
        VerbalizedConfidenceExports,
        analysis_import("verbalized_confidence"),
    )
    client: Mock = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
    )
    assert module.ask_conf(client, "question", "response", retries=1) == expected


def test_certificate_mask_and_artifact(
    analysis_import: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module: CertificateEvalExports = cast(
        CertificateEvalExports,
        analysis_import("certificate_eval"),
    )
    rows: list[CertificateRow] = [
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
    result: CertificateReport = json.loads(
        (tmp_path / "results/analyses/certificate_eval.json").read_text(),
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
    module: GraphStateDiversityAnalysisExports = cast(
        GraphStateDiversityAnalysisExports,
        analysis_import("graph_state_diversity_analysis"),
    )
    details: list[DiversityDetail] = [
        {
            "question_id": str(i),
            "kg_correct": bool(i % 2),
            "kg_subgraph_entropy": float(i),
            "kg_discrete_semantic_entropy": i / 2,
            "kg_graph_state_sample_count": 2,
        }
        for i in range(6)
    ]
    absent_details: list[DiversityDetail] = [{"question_id": "absent"}]
    details += absent_details
    monkeypatch.setattr(
        module,
        "_iter_result_details",
        lambda: iter([("unit", "adaptive", details)]),
    )
    monkeypatch.setattr(module, "FIG_DIR", str(tmp_path / "figures"))
    monkeypatch.setattr(module, "OUT_JSON", str(tmp_path / "summary.json"))
    assert module.plot([]) == []
    module.main()
    result: DiversityReport = json.loads((tmp_path / "summary.json").read_text())
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
    module: PrepareHotpotqaFullwikiCorpusExports = cast(
        PrepareHotpotqaFullwikiCorpusExports,
        analysis_import("prepare_hotpotqa_fullwiki_corpus"),
    )
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
            ],
        ),
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
    result: CorpusSummary = module.prepare_corpus(**opts)
    assert result["selected_questions"] == 2
    assert result["corpus_documents"] == 1
    doc: CorpusDocument = json.loads(opts["output"].read_text())
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
    analysis_import: Callable[[str], ModuleType],
    changes: AccountingRow,
    bucket: str,
) -> None:
    module: LockinAccountingAnalysisExports = cast(
        LockinAccountingAnalysisExports,
        analysis_import("lockin_accounting_analysis"),
    )
    row: AccountingRow = {
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
    result: AccountingSummary = module.account([row])
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
    module: DiversityRunAnalysisExports = cast(
        DiversityRunAnalysisExports,
        analysis_import("diversity_run_analysis"),
    )
    path: Path = tmp_path / "results.json"
    rows: list[TraceDetail] = [
        {
            "kg_graph_state_sample_count": 2,
            "kg_correct": False,
            "kg_sd_uq": i / 10,
            "kg_chunk_jaccard": 1 - i / 10,
        }
        for i in range(6)
    ]
    unavailable_rows: list[TraceDetail] = [
        {"kg_generation_failed": True},
        {"kg_graph_state_sample_count": 1},
    ]
    rows += unavailable_rows
    path.write_text(
        json.dumps(
            {
                "config_results": [
                    {"config": {"name": "kg_entity_first"}, "details": rows},
                ],
            },
        ),
    )
    monkeypatch.setattr(module, "OUT", str(tmp_path / "analysis.json"))
    module.main(str(path))
    result: TraceReport = json.loads((tmp_path / "analysis.json").read_text())
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
    analysis_import: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "make_figures", ModuleType("make_figures"))
    module: MakeConcentrationFigureExports = cast(
        MakeConcentrationFigureExports,
        analysis_import("make_concentration_figure"),
    )
    rows: list[ConcentrationDetail] = [
        {"kg_retrieval_overlap": 0.5, "kg_sd_uq": 0, "kg_correct": False},
        {"kg_retrieval_overlap": 0.8, "kg_sd_uq": 0.1, "kg_correct": True},
        {},
    ]
    ow: NDArray[np.float64]
    sw: NDArray[np.float64]
    oc: NDArray[np.float64]
    sc: NDArray[np.float64]
    ow, sw, oc, sc = module.points(rows, "kg")
    assert ow.tolist() == [0.5]
    assert sw.tolist() == [-12]
    assert oc.tolist() == [0.8]
    assert sc[0] == pytest.approx(-1)
    assert module.binned_means([], []) == ([], [], [])
    monkeypatch.setattr(module, "RNG", np.random.default_rng(1))
    mids: list[float]
    means: list[float]
    cis: list[tuple[float, float]]
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
    module: AnchorMaskingInterventionExports = cast(
        AnchorMaskingInterventionExports,
        analysis_import("anchor_masking_intervention"),
    )
    base: CandidateDetail = {
        "question_id": "q",
        "kg_sd_uq": 0,
        "kg_support_entailment_uncertainty": 0.9,
        "kg_retrieval_route": "entity_first",
    }
    rows: list[CandidateDetail] = [
        {**base, "kg_generation_failed": True},
        {},
        {**base, "kg_sd_uq": 1},
        {**base, "kg_correct": True},
        {**base, "kg_retrieval_route": "entity_first_empty"},
        base,
    ]

    def fake_result_details(*a: object) -> list[CandidateDetail]:
        return rows

    monkeypatch.setattr(module, "_result_details", fake_result_details)
    selected: list[InterventionCandidate] = module.select_candidates(
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
    analysis_import: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module: GpsSensitivityExports = cast(
        GpsSensitivityExports,
        analysis_import("gps_sensitivity"),
    )
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

    def fake_load_artifact(*a: object, **kw: object) -> ReplayArtifact:
        return {"source_result": "unit", "questions": questions}

    monkeypatch.setattr(
        module,
        "load_artifact",
        fake_load_artifact,
    )

    def fake_row_map(*a: object) -> dict[str, gps_v3_depth_matched_HopRow]:
        return {}

    monkeypatch.setattr(module, "row_map", fake_row_map)
    assert module.auroc_at("hotpotqa", False, 0.6, 0.4) == (1, 2)
    questions[1]["kg_correct"] = True
    assert module.auroc_at("hotpotqa", False, 0.6, 0.4) == (None, 2)


def test_wave1_wilson_and_silent_counts(
    analysis_import: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module: Wave1AnalysisExports = cast(
        Wave1AnalysisExports,
        analysis_import("wave1_analysis"),
    )
    assert module.wilson_ci(0, 0) is None
    lo: float
    hi: float
    # wilson_ci returns None only for n == 0; this fixture supplies n == 10.
    lo, hi = cast(list[float], module.wilson_ci(5, 10))
    assert lo == pytest.approx(1 - hi)
    assert 0.2 < lo < 0.3
    monkeypatch.setattr(module, "KEY", {"unit": "Unit"})
    monkeypatch.setattr(module, "STRICT_RUNS", {})

    def fake_rows_for(*a: object) -> list[SilentRow]:
        return [
            {"correct": False, "dse": 0, "sd": 0},
            {"correct": False, "dse": 1, "sd": 0.5},
            {"correct": True, "dse": 0, "sd": 0},
        ]

    monkeypatch.setattr(
        module,
        "rows_for",
        fake_rows_for,
    )
    result: dict[str, dict[str, SilentCI]] = module.silent_cis()
    assert result["unit"]["kg"]["rate"] == 0.5
    assert result["pooled"]["kg"]["n"] == 2


def test_reporting_odds_ratio_and_answered_denominators(
    analysis_import: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module: ReportingRobustnessExports = cast(
        ReportingRobustnessExports,
        analysis_import("reporting_robustness"),
    )
    assert module.odds_ratio_mh({}) is None
    assert module.odds_ratio_mh(
        {
            "unit": {
                "cert_correct": 4,
                "cert_wrong": 1,
                "uncert_correct": 2,
                "uncert_wrong": 3,
            },
        },
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
                    },
                ],
            },
        ),
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
    analysis_import: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module: RejudgeWave1Exports = cast(
        RejudgeWave1Exports,
        analysis_import("rejudge_wave1"),
    )
    rows: list[RejudgeRow] = [
        {"qid": str(i), "sd": i / 12, "seu": i / 12} for i in range(12)
    ]

    def fake_rows_for(*a: object) -> list[RejudgeRow]:
        return rows

    monkeypatch.setattr(module, "rows_for", fake_rows_for)

    def fake_gps_scores(*a: object, **kw: object) -> dict[str, float]:
        return {str(i): i / 12 for i in range(12)}

    monkeypatch.setattr(module, "gps_scores", fake_gps_scores)
    result: dict[str, tuple[float | None, int]] = module.metric_aurocs(
        "unit",
        "kg",
        {str(i): i < 6 for i in range(12)},
    )
    assert result == {"sd_uq": (1, 12), "seu": (1, 12), "gps": (1, 12)}
    assert module.metric_aurocs("unit", "kg", {})["gps"] == (None, 0)


def test_residue_audit_filter_and_artifact(
    analysis_import: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module: ResidueAuditExports = cast(
        ResidueAuditExports,
        analysis_import("residue_audit"),
    )
    monkeypatch.setattr(module, "REPO", str(tmp_path))
    monkeypatch.setattr(module, "KEY", {"unit": "Unit"})
    monkeypatch.setattr(module, "MANIFEST", {"Unit": {"result_path": "result.json"}})
    monkeypatch.setattr(module, "SLUGS", ["unit"])
    monkeypatch.setattr(module, "OUT_PATH", str(tmp_path / "out.json"))

    def fake_gps_cache(*a: object) -> dict[str, int]:
        return {"q1": 0, "q2": 0}

    monkeypatch.setattr(module, "gps_cache", fake_gps_cache)
    base: ResidueDetail = {
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
                    },
                ],
            },
        ),
    )
    module.main()
    result: ResidueReport = json.loads((tmp_path / "out.json").read_text())
    assert result["count"] == 1
    assert result["cases"][0]["response"] == "b"
    assert "cannot be confirmed" in result["caveat"]


def test_migration_figure_artifacts(
    analysis_import: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(sys.modules, "make_figures", ModuleType("make_figures"))
    module: MakeMigrationFigureExports = cast(
        MakeMigrationFigureExports,
        analysis_import("make_migration_figure"),
    )
    monkeypatch.setattr(module, "OUT", str(tmp_path))

    def fake_rows_for(*a: object) -> list[MigrationRow]:
        return [
            {"qid": "q1", "correct": False, "sd": 0, "seu": 0.2},
            {"qid": "q2", "correct": False, "sd": 0.001, "seu": 0.8},
        ]

    monkeypatch.setattr(
        module,
        "rows_for",
        fake_rows_for,
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
    module: VerbalizedConfidenceExports = cast(
        VerbalizedConfidenceExports,
        analysis_import("verbalized_confidence"),
    )
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
