"""Real experiment orchestration with provider/telemetry/storage boundaries isolated."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

if TYPE_CHECKING:
    from experiments.experiment import MIRAGEEvaluationPipeline


@pytest.fixture
def pipeline(
    experiment_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> MIRAGEEvaluationPipeline:
    monkeypatch.setattr(
        experiment_module, "get_model_provider", Mock(return_value=Mock())
    )
    monkeypatch.setattr(experiment_module, "VanillaRAGSystem", Mock())
    monkeypatch.setattr(experiment_module, "EnhancedRAGSystem", Mock())
    return experiment_module.MIRAGEEvaluationPipeline(
        entropy_samples=100, evaluation_mode="accuracy_only", multi_temperature=True
    )


def test_real_pipeline_constructor_and_distinct_judge(
    experiment_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    pipeline: MIRAGEEvaluationPipeline,
) -> None:
    assert pipeline.entropy_samples == 20
    assert not pipeline.compute_metrics
    assert not pipeline.multi_temperature
    assert pipeline.judge_llm_provider is pipeline._base_llm_provider
    assert pipeline.eval_configs[0]["max_chunks"] == 10
    provider = Mock()
    monkeypatch.setattr(experiment_module, "get_model_provider", provider)
    p = experiment_module.MIRAGEEvaluationPipeline(
        entropy_samples=0, judge_provider="other", judge_model="judge"
    )
    assert p.entropy_samples == 1
    assert p.compute_metrics
    assert provider.call_count == 2
    provider.assert_called_with("other", model="judge")


@pytest.mark.parametrize(
    "kwargs,field",
    [
        ({"evaluation_mode": "bad"}, "evaluation_mode"),
        ({"dataset_kg_scope": "bad"}, "dataset_kg_scope"),
        ({"kg_builder_profile": "bad"}, "kg_builder_profile"),
    ],
)
def test_pipeline_invalid_modes_fail_before_provider(
    experiment_module: ModuleType, kwargs: dict[str, str], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        experiment_module.MIRAGEEvaluationPipeline(**kwargs)


def test_temporary_environment_restores_even_on_failure(
    pipeline: MIRAGEEvaluationPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_EXISTING", "original")
    monkeypatch.delenv("TEST_ABSENT", raising=False)
    with (
        pytest.raises(ValueError, match="abort"),
        pipeline._temporary_env({"TEST_EXISTING": None, "TEST_ABSENT": 5}),
    ):
        assert "TEST_EXISTING" not in os.environ
        assert os.environ["TEST_ABSENT"] == "5"
        raise ValueError("abort")
    assert os.environ["TEST_EXISTING"] == "original"
    assert "TEST_ABSENT" not in os.environ


def test_retrieval_environment_and_constructor_contract(
    pipeline: MIRAGEEvaluationPipeline, experiment_module: ModuleType
) -> None:
    overrides = pipeline._retrieval_env_overrides(
        {
            "retrieval_stack": {
                "query_fusion": True,
                "late_interaction": False,
                "first_stage_late_interaction": True,
                "reranker": False,
                "retrieval_profile": "strict",
            }
        }
    )
    assert overrides == {
        "ONTOGRAPHRAG_QUERY_FUSION": "1",
        "ONTOGRAPHRAG_LATE_INTERACTION": "0",
        "ONTOGRAPHRAG_FIRST_STAGE_LATE_INTERACTION": "1",
        "ONTOGRAPHRAG_RERANKER": "0",
        "ONTOGRAPHRAG_RETRIEVAL_PROFILE": "strict",
    }
    pipeline._build_rag_systems_for_config(
        {
            "kg_system": {
                "allow_vector_fallback": False,
                "max_chunks_per_passage": "3",
                "traversal_chunk_min_sim": "0.2",
            }
        }
    )
    kwargs = experiment_module.EnhancedRAGSystem.call_args.kwargs
    assert kwargs["allow_vector_fallback"] is False
    assert kwargs["max_chunks_per_passage"] == 3
    assert kwargs["traversal_chunk_min_sim"] == 0.2


@pytest.mark.parametrize(
    "config,signature",
    [
        (None, "vanilla_kg"),
        ({"executed_systems": ["kg_rag"]}, "kg"),
        ({"executed_systems": ["invalid"]}, "vanilla_kg"),
    ],
)
def test_execution_signature(
    pipeline: MIRAGEEvaluationPipeline, config: dict[str, object] | None, signature: str
) -> None:
    assert pipeline._execution_signature(config) == signature


def test_routing_distribution_excludes_generation_failures(
    pipeline: MIRAGEEvaluationPipeline,
) -> None:
    result = pipeline._compute_kg_routing_distribution(
        [
            {"kg_retrieval_route": "entity_first"},
            {"kg_retrieval_route": "vector_only"},
            {},
            {"kg_generation_failed": True, "kg_retrieval_route": "vector_only"},
        ]
    )
    assert result["n"] == 3
    assert result["pure_entity_first_rate"] == pytest.approx(1 / 3)
    assert result["dense_fallback_rate"] == pytest.approx(1 / 3)
    assert result["unknown_route_rate"] == pytest.approx(1 / 3)
    assert pipeline._compute_kg_routing_distribution([])["graph_route_rate"] == 0


def test_run_index_append_and_corrupt_recovery(
    experiment_module: ModuleType, tmp_path: Path
) -> None:
    run = tmp_path / "run"
    manifest = {"run_id": "r1", "created_at": "2026-10-08", "datasets": ["unit"]}
    experiment_module.update_runs_index(run, manifest)
    experiment_module.update_runs_index(
        run, {**manifest, "run_id": "r2", "model": "fake"}
    )
    entries = json.loads((tmp_path / "index.json").read_text())
    assert [r["run_id"] for r in entries] == ["r1", "r2"]
    assert entries[0]["model"] == "unknown"
    (tmp_path / "index.json").write_text("broken")
    experiment_module.update_runs_index(run, manifest)
    assert len(json.loads((tmp_path / "index.json").read_text())) == 1


def test_invalid_retrieval_profile(experiment_module: ModuleType) -> None:
    with pytest.raises(ValueError, match="Unsupported retrieval study"):
        experiment_module.MIRAGEEvaluationPipeline.build_retrieval_study_eval_configs(
            profile="bad",
            similarity_thresholds=[0.1],
            max_chunks_values=[2],
            retrieval_temperature_values=[0],
            retrieval_shortlist_factor=1,
        )


@pytest.mark.parametrize(
    "mode",
    ["new", "reuse", "rebuild", "build_failed", "policy_denied", "evaluation_failed"],
)
def test_pipeline_run_artifacts_and_control_flow(
    pipeline: MIRAGEEvaluationPipeline,
    experiment_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    run = Mock(id="offline-unit-run")
    run.summary = {}
    monkeypatch.setattr(experiment_module.wandb, "init", Mock(return_value=run))
    monkeypatch.setattr(experiment_module, "_git_commit", lambda: "unit-revision")
    monkeypatch.setattr(
        pipeline, "_dataset_corpus_profile", lambda name: {"corpus_source": "synthetic"}
    )
    monkeypatch.setattr(
        pipeline, "_selection_path", lambda name: tmp_path / "absent-selection.json"
    )
    monkeypatch.setattr(
        pipeline,
        "_dataset_kg_exists",
        Mock(return_value=2 if mode in {"reuse", "rebuild"} else 0),
    )
    monkeypatch.setattr(
        pipeline,
        "_enforce_dataset_corpus_policy",
        Mock(return_value=(mode != "policy_denied", False)),
    )
    monkeypatch.setattr(pipeline, "_get_dataset_corpus_metadata", Mock(return_value={}))
    monkeypatch.setattr(
        pipeline, "_prepare_dataset_kg_contract", Mock(return_value={"build_meta": {}})
    )
    monkeypatch.setattr(
        pipeline, "_assess_dataset_kg_compatibility", Mock(return_value=(True, []))
    )
    build = Mock(return_value=mode != "build_failed")
    delete = Mock()
    monkeypatch.setattr(pipeline, "_build_kg_for_dataset", build)
    monkeypatch.setattr(pipeline, "_delete_dataset_kg", delete)
    monkeypatch.setattr(pipeline, "_verify_kg_quality", Mock(return_value=True))
    result = {
        "config": {"name": "default"},
        "vanilla_accuracy": 0.5,
        "kg_accuracy": 1.0,
        "num_questions_logged": 2,
        "details": [{"question_id": "q", "internal_not_for_export": "private"}],
    }
    evaluate = Mock(
        return_value={"error": "generation unavailable"}
        if mode == "evaluation_failed"
        else result
    )
    monkeypatch.setattr(pipeline, "_run_evaluation_on_dataset", evaluate)
    # Telemetry tables/plots are separate contracts, not part of run sequencing.
    monkeypatch.setattr(pipeline, "_log_config_summary_to_wandb", Mock())
    monkeypatch.setattr(pipeline, "_log_final_summary_to_wandb", Mock())
    pipeline.rebuild_kg = mode == "rebuild"
    pipeline.run_pipeline(["pubmedqa"], output_dir=str(tmp_path))
    manifest = json.loads((pipeline._run_dir / "manifest.json").read_text())
    summary = json.loads(
        (pipeline._run_dir / "mirage_evaluation_summary.json").read_text()
    )
    assert manifest["git_commit"] == "unit-revision"
    assert manifest["wandb_run_id"] == "offline-unit-run"
    assert "completed_at" in manifest
    assert (
        json.loads((tmp_path / "runs/index.json").read_text())[0]["run_id"]
        == manifest["run_id"]
    )
    run.finish.assert_called_once()
    if mode in {"policy_denied", "build_failed"}:
        evaluate.assert_not_called()
        assert summary["results"] == []
    elif mode == "evaluation_failed":
        assert summary["results"][0]["config_results"] == []
    else:
        assert manifest["accuracy"]["pubmedqa/default"]["kg"] == 1.0
        saved = json.loads(
            (pipeline._run_dir / "mirage_pubmedqa_results.json").read_text()
        )
        assert saved["config_results"][0]["details"] == [{"question_id": "q"}]
        assert summary["results"][0]["track"] == "biomedical_grounding"
    if mode == "reuse":
        build.assert_not_called()
    if mode == "rebuild":
        delete.assert_called_once_with("pubmedqa")
        build.assert_called_once_with("pubmedqa")


@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "generation_failed",
        "kg_only",
        "missing_selection",
        "resume",
        "rebuild",
        "metrics",
        "metrics_failed",
    ],
)
def test_evaluation_loop_scopes_retrieval_and_serializes_answers(
    pipeline: MIRAGEEvaluationPipeline,
    experiment_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    pipeline.use_llm_judge = False
    if mode in {"metrics", "metrics_failed"}:
        pipeline.compute_metrics = True

        def collect(**kwargs: Any) -> tuple:
            base = (["yes", "yes"], ["retrieved evidence"], 1.0)
            assert kwargs["question_id"] == "q1" and kwargs["kg_name"] == "pubmedqa"
            return (*base, []) if kwargs.get("return_graph_state_traces") else base

        monkeypatch.setattr(pipeline, "_collect_sample_responses", collect)
        monkeypatch.setattr(
            experiment_module,
            "compute_all_uncertainty_metrics",
            Mock(
                return_value={
                    "semantic_entropy": 0.2,
                    "compute_times": {"semantic_entropy": 0.01},
                }
            ),
        )
        for name in [
            "compute_graph_path_support_detailed",
            "compute_subgraph_perturbation_stability_detailed",
        ]:
            monkeypatch.setattr(
                experiment_module,
                name,
                Mock(return_value={"score": 0.25, "null_reason": None}),
            )
        for name in [
            "compute_graph_path_disagreement",
            "compute_competing_answer_alternatives",
            "compute_evidence_vn_entropy",
            "compute_subgraph_informativeness",
            "compute_support_entailment_uncertainty",
            "compute_evidence_conflict_uncertainty",
        ]:
            monkeypatch.setattr(experiment_module, name, Mock(return_value=0.3))
    data = {
        "q1": {
            "question": "Does this work?",
            "final_decision": "yes",
            "contexts": ["synthetic gold evidence"],
        }
    }
    monkeypatch.setattr(pipeline, "_load_mirage_raw_data", Mock(return_value=data))
    monkeypatch.setattr(pipeline, "_dataset_kg_exists", Mock(return_value=0))
    monkeypatch.setattr(
        pipeline,
        "_resolve_selected_question_ids",
        Mock(
            return_value=["q1"],
            side_effect=ValueError("selection missing")
            if mode == "missing_selection"
            else None,
        ),
    )
    monkeypatch.setattr(
        pipeline, "_subset_metadata", Mock(return_value={"subset_tag": "unit"})
    )
    monkeypatch.setattr(
        pipeline, "_selection_path", lambda name: tmp_path / "selection.json"
    )
    monkeypatch.setattr(
        pipeline, "_get_normalized_dataset", Mock(return_value=([], [], False))
    )
    monkeypatch.setattr(
        experiment_module, "build_global_corpus_passages", Mock(return_value=None)
    )
    vanilla = Mock()
    kg = Mock()
    for rag in [vanilla, kg]:
        rag.generate_response.return_value = {
            "response": "Yes",
            "context": {
                "chunks": [
                    {
                        "text": "retrieved evidence",
                        "question_id": "q1",
                        "kg_name": "pubmedqa",
                    }
                ]
            },
        }
        if mode in {"generation_failed", "metrics_failed"}:
            rag.generate_response.side_effect = RuntimeError("provider unavailable")
    monkeypatch.setattr(
        pipeline, "_build_rag_systems_for_config", Mock(return_value=(vanilla, kg))
    )
    if mode in {"resume", "rebuild"}:
        checkpoint_dir = tmp_path / "results/checkpoints"
        checkpoint_dir.mkdir(parents=True)
        signature = pipeline._execution_signature({"name": "unit"})
        checkpoint = checkpoint_dir / f"pubmedqa_unit_unit_{signature}.jsonl"
        row = {
            "question_id": "q1",
            "question": "Does this work?",
            "vanilla_correct": True,
            "kg_correct": True,
            "kg_generation_failed": False,
            "vanilla_generation_failed": False,
            "kg_response": "Yes",
            "vanilla_response": "Yes",
        }
        checkpoint.write_text(
            json.dumps(row) + "\nmalformed-line\n" + json.dumps(row) + "\n"
        )
        pipeline.rebuild_kg = mode == "rebuild"
    result = pipeline._run_evaluation_on_dataset(
        "pubmedqa",
        config={
            "name": "unit",
            **({"executed_systems": ["kg_rag"]} if mode == "kg_only" else {}),
        },
    )
    if mode == "missing_selection":
        assert result["error"] == "selection missing"
        kg.generate_response.assert_not_called()
        return
    if mode in {"metrics", "metrics_failed"}:
        detail = result["details"][0]
        assert detail["kg_graph_path_support"] == (
            0.5 if mode == "metrics_failed" else 0.25
        )
        if mode == "metrics":
            assert detail["kg_semantic_entropy"] == 0.2
            assert (
                experiment_module.compute_graph_path_support_detailed.call_args.kwargs[
                    "question_id"
                ]
                == "q1"
            )
        else:
            experiment_module.compute_graph_path_support_detailed.assert_not_called()
    assert result["num_questions_logged"] == 1
    assert result["kg_accuracy"] == (
        0 if mode in {"generation_failed", "metrics_failed"} else 1
    )
    assert result["details"][0]["kg_generation_failed"] is (
        mode in {"generation_failed", "metrics_failed"}
    )
    if mode == "resume":
        kg.generate_response.assert_not_called()
        vanilla.generate_response.assert_not_called()
        assert not list((tmp_path / "results/checkpoints").glob("*.jsonl"))
        return
    assert kg.generate_response.call_args.kwargs["question_id"] == "q1"
    assert kg.generate_response.call_args.kwargs["extra_context_texts"] is None
    assert kg.generate_response.call_args.kwargs["allow_decomposition"] is False
    remaining = list((tmp_path / "results/checkpoints").glob("*.jsonl"))
    if mode == "rebuild":
        assert len(remaining) == 1 and ".pre_rebuild_" in remaining[0].name
    else:
        assert not remaining
    saved = json.loads(Path(result["question_details_file"]).read_text())
    assert saved["num_questions"] == 1
    assert saved["questions"][0]["question_id"] == "q1"
    if mode == "kg_only":
        vanilla.generate_response.assert_not_called()
        assert result["details"][0]["vanilla_system_skipped"]
