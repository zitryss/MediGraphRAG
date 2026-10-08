"""Plot real synthetic data using Agg and verify native figure artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from experiments import visualize_results as visual


@pytest.fixture
def result_blocks() -> list[visual.ResultBlock]:
    metrics: list[str] = [spec[0] for spec in visual.UNCERTAINTY_METRIC_SPECS]
    details: list[visual.PlotDetail] = []
    for i in range(8):
        row: visual.PlotDetail = {
            "question_id": str(i),
            "vanilla_correct": i % 2 == 0,
            "kg_correct": i % 3 == 0,
            "task_type": "factoid",
        }
        for prefix in ["vanilla", "kg"]:
            # Names come from the producer's finite UNCERTAINTY_METRIC_SPECS;
            # each generated key is one optional numeric PlotDetail field.
            row.update(
                cast(
                    visual.PlotDetail, {f"{prefix}_{m}": (i + 1) / 10 for m in metrics}
                )
            )
        details.append(row)
    config: visual.ConfigResult = {
        "config": {"name": "unit", "similarity_threshold": 0.1, "max_chunks": 2},
        "details": details,
        "vanilla_accuracy": 0.5,
        "kg_accuracy": 0.75,
        "metrics_by_approach": {
            "vanilla_rag": {"accuracy": 0.5},
            "kg_rag": {"accuracy": 0.75},
        },
        "auroc_aurec": {
            system: {
                key: value
                for m in metrics
                for key, value in [(f"{m}_auroc", 0.75), (f"{m}_aurec", 0.2)]
            }
            for system in ["vanilla_rag", "kg_rag"]
        },
        "accuracy_by_task_type": {
            name: {"n": 4, "vanilla_accuracy": 0.5, "kg_accuracy": 0.75}
            for name in ["factoid", "yesno"]
        },
        "complementarity": {
            "both_correct": 2,
            "vanilla_only": 2,
            "kg_only": 2,
            "neither": 2,
        },
    }
    for prefix in ["vanilla", "kg"]:
        # Same finite metric names populate the matching average fields.
        config.update(
            cast(visual.ConfigResult, {f"{prefix}_avg_{m}": 0.4 for m in metrics})
        )
        config.update(
            cast(
                visual.ConfigResult,
                {
                    f"{prefix}_avg_compute_times": {
                        m: 0.01 * (i + 1) for i, m in enumerate(metrics)
                    }
                },
            )
        )
    return [{"dataset": "unit", "config_results": [config]}]


@pytest.mark.parametrize(
    "function_name",
    [
        "plot_metric_bar_charts",
        "plot_auroc_aurec_heatmaps",
        "plot_metric_correlation_matrix",
        "plot_reliability_diagrams",
        "plot_compute_time_chart",
        "plot_auroc_vs_compute_time",
        "plot_complementarity_matrix",
        "plot_query_type_stratification",
        "plot_per_system_auroc_comparison",
        "plot_metric_spearman_matrix",
    ],
)
def test_plots_emit_png_artifacts(
    function_name: str, result_blocks: list[visual.ResultBlock], tmp_path: Path
) -> None:
    paths: list[str] = getattr(visual, function_name)(
        result_blocks, output_dir=str(tmp_path)
    )
    assert paths, function_name
    for path in paths:
        data: bytes = Path(path).read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(data) > 1000
    plt.close("all")


@pytest.mark.parametrize("multi", [False, True])
def test_legacy_aggregation_means(multi: bool, tmp_path: Path) -> None:
    rows: list[visual.ResultBlock] = [
        {
            "config": {"similarity_threshold": 0.1, "max_chunks": 2},
            "evaluation": {
                "vanilla_rag_metrics": {"accuracy": x},
                "kg_rag_metrics": {"accuracy": 0.8},
            },
        }
        for x in [0.2, 0.6]
    ]
    results: visual.VisualizationResults = (
        {"per_dataset_results": {"unit": {"results": rows}}}
        if multi
        else {"results": rows}
    )
    assert visual.aggregate_by_config(results)["thresh=0.1_chunks=2"]["vanilla_rag"][
        "accuracy"
    ] == pytest.approx(0.4)
    overall: visual.OverallMetrics = visual.aggregate_overall(results)
    assert overall["kg_rag"]["accuracy"] == 0.8
    figure: Figure | None = visual.create_overall_comparison_chart(
        overall, ["accuracy"], str(tmp_path / "overall.png")
    )
    assert figure is not None
    bars: list[Rectangle] = [
        patch for patch in figure.axes[0].patches if isinstance(patch, Rectangle)
    ]
    assert len(bars) == 2
    assert [patch.get_height() for patch in bars] == pytest.approx([0.4, 0.8])
    plt.close("all")


def test_new_schema_aggregation_and_bar_values(
    result_blocks: list[visual.ResultBlock], tmp_path: Path
) -> None:
    results: visual.VisualizationResults = {
        "results": result_blocks,
        "metric_names": ["accuracy"],
    }
    aggregated: visual.AggregatedMetrics = visual.aggregate_by_config(results)
    assert aggregated["unit"]["kg_rag"]["accuracy"] == 0.75
    assert visual.aggregate_overall(results)["vanilla_rag"]["accuracy"] == 0.5
    figure: Figure | None = visual.create_comparison_chart(
        aggregated, ["accuracy"], str(tmp_path / "comparison.png")
    )
    assert figure is not None
    bars: list[Rectangle] = [
        patch for patch in figure.axes[0].patches if isinstance(patch, Rectangle)
    ]
    assert len(bars) == 2
    assert [patch.get_height() for patch in bars] == [0.5, 0.75]
    plt.close("all")
