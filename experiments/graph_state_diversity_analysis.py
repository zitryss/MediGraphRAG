"""Graph-state diversity analysis for runs with per-sample graph traces.

The current headline run logs predate graph-state tracing, so this script is
also an availability check: it reports which saved runs contain the required
fields and only draws the phase plot when enough rows are present.
"""

import json
import os
import sys
from collections.abc import Iterator
from typing import (
    TYPE_CHECKING,
    Protocol,
    TypedDict,
    cast,
)

import numpy as np
from numpy.typing import NDArray
from typing_extensions import ReadOnly

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


if TYPE_CHECKING:
    from experiments.dataset_adapters import DefaultJSONCodec, JSONValue


class OptionalPlotting(Protocol):
    """The two pyplot operations used by the optional phase plot."""

    def subplots(self, *, figsize: tuple[float, float]) -> tuple["Figure", "Axes"]: ...

    def close(self, fig: "Figure") -> None: ...


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as _pyplot

    plt: OptionalPlotting | None = _pyplot
except Exception:  # pragma: no cover - plotting is optional
    plt = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.trust_analysis import KEY, MANIFEST, REPO, STRICT_RUNS
from ontographrag.rag.graph_state import (
    GraphStateSample,
    GraphStateSummary,
    graph_state_diversity,
)


class DiversityDetail(TypedDict, total=False):
    question_id: str | int
    kg_correct: bool
    kg_graph_state_traces: list[GraphStateSample | GraphStateSummary | None] | None
    kg_graph_state_sample_count: int | None
    kg_subgraph_entropy: float | None
    kg_path_entropy: float | None
    kg_seed_entity_entropy: float | None
    kg_chunk_entropy: float | None
    kg_subgraph_jaccard: float | None
    kg_discrete_semantic_entropy: float | None
    kg_sd_uq: float | None


class DiversityConfig(TypedDict):
    config: dict[str, str]
    details: list[DiversityDetail]


class DiversityDocument(TypedDict):
    config_results: list[DiversityConfig]


class DiversityProjection(TypedDict, total=False):
    sample_count: ReadOnly[int | None]
    seed_entity_entropy: ReadOnly[float | None]
    path_entropy: ReadOnly[float | None]
    subgraph_entropy: ReadOnly[float | None]
    chunk_entropy: ReadOnly[float | None]
    subgraph_jaccard: ReadOnly[float | None]


class DiversityRow(TypedDict):
    dataset: str
    policy: str
    question_id: str
    correct: bool
    output_discrete_semantic_entropy: float
    sd_uq: float
    subgraph_entropy: float
    path_entropy: float
    seed_entity_entropy: float
    subgraph_jaccard: float
    sample_count: int


class DiversityReport(TypedDict):
    status: str
    note: str
    row_count: int
    rows_by_dataset_policy: dict[str, int]
    figures: list[str]
    rows_preview: list[DiversityRow]


OUT_JSON: str = os.path.join(
    REPO,
    "results",
    "analyses",
    "graph_state_diversity_analysis.json",
)
FIG_DIR: str = os.path.join(REPO, "paper", "figures")


def _iter_result_details() -> "Iterator[tuple[str, str, JSONValue]]":
    for slug, manifest_key in KEY.items():
        doc: "JSONValue" = cast("DefaultJSONCodec", json).load(
            open(os.path.join(REPO, MANIFEST[manifest_key]["result_path"])),
        )
        for cfg in doc["config_results"]:
            if not cfg["config"]["name"].startswith("kg_entity_first"):
                continue
            yield slug, "adaptive", cfg["details"]
    for slug, rel_path in STRICT_RUNS.items():
        doc = cast("DefaultJSONCodec", json).load(open(os.path.join(REPO, rel_path)))
        yield slug, "strict", doc["config_results"][0]["details"]


def _row_diversity(row: DiversityDetail) -> DiversityProjection:
    if row.get("kg_graph_state_traces"):
        return graph_state_diversity(row.get("kg_graph_state_traces") or [])
    if row.get("kg_subgraph_entropy") is not None:
        return {
            "sample_count": row.get("kg_graph_state_sample_count", 0),
            "seed_entity_entropy": row.get("kg_seed_entity_entropy", 0.0),
            "path_entropy": row.get("kg_path_entropy", 0.0),
            "subgraph_entropy": row.get("kg_subgraph_entropy", 0.0),
            "chunk_entropy": row.get("kg_chunk_entropy", 0.0),
            "subgraph_jaccard": row.get("kg_subgraph_jaccard", 0.0),
        }
    return {}


def collect_rows() -> list[DiversityRow]:
    rows: list[DiversityRow] = []
    for dataset, policy, details in _iter_result_details():
        for row in details:
            diversity: DiversityProjection = _row_diversity(row)
            if not diversity:
                continue
            output_entropy: "JSONValue" = row.get("kg_discrete_semantic_entropy")
            if output_entropy is None:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "policy": policy,
                    "question_id": str(row.get("question_id")),
                    "correct": bool(row.get("kg_correct")),
                    "output_discrete_semantic_entropy": float(output_entropy),
                    "sd_uq": float(row.get("kg_sd_uq") or 0.0),
                    "subgraph_entropy": float(diversity.get("subgraph_entropy") or 0.0),
                    "path_entropy": float(diversity.get("path_entropy") or 0.0),
                    "seed_entity_entropy": float(
                        diversity.get("seed_entity_entropy") or 0.0,
                    ),
                    "subgraph_jaccard": float(diversity.get("subgraph_jaccard") or 0.0),
                    "sample_count": int(diversity.get("sample_count") or 0),
                },
            )
    return rows


def plot(rows: list[DiversityRow]) -> list[str]:
    if plt is None or len(rows) < 5:
        return []
    os.makedirs(FIG_DIR, exist_ok=True)
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    x: NDArray[np.float64] = np.array(
        [r["subgraph_entropy"] for r in rows],
        dtype=float,
    )
    y: NDArray[np.float64] = np.array(
        [r["output_discrete_semantic_entropy"] for r in rows],
        dtype=float,
    )
    wrong: NDArray[np.bool_] = np.array([not r["correct"] for r in rows], dtype=bool)
    ax.scatter(x[~wrong], y[~wrong], s=18, c="#4C78A8", alpha=0.55, label="correct")
    ax.scatter(x[wrong], y[wrong], s=22, c="#E45756", alpha=0.72, label="wrong")
    ax.set_xlabel("Graph-state entropy (subgraph signature, bits)")
    ax.set_ylabel("Output semantic entropy (DSE)")
    ax.set_title("Graph-State Diversity vs Output Dispersion")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    outputs: list[str] = []
    for ext in ("pdf", "png"):
        path: str = os.path.join(
            FIG_DIR,
            f"graph_state_entropy_vs_output_entropy.{ext}",
        )
        fig.savefig(path, bbox_inches="tight", dpi=180)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    rows: list[DiversityRow] = collect_rows()
    by_dataset: dict[str, int] = {}
    for row in rows:
        key: str = f"{row['dataset']}:{row['policy']}"
        by_dataset[key] = by_dataset.get(key, 0) + 1
    figures: list[str] = plot(rows)
    out: DiversityReport = {
        "status": "ok" if rows else "no_graph_state_trace_fields",
        "note": (
            "Requires per-sample kg_graph_state_traces or derived kg_*_entropy "
            "fields from runs produced after graph-state tracing was added."
        ),
        "row_count": len(rows),
        "rows_by_dataset_policy": by_dataset,
        "figures": figures,
        "rows_preview": rows[:20],
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
