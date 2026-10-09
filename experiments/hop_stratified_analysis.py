"""
Hop-stratified AUROC and Precision@k (PPV) analysis.

Tests the hypothesis:
  - Single-hop questions: grounding metrics (SEU, ECU) dominate
  - Multi-hop questions:  structural metrics (GPS, SPS) also become informative

Produces:
  paper/figures/hop_auroc.pdf    -- AUROC by metric family × hop count
  paper/figures/hop_ppv.pdf      -- Precision@20% by metric family × hop count
  paper/figures/hop_auroc_detail.pdf  -- full per-metric view for appendix

Run from project root:
    python experiments/hop_stratified_analysis.py
"""

import json
import os
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    NotRequired,
    Protocol,
    TypeAlias,
    TypedDict,
    cast,
)

import numpy as np
from numpy.typing import NDArray
from typing_extensions import ReadOnly

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.container import BarContainer
    from matplotlib.figure import Figure

if TYPE_CHECKING:
    from experiments.dataset_adapters import DefaultJSONCodec, JSONValue


MetricKey: TypeAlias = Literal[
    "semantic_entropy",
    "discrete_semantic_entropy",
    "sre_uq",
    "p_true",
    "selfcheckgpt",
    "vn_entropy",
    "sd_uq",
    "graph_path_support",
    "subgraph_perturbation_stability",
    "support_entailment_uncertainty",
    "evidence_conflict_uncertainty",
]
FamilyKey: TypeAlias = Literal["output", "structural", "grounding"]
StatKey: TypeAlias = Literal["auroc", "ppv"]


class HopDetail(TypedDict, total=False):
    id: str | int
    question_id: str | int
    hop_count: int | None
    kg_correct: bool
    kg_generation_failed: bool
    kg_semantic_entropy: float | str | None
    kg_discrete_semantic_entropy: float | str | None
    kg_sre_uq: float | str | None
    kg_p_true: float | str | None
    kg_selfcheckgpt: float | str | None
    kg_vn_entropy: float | str | None
    kg_sd_uq: float | str | None
    kg_graph_path_support: float | str | None
    kg_subgraph_perturbation_stability: float | str | None
    kg_support_entailment_uncertainty: float | str | None
    kg_evidence_conflict_uncertainty: float | str | None
    kg_graph_path_support_null_reason: str | None
    kg_subgraph_perturbation_stability_null_reason: str | None
    vanilla_correct: bool
    vanilla_generation_failed: bool
    vanilla_semantic_entropy: float | str | None
    vanilla_discrete_semantic_entropy: float | str | None
    vanilla_sre_uq: float | str | None
    vanilla_p_true: float | str | None
    vanilla_selfcheckgpt: float | str | None
    vanilla_vn_entropy: float | str | None
    vanilla_sd_uq: float | str | None
    vanilla_graph_path_support: float | str | None
    vanilla_subgraph_perturbation_stability: float | str | None
    vanilla_support_entailment_uncertainty: float | str | None
    vanilla_evidence_conflict_uncertainty: float | str | None
    vanilla_graph_path_support_null_reason: str | None
    vanilla_subgraph_perturbation_stability_null_reason: str | None


class HopConfig(TypedDict, total=False):
    config: dict[str, str]
    details: list[HopDetail]


class HopDocument(TypedDict, total=False):
    config_results: list[HopConfig]


class MetricStats(TypedDict):
    auroc: float
    ppv: float
    label: str
    family: str


class BuildingHopStats(TypedDict):
    n: int
    error_rate: float
    semantic_entropy: NotRequired[MetricStats]
    discrete_semantic_entropy: NotRequired[MetricStats]
    sre_uq: NotRequired[MetricStats]
    p_true: NotRequired[MetricStats]
    selfcheckgpt: NotRequired[MetricStats]
    vn_entropy: NotRequired[MetricStats]
    sd_uq: NotRequired[MetricStats]
    graph_path_support: NotRequired[MetricStats]
    subgraph_perturbation_stability: NotRequired[MetricStats]
    support_entailment_uncertainty: NotRequired[MetricStats]
    evidence_conflict_uncertainty: NotRequired[MetricStats]


class HopStats(TypedDict):
    """Complete compute_stats output: every declared metric receives a record."""

    n: int
    error_rate: float
    semantic_entropy: MetricStats
    discrete_semantic_entropy: MetricStats
    sre_uq: MetricStats
    p_true: MetricStats
    selfcheckgpt: MetricStats
    vn_entropy: MetricStats
    sd_uq: MetricStats
    graph_path_support: MetricStats
    subgraph_perturbation_stability: MetricStats
    support_entailment_uncertainty: MetricStats
    evidence_conflict_uncertainty: MetricStats


class FamilyStatsView(TypedDict, total=False):
    semantic_entropy: ReadOnly[MetricStats]
    discrete_semantic_entropy: ReadOnly[MetricStats]
    sre_uq: ReadOnly[MetricStats]
    p_true: ReadOnly[MetricStats]
    selfcheckgpt: ReadOnly[MetricStats]
    vn_entropy: ReadOnly[MetricStats]
    sd_uq: ReadOnly[MetricStats]
    graph_path_support: ReadOnly[MetricStats]
    subgraph_perturbation_stability: ReadOnly[MetricStats]
    support_entailment_uncertainty: ReadOnly[MetricStats]
    evidence_conflict_uncertainty: ReadOnly[MetricStats]


class FamilyMetricLookup(Protocol):
    """Read-only indexing for the finite metric fields, not other row metadata."""

    def __getitem__(self, key: MetricKey, /) -> MetricStats: ...


class HopSummary(TypedDict):
    dataset: str
    hop_label: str
    n: int
    error_rate: float
    output_auroc: float
    structural_auroc: float
    grounding_auroc: float
    output_ppv: float
    structural_ppv: float
    grounding_ppv: float
    _stats: NotRequired[HopStats]


class PerMetricRow(TypedDict):
    dataset: str
    hop_label: str
    metric: str
    family: str
    n: int
    auroc: float
    ppv: float
    error_rate: float


class HopReport(TypedDict):
    rows: list[HopSummary]
    rows_detail: list[PerMetricRow]
    saved_figures: list[str]


# ── Results paths ──────────────────────────────────────────────────────────────
RUNS: dict[str, str] = {
    "BioASQ": "results/runs/20260402-172730-bioasq-n100-full-metrics-evaluation-subset/mirage_bioasq_results.json",
    "RealMedQA": "results/runs/20260402-155138-realmedqa-n100-full-metrics-evaluation-subset/mirage_realmedqa_results.json",
    "PubMedQA": "results/runs/20260401-074100-pubmedqa-n100-full-metrics-evaluation-subset-rebuildkg/mirage_pubmedqa_results.json",
    "2Wiki": "results/runs/20260402-185601-2wikimultihopqa-n100-full-metrics-evaluation-subset/mirage_2wikimultihopqa_results.json",
    "HotpotQA": "results/runs/20260403-103006-hotpotqa-n100-full-metrics-evaluation-subset/mirage_hotpotqa_results.json",
    "MultiHopRAG": "results/runs/20260402-201208-multihoprag-n100-full-metrics-evaluation-subset/mirage_multihoprag_results.json",
}

# ── Metric layout ──────────────────────────────────────────────────────────────
# (internal_key, display_label, family, higher_is_more_certain)
METRICS: list[tuple[MetricKey, str, FamilyKey, bool]] = [
    # Output estimators
    ("semantic_entropy", "SE", "output", False),
    ("discrete_semantic_entropy", "DSE", "output", False),
    ("sre_uq", "SRE-UQ", "output", False),
    ("p_true", "P(True)", "output", True),
    ("selfcheckgpt", "SelfCk", "output", False),
    ("vn_entropy", "VN-Ent", "output", False),
    ("sd_uq", "SD-UQ", "output", False),
    # Structural measures
    ("graph_path_support", "GPS", "structural", False),
    ("subgraph_perturbation_stability", "SPS-UQ", "structural", False),
    # Grounding measures
    ("support_entailment_uncertainty", "SEU", "grounding", False),
    ("evidence_conflict_uncertainty", "ECU", "grounding", False),
]

FAMILY_COLORS: dict[str, str] = {
    "output": "#3A7DC9",
    "structural": "#E07B39",
    "grounding": "#4BAE8A",
}

FAMILY_LABELS: dict[str, str] = {
    "output": "Output",
    "structural": "Structural",
    "grounding": "Grounding",
}

SYSTEM: str = "kg_rag"  # primary system for the analysis
SYSTEM_PREFIX: str = "kg"

# Biomedical datasets have no hop metadata → treat as "single-hop proxy"
SINGLE_HOP_DATASETS: set[str] = {"BioASQ", "RealMedQA", "PubMedQA"}
# HotpotQA is 2-hop by design even though hop_count metadata is absent
HOTPOTQA_DEFAULT_HOP: int = 2
PREFERRED_CONFIG_NAME: str | None = os.environ.get("MIRAGE_CONFIG_NAME")


# ── Data loading ───────────────────────────────────────────────────────────────


def _select_config_result(path: str, results_doc: "JSONValue") -> "JSONValue":
    config_results: "JSONValue" = results_doc.get("config_results", [])
    if not config_results:
        raise ValueError(f"{path} contains no config_results")
    if len(config_results) == 1:
        return config_results[0]

    if PREFERRED_CONFIG_NAME:
        for cfg_res in config_results:
            if cfg_res.get("config", {}).get("name") == PREFERRED_CONFIG_NAME:
                return cfg_res

    for cfg_res in config_results:
        if cfg_res.get("config", {}).get("name") == "default":
            return cfg_res

    config_names: list[str] = [
        cfg.get("config", {}).get("name", "<unnamed>") for cfg in config_results
    ]
    raise ValueError(
        f"{path} contains multiple configs {config_names}; "
        "set MIRAGE_CONFIG_NAME to choose one explicitly",
    )


def load_details(path: str) -> "JSONValue":
    with open(path) as f:
        results_doc: "JSONValue" = cast("DefaultJSONCodec", json).load(f)
    return _select_config_result(path, results_doc).get("details", [])


def hop_label(hop: object, ds_name: str) -> str:
    """Canonical hop label for a record."""
    if hop is None:
        if ds_name in SINGLE_HOP_DATASETS:
            return "1-hop (proxy)"
        if ds_name == "HotpotQA":
            return "2-hop"
        return "unknown"
    return f"{hop}-hop"


def to_uncertainty(score: float, higher_is_more_certain: bool) -> float:
    return -score if higher_is_more_certain else score


# ── Core stats ─────────────────────────────────────────────────────────────────


def auroc(
    y_true: NDArray[np.float64 | np.int64],
    uncertainty: NDArray[np.float64 | np.int64],
) -> float:
    """AUROC: predicting incorrectness from uncertainty. 0.5 = chance."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")
    if len(np.unique(y_true)) < 2 or len(y_true) < 4:
        return float("nan")
    return float(roc_auc_score(y_true, -uncertainty))


def ppv_at_k(
    y_true: NDArray[np.float64 | np.int64],
    uncertainty: NDArray[np.float64 | np.int64],
    frac: float = 0.20,
) -> float:
    """Precision@k: among the top-frac most uncertain questions, what fraction are wrong?
    Returns NaN if fewer than 2 questions in the pool.
    """
    n: int = len(y_true)
    k: int = max(1, int(np.ceil(n * frac)))
    if n < 2:
        return float("nan")
    order: NDArray[np.intp] = np.argsort(-uncertainty)  # most uncertain first
    top_k: NDArray[np.float64 | np.int64] = y_true[order[:k]]
    wrong_in_top_k: np.int64 = np.sum(top_k == 0)
    return float(wrong_in_top_k / k)


def baseline_error_rate(y_true: NDArray[np.float64 | np.int64]) -> float:
    return float(np.mean(y_true == 0)) if len(y_true) > 0 else float("nan")


def compute_stats(
    details: Sequence[HopDetail | Mapping[str, object]],
    hop_label_: str,
    ds_name: str,
    ppv_frac: float = 0.20,
    system_prefix: str = SYSTEM_PREFIX,
) -> HopStats:
    """For a slice of details (same hop bucket), compute AUROC and PPV for each metric."""
    clean_details: list[HopDetail | Mapping[str, object]] = [
        d
        for d in details
        if not bool(d.get(f"{system_prefix}_generation_failed", False))
    ]
    y_true_global: NDArray[np.float64] = np.array(
        [
            1.0 if d.get(f"{system_prefix}_correct", False) else 0.0
            for d in clean_details
        ],
    )
    results: BuildingHopStats = {
        "n": len(clean_details),
        "error_rate": baseline_error_rate(y_true_global),
    }

    for key, label, family, higher_is_certain in METRICS:
        metric_details: list[HopDetail | Mapping[str, object]] = clean_details
        if key == "graph_path_support":
            metric_details = [
                d
                for d in clean_details
                if not str(d.get(f"{system_prefix}_graph_path_support_null_reason", ""))
            ]
        elif key == "subgraph_perturbation_stability":
            null_key: str = (
                f"{system_prefix}_subgraph_perturbation_stability_null_reason"
            )
            if any(null_key in d for d in clean_details):
                metric_details = [
                    d for d in clean_details if not str(d.get(null_key, ""))
                ]
            else:
                metric_details = []
                for d in clean_details:
                    legacy_value: object = d.get(f"{system_prefix}_{key}")
                    try:
                        if (
                            legacy_value is not None
                            and abs(float(legacy_value) - 0.5) > 1e-12
                        ):
                            metric_details.append(d)
                    except (TypeError, ValueError):
                        continue
        raw: NDArray[np.float64] = np.array(
            [
                float(d.get(f"{system_prefix}_{key}", float("nan")))
                for d in metric_details
            ],
        )
        valid: NDArray[np.bool_] = ~np.isnan(raw)
        if valid.sum() < 4:
            results[key] = {
                "auroc": float("nan"),
                "ppv": float("nan"),
                "label": label,
                "family": family,
            }
            continue
        uncertainty_fn: Callable[[float], float] = lambda s: to_uncertainty(
            s,
            higher_is_certain,
        )
        unc: NDArray[np.float64] = np.where(
            valid,
            np.vectorize(uncertainty_fn)(raw),
            float("nan"),
        )
        yt: NDArray[np.float64] = np.array(
            [
                1.0 if d.get(f"{system_prefix}_correct", False) else 0.0
                for d in metric_details
            ],
        )[valid]
        un: NDArray[np.float64] = unc[valid]
        results[key] = {
            "auroc": auroc(yt, un),
            "ppv": ppv_at_k(yt, un, ppv_frac),
            "label": label,
            "family": family,
        }
    # Both loop branches populate every key enumerated by METRICS.
    return cast(HopStats, results)


def family_avg(stats: FamilyStatsView, stat_key: StatKey) -> dict[str, float]:
    """Average a stat (auroc or ppv) across metrics within each family."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for key, label, family, _ in METRICS:
        if key in stats and not np.isnan(
            cast(FamilyMetricLookup, stats)[key].get(stat_key, float("nan")),
        ):
            buckets[family].append(cast(FamilyMetricLookup, stats)[key][stat_key])
    return {f: float(np.mean(vs)) for f, vs in buckets.items() if vs}


# ── Build analysis table ────────────────────────────────────────────────────────


def build_hop_table() -> list[HopSummary]:
    """
    Returns a list of rows:
      {dataset, hop_label, n, error_rate,
       output_auroc, structural_auroc, grounding_auroc,
       output_ppv, structural_ppv, grounding_ppv}
    """
    rows: list[HopSummary] = []
    for ds, path in RUNS.items():
        try:
            details: list[HopDetail] = load_details(path)
        except Exception as e:
            print(f"  [skip] {ds}: {e}")
            continue

        # Group by hop bucket
        buckets: dict[str, list[HopDetail]] = defaultdict(list)
        for d in details:
            hc: int | None = d.get("hop_count")
            hl: str = hop_label(hc, ds)
            if hl != "unknown":
                buckets[hl].append(d)

        for hl, bucket_details in sorted(buckets.items()):
            stats: HopStats = compute_stats(bucket_details, hl, ds)
            fa: dict[str, float] = family_avg(stats, "auroc")
            fp: dict[str, float] = family_avg(stats, "ppv")
            rows.append(
                {
                    "dataset": ds,
                    "hop_label": hl,
                    "n": stats["n"],
                    "error_rate": stats["error_rate"],
                    "output_auroc": fa.get("output", float("nan")),
                    "structural_auroc": fa.get("structural", float("nan")),
                    "grounding_auroc": fa.get("grounding", float("nan")),
                    "output_ppv": fp.get("output", float("nan")),
                    "structural_ppv": fp.get("structural", float("nan")),
                    "grounding_ppv": fp.get("grounding", float("nan")),
                    "_stats": stats,
                },
            )
    return rows


def build_per_metric_table() -> list[PerMetricRow]:
    """Full per-metric table for appendix figure."""
    rows: list[PerMetricRow] = []
    for ds, path in RUNS.items():
        try:
            details: list[HopDetail] = load_details(path)
        except Exception:
            continue
        buckets: dict[str, list[HopDetail]] = defaultdict(list)
        for d in details:
            hc: int | None = d.get("hop_count")
            hl: str = hop_label(hc, ds)
            if hl != "unknown":
                buckets[hl].append(d)
        for hl, bucket_details in sorted(buckets.items()):
            stats: HopStats = compute_stats(bucket_details, hl, ds)
            for key, label, family, _ in METRICS:
                if key not in stats:
                    continue
                rows.append(
                    {
                        "dataset": ds,
                        "hop_label": hl,
                        "metric": label,
                        "family": family,
                        "n": stats["n"],
                        "auroc": stats[key]["auroc"],
                        "ppv": stats[key]["ppv"],
                        "error_rate": stats["error_rate"],
                    },
                )
    return rows


# ── Print narrative table ──────────────────────────────────────────────────────


def print_table(rows: list[HopSummary]) -> None:
    print("\n" + "=" * 88)
    print(
        f"{'Dataset':<14} {'Hop':>10}  {'n':>4}  {'ErrRate':>7}  "
        f"{'AUROC-Out':>9}  {'AUROC-Str':>9}  {'AUROC-Grd':>9}  "
        f"{'PPV-Out':>7}  {'PPV-Str':>7}  {'PPV-Grd':>7}",
    )
    print("-" * 88)

    def fmt(v: float) -> str:
        return f"{v:.3f}" if not np.isnan(v) else "   — "

    for r in rows:
        print(
            f"{r['dataset']:<14} {r['hop_label']:>10}  {r['n']:>4}  {fmt(r['error_rate']):>7}  "
            f"{fmt(r['output_auroc']):>9}  {fmt(r['structural_auroc']):>9}  {fmt(r['grounding_auroc']):>9}  "
            f"{fmt(r['output_ppv']):>7}  {fmt(r['structural_ppv']):>7}  {fmt(r['grounding_ppv']):>7}",
        )
    print("=" * 88)


def _family_value(row: HopSummary, family: str, stat: str) -> float:
    if family == "output":
        if stat == "auroc":
            return row.get("output_auroc", float("nan"))
        if stat == "ppv":
            return row.get("output_ppv", float("nan"))
    if family == "structural":
        if stat == "auroc":
            return row.get("structural_auroc", float("nan"))
        if stat == "ppv":
            return row.get("structural_ppv", float("nan"))
    if family == "grounding":
        if stat == "auroc":
            return row.get("grounding_auroc", float("nan"))
        if stat == "ppv":
            return row.get("grounding_ppv", float("nan"))
    return float("nan")


def print_narrative(rows: list[HopSummary]) -> None:
    """Print the story: does structural AUROC increase with hop count?"""
    print("\n── Structural AUROC lift (multi-hop vs single-hop) ──────────────────")

    single_hop: list[HopSummary] = [r for r in rows if "1-hop" in r["hop_label"]]
    two_hop: list[HopSummary] = [r for r in rows if r["hop_label"] in ("2-hop",)]
    many_hop: list[HopSummary] = [
        r for r in rows if r["hop_label"] in ("3-hop", "4-hop", "5+-hop")
    ]

    def mean_family(subset: list[HopSummary], family: str, key: str) -> float:
        vals: list[float] = [
            _family_value(r, family, key)
            for r in subset
            if not np.isnan(_family_value(r, family, key))
        ]
        return float(np.mean(vals)) if vals else float("nan")

    for stat_key, label in [("auroc", "AUROC"), ("ppv", "PPV@20%")]:
        print(f"\n  {label}:")
        for family in ["output", "structural", "grounding"]:
            s1: float = mean_family(single_hop, family, stat_key)
            s2: float = mean_family(two_hop, family, stat_key)
            sm: float = mean_family(many_hop, family, stat_key)

            def f(v: float) -> str:
                return f"{v:.3f}" if not np.isnan(v) else "  —  "

            lift_2: str = (
                f"{s2 - s1:+.3f}" if not np.isnan(s1) and not np.isnan(s2) else "  —  "
            )
            lift_m: str = (
                f"{sm - s1:+.3f}" if not np.isnan(s1) and not np.isnan(sm) else "  —  "
            )
            print(
                f"    {FAMILY_LABELS[family]:>10}: 1-hop={f(s1)}  2-hop={f(s2)} ({lift_2})  "
                f"3+-hop={f(sm)} ({lift_m})",
            )


def print_ppv_story(rows: list[HopSummary]) -> None:
    """Print PPV > baseline check: when flagged uncertain, are they really more wrong?"""
    print("\n── Positive Predictive Power: PPV vs. baseline error rate ─────────────")
    print(
        "  (PPV@20% means: top-20% most uncertain questions → fraction actually wrong)",
    )
    print(
        "  A good metric has PPV >> baseline (random 20% would match baseline error rate)\n",
    )

    for r in rows:
        base: float = r["error_rate"]
        ds: str = r["dataset"]
        hl: str = r["hop_label"]
        print(f"  {ds} [{hl}]  base_err={base:.2f}")
        for family in ["output", "structural", "grounding"]:
            ppv: float = _family_value(r, family, "ppv")
            lift: float = (
                ppv - base if not np.isnan(ppv) and not np.isnan(base) else float("nan")
            )
            lift_str: str = f"{lift:+.2f}" if not np.isnan(lift) else "  —"
            print(
                f"       {FAMILY_LABELS[family]:>10}: PPV={ppv:.2f}  (lift vs base: {lift_str})",
            )
        print()


# ── Plotting ───────────────────────────────────────────────────────────────────


def plot_hop_auroc(
    rows: list[HopSummary],
    out_dir: Path,
    filename_stem: str = "hop_auroc",
) -> None:
    """Main figure: 3-family AUROC grouped by hop bucket with dataset scatter."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    # Collect data: hop_label → family → list of (dataset, auroc)
    hop_order: list[str] = ["1-hop (proxy)", "2-hop", "3-hop", "4-hop"]
    present_hops: list[str] = sorted(
        set(r["hop_label"] for r in rows),
        key=lambda h: hop_order.index(h) if h in hop_order else 99,
    )
    families: list[FamilyKey] = ["output", "structural", "grounding"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    fig.suptitle(
        "AUROC by metric family across hop complexity (KG-RAG)",
        fontsize=12,
        fontweight="bold",
    )

    import matplotlib.ticker as mticker

    for ax, family in zip(axes, families):
        color: str = FAMILY_COLORS[family]

        xs: list[int]
        ys: list[float]
        ds_labels: list[str]
        xs, ys, ds_labels = [], [], []
        for xi, hl in enumerate(present_hops):
            bucket: list[HopSummary] = [r for r in rows if r["hop_label"] == hl]
            for r in bucket:
                val: float = _family_value(r, family, "auroc")
                if not np.isnan(val):
                    xs.append(xi)
                    ys.append(val)
                    ds_labels.append(r["dataset"])

        # Scatter points
        ax.scatter(xs, ys, color=color, s=70, alpha=0.75, zorder=3)

        # Mean line
        means: list[float] = []
        for xi, hl in enumerate(present_hops):
            bucket_vals: list[float] = [
                _family_value(r, family, "auroc")
                for r in rows
                if r["hop_label"] == hl
                and not np.isnan(_family_value(r, family, "auroc"))
            ]
            means.append(float(np.mean(bucket_vals)) if bucket_vals else float("nan"))

        valid_xi: list[int] = [xi for xi, m in enumerate(means) if not np.isnan(m)]
        valid_m: list[float] = [m for m in means if not np.isnan(m)]
        if len(valid_xi) >= 2:
            ax.plot(
                valid_xi,
                valid_m,
                color=color,
                linewidth=2.5,
                zorder=4,
                marker="D",
                markersize=7,
            )

        # Dataset labels
        for xi, yi, lbl in zip(xs, ys, ds_labels):
            ax.annotate(
                lbl,
                (xi, yi),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=6.5,
                color="#444444",
                alpha=0.9,
            )

        ax.axhline(
            0.5,
            color="#aaaaaa",
            linewidth=1.0,
            linestyle="--",
            zorder=1,
            label="chance",
        )
        ax.set_xticks(range(len(present_hops)))
        ax.set_xticklabels(present_hops, rotation=20, ha="right", fontsize=9)
        ax.set_title(FAMILY_LABELS[family], fontsize=11, fontweight="bold", color=color)
        ax.set_ylim(0.25, 1.0)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        if ax == axes[0]:
            ax.set_ylabel("AUROC", fontsize=10)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{filename_stem}.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{filename_stem}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/{filename_stem}.pdf")


def plot_ppv_bars(
    rows: list[HopSummary],
    out_dir: Path,
    filename_stem: str = "hop_ppv",
) -> None:
    """PPV@20% comparison: single-hop vs multi-hop, per family."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    hop_order: list[str] = ["1-hop (proxy)", "2-hop", "3-hop", "4-hop"]
    present_hops: list[str] = sorted(
        set(r["hop_label"] for r in rows),
        key=lambda h: hop_order.index(h) if h in hop_order else 99,
    )
    families: list[FamilyKey] = ["output", "structural", "grounding"]
    n_hops: int = len(present_hops)
    n_fams: int = len(families)

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bar_width: float = 0.22
    x: NDArray[np.int64] = np.arange(n_hops)

    for fi, family in enumerate(families):
        means: list[float] = []
        for hl in present_hops:
            bucket_vals: list[float] = [
                _family_value(r, family, "ppv")
                for r in rows
                if r["hop_label"] == hl
                and not np.isnan(_family_value(r, family, "ppv"))
            ]
            means.append(float(np.mean(bucket_vals)) if bucket_vals else float("nan"))

        offsets: NDArray[np.float64] = cast(
            NDArray[np.float64],
            x + (fi - 1) * bar_width,
        )
        bars: BarContainer = ax.bar(
            offsets,
            [m if not np.isnan(m) else 0 for m in means],
            width=bar_width,
            color=FAMILY_COLORS[family],
            label=FAMILY_LABELS[family],
            alpha=0.85,
            zorder=3,
        )
        for bar_x, m in zip(offsets, means):
            if not np.isnan(m):
                ax.text(
                    bar_x,
                    m + 0.01,
                    f"{m:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#333333",
                )

    # Mean baseline per hop
    for xi, hl in enumerate(present_hops):
        base_vals: list[float] = [
            r["error_rate"]
            for r in rows
            if r["hop_label"] == hl and not np.isnan(r.get("error_rate", float("nan")))
        ]
        if base_vals:
            base: float = float(np.mean(base_vals))
            ax.hlines(
                base,
                xi - 1.5 * bar_width,
                xi + 1.5 * bar_width,
                colors="#666666",
                linewidths=1.5,
                linestyles="--",
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(present_hops, fontsize=10)
    ax.set_ylabel("Precision@20% (fraction wrong in top-20% uncertain)", fontsize=9)
    ax.set_title(
        "Positive Predictive Power by metric family × hop complexity (KG-RAG)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.annotate(
        "Dashed line = baseline error rate per hop group",
        xy=(0.01, 0.97),
        xycoords="axes fraction",
        fontsize=7.5,
        color="#555555",
        va="top",
    )

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{filename_stem}.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{filename_stem}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/{filename_stem}.pdf")


def plot_per_metric_detail(
    rows_detail: list[PerMetricRow],
    out_dir: Path,
    filename_stem: str = "hop_auroc_detail",
) -> None:
    """Appendix figure: per-metric AUROC heatmap across (metric × hop bucket)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        print("matplotlib/seaborn not available — skipping detail plot")
        return

    hop_order: list[str] = ["1-hop (proxy)", "2-hop", "3-hop", "4-hop"]
    all_metrics: list[str] = [m[1] for m in METRICS]
    families: list[FamilyKey] = [m[2] for m in METRICS]

    present_hops: list[str] = [
        h for h in hop_order if any(r["hop_label"] == h for r in rows_detail)
    ]

    # Build matrix: metrics × hops (mean AUROC across datasets)
    mat: NDArray[np.float64] = np.full((len(all_metrics), len(present_hops)), np.nan)
    for i, (_, label, _, _) in enumerate(METRICS):
        for j, hl in enumerate(present_hops):
            vals: list[float] = [
                r["auroc"]
                for r in rows_detail
                if r["metric"] == label
                and r["hop_label"] == hl
                and not np.isnan(r.get("auroc", float("nan")))
            ]
            if vals:
                mat[i, j] = float(np.mean(vals))

    norm: TwoSlopeNorm = TwoSlopeNorm(vmin=0.25, vcenter=0.50, vmax=0.95)
    cmap: Colormap = sns.diverging_palette(10, 130, s=80, l=50, as_cmap=True)

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(max(4, len(present_hops) * 1.4), 6))
    mask: NDArray[np.bool_] = np.isnan(mat)
    sns.heatmap(
        mat,
        ax=ax,
        mask=mask,
        cmap=cmap,
        norm=norm,
        linewidths=0.4,
        linecolor="#dddddd",
        cbar_kws={"label": "Mean AUROC (across datasets)"},
        xticklabels=present_hops,
        yticklabels=all_metrics,
    )
    for r in range(len(all_metrics)):
        for c in range(len(present_hops)):
            v: np.float64 = mat[r, c]
            if not np.isnan(v):
                ax.text(
                    c + 0.5,
                    r + 0.5,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white" if abs(v - 0.5) > 0.2 else "#1a1a1a",
                )
            else:
                ax.text(
                    c + 0.5,
                    r + 0.5,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#bbbbbb",
                )

    # Family separator lines
    for ri in range(1, len(METRICS)):
        if families[ri] != families[ri - 1]:
            ax.axhline(ri, color="white", linewidth=3.0, zorder=3)

    ax.set_title(
        "Per-metric AUROC by hop complexity (KG-RAG, mean across datasets)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.tick_params(axis="x", rotation=20, labelsize=9, length=0)
    ax.tick_params(axis="y", rotation=0, labelsize=9, length=0)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{filename_stem}.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{filename_stem}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/{filename_stem}.pdf")


def run_hop_stratified_analysis(
    details_by_dataset: Mapping[str, Sequence[HopDetail | Mapping[str, object]]],
    output_dir: str = "paper/figures",
    system_prefix: str = SYSTEM_PREFIX,
    figure_suffix: str = "",
) -> HopReport:
    """Callable API — run the full hop-stratified analysis on in-memory details.

    Parameters
    ----------
    details_by_dataset : mapping of dataset_name → list of per-question detail dicts.
        Each dict must contain the same keys written by experiment.py (e.g.
        ``kg_correct``, ``kg_semantic_entropy``, ``hop_count``, etc.).
    output_dir : directory where PDF/PNG figures are written.
    system_prefix : prefix for metric keys in detail dicts (default "kg").

    Returns
    -------
    dict with keys:
        "rows"         — hop-summary rows (one per dataset × hop bucket)
        "rows_detail"  — per-metric rows
        "saved_figures"— list of written file paths
    """
    # Build per-bucket stats from in-memory dicts (avoids re-reading JSON files)
    rows: list[HopSummary] = []
    rows_detail: list[PerMetricRow] = []

    for ds_name, details in details_by_dataset.items():
        # Group by hop bucket
        buckets: dict[str, list[HopDetail | Mapping[str, object]]] = defaultdict(list)
        for d in details:
            hc: object = d.get("hop_count")
            hl: str = hop_label(hc, ds_name)
            if hl != "unknown":
                buckets[hl].append(d)

        for hl, bucket_details in sorted(buckets.items()):
            stats: HopStats = compute_stats(
                bucket_details,
                hl,
                ds_name,
                system_prefix=system_prefix,
            )
            fa: dict[str, float] = family_avg(stats, "auroc")
            fp: dict[str, float] = family_avg(stats, "ppv")
            rows.append(
                {
                    "dataset": ds_name,
                    "hop_label": hl,
                    "n": stats["n"],
                    "error_rate": stats["error_rate"],
                    "output_auroc": fa.get("output", float("nan")),
                    "structural_auroc": fa.get("structural", float("nan")),
                    "grounding_auroc": fa.get("grounding", float("nan")),
                    "output_ppv": fp.get("output", float("nan")),
                    "structural_ppv": fp.get("structural", float("nan")),
                    "grounding_ppv": fp.get("grounding", float("nan")),
                    "_stats": stats,
                },
            )
            for key, label, family, _ in METRICS:
                if key not in stats:
                    continue
                rows_detail.append(
                    {
                        "dataset": ds_name,
                        "hop_label": hl,
                        "metric": label,
                        "family": family,
                        "n": stats["n"],
                        "auroc": stats[key]["auroc"],
                        "ppv": stats[key]["ppv"],
                        "error_rate": stats["error_rate"],
                    },
                )

    out_path: Path = Path(output_dir)
    saved: list[str] = []
    suffix: str = f"_{figure_suffix}" if figure_suffix else ""

    if rows:
        print_table(rows)
        print_narrative(rows)
        print_ppv_story(rows)

        plot_hop_auroc(rows, out_path, filename_stem=f"hop_auroc{suffix}")
        hop_auroc_pdf: str = str(out_path / f"hop_auroc{suffix}.pdf")
        hop_auroc_png: str = str(out_path / f"hop_auroc{suffix}.png")
        if Path(hop_auroc_pdf).exists():
            saved.append(hop_auroc_pdf)
        if Path(hop_auroc_png).exists():
            saved.append(hop_auroc_png)

        plot_ppv_bars(rows, out_path, filename_stem=f"hop_ppv{suffix}")
        ppv_pdf: str = str(out_path / f"hop_ppv{suffix}.pdf")
        ppv_png: str = str(out_path / f"hop_ppv{suffix}.png")
        if Path(ppv_pdf).exists():
            saved.append(ppv_pdf)
        if Path(ppv_png).exists():
            saved.append(ppv_png)

    if rows_detail:
        try:
            plot_per_metric_detail(
                rows_detail,
                out_path,
                filename_stem=f"hop_auroc_detail{suffix}",
            )
            detail_pdf: str = str(out_path / f"hop_auroc_detail{suffix}.pdf")
            if Path(detail_pdf).exists():
                saved.append(detail_pdf)
        except Exception as e:
            print(f"  [warn] hop_auroc_detail failed: {e}")

    return {"rows": rows, "rows_detail": rows_detail, "saved_figures": saved}


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    print("Loading per-question data...")
    rows: list[HopSummary] = build_hop_table()
    rows_detail: list[PerMetricRow] = build_per_metric_table()

    print_table(rows)
    print_narrative(rows)
    print_ppv_story(rows)

    out_dir: Path = Path("paper/figures")
    plot_hop_auroc(rows, out_dir)
    plot_ppv_bars(rows, out_dir)
    plot_per_metric_detail(rows_detail, out_dir)


if __name__ == "__main__":
    main()
