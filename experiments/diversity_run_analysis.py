"""Per-sample graph-state diversity analysis for a fresh diversity run.

Reads a run produced with graph-state tracing enabled (current experiment.py
emits per-question kg_seed_entity_jaccard / kg_path_jaccard / kg_subgraph_jaccard
/ kg_chunk_jaccard and the entropy variants).  Computes:

  1. how stable the retrieval STATE is across the N=5 samples (mean Jaccard per
     family) -- the direct grounding of Eq. 2 beyond chunk overlap; and
  2. among WRONG KG answers, the Spearman coupling between retrieval-state
     overlap and answer-state dispersion (SD-UQ) -- higher overlap should
     predict lower dispersion (the lock-in mechanism).

No generation/retrieval rerun here; this only reads the saved run JSON.
Usage: python -m experiments.diversity_run_analysis <glob-or-path-to-results.json>
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import (
    TYPE_CHECKING,
    Literal,
    NotRequired,
    TypeAlias,
    TypedDict,
    cast,
)

import numpy as np
from scipy.stats import spearmanr

if TYPE_CHECKING:
    from experiments.dataset_adapters import DefaultJSONCodec, JSONValue


OverlapKey: TypeAlias = Literal[
    "kg_seed_entity_jaccard",
    "kg_path_jaccard",
    "kg_subgraph_jaccard",
    "kg_chunk_jaccard",
]


class TraceDetail(TypedDict, total=False):
    kg_generation_failed: bool
    kg_system_skipped: bool
    kg_correct: bool
    kg_graph_state_sample_count: int | None
    kg_seed_entity_jaccard: float | None
    kg_path_jaccard: float | None
    kg_subgraph_jaccard: float | None
    kg_chunk_jaccard: float | None
    kg_sd_uq: float | None
    kg_dominant_seed_entity_fraction: float | None


class TraceConfig(TypedDict):
    config: dict[str, str]
    details: list[TraceDetail]


class TraceDocument(TypedDict):
    config_results: list[TraceConfig]


class StabilitySummary(TypedDict):
    mean: float | None
    median: float | None
    n: int


class CouplingSummary(TypedDict):
    spearman_rho: float | None
    p: float | None
    n: int
    note: NotRequired[str]


class TraceReport(TypedDict):
    run: str
    n_answered: int
    n_with_traces: int
    n_wrong_with_traces: int
    stability_mean_jaccard: dict[str, StabilitySummary]
    coupling_overlap_vs_sduq_amongwrong: dict[str, CouplingSummary]
    dominant_seed_entity_fraction_mean: NotRequired[float | None]


REPO: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT: str = os.path.join(REPO, "results", "analyses", "diversity_run_analysis.json")

OVERLAPS: list[tuple[str, OverlapKey]] = [
    ("seed_entity", "kg_seed_entity_jaccard"),
    ("path", "kg_path_jaccard"),
    ("subgraph", "kg_subgraph_jaccard"),
    ("chunk", "kg_chunk_jaccard"),
]


def load_details(path: str) -> JSONValue:
    doc: JSONValue = cast("DefaultJSONCodec", json).load(open(path))
    for cfg in doc["config_results"]:
        if cfg["config"]["name"].startswith("kg_entity_first"):
            return cfg["details"]
    raise SystemExit("no kg_entity_first config in " + path)


def fin(x: float | str | None) -> bool | np.bool_:
    try:
        return cast(np.bool_, np.isfinite(float(x)))
    except Exception:
        return False


def main(pattern: str) -> None:
    paths: list[str] = (
        sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
    )
    if not paths:
        raise SystemExit("no run JSON matched: " + pattern)
    path: str = paths[-1]
    details: list[JSONValue] = [
        r
        for r in load_details(path)
        if not r.get("kg_generation_failed") and not r.get("kg_system_skipped")
    ]
    # rows with a real multi-sample graph trace
    rows: list[JSONValue] = [
        r for r in details if (r.get("kg_graph_state_sample_count") or 0) >= 2
    ]
    wrong: list[JSONValue] = [r for r in rows if not r.get("kg_correct")]

    out: TraceReport = {
        "run": os.path.relpath(path, REPO),
        "n_answered": len(details),
        "n_with_traces": len(rows),
        "n_wrong_with_traces": len(wrong),
        "stability_mean_jaccard": {},
        "coupling_overlap_vs_sduq_amongwrong": {},
    }

    for name, key in OVERLAPS:
        vals: list[float] = [
            float(value)
            for r in rows
            if (value := r.get(key)) is not None and fin(value)
        ]
        out["stability_mean_jaccard"][name] = {
            "mean": float(np.mean(vals)) if vals else None,
            "median": float(np.median(vals)) if vals else None,
            "n": len(vals),
        }

    for name, key in OVERLAPS:
        xs: list[float]
        ys: list[float]
        xs, ys = [], []
        for r in wrong:
            overlap: float | None = r.get(key)
            sd_uq: float | None = r.get("kg_sd_uq")
            if (
                overlap is not None
                and sd_uq is not None
                and fin(overlap)
                and fin(sd_uq)
            ):
                xs.append(float(overlap))
                ys.append(float(sd_uq))
        if len(xs) >= 5 and len(set(xs)) > 1 and len(set(ys)) > 1:
            rho, p = spearmanr(xs, ys)
            out["coupling_overlap_vs_sduq_amongwrong"][name] = {
                "spearman_rho": float(rho),
                "p": float(p),
                "n": len(xs),
            }
        else:
            out["coupling_overlap_vs_sduq_amongwrong"][name] = {
                "spearman_rho": None,
                "p": None,
                "n": len(xs),
                "note": "too few/constant",
            }

    dsf: list[float] = [
        float(fraction)
        for r in rows
        if (fraction := r.get("kg_dominant_seed_entity_fraction")) is not None
        and fin(fraction)
    ]
    out["dominant_seed_entity_fraction_mean"] = float(np.mean(dsf)) if dsf else None

    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1))
    print("wrote", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    pat: str = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(
            REPO,
            "results",
            "diversity_run",
            "runs",
            "*realmedqa*",
            "mirage_realmedqa_results.json",
        )
    )
    main(pat)
