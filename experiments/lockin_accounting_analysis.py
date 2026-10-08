"""Silent-error accounting from saved paper artifacts.

No retrieval, generation, entity linking, or KG construction is run here.  The
script only reads completed result JSONs and the saved GPS replay artifacts used
by the paper's current depth-matched GPS diagnostic.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import NotRequired, TypeGuard, TypedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.gps_v3_depth_matched import scores_for as gps_scores_for


class ManifestEntry(TypedDict):
    result_path: str


class NamedConfig(TypedDict):
    name: str


class AccountingDetail(TypedDict):
    question_id: str | int
    kg_generation_failed: NotRequired[bool]
    vanilla_generation_failed: NotRequired[bool]
    kg_correct: NotRequired[bool]
    vanilla_correct: NotRequired[bool]
    kg_discrete_semantic_entropy: NotRequired[float | None]
    vanilla_discrete_semantic_entropy: NotRequired[float | None]
    kg_sd_uq: NotRequired[float | None]
    vanilla_sd_uq: NotRequired[float | None]
    kg_support_entailment_uncertainty: NotRequired[float | None]
    vanilla_support_entailment_uncertainty: NotRequired[float | None]
    kg_system_skipped: NotRequired[bool]
    vanilla_system_skipped: NotRequired[bool]
    kg_retrieval_route: NotRequired[str | None]
    kg_route_reason: NotRequired[str | None]


class AccountingConfig(TypedDict):
    config: NamedConfig
    details: list[AccountingDetail]


class AccountingDocument(TypedDict):
    config_results: list[AccountingConfig]


class AccountingRow(TypedDict, total=False):
    qid: str
    slug: str
    policy: str
    correct: bool
    dse: float | None
    sd: float | None
    seu: float | None
    gps: float | None
    route: str
    route_reason: str


class AccountingSummary(TypedDict):
    answered: int
    wrong: int
    silent_wrong: int
    not_silent_wrong: int
    answer_state_max_recall: float | None
    silent_upper_bound_rate: float | None
    inspectable_lower_bound: int
    inspectable_lower_bound_rate: float | None
    fully_deceptive_residue: int
    fully_deceptive_residue_rate_of_silent: float | None
    route_known_silent: int
    buckets: dict[str, int]
    silent_flags: dict[str, int]
    route_states_among_silent: dict[str, int]


class MigrationSummary(TypedDict):
    shared: int
    state_transitions: dict[str, int]
    route_transitions: dict[str, int]
    silent_indicator_transitions: dict[str, int]


class AccountingReport(TypedDict):
    note: str
    definitions: dict[str, str]
    per_run: dict[str, dict[str, AccountingSummary]]
    pooled: dict[str, AccountingSummary]
    paired_migration: dict[str, MigrationSummary]


REPO: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON: str = os.path.join(REPO, "results", "analyses", "lockin_accounting.json")
OUT_TEX: str = os.path.join(REPO, "results", "analyses", "lockin_accounting_tables.tex")
MANIFEST: dict[str, ManifestEntry] = json.load(
    open(os.path.join(REPO, "paper", "figures", "latest_results_manifest.json"))
)

KEY: dict[str, str] = {
    "pubmedqa": "Pubmedqa",
    "realmedqa": "Realmedqa",
    "hotpotqa": "Hotpotqa",
    "hotpotqa_fullwiki": "HotpotqaFullWiki",
    "2wikimultihopqa": "2Wikimultihopqa",
    "musique": "Musique",
}
LABEL: dict[str, str] = {
    "pubmedqa": "PubMedQA",
    "realmedqa": "RealMedQA",
    "hotpotqa": "HotpotQA",
    "hotpotqa_fullwiki": "HotpotQA-FW",
    "2wikimultihopqa": "2WikiMHQA",
    "musique": "MuSiQue",
}
STRICT_RUNS: dict[str, str] = {
    "realmedqa": "results/latest_kg_design_final_metrics/runs/"
    "20260528-174538-realmedqa-n230-full-metrics-evaluation-subset/"
    "mirage_realmedqa_results.json",
    "2wikimultihopqa": "results/latest_kg_design_final_metrics/runs/"
    "20260529-120453-2wikimultihopqa-n250-full-metrics-evaluation-subset/"
    "mirage_2wikimultihopqa_results.json",
}

SD_FLOOR: float = 1e-9


def finite(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def is_silent(row: AccountingRow) -> bool:
    dse: float | None
    sd: float | None
    return (
        finite(dse := row.get("dse"))
        and abs(float(dse)) < 1e-9
        and finite(sd := row.get("sd"))
        and float(sd) <= SD_FLOOR
    )


def gps_cache(slug: str, policy: str, side: str) -> dict[str, float | None]:
    if policy == "vanilla":
        return {}
    try:
        return gps_scores_for(slug, strict=(policy == "strict"), side=side)
    except Exception:
        return {}


def rows_for(slug: str, policy: str) -> list[AccountingRow]:
    if policy == "strict":
        path: str = os.path.join(REPO, STRICT_RUNS[slug])
        cfg_prefix: str = "kg_strict_entity_first"
        prefix: str = "kg"
    else:
        path = os.path.join(REPO, MANIFEST[KEY[slug]]["result_path"])
        cfg_prefix = "kg_entity_first" if policy == "kg" else "dense_floor"
        prefix = "kg" if policy == "kg" else "vanilla"

    doc: AccountingDocument = json.load(open(path))
    details: list[AccountingDetail] = next(
        cfg["details"]
        for cfg in doc["config_results"]
        if cfg["config"]["name"].startswith(cfg_prefix)
    )
    gps: dict[str, float | None] = gps_cache(slug, policy, prefix)
    out: list[AccountingRow] = []
    for r in details:
        if (
            r.get("kg_generation_failed")
            if prefix == "kg"
            else r.get("vanilla_generation_failed")
        ) or (
            r.get("kg_system_skipped")
            if prefix == "kg"
            else r.get("vanilla_system_skipped")
        ):
            continue
        qid: str = str(r["question_id"])
        out.append(
            {
                "qid": qid,
                "slug": slug,
                "policy": policy,
                "correct": bool(
                    (
                        r.get("kg_correct")
                        if prefix == "kg"
                        else r.get("vanilla_correct")
                    )
                ),
                "dse": (
                    r.get("kg_discrete_semantic_entropy")
                    if prefix == "kg"
                    else r.get("vanilla_discrete_semantic_entropy")
                ),
                "sd": (r.get("kg_sd_uq") if prefix == "kg" else r.get("vanilla_sd_uq")),
                "seu": (
                    r.get("kg_support_entailment_uncertainty")
                    if prefix == "kg"
                    else r.get("vanilla_support_entailment_uncertainty")
                ),
                "gps": gps.get(qid) if policy != "vanilla" else None,
                "route": str(r.get("kg_retrieval_route") or "")
                if prefix == "kg"
                else "",
                "route_reason": str(r.get("kg_route_reason") or "")
                if prefix == "kg"
                else "",
            }
        )
    return out


def route_state(row: AccountingRow) -> str:
    route: str = row.get("route") or ""
    reason: str = row.get("route_reason") or ""
    if (
        route in {"entity_first_empty", "empty", "no_graph"}
        or reason == "strict_no_graph_signal"
    ):
        return "empty"
    if route:
        return "populated"
    return "unknown"


def seu_flags(row: AccountingRow) -> bool:
    seu: float | None
    return finite(seu := row.get("seu")) and float(seu) > 0.5


def gps_flags(row: AccountingRow) -> bool:
    gps: float | None
    # For KG policies, GPS abstention is diagnostically meaningful; for dense it
    # is unavailable rather than graph evidence.
    if row["policy"] == "vanilla":
        return False
    return row.get("gps") is None or (
        finite(gps := row.get("gps")) and float(gps) > 0.5
    )


def silent_bucket(row: AccountingRow) -> str:
    if row["correct"]:
        return "correct"
    if not is_silent(row):
        return "wrong_not_silent"
    rstate: str = route_state(row)
    if rstate == "empty":
        return "silent_empty_route"
    if rstate == "unknown":
        if seu_flags(row) or gps_flags(row):
            return "silent_route_unknown_flagged"
        return "silent_route_unknown_calm"
    if seu_flags(row):
        return "silent_populated_seu_contradiction"
    if gps_flags(row):
        return "silent_populated_gps_weak_or_abstained"
    return "silent_all_families_calm"


def account(rows: Iterable[AccountingRow]) -> AccountingSummary:
    rows = list(rows)
    wrong: list[AccountingRow] = [r for r in rows if not r["correct"]]
    silent: list[AccountingRow] = [r for r in wrong if is_silent(r)]
    buckets: Counter[str] = Counter(silent_bucket(r) for r in rows)
    route_known_silent: list[AccountingRow] = [
        r for r in silent if route_state(r) != "unknown"
    ]
    inspectable: list[AccountingRow] = [
        r for r in silent if route_state(r) == "empty" or seu_flags(r) or gps_flags(r)
    ]
    fully_calm: list[AccountingRow] = [
        r
        for r in silent
        if route_state(r) != "empty" and not seu_flags(r) and not gps_flags(r)
    ]
    return {
        "answered": len(rows),
        "wrong": len(wrong),
        "silent_wrong": len(silent),
        "not_silent_wrong": len(wrong) - len(silent),
        "answer_state_max_recall": (len(wrong) - len(silent)) / len(wrong)
        if wrong
        else None,
        "silent_upper_bound_rate": len(silent) / len(wrong) if wrong else None,
        "inspectable_lower_bound": len(inspectable),
        "inspectable_lower_bound_rate": len(inspectable) / len(wrong)
        if wrong
        else None,
        "fully_deceptive_residue": len(fully_calm),
        "fully_deceptive_residue_rate_of_silent": len(fully_calm) / len(silent)
        if silent
        else None,
        "route_known_silent": len(route_known_silent),
        "buckets": dict(buckets),
        "silent_flags": {
            "seu_contradiction": sum(seu_flags(r) for r in silent),
            "gps_weak_or_abstained": sum(gps_flags(r) for r in silent),
            "either_non_answer_family": len(inspectable),
        },
        "route_states_among_silent": dict(Counter(route_state(r) for r in silent)),
    }


def transition_state(row: AccountingRow) -> str:
    if row["correct"]:
        return "correct"
    return "wrong_silent" if is_silent(row) else "wrong_not_silent"


def paired_migration(slug: str) -> MigrationSummary:
    adaptive: dict[str, AccountingRow] = {r["qid"]: r for r in rows_for(slug, "kg")}
    strict: dict[str, AccountingRow] = {r["qid"]: r for r in rows_for(slug, "strict")}
    shared: list[str] = sorted(set(adaptive) & set(strict))
    state: Counter[tuple[str, str]] = Counter()
    route: Counter[tuple[str, str]] = Counter()
    floor: Counter[tuple[bool, bool]] = Counter()
    for qid in shared:
        a, s = adaptive[qid], strict[qid]
        state[(transition_state(a), transition_state(s))] += 1
        route[(route_state(a), route_state(s))] += 1
        floor[(is_silent(a), is_silent(s))] += 1
    return {
        "shared": len(shared),
        "state_transitions": {f"{a}->{b}": n for (a, b), n in sorted(state.items())},
        "route_transitions": {f"{a}->{b}": n for (a, b), n in sorted(route.items())},
        "silent_indicator_transitions": {
            f"{int(a)}->{int(b)}": n for (a, b), n in sorted(floor.items())
        },
    }


def table_row(label: str, d: AccountingSummary) -> str:
    b: dict[str, int] = d["buckets"]
    return (
        f"{label} & {d['wrong']} & {d['silent_wrong']} & "
        f"{d['answer_state_max_recall']:.2f} & "
        f"{b.get('silent_empty_route', 0)} & "
        f"{b.get('silent_populated_seu_contradiction', 0)} & "
        f"{b.get('silent_populated_gps_weak_or_abstained', 0)} & "
        f"{b.get('silent_all_families_calm', 0)} & "
        f"{b.get('silent_route_unknown_flagged', 0) + b.get('silent_route_unknown_calm', 0)} \\\\"
    )


def bounds_row(label: str, d: AccountingSummary) -> str:
    return (
        f"{label} & {d['wrong']} & {d['silent_wrong']} "
        f"({100 * d['silent_upper_bound_rate']:.0f}\\%) & "
        f"{d['inspectable_lower_bound']} ({100 * d['inspectable_lower_bound_rate']:.0f}\\%) & "
        f"{d['fully_deceptive_residue']} "
        f"({100 * d['fully_deceptive_residue_rate_of_silent']:.0f}\\% of silent) \\\\"
    )


def write_tex(out: AccountingReport) -> None:
    pooled: dict[str, AccountingSummary] = out["pooled"]
    lines: list[str] = [
        "% Auto-generated by experiments/lockin_accounting_analysis.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Silent-error accounting. Buckets are mutually exclusive. "
        "The max-recall column is the structural upper bound for answer-dispersion "
        "methods: silent wrong answers provide no sampled-answer disagreement signal. "
        "Route-unknown rows retain missing route metadata rather than being reassigned.}",
        "\\label{tab:silent_accounting}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lrrrrrrrr@{}}",
        "\\toprule",
        "Run & Wrong & Silent & Max answer recall & Empty & Pop.+SEU & Pop.+GPS & Calm & Route unk. \\\\",
        "\\midrule",
        table_row("Dense pooled", pooled["vanilla"]),
        table_row("Adaptive KG pooled", pooled["kg"]),
        table_row("Strict KG pooled", pooled["strict"]),
        table_row("RealMedQA strict", out["per_run"]["realmedqa"]["strict"]),
        table_row("2WikiMHQA strict", out["per_run"]["2wikimultihopqa"]["strict"]),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Lock-in bounds from existing observables. Silent errors are an upper bound; "
        "the inspectable lower bound counts silent errors flagged by an empty route, SEU contradiction, "
        "or weak/abstained GPS.}",
        "\\label{tab:lockin_bounds}",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "Run & Wrong & Silent upper & Inspectable lower & Fully calm residue \\\\",
        "\\midrule",
        bounds_row("Adaptive KG", pooled["kg"]),
        bounds_row("Strict KG", pooled["strict"]),
        bounds_row("RealMedQA strict", out["per_run"]["realmedqa"]["strict"]),
        bounds_row("2WikiMHQA strict", out["per_run"]["2wikimultihopqa"]["strict"]),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    per_run: dict[str, dict[str, AccountingSummary]] = {}
    pooled_rows: dict[str, list[AccountingRow]] = defaultdict(list)

    for slug in KEY:
        per_run[slug] = {}
        for policy in ("vanilla", "kg") + (("strict",) if slug in STRICT_RUNS else ()):
            rows: list[AccountingRow] = rows_for(slug, policy)
            per_run[slug][policy] = account(rows)
            pooled_rows[policy].extend(rows)

    pooled: dict[str, AccountingSummary] = {
        policy: account(rows) for policy, rows in pooled_rows.items()
    }
    migration: dict[str, MigrationSummary] = {
        slug: paired_migration(slug) for slug in STRICT_RUNS
    }
    out: AccountingReport = {
        "note": "Saved-artifact analysis only; no retrieval/generation reruns.",
        "definitions": {
            "silent": "wrong and DSE=0 and SD-UQ <= 1e-9",
            "seu_contradiction": "SEU > 0.5",
            "gps_weak_or_abstained": "GPS undefined/abstained or GPS > 0.5",
            "answer_state_max_recall": "1 - silent_wrong / wrong",
        },
        "per_run": per_run,
        "pooled": pooled,
        "paired_migration": migration,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1, default=float)
    write_tex(out)
    print(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
