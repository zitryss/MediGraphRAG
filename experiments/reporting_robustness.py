"""Reporting robustness checks from saved artefacts only.

This script adds manuscript-facing bookkeeping that reviewers commonly ask
for: answered/failure counts, a small complete-case check, and a
dataset-stratified certificate effect size. It does not rerun retrieval,
generation, judging, or KG construction.
"""

import json
import os
import sys
from collections.abc import Mapping, Sequence
from typing import NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.certificate_eval import CertificateRow, certificate_mask, load_all
from experiments.trust_analysis import (
    KEY,
    MANIFEST,
    REPO,
    STRICT_RUNS,
    CompleteTrustRow,
    TrustConfig,
    TrustDetail,
    TrustDocument,
    TrustRow,
    has_complete_metrics,
    rows_for,
)


class AnsweredCounts(TypedDict):
    total: int
    answered: int
    failures: int
    failure_rate: float | None


class CompleteCase(TypedDict):
    adaptive_answered: int
    strict_answered: int
    both_answered: int


class ContingencyTable(TypedDict):
    cert_correct: int
    cert_wrong: int
    uncert_correct: int
    uncert_wrong: int
    n: NotRequired[int]
    odds_ratio_ha: NotRequired[float]


class CertificateEffect(TypedDict):
    per_dataset_2x2: dict[str, ContingencyTable]
    pooled_2x2: ContingencyTable
    mantel_haenszel_or: float | None
    bootstrap_ci: list[float]
    bootstrap_B: int


class HeadlineEffect(TypedDict):
    auroc: float
    rank_biserial_r: float
    n: int


class HeadlineEffects(TypedDict):
    note: str
    realmedqa_adaptive_sd: HeadlineEffect
    realmedqa_strict_sd: HeadlineEffect
    realmedqa_strict_seu: HeadlineEffect


class RobustnessReport(TypedDict):
    note: str
    answered_counts: dict[str, dict[str, AnsweredCounts]]
    realmedqa_adaptive_strict_complete_case: CompleteCase
    certificate_effect: CertificateEffect
    headline_effect_sizes: HeadlineEffects


RNG: np.random.Generator = np.random.default_rng(42)
B: int = 5000


def answered_counts() -> dict[str, dict[str, AnsweredCounts]]:
    out: dict[str, dict[str, AnsweredCounts]] = {}
    for slug, manifest_key in KEY.items():
        doc: TrustDocument = json.load(
            open(os.path.join(REPO, MANIFEST[manifest_key]["result_path"])),
        )
        out[slug] = {}
        for cfg in doc["config_results"]:
            name: str = cfg["config"]["name"]
            prefix: str = "kg" if name.startswith("kg") else "vanilla"
            policy: str = "kg" if prefix == "kg" else "dense"
            total: int = len(cfg["details"])
            failed: int = sum(
                1 for row in cfg["details"] if row.get(f"{prefix}_generation_failed")
            )
            out[slug][policy] = {
                "total": total,
                "answered": total - failed,
                "failures": failed,
                "failure_rate": failed / total if total else None,
            }
    for slug, rel_path in STRICT_RUNS.items():
        doc = json.load(open(os.path.join(REPO, rel_path)))
        cfg = doc["config_results"][0]
        total = len(cfg["details"])
        failed = sum(1 for row in cfg["details"] if row.get("kg_generation_failed"))
        out.setdefault(slug, {})["strict_kg"] = {
            "total": total,
            "answered": total - failed,
            "failures": failed,
            "failure_rate": failed / total if total else None,
        }
    return out


def realmedqa_adaptive_strict_complete_case() -> CompleteCase:
    adaptive_doc: TrustDocument = json.load(
        open(os.path.join(REPO, MANIFEST[KEY["realmedqa"]]["result_path"])),
    )
    strict_doc: TrustDocument = json.load(
        open(os.path.join(REPO, STRICT_RUNS["realmedqa"])),
    )
    adaptive_rows: list[TrustDetail] = next(
        cfg["details"]
        for cfg in adaptive_doc["config_results"]
        if cfg["config"]["name"].startswith("kg")
    )
    strict_rows: list[TrustDetail] = strict_doc["config_results"][0]["details"]
    adaptive_answered: set[str] = {
        str(row["question_id"])
        for row in adaptive_rows
        if not row.get("kg_generation_failed")
    }
    strict_answered: set[str] = {
        str(row["question_id"])
        for row in strict_rows
        if not row.get("kg_generation_failed")
    }
    return {
        "adaptive_answered": len(adaptive_answered),
        "strict_answered": len(strict_answered),
        "both_answered": len(adaptive_answered & strict_answered),
    }


def odds_ratio_mh(tables: Mapping[str, ContingencyTable]) -> float | None:
    """Mantel-Haenszel common odds ratio over per-dataset 2x2 tables."""
    num: float = 0.0
    den: float = 0.0
    for table in tables.values():
        a: int = table["cert_correct"]
        b: int = table["cert_wrong"]
        c: int = table["uncert_correct"]
        d: int = table["uncert_wrong"]
        n: int = a + b + c + d
        if n:
            num += (a * d) / n
            den += (b * c) / n
    if den == 0:
        return None
    return num / den


def certificate_effect() -> CertificateEffect:
    data: dict[str, list[CertificateRow]] = load_all()
    tables: dict[str, ContingencyTable] = {}
    row_cache: dict[str, NDArray[np.object_]] = {}
    for slug, rows in data.items():
        sd_med: float = float(np.median([sd for sd, _, _, _ in rows]))
        selected: NDArray[np.bool_ | np.float64] = certificate_mask(rows, sd_med)
        correct: NDArray[np.bool_] = np.array([c for _, _, _, c in rows], dtype=bool)
        tables[slug] = {
            "n": len(rows),
            "cert_correct": int((selected & correct).sum()),
            "cert_wrong": int((selected & ~correct).sum()),
            "uncert_correct": int((~selected & correct).sum()),
            "uncert_wrong": int((~selected & ~correct).sum()),
        }
        row_cache[slug] = np.array(
            [(bool(selected[i]), bool(correct[i])) for i in range(len(rows))],
            dtype=object,
        )

    pooled: ContingencyTable = {
        "cert_correct": sum(t["cert_correct"] for t in tables.values()),
        "cert_wrong": sum(t["cert_wrong"] for t in tables.values()),
        "uncert_correct": sum(t["uncert_correct"] for t in tables.values()),
        "uncert_wrong": sum(t["uncert_wrong"] for t in tables.values()),
    }
    pooled["odds_ratio_ha"] = (
        (pooled["cert_correct"] + 0.5)
        * (pooled["uncert_wrong"] + 0.5)
        / ((pooled["cert_wrong"] + 0.5) * (pooled["uncert_correct"] + 0.5))
    )
    point: float | None = odds_ratio_mh(tables)

    boots: list[float] = []
    for _ in range(B):
        boot_tables: dict[str, ContingencyTable] = {}
        for slug, cached_rows in row_cache.items():
            idx: NDArray[np.int64] = RNG.integers(0, len(cached_rows), len(cached_rows))
            sample: NDArray[np.object_] = cached_rows[idx]
            selected = np.array([x[0] for x in sample], dtype=bool)
            correct = np.array([x[1] for x in sample], dtype=bool)
            boot_tables[slug] = {
                "cert_correct": int((selected & correct).sum()),
                "cert_wrong": int((selected & ~correct).sum()),
                "uncert_correct": int((~selected & correct).sum()),
                "uncert_wrong": int((~selected & ~correct).sum()),
            }
        value: float | None = odds_ratio_mh(boot_tables)
        if value is not None and np.isfinite(value):
            boots.append(value)

    return {
        "per_dataset_2x2": tables,
        "pooled_2x2": pooled,
        "mantel_haenszel_or": point,
        "bootstrap_ci": [
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)),
        ],
        "bootstrap_B": B,
    }


def auroc(scores: Sequence[float], wrong: Sequence[bool]) -> float:
    return float(roc_auc_score([int(x) for x in wrong], scores))


def headline_effect_sizes() -> HeadlineEffects:
    adaptive: list[TrustRow] = [
        row for row in rows_for("realmedqa", "kg") if row["sd"] is not None
    ]
    strict: list[CompleteTrustRow] = [
        row for row in rows_for("realmedqa", "strict") if has_complete_metrics(row)
    ]
    adaptive_sd: float = auroc(
        [float(sd) for row in adaptive if (sd := row["sd"]) is not None],
        [not row["correct"] for row in adaptive],
    )
    strict_sd: float = auroc(
        [float(row["sd"]) for row in strict],
        [not row["correct"] for row in strict],
    )
    strict_seu: float = auroc(
        [float(row["seu"]) for row in strict],
        [not row["correct"] for row in strict],
    )

    def rb(value: float) -> float:
        return 2.0 * value - 1.0

    return {
        "note": "Rank-biserial r is reported as 2*AUROC-1, with incorrect answers as the positive class.",
        "realmedqa_adaptive_sd": {
            "auroc": adaptive_sd,
            "rank_biserial_r": rb(adaptive_sd),
            "n": len(adaptive),
        },
        "realmedqa_strict_sd": {
            "auroc": strict_sd,
            "rank_biserial_r": rb(strict_sd),
            "n": len(strict),
        },
        "realmedqa_strict_seu": {
            "auroc": strict_seu,
            "rank_biserial_r": rb(strict_seu),
            "n": len(strict),
        },
    }


def main() -> None:
    out: RobustnessReport = {
        "note": "Saved-artifact reporting checks; no KG/retrieval/generation reruns.",
        "answered_counts": answered_counts(),
        "realmedqa_adaptive_strict_complete_case": realmedqa_adaptive_strict_complete_case(),
        "certificate_effect": certificate_effect(),
        "headline_effect_sizes": headline_effect_sizes(),
    }
    dest: str = os.path.join(REPO, "results", "analyses", "reporting_robustness.json")
    json.dump(out, open(dest, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
