"""Certificate evaluation (review-6 Issue 3): baselines + split-half.

(a) Matched-coverage baselines: SD-UQ-only gate (per-dataset percentile)
    accepting the same pooled coverage as the conjunctive certificate.
(b) Split-half validation: the only data-dependent threshold (the
    per-dataset SD-UQ median) is computed on a random half and the
    certificate is evaluated on the other half (B repeats).
(c) Per-dataset certificate coverage/precision.

Writes results/analyses/certificate_eval.json.
"""

import json
import os
import sys
from collections.abc import Callable, Sequence
from typing import NotRequired, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.trust_analysis import (
    KEY,
    REPO,
    CompleteTrustRow,
    gps_scores,
    has_complete_metrics,
    rows_for,
)

CertificateRow: TypeAlias = tuple[float, float, float | None, bool]


class CertificateSummary(TypedDict):
    coverage: float
    precision: float | None
    selected: int
    n: NotRequired[int]


class SplitHalfSummary(TypedDict):
    precision_mean: float
    precision_p5: float
    precision_p95: float
    coverage_mean: float
    B: int


class CertificateReport(TypedDict, total=False):
    per_dataset_certificate: dict[str, CertificateSummary]
    pooled_certificate: CertificateSummary
    baseline_sd_only: CertificateSummary
    baseline_combined_rank: CertificateSummary
    split_half: SplitHalfSummary


RNG: np.random.Generator = np.random.default_rng(42)
B: int = 200


def load_all() -> dict[str, list[CertificateRow]]:
    data: dict[str, list[CertificateRow]] = {}
    for slug in KEY:
        rows: list[CompleteTrustRow] = [
            r for r in rows_for(slug, "kg") if has_complete_metrics(r)
        ]
        gps: dict[str, float | None] = gps_scores(slug)
        data[slug] = [
            (float(r["sd"]), float(r["seu"]), gps.get(r["qid"]), bool(r["correct"]))
            for r in rows
        ]
    return data


def certificate_mask(
    rows: Sequence[CertificateRow],
    sd_thresh: float,
) -> NDArray[np.bool_ | np.float64]:
    return np.array(
        [
            sd <= sd_thresh and seu <= 0.5 and g is not None and g <= 0.5
            for sd, seu, g, _ in rows
        ],
    )


def main() -> None:
    data: dict[str, list[CertificateRow]] = load_all()
    out: CertificateReport = {}

    # (c) per-dataset certificate, thresholds in-sample (as in the paper)
    per_ds: dict[str, CertificateSummary] = {}
    sel_total: int
    corr_total: int
    n_total: int
    sel_total = corr_total = n_total = 0
    for slug, rows in data.items():
        sd_med: float = float(np.median([sd for sd, _, _, _ in rows]))
        m: NDArray[np.bool_ | np.float64] = certificate_mask(rows, sd_med)
        y: NDArray[np.bool_ | np.float64] = np.array([c for _, _, _, c in rows])
        per_ds[slug] = {
            "n": len(rows),
            "coverage": float(m.mean()),
            "selected": int(m.sum()),
            "precision": float(y[m].mean()) if m.sum() else None,
        }
        sel_total += int(m.sum())
        corr_total += int(y[m].sum())
        n_total += len(rows)
    pooled_cov: float = sel_total / n_total
    out["per_dataset_certificate"] = per_ds
    out["pooled_certificate"] = {
        "coverage": pooled_cov,
        "precision": corr_total / sel_total,
        "selected": sel_total,
        "n": n_total,
    }

    # (a) matched-coverage baselines at the same pooled coverage
    def gate_baseline(
        score_fn: Callable[[CertificateRow], float],
        name: str,
    ) -> CertificateSummary:
        sel: int
        corr: int
        sel = corr = 0
        for slug, rows in data.items():
            scores: NDArray[np.float64] = np.array([score_fn(r) for r in rows])
            y: NDArray[np.bool_ | np.float64] = np.array([c for _, _, _, c in rows])
            k: int = int(round(pooled_cov * len(rows)))
            if k == 0:
                continue
            idx: NDArray[np.intp] = np.argsort(scores)[:k]
            sel += k
            corr += int(y[idx].sum())
        return {"coverage": sel / n_total, "precision": corr / sel, "selected": sel}

    out["baseline_sd_only"] = gate_baseline(lambda r: r[0], "sd")

    # percentile-rank combined (the audit score) at matched coverage
    def combined_rank(rows: Sequence[CertificateRow]) -> NDArray[np.float64]:
        sd: NDArray[np.float64] = np.array([r[0] for r in rows])
        seu: NDArray[np.float64] = np.array([r[1] for r in rows])
        gpsr: NDArray[np.float64] = np.array(
            [1.0 if r[2] is None else r[2] for r in rows],
        )

        def pr(v: NDArray[np.float64]) -> NDArray[np.float64]:
            order: NDArray[np.intp] = v.argsort().argsort()
            return order / max(1, len(v) - 1)

        return (pr(sd) + pr(seu) + pr(gpsr)) / 3

    sel: int
    corr: int
    sel = corr = 0
    for slug, rows in data.items():
        scores: NDArray[np.float64] = combined_rank(rows)
        y = np.array([c for _, _, _, c in rows])
        k: int = int(round(pooled_cov * len(rows)))
        if k == 0:
            continue
        idx: NDArray[np.intp] = np.argsort(scores)[:k]
        sel += k
        corr += int(y[idx].sum())
    out["baseline_combined_rank"] = {
        "coverage": sel / n_total,
        "precision": corr / sel,
        "selected": sel,
    }

    # (b) split-half threshold validation
    precs: list[float]
    covs: list[float]
    precs, covs = [], []
    for _ in range(B):
        n_eval: int
        sel = corr = n_eval = 0
        for slug, rows in data.items():
            idx = RNG.permutation(len(rows))
            half: int = len(rows) // 2
            dev: NDArray[np.int64]
            test: NDArray[np.int64]
            dev, test = idx[:half], idx[half:]
            sd_med = float(np.median([rows[i][0] for i in dev]))
            test_rows: list[CertificateRow] = [rows[i] for i in test]
            m = certificate_mask(test_rows, sd_med)
            y = np.array([c for _, _, _, c in test_rows])
            sel += int(m.sum())
            corr += int(y[m].sum())
            n_eval += len(test_rows)
        if sel:
            precs.append(corr / sel)
            covs.append(sel / n_eval)
    out["split_half"] = {
        "precision_mean": float(np.mean(precs)),
        "precision_p5": float(np.percentile(precs, 5)),
        "precision_p95": float(np.percentile(precs, 95)),
        "coverage_mean": float(np.mean(covs)),
        "B": B,
    }

    dest: str = os.path.join(REPO, "results", "analyses", "certificate_eval.json")
    json.dump(out, open(dest, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
