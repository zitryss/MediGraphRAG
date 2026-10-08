"""Compute every paper-facing number that changes under the final GPS metric.

Reads the GPS-v2 replay artifacts (experiments/gps_v2_replay.py) and the
canonical run JSONs, then emits one JSON with:
  - per-dataset adaptive GPS: usable/answered, AUROC + bootstrap CI, AUREC
  - strict-run GPS (KG side)
  - 2Wiki hop-wise GPS AUC per policy
  - pooled Spearman correlations between families (KG side, non-abstained)
  - composite audit table (SD-UQ / SEU / GPS-risk / combined, AUC + AUREC)
  - case-study question lookups

Frozen GPS parameters: tau=0.60, gamma=0.4 (selected on RealMedQA only).
"""

import glob
import json
import os
import sys
from collections.abc import Sequence
from typing import NotRequired, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.gps_v2_replay import (
    score_from_store,
    ReplayArtifact,
    ReplayScore,
    ReplayStore,
    LoggedSummary,
)


class PaperDetail(TypedDict, total=False):
    question_id: str | int
    question: NotRequired[str]
    hop_count: NotRequired[int | None]
    kg_sd_uq: NotRequired[float | None]
    kg_support_entailment_uncertainty: NotRequired[float | None]


class PaperConfig(TypedDict):
    config: dict[str, str]
    details: list[PaperDetail]


class PaperDocument(TypedDict):
    config_results: list[PaperConfig]


class PaperScoreSummary(TypedDict):
    answered: int
    usable: int
    auroc: float | None
    ci: list[float | None]
    aurec: NotRequired[float | None]
    v1_logged: NotRequired[LoggedSummary]


class PaperComposite(TypedDict):
    n: int
    sd_auc: float | None
    seu_auc: float | None
    gps_risk_auc: float | None
    combined_auc: float | None
    combined_aurec: float


class PaperCorrelation(TypedDict):
    rho: float
    ci: list[float]
    n: int


class HopSummary(TypedDict):
    n: int
    auroc: float | None


class CaseLink(TypedDict):
    name: str | None
    cos: float
    L: int | None
    source: str | None


class CaseSummary(TypedDict):
    question_id: str | int
    question: str
    gps_v2: float | None
    null: str | None
    links: list[CaseLink]


class CaseReplayScore(TypedDict):
    score: float | None
    null_reason: str | None


class PaperReport(TypedDict):
    frozen: dict[str, float]
    adaptive: dict[str, dict[str, PaperScoreSummary]]
    strict: dict[str, PaperScoreSummary]
    hopwise_2wiki: dict[str, HopSummary]
    correlations: dict[str, PaperCorrelation]
    composite: dict[str, PaperComposite]
    cases: dict[str, CaseSummary]


REPO: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS: str = os.path.join(REPO, "results", "latest_kg_design_final_metrics", "runs")
TAU, GAMMA = 0.60, 0.4
B: int = 2000
RNG: np.random.Generator = np.random.default_rng(42)

SLUGS: list[str] = [
    "pubmedqa",
    "realmedqa",
    "hotpotqa",
    "hotpotqa_fullwiki",
    "2wikimultihopqa",
    "musique",
]


def latest_artifact(slug: str, strict: bool = False) -> str | None:
    tag: str = (
        f"gps_v2_replay_{slug}_strict_[0-9]*.json"
        if strict
        else f"gps_v2_replay_{slug}_[0-9]*.json"
    )
    paths: list[str] = glob.glob(os.path.join(RUNS, "*", tag))
    return sorted(paths)[-1] if paths else None


def v2_scores(art: ReplayArtifact, side: str) -> list[tuple[str, float | None, bool]]:
    """(question_id, score|None, correct, answered) per question."""
    out: list[tuple[str, float | None, bool]] = []
    for q in art["questions"]:
        st: ReplayStore | None = (q.get("store") or {}).get(side)
        if st is None or st.get("null_reason") == "generation_failed":
            continue
        res: ReplayScore = score_from_store(st, TAU, GAMMA)
        score: float | None = None if res["null_reason"] else res["score"]
        out.append((str(q["question_id"]), score, bool(q.get(f"{side}_correct"))))
    return out


def auroc_ci(
    scores: ArrayLike, wrong: ArrayLike
) -> tuple[float | None, float | None, float | None]:
    s, w = np.asarray(scores, float), np.asarray(wrong, int)
    if len(set(w)) < 2:
        return None, None, None
    a: float = float(roc_auc_score(w, s))
    boot: list[float] = []
    for _ in range(B):
        idx: NDArray[np.int64] = RNG.integers(0, len(s), len(s))
        if len(set(w[idx])) < 2:
            continue
        boot.append(roc_auc_score(w[idx], s[idx]))
    return a, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def aurec(correct: ArrayLike, uncertainty: ArrayLike) -> float:
    """Mean error over rejection levels, rejecting most-uncertain first."""
    y: NDArray[np.float64] = np.asarray(correct, float)
    u: NDArray[np.float64] = np.asarray(uncertainty, float)
    order: NDArray[np.intp] = np.argsort(-u)
    errors: NDArray[np.float64] = 1 - y[order]
    suffix: NDArray[np.float64] = np.cumsum(errors[::-1])[::-1]
    remaining: NDArray[np.float64] = np.arange(len(y), 0, -1, dtype=float)
    return float((suffix / remaining).mean())


def percentile_ranks(values: Sequence[float] | NDArray[np.float64]) -> list[float]:
    order: list[int] = sorted(range(len(values)), key=lambda i: values[i])
    ranks: list[float] = [0.0] * len(values)
    if len(values) <= 1:
        return ranks
    for rank, idx in enumerate(order):
        ranks[idx] = rank / (len(values) - 1)
    return ranks


def main() -> None:
    out: PaperReport = {
        "frozen": {"tau": TAU, "gamma": GAMMA},
        "adaptive": {},
        "strict": {},
        "hopwise_2wiki": {},
        "correlations": {},
        "composite": {},
        "cases": {},
    }

    manifest: dict[str, dict[str, str]] = json.load(
        open(os.path.join(REPO, "paper", "figures", "latest_results_manifest.json"))
    )
    key_for: dict[str, str] = {
        "pubmedqa": "Pubmedqa",
        "realmedqa": "Realmedqa",
        "hotpotqa": "Hotpotqa",
        "hotpotqa_fullwiki": "HotpotqaFullWiki",
        "2wikimultihopqa": "2Wikimultihopqa",
        "musique": "Musique",
    }

    pooled: dict[str, list[float]] = {"sd": [], "seu": [], "gps": []}
    for slug in SLUGS:
        art: ReplayArtifact = json.load(open(latest_artifact(slug)))
        run_doc: PaperDocument = json.load(
            open(os.path.join(REPO, manifest[key_for[slug]]["result_path"]))
        )
        kg_rows: dict[str, PaperDetail] = {}
        for cfg in run_doc["config_results"]:
            if cfg["config"]["name"].startswith("kg_entity_first"):
                kg_rows = {str(r["question_id"]): r for r in cfg["details"]}
        ds: dict[str, PaperScoreSummary] = {}
        for side in ("kg", "vanilla"):
            rows: list[tuple[str, float | None, bool]] = v2_scores(art, side)
            answered: int = len(rows)
            defined: list[tuple[float, bool]] = [
                (s, not c) for _, s, c in rows if s is not None
            ]
            scores: list[float] = [s for s, _ in defined]
            wrong: list[bool] = [w for _, w in defined]
            a, lo, hi = auroc_ci(scores, wrong) if defined else (None, None, None)
            ds[side] = {
                "answered": answered,
                "usable": len(defined),
                "auroc": a,
                "ci": [lo, hi],
                "aurec": aurec([not w for w in wrong], scores) if defined else None,
            }
        out["adaptive"][slug] = ds

        # pooled correlations + composite (KG side)
        comp_rows: list[tuple[float, float, float, bool]] = []
        for qid, s, c in v2_scores(art, "kg"):
            r: PaperDetail = kg_rows.get(qid, {})
            sd: float | None = r.get("kg_sd_uq")
            seu: float | None = r.get("kg_support_entailment_uncertainty")
            if sd is None or seu is None:
                continue
            if s is not None:
                pooled["sd"].append(float(sd))
                pooled["seu"].append(float(seu))
                pooled["gps"].append(float(s))
            comp_rows.append(
                (float(sd), float(seu), 1.0 if s is None else float(s), bool(c))
            )
        if len(comp_rows) >= 12:
            sd_r: list[float] = percentile_ranks([r[0] for r in comp_rows])
            seu_r: list[float] = percentile_ranks([r[1] for r in comp_rows])
            gps_r: list[float] = percentile_ranks([r[2] for r in comp_rows])
            combined: list[float] = [
                (a + b + c) / 3 for a, b, c in zip(sd_r, seu_r, gps_r)
            ]
            wrong = [not r[3] for r in comp_rows]
            correct: list[bool] = [r[3] for r in comp_rows]
            out["composite"][slug] = {
                "n": len(comp_rows),
                "sd_auc": auroc_ci([r[0] for r in comp_rows], wrong)[0],
                "seu_auc": auroc_ci([r[1] for r in comp_rows], wrong)[0],
                "gps_risk_auc": auroc_ci([r[2] for r in comp_rows], wrong)[0],
                "combined_auc": auroc_ci(combined, wrong)[0],
                "combined_aurec": aurec(correct, combined),
            }

    # pooled Spearman with bootstrap CIs
    def spearman_ci(x: ArrayLike, y: ArrayLike) -> PaperCorrelation:
        x, y = np.asarray(x), np.asarray(y)
        rho: float = float(spearmanr(x, y).statistic)
        boot: list[float] = []
        for _ in range(B):
            idx: NDArray[np.int64] = RNG.integers(0, len(x), len(x))
            boot.append(spearmanr(x[idx], y[idx]).statistic)
        return {
            "rho": rho,
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n": int(len(x)),
        }

    out["correlations"]["sd_vs_seu"] = spearman_ci(pooled["sd"], pooled["seu"])
    out["correlations"]["sd_vs_gps"] = spearman_ci(pooled["sd"], pooled["gps"])
    out["correlations"]["seu_vs_gps"] = spearman_ci(pooled["seu"], pooled["gps"])

    # strict runs (KG side only)
    for slug in ("realmedqa", "2wikimultihopqa"):
        path: str | None = latest_artifact(slug, strict=True)
        art = json.load(open(path))
        rows = v2_scores(art, "kg")
        defined = [(s, not c) for _, s, c in rows if s is not None]
        a, lo, hi = auroc_ci([s for s, _ in defined], [w for _, w in defined])
        out["strict"][slug] = {
            "answered": len(rows),
            "usable": len(defined),
            "auroc": a,
            "ci": [lo, hi],
            "v1_logged": art["v1_logged"]["kg"],
        }

    # 2wiki hop-wise GPS AUC per policy
    adaptive_art: ReplayArtifact = json.load(open(latest_artifact("2wikimultihopqa")))
    strict_art: ReplayArtifact = json.load(
        open(latest_artifact("2wikimultihopqa", strict=True))
    )
    run_doc = json.load(
        open(os.path.join(REPO, manifest["2Wikimultihopqa"]["result_path"]))
    )
    hop_by_qid: dict[str, int | None] = {}
    for cfg in run_doc["config_results"]:
        for r in cfg["details"]:
            hop_by_qid[str(r["question_id"])] = r.get("hop_count")
    for label, art, side in [
        ("dense", adaptive_art, "vanilla"),
        ("adaptive_kg", adaptive_art, "kg"),
        ("strict_kg", strict_art, "kg"),
    ]:
        for hop in (2, 4):
            sel: list[tuple[float, bool]] = [
                (s, not c)
                for qid, s, c in v2_scores(art, side)
                if s is not None and hop_by_qid.get(qid) == hop
            ]
            a = (
                auroc_ci([s for s, _ in sel], [w for _, w in sel])[0]
                if len(sel) >= 5
                else None
            )
            out["hopwise_2wiki"][f"{label}_{hop}hop"] = {"n": len(sel), "auroc": a}

    # case-study lookups
    musique_art: ReplayArtifact = json.load(open(latest_artifact("musique")))
    twiki_art: ReplayArtifact = adaptive_art
    case_specs: dict[str, tuple[ReplayArtifact, str]] = {
        "mickey_mouse": (musique_art, "Mickey Mouse"),
        "fire_eater": (twiki_art, "Fire-Eater"),
        "the_cup": (twiki_art, "The Cup (1999"),
    }
    qtext: dict[str, str] = {}
    for cfg in json.load(open(os.path.join(REPO, manifest["Musique"]["result_path"])))[
        "config_results"
    ]:
        for r in cfg["details"]:
            qtext[str(r["question_id"])] = r.get("question", "")
    for cfg in run_doc["config_results"]:
        for r in cfg["details"]:
            qtext[str(r["question_id"])] = r.get("question", "")
    for name, (art, needle) in case_specs.items():
        for q in art["questions"]:
            if needle.lower() in qtext.get(str(q["question_id"]), "").lower():
                st: ReplayStore | None = (q.get("store") or {}).get("kg")
                res: ReplayScore | CaseReplayScore = (
                    score_from_store(st, TAU, GAMMA)
                    if st
                    else {"score": None, "null_reason": "missing"}
                )
                out["cases"][name] = {
                    "question_id": q["question_id"],
                    "question": qtext.get(str(q["question_id"]), "")[:90],
                    "gps_v2": res["score"],
                    "null": res["null_reason"],
                    "links": [
                        {
                            "name": a.get("name"),
                            "cos": round(float(a.get("cos", 0)), 2),
                            "L": a.get("L"),
                            "source": a.get("source"),
                        }
                        for a in (st or {}).get("answers", [])[:4]
                    ],
                }
                break

    dest: str = os.path.join(REPO, "results", "analyses", "gps_v2_paper_numbers.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
