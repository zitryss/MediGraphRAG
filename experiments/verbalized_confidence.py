"""V3.4 / review M4: verbalised-confidence P(True) baseline from saved answers.

The paper's P(True) proxy is cluster agreement, which is monotone in DSE
under black-box sampling.  The original P(True) asks the model for a
probability.  This script queries the same generator family (GPT-4o-mini via
OpenRouter) once per saved question/answer pair:

    "Probability that the proposed answer is correct: <float>"

and evaluates AUROC of (1 - p) against the paper's correctness labels, per
dataset, for the adaptive-KG and dense policies.  No retrieval or answer
generation is rerun.  Writes results/analyses/verbalized_confidence.json.
"""

import json
import os
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal, NotRequired, TypeAlias, TypedDict

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.trust_analysis import REPO, MANIFEST, KEY, NamedConfig
from experiments.rejudge_independent import get_client

ConfidenceRow: TypeAlias = tuple[str, str, str, bool]


class ConfidenceDetail(TypedDict):
    question_id: str | int
    question: str
    kg_response: NotRequired[str]
    vanilla_response: NotRequired[str]
    kg_correct: NotRequired[bool]
    vanilla_correct: NotRequired[bool]
    kg_generation_failed: NotRequired[bool]
    vanilla_generation_failed: NotRequired[bool]


class ConfidenceConfig(TypedDict):
    config: NamedConfig
    details: list[ConfidenceDetail]


class ConfidenceDocument(TypedDict):
    config_results: list[ConfidenceConfig]


class ConfidenceSummary(TypedDict):
    n: int
    auroc: float | None
    n_failed: NotRequired[int]
    mean_conf_correct: NotRequired[float | None]
    mean_conf_wrong: NotRequired[float | None]


class ConfidenceReport(TypedDict):
    model: str
    kg: NotRequired[dict[str, ConfidenceSummary]]
    vanilla: NotRequired[dict[str, ConfidenceSummary]]


MODEL: str = "openai/gpt-4o-mini"
SYSTEM: str = (
    "You estimate the probability that a proposed answer to a question "
    "is correct. Respond with ONLY a number between 0 and 1."
)
USER_TMPL: str = (
    "Question: {question}\n"
    "Proposed answer: {response}\n\n"
    "Probability that the proposed answer is correct (0 to 1):"
)


def ask_conf(
    client: "OpenAI", question: str, response: str, retries: int = 3
) -> float | None:
    for attempt in range(retries):
        try:
            r: "ChatCompletion" = client.chat.completions.create(
                model=MODEL,
                temperature=0.0,
                max_tokens=8,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": USER_TMPL.format(
                            question=question, response=response
                        ),
                    },
                ],
            )
            txt: str = (r.choices[0].message.content or "").strip()
            m: re.Match[str] | None = re.search(r"\d*\.?\d+", txt)
            if m:
                p: float | None
                p = float(m.group(0))
                if p > 1:
                    p = p / 100.0 if p <= 100 else None
                if p is not None and 0.0 <= p <= 1.0:
                    return p
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def load_details(slug: str, policy: str) -> list[ConfidenceRow]:
    doc: ConfidenceDocument = json.load(
        open(os.path.join(REPO, MANIFEST[KEY[slug]]["result_path"]))
    )
    cfg_prefix: str = "kg_entity_first" if policy == "kg" else "dense_floor"
    details: list[ConfidenceDetail] = next(
        c for c in doc["config_results"] if c["config"]["name"].startswith(cfg_prefix)
    )["details"]
    prefix: Literal["kg", "vanilla"] = "kg" if policy == "kg" else "vanilla"
    return [
        (
            str(r["question_id"]),
            r["question"],
            r["kg_response"] if prefix == "kg" else r["vanilla_response"],
            bool(r["kg_correct"] if prefix == "kg" else r["vanilla_correct"]),
        )
        for r in details
        if not (
            r.get("kg_generation_failed")
            if prefix == "kg"
            else r.get("vanilla_generation_failed")
        )
    ]


def main() -> None:
    client: "OpenAI" = get_client()
    out: ConfidenceReport = {"model": MODEL}
    policies: tuple[Literal["kg"], Literal["vanilla"]] = ("kg", "vanilla")
    for policy in policies:
        policy_result: dict[str, ConfidenceSummary] = {}
        out[policy] = policy_result
        pooled_u: list[float]
        pooled_w: list[int]
        pooled_u, pooled_w = [], []
        for slug in KEY:
            rows: list[ConfidenceRow] = load_details(slug, policy)
            print(f"[{slug}/{policy}] querying {len(rows)} confidences ...", flush=True)
            with ThreadPoolExecutor(max_workers=8) as ex:
                scorer: Callable[[ConfidenceRow], float | None] = lambda t: ask_conf(
                    client, t[1], t[2]
                )
                ps: list[float | None] = list(ex.map(scorer, rows))
            ok: list[tuple[ConfidenceRow, float]] = [
                (r, p) for r, p in zip(rows, ps) if p is not None
            ]
            u: list[float] = [1.0 - p for _, p in ok]
            w: list[int] = [int(not r[3]) for r, _ in ok]
            auc: float | None = (
                float(roc_auc_score(w, u)) if len(set(w)) > 1 and len(w) > 10 else None
            )
            policy_result[slug] = {
                "n": len(ok),
                "n_failed": len(rows) - len(ok),
                "auroc": auc,
                "mean_conf_correct": float(np.mean([p for (r, p) in ok if r[3]]))
                if any(r[3] for r, _ in ok)
                else None,
                "mean_conf_wrong": float(np.mean([p for (r, p) in ok if not r[3]]))
                if any(not r[3] for r, _ in ok)
                else None,
            }
            pooled_u += u
            pooled_w += w
            print(
                f"[{slug}/{policy}] AUROC={auc if auc is None else round(auc, 3)} "
                f"(n={len(ok)})",
                flush=True,
            )
        policy_result["pooled"] = {
            "n": len(pooled_w),
            "auroc": (
                float(roc_auc_score(pooled_w, pooled_u))
                if len(set(pooled_w)) > 1
                else None
            ),
        }
    dest: str = os.path.join(REPO, "results", "analyses", "verbalized_confidence.json")
    json.dump(out, open(dest, "w"), indent=1, default=float)
    print("wrote", dest)


if __name__ == "__main__":
    main()
