"""GPS-v2 structural replay — recompute Graph Path Support from saved answer logs.

No generation or retrieval is rerun.  For each answered question in the
canonical reported runs, this script re-resolves question/answer entities
against the persistent dataset KG in Neo4j and recomputes structural support
with two upgrades over GPS-v1:

  (a) soft answer-entity linking: in addition to the v1 surface/fuzzy matcher,
      candidate answer spans are embedded (all-MiniLM-L6-v2) and linked to KG
      entities by cosine similarity >= tau.  Soft links carry their cosine as a
      link weight; v1 surface matches carry weight 1.0.
  (b) graded path support: reachability is depth-probed (L = 1..max_hops) and
      a reachable answer entity contributes gamma^(L-1) instead of 1.

      GPS-v2 = 1 - sum_e(w_e * gamma^(L_e - 1) * reach_e) / sum_e(w_e)

Anti-overfitting protocol (paper revision_plan item 12): tau and gamma are
tuned on the RealMedQA replay only (--grid), frozen, then evaluated once on
the remaining datasets (--frozen).  The question side uses the v1 matcher with
a fixed-threshold soft fallback (TAU_Q) that is NOT part of the grid, so the
per-question path-length store is reusable across grid points.

Usage:
  .venv/bin/python experiments/gps_v2_replay.py --dataset realmedqa --grid
  .venv/bin/python experiments/gps_v2_replay.py --dataset all --tau 0.72 --gamma 0.7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Sequence
from datetime import date
from typing import (
    TYPE_CHECKING,
    Literal,
    LiteralString,
    NotRequired,
    TypeAlias,
    TypedDict,
    cast,
)

import numpy as np
from numpy.typing import NDArray
from typing_extensions import ReadOnly

if TYPE_CHECKING:
    from neo4j import Driver, Result, Session
    from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.uncertainty_metrics import (
    EmbeddingArray,
    ScopedEntity,
    SentenceEncoder,
    _candidate_structural_spans,
    _load_scoped_entities,
    _match_scoped_entities_to_text,
    _primary_answer_span,
    _question_local_entity_support_predicate,
    _question_local_path_support_predicate,
)

if TYPE_CHECKING:
    from experiments.dataset_adapters import DefaultJSONCodec, JSONValue


class SoftMatch(TypedDict):
    id: str
    name: str
    cos: float


class ReplayRowFields(TypedDict):
    question_id: str | int
    question: str
    kg_response: NotRequired[str]
    vanilla_response: NotRequired[str]
    kg_correct: NotRequired[bool]
    vanilla_correct: NotRequired[bool]
    kg_generation_failed: NotRequired[bool]
    vanilla_generation_failed: NotRequired[bool]
    kg_graph_path_support: NotRequired[float | None]
    vanilla_graph_path_support: NotRequired[float | None]


class RawReplayRowFields(TypedDict):
    """Merge reads question_id but does not validate persisted field values."""

    question_id: JSONValue
    question: NotRequired[JSONValue]
    kg_response: NotRequired[JSONValue]
    vanilla_response: NotRequired[JSONValue]
    kg_correct: NotRequired[JSONValue]
    vanilla_correct: NotRequired[JSONValue]
    kg_generation_failed: NotRequired[JSONValue]
    vanilla_generation_failed: NotRequired[JSONValue]
    kg_graph_path_support: NotRequired[JSONValue]
    vanilla_graph_path_support: NotRequired[JSONValue]


class MergedReplayRow(RawReplayRowFields, total=False):
    store: JSONValue | dict[str, V2ReplayStore]


class CompletedReplayRow(RawReplayRowFields):
    store: dict[str, V2ReplayStore]


class KGResponseView(TypedDict):
    kg_response: ReadOnly[JSONValue]


class VanillaResponseView(TypedDict):
    vanilla_response: ReadOnly[JSONValue]


class ReplayConfig(TypedDict):
    """Runner config output always emits its detail list, including empty runs."""

    config: NotRequired[dict[str, str]]
    details: list[MergedReplayRow]


class ReplayDocument(TypedDict, total=False):
    config_results: list[ReplayConfig]


class StoredAnswer(TypedDict, total=False):
    """Sparse saved input accepted by depth-matched v3 scoring."""

    q_sources: list[str]
    id: str
    name: object
    cos: float
    source: str
    L: int | None


class ReplayStore(TypedDict, total=False):
    """Sparse imported store; not the completed v2 producer contract."""

    null_reason: str | None
    q_ids: list[str]
    q_source: str
    answers: list[StoredAnswer]


class LinkedAnswer(TypedDict):
    """Mutable link record before path resolution adds L."""

    id: str
    name: object
    cos: float
    source: str
    q_sources: NotRequired[list[str]]
    L: NotRequired[int | None]


class PathPendingAnswer(TypedDict):
    id: str
    name: object
    cos: float
    source: str
    q_sources: list[str]
    L: NotRequired[int | None]


class V2StoredAnswer(TypedDict):
    """Serialized completed v2 link: path length is present, possibly null."""

    id: str
    name: object
    cos: float
    source: str
    L: int | None
    q_sources: NotRequired[ReadOnly[list[str]]]


class V2NullStore(TypedDict):
    null_reason: Literal["generation_failed", "no_q_entities", "no_a_entities"]
    q_ids: NotRequired[list[str]]
    q_source: NotRequired[str]
    answers: NotRequired[ReadOnly[list[V2StoredAnswer]]]


class V2AnsweredStore(TypedDict):
    null_reason: None
    q_ids: list[str]
    q_source: str
    answers: list[V2StoredAnswer]


V2ReplayStore: TypeAlias = V2NullStore | V2AnsweredStore


class PresentNullReason(TypedDict):
    """Read view after the existing truthy null_reason check."""

    null_reason: ReadOnly[str]


class ReplayScore(TypedDict):
    score: float
    null_reason: str | None


class ReplayQuestion(TypedDict):
    question_id: str | int
    kg_correct: NotRequired[bool | None]
    vanilla_correct: NotRequired[bool | None]
    store: NotRequired[dict[str, ReplayStore]]


class LoggedSummary(TypedDict):
    usable: int
    auroc: float | None


class ReplayArtifact(TypedDict):
    source_result: str
    questions: list[ReplayQuestion]
    _artifact_path: NotRequired[ReadOnly[str]]
    v1_logged: NotRequired[dict[str, LoggedSummary]]


class V2ReplayQuestion(TypedDict):
    question_id: JSONValue
    kg_correct: JSONValue
    vanilla_correct: JSONValue
    store: dict[str, V2ReplayStore]


class V2ReplayArtifact(TypedDict):
    source_result: str
    questions: list[V2ReplayQuestion]
    v1_logged: dict[str, LoggedSummary]


class ReplaySummary(TypedDict):
    tau: float
    gamma: float
    answered: int
    usable: int
    coverage: float | None
    auroc: float | None
    abstain_reasons: dict[str, int]


class ReplayParameters(TypedDict):
    min_name_length: int
    max_entities: int
    tau_loose_store: float
    tau_q_fixed: float
    max_hops: int
    mode: str


class ReplayOutput(V2ReplayArtifact):
    replay: str
    date: str
    dataset: str
    params: ReplayParameters
    summary: dict[str, list[ReplaySummary]]


REPO_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST: str = os.path.join(
    REPO_ROOT,
    "paper",
    "figures",
    "latest_results_manifest.json",
)

DATASETS: dict[str, tuple[str, bool, int]] = {
    # slug -> (manifest key, question_scoped, structural max_hops)
    "pubmedqa": ("Pubmedqa", True, 2),
    "realmedqa": ("Realmedqa", False, 2),
    "hotpotqa": ("Hotpotqa", True, 2),
    "hotpotqa_fullwiki": ("HotpotqaFullWiki", False, 2),
    "2wikimultihopqa": ("2Wikimultihopqa", True, 2),
    "musique": ("Musique", True, 4),
}

DENSE_CFG: str = "dense_floor_thr0.1_k10_rt0p0"
KG_CFG: str = "kg_entity_first_thr0.1_k10_rt0p0"

MIN_NAME_LENGTH: int = 4
MAX_ENTITIES: int = 5  # per side, matching GPS-v1
MAX_SOFT_SPANS: int = 24  # candidate spans embedded per text
TAU_LOOSE: float = float(
    os.environ.get("GPS_TAU_LOOSE", 0.60),
)  # store floor; grid taus filter above this
TAU_Q: float = 0.75  # fixed question-side soft-fallback threshold (not tuned)
GRID_TAU: tuple[float, ...] = tuple(
    float(x) for x in os.environ.get("GPS_GRID_TAU", "0.60,0.65,0.72,0.80").split(",")
)
GRID_GAMMA: tuple[float, ...] = tuple(
    float(x) for x in os.environ.get("GPS_GRID_GAMMA", "0.5,0.7,1.0").split(",")
)


def _load_env() -> dict[str, str]:
    env_path: str = os.path.join(REPO_ROOT, ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)
    return {
        "uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.environ.get("NEO4J_USERNAME", "neo4j"),
        "password": os.environ.get("NEO4J_PASSWORD", "password"),
        "database": os.environ.get("NEO4J_DATABASE", "neo4j"),
    }


def _select_config(doc: JSONValue, name: str) -> JSONValue:
    for cfg in doc.get("config_results", []):
        if cfg.get("config", {}).get("name") == name:
            return cfg
    raise ValueError(f"missing config {name}")


def _merged_rows(
    result_path: str,
    kg_cfg: str = KG_CFG,
    with_dense: bool = True,
) -> list[MergedReplayRow]:
    """vanilla_* fields from the dense config, kg_* from entity-first (paper convention)."""
    doc: JSONValue = cast("DefaultJSONCodec", json).load(open(result_path))
    dense: dict[None | bool | int | float | str, JSONValue] = (
        {r["question_id"]: r for r in _select_config(doc, DENSE_CFG)["details"]}
        if with_dense
        else {}
    )
    rows: list[MergedReplayRow] = []
    for r in _select_config(doc, kg_cfg)["details"]:
        row: JSONValue = r.copy()
        d: JSONValue = dense.get(r["question_id"], {})
        for k, v in d.items():
            if k.startswith("vanilla_"):
                row[k] = v
        # Both the successful string-key read and copy above require a JSON object.
        rows.append(cast(MergedReplayRow, row))
    return rows


class ScopedEntityIndex:
    """Entities (with embeddings) visible in one (kg_name, question_id) scope."""

    def __init__(self, session: Session, kg_name: str, question_id: str | None) -> None:
        scope: LiteralString = _question_local_entity_support_predicate(
            "e",
            kg_name=kg_name,
            question_id=question_id,
        )
        self.rows: Sequence[ScopedEntity] = _load_scoped_entities(
            session,
            entity_scope=scope,
            kg_name=kg_name,
            question_id=question_id,
            min_name_length=MIN_NAME_LENGTH,
        )
        params: dict[str, int | str] = {"min_len": MIN_NAME_LENGTH}
        if kg_name:
            params["kg_name"] = kg_name
        if question_id:
            params["question_id"] = question_id
        emb_rows: Result = session.run(
            f"""
            MATCH (e:__Entity__)
            WHERE size(e.name) >= $min_len AND e.embedding IS NOT NULL AND {scope}
            RETURN DISTINCT e.id AS id, e.name AS name, e.embedding AS emb
            LIMIT 20000
            """,
            params,
        )
        ids: list[str]
        names: list[str]
        embs: list[NDArray[np.float32]]
        ids, names, embs = [], [], []
        for r in emb_rows:
            ids.append(str(r["id"]))
            names.append(str(r["name"]))
            embs.append(np.asarray(r["emb"], dtype=np.float32))
        self.emb_ids: list[str] = ids
        self.emb_names: list[str] = names
        self.emb_matrix: NDArray[np.float32]
        if embs:
            m: NDArray[np.float32] = np.vstack(embs)
            norms: NDArray[np.float32] = np.linalg.norm(m, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.emb_matrix = m / norms
        else:
            self.emb_matrix = np.zeros((0, 384), dtype=np.float32)

    def surface_match(self, text: str) -> list[ScopedEntity]:
        return _match_scoped_entities_to_text(
            text,
            self.rows,
            min_name_length=MIN_NAME_LENGTH,
            max_entities=MAX_ENTITIES,
        )

    def soft_match(
        self,
        text: str,
        encoder: SentenceTransformer,
        tau: float,
    ) -> list[SoftMatch]:
        """Top entities by max cosine between any candidate span and the entity name."""
        if self.emb_matrix.shape[0] == 0:
            return []
        spans: list[str] = _candidate_structural_spans(
            text,
            min_name_length=MIN_NAME_LENGTH,
        )[:MAX_SOFT_SPANS]
        if not spans:
            return []
        span_embs: EmbeddingArray = cast(SentenceEncoder, encoder).encode(
            spans,
            normalize_embeddings=True,
        )
        sims: EmbeddingArray = span_embs @ self.emb_matrix.T  # (spans, entities)
        best: EmbeddingArray = sims.max(axis=0)  # best span per entity
        order: NDArray[np.intp] = np.argsort(-best)
        out: list[SoftMatch] = []
        for idx in order[: MAX_ENTITIES * 3]:
            cos: float = float(best[idx])
            if cos < tau:
                break
            out.append(
                {"id": self.emb_ids[idx], "name": self.emb_names[idx], "cos": cos},
            )
        return out


def _min_path_length(
    session: Session,
    q_ids: list[str],
    a_id: str,
    kg_name: str,
    question_id: str | None,
    max_hops: int,
) -> int | None:
    """Smallest L in 1..max_hops with a qualifying path, mirroring GPS-v1 filters."""
    path_scope: LiteralString = _question_local_path_support_predicate(
        path_var="p",
        kg_name=kg_name,
        question_id=question_id,
    )
    params: dict[str, str | list[str]] = {"q_ids": q_ids, "a_id": a_id}
    if kg_name:
        params["kg_name"] = kg_name
    if question_id:
        params["question_id"] = question_id
    for hop in range(1, max_hops + 1):
        query: str = f"""
        UNWIND $q_ids AS q_id
        MATCH (q_e:__Entity__ {{id: q_id}})
        MATCH (a_e:__Entity__ {{id: $a_id}})
        MATCH p = (q_e)-[*{hop}..{hop}]-(a_e)
        WHERE ALL(r IN relationships(p) WHERE coalesce(r.confidence, 1.0) >= 0.4)
          {path_scope}
        RETURN 1 LIMIT 1
        """
        if list(session.run(query, params, timeout=30)):
            return hop
    return None


def replay_question(
    session: Session,
    encoder: SentenceTransformer,
    index: ScopedEntityIndex,
    question: str,
    answer: str,
    kg_name: str,
    question_id: str | None,
    max_hops: int,
) -> V2ReplayStore:
    """Resolve links + path lengths once, at the loose threshold (store row)."""
    q_matches: list[ScopedEntity] = index.surface_match(question)
    q_source: str = "surface"
    if not q_matches:
        q_matches = [
            {"id": m["id"], "name": m["name"]}
            for m in index.soft_match(question, encoder, TAU_Q)[:MAX_ENTITIES]
        ]
        q_source = "soft"
    q_ids: list[str] = [str(m.get("id")) for m in q_matches if m.get("id")]
    if not q_ids:
        return {"null_reason": "no_q_entities"}

    span: str = _primary_answer_span(answer)
    a_links: dict[str, LinkedAnswer] = {}
    m: ScopedEntity | SoftMatch
    for m in index.surface_match(span):
        a_id: str = str(m.get("id"))
        if a_id:
            a_links[a_id] = {
                "id": a_id,
                "name": m.get("name"),
                "cos": 1.0,
                "source": "surface",
            }
    for m in index.soft_match(span, encoder, TAU_LOOSE):
        if m["id"] not in a_links:
            # soft_match emits exactly id/name/cos; this expansion only adds source.
            a_links[m["id"]] = cast(LinkedAnswer, {**m, "source": "soft"})
    # v1 trivial-overlap rule: an answer entity that is also a question entity
    # does not count, and its path sources exclude itself.
    answers: list[LinkedAnswer] = []
    for a in a_links.values():
        sources: list[str] = [q for q in q_ids if q != a["id"]]
        if not sources:
            continue
        a["q_sources"] = sources
        answers.append(a)
    answers.sort(key=lambda a: (-a["cos"], a["id"]))
    answers = answers[: MAX_ENTITIES * 2]
    if not answers:
        return {"null_reason": "no_a_entities", "q_ids": q_ids, "q_source": q_source}

    for a in answers:
        a["L"] = _min_path_length(
            session,
            # Each retained link received q_sources immediately before append.
            cast(PathPendingAnswer, a)["q_sources"],
            a["id"],
            kg_name,
            question_id,
            max_hops,
        )
        del a["q_sources"]
    return {
        "null_reason": None,
        "q_ids": q_ids,
        "q_source": q_source,
        # The loop assigns L to every retained link and removes q_sources.
        "answers": cast(list[V2StoredAnswer], answers),
    }


def score_from_store(
    store_row: ReplayStore | V2ReplayStore,
    tau: float,
    gamma: float,
) -> ReplayScore:
    """Evaluate one (tau, gamma) grid point from a stored replay row."""
    if store_row.get("null_reason"):
        return {
            "score": 0.5,
            "null_reason": cast(PresentNullReason, store_row)["null_reason"],
        }
    kept: list[StoredAnswer | V2StoredAnswer] = [
        a for a in store_row["answers"] if a["source"] == "surface" or a["cos"] >= tau
    ][:MAX_ENTITIES]
    if not kept:
        return {"score": 0.5, "null_reason": "no_a_entities"}
    num: float = sum(
        a["cos"] * (gamma ** (a["L"] - 1)) for a in kept if a["L"] is not None
    )
    den: float = sum(a["cos"] for a in kept)
    return {"score": float(1.0 - num / den), "null_reason": None}


def auroc(scores: list[float], wrong: list[bool]) -> float | None:
    from sklearn.metrics import roc_auc_score

    if len(set(wrong)) < 2:
        return None
    return float(roc_auc_score([int(w) for w in wrong], scores))


def summarize(
    rows: list[CompletedReplayRow],
    side: str,
    tau: float,
    gamma: float,
) -> ReplaySummary:
    scored: list[float]
    wrong: list[bool]
    reasons: Counter[str]
    scored, wrong, reasons = [], [], Counter()
    answered: int = 0
    for r in rows:
        if (
            r.get("kg_generation_failed")
            if side == "kg"
            else r.get("vanilla_generation_failed")
        ):
            continue
        answered += 1
        res: ReplayScore = score_from_store(r["store"][side], tau, gamma)
        if res["null_reason"]:
            reasons[res["null_reason"]] += 1
            continue
        scored.append(res["score"])
        wrong.append(
            not bool(
                (r.get("kg_correct") if side == "kg" else r.get("vanilla_correct")),
            ),
        )
    return {
        "tau": tau,
        "gamma": gamma,
        "answered": answered,
        "usable": len(scored),
        "coverage": round(len(scored) / answered, 3) if answered else None,
        "auroc": auroc(scored, wrong),
        "abstain_reasons": dict(reasons),
    }


def run_dataset(
    slug: str,
    env: dict[str, str],
    encoder: SentenceTransformer,
    grid: bool,
    tau: float,
    gamma: float,
) -> None:
    from neo4j import GraphDatabase

    manifest_key, scoped, max_hops = DATASETS[slug]
    runfile: str = os.environ.get("GPS_RUNFILE", "")
    kg_cfg: str = os.environ.get("GPS_KG_CONFIG", KG_CFG)
    if runfile:
        result_path: str = os.path.join(REPO_ROOT, runfile)
        rows: list[MergedReplayRow] = _merged_rows(
            result_path,
            kg_cfg=kg_cfg,
            with_dense=False,
        )
    else:
        manifest: JSONValue = cast("DefaultJSONCodec", json).load(open(MANIFEST))
        result_path = os.path.join(REPO_ROOT, manifest[manifest_key]["result_path"])
        rows = _merged_rows(result_path)
    print(f"[{slug}] {len(rows)} questions from {os.path.basename(result_path)}")

    driver: Driver = GraphDatabase.driver(
        env["uri"],
        auth=(env["user"], env["password"]),
    )
    shared_index: ScopedEntityIndex | None = None
    t0: float = time.time()
    with driver.session(database=env["database"]) as session:
        if not scoped:
            shared_index = ScopedEntityIndex(session, slug, None)
            print(
                f"[{slug}] shared index: {len(shared_index.emb_ids)} embedded entities",
            )
        for i, r in enumerate(rows):
            qid: str | None = str(r["question_id"]) if scoped else None
            index: ScopedEntityIndex = shared_index or ScopedEntityIndex(
                session,
                slug,
                qid,
            )
            r["store"] = {}
            sides: tuple[
                tuple[
                    Literal["kg", "vanilla"],
                    Literal["kg_response", "vanilla_response"],
                ],
                ...,
            ] = (
                (("kg", "kg_response"),)
                if runfile
                else (("kg", "kg_response"), ("vanilla", "vanilla_response"))
            )
            for side, resp_key in sides:
                if (
                    r.get("kg_generation_failed")
                    if side == "kg"
                    else r.get("vanilla_generation_failed")
                ) or not r.get(resp_key):
                    r["store"][side] = {"null_reason": "generation_failed"}
                    continue
                r["store"][side] = replay_question(
                    session,
                    encoder,
                    index,
                    r["question"],
                    # The false failed/missing-response guard proves this selected key.
                    cast(KGResponseView, r)["kg_response"]
                    if resp_key == "kg_response"
                    else cast(VanillaResponseView, r)["vanilla_response"],
                    slug,
                    qid,
                    max_hops,
                )
            if (i + 1) % 25 == 0:
                print(f"[{slug}] {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)")
    driver.close()

    # Every row gets store={} before either side is resolved or marked failed.
    completed_rows: list[CompletedReplayRow] = cast(list[CompletedReplayRow], rows)

    grid_points: list[tuple[float, float]] = (
        [(t, g) for t in GRID_TAU for g in GRID_GAMMA] if grid else [(tau, gamma)]
    )
    eval_sides: tuple[Literal["kg", "vanilla"], ...] = (
        ("kg",) if runfile else ("kg", "vanilla")
    )
    summary: dict[str, list[ReplaySummary]] = {
        side: [summarize(completed_rows, side, t, g) for (t, g) in grid_points]
        for side in eval_sides
    }
    # v1 comparison from the logged scores (0.5 sentinel rows excluded — the
    # logs do not retain null reasons, so exact-0.5 is used as the abstention
    # marker, matching how the paper's AUROC pipeline drops sentinel rows).
    v1: dict[str, LoggedSummary] = {}
    s: list[float] | ReplaySummary
    w: list[bool]
    for side in eval_sides:
        s, w = [], []
        for r in rows:
            if (
                r.get("kg_generation_failed")
                if side == "kg"
                else r.get("vanilla_generation_failed")
            ):
                continue
            val: JSONValue = (
                r.get("kg_graph_path_support")
                if side == "kg"
                else r.get("vanilla_graph_path_support")
            )
            if val is None or abs(val - 0.5) < 1e-9:
                continue
            s.append(float(val))
            w.append(
                not bool(
                    (r.get("kg_correct") if side == "kg" else r.get("vanilla_correct")),
                ),
            )
        v1[side] = {"usable": len(s), "auroc": auroc(s, w)}

    out: ReplayOutput = {
        "replay": "gps_v2",
        "date": str(date.today()),
        "dataset": slug,
        "source_result": os.path.relpath(result_path, REPO_ROOT),
        "params": {
            "min_name_length": MIN_NAME_LENGTH,
            "max_entities": MAX_ENTITIES,
            "tau_loose_store": TAU_LOOSE,
            "tau_q_fixed": TAU_Q,
            "max_hops": max_hops,
            "mode": "grid" if grid else "frozen",
        },
        "v1_logged": v1,
        "summary": summary,
        "questions": [
            {
                "question_id": r["question_id"],
                "kg_correct": r.get("kg_correct"),
                "vanilla_correct": r.get("vanilla_correct"),
                "store": r["store"],
            }
            for r in completed_rows
        ],
    }
    suffix: str = "_strict" if runfile else ""
    out_path: str = os.path.join(
        os.path.dirname(result_path),
        f"gps_v2_replay_{slug}{suffix}_{date.today():%Y%m%d}.json",
    )
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"[{slug}] wrote {out_path}")
    for side in eval_sides:
        print(
            f"[{slug}] {side}: v1 usable={v1[side]['usable']} auroc={v1[side]['auroc']}",
        )
        for s in summary[side]:
            print(
                f"[{slug}] {side}: v2 tau={s['tau']} gamma={s['gamma']} "
                f"usable={s['usable']}/{s['answered']} auroc={s['auroc']}",
            )


def main() -> None:
    ap: argparse.ArgumentParser = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="slug or 'all'")
    ap.add_argument("--grid", action="store_true", help="dev grid (RealMedQA only)")
    ap.add_argument("--tau", type=float, default=0.72)
    ap.add_argument("--gamma", type=float, default=0.7)
    args: argparse.Namespace = ap.parse_args()

    if args.grid and args.dataset != "realmedqa":
        ap.error("--grid is restricted to realmedqa (anti-overfitting protocol)")

    env: dict[str, str] = _load_env()
    from sentence_transformers import SentenceTransformer

    encoder: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")
    slugs: list[str] = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for slug in slugs:
        run_dataset(slug, env, encoder, grid=args.grid, tau=args.tau, gamma=args.gamma)


if __name__ == "__main__":
    main()
