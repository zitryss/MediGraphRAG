"""
retrieval_audit.py — evidence-level diagnostic for the KG vs vanilla recall gap.

For each question in an evaluation run:
  1. Classify: vanilla-only / kg-only / both-correct / neither
  2. Identify gold supporting-fact passages (from raw HotpotQA supporting_facts)
  3. For each SF passage, check:
       a. In graph?         (chunk with questionId + passageIndex exists in Neo4j)
       b. Has entities?     (chunk linked via HAS_ENTITY)
       c. Has embedding?    (chunk.embedding IS NOT NULL)
       d. Retrieved by KG?  (chunk text appears in what vanilla vector search returns
                             -- used as a proxy for retrievability)
  4. Compute per-question SF recall for vanilla and KG retrieval

Outputs:
  - retrieval_audit_<run_id>.json   full per-question detail
  - retrieval_audit_<run_id>.txt    human-readable summary

Usage:
    python -m experiments.retrieval_audit \\
        --run-dir results/runs/<run_id> \\
        --questions-file <path-to-questions-json> \\
        [--neo4j-uri bolt://localhost:7687] \\
        [--neo4j-user neo4j] \\
        [--neo4j-password password] \\
        [--kg-name hotpotqa]

The questions file is the per-config questions JSON produced by the experiment runner,
e.g. results/runs/.../questions/hotpotqa_..._questions.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    LiteralString,
    NotRequired,
    TypeAlias,
    TypedDict,
    cast,
)

from typing_extensions import ReadOnly

if TYPE_CHECKING:
    from neo4j import Driver, Result
    from sentence_transformers import SentenceTransformer

CorrectnessClass: TypeAlias = Literal[
    "both_correct",
    "vanilla_only",
    "kg_only",
    "neither",
]
RecallKey: TypeAlias = Literal[
    "sf_extraction_recall",
    "sf_vector_retrieval_recall",
    "grounding_quality",
    "seed_entity_count",
]


class SupportingFacts(TypedDict, total=False):
    title: list[str]
    sent_id: list[int]


class SupportingFactsQuestion(TypedDict, total=False):
    supporting_facts: SupportingFacts | list[list[str | int]] | None


class HotpotQuestion(SupportingFactsQuestion):
    _id: str
    context: NotRequired[list[list[str | list[str]]]]


class CorrectnessInput(TypedDict, total=False):
    vanilla_correct: ReadOnly[object]
    kg_correct: ReadOnly[object]


class EvaluationQuestion(TypedDict):
    question_id: str
    question: str
    vanilla_correct: NotRequired[bool]
    kg_correct: NotRequired[bool]
    grounding_quality: NotRequired[float | None]
    seed_entity_count: NotRequired[int | None]


class EvaluationQuestions(TypedDict, total=False):
    questions: list[EvaluationQuestion]


class ChunkCountProjection(TypedDict):
    chunk_count: int
    has_embedding: bool


class EntityCountProjection(TypedDict):
    entity_count: int


class ElementProjection(TypedDict):
    eid: str
    # Uninspected stored properties have no scalar schema guarantee.
    text: object


class VectorProjection(TypedDict):
    eid: str
    score: float


class EntityIDProjection(TypedDict):
    # This helper returns the raw stored property, not the driver element ID.
    eid: object


class PathCountProjection(TypedDict):
    n: int


class GraphPresence(TypedDict):
    chunk_count: int
    entity_count: int
    has_embedding: bool


class SupportingFactDetail(TypedDict):
    title: str
    passage_index: int
    in_graph: bool
    entity_count: int
    has_embedding: bool
    vector_retrievable_top40: bool
    vector_score: float | None


class QuestionAudit(TypedDict):
    question_id: str
    question: str
    vanilla_correct: bool | None
    kg_correct: bool | None
    correctness_class: CorrectnessClass
    sf_titles_all: list[str]
    sf_titles_in_context: list[str]
    sf_titles_missing_from_context: list[str]
    sf_passage_indices: list[int]
    sf_in_context_count: int
    sf_in_graph_count: int
    sf_vector_retrievable_count: int
    sf_extraction_recall: float | None
    sf_vector_retrieval_recall: float | None
    sf_details: list[SupportingFactDetail]
    grounding_quality: float | None
    seed_entity_count: int | None


class ClassAggregate(TypedDict):
    n: int
    avg_sf_extraction_recall: float | None
    avg_sf_vector_retrieval_recall: float | None
    avg_grounding_quality: float | None
    avg_seed_entity_count: float | None
    all_sf_vector_retrievable_pct: float | None


class OverallAggregate(TypedDict):
    avg_sf_extraction_recall: float | None
    avg_sf_vector_retrieval_recall: float | None
    sf_fully_in_graph_pct: float | None
    sf_fully_vector_retrievable_pct: float | None


class FailureAnalysis(TypedDict):
    total_vanilla_only: int
    extraction_failures: int
    retrieval_failures_given_in_graph: int
    extraction_failure_qids: list[str]
    retrieval_failure_qids: list[str]


class ClassReportSummary(TypedDict, total=False):
    n: ReadOnly[int]
    avg_sf_extraction_recall: ReadOnly[float | None]
    avg_sf_vector_retrieval_recall: ReadOnly[float | None]
    avg_grounding_quality: ReadOnly[float | None]
    avg_seed_entity_count: ReadOnly[float | None]
    all_sf_vector_retrievable_pct: ReadOnly[float | None]


class OverallReportSummary(TypedDict, total=False):
    avg_sf_extraction_recall: ReadOnly[float | None]
    avg_sf_vector_retrieval_recall: ReadOnly[float | None]
    sf_fully_in_graph_pct: ReadOnly[float | None]
    sf_fully_vector_retrievable_pct: ReadOnly[float | None]


class FailureReportSummary(TypedDict, total=False):
    total_vanilla_only: ReadOnly[int]
    extraction_failures: ReadOnly[int]
    retrieval_failures_given_in_graph: ReadOnly[int]
    extraction_failure_qids: ReadOnly[list[str]]
    retrieval_failure_qids: ReadOnly[list[str]]


class AuditAggregate(TypedDict):
    total_questions: int
    correctness_breakdown: dict[CorrectnessClass, int]
    both_correct: NotRequired[ClassAggregate]
    vanilla_only: NotRequired[ClassAggregate]
    kg_only: NotRequired[ClassAggregate]
    neither: NotRequired[ClassAggregate]
    overall: NotRequired[OverallAggregate]
    vanilla_only_failure_analysis: NotRequired[FailureAnalysis]


class AuditReport(TypedDict):
    aggregates: AuditAggregate
    questions: list[QuestionAudit]


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------


def _make_driver(uri: str, user: str, password: str) -> Driver:
    try:
        from neo4j import GraphDatabase

        return GraphDatabase.driver(uri, auth=(user, password))
    except ImportError:
        log.error("neo4j Python driver not installed; run: pip install neo4j")
        sys.exit(1)


def _run_query(
    driver: Driver,
    cypher: LiteralString,
    params: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    with driver.session() as session:
        result: Result = session.run(cypher, params or {})
        return [dict(r) for r in result]


# ---------------------------------------------------------------------------
# Load raw HotpotQA data
# ---------------------------------------------------------------------------


def _load_raw_hotpotqa(raw_path: str) -> dict[str, HotpotQuestion]:
    """Return {question_id: raw_question_dict}."""
    with open(raw_path) as f:
        raw: list[HotpotQuestion] = json.load(f)
    return {q["_id"]: q for q in raw}


def _get_supporting_facts(raw_q: SupportingFactsQuestion) -> list[tuple[str, int]]:
    """
    Return [(title, sent_id), ...] for each supporting fact.

    HotpotQA stores supporting_facts as either:
      - {"title": [...], "sent_id": [...]}  (object form, most common)
      - [[title, sent_id], ...]             (array form, some versions)
    """
    empty_supporting_facts: SupportingFacts = {}
    sf: SupportingFacts | list[list[str | int]] | None = raw_q.get(
        "supporting_facts",
        empty_supporting_facts,
    )
    if isinstance(sf, dict):
        titles: list[str] = sf.get("title", [])
        sent_ids: list[int] = sf.get("sent_id", [])
        return list(zip(titles, sent_ids))
    elif isinstance(sf, list):
        return [(item[0], item[1]) for item in sf if len(item) >= 2]
    return []


def _context_title_to_passage_index(raw_q: HotpotQuestion) -> dict[str, int]:
    """Map passage title -> passageIndex (0-based position in context list)."""
    return {title: idx for idx, (title, _) in enumerate(raw_q.get("context", []))}


# ---------------------------------------------------------------------------
# Graph checks
# ---------------------------------------------------------------------------


def _check_chunks_in_graph(
    driver: Driver,
    question_id: str,
    passage_indices: list[int],
    kg_name: str,
) -> dict[int, GraphPresence]:
    """
    For each passageIndex, return dict with:
      chunk_count, entity_count, has_embedding
    """
    results: dict[int, GraphPresence] = {}
    for pidx in passage_indices:
        # count and IS NOT NULL project an integer and boolean, respectively.
        rows: list[ChunkCountProjection] = cast(
            list[ChunkCountProjection],
            _run_query(
                driver,
                """
            MATCH (c:Chunk)
            WHERE c.questionId = $qid AND c.passageIndex = $pidx
              AND (c.kgName = $kg_name OR $kg_name IS NULL)
            RETURN
              count(c) AS chunk_count,
              c.embedding IS NOT NULL AS has_embedding
            LIMIT 1
            """,
                {"qid": question_id, "pidx": pidx, "kg_name": kg_name},
            ),
        )
        chunk_count: int = rows[0]["chunk_count"] if rows else 0
        has_embedding: bool = rows[0]["has_embedding"] if rows else False

        # DISTINCT does not change the integer result of Cypher count.
        entity_rows: list[EntityCountProjection] = cast(
            list[EntityCountProjection],
            _run_query(
                driver,
                """
            MATCH (c:Chunk)-[:HAS_ENTITY]->(e:__Entity__)
            WHERE c.questionId = $qid AND c.passageIndex = $pidx
              AND (c.kgName = $kg_name OR $kg_name IS NULL)
            RETURN count(DISTINCT e) AS entity_count
            """,
                {"qid": question_id, "pidx": pidx, "kg_name": kg_name},
            ),
        )
        entity_count: int = entity_rows[0]["entity_count"] if entity_rows else 0

        results[pidx] = {
            "chunk_count": chunk_count,
            "entity_count": entity_count,
            "has_embedding": bool(has_embedding),
        }
    return results


def _check_chunk_retrievable_via_vector(
    driver: Driver,
    question_id: str,
    passage_index: int,
    query_embedding: list[float],
    kg_name: str,
    top_k: int = 40,
) -> tuple[bool, float | None]:
    """
    Check whether the chunk at (questionId, passageIndex) ranks in the top-k
    of a vector similarity search for the question embedding.

    Returns (found_in_top_k, similarity_score_or_None).
    """
    # Get the chunk text for this passage
    # elementId is a string; the unused text property remains unconstrained.
    chunk_rows: list[ElementProjection] = cast(
        list[ElementProjection],
        _run_query(
            driver,
            """
        MATCH (c:Chunk)
        WHERE c.questionId = $qid AND c.passageIndex = $pidx
          AND (c.kgName = $kg_name OR $kg_name IS NULL)
        RETURN elementId(c) AS eid, c.text AS text
        LIMIT 1
        """,
            {"qid": question_id, "pidx": passage_index, "kg_name": kg_name},
        ),
    )
    if not chunk_rows:
        return False, None

    target_eid: str = chunk_rows[0]["eid"]

    # Run vector search and check if this chunk appears
    try:
        # The built-in vector procedure yields a numeric similarity score.
        vector_rows: list[VectorProjection] = cast(
            list[VectorProjection],
            _run_query(
                driver,
                """
            CALL db.index.vector.queryNodes('vector', $k, $embedding)
            YIELD node AS chunk, score
            WHERE chunk.questionId = $qid
            RETURN elementId(chunk) AS eid, score
            LIMIT $k
            """,
                {"k": top_k, "embedding": query_embedding, "qid": question_id},
            ),
        )
        for row in vector_rows:
            if row["eid"] == target_eid:
                return True, float(row["score"])
    except Exception as e:
        log.debug(
            "Vector retrievability check failed for %s p%d: %s",
            question_id,
            passage_index,
            e,
        )

    return False, None


def _get_entity_seeds_for_question(
    driver: Driver,
    question_id: str,
    kg_name: str,
    entity_names: list[str],
    limit: int = 12,
) -> list[object]:
    """
    Look up entity IDs that match any of the entity_names for this question's scope.
    Approximates what the entity-first retriever would use as seeds.
    """
    if not entity_names:
        return []
    # Only the alias is fixed: e.id may be absent or any stored property value.
    rows: list[EntityIDProjection] = cast(
        list[EntityIDProjection],
        _run_query(
            driver,
            """
        MATCH (e:__Entity__)
        WHERE toLower(e.name) IN $names
        AND EXISTS {
            MATCH (e)<-[:HAS_ENTITY]-(c:Chunk)
            WHERE c.questionId = $qid
        }
        RETURN e.id AS eid
        LIMIT $limit
        """,
            {
                "names": [n.lower() for n in entity_names],
                "qid": question_id,
                "limit": limit,
            },
        ),
    )
    return [r["eid"] for r in rows]


def _check_passage_reachable_via_entity_traversal(
    driver: Driver,
    question_id: str,
    passage_index: int,
    seed_entity_ids: list[str],
    kg_name: str,
    max_hops: int = 2,
) -> bool:
    """
    Check whether any entity in the SF passage is reachable from seed entities
    within max_hops in the knowledge graph.
    """
    if not seed_entity_ids:
        return False
    try:
        # Counting paths projects an integer independently of path contents.
        rows: list[PathCountProjection] = cast(
            list[PathCountProjection],
            _run_query(
                driver,
                f"""
            MATCH (seed:__Entity__)
            WHERE seed.id IN $seed_ids
            MATCH path = (seed)-[*1..{max_hops}]-(neighbor:__Entity__)
            WHERE EXISTS {{
                MATCH (neighbor)<-[:HAS_ENTITY]-(c:Chunk)
                WHERE c.questionId = $qid AND c.passageIndex = $pidx
            }}
            RETURN count(path) AS n
            LIMIT 1
            """,
                {
                    "seed_ids": seed_entity_ids,
                    "qid": question_id,
                    "pidx": passage_index,
                },
            ),
        )
        return bool(rows and rows[0]["n"] > 0)
    except Exception as e:
        log.debug(
            "Graph traversal check failed for %s p%d: %s",
            question_id,
            passage_index,
            e,
        )
        return False


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------


def _get_embedding(text: str, model: SentenceTransformer) -> list[float] | None:
    try:
        return model.encode(text).tolist()
    except Exception as e:
        log.debug("Embedding failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Main audit logic
# ---------------------------------------------------------------------------


def _audit_question(
    raw_q: HotpotQuestion,
    eval_q: EvaluationQuestion,
    driver: Driver,
    embed_model: SentenceTransformer | None,
    kg_name: str,
) -> QuestionAudit:
    question_id: str = eval_q["question_id"]
    question_text: str = eval_q["question"]

    sf_pairs: list[tuple[str, int]] = _get_supporting_facts(raw_q)
    title_to_idx: dict[str, int] = _context_title_to_passage_index(raw_q)

    # Deduplicate SF titles (a title can appear multiple times for different sentences)
    sf_titles_in_context: dict[str, int] = {}
    sf_titles_missing: list[str] = []
    for title, _sent_id in sf_pairs:
        if title in title_to_idx:
            sf_titles_in_context[title] = title_to_idx[title]
        elif title not in sf_titles_missing:
            sf_titles_missing.append(title)

    unique_sf_passage_indices: list[int] = list(set(sf_titles_in_context.values()))

    # Graph presence check
    graph_checks: dict[int, GraphPresence] = _check_chunks_in_graph(
        driver,
        question_id,
        unique_sf_passage_indices,
        kg_name,
    )

    # Embedding for vector retrievability check
    query_embedding: list[float] | None = (
        _get_embedding(question_text, embed_model) if embed_model else None
    )

    sf_details: list[SupportingFactDetail] = []
    for title, pidx in sf_titles_in_context.items():
        gc: GraphPresence = graph_checks.get(
            pidx,
            {"chunk_count": 0, "entity_count": 0, "has_embedding": False},
        )
        in_graph: bool = gc["chunk_count"] > 0

        vector_found: bool
        vector_score: float | None
        vector_found, vector_score = False, None
        if in_graph and query_embedding:
            vector_found, vector_score = _check_chunk_retrievable_via_vector(
                driver,
                question_id,
                pidx,
                query_embedding,
                kg_name,
            )

        sf_details.append(
            {
                "title": title,
                "passage_index": pidx,
                "in_graph": in_graph,
                "entity_count": gc["entity_count"],
                "has_embedding": gc["has_embedding"],
                "vector_retrievable_top40": vector_found,
                "vector_score": vector_score,
            },
        )

    # Summary booleans
    sf_in_context_count: int = len(unique_sf_passage_indices)
    sf_in_graph_count: int = sum(1 for d in sf_details if d["in_graph"])
    sf_vector_retrievable_count: int = sum(
        1 for d in sf_details if d["vector_retrievable_top40"]
    )

    return {
        "question_id": question_id,
        "question": question_text,
        "vanilla_correct": eval_q.get("vanilla_correct"),
        "kg_correct": eval_q.get("kg_correct"),
        "correctness_class": _correctness_class(eval_q),
        "sf_titles_all": [t for t, _ in sf_pairs],
        "sf_titles_in_context": list(sf_titles_in_context.keys()),
        "sf_titles_missing_from_context": sf_titles_missing,
        "sf_passage_indices": unique_sf_passage_indices,
        "sf_in_context_count": sf_in_context_count,
        "sf_in_graph_count": sf_in_graph_count,
        "sf_vector_retrievable_count": sf_vector_retrievable_count,
        "sf_extraction_recall": sf_in_graph_count / sf_in_context_count
        if sf_in_context_count
        else None,
        "sf_vector_retrieval_recall": sf_vector_retrievable_count / sf_in_context_count
        if sf_in_context_count
        else None,
        "sf_details": sf_details,
        "grounding_quality": eval_q.get("grounding_quality"),
        "seed_entity_count": eval_q.get("seed_entity_count"),
    }


def _correctness_class(eval_q: CorrectnessInput) -> CorrectnessClass:
    v: object = eval_q.get("vanilla_correct")
    k: object = eval_q.get("kg_correct")
    if v and k:
        return "both_correct"
    if v and not k:
        return "vanilla_only"
    if not v and k:
        return "kg_only"
    return "neither"


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------


def _aggregate(results: list[QuestionAudit]) -> AuditAggregate:
    by_class: dict[CorrectnessClass, list[QuestionAudit]] = defaultdict(list)
    for r in results:
        by_class[r["correctness_class"]].append(r)

    def _avg(lst: list[QuestionAudit], key: RecallKey) -> float | None:
        vals: list[float | int] = [
            cast(float | int, x[key]) for x in lst if x.get(key) is not None
        ]
        return sum(vals) / len(vals) if vals else None

    agg: AuditAggregate = {
        "total_questions": len(results),
        "correctness_breakdown": {cls: len(qs) for cls, qs in by_class.items()},
    }

    for cls, qs in by_class.items():
        agg[cls] = {
            "n": len(qs),
            "avg_sf_extraction_recall": _avg(qs, "sf_extraction_recall"),
            "avg_sf_vector_retrieval_recall": _avg(qs, "sf_vector_retrieval_recall"),
            "avg_grounding_quality": _avg(qs, "grounding_quality"),
            "avg_seed_entity_count": _avg(qs, "seed_entity_count"),
            # How often ALL SF passages were vector-retrievable
            "all_sf_vector_retrievable_pct": (
                sum(
                    1
                    for q in qs
                    if q["sf_vector_retrievable_count"] >= q["sf_in_context_count"] > 0
                )
                / len(qs)
                if qs
                else None
            ),
        }

    # Overall extraction recall (are SF passages in the graph at all?)
    agg["overall"] = {
        "avg_sf_extraction_recall": _avg(results, "sf_extraction_recall"),
        "avg_sf_vector_retrieval_recall": _avg(results, "sf_vector_retrieval_recall"),
        "sf_fully_in_graph_pct": (
            sum(1 for r in results if r["sf_extraction_recall"] == 1.0) / len(results)
            if results
            else None
        ),
        "sf_fully_vector_retrievable_pct": (
            sum(
                1
                for r in results
                if r["sf_vector_retrieval_recall"] is not None
                and r["sf_vector_retrieval_recall"] == 1.0
            )
            / len(results)
            if results
            else None
        ),
    }

    # Extraction vs retrieval breakdown for vanilla-only failures
    vanilla_only: list[QuestionAudit] = by_class.get("vanilla_only", [])
    extraction_failures: list[QuestionAudit] = [
        q
        for q in vanilla_only
        if q.get("sf_extraction_recall") is not None
        and cast(float, q["sf_extraction_recall"]) < 1.0
    ]
    retrieval_failures: list[QuestionAudit] = [
        q
        for q in vanilla_only
        if q.get("sf_extraction_recall") == 1.0
        and q.get("sf_vector_retrieval_recall") is not None
        and cast(float, q["sf_vector_retrieval_recall"]) < 1.0
    ]
    agg["vanilla_only_failure_analysis"] = {
        "total_vanilla_only": len(vanilla_only),
        "extraction_failures": len(extraction_failures),
        "retrieval_failures_given_in_graph": len(retrieval_failures),
        "extraction_failure_qids": [q["question_id"] for q in extraction_failures],
        "retrieval_failure_qids": [q["question_id"] for q in retrieval_failures],
    }

    return agg


def _print_report(agg: AuditAggregate, results: list[QuestionAudit]) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("RETRIEVAL AUDIT — Evidence-Level Diagnostic")
    lines.append("=" * 70)
    lines.append(f"\nTotal questions audited: {agg['total_questions']}")

    cb: dict[CorrectnessClass, int] = agg.get("correctness_breakdown", {})
    lines.append("\nCorrectness breakdown:")
    cls: CorrectnessClass
    for cls in ("both_correct", "vanilla_only", "kg_only", "neither"):
        lines.append(f"  {cls}: {cb.get(cls, 0)}")

    ov: OverallReportSummary = agg.get("overall", {})
    lines.append("\nOverall supporting-fact (SF) coverage:")
    lines.append(
        f"  SF passages in graph (extraction recall):  {ov.get('avg_sf_extraction_recall', 'N/A'):.1%}"
        if ov.get("avg_sf_extraction_recall") is not None
        else "  SF extraction recall: N/A",
    )
    lines.append(
        f"  SF fully in graph (all SF present):        {ov.get('sf_fully_in_graph_pct', 'N/A'):.1%}"
        if ov.get("sf_fully_in_graph_pct") is not None
        else "  SF fully in graph: N/A",
    )
    lines.append(
        f"  SF vector-retrievable (top-40 recall):     {ov.get('avg_sf_vector_retrieval_recall', 'N/A'):.1%}"
        if ov.get("avg_sf_vector_retrieval_recall") is not None
        else "  SF vector retrieval recall: N/A",
    )
    lines.append(
        f"  SF fully vector-retrievable:               {ov.get('sf_fully_vector_retrievable_pct', 'N/A'):.1%}"
        if ov.get("sf_fully_vector_retrievable_pct") is not None
        else "  SF fully vector-retrievable: N/A",
    )

    lines.append("\nBy correctness class:")
    for cls in ("both_correct", "vanilla_only", "kg_only", "neither"):
        class_summary: ClassReportSummary = agg.get(cls, {})
        if not class_summary:
            continue
        lines.append(
            f"\n  [{cls.upper()}] n={cast(ClassAggregate, class_summary)['n']}",
        )
        er: float | None = class_summary.get("avg_sf_extraction_recall")
        vr: float | None = class_summary.get("avg_sf_vector_retrieval_recall")
        af: float | None = class_summary.get("all_sf_vector_retrievable_pct")
        lines.append(
            f"    SF extraction recall:      {er:.1%}"
            if er is not None
            else "    SF extraction recall:      N/A",
        )
        lines.append(
            f"    SF vector retrieval recall:{vr:.1%}"
            if vr is not None
            else "    SF vector retrieval recall: N/A",
        )
        lines.append(
            f"    All SF vector-retrievable: {af:.1%}"
            if af is not None
            else "    All SF vector-retrievable:  N/A",
        )
        lines.append(
            f"    Avg grounding_quality:     {class_summary.get('avg_grounding_quality', 'N/A'):.3f}"
            if class_summary.get("avg_grounding_quality") is not None
            else "    Avg grounding_quality:     N/A",
        )
        lines.append(
            f"    Avg seed entity count:     {class_summary.get('avg_seed_entity_count', 'N/A'):.1f}"
            if class_summary.get("avg_seed_entity_count") is not None
            else "    Avg seed entity count:     N/A",
        )

    fa: FailureReportSummary = agg.get("vanilla_only_failure_analysis", {})
    lines.append("\nVanilla-only failure analysis:")
    lines.append(f"  Total vanilla-only questions: {fa.get('total_vanilla_only', 0)}")
    lines.append(
        f"  Extraction failures (SF not in graph): {fa.get('extraction_failures', 0)}",
    )
    lines.append(
        f"  Retrieval failures (in graph, not retrieved): {fa.get('retrieval_failures_given_in_graph', 0)}",
    )
    if fa.get("retrieval_failure_qids"):
        lines.append(
            f"  Retrieval-failure QIDs: {cast(FailureAnalysis, fa)['retrieval_failure_qids']}",
        )

    lines.append("\nVanilla-only question details:")
    vanilla_only_qs: list[QuestionAudit] = [
        r for r in results if r["correctness_class"] == "vanilla_only"
    ]
    for q in vanilla_only_qs:
        lines.append(f"\n  QID: {q['question_id']}")
        lines.append(f"  Q:   {q['question'][:90]}")
        lines.append(f"  SF in context: {q['sf_titles_in_context']}")
        lines.append(
            f"  SF missing from context: {q['sf_titles_missing_from_context']}",
        )
        for d in q["sf_details"]:
            lines.append(
                f"    p{d['passage_index']} '{d['title'][:40]}': "
                f"in_graph={d['in_graph']}, "
                f"entities={d['entity_count']}, "
                f"emb={d['has_embedding']}, "
                f"vector_top40={d['vector_retrievable_top40']} "
                f"(score={d['vector_score']:.4f})"
                if d["vector_score"] is not None
                else f"    p{d['passage_index']} '{d['title'][:40]}': "
                f"in_graph={d['in_graph']}, entities={d['entity_count']}, "
                f"emb={d['has_embedding']}, vector_top40={d['vector_retrievable_top40']}",
            )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Evidence-level retrieval audit for KG vs vanilla recall gap",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to evaluation run directory",
    )
    parser.add_argument(
        "--questions-file",
        required=True,
        help="Path to per-config questions JSON",
    )
    parser.add_argument(
        "--raw-data",
        default="MIRAGE/rawdata/hotpotqa/hotpot_dev_fullwiki_v1.json",
        help="Path to raw HotpotQA dev data",
    )
    parser.add_argument(
        "--neo4j-uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    )
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USERNAME", "neo4j"))
    parser.add_argument(
        "--neo4j-password",
        default=os.getenv("NEO4J_PASSWORD", "password"),
    )
    parser.add_argument("--kg-name", default="hotpotqa")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name for vector retrievability checks",
    )
    parser.add_argument(
        "--skip-vector-check",
        action="store_true",
        help="Skip vector retrievability check (faster, no embedding model needed)",
    )
    parser.add_argument(
        "--top-k-vector",
        type=int,
        default=40,
        help="Top-K for vector retrievability check",
    )
    args: argparse.Namespace = parser.parse_args()

    run_dir: Path = Path(args.run_dir)
    out_json: Path = run_dir / f"retrieval_audit_{run_dir.name}.json"
    out_txt: Path = run_dir / f"retrieval_audit_{run_dir.name}.txt"

    # Load raw data
    raw_path: str = args.raw_data
    if not os.path.isabs(raw_path):
        raw_path = str(Path(__file__).parent.parent / raw_path)
    log.info("Loading raw HotpotQA from %s", raw_path)
    raw_by_id: dict[str, HotpotQuestion] = _load_raw_hotpotqa(raw_path)

    # Load eval questions
    log.info("Loading eval questions from %s", args.questions_file)
    with open(args.questions_file) as f:
        qs_data: EvaluationQuestions = json.load(f)
    eval_questions: list[EvaluationQuestion] = qs_data.get("questions", [])
    log.info("%d questions in eval file", len(eval_questions))

    # Connect to Neo4j
    log.info("Connecting to Neo4j at %s", args.neo4j_uri)
    driver: Driver = _make_driver(args.neo4j_uri, args.neo4j_user, args.neo4j_password)

    # Load embedding model
    embed_model: SentenceTransformer | None = None
    if not args.skip_vector_check:
        try:
            from sentence_transformers import SentenceTransformer

            log.info("Loading embedding model %s", args.embedding_model)
            embed_model = SentenceTransformer(args.embedding_model)
        except ImportError:
            log.warning(
                "sentence-transformers not installed; skipping vector retrievability check",
            )

    # Audit each question
    results: list[QuestionAudit] = []
    for i, eval_q in enumerate(eval_questions):
        qid: str = eval_q["question_id"]
        raw_q: HotpotQuestion | None = raw_by_id.get(qid)
        if not raw_q:
            log.warning("QID %s not found in raw data, skipping", qid)
            continue
        log.info("[%d/%d] Auditing %s", i + 1, len(eval_questions), qid)
        result: QuestionAudit = _audit_question(
            raw_q,
            eval_q,
            driver,
            embed_model,
            args.kg_name,
        )
        results.append(result)

    driver.close()

    # Aggregate and report
    agg: AuditAggregate = _aggregate(results)
    report_text: str = _print_report(agg, results)
    print(report_text)

    # Write outputs
    output: AuditReport = {"aggregates": agg, "questions": results}
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    with open(out_txt, "w") as f:
        f.write(report_text)

    log.info("Written to %s and %s", out_json, out_txt)


if __name__ == "__main__":
    main()
