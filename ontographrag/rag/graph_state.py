"""Utilities for logging symbolic retrieval-state diversity.

The functions here are deliberately dependency-light so they can be used both
inside the live RAG system and by offline experiment scripts.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from typing import Literal

from typing_extensions import ReadOnly, TypedDict


class StateRelationship(TypedDict, total=False):
    key: ReadOnly[object]
    element_id: ReadOnly[object]
    source: ReadOnly[object]
    type: ReadOnly[object]
    target: ReadOnly[object]
    negated: ReadOnly[object]


class StatePath(TypedDict, total=False):
    node_ids: ReadOnly[Iterable[object] | None]
    path: ReadOnly[object]


class StateEntity(TypedDict, total=False):
    id: ReadOnly[object]
    min_hops: ReadOnly[int | float | str | None]


class StateChunk(TypedDict, total=False):
    chunk_id: ReadOnly[object]
    chunk_element_id: ReadOnly[object]
    text: ReadOnly[object]
    linked_entity_ids: ReadOnly[Iterable[object] | None]
    entities: ReadOnly[Sequence[StateEntity | None] | None]


class StateDiagnostics(TypedDict, total=False):
    seed_entity_ids: ReadOnly[Iterable[object] | None]
    seed_entity_names: ReadOnly[Iterable[object] | None]
    seed_entities: ReadOnly[Iterable[object] | None]


class GraphStateContext(TypedDict, total=False):
    diagnostics: ReadOnly[StateDiagnostics | None]
    chunks: ReadOnly[Sequence[StateChunk | None] | None]
    entities: ReadOnly[Mapping[str, StateEntity | None] | None]
    relationships: ReadOnly[Sequence[StateRelationship | None] | None]
    traversal_paths: ReadOnly[Sequence[StatePath | None] | None]
    retrieval_route: ReadOnly[object]
    search_method: ReadOnly[object]


class GraphStateSample(TypedDict, total=False):
    seed_entity_ids: ReadOnly[Sequence[str] | None]
    entity_ids: ReadOnly[Sequence[str] | None]
    relationship_ids: ReadOnly[Sequence[str] | None]
    path_ids: ReadOnly[Sequence[str] | None]
    chunk_ids: ReadOnly[Sequence[str] | None]
    seed_signature: ReadOnly[str | None]
    path_signature: ReadOnly[str | None]
    subgraph_signature: ReadOnly[str | None]
    chunk_signature: ReadOnly[str | None]
    route: ReadOnly[str | None]


class GraphStateSummary(TypedDict):
    route: str
    search_method: str
    seed_entity_ids: list[str]
    seed_entity_names: list[str]
    entity_ids: list[str]
    relationship_ids: list[str]
    path_ids: list[str]
    chunk_ids: list[str]
    seed_signature: str
    path_signature: str
    subgraph_signature: str
    chunk_signature: str
    entity_count: int
    relationship_count: int
    path_count: int
    chunk_count: int


class EmptyGraphStateDiversity(TypedDict):
    sample_count: int
    seed_entity_entropy: float
    path_entropy: float
    subgraph_entropy: float
    chunk_entropy: float
    route_entropy: float


class GraphStateDiversity(EmptyGraphStateDiversity):
    empty_state_count: int
    seed_entity_entropy_norm: float
    path_entropy_norm: float
    subgraph_entropy_norm: float
    chunk_entropy_norm: float
    route_entropy_norm: float
    seed_entity_jaccard: float
    entity_jaccard: float
    path_jaccard: float
    subgraph_jaccard: float
    chunk_jaccard: float
    dominant_seed_entity_id: str
    dominant_seed_entity_fraction: float


def _clean_id(value: object) -> str:
    text: str = str(value or "").strip()
    return text


def _set_from(values: Iterable[object]) -> set[str]:
    return {text for text in (_clean_id(v) for v in values) if text}


def _signature(values: Iterable[object]) -> str:
    items: list[str] = sorted(_set_from(values))
    return "||".join(items) if items else "<empty>"


def _relationship_id(rel: StateRelationship | None) -> str:
    if not isinstance(rel, dict):
        return ""
    return (
        _clean_id(rel.get("key"))
        or _clean_id(rel.get("element_id"))
        or "::".join(
            part
            for part in (
                _clean_id(rel.get("source")),
                _clean_id(rel.get("type")),
                _clean_id(rel.get("target")),
                f"neg={bool(rel.get('negated', False))}",
            )
            if part
        )
    )


def _path_id(path_entry: StatePath | None) -> str:
    if not isinstance(path_entry, dict):
        return ""
    node_ids: Iterable[object] = path_entry.get("node_ids") or []
    if node_ids:
        return "->".join(
            _clean_id(node_id) for node_id in node_ids if _clean_id(node_id)
        )
    return _clean_id(path_entry.get("path"))


def _chunk_id(chunk: StateChunk | None) -> str:
    if not isinstance(chunk, dict):
        return ""
    return (
        _clean_id(chunk.get("chunk_id"))
        or _clean_id(chunk.get("chunk_element_id"))
        or _clean_id(chunk.get("text"))[:80]
    )


def _entity_ids_from_chunks(chunks: Sequence[StateChunk | None]) -> list[str]:
    ids: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        ids.extend(_clean_id(eid) for eid in (chunk.get("linked_entity_ids") or []))
        for entity in chunk.get("entities") or []:
            if isinstance(entity, dict):
                ids.append(_clean_id(entity.get("id")))
    return [eid for eid in ids if eid]


def summarize_context_graph_state(
    context: GraphStateContext | None,
) -> GraphStateSummary:
    """Return a compact symbolic state summary for one retrieved context."""
    context = context or {}
    diagnostics: StateDiagnostics = context.get("diagnostics") or {}
    chunks: Sequence[StateChunk | None] = context.get("chunks") or []
    entities: Mapping[str, StateEntity | None] = context.get("entities") or {}
    relationships: Sequence[StateRelationship | None] = (
        context.get("relationships") or []
    )
    traversal_paths: Sequence[StatePath | None] = context.get("traversal_paths") or []

    seed_entity_ids: list[str] = [
        _clean_id(eid)
        for eid in (diagnostics.get("seed_entity_ids") or [])
        if _clean_id(eid)
    ]
    if not seed_entity_ids:
        seed_entity_ids = [
            _clean_id(eid)
            for eid, info in entities.items()
            if isinstance(info, dict) and int(info.get("min_hops", 99) or 99) == 0
        ]

    seed_entity_names: list[str] = [
        _clean_id(name)
        for name in (
            diagnostics.get("seed_entity_names")
            or diagnostics.get("seed_entities")
            or []
        )
        if _clean_id(name)
    ]
    entity_ids: list[str] = list(entities.keys()) or _entity_ids_from_chunks(chunks)
    relationship_ids: list[str] = [_relationship_id(rel) for rel in relationships]
    path_ids: list[str] = [_path_id(path) for path in traversal_paths]
    chunk_ids: list[str] = [_chunk_id(chunk) for chunk in chunks]

    subgraph_items: list[str] = list(entity_ids) + relationship_ids + path_ids
    return {
        "route": _clean_id(context.get("retrieval_route")),
        "search_method": _clean_id(context.get("search_method")),
        "seed_entity_ids": sorted(_set_from(seed_entity_ids)),
        "seed_entity_names": sorted(_set_from(seed_entity_names)),
        "entity_ids": sorted(_set_from(entity_ids)),
        "relationship_ids": sorted(_set_from(relationship_ids)),
        "path_ids": sorted(_set_from(path_ids)),
        "chunk_ids": sorted(_set_from(chunk_ids)),
        "seed_signature": _signature(seed_entity_ids),
        "path_signature": _signature(path_ids),
        "subgraph_signature": _signature(subgraph_items),
        "chunk_signature": _signature(chunk_ids),
        "entity_count": len(_set_from(entity_ids)),
        "relationship_count": len(_set_from(relationship_ids)),
        "path_count": len(_set_from(path_ids)),
        "chunk_count": len(_set_from(chunk_ids)),
    }


def shannon_entropy(labels: Sequence[str]) -> float:
    labels = [str(label) for label in labels]
    if not labels:
        return 0.0
    n: int = len(labels)
    counts: Counter[str] = Counter(labels)
    value: float = float(-sum((c / n) * math.log(c / n, 2) for c in counts.values()))
    return 0.0 if abs(value) < 1e-12 else value


def normalized_entropy(labels: Sequence[str]) -> float:
    labels = [str(label) for label in labels]
    if len(labels) <= 1:
        return 0.0
    denom: float = math.log(len(labels), 2)
    return float(shannon_entropy(labels) / denom) if denom > 0 else 0.0


def mean_pairwise_jaccard(sets: Sequence[Iterable[object]]) -> float:
    clean_sets: list[set[str]] = [_set_from(values) for values in sets]
    pairs: list[tuple[set[str], set[str]]] = list(combinations(clean_sets, 2))
    if not pairs:
        return 1.0 if clean_sets else 0.0
    vals: list[float] = []
    for left, right in pairs:
        union: set[str] = left | right
        vals.append((len(left & right) / len(union)) if union else 1.0)
    return float(sum(vals) / len(vals)) if vals else 0.0


def graph_state_diversity(
    sample_states: Sequence[GraphStateSample | GraphStateSummary | None],
) -> EmptyGraphStateDiversity | GraphStateDiversity:
    """Compute entropy/Jaccard diversity across per-sample graph states."""
    states: list[GraphStateSample | GraphStateSummary] = [
        state or {} for state in sample_states
    ]
    if not states:
        return {
            "sample_count": 0,
            "seed_entity_entropy": 0.0,
            "path_entropy": 0.0,
            "subgraph_entropy": 0.0,
            "chunk_entropy": 0.0,
            "route_entropy": 0.0,
        }

    def labels(
        key: Literal[
            "seed_signature",
            "path_signature",
            "subgraph_signature",
            "chunk_signature",
            "route",
        ],
    ) -> list[str]:
        return [str(state.get(key) or "<empty>") for state in states]

    seed_sets: list[Sequence[str]] = [
        state.get("seed_entity_ids") or [] for state in states
    ]
    entity_sets: list[Sequence[str]] = [
        state.get("entity_ids") or [] for state in states
    ]
    path_sets: list[Sequence[str]] = [state.get("path_ids") or [] for state in states]
    subgraph_sets: list[list[str]] = [
        list(state.get("entity_ids") or [])
        + list(state.get("relationship_ids") or [])
        + list(state.get("path_ids") or [])
        for state in states
    ]
    chunk_sets: list[Sequence[str]] = [state.get("chunk_ids") or [] for state in states]
    seed_counter: Counter[str] = Counter(
        eid for state in states for eid in (state.get("seed_entity_ids") or [])
    )
    dominant_seed_id, dominant_seed_count = ("", 0)
    if seed_counter:
        dominant_seed_id, dominant_seed_count = seed_counter.most_common(1)[0]

    return {
        "sample_count": len(states),
        "empty_state_count": sum(1 for state in states if not state.get("chunk_ids")),
        "seed_entity_entropy": shannon_entropy(labels("seed_signature")),
        "seed_entity_entropy_norm": normalized_entropy(labels("seed_signature")),
        "path_entropy": shannon_entropy(labels("path_signature")),
        "path_entropy_norm": normalized_entropy(labels("path_signature")),
        "subgraph_entropy": shannon_entropy(labels("subgraph_signature")),
        "subgraph_entropy_norm": normalized_entropy(labels("subgraph_signature")),
        "chunk_entropy": shannon_entropy(labels("chunk_signature")),
        "chunk_entropy_norm": normalized_entropy(labels("chunk_signature")),
        "route_entropy": shannon_entropy(labels("route")),
        "route_entropy_norm": normalized_entropy(labels("route")),
        "seed_entity_jaccard": mean_pairwise_jaccard(seed_sets),
        "entity_jaccard": mean_pairwise_jaccard(entity_sets),
        "path_jaccard": mean_pairwise_jaccard(path_sets),
        "subgraph_jaccard": mean_pairwise_jaccard(subgraph_sets),
        "chunk_jaccard": mean_pairwise_jaccard(chunk_sets),
        "dominant_seed_entity_id": dominant_seed_id,
        "dominant_seed_entity_fraction": (
            float(dominant_seed_count / len(states)) if states else 0.0
        ),
    }


def dominant_anchor_id(
    sample_states: Sequence[GraphStateSample | GraphStateSummary | None],
) -> str:
    """Return the most frequent seed entity id across sample states."""
    diversity: EmptyGraphStateDiversity | GraphStateDiversity = graph_state_diversity(
        sample_states
    )
    return str(diversity.get("dominant_seed_entity_id") or "")
