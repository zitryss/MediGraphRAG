"""Static conformance witnesses checked by both whole-project typing gates.

These functions are not runtime tests and never construct SDK clients. Returning
SDK values as the consumed protocols checks the real interface without casts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ontographrag.rag.graph_state import GraphStateContext, StateChunk, StateEntity
from ontographrag.rag.retrieval_sampling import select_ranked_subset
from ontographrag.rag.systems._records import RetrievalContext

if TYPE_CHECKING:
    from langchain_community.graphs.graph_document import GraphDocument
    from langchain_neo4j import Neo4jGraph

    from ontographrag.kg.builders.ontology_guided_kg_creator import (
        OntologyGuidedKGCreator,
    )
    from ontographrag.kg.chunking import ChunkingHost
    from shared.common_fn import GraphQuery, MergeDocument


def graph_merge_contracts(
    graph: Neo4jGraph,
    document: GraphDocument,
) -> tuple[GraphQuery, MergeDocument]:
    return graph, document


def chunking_host_contract(host: OntologyGuidedKGCreator) -> ChunkingHost:
    return host


def sampled_score_rows(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    return select_ranked_subset(
        rows,
        max_items=1,
        retrieval_temperature=1.0,
        shortlist_factor=2,
    )


def sampled_numbers(rows: list[int]) -> list[int]:
    return select_ranked_subset(
        rows,
        max_items=1,
        retrieval_temperature=1.0,
        shortlist_factor=2,
        score_getter=float,
    )


def sampled_deterministic_numbers(rows: list[int]) -> list[int]:
    return select_ranked_subset(
        rows,
        max_items=1,
        retrieval_temperature=0,
        shortlist_factor=2,
    )


class ConcreteStateEntity(StateEntity):
    id: str
    min_hops: int


class ConcreteStateChunk(StateChunk):
    chunk_id: str
    text: str
    entities: list[ConcreteStateEntity]


class ConcreteStateContext(GraphStateContext):
    chunks: list[ConcreteStateChunk]
    entities: dict[str, ConcreteStateEntity]


def graph_state_projection(context: ConcreteStateContext) -> GraphStateContext:
    return context


def retrieval_graph_projection(context: RetrievalContext) -> GraphStateContext:
    return context
