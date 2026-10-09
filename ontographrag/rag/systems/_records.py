"""Dictionary records produced and combined by the retrieval systems."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, NotRequired, Protocol, TypeAlias, TypeVar

from typing_extensions import TypedDict

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ontographrag.rag.reranking import RerankingMetadata

from ontographrag.rag.graph_state import GraphStateSummary

DecodedJSON: TypeAlias = (
    str | int | float | bool | None | list["DecodedJSON"] | dict[str, "DecodedJSON"]
)


# The maintained storage sanitizer admits these values for extensible properties.
# This is not a universal Neo4j driver-value model: temporal/node/relationship
# objects remain outside the per-query property projections below.
GraphPrimitive: TypeAlias = str | int | float | bool
StoredGraphValue: TypeAlias = GraphPrimitive | list[GraphPrimitive]
StoredGraphProperty: TypeAlias = StoredGraphValue | None


class EntityExtensions(TypedDict, total=False):
    name: StoredGraphProperty
    source: str
    min_hops: int
    seed_connections: int
    score: float


class PartialRetrievedEntity(EntityExtensions, total=False):
    id: StoredGraphValue
    element_id: str | None
    type: str | None
    description: StoredGraphProperty
    mentioned_in_chunks: list[str | None]


class EntityProjectionFields(EntityExtensions):
    id: StoredGraphValue
    element_id: str | None
    type: str | None
    description: StoredGraphProperty


class EntityProjection(EntityProjectionFields, total=False):
    mentioned_in_chunks: list[str | None]


class MentionedEntity(EntityProjectionFields):
    mentioned_in_chunks: list[str | None]


class HoppedEntity(TypedDict):
    id: StoredGraphValue
    element_id: str | None
    type: str | None
    description: StoredGraphProperty
    mentioned_in_chunks: list[str | None]
    source: str
    min_hops: int
    name: NotRequired[StoredGraphProperty]
    seed_connections: NotRequired[int]
    score: NotRequired[float]


class GraphNeighbor(TypedDict):
    id: StoredGraphValue
    element_id: str
    type: str
    name: StoredGraphProperty
    min_hops: int
    seed_connections: int
    source: str
    description: NotRequired[StoredGraphProperty]
    mentioned_in_chunks: NotRequired[list[str | None]]
    score: NotRequired[float]


RetrievedEntity: TypeAlias = (
    PartialRetrievedEntity
    | EntityProjection
    | MentionedEntity
    | HoppedEntity
    | GraphNeighbor
)


class RetrievedChunkExtensions(TypedDict, total=False):
    parent_chunk_id: str | None
    parent_chunk_element_id: str | None
    question_id: str | None
    passage_index: int | None
    chunk_local_index: int | None
    retrieval_local_index: int | None
    position: int | None
    min_hop: int
    source: str | None
    kg_name: StoredGraphProperty
    entities: list[RetrievedEntity] | list[EntityProjection]
    linked_entity_ids: Sequence[StoredGraphProperty]
    linked_entity_count: int
    adjacent: bool
    reranker_score: float
    late_interaction_score: float | None
    rrf_score: float
    retrieval_support_count: int


class PartialRetrievedChunk(RetrievedChunkExtensions, total=False):
    text: str
    document: StoredGraphValue
    chunk_id: str | None
    chunk_element_id: str | None
    score: float


class CompleteRetrievedChunk(RetrievedChunkExtensions):
    text: str
    document: StoredGraphValue
    chunk_id: str | None
    chunk_element_id: str | None
    score: NotRequired[float]


class ScoredRetrievedChunk(RetrievedChunkExtensions):
    text: str
    document: StoredGraphValue
    chunk_id: str | None
    chunk_element_id: str | None
    score: float


RetrievedChunk: TypeAlias = (
    PartialRetrievedChunk | CompleteRetrievedChunk | ScoredRetrievedChunk
)
ChunkT = TypeVar("ChunkT", bound=RetrievedChunk)


class RetrievalProvenance(TypedDict, total=False):
    question_ids: list[str] | list[GraphPrimitive]
    passage_keys: list[str] | list[GraphPrimitive]
    provenance_positions: list[int] | list[GraphPrimitive]


class RetrievedRelationship(RetrievalProvenance, total=False):
    key: str
    source: str
    source_element_id: str | None
    target: str
    target_element_id: str | None
    type: str
    element_id: str | None
    negated: bool
    condition: StoredGraphProperty
    quantitative: StoredGraphProperty
    confidence: float | None


class TraversalPath(RetrievalProvenance, total=False):
    path: str
    node_ids: list[str]
    hops: int
    seed: str
    target: str
    score: float


class RetrievalDiagnostics(TypedDict, total=False):
    seed_entity_ids: list[str]
    seed_entity_names: list[str] | list[str | None] | list[StoredGraphProperty]
    seed_entities: list[str] | list[str | None] | list[StoredGraphProperty]
    anchor_mask_entity_ids: list[str]
    anchor_mask_entity_names: list[str]
    ann_match_count: int
    symbolic_match_count: int
    rfge_fired: bool
    subgraph_edge_count: int
    top_ppr_entities: list[tuple[str, float]]
    retrieval_mode_config: str
    subquestion_count: int
    active_hop_count: int


class RetrievalSampling(TypedDict):
    temperature: float
    shortlist_factor: int
    sample_id: int
    candidate_limit: int


class RetrievalContextExtensions(TypedDict, total=False):
    query: str
    search_method: str
    retrieval_route: str
    route_reason: str
    late_interaction_stage: "RerankingMetadata"
    late_interaction: "RerankingMetadata"
    reranker: "RerankingMetadata"
    kg_name: str | None
    seed_entity_count: int
    grounding_quality: float
    diagnostics: RetrievalDiagnostics
    retrieval_sampling: RetrievalSampling
    graph_state: GraphStateSummary
    error: str


class RetrievalCore(TypedDict):
    chunks: list[RetrievedChunk]
    documents: list[StoredGraphProperty]
    total_score: float


class PartialRetrievalContext(RetrievalContextExtensions, total=False):
    chunks: list[RetrievedChunk]
    documents: list[StoredGraphProperty]
    total_score: float
    entities: dict[str, RetrievedEntity]
    relationships: list[RetrievedRelationship]
    entity_count: int
    relationship_count: int

    graph_neighbors: dict[str, RetrievedEntity]
    traversal_paths: list[TraversalPath]


class CompleteRetrievalContext(RetrievalContextExtensions, RetrievalCore, total=False):
    entities: dict[str, RetrievedEntity]
    relationships: list[RetrievedRelationship]
    entity_count: int
    relationship_count: int

    graph_neighbors: dict[str, RetrievedEntity]
    traversal_paths: list[TraversalPath]


class ScoredRetrievalCore(TypedDict):
    chunks: list[ScoredRetrievedChunk]
    documents: list[StoredGraphProperty]
    total_score: float


class GraphRetrievalContext(RetrievalContextExtensions, ScoredRetrievalCore):
    entities: dict[str, RetrievedEntity]
    relationships: list[RetrievedRelationship]
    entity_count: NotRequired[int]
    relationship_count: NotRequired[int]

    graph_neighbors: dict[str, RetrievedEntity]
    traversal_paths: list[TraversalPath]


class CountedGraphRetrievalContext(RetrievalContextExtensions, RetrievalCore):
    entities: dict[str, RetrievedEntity]
    relationships: list[RetrievedRelationship]
    entity_count: int
    relationship_count: int

    graph_neighbors: dict[str, RetrievedEntity]
    traversal_paths: list[TraversalPath]


class ScoredCompleteRetrievalContext(
    RetrievalContextExtensions,
    ScoredRetrievalCore,
    total=False,
):
    entities: dict[str, RetrievedEntity]
    relationships: list[RetrievedRelationship]
    entity_count: int
    relationship_count: int

    graph_neighbors: dict[str, RetrievedEntity]
    traversal_paths: list[TraversalPath]


class ProducedGraphRetrievalContext(RetrievalContextExtensions, ScoredRetrievalCore):
    entities: dict[str, RetrievedEntity]
    relationships: list[RetrievedRelationship]
    entity_count: int
    relationship_count: int

    graph_neighbors: dict[str, RetrievedEntity]
    traversal_paths: list[TraversalPath]


RetrievalContext: TypeAlias = (
    PartialRetrievalContext
    | CompleteRetrievalContext
    | GraphRetrievalContext
    | CountedGraphRetrievalContext
    | ScoredCompleteRetrievalContext
    | ProducedGraphRetrievalContext
)
ContextT = TypeVar("ContextT", bound=RetrievalContext)
StrongContextT = TypeVar(
    "StrongContextT",
    bound=GraphRetrievalContext
    | ProducedGraphRetrievalContext
    | ScoredCompleteRetrievalContext,
)


class RetrievalWorkingContext(RetrievalContextExtensions):
    """Vanilla assembly phase before document-set materialization."""

    chunks: list[RetrievedChunk]
    documents: set[StoredGraphProperty] | list[StoredGraphProperty]
    total_score: float


class GraphWorkingContext(RetrievalContextExtensions):
    """Graph assembly phase before document-set materialization."""

    chunks: list[RetrievedChunk]
    documents: set[StoredGraphProperty] | list[StoredGraphProperty]
    total_score: float
    entities: dict[str, MentionedEntity]
    relationships: list[RetrievedRelationship]
    entity_count: NotRequired[int]
    relationship_count: NotRequired[int]

    graph_neighbors: NotRequired[dict[str, RetrievedEntity]]
    traversal_paths: NotRequired[list[TraversalPath]]


class FusedChunkRecord(TypedDict):
    chunk: RetrievedChunk
    rrf: float
    support_count: int
    best_score: float


class EvidencePath(TypedDict):
    path_str: str
    node_ids: set[str]
    hops: int
    provenance_positions: list[int] | list[GraphPrimitive]
    passage_keys: list[str] | list[GraphPrimitive]


class GraphExpansion(TypedDict):
    neighbors: dict[str, GraphNeighbor]
    paths: list[TraversalPath]


class RelationshipRow(RetrievalProvenance):
    source: str
    source_element_id: str
    target: str
    target_element_id: str
    relationship_type: str
    relationship_element_id: str
    negated: bool | None
    condition: StoredGraphProperty
    quantitative: StoredGraphProperty
    confidence: float | None


class HopBucket(TypedDict):
    all_chunk_ids: set[str]
    new_chunks: list[ScoredRetrievedChunk]
    entities: dict[str, RetrievedEntity]
    relationships: list[RetrievedRelationship]
    graph_neighbors: dict[str, RetrievedEntity]
    traversal_paths: list[TraversalPath]


class ChunkProjectionFields(TypedDict):
    """Common columns projected from Chunk/Document matches; route extras vary."""

    text: str
    chunk_id: str | None
    chunk_element_id: str
    document: StoredGraphValue
    parent_chunk_id: NotRequired[str | None]
    parent_chunk_element_id: NotRequired[str | None]
    question_id: NotRequired[str | None]
    passage_index: NotRequired[int | None]
    chunk_local_index: NotRequired[int | None]
    retrieval_local_index: NotRequired[int | None]
    position: NotRequired[int | None]
    source: NotRequired[str | None]
    kg_name: NotRequired[StoredGraphProperty]
    entities: NotRequired[list[EntityProjection]]
    linked_entity_ids: NotRequired[list[str]]
    linked_entity_count: NotRequired[int]
    seed_entity_id: NotRequired[str]


class ChunkProjection(ChunkProjectionFields, total=False):
    score: float


class ScoredChunkProjection(ChunkProjectionFields):
    score: float


class RelationTypeProjection(TypedDict):
    rel_type: str
    cnt: int


class TraversalSampleProjection(TypedDict):
    nodes: list[StoredGraphProperty]
    node_ids: list[StoredGraphProperty]
    rels: list[RetrievedRelationship]


class TraversalProjection(TypedDict):
    id: StoredGraphValue
    name: StoredGraphProperty
    type: str
    element_id: str
    min_hops: int
    seed_connections: int
    sample_paths: list[TraversalSampleProjection]


class EdgeProjection(TypedDict):
    src: str
    tgt: str
    conf: float


class EmbeddingProjection(TypedDict):
    eid: str
    emb: list[float] | None


class QueryEncoder(Protocol):
    """Legacy single-query encode capability accepted by the RAG fallback."""

    def encode(
        self,
        query: str,
        /,
        *,
        convert_to_numpy: Literal[True],
    ) -> "NDArray[np.float32] | list[float]": ...


VectorIndexConfiguration = TypedDict(
    "VectorIndexConfiguration",
    {
        "vector.dimensions": NotRequired[int],
        "vector.similarity_function": NotRequired[str],
    },
)


class VectorIndexOptions(TypedDict, total=False):
    indexProvider: str
    indexConfig: VectorIndexConfiguration


class VectorIndexOptionsProjection(TypedDict):
    options: VectorIndexOptions


class FloatListConvertible(Protocol):
    def tolist(self) -> list[float]: ...
