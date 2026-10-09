"""Dictionary records produced and combined by the retrieval systems."""

from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Literal,
    NotRequired,
    Protocol,
    Required,
    TypeAlias,
    TypeVar,
)

from typing_extensions import TypedDict

if TYPE_CHECKING:
    import numpy as np
    from langchain_core.prompts import ChatPromptTemplate
    from numpy.typing import NDArray

    from ontographrag.rag.reranking import RerankingMetadata

from ontographrag.rag.graph_state import GraphStateSummary
from shared.graph_types import (
    Neo4jProperty,
    Neo4jPropertyValue,
)
from shared.graph_types import (
    Neo4jPropertyScalar as ImportedPropertyScalar,
)
from shared.graph_types import (
    Neo4jPropertyValue as ImportedPropertyValue,
)

DecodedJSON: TypeAlias = (
    str | int | float | bool | None | list["DecodedJSON"] | dict[str, "DecodedJSON"]
)


# Imported graphs bypass the normalized storage writer. These aliases cover
# hydrated property values, not arbitrary Cypher row maps or driver entities.
GraphPrimitive: TypeAlias = str | int | float | bool
StoredGraphValue: TypeAlias = Neo4jPropertyValue
StoredGraphProperty: TypeAlias = Neo4jProperty
# Successful dict/set insertion excludes list properties; missing IDs may be None.
StoredGraphKey: TypeAlias = "ImportedPropertyScalar | None"
SlicedGraphValue: TypeAlias = (
    str | bytes | tuple[int, ...] | tuple[float, ...] | list[ImportedPropertyScalar]
)
ChunkIdentity: TypeAlias = "StoredGraphKey | tuple[int, ...] | tuple[float, ...]"


class EntityExtensions(TypedDict, total=False):
    name: StoredGraphProperty
    source: str
    min_hops: int
    seed_connections: int
    score: float


class PartialRetrievedEntity(EntityExtensions, total=False):
    id: StoredGraphProperty
    element_id: str | None
    type: StoredGraphProperty
    description: StoredGraphProperty
    mentioned_in_chunks: list[StoredGraphProperty]


class EntityProjectionFields(EntityExtensions):
    id: NotRequired[StoredGraphProperty]
    element_id: str
    type: NotRequired[StoredGraphValue]
    description: NotRequired[StoredGraphValue]


class EntityProjection(EntityProjectionFields, total=False):
    mentioned_in_chunks: list[StoredGraphProperty]


class MentionedEntity(EntityExtensions):
    id: StoredGraphProperty
    element_id: str | None
    type: StoredGraphProperty
    description: StoredGraphProperty
    mentioned_in_chunks: list[StoredGraphProperty]


class HoppedEntity(TypedDict):
    id: StoredGraphProperty
    element_id: str | None
    type: StoredGraphProperty
    description: StoredGraphProperty
    mentioned_in_chunks: list[StoredGraphProperty]
    source: str
    min_hops: int
    name: NotRequired[StoredGraphProperty]
    seed_connections: NotRequired[int]
    score: NotRequired[float]


class GraphNeighbor(TypedDict):
    id: StoredGraphProperty
    element_id: str
    type: StoredGraphValue
    name: StoredGraphProperty
    min_hops: int
    seed_connections: int
    source: str
    description: NotRequired[StoredGraphProperty]
    mentioned_in_chunks: NotRequired[list[StoredGraphProperty]]
    score: NotRequired[float]


RetrievedEntity: TypeAlias = (
    PartialRetrievedEntity
    | EntityProjection
    | MentionedEntity
    | HoppedEntity
    | GraphNeighbor
)


class RetrievedChunkExtensions(TypedDict, total=False):
    parent_chunk_id: StoredGraphProperty
    parent_chunk_element_id: str | None
    question_id: StoredGraphProperty
    passage_index: StoredGraphProperty
    chunk_local_index: StoredGraphProperty
    retrieval_local_index: StoredGraphProperty
    position: StoredGraphProperty
    min_hop: int
    source: StoredGraphProperty
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
    text: StoredGraphProperty
    document: StoredGraphProperty
    chunk_id: StoredGraphProperty
    chunk_element_id: str | None
    score: float


class IdentifiedRetrievedChunk(RetrievedChunkExtensions, total=False):
    """Read view after an existing truthy chunk_id guard; no value coercion."""

    chunk_id: Required[StoredGraphProperty]
    text: StoredGraphProperty
    document: StoredGraphProperty
    chunk_element_id: str | None
    score: float


class ElementIdentifiedRetrievedChunk(RetrievedChunkExtensions, total=False):
    """Read view after the existing truthy chunk_element_id guard."""

    chunk_id: StoredGraphProperty
    text: StoredGraphProperty
    document: StoredGraphProperty
    chunk_element_id: Required[str | None]
    score: float


class CompleteRetrievedChunk(RetrievedChunkExtensions):
    text: StoredGraphProperty
    document: StoredGraphProperty
    chunk_id: StoredGraphProperty
    chunk_element_id: str | None
    score: NotRequired[float]


class ScoredRetrievedChunk(RetrievedChunkExtensions):
    text: StoredGraphProperty
    document: StoredGraphProperty
    chunk_id: StoredGraphProperty
    chunk_element_id: str | None
    score: float


RetrievedChunk: TypeAlias = (
    PartialRetrievedChunk
    | IdentifiedRetrievedChunk
    | CompleteRetrievedChunk
    | ScoredRetrievedChunk
)
ChunkT = TypeVar("ChunkT", bound=RetrievedChunk)


class RetrievalProvenance(TypedDict, total=False):
    question_ids: ImportedPropertyValue
    passage_keys: ImportedPropertyValue
    provenance_positions: ImportedPropertyValue


class RetrievedRelationship(RetrievalProvenance, total=False):
    key: str
    source: StoredGraphKey
    source_element_id: str | None
    target: StoredGraphKey
    target_element_id: str | None
    type: str
    element_id: str | None
    negated: bool
    condition: "ImportedPropertyValue | None"
    quantitative: "ImportedPropertyValue | None"
    confidence: float | None


class ProjectedRelationship(RetrievalProvenance, total=False):
    """Sanitized traversal map; imported properties are not normalized."""

    type: Required[str]
    negated: ImportedPropertyValue
    condition: "ImportedPropertyValue | None"
    quantitative: "ImportedPropertyValue | None"
    confidence: Required[float]


class TraversalPath(RetrievalProvenance, total=False):
    path: str
    node_ids: list[StoredGraphProperty]
    hops: int
    seed: str
    target: str
    score: float


class RetrievalDiagnostics(TypedDict, total=False):
    seed_entity_ids: list[StoredGraphKey]
    seed_entity_names: (
        list[str] | list[str | None] | list[StoredGraphProperty] | list[StoredGraphKey]
    )
    seed_entities: (
        list[str] | list[str | None] | list[StoredGraphProperty] | list[StoredGraphKey]
    )
    anchor_mask_entity_ids: list[str]
    anchor_mask_entity_names: list[str]
    ann_match_count: int
    symbolic_match_count: int
    rfge_fired: bool
    subgraph_edge_count: int
    top_ppr_entities: list[tuple[StoredGraphKey, float]]
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
    entities: dict[StoredGraphKey, RetrievedEntity]
    relationships: list[RetrievedRelationship]
    entity_count: int
    relationship_count: int

    graph_neighbors: dict[StoredGraphKey, RetrievedEntity]
    traversal_paths: list[TraversalPath]


class CompleteRetrievalContext(RetrievalContextExtensions, RetrievalCore, total=False):
    entities: dict[StoredGraphKey, RetrievedEntity]
    relationships: list[RetrievedRelationship]
    entity_count: int
    relationship_count: int

    graph_neighbors: dict[StoredGraphKey, RetrievedEntity]
    traversal_paths: list[TraversalPath]


class ScoredRetrievalCore(TypedDict):
    chunks: list[ScoredRetrievedChunk]
    documents: list[StoredGraphProperty]
    total_score: float


class GraphRetrievalContext(RetrievalContextExtensions, ScoredRetrievalCore):
    entities: dict[StoredGraphKey, RetrievedEntity]
    relationships: list[RetrievedRelationship]
    entity_count: NotRequired[int]
    relationship_count: NotRequired[int]

    graph_neighbors: dict[StoredGraphKey, RetrievedEntity]
    traversal_paths: list[TraversalPath]


class CountedGraphRetrievalContext(RetrievalContextExtensions, RetrievalCore):
    entities: dict[StoredGraphKey, RetrievedEntity]
    relationships: list[RetrievedRelationship]
    entity_count: int
    relationship_count: int

    graph_neighbors: dict[StoredGraphKey, RetrievedEntity]
    traversal_paths: list[TraversalPath]


class ScoredCompleteRetrievalContext(
    RetrievalContextExtensions,
    ScoredRetrievalCore,
    total=False,
):
    entities: dict[StoredGraphKey, RetrievedEntity]
    relationships: list[RetrievedRelationship]
    entity_count: int
    relationship_count: int

    graph_neighbors: dict[StoredGraphKey, RetrievedEntity]
    traversal_paths: list[TraversalPath]


class ProducedGraphRetrievalContext(RetrievalContextExtensions, ScoredRetrievalCore):
    entities: dict[StoredGraphKey, RetrievedEntity]
    relationships: list[RetrievedRelationship]
    entity_count: int
    relationship_count: int

    graph_neighbors: dict[StoredGraphKey, RetrievedEntity]
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
    entities: dict[StoredGraphKey, MentionedEntity]
    relationships: list[RetrievedRelationship]
    entity_count: NotRequired[int]
    relationship_count: NotRequired[int]

    graph_neighbors: NotRequired[dict[StoredGraphKey, RetrievedEntity]]
    traversal_paths: NotRequired[list[TraversalPath]]


class FusedChunkRecord(TypedDict):
    chunk: RetrievedChunk
    rrf: float
    support_count: int
    best_score: float


class EvidencePath(TypedDict):
    path_str: str
    node_ids: set[StoredGraphKey]
    hops: int
    provenance_positions: list[ImportedPropertyScalar]
    passage_keys: list[ImportedPropertyScalar]


class GraphExpansion(TypedDict):
    neighbors: dict[StoredGraphKey, GraphNeighbor]
    paths: list[TraversalPath]


class RelationshipRow(RetrievalProvenance):
    """Sanitized Cypher row; only predicate/computed scalar keys are guaranteed."""

    source: StoredGraphKey
    source_element_id: str
    target: StoredGraphKey
    target_element_id: str
    relationship_type: str
    relationship_element_id: str
    negated: NotRequired[ImportedPropertyValue]
    condition: NotRequired["ImportedPropertyValue | None"]
    quantitative: NotRequired["ImportedPropertyValue | None"]
    confidence: float


class HopBucket(TypedDict):
    all_chunk_ids: set[ChunkIdentity]
    new_chunks: list[ScoredRetrievedChunk]
    entities: dict[StoredGraphKey, RetrievedEntity]
    relationships: list[RetrievedRelationship]
    graph_neighbors: dict[StoredGraphKey, RetrievedEntity]
    traversal_paths: list[TraversalPath]


class ChunkProjectionFields(TypedDict):
    """Sanitized raw rows; imported array properties may be omitted."""

    text: NotRequired[StoredGraphProperty]
    chunk_id: NotRequired[StoredGraphProperty]
    chunk_element_id: str
    document: NotRequired[StoredGraphProperty]
    parent_chunk_id: NotRequired[StoredGraphProperty]
    parent_chunk_element_id: NotRequired[str | None]
    question_id: NotRequired[StoredGraphProperty]
    passage_index: NotRequired[StoredGraphProperty]
    chunk_local_index: NotRequired[StoredGraphProperty]
    retrieval_local_index: NotRequired[StoredGraphProperty]
    position: NotRequired[StoredGraphProperty]
    source: NotRequired[StoredGraphProperty]
    kg_name: NotRequired[StoredGraphProperty]
    entities: NotRequired[list[EntityProjection]]
    linked_entity_ids: NotRequired[list[StoredGraphKey]]
    linked_entity_count: NotRequired[int]
    seed_entity_id: NotRequired[StoredGraphKey]


class ChunkProjection(ChunkProjectionFields, total=False):
    score: float


class ScoredChunkProjection(ChunkProjectionFields):
    score: float


class RelationTypeProjection(TypedDict):
    rel_type: str
    cnt: int


class TraversalSampleProjection(TypedDict, total=False):
    nodes: list[StoredGraphProperty]
    node_ids: list[StoredGraphProperty]
    rels: list[ProjectedRelationship]


class TraversalProjection(TypedDict):
    id: NotRequired[StoredGraphValue]
    name: NotRequired[StoredGraphProperty]
    type: NotRequired[StoredGraphValue]
    element_id: str
    min_hops: int
    seed_connections: int
    sample_paths: list[TraversalSampleProjection]


class EdgeProjection(TypedDict):
    src: StoredGraphKey
    tgt: StoredGraphKey
    conf: float


class EmbeddingProjection(TypedDict):
    eid: StoredGraphKey
    emb: NotRequired[StoredGraphProperty]


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


class PromptTemplateFactory(Protocol):
    """Installed chat-template factory restricted to the string pairs we pass."""

    def from_messages(
        self,
        messages: Sequence[tuple[str, str]],
        template_format: Literal["f-string", "mustache", "jinja2"] = "f-string",
    ) -> "ChatPromptTemplate": ...
