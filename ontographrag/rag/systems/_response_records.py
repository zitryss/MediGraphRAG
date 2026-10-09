"""Wire dictionaries assembled by the enhanced and vanilla RAG systems."""

from typing import TYPE_CHECKING, Never, NotRequired, TypedDict

from ontographrag.rag.systems._records import (
    RetrievalContext,
    RetrievedRelationship,
    StoredGraphKey,
    StoredGraphProperty,
    StoredGraphValue,
)

if TYPE_CHECKING:
    from ontographrag.rag.answer_guardrails import GuardrailVerdict
    from ontographrag.rag.reranking import RerankingMetadata


class EmptyRerankingMetadata(TypedDict):
    pass


class GuardrailMetadata(TypedDict):
    enabled: bool
    mode: str
    initial_verdict: NotRequired["GuardrailVerdict"]
    final_decision: NotRequired[str]
    retried: NotRequired[bool]
    retried_insufficient_info: NotRequired[bool]
    retry_verdict: NotRequired["GuardrailVerdict"]


class UsedEntity(TypedDict):
    id: StoredGraphKey
    element_id: str | None
    type: StoredGraphProperty
    description: StoredGraphProperty
    reasoning_context: str


class UsedChunk(TypedDict):
    id: StoredGraphProperty
    element_id: str | None
    text: StoredGraphProperty
    reasoning_context: str


ReasoningEdge = TypedDict(
    "ReasoningEdge",
    {
        "from": StoredGraphKey,
        "to": StoredGraphKey,
        "from_name": StoredGraphProperty,
        "to_name": StoredGraphProperty,
        "relationship": str,
        "reasoning_context": str,
    },
)


class UsedEvidence(TypedDict):
    used_entities: list[UsedEntity]
    used_chunks: list[UsedChunk]
    reasoning_edges: list[ReasoningEdge]
    total_filtered_entities: NotRequired[int]
    total_filtered_chunks: NotRequired[int]
    total_reasoning_edges: NotRequired[int]
    error: NotRequired[str]


class GenerationRetrievalParameters(TypedDict, total=False):
    question_type: str
    similarity_threshold: float
    max_chunks: int
    timeout: float | None
    retrieval_temperature: float
    retrieval_shortlist_factor: int
    retrieval_sample_id: int


class GenerationResponseFields(TypedDict):
    response: str
    context: RetrievalContext
    sources: list[StoredGraphProperty]
    confidence: float
    error: NotRequired[str]
    entities: NotRequired[list[StoredGraphKey]]
    used_entities: NotRequired[list[UsedEntity]]
    used_chunks: NotRequired[list[UsedChunk]]
    reasoning_edges: NotRequired[list[ReasoningEdge]]
    relationships: NotRequired[list[RetrievedRelationship]]
    chunk_count: NotRequired[int]
    entity_count: NotRequired[int]
    relationship_count: NotRequired[int]
    retrieval_params: NotRequired[GenerationRetrievalParameters]
    late_interaction_stage: NotRequired["RerankingMetadata | EmptyRerankingMetadata"]
    late_interaction: NotRequired["RerankingMetadata | EmptyRerankingMetadata"]
    reranker: NotRequired["RerankingMetadata | EmptyRerankingMetadata"]


class GenerationResponse(GenerationResponseFields):
    guardrail: NotRequired[GuardrailMetadata]


class ErrorPayload(TypedDict):
    error: str


class EntityRelationshipDetail(TypedDict):
    related_id: NotRequired[StoredGraphProperty]
    related_type: NotRequired[StoredGraphProperty]
    relationship_type: str | None
    relationship_element_id: str | None


class EntityMention(TypedDict):
    chunk_id: NotRequired[StoredGraphProperty]
    document: NotRequired[StoredGraphProperty]


class EntityDetails(TypedDict):
    id: str
    type: StoredGraphProperty
    description: StoredGraphProperty
    element_id: str
    relationships: list[EntityRelationshipDetail]
    mentions: list[EntityMention]


class KnowledgeGraphStats(TypedDict):
    documents: int
    chunks: int
    entities: int | list[Never]
    relationships: int
    document_names: list[StoredGraphValue]
    has_embeddings: bool


class EntitySearchHit(TypedDict):
    id: StoredGraphProperty
    element_id: str
    type: StoredGraphProperty
    description: str
    score: float
    chunk_mentions: int
    documents: list[StoredGraphValue]


class KnowledgeGraphStatsProjection(TypedDict):
    documents: int
    chunks: int
    entities: int
    relationships: int
    document_names: NotRequired[list[StoredGraphValue]]


class EntitySearchProjection(TypedDict):
    id: NotRequired[StoredGraphProperty]
    element_id: str
    type: NotRequired[StoredGraphProperty]
    name: str
    chunk_mentions: int
    documents: NotRequired[list[StoredGraphValue]]


class EntityDetailsProjection(TypedDict):
    id: str
    element_id: str
    type: NotRequired[StoredGraphProperty]
    description: NotRequired[StoredGraphProperty]
    relationships: NotRequired[list[EntityRelationshipDetail]]
    mentions: NotRequired[list[EntityMention]]
