"""Wire dictionaries assembled by the enhanced and vanilla RAG systems."""

from typing import TYPE_CHECKING, Never, NotRequired, TypedDict

from ontographrag.rag.systems._records import RetrievalContext, RetrievedRelationship

if TYPE_CHECKING:
    from ontographrag.rag.answer_guardrails import GuardrailVerdict
    from ontographrag.rag.reranking import RerankingMetadata


class GuardrailMetadata(TypedDict):
    enabled: bool
    mode: str
    initial_verdict: NotRequired["GuardrailVerdict"]
    final_decision: NotRequired[str]
    retried: NotRequired[bool]
    retry_verdict: NotRequired["GuardrailVerdict"]


class UsedEntity(TypedDict):
    id: str
    element_id: str | None
    type: str
    description: str
    reasoning_context: str


class UsedChunk(TypedDict):
    id: str | None
    element_id: str | None
    text: str
    reasoning_context: str


ReasoningEdge = TypedDict(
    "ReasoningEdge",
    {
        "from": str,
        "to": str,
        "from_name": str,
        "to_name": str,
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
    timeout: int
    retrieval_temperature: float
    retrieval_shortlist_factor: int
    retrieval_sample_id: int


class GenerationResponse(TypedDict):
    response: str
    context: RetrievalContext
    sources: list[str | None]
    confidence: float
    error: NotRequired[str]
    entities: NotRequired[list[str]]
    used_entities: NotRequired[list[UsedEntity]]
    used_chunks: NotRequired[list[UsedChunk]]
    reasoning_edges: NotRequired[list[ReasoningEdge]]
    relationships: NotRequired[list[RetrievedRelationship]]
    chunk_count: NotRequired[int]
    entity_count: NotRequired[int]
    relationship_count: NotRequired[int]
    guardrail: NotRequired[GuardrailMetadata]
    retrieval_params: NotRequired[GenerationRetrievalParameters]
    late_interaction_stage: NotRequired["RerankingMetadata | dict[Never, Never]"]
    late_interaction: NotRequired["RerankingMetadata | dict[Never, Never]"]
    reranker: NotRequired["RerankingMetadata | dict[Never, Never]"]
