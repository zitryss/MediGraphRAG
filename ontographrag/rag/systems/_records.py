"""Dictionary records produced and combined by the retrieval systems."""

from typing import TypedDict

from ontographrag.rag.graph_state import GraphStateSummary


class RetrievedEntity(TypedDict, total=False):
    id: str
    element_id: str | None
    type: str
    description: str
    name: str
    mentioned_in_chunks: list[str]
    source: str
    min_hops: int
    score: float


class RetrievedChunk(TypedDict, total=False):
    text: str
    chunk_id: str | None
    chunk_element_id: str | None
    parent_chunk_id: str | None
    parent_chunk_element_id: str | None
    question_id: str | None
    passage_index: int | None
    chunk_local_index: int | None
    retrieval_local_index: int | None
    position: int | None
    source: str | None
    score: float
    document: str
    kg_name: str | None
    entities: list[RetrievedEntity]
    linked_entity_ids: list[str]
    linked_entity_count: int
    adjacent: bool
    reranker_score: float
    late_interaction_score: float
    rrf_score: float
    retrieval_support_count: int


class RetrievalProvenance(TypedDict, total=False):
    question_ids: list[str]
    passage_keys: list[str]
    provenance_positions: list[int]


class RetrievedRelationship(RetrievalProvenance, total=False):
    key: str
    source: str
    source_element_id: str | None
    target: str
    target_element_id: str | None
    type: str
    element_id: str | None
    negated: bool
    condition: str | None
    quantitative: str | None
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
    seed_entity_names: list[str]
    seed_entities: list[str]


class RetrievalSampling(TypedDict):
    temperature: float
    shortlist_factor: int
    sample_id: int
    candidate_limit: int


class RetrievalContext(TypedDict, total=False):
    query: str
    chunks: list[RetrievedChunk]
    entities: dict[str, RetrievedEntity]
    relationships: list[RetrievedRelationship]
    graph_neighbors: dict[str, RetrievedEntity]
    traversal_paths: list[TraversalPath]
    documents: list[str | None]
    total_score: float
    search_method: str
    retrieval_route: str
    kg_name: str | None
    entity_count: int
    relationship_count: int
    seed_entity_count: int
    grounding_quality: float
    diagnostics: RetrievalDiagnostics
    retrieval_sampling: RetrievalSampling
    graph_state: GraphStateSummary
    error: str


class FusedChunkRecord(TypedDict):
    chunk: RetrievedChunk
    rrf: float
    support_count: int
    best_score: float


class EvidencePath(TypedDict):
    path_str: str
    node_ids: set[str]
    hops: int
    provenance_positions: list[int]
    passage_keys: list[str]
