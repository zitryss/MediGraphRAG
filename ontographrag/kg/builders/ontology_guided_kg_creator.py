import difflib
import hashlib
import importlib
import logging
import math
import os
import re
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Literal,
    Required,
    TypeAlias,
    TypedDict,
    TypeVar,
    cast,
    overload,
)

from typing_extensions import ReadOnly

# Shared helpers re-exported for backward compatibility (tests and callers
# access them through this module).
from ontographrag.kg.builders._creator_shared import (
    _ENTITY_MIN_NAME_LENGTH as _ENTITY_MIN_NAME_LENGTH,
)
from ontographrag.kg.builders._creator_shared import (
    _GENERIC_HUB_ENTITY_BLOCKLIST as _GENERIC_HUB_ENTITY_BLOCKLIST,
)
from ontographrag.kg.builders._creator_shared import (
    AnchorSpan,
    AugmentedPropertyKey,
    ContextPassageInput,
    DecodedJSONValue,
    ExactAnchorSpan,
    PassageGroup,
    RestorationRecord,
    _env_flag,
    _is_valid_entity_name,
)
from ontographrag.kg.builders._enrichment import EnrichmentMixin
from ontographrag.kg.builders._evaluation import EvaluationMixin
from ontographrag.kg.builders._extraction import ExtractionMixin
from ontographrag.kg.builders._ontology import OntologySchemaMixin, SchemaCard
from ontographrag.kg.builders._storage import QueryGraph, StorageMixin
from ontographrag.kg.chunking import TextChunk
from ontographrag.kg.utils.common_functions import load_embedding_model
from ontographrag.schemas.models import (
    OntologySchema,
)
from shared.graph_types import Neo4jProperty
from shared.model_contracts import GenerationProvider

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import NDArray
    from spacy.language import Language, PipeCallable
    from spacy.tokens import Doc


class SectionSegment(TypedDict):
    index: int
    name: str
    start_pos: int
    end_pos: int
    text: str


class NodeFont(TypedDict):
    size: int
    color: str


class RelationshipFont(TypedDict):
    size: int
    align: str


class RelationshipColor(TypedDict):
    color: str


GraphProperties: TypeAlias = dict[AugmentedPropertyKey, object]
RawGraphProperties: TypeAlias = DecodedJSONValue | GraphProperties | dict[str, object]


class PostprocessedEntity(TypedDict, total=False):
    """Extraction guarantees id/type, but retains supplied properties and extras."""

    id: Required[str]
    type: Required[str]
    properties: Required[RawGraphProperties]
    uuid: DecodedJSONValue
    name: DecodedJSONValue
    content_hash: DecodedJSONValue
    provenance_positions: DecodedJSONValue | list[int]
    label: DecodedJSONValue
    embedding: DecodedJSONValue | list[float]
    color: DecodedJSONValue
    size: DecodedJSONValue
    font: DecodedJSONValue | NodeFont
    title: DecodedJSONValue


class IdentifiedEntity(TypedDict, total=False):
    """Entity staging after the existing id string conversion, before type coercion."""

    id: Required[str]
    type: DecodedJSONValue
    properties: RawGraphProperties
    uuid: DecodedJSONValue
    name: DecodedJSONValue
    content_hash: DecodedJSONValue
    provenance_positions: DecodedJSONValue | list[int]
    label: DecodedJSONValue
    embedding: DecodedJSONValue | list[float]
    color: DecodedJSONValue
    size: DecodedJSONValue
    font: DecodedJSONValue | NodeFont
    title: DecodedJSONValue


ResolvedExtractionRelationship = TypedDict(
    "ResolvedExtractionRelationship",
    {
        "source": Required[str],
        "target": Required[str],
        "properties": Required[GraphProperties],
        "type": DecodedJSONValue,
        "id": DecodedJSONValue,
        "from": DecodedJSONValue,
        "to": DecodedJSONValue,
        "label": DecodedJSONValue,
        "negated": DecodedJSONValue,
        "condition": DecodedJSONValue,
        "quantitative": DecodedJSONValue,
        "contradiction_detected": DecodedJSONValue,
        "provenance_positions": DecodedJSONValue | list[int],
        "arrows": DecodedJSONValue,
        "color": DecodedJSONValue | RelationshipColor,
        "font": DecodedJSONValue | RelationshipFont,
    },
    total=False,
)

ReflectedRelationship = TypedDict(
    "ReflectedRelationship",
    {
        "source": Required[DecodedJSONValue],
        "target": Required[DecodedJSONValue],
        "type": Required[DecodedJSONValue],
        "properties": RawGraphProperties,
        "id": DecodedJSONValue,
        "from": DecodedJSONValue,
        "to": DecodedJSONValue,
        "label": DecodedJSONValue,
        "negated": DecodedJSONValue,
        "condition": DecodedJSONValue,
        "quantitative": DecodedJSONValue,
        "contradiction_detected": DecodedJSONValue,
        "provenance_positions": DecodedJSONValue | list[int],
        "arrows": DecodedJSONValue,
        "color": DecodedJSONValue | RelationshipColor,
        "font": DecodedJSONValue | RelationshipFont,
    },
    total=False,
)


RawExtractionRelationship = TypedDict(
    "RawExtractionRelationship",
    {
        "source": DecodedJSONValue,
        "target": DecodedJSONValue,
        "type": DecodedJSONValue,
        "properties": RawGraphProperties,
        "id": DecodedJSONValue,
        "from": DecodedJSONValue,
        "to": DecodedJSONValue,
        "label": DecodedJSONValue,
        "negated": DecodedJSONValue,
        "condition": DecodedJSONValue,
        "quantitative": DecodedJSONValue,
        "contradiction_detected": DecodedJSONValue,
        "provenance_positions": DecodedJSONValue | list[int],
        "arrows": DecodedJSONValue,
        "color": DecodedJSONValue | RelationshipColor,
        "font": DecodedJSONValue | RelationshipFont,
    },
    total=False,
)


class GraphNode(TypedDict, total=False):
    """Entity and visualization fields; properties remain an extensible boundary."""

    id: str
    uuid: str | None
    name: str
    content_hash: str
    provenance_positions: list[int]
    label: str
    type: str
    properties: GraphProperties
    embedding: list[float] | None
    color: str
    size: int
    font: NodeFont
    title: str


class ConstructedNode(TypedDict, total=False):
    """Entities assembled by deterministic extractors with guaranteed core fields."""

    id: Required[str]
    uuid: str
    name: str
    content_hash: str
    provenance_positions: list[int]
    label: str
    type: Required[str]
    properties: Required[dict[str, object]]
    embedding: list[float] | None
    color: str
    size: int
    font: NodeFont
    title: str


class GeneratedNode(TypedDict, total=False):
    """Formatted creator output; metadata inside properties remains arbitrary."""

    id: Required[str]
    uuid: str
    name: str
    content_hash: str
    provenance_positions: list[int]
    label: Required[str]
    type: str
    properties: Required[GraphProperties]
    embedding: Required[list[float] | DecodedJSONValue]
    color: Required[str]
    size: Required[int]
    font: Required[NodeFont]
    title: Required[str]


class EmbeddingMutation(TypedDict, total=False):
    """Field-only view for copying embeddings within one NodeT specialization."""

    embedding: list[float] | DecodedJSONValue


class HarmonizationCandidate(TypedDict, total=False):
    """Entity after successful id/type reads and description mapping access."""

    id: Required[str]
    uuid: DecodedJSONValue
    name: DecodedJSONValue
    content_hash: DecodedJSONValue
    provenance_positions: DecodedJSONValue | list[int]
    label: DecodedJSONValue
    type: Required[str]
    properties: GraphProperties
    embedding: list[float] | DecodedJSONValue
    color: DecodedJSONValue
    size: DecodedJSONValue
    font: DecodedJSONValue | NodeFont
    title: DecodedJSONValue


class NormalizedNode(TypedDict, total=False):
    """Post-harmonization entity; required fields are retained by linker passes."""

    id: Required[str]
    uuid: Required[str]
    name: DecodedJSONValue
    content_hash: DecodedJSONValue
    provenance_positions: DecodedJSONValue | list[int]
    label: DecodedJSONValue
    type: Required[str]
    properties: Required[GraphProperties]
    embedding: list[float] | DecodedJSONValue
    color: DecodedJSONValue
    size: DecodedJSONValue
    font: DecodedJSONValue | NodeFont
    title: DecodedJSONValue


NodeT = TypeVar("NodeT", GraphNode, NormalizedNode)
EntityNode: TypeAlias = GraphNode | NormalizedNode | GeneratedNode | ConstructedNode

GraphRelationship = TypedDict(
    "GraphRelationship",
    {
        "id": str,
        "from": str,
        "to": str,
        "source": str,
        "target": str,
        "type": str,
        "label": str,
        "negated": bool,
        "condition": str | None,
        "quantitative": str | None,
        "contradiction_detected": bool,
        "properties": GraphProperties,
        "provenance_positions": list[int],
        "arrows": str,
        "color": RelationshipColor,
        "font": RelationshipFont,
    },
    total=False,
)

ConstructedRelationship = TypedDict(
    "ConstructedRelationship",
    {
        "id": str,
        "from": str,
        "to": str,
        "source": Required[str],
        "target": Required[str],
        "type": Required[str],
        "label": str,
        "negated": bool,
        "condition": str | None,
        "quantitative": str | None,
        "contradiction_detected": bool,
        "properties": Required[dict[str, object]],
        "provenance_positions": list[int],
        "arrows": str,
        "color": RelationshipColor,
        "font": RelationshipFont,
    },
    total=False,
)

GeneratedRelationship = TypedDict(
    "GeneratedRelationship",
    {
        "id": Required[str],
        "from": Required[str],
        "to": Required[str],
        "source": Required[str],
        "target": Required[str],
        "type": Required[str],
        "label": Required[str],
        "negated": Required[DecodedJSONValue],
        "condition": str | None,
        "quantitative": str | None,
        "contradiction_detected": bool,
        "properties": Required[GraphProperties],
        "provenance_positions": DecodedJSONValue | list[int],
        "arrows": Required[str],
        "color": Required[RelationshipColor],
        "font": Required[RelationshipFont],
    },
    total=False,
)


NormalizedRelationship = TypedDict(
    "NormalizedRelationship",
    {
        "id": DecodedJSONValue,
        "from": DecodedJSONValue,
        "to": DecodedJSONValue,
        "source": Required[str],
        "target": Required[str],
        "type": Required[str],
        "label": DecodedJSONValue,
        "negated": DecodedJSONValue,
        "condition": DecodedJSONValue,
        "quantitative": DecodedJSONValue,
        "contradiction_detected": DecodedJSONValue,
        "properties": Required[GraphProperties],
        "provenance_positions": DecodedJSONValue | list[int],
        "arrows": DecodedJSONValue,
        "color": DecodedJSONValue | RelationshipColor,
        "font": DecodedJSONValue | RelationshipFont,
    },
    total=False,
)


RelationshipT = TypeVar(
    "RelationshipT",
    bound=GraphRelationship
    | NormalizedRelationship
    | ResolvedExtractionRelationship
    | ReflectedRelationship
    | RawExtractionRelationship,
)
RelationshipRecord: TypeAlias = (
    GraphRelationship
    | NormalizedRelationship
    | GeneratedRelationship
    | ConstructedRelationship
)

EntityLookupKey: TypeAlias = str | tuple[str, str]
EntityLookup: TypeAlias = dict[EntityLookupKey, GraphNode]
NormalizedEntityLookup: TypeAlias = dict[EntityLookupKey, NormalizedNode]
EntityGroupKey: TypeAlias = (
    str | tuple[str, str, str] | tuple[str | tuple[str, str, str], str]
)


ExtractionEntity: TypeAlias = GraphNode | PostprocessedEntity
ExtractionRelationship: TypeAlias = (
    GraphRelationship
    | ResolvedExtractionRelationship
    | ReflectedRelationship
    | RawExtractionRelationship
)
RelationshipRecoveryInput: TypeAlias = (
    DecodedJSONValue | Sequence[ExtractionRelationship | DecodedJSONValue]
)
HarmonizableEntity: TypeAlias = (
    EntityNode | PostprocessedEntity | HarmonizationCandidate
)
HarmonizableRelationship: TypeAlias = (
    RelationshipRecord
    | ResolvedExtractionRelationship
    | ReflectedRelationship
    | RawExtractionRelationship
)


class ExtractionResult(TypedDict):
    entities: ReadOnly[Sequence[ExtractionEntity]]
    relationships: ReadOnly[Sequence[ExtractionRelationship]]


class PostprocessedExtractionResult(TypedDict):
    entities: list[PostprocessedEntity]
    relationships: list[ResolvedExtractionRelationship | ReflectedRelationship]


class ComponentSummary(TypedDict):
    id: str
    component_index: int
    entity_ids: list[str]
    top_entities: list[str]
    entity_count: int
    relationship_count: int
    text: str


class GraphSummary(TypedDict):
    id: str
    text: str
    component_count: int


class ClaimRecord(TypedDict):
    id: str
    text: str
    polarity: str
    confidence: float
    entity_ids: list[str]
    provenance_positions: list[DecodedJSONValue]
    component_index: int
    relationship_index: int


class FragmentationBridge(TypedDict):
    source_id: str
    target_id: str
    source_name: str
    target_name: str
    confidence: float
    reason: str
    shared_alias_keys: list[str]
    left_component: int
    right_component: int


class GraphComponent(TypedDict):
    entity_ids: list[str]
    relationships: list[RelationshipRecord]
    entity_count: int
    relationship_count: int
    degree: Counter[str]
    top_entities: list[str]


class GraphEnrichment(TypedDict, total=False):
    component_summaries: list[ComponentSummary]
    claims: list[ClaimRecord]
    graph_summary: GraphSummary | None
    fragmentation_bridges: list[FragmentationBridge]


class GraphMetadata(TypedDict, total=False):
    total_chunks: int
    total_passages: int
    total_entities: int
    total_relationships: int
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    embedding_dimension: int
    ontology_classes: int
    ontology_relationships: int
    extraction_method: str
    kg_name: str | None
    provider: str
    model: str
    max_chunks_setting: int | None
    created_at: str
    visualization_ready: bool
    file_name: str | None
    doc_hash: str | None
    schema_card: SchemaCard
    schema_enforcement_dropped_entities: int
    schema_enforcement_dropped_relationships: int
    harmonization_relationships_dropped_unmapped: int
    harmonization_relationships_dropped_schema_mismatch: int
    harmonization_relationships_deduped: int
    harmonization_relationship_contradiction_groups: int
    harmonization_relationship_contradiction_edges: int
    anchor_grounded_entities: int
    restoration_full_relationships: int
    restoration_partial_relationships: int
    graph_enrichment: GraphEnrichment
    component_summary_count: int
    claim_count: int
    fragmentation_bridge_count: int
    doc_metadata: Mapping[str, object]
    stored_in_neo4j: bool
    synonym_merges: int
    kg_type: str
    source_knowledge_graphs: int
    stored_relationships: int
    relationship_store_failures: int
    relationships_skipped_low_confidence: int
    relationships_skipped_schema_mismatch: int
    relationships_reverified_kept: int
    relationships_reverified_rejected: int
    relationship_store_ratio: float | str | None
    relationship_store_failure_ratio: float
    relationship_store_degraded: bool


class GeneratedGraphMetadata(TypedDict, total=False):
    total_chunks: Required[int]
    total_passages: int
    total_entities: Required[int]
    total_relationships: Required[int]
    chunk_size: Required[int]
    chunk_overlap: Required[int]
    embedding_model: Required[str]
    embedding_dimension: Required[int]
    ontology_classes: Required[int]
    ontology_relationships: Required[int]
    extraction_method: Required[str]
    kg_name: Required[str | None]
    provider: str
    model: str
    max_chunks_setting: int | None
    created_at: Required[str]
    visualization_ready: Required[bool]
    file_name: Required[str | None]
    doc_hash: Required[str | None]
    schema_card: SchemaCard
    schema_enforcement_dropped_entities: int
    schema_enforcement_dropped_relationships: int
    harmonization_relationships_dropped_unmapped: int
    harmonization_relationships_dropped_schema_mismatch: int
    harmonization_relationships_deduped: int
    harmonization_relationship_contradiction_groups: int
    harmonization_relationship_contradiction_edges: int
    anchor_grounded_entities: int
    restoration_full_relationships: int
    restoration_partial_relationships: int
    graph_enrichment: GraphEnrichment
    component_summary_count: int
    claim_count: int
    fragmentation_bridge_count: int
    doc_metadata: Mapping[str, object]
    stored_in_neo4j: bool
    synonym_merges: int
    kg_type: str
    source_knowledge_graphs: int
    stored_relationships: int
    relationship_store_failures: int
    relationships_skipped_low_confidence: int
    relationships_skipped_schema_mismatch: int
    relationships_reverified_kept: int
    relationships_reverified_rejected: int
    relationship_store_ratio: float | str | None
    relationship_store_failure_ratio: float
    relationship_store_degraded: bool


class KnowledgeGraph(TypedDict):
    nodes: ReadOnly[Sequence[EntityNode]]
    relationships: ReadOnly[Sequence[RelationshipRecord]]
    chunks: list[TextChunk]
    metadata: ReadOnly[GraphMetadata | GeneratedGraphMetadata]


class GeneratedKnowledgeGraph(TypedDict):
    """Base creator output, distinct from patient/bulk harmonized graph records."""

    nodes: list[GeneratedNode]
    relationships: list[GeneratedRelationship]
    chunks: list[TextChunk]
    metadata: GeneratedGraphMetadata


class GraphChunkInput(TypedDict, total=False):
    """Legacy or caller-owned chunk input; missing provenance remains missing."""

    text: ReadOnly[str]
    chunk_id: ReadOnly[int]
    position: ReadOnly[int]
    start_pos: ReadOnly[int]
    end_pos: ReadOnly[int]
    total_chunks: ReadOnly[int]
    embedding: ReadOnly[list[float] | None]
    section_name: ReadOnly[str]
    source: ReadOnly[str | None]
    dataset: ReadOnly[str]
    question_id: ReadOnly[str | None]
    passage_index: ReadOnly[int]
    chunk_local_index: ReadOnly[int]
    source_title: ReadOnly[object]
    source_scope_key: ReadOnly[str | None]


class KnowledgeGraphInput(TypedDict, total=False):
    """Partial legacy graph input, separate from guaranteed generation output."""

    nodes: ReadOnly[Sequence[EntityNode]]
    relationships: ReadOnly[Sequence[RelationshipRecord]]
    chunks: ReadOnly[Sequence[GraphChunkInput]]
    metadata: ReadOnly[GraphMetadata | GeneratedGraphMetadata]


class MetadataReadyKnowledgeGraphInput(KnowledgeGraphInput):
    """Read view after setdefault establishes only the metadata key."""

    metadata: ReadOnly[GraphMetadata | GeneratedGraphMetadata]


class MutableKnowledgeGraphInput(TypedDict, total=False):
    """Exact mutation view for inserting absent metadata without replacing it."""

    nodes: ReadOnly[Sequence[EntityNode]]
    relationships: ReadOnly[Sequence[RelationshipRecord]]
    chunks: ReadOnly[Sequence[GraphChunkInput]]
    metadata: GraphMetadata | GeneratedGraphMetadata


class RelationshipProvenance(TypedDict):
    provenance_positions: list[int]
    question_ids: list[str]
    passage_keys: list[str]


class UMLSLink(TypedDict):
    cui: str
    score: float
    name: str | None


class SynonymQueryRow(TypedDict):
    eid: str
    name: Neo4jProperty
    emb: Neo4jProperty
    degree: int
    etype: Neo4jProperty


class CreatorContextEntity(TypedDict):
    id: Neo4jProperty
    type: Neo4jProperty
    description: Neo4jProperty


class CreatorContextChunk(TypedDict):
    text: Neo4jProperty
    chunk_id: Neo4jProperty
    score: float
    document: Neo4jProperty
    entities: list[CreatorContextEntity]


class CreatorRagContext(TypedDict):
    query: str
    chunks: list[CreatorContextChunk]
    entities: list[Neo4jProperty]
    documents: list[Neo4jProperty]


class _CreatorContextAccumulator(TypedDict):
    query: str
    chunks: list[CreatorContextChunk]
    entities: set[Neo4jProperty]
    documents: set[Neo4jProperty]


class OntologyGuidedKGCreator(
    EnrichmentMixin,
    EvaluationMixin,
    ExtractionMixin,
    OntologySchemaMixin,
    StorageMixin,
):
    """
    Ontology-Guided Knowledge Graph Creator that properly extracts entities from PDF content
    using LLM with ontology guidance for better entity classification and relationships
    """

    _SECTION_HEADER_RE: re.Pattern[str] = re.compile(
        r"(?m)^(?:"
        r"\d+[\.\d]*\s*"  # optional numbering: "2.", "2.1."
        r")?"
        r"(?P<header>"
        r"abstract|introduction|background|methods?|materials?\s+and\s+methods?|"
        r"experimental\s+(?:procedures?|design)|study\s+design|"
        r"results?(?:\s+and\s+discussion)?|"
        r"discussion|conclusions?|summary|"
        r"statistical\s+analysis|data\s+analysis|"
        r"supplementary|acknowledgements?|references?"
        r")"
        r"[:\s]*$",
        re.IGNORECASE,
    )
    _QUALIFIER_KEYWORDS: re.Pattern[str] = re.compile(
        r"\b(?:condition|experiment|treat(?:ed|ment)|knockout|knock[- ]?out|"
        r"mutant|patient|cohort|model|culture|in\s+vitro|in\s+vivo|"
        r"hypox|normox|baseline|control|express(?:ed|ion)|stimulat|"
        r"inhibit|activat|induc|depleted|overexpress|transfect|"
        r"under\s+these|such\s+conditions?|this\s+(?:model|system|context|protocol)|"
        r"these\s+(?:cells?|conditions?|animals?|patients?|mice|rats?))\b",
        re.IGNORECASE,
    )
    _COREF_MARKERS: re.Pattern[str] = re.compile(
        r"\b(?:"
        r"these\s+(?:conditions?|cells?|animals?|mice|rats?|patients?|results?|findings?|data)|"
        r"this\s+(?:model|system|treatment|context|approach|protocol|setup|condition|disease|disorder|syndrome|gene|protein|enzyme|receptor|biomarker)|"
        r"such\s+conditions?|"
        r"under\s+these\s+(?:conditions?|circumstances?)|"
        r"the\s+(?:treated|knockout|mutant|control)\s+(?:group|cells?|mice|animals?)|"
        r"the\s+above[-\s](?:mentioned\s+)?(?:conditions?|treatment|model|protocol)|"
        r"as\s+(?:described|mentioned|stated)\s+(?:above|previously|earlier)"
        r")\b",
        re.IGNORECASE,
    )
    _TITLE_AWARE_BUNDLE_DATASETS: set[str] = {"hotpotqa", "2wikimultihopqa", "musique"}

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str | None = "password",
        neo4j_database: str = "neo4j",
        embedding_model: str = "sentence_transformers",
        ontology_path: str | None = None,
        enable_coreference_resolution: bool = False,
        enable_heuristic_coreference_resolution: bool = True,
        retrieval_chunk_size: int | None = None,
        retrieval_chunk_overlap: int | None = None,
        strict_ontology: bool = True,
        self_consistency_n: int = 1,
        few_shot_example_count: int = 2,
        min_triple_confidence: float = 0.15,
        relationship_type_similarity_threshold: float = 0.62,
        enable_low_confidence_triple_reverification: bool = False,
        low_confidence_reverify_threshold: float = 0.4,
        enable_umls_linking: bool = False,
        umls_spacy_model: str | None = None,
        enable_anchor_constrained_extraction: bool = True,
        enable_self_reflection: bool = True,
        enable_anchor_coverage_supplement: bool = True,
        enable_cross_passage_relation_recovery: bool = True,
        enable_soft_entity_linking: bool = False,
        soft_entity_similarity_threshold: float = 0.88,
        enable_fragmentation_repair: bool = False,
        fragmentation_bridge_similarity_threshold: float = 0.92,
        max_fragmentation_bridges: int = 8,
        enable_graph_summaries: bool = False,
        enable_claim_extraction: bool = False,
        max_summary_entities: int = 6,
        max_summary_relationships: int = 6,
    ) -> None:
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        env_retrieval_chunk_size: str | None = os.getenv("RETRIEVAL_CHUNK_SIZE")
        env_retrieval_chunk_overlap: str | None = os.getenv("RETRIEVAL_CHUNK_OVERLAP")
        resolved_retrieval_chunk_size: int | None = retrieval_chunk_size
        if resolved_retrieval_chunk_size is None and env_retrieval_chunk_size:
            try:
                resolved_retrieval_chunk_size = int(env_retrieval_chunk_size)
            except ValueError:
                logging.warning(
                    "Invalid RETRIEVAL_CHUNK_SIZE=%r; falling back to 256",
                    env_retrieval_chunk_size,
                )
        resolved_retrieval_chunk_overlap: int | None = retrieval_chunk_overlap
        if resolved_retrieval_chunk_overlap is None and env_retrieval_chunk_overlap:
            try:
                resolved_retrieval_chunk_overlap = int(env_retrieval_chunk_overlap)
            except ValueError:
                logging.warning(
                    "Invalid RETRIEVAL_CHUNK_OVERLAP=%r; falling back to 64",
                    env_retrieval_chunk_overlap,
                )
        self.retrieval_chunk_size: int = max(
            64,
            int(resolved_retrieval_chunk_size or 256),
        )
        default_retrieval_overlap: int | None = resolved_retrieval_chunk_overlap
        if default_retrieval_overlap is None:
            default_retrieval_overlap = min(64, max(16, self.retrieval_chunk_size // 4))
        self.retrieval_chunk_overlap: int = max(
            0,
            min(int(default_retrieval_overlap), self.retrieval_chunk_size - 1),
        )
        self.cross_chunk_relation_window: int = max(
            2,
            int(os.getenv("KG_CROSS_CHUNK_RELATION_WINDOW", "3") or 3),
        )
        self.cross_section_relation_window: int = max(
            2,
            int(os.getenv("KG_CROSS_SECTION_RELATION_WINDOW", "2") or 2),
        )
        self.cross_passage_relation_window: int = max(
            1,
            int(os.getenv("KG_CROSS_PASSAGE_RELATION_WINDOW", "2") or 2),
        )
        self.max_relationship_prompt_entities: int = max(
            8,
            int(os.getenv("KG_RELATION_PROMPT_ENTITY_CAP", "40") or 40),
        )
        self.enable_coreference_resolution: bool = enable_coreference_resolution
        self.enable_heuristic_coreference_resolution: bool = (
            enable_heuristic_coreference_resolution
        )
        self.neo4j_uri: str = neo4j_uri
        self.neo4j_user: str = neo4j_user
        self.neo4j_password: str | None = neo4j_password
        self.neo4j_database: str = neo4j_database
        self.embedding_model: str = embedding_model
        self.ontology_path: str | None = ontology_path
        self.strict_ontology: bool = strict_ontology
        self.self_consistency_n: int = max(
            1,
            int(
                os.getenv("KG_SELF_CONSISTENCY_N", str(self_consistency_n))
                or self_consistency_n,
            ),
        )
        self.few_shot_example_count: int = max(
            0,
            int(
                os.getenv("KG_FEW_SHOT_EXAMPLE_COUNT", str(few_shot_example_count))
                or few_shot_example_count,
            ),
        )
        try:
            self.min_triple_confidence: float = max(
                0.0,
                min(
                    1.0,
                    float(
                        os.getenv(
                            "KG_MIN_TRIPLE_CONFIDENCE",
                            str(min_triple_confidence),
                        )
                        or min_triple_confidence,
                    ),
                ),
            )
        except ValueError:
            self.min_triple_confidence = min_triple_confidence
        try:
            self.relationship_type_similarity_threshold: float = max(
                0.0,
                min(
                    1.0,
                    float(
                        os.getenv(
                            "KG_RELATIONSHIP_TYPE_SIMILARITY_THRESHOLD",
                            str(relationship_type_similarity_threshold),
                        )
                        or relationship_type_similarity_threshold,
                    ),
                ),
            )
        except ValueError:
            self.relationship_type_similarity_threshold = (
                relationship_type_similarity_threshold
            )
        self.enable_low_confidence_triple_reverification: bool = _env_flag(
            os.getenv("KG_ENABLE_LOW_CONFIDENCE_TRIPLE_REVERIFY"),
            enable_low_confidence_triple_reverification,
        )
        try:
            self.low_confidence_reverify_threshold: float = max(
                0.0,
                min(
                    1.0,
                    float(
                        os.getenv(
                            "KG_LOW_CONFIDENCE_REVERIFY_THRESHOLD",
                            str(low_confidence_reverify_threshold),
                        )
                        or low_confidence_reverify_threshold,
                    ),
                ),
            )
        except ValueError:
            self.low_confidence_reverify_threshold = low_confidence_reverify_threshold
        self.enable_umls_linking: bool = _env_flag(
            os.getenv("KG_ENABLE_UMLS_LINKING"),
            enable_umls_linking,
        )
        self.umls_spacy_model: str = (
            os.getenv("KG_UMLS_SPACY_MODEL", umls_spacy_model or "en_core_sci_sm")
            or "en_core_sci_sm"
        )
        self.enable_anchor_constrained_extraction: bool = _env_flag(
            os.getenv("KG_ENABLE_ANCHOR_CONSTRAINED_EXTRACTION"),
            enable_anchor_constrained_extraction,
        )
        self.enable_self_reflection: bool = _env_flag(
            os.getenv("KG_ENABLE_SELF_REFLECTION"),
            enable_self_reflection,
        )
        self.enable_anchor_coverage_supplement: bool = _env_flag(
            os.getenv("KG_ENABLE_ANCHOR_COVERAGE_SUPPLEMENT"),
            enable_anchor_coverage_supplement,
        )
        self.enable_cross_passage_relation_recovery: bool = _env_flag(
            os.getenv("KG_ENABLE_CROSS_PASSAGE_RELATION_RECOVERY"),
            enable_cross_passage_relation_recovery,
        )
        self.enable_soft_entity_linking: bool = _env_flag(
            os.getenv("KG_ENABLE_SOFT_ENTITY_LINKING"),
            enable_soft_entity_linking,
        )
        try:
            self.soft_entity_similarity_threshold: float = max(
                0.0,
                min(
                    1.0,
                    float(
                        os.getenv(
                            "KG_SOFT_ENTITY_SIMILARITY_THRESHOLD",
                            str(soft_entity_similarity_threshold),
                        )
                        or soft_entity_similarity_threshold,
                    ),
                ),
            )
        except ValueError:
            self.soft_entity_similarity_threshold = soft_entity_similarity_threshold
        self.enable_fragmentation_repair: bool = _env_flag(
            os.getenv("KG_ENABLE_FRAGMENTATION_REPAIR"),
            enable_fragmentation_repair,
        )
        try:
            self.fragmentation_bridge_similarity_threshold: float = max(
                0.0,
                min(
                    1.0,
                    float(
                        os.getenv(
                            "KG_FRAGMENTATION_BRIDGE_SIMILARITY_THRESHOLD",
                            str(fragmentation_bridge_similarity_threshold),
                        )
                        or fragmentation_bridge_similarity_threshold,
                    ),
                ),
            )
        except ValueError:
            self.fragmentation_bridge_similarity_threshold = (
                fragmentation_bridge_similarity_threshold
            )
        try:
            self.max_fragmentation_bridges: int = max(
                0,
                int(
                    os.getenv(
                        "KG_MAX_FRAGMENTATION_BRIDGES",
                        str(max_fragmentation_bridges),
                    )
                    or max_fragmentation_bridges,
                ),
            )
        except ValueError:
            self.max_fragmentation_bridges = max_fragmentation_bridges
        self.enable_graph_summaries: bool = _env_flag(
            os.getenv("KG_ENABLE_GRAPH_SUMMARIES"),
            enable_graph_summaries,
        )
        self.enable_claim_extraction: bool = _env_flag(
            os.getenv("KG_ENABLE_CLAIM_EXTRACTION"),
            enable_claim_extraction,
        )
        try:
            self.max_summary_entities: int = max(
                1,
                int(
                    os.getenv(
                        "KG_MAX_SUMMARY_ENTITIES",
                        str(max_summary_entities),
                    )
                    or max_summary_entities,
                ),
            )
        except ValueError:
            self.max_summary_entities = max_summary_entities
        try:
            self.max_summary_relationships: int = max(
                1,
                int(
                    os.getenv(
                        "KG_MAX_SUMMARY_RELATIONSHIPS",
                        str(max_summary_relationships),
                    )
                    or max_summary_relationships,
                ),
            )
        except ValueError:
            self.max_summary_relationships = max_summary_relationships
        self._umls_linker_state: str = "disabled"
        self._umls_nlp: Language | None = None
        self._triple_reverification_cache: dict[str, bool] = {}
        self._last_schema_enforcement_stats: dict[str, int] = {
            "dropped_entities": 0,
            "dropped_relationships": 0,
            "kept_entities": 0,
            "kept_relationships": 0,
        }
        self._last_relationship_harmonization_stats: dict[str, int] = {
            "kept": 0,
            "dropped_unmapped": 0,
            "dropped_schema_mismatch": 0,
            "deduped": 0,
        }
        self._last_relationship_contradiction_stats: dict[str, int] = {
            "contradiction_groups": 0,
            "contradiction_edges": 0,
        }

        # Initialize embedding model
        self.embedding_function, self.embedding_dimension = load_embedding_model(
            embedding_model,
        )
        logging.info(
            f"Initialized embedding model: {embedding_model}, dimension: {self.embedding_dimension}",
        )

        # Load ontology if provided
        self.ontology_classes: list[dict[str, str]] = []
        self.ontology_relationships: list[dict[str, str]] = []
        self._ontology_schema: OntologySchema | None = None
        if ontology_path and os.path.exists(ontology_path):
            logging.info(
                f"Ontology file exists at: {ontology_path} (size: {os.path.getsize(ontology_path)} bytes)",
            )
            try:
                self._load_ontology(ontology_path)
                logging.info(
                    f"✅ Successfully loaded ontology: {len(self.ontology_classes)} classes, {len(self.ontology_relationships)} relationships",
                )

                # Validate ontology structure to prevent "string indices must be integers" errors
                self._validate_ontology_structure()

            except Exception as e:
                logging.error(f"❌ Failed to load ontology: {e}")
                if self.strict_ontology:
                    raise
                # Continue with empty ontology - LLM extraction will still work
                self.ontology_classes = []
                self.ontology_relationships = []
        else:
            if ontology_path:
                logging.warning(f"Ontology file not found: {ontology_path}")
                logging.info(
                    f"Available files in temp dir: {os.listdir(os.path.dirname(ontology_path)) if ontology_path else 'N/A'}",
                )
                if self.strict_ontology:
                    raise FileNotFoundError(f"Ontology file not found: {ontology_path}")
            else:
                logging.info("No ontology provided - using basic LLM entity extraction")
            # No ontology - use empty lists (will fall back to pattern matching)

        # Pre-compute ontology class label embeddings for semantic classification
        self._ontology_class_embeddings: list[
            tuple[str, list[float]]
        ] = []  # [(class_id, embedding), ...]
        self._ontology_relationship_embeddings: list[
            tuple[str, str, list[float]]
        ] = []  # [(id, label, embedding), ...]
        if self.ontology_classes and self.embedding_function:
            try:
                labels: list[str] = [cls["label"] for cls in self.ontology_classes]
                embeddings: list[list[float]] = self.embedding_function.embed_documents(
                    labels,
                )
                self._ontology_class_embeddings = [
                    (cls["id"], emb)
                    for cls, emb in zip(self.ontology_classes, embeddings)
                ]
                logging.info(
                    f"Pre-computed embeddings for {len(self._ontology_class_embeddings)} ontology classes",
                )
            except Exception as e:
                logging.warning(
                    f"Could not pre-compute ontology class embeddings: {e}. Falling back to keyword matching.",
                )
        if self.embedding_function:
            try:
                if self._ontology_schema and self._ontology_schema.relationship_types:
                    rel_labels: list[str] = [
                        rt.label for rt in self._ontology_schema.relationship_types
                    ]
                    embeddings = self.embedding_function.embed_documents(rel_labels)
                    self._ontology_relationship_embeddings = [
                        (rt.id, rt.label, emb)
                        for rt, emb in zip(
                            self._ontology_schema.relationship_types,
                            embeddings,
                        )
                    ]
                elif self.ontology_relationships:
                    rel_labels = [
                        rel.get("label", rel.get("id", ""))
                        for rel in self.ontology_relationships
                    ]
                    embeddings = self.embedding_function.embed_documents(rel_labels)
                    self._ontology_relationship_embeddings = [
                        (rel.get("id", ""), rel.get("label", rel.get("id", "")), emb)
                        for rel, emb in zip(self.ontology_relationships, embeddings)
                    ]
                if self._ontology_relationship_embeddings:
                    logging.info(
                        "Pre-computed embeddings for %d ontology relationship types",
                        len(self._ontology_relationship_embeddings),
                    )
            except Exception as e:
                logging.warning(
                    "Could not pre-compute ontology relationship embeddings: %s. Falling back to string matching.",
                    e,
                )

    @staticmethod
    def _boundary_pattern(name: str, *, ignore_case: bool = False) -> re.Pattern[str]:
        """Boundary-aware regex for entity/relation span restoration."""
        flags: int = re.IGNORECASE if ignore_case else 0
        prefix: str = (
            r"(?<!\w)" if not name[:1].isalnum() and name[:1] != "_" else r"\b"
        )
        suffix: str = (
            r"(?!\w)" if not name[-1:].isalnum() and name[-1:] != "_" else r"\b"
        )
        return re.compile(prefix + re.escape(name) + suffix, flags)

    @staticmethod
    def _surface_forms(name: str) -> set[str]:
        """Return underscore/space variants for exact-text restoration."""
        forms: set[str] = {name}
        forms.add(name.replace("_", " "))
        forms.add(name.replace(" ", "_"))
        return {f for f in forms if isinstance(f, str) and f.strip()}

    def _find_exact_text_spans(
        self,
        text: str,
        candidates: list[str],
        *,
        start_offset: int = 0,
        chunk_position: int | None = None,
        max_matches_per_candidate: int = 12,
    ) -> list[ExactAnchorSpan]:
        """Locate exact candidate text spans with absolute character offsets."""
        if not isinstance(text, str) or not text.strip():
            return []

        unique_candidates: list[str] = []
        seen_candidates: set[str] = set()
        for candidate in candidates or []:
            if not isinstance(candidate, str):
                continue
            cleaned: str = candidate.strip()
            if not cleaned or cleaned.lower() in seen_candidates:
                continue
            seen_candidates.add(cleaned.lower())
            unique_candidates.append(cleaned)

        spans: list[ExactAnchorSpan] = []
        for candidate in sorted(unique_candidates, key=len, reverse=True):
            forms: list[str] = sorted(
                self._surface_forms(candidate),
                key=len,
                reverse=True,
            )
            candidate_matches: int = 0
            for form in forms:
                pattern: re.Pattern[str] = self._boundary_pattern(
                    form,
                    ignore_case=True,
                )
                for match in pattern.finditer(text):
                    candidate_matches += 1
                    spans.append(
                        {
                            "text": text[match.start() : match.end()],
                            "start": start_offset + match.start(),
                            "end": start_offset + match.end(),
                            "chunk_position": chunk_position,
                            "local_start": match.start(),
                            "local_end": match.end(),
                        },
                    )
                    if candidate_matches >= max_matches_per_candidate:
                        break
                if candidate_matches >= max_matches_per_candidate:
                    break
        return cast(list[ExactAnchorSpan], self._merge_anchor_spans(spans))

    def _detect_section_headers(self, text: str) -> list[tuple[int, str]]:
        """Return list of (char_position, normalised_section_name) for every section
        header found in *text*, in document order.

        e.g. [(0, 'Abstract'), (412, 'Introduction'), (2105, 'Methods'), ...]
        """
        headers: list[tuple[int, str]] = []
        for m in self._SECTION_HEADER_RE.finditer(text):
            raw: str = m.group("header").strip()
            # Normalise to title case; collapse "materials and methods" variants
            if re.match(r"materials?\s+and\s+methods?", raw, re.IGNORECASE):
                normalised: str = "Methods"
            elif re.match(
                r"experimental\s+(?:procedures?|design)|study\s+design",
                raw,
                re.IGNORECASE,
            ):
                normalised = "Methods"
            elif re.match(r"results?\s+and\s+discussion", raw, re.IGNORECASE):
                normalised = "Results"
            else:
                normalised = raw.title()
            headers.append((m.start(), normalised))
        return headers

    def _get_section_for_position(
        self,
        pos: int,
        section_headers: list[tuple[int, str]],
    ) -> str | None:
        """Return the section name that covers character position *pos*."""
        current: str | None = None
        for header_pos, name in section_headers:
            if header_pos <= pos:
                current = name
            else:
                break
        return current

    def _has_coreference_markers(self, text: str) -> bool:
        """Return True if *text* contains demonstrative coreference markers."""
        return bool(self._COREF_MARKERS.search(text))

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Return cosine similarity for two dense vectors."""
        denom: float = (sum(a * a for a in vec_a) ** 0.5 + 1e-9) * (
            sum(b * b for b in vec_b) ** 0.5 + 1e-9
        )
        return float(sum(a * b for a, b in zip(vec_a, vec_b)) / denom)

    def _resolve_coreferences_heuristic(
        self,
        chunk_text: str,
        previous_entities: "Sequence[ExtractionEntity] | None",
    ) -> str:
        """Cheap pronoun/coreference resolver using previously seen entities.

        This is intentionally conservative. It only rewrites typed demonstrative
        phrases such as "this disease" or "these patients" when a recent entity
        of the matching coarse type exists in the preceding context.
        """
        if not previous_entities or not self._has_coreference_markers(chunk_text):
            return chunk_text

        type_aliases: dict[str, set[str]] = {
            "disease": {"disease", "condition", "disorder", "illness", "syndrome"},
            "treatment": {"treatment", "therapy", "drug", "medication", "procedure"},
            "gene": {"gene", "mutation", "variant", "allele"},
            "protein": {"protein", "enzyme", "receptor", "biomarker"},
            "patient": {"patient", "patients", "subject", "subjects", "cohort"},
            "cell": {"cell", "cells", "line", "lines"},
            "animal": {"animal", "animals", "mouse", "mice", "rat", "rats"},
            "model": {"model", "system", "context", "protocol", "setup"},
            "person": {"person", "people", "actor", "director", "scientist"},
            "organization": {
                "organization",
                "company",
                "museum",
                "institute",
                "university",
            },
            "location": {"city", "country", "state", "province", "location"},
            "work": {"film", "movie", "book", "song", "album", "series"},
        }
        type_preferences: dict[str, tuple[str, ...]] = {
            "disease": ("Disease", "MedicalCondition", "Condition", "Disorder"),
            "treatment": ("Treatment", "Drug", "Medication", "MedicalProcedure"),
            "gene": ("Gene", "Genetic Mutation", "Genetic Variant", "Genetic Disorder"),
            "protein": ("Protein", "Biomarker", "Enzyme", "Receptor"),
            "patient": ("Patient", "PopulationGroup", "Person"),
            "cell": ("Cell", "Cell Type", "CellType", "Cell Line"),
            "animal": ("Organism", "PopulationGroup"),
            "model": ("Model", "Concept", "Biological System"),
            "person": ("Person", "Physician"),
            "organization": ("Organization", "Hospital", "Institute"),
            "location": ("Location", "Anatomical Structure", "AnatomicalStructure"),
            "work": ("Work", "CreativeWork", "Film"),
        }

        def _choose_antecedent(category: str) -> str | None:
            preferred_types: tuple[str, ...] = type_preferences.get(category, ())
            category_aliases: set[str] = type_aliases.get(category, {category})
            for entity in reversed(previous_entities):
                entity_name: str = str(entity.get("id") or "").strip()
                entity_type: str = str(entity.get("type") or "")
                if not entity_name:
                    continue
                if entity_type in preferred_types:
                    return entity_name
                entity_desc: str = str(
                    (entity.get("properties") or {}).get("description") or "",
                ).lower()
                if any(alias in entity_desc for alias in category_aliases):
                    return entity_name
            for entity in reversed(previous_entities):
                entity_name = str(entity.get("id") or "").strip()
                if entity_name:
                    return entity_name
            return None

        resolved: str = chunk_text
        replacement_patterns: list[tuple[str, re.Pattern[str]]] = [
            (
                "disease",
                re.compile(
                    r"\b(?:this|the)\s+(?:disease|condition|disorder|syndrome)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "treatment",
                re.compile(
                    r"\b(?:this|the)\s+(?:treatment|therapy|drug|medication|procedure)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "gene",
                re.compile(
                    r"\b(?:this|the)\s+(?:gene|mutation|variant|allele)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "protein",
                re.compile(
                    r"\b(?:this|the)\s+(?:protein|enzyme|receptor|biomarker)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "patient",
                re.compile(
                    r"\b(?:these|the)\s+(?:patients|subjects|cases|participants)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "cell",
                re.compile(r"\b(?:these|the)\s+(?:cells|cell lines)\b", re.IGNORECASE),
            ),
            (
                "animal",
                re.compile(r"\b(?:these|the)\s+(?:animals|mice|rats)\b", re.IGNORECASE),
            ),
            (
                "model",
                re.compile(
                    r"\b(?:this|the)\s+(?:model|system|protocol|setup|context)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "person",
                re.compile(
                    r"\b(?:this|the)\s+(?:person|actor|director|scientist)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "organization",
                re.compile(
                    r"\b(?:this|the)\s+(?:organization|company|museum|institute|university)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "location",
                re.compile(
                    r"\b(?:this|the)\s+(?:city|country|state|province|location)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "work",
                re.compile(
                    r"\b(?:this|the)\s+(?:film|movie|book|song|album|series)\b",
                    re.IGNORECASE,
                ),
            ),
        ]

        for category, pattern in replacement_patterns:
            antecedent: str | None = _choose_antecedent(category)
            if antecedent:
                resolved = pattern.sub(antecedent, resolved)

        return resolved

    def _build_section_segments(
        self,
        text: str,
        *,
        section_headers: list[tuple[int, str]] | None = None,
    ) -> list[SectionSegment]:
        """Return ordered section-aligned text segments for upstream relation recovery."""
        headers: list[tuple[int, str]] = list(
            section_headers or self._detect_section_headers(text),
        )
        if not text:
            return []

        boundaries: list[int] = [0]
        for pos, _ in headers:
            if pos not in boundaries and 0 < pos < len(text):
                boundaries.append(pos)
        boundaries.append(len(text))
        boundaries = sorted(set(boundaries))

        segments: list[SectionSegment] = []
        for idx, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            segment_text: str = text[start:end]
            if not segment_text.strip():
                continue
            section_name: str | None = self._get_section_for_position(start, headers)
            segments.append(
                {
                    "index": idx,
                    "name": section_name or "Preamble",
                    "start_pos": start,
                    "end_pos": end,
                    "text": segment_text,
                },
            )
        return segments

    def merge_synonym_entities(
        self,
        graph: QueryGraph,
        similarity_threshold: float = 0.82,
        kg_name: str | None = None,
    ) -> int:
        """
        HippoRAG-style synonym merging: cluster entity nodes that share nearly-identical
        embeddings (surface-form variants like "TBK1" / "TBK1 kinase" / "TBK1 protein")
        and merge them into a single canonical node.

        Uses the existing entity_vector index.  Merges the lower-degree node into the
        higher-degree one so that the richer node survives.

        Returns the number of merges performed.
        """
        try:
            import numpy as np

            # Fetch all entity nodes with their embeddings
            params: dict[str, object] = {"kg_name": kg_name} if kg_name else {}
            kg_filter: str = "AND e.kgName = $kg_name" if kg_name else ""
            fetch_q: str = f"""
            MATCH (e:__Entity__)
            WHERE e.embedding IS NOT NULL {kg_filter}
            RETURN elementId(e) AS eid, coalesce(e.name, e.original_id, e.id) AS name, e.embedding AS emb,
                   COUNT {{ (e)--() }} AS degree,
                   coalesce(e.type, e.ontology_class, '') AS etype
            """
            rows: list[SynonymQueryRow] = cast(
                "list[SynonymQueryRow]",
                graph.query(fetch_q, params),
            )
            if not rows:
                logging.info("No entity embeddings found; skipping synonym merging.")
                return 0

            rows = [r for r in rows if r.get("emb") is not None]
            if not rows:
                logging.info(
                    "No entity embeddings found after filtering; skipping synonym merging.",
                )
                return 0
            eids: list[str] = [r["eid"] for r in rows]
            names: list[Neo4jProperty] = [r["name"] for r in rows]
            degrees: list[int] = [r["degree"] for r in rows]
            etypes: list[Neo4jProperty] = [r.get("etype", "") or "" for r in rows]
            embeddings: NDArray[np.float64] = np.array(
                [r["emb"] for r in rows],
                dtype=float,
            )

            # Generic types that should not block merging (the LLM falls back to these
            # when it cannot classify an entity, so two nodes might genuinely represent
            # the same concept even though one was labelled "Entity" and the other "Concept").
            _generic_types: set[str] = {"entity", "concept", "unknown", "other", ""}

            # Normalise for cosine similarity
            norms: NDArray[np.float64] = np.linalg.norm(
                embeddings,
                axis=1,
                keepdims=True,
            )
            norms[norms == 0] = 1e-9
            normed: NDArray[np.float64] = embeddings / norms

            # Build equivalence clusters via union-find so that synonym chains
            # (A≈B, B≈C) are merged atomically rather than pairwise.
            # Pairwise sequential merges leave dangling references when later pairs
            # reference nodes already merged away.
            sim_matrix: NDArray[np.float64] = normed @ normed.T
            n: int = len(eids)

            parent: list[int] = list(range(n))

            def _find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]  # path compression
                    x = parent[x]
                return x

            def _union(x: int, y: int) -> None:
                px, py = _find(x), _find(y)
                if px != py:
                    parent[px] = py

            for i in range(n):
                for j in range(i + 1, n):
                    if sim_matrix[i, j] >= similarity_threshold:
                        ti, tj = etypes[i].lower(), etypes[j].lower()
                        if (
                            ti not in _generic_types
                            and tj not in _generic_types
                            and ti != tj
                        ):
                            continue
                        if not self._names_pass_synonym_guard(names[i], names[j]):
                            continue
                        _union(i, j)

            # Group indices by cluster root
            from collections import defaultdict as _dd

            clusters: dict[int, list[int]] = _dd(list)
            for i in range(n):
                clusters[_find(i)].append(i)

            # Build one (canonical, [duplicates]) tuple per multi-node cluster.
            # Canonical = highest-degree node in the cluster.
            merge_tasks: list[tuple[int, list[int]]] = []
            for members in clusters.values():
                if len(members) < 2:
                    continue
                canonical_idx: int = max(members, key=lambda i: degrees[i])
                dups: list[int] = [i for i in members if i != canonical_idx]
                merge_tasks.append((canonical_idx, dups))

            if not merge_tasks:
                logging.info(
                    "No synonym clusters found above threshold %.2f",
                    similarity_threshold,
                )
                return 0

            merged: int = 0
            for canonical_idx, dup_indices in merge_tasks:
                canonical_name: object = names[canonical_idx]
                for dup_idx in dup_indices:
                    duplicate_name: object = names[dup_idx]
                    try:
                        # Stamp synonym alias on canonical BEFORE merge (dup ceases to exist after).
                        # Then use apoc.refactor.mergeNodes to atomically rewire all relationships
                        # and delete the duplicate.  {properties: 'discard'} means the first
                        # node in the list (canonical) wins all property conflicts.
                        merge_q: str = """
                        MATCH (can:__Entity__) WHERE elementId(can) = $can_eid
                        MATCH (dup:__Entity__) WHERE elementId(dup) = $dup_eid
                        SET can.synonyms = coalesce(can.synonyms, []) + [dup.id]
                        WITH [can, dup] AS nodes
                        CALL apoc.refactor.mergeNodes(nodes, {properties: 'discard', mergeRels: true})
                        YIELD node
                        RETURN node.id AS merged_id
                        """
                        graph.query(
                            merge_q,
                            {
                                "dup_eid": eids[dup_idx],
                                "can_eid": eids[canonical_idx],
                            },
                        )
                        logging.info(
                            "Merged synonym '%s' → '%s'",
                            duplicate_name,
                            canonical_name,
                        )
                        merged += 1
                    except Exception as merge_err:
                        logging.warning(
                            "Failed to merge '%s' → '%s': %s",
                            duplicate_name,
                            canonical_name,
                            merge_err,
                        )

            logging.info(
                f"Synonym merging complete: {merged}/{len(merge_tasks)} pairs merged",
            )
            return merged

        except Exception as e:
            logging.error(f"Synonym merging failed: {e}")
            return 0

    def _semantic_relationship_match(
        self,
        raw_type: str,
        candidate_ids: list[str],
    ) -> tuple[str, float] | None:
        """Return the best semantic relationship-type match if embeddings exist."""
        embeddings: list[tuple[str, str, list[float]]] = (
            getattr(self, "_ontology_relationship_embeddings", []) or []
        )
        if (
            not embeddings
            or not self.embedding_function
            or not raw_type
            or not candidate_ids
        ):
            return None

        try:
            raw_embedding: list[float] = self.embedding_function.embed_query(raw_type)
        except Exception as exc:
            logging.debug(
                "Relationship embedding lookup failed for '%s': %s",
                raw_type,
                exc,
            )
            return None

        candidate_id_set: set[str] = set(candidate_ids)
        best_id: str | None = None
        best_score: float = 0.0
        for rel_id, _, rel_embedding in embeddings:
            if rel_id not in candidate_id_set:
                continue
            score: float = self._cosine_similarity(raw_embedding, rel_embedding)
            if score > best_score:
                best_id = rel_id
                best_score = score

        if best_id and best_score >= getattr(
            self,
            "relationship_type_similarity_threshold",
            0.62,
        ):
            return best_id, best_score
        return None

    def _generate_entity_id(self, entity: "HarmonizableEntity") -> str:
        """Generate a deterministic UUID scoped to both normalized text AND type.

        Including the type means same-surface/different-type entities (e.g. "depression"
        as Disease vs GeologicalFeature) get distinct UUIDs after the harmonization split.
        For merged entities (LLM type-drift already resolved to one specific type by
        _harmonize_entities) the representative carries a single resolved type, so
        chunk-level variants still converge to the same UUID.
        """
        norm_text: str = self._normalize_entity_text(entity["id"])
        entity_type: str = (entity.get("type") or "").strip()
        properties: GraphProperties = dict[AugmentedPropertyKey, object](
            entity.get("properties") or {},
        )
        scope_key: str = str(
            properties.get("disambiguation_scope")
            or properties.get("title_scope_key")
            or "",
        ).strip()
        unique_seed: str = f"{norm_text}:{entity_type}" if entity_type else norm_text
        if scope_key:
            unique_seed = f"{unique_seed}:{scope_key}"
        return str(uuid.uuid5(uuid.NAMESPACE_OID, unique_seed))

    def _bundle_title_variants(self, raw_title: str) -> list[str]:
        """Return normalized title variants for bundle-style article titles."""
        if not isinstance(raw_title, str):
            return []
        cleaned: str = raw_title.strip()
        if not cleaned:
            return []
        variants: set[str] = {cleaned}
        # Hotpot/2Wiki titles often carry parenthetical disambiguators.
        variants.add(re.sub(r"\s+\([^)]*\)\s*$", "", cleaned).strip())
        normalized: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            norm: str = self._normalize_entity_text(variant)
            if norm and norm not in seen:
                normalized.append(norm)
                seen.add(norm)
        return normalized

    def _infer_passage_title(self, passage_text: str, dataset: str) -> str | None:
        """Infer a source/article title from a benchmark passage string when present."""
        if not isinstance(passage_text, str):
            return None
        text: str = passage_text.strip()
        if not text:
            return None
        if text.lower().startswith("title:"):
            first_line: str = text.splitlines()[0]
            title: str = first_line.split(":", 1)[1].strip()
            return title or None
        if dataset in self._TITLE_AWARE_BUNDLE_DATASETS:
            prefix, sep, _ = text.partition(". ")
            if sep and prefix and len(prefix) <= 160:
                return prefix.strip()
        return None

    def _build_passage_scope_key(
        self,
        *,
        dataset: object,
        question_id: object,
        passage_index: object,
    ) -> str | None:
        dataset_str: str = str(dataset or "").strip()
        question_str: str = str(question_id or "").strip()
        if not dataset_str or not question_str or passage_index is None:
            return None
        return f"{dataset_str}/{question_str}/p{passage_index}"

    def _entity_matches_own_source_title(
        self,
        normalized_text: str,
        entity: "HarmonizableEntity",
    ) -> bool:
        properties: GraphProperties = dict[AugmentedPropertyKey, object](
            entity.get("properties") or {},
        )
        source_title: object = properties.get("source_title")
        if not isinstance(source_title, str) or not source_title.strip():
            return False
        return normalized_text in self._bundle_title_variants(source_title)

    def _names_pass_synonym_guard(self, left_name: str, right_name: str) -> bool:
        """Require some lexical agreement before merging embedding-near entities."""
        left_norm: str = self._normalize_entity_text(left_name or "")
        right_norm: str = self._normalize_entity_text(right_name or "")
        if not left_norm or not right_norm:
            return False
        if left_norm == right_norm:
            return True
        if len(left_norm) >= 4 and left_norm in right_norm:
            return True
        if len(right_norm) >= 4 and right_norm in left_norm:
            return True

        left_tokens: list[str] = [tok for tok in re.split(r"[_\s]+", left_norm) if tok]
        right_tokens: list[str] = [
            tok for tok in re.split(r"[_\s]+", right_norm) if tok
        ]
        left_set: set[str] = set(left_tokens)
        right_set: set[str] = set(right_tokens)
        shared: set[str] = left_set & right_set
        if shared:
            longest_shared: int = max(len(token) for token in shared)
            union: set[str] = left_set | right_set
            if longest_shared >= 4 and (
                left_set.issubset(right_set)
                or right_set.issubset(left_set)
                or (len(union) > 0 and len(shared) / len(union) >= 0.5)
            ):
                return True
            if any(any(ch.isdigit() for ch in token) for token in shared):
                return True

        def _initialism(tokens: list[str]) -> str:
            return "".join(token[0] for token in tokens if token)

        if len(left_tokens) == 1 and left_tokens[0] == _initialism(right_tokens):
            return True
        if len(right_tokens) == 1 and right_tokens[0] == _initialism(left_tokens):
            return True

        return difflib.SequenceMatcher(None, left_norm, right_norm).ratio() >= 0.92

    def _entity_name_variants(self, name: str) -> list[str]:
        """Generate robust surface-form variants for entity resolution."""
        if not isinstance(name, str):
            return []
        base: str = name.strip()
        if not base:
            return []

        variants: set[str] = {
            base,
            base.lower(),
            base.replace("_", " "),
            base.replace(" ", "_"),
            self._normalize_entity_text(base),
        }
        return [v for v in variants if isinstance(v, str) and v.strip()]

    def _entity_candidate_names(self, entity: "HarmonizableEntity") -> list[str]:
        """Collect plausible surface forms for an extracted entity."""
        if not isinstance(entity, dict):
            return []

        candidates: list[str] = []
        for candidate in (
            entity.get("id"),
            entity.get("name"),
            (entity.get("properties") or {}).get("name"),
        ):
            if isinstance(candidate, str) and candidate.strip():
                candidates.append(candidate)

        props: RawGraphProperties | dict[str, object] = entity.get("properties") or {}
        all_names: object = props.get("all_names") or []
        if isinstance(all_names, list):
            for alias in all_names:
                if isinstance(alias, str) and alias.strip():
                    candidates.append(alias)

        # Include LLM-extracted aliases (alternate surface forms / abbreviations)
        llm_aliases: object = props.get("aliases") or []
        if isinstance(llm_aliases, list):
            for alias in llm_aliases:
                if isinstance(alias, str) and alias.strip():
                    candidates.append(alias)

        seen: set[str] = set()
        deduped: list[str] = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                deduped.append(candidate)
        return deduped

    def _entity_appears_in_text(self, entity: "HarmonizableEntity", text: str) -> bool:
        """Check whether an entity or any of its aliases appears in text."""
        if not isinstance(text, str) or not text.strip():
            return False

        text_lower: str = text.lower()
        seen_variants: set[str] = set()
        for candidate in self._entity_candidate_names(entity):
            for variant in self._entity_name_variants(candidate):
                normalized: str = variant.strip().lower()
                if len(normalized) < 3 or normalized in seen_variants:
                    continue
                seen_variants.add(normalized)
                prefix: str = (
                    r"(?<!\w)"
                    if not normalized[:1].isalnum() and normalized[:1] != "_"
                    else r"\b"
                )
                suffix: str = (
                    r"(?!\w)"
                    if not normalized[-1:].isalnum() and normalized[-1:] != "_"
                    else r"\b"
                )
                pattern: re.Pattern[str] = re.compile(
                    prefix + re.escape(normalized) + suffix,
                )
                if pattern.search(text_lower):
                    return True
        return False

    def _resolve_relationship_endpoint(
        self,
        raw_name: str,
        entities: "Sequence[ExtractionEntity]",
    ) -> str | None:
        """
        Resolve a relationship endpoint to the best matching extracted entity ID.

        The LLM often uses short forms in relationship triples while the entity
        list carries a fuller mention, so we allow safe fuzzy matching here.
        """
        if not isinstance(raw_name, str) or not raw_name.strip():
            return None

        raw_name = raw_name.strip()
        raw_variants: list[str] = self._entity_name_variants(raw_name)
        raw_norm: str = self._normalize_entity_text(raw_name)

        direct_matches: list[str] = []
        best_entity_id: str | None = None
        best_score: float = 0.0
        second_best: float = 0.0

        for entity in entities:
            entity_id: str | None = entity.get("id")
            if not isinstance(entity_id, str) or not entity_id.strip():
                continue

            candidate_variants: set[str] = set()
            for candidate in self._entity_candidate_names(entity):
                candidate_variants.update(self._entity_name_variants(candidate))

            if any(
                rv.lower() == cv.lower()
                for rv in raw_variants
                for cv in candidate_variants
            ):
                direct_matches.append(entity_id)
                continue

            entity_best: float = 0.0
            for candidate in candidate_variants:
                candidate_norm: str = self._normalize_entity_text(candidate)
                if not candidate_norm or not raw_norm:
                    continue
                if candidate_norm == raw_norm:
                    entity_best = 1.0
                    break

                score: float = 0.0
                if raw_norm in candidate_norm or candidate_norm in raw_norm:
                    if min(len(raw_norm), len(candidate_norm)) >= 4:
                        score = 0.94
                else:
                    score = difflib.SequenceMatcher(
                        None,
                        raw_norm,
                        candidate_norm,
                    ).ratio()
                    raw_tokens: set[str] = set(raw_norm.split("_"))
                    candidate_tokens: set[str] = set(candidate_norm.split("_"))
                    if raw_tokens and candidate_tokens:
                        overlap: float = len(raw_tokens & candidate_tokens) / max(
                            1,
                            min(len(raw_tokens), len(candidate_tokens)),
                        )
                        score = max(score, overlap * 0.92)

                entity_best = max(entity_best, score)

            if entity_best > best_score:
                second_best = best_score
                best_score = entity_best
                best_entity_id = entity_id
            elif entity_best > second_best:
                second_best = entity_best

        unique_direct: list[str] = list(dict.fromkeys(direct_matches))
        if len(unique_direct) == 1:
            return unique_direct[0]
        if len(unique_direct) > 1:
            for entity_id in unique_direct:
                if entity_id.lower() == raw_name.lower():
                    return entity_id
            return None

        if best_entity_id and best_score >= 0.90 and (best_score - second_best) >= 0.03:
            return best_entity_id
        return None

    @staticmethod
    def _relationship_evidence_scope(
        verification_chunks: Sequence[GraphChunkInput],
        triple_confidence: float,
    ) -> str:
        """Categorize how directly a relation is text-supported.

        This intentionally does not collapse weak evidence into cross-section.
        In the current confidence scheme:
        - 1.0 means same sentence
        - 0.7 means same chunk
        - 0.4 means both entities found across multiple chunks
        - 0.3 means only one side grounded
        - 0.1 means unsupported
        """
        if not verification_chunks:
            return "unknown"
        if triple_confidence >= 0.95:
            return "sentence"
        if triple_confidence >= 0.65:
            return "chunk"
        if triple_confidence >= 0.35:
            section_names: set[str] = {
                str(chunk.get("section_name")).strip()
                for chunk in (verification_chunks or [])
                if str(chunk.get("section_name") or "").strip()
            }
            if len(section_names) > 1:
                return "cross_section"
            return "cross_chunk"
        if triple_confidence >= 0.25:
            return "partial_grounding"
        return "unsupported"

    def _ensure_umls_linker(self) -> "Language | None":
        """Lazily initialize an optional SciSpaCy UMLS linker."""
        if not getattr(self, "enable_umls_linking", False):
            return None
        if getattr(self, "_umls_linker_state", "disabled") == "ready":
            return self._umls_nlp
        if getattr(self, "_umls_linker_state", "disabled") == "unavailable":
            return None

        try:
            spacy: ModuleType = importlib.import_module("spacy")
            importlib.import_module("scispacy")
            nlp: Language = spacy.load(self.umls_spacy_model)
            if "scispacy_linker" not in getattr(nlp, "pipe_names", []):
                nlp.add_pipe(
                    "scispacy_linker",
                    config={
                        "resolve_abbreviations": True,
                        "linker_name": "umls",
                    },
                )
            self._umls_nlp = nlp
            self._umls_linker_state = "ready"
            logging.info(
                "UMLS linker enabled using spaCy model '%s'",
                self.umls_spacy_model,
            )
            return nlp
        except Exception as exc:
            self._umls_linker_state = "unavailable"
            self._umls_nlp = None
            logging.warning("UMLS linker unavailable, continuing without it: %s", exc)
            return None

    def _link_entity_to_umls(
        self,
        entity_name: str,
    ) -> UMLSLink | None:
        """Return top UMLS link metadata for an entity name when available."""
        nlp: Language | None = self._ensure_umls_linker()
        if nlp is None or not entity_name.strip():
            return None
        try:
            doc: Doc = nlp(entity_name)
        except Exception as exc:
            logging.debug("UMLS linking failed for '%s': %s", entity_name, exc)
            return None

        kb_ents: list[object] = []
        for ent in getattr(doc, "ents", []):
            kb_ents.extend(list(getattr(getattr(ent, "_", None), "kb_ents", []) or []))
        if not kb_ents:
            kb_ents = list(getattr(getattr(doc, "_", None), "kb_ents", []) or [])
        if not kb_ents:
            return None

        cui, score = kb_ents[0]
        if score < 0.75:
            return None

        kb_name: str | None = None
        try:
            linker: PipeCallable = nlp.get_pipe("scispacy_linker")
            kb_entry: object = linker.kb.cui_to_entity.get(cui)
            if kb_entry:
                kb_name = kb_entry[0]
        except Exception:
            kb_name = None

        return {
            "cui": cui,
            "score": float(score),
            "name": kb_name,
        }

    def _apply_optional_umls_linking(
        self,
        harmonized_entities: "list[NodeT]",
        entity_map: "dict[EntityLookupKey, NodeT]",
    ) -> "tuple[list[NodeT], dict[EntityLookupKey, NodeT]]":
        """Attach optional UMLS metadata and merge entities sharing a confident CUI."""
        if not getattr(self, "enable_umls_linking", False):
            return harmonized_entities, entity_map
        if self._ensure_umls_linker() is None:
            return harmonized_entities, entity_map

        generic_types: set[str] = {"Concept", "Entity", "Unknown", "Other"}
        entities_with_links: list[NodeT] = []
        for entity in harmonized_entities:
            entity_copy: NodeT = cast(NodeT, dict(entity))
            properties: GraphProperties = dict[AugmentedPropertyKey, object](
                entity_copy.get("properties") or {},
            )
            link: UMLSLink | None = self._link_entity_to_umls(
                str(entity_copy.get("id") or ""),
            )
            if link:
                properties["umls_cui"] = link["cui"]
                properties["umls_score"] = round(link["score"], 4)
                if link.get("name"):
                    properties["umls_name"] = link["name"]
            entity_copy["properties"] = properties
            entities_with_links.append(entity_copy)

        grouped_by_cui: dict[str, list[NodeT]] = defaultdict(list)
        ungrouped: list[NodeT] = []
        for entity in entities_with_links:
            cui: object = (entity.get("properties") or {}).get("umls_cui")
            if cui:
                grouped_by_cui[str(cui)].append(entity)
            else:
                ungrouped.append(entity)

        merged_entities: list[NodeT] = []
        remapped_entities_by_uuid: dict[str | None, NodeT] = {}

        for cui, cui_group in grouped_by_cui.items():
            specific_types: set[str | None] = {
                entity.get("type")
                for entity in cui_group
                if entity.get("type") not in generic_types
            }
            if len(cui_group) == 1 or len(specific_types) > 1:
                for entity in cui_group:
                    merged_entities.append(entity)
                    if entity.get("uuid"):
                        remapped_entities_by_uuid[entity["uuid"]] = entity
                continue

            representative: NodeT = max(
                cui_group,
                key=lambda e: (
                    0 if e.get("type") in generic_types else 1,
                    len(str((e.get("properties") or {}).get("description") or "")),
                    len(str(e.get("id") or "")),
                ),
            ).copy()
            properties = dict(representative.get("properties") or {})
            all_names: set[object] = set(
                properties.get("all_names") or [representative.get("id")],
            )
            all_descriptions: set[object] = set(
                properties.get("all_descriptions") or [],
            )
            original_ids: set[object] = set(
                properties.get("original_ids") or [representative.get("id")],
            )
            merged_anchor_spans: list[AnchorSpan] = self._merge_anchor_spans(
                properties.get("anchor_spans"),
            )

            for entity in cui_group:
                entity_props: GraphProperties = dict[AugmentedPropertyKey, object](
                    entity.get("properties") or {},
                )
                all_names.update(entity_props.get("all_names") or [entity.get("id")])
                all_descriptions.update(entity_props.get("all_descriptions") or [])
                description: object = entity_props.get("description")
                if description:
                    all_descriptions.add(description)
                original_ids.update(
                    entity_props.get("original_ids") or [entity.get("id")],
                )
                merged_anchor_spans = self._merge_anchor_spans(
                    merged_anchor_spans,
                    entity_props.get("anchor_spans"),
                )
                if entity.get("embedding") and representative.get("embedding") is None:
                    cast(EmbeddingMutation, representative)["embedding"] = entity.get(
                        "embedding",
                    )
                if entity.get("uuid"):
                    remapped_entities_by_uuid[entity["uuid"]] = representative

            properties["all_names"] = sorted(
                {
                    name
                    for name in all_names
                    if isinstance(name, str) and name and str(name).strip()
                },
            )
            properties["all_descriptions"] = sorted(
                {
                    desc
                    for desc in all_descriptions
                    if isinstance(desc, str) and desc.strip()
                },
                key=len,
                reverse=True,
            )
            properties["original_ids"] = sorted(
                {
                    value
                    for value in original_ids
                    if isinstance(value, str) and value and str(value).strip()
                },
            )
            if merged_anchor_spans:
                properties["anchor_spans"] = merged_anchor_spans
                properties["anchor_mention_count"] = len(merged_anchor_spans)
            representative["properties"] = properties
            merged_entities.append(representative)
            logging.info(
                "Merged %d entities through shared UMLS CUI %s",
                len(cui_group),
                cui,
            )

        merged_entities.extend(ungrouped)
        updated_entity_map: dict[EntityLookupKey, NodeT] = {}
        for variant_name, representative in entity_map.items():
            uuid_value: str | None = (
                representative.get("uuid") if isinstance(representative, dict) else None
            )
            updated_entity_map[variant_name] = remapped_entities_by_uuid.get(
                uuid_value,
                representative,
            )

        return merged_entities, updated_entity_map

    def _soft_entity_alias_keys(self, name: str) -> list[str]:
        """Return conservative soft-canonical keys for a surface form."""
        if not isinstance(name, str):
            return []
        cleaned: str = name.strip()
        if not cleaned:
            return []

        keys: set[str] = {self._normalize_entity_text(cleaned)}
        stripped_parenthetical: str = re.sub(r"\s+\([^)]*\)\s*$", "", cleaned).strip()
        if stripped_parenthetical and stripped_parenthetical != cleaned:
            keys.add(self._normalize_entity_text(stripped_parenthetical))

        tokens: list[str] = [
            tok
            for tok in re.split(r"[_\s]+", self._normalize_entity_text(cleaned))
            if tok
        ]
        if len(tokens) >= 2:
            keys.add("".join(token[0] for token in tokens if token))

        removable_suffixes: set[str] = {
            "protein",
            "proteins",
            "gene",
            "genes",
            "kinase",
            "kinases",
            "receptor",
            "receptors",
            "complex",
            "complexes",
            "subunit",
            "subunits",
            "pathway",
            "pathways",
            "signaling",
            "signalling",
            "isoform",
            "isoforms",
        }
        stripped_tokens: list[str] = list(tokens)
        while len(stripped_tokens) > 1 and stripped_tokens[-1] in removable_suffixes:
            stripped_tokens = stripped_tokens[:-1]
            keys.add("_".join(stripped_tokens))
            if len(stripped_tokens) >= 2:
                keys.add("".join(token[0] for token in stripped_tokens if token))

        return [key for key in keys if isinstance(key, str) and key.strip()]

    @staticmethod
    def _cosine_similarity(
        left: Sequence[float | str] | DecodedJSONValue,
        right: Sequence[float | str] | DecodedJSONValue,
    ) -> float:
        """Lightweight cosine similarity without a hard NumPy dependency."""
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return 0.0
        if not left or not right or len(left) != len(right):
            return 0.0
        dot: float = 0.0
        left_norm: float = 0.0
        right_norm: float = 0.0
        for l_val, r_val in zip(left, right):
            try:
                l_float: float = float(l_val)
                r_float: float = float(r_val)
            except (TypeError, ValueError):
                return 0.0
            dot += l_float * r_float
            left_norm += l_float * l_float
            right_norm += r_float * r_float
        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0
        return dot / math.sqrt(left_norm * right_norm)

    @staticmethod
    def _confidence_score(value: object) -> float:
        """Normalize numeric or verbal confidence labels onto a sortable [0, 1] scale."""
        if isinstance(value, (int, float)):
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                return 0.0

        text: str = str(value or "").strip().lower()
        if not text or text in {"none", "null", "nan", "unknown", "n/a"}:
            return 0.0

        try:
            return max(0.0, min(1.0, float(text)))
        except (TypeError, ValueError):
            pass

        verbal_scale: dict[str, float] = {
            "demonstrated": 0.95,
            "supported": 0.85,
            "high": 0.8,
            "strong": 0.8,
            "suggested": 0.6,
            "moderate": 0.55,
            "hypothesized": 0.35,
            "hypothesis": 0.35,
            "speculative": 0.25,
            "weak": 0.2,
            "low": 0.2,
        }
        return verbal_scale.get(text, 0.0)

    def _merge_entity_cluster(self, entities: "list[NodeT]") -> "NodeT":
        """Merge a soft-linked entity cluster into one representative entity."""
        representative: NodeT = max(
            entities,
            key=lambda entity: (
                self._entity_type_specificity(entity.get("type")),
                len(str((entity.get("properties") or {}).get("description") or "")),
                len(str(entity.get("id") or "")),
            ),
        ).copy()

        properties: GraphProperties = dict[AugmentedPropertyKey, object](
            representative.get("properties") or {},
        )
        merged_anchor_spans: list[AnchorSpan] = self._merge_anchor_spans(
            *[
                (entity.get("properties") or {}).get("anchor_spans")
                for entity in entities
            ],
        )

        all_names: set[str] = set()
        all_descriptions: set[str] = set()
        aliases: set[str] = set()
        source_scope_keys: set[str] = set()
        source_titles: set[str] = set()
        question_ids: set[str] = set()
        passage_indexes: set[int] = set()
        umls_cuis: set[str] = set()
        umls_names: set[str] = set()
        best_embedding: list[float] | DecodedJSONValue = representative.get("embedding")

        for entity in entities:
            entity_properties: GraphProperties = dict[AugmentedPropertyKey, object](
                entity.get("properties") or {},
            )
            if not best_embedding and entity.get("embedding") is not None:
                best_embedding = entity.get("embedding")

            for name in self._entity_candidate_names(entity):
                if isinstance(name, str) and name.strip():
                    all_names.add(name.strip())
            description: str = str(entity_properties.get("description") or "").strip()
            if description:
                all_descriptions.add(description)
            for alias in entity_properties.get("aliases") or []:
                if isinstance(alias, str) and alias.strip():
                    aliases.add(alias.strip())
            for key in entity_properties.get("source_scope_keys") or []:
                if isinstance(key, str) and key.strip():
                    source_scope_keys.add(key.strip())
            source_scope_key: str = str(
                entity_properties.get("source_scope_key") or "",
            ).strip()
            if source_scope_key:
                source_scope_keys.add(source_scope_key)
            for title in entity_properties.get("source_titles") or []:
                if isinstance(title, str) and title.strip():
                    source_titles.add(title.strip())
            source_title: str = str(entity_properties.get("source_title") or "").strip()
            if source_title:
                source_titles.add(source_title)
            for question_id in entity_properties.get("question_ids") or []:
                if isinstance(question_id, str) and question_id.strip():
                    question_ids.add(question_id.strip())
            single_question_id: str = str(
                entity_properties.get("question_id") or "",
            ).strip()
            if single_question_id:
                question_ids.add(single_question_id)
            for passage_index in entity_properties.get("passage_indexes") or []:
                if isinstance(passage_index, int):
                    passage_indexes.add(int(passage_index))
            if isinstance(entity_properties.get("passage_index"), int):
                passage_indexes.add(int(entity_properties["passage_index"]))
            cui: str = str(entity_properties.get("umls_cui") or "").strip()
            if cui:
                umls_cuis.add(cui)
            umls_name: str = str(entity_properties.get("umls_name") or "").strip()
            if umls_name:
                umls_names.add(umls_name)

            for key, value in entity_properties.items():
                if key in {
                    "description",
                    "all_names",
                    "aliases",
                    "anchor_spans",
                    "source_scope_keys",
                    "source_titles",
                    "question_ids",
                    "passage_indexes",
                    "umls_cui",
                    "umls_name",
                }:
                    continue
                properties.setdefault(key, value)

        if all_names:
            properties["all_names"] = sorted(all_names)
            properties["original_ids"] = sorted(all_names)
        if all_descriptions:
            properties["all_descriptions"] = sorted(
                all_descriptions,
                key=len,
                reverse=True,
            )
            properties.setdefault("description", max(all_descriptions, key=len))
        if aliases:
            properties["aliases"] = sorted(aliases)
        if merged_anchor_spans:
            properties["anchor_spans"] = merged_anchor_spans
            properties["anchor_mention_count"] = len(merged_anchor_spans)
        if source_scope_keys:
            properties["source_scope_keys"] = sorted(source_scope_keys)
        if source_titles:
            properties["source_titles"] = sorted(source_titles)
        if question_ids:
            properties["question_ids"] = sorted(question_ids)
        if passage_indexes:
            properties["passage_indexes"] = sorted(passage_indexes)
        if len(umls_cuis) == 1:
            properties["umls_cui"] = next(iter(umls_cuis))
        if len(umls_names) == 1:
            properties["umls_name"] = next(iter(umls_names))
        properties["soft_linked"] = True
        properties["soft_link_cluster_size"] = len(entities)

        representative["properties"] = properties
        cast(EmbeddingMutation, representative)["embedding"] = best_embedding
        representative.setdefault("uuid", entities[0].get("uuid"))
        return representative

    def _apply_soft_entity_linking(
        self,
        harmonized_entities: "list[NodeT]",
        entity_map: "dict[EntityLookupKey, NodeT]",
    ) -> "tuple[list[NodeT], dict[EntityLookupKey, NodeT]]":
        """Apply conservative soft canonicalisation before UUID endpoints freeze."""
        if not getattr(self, "enable_soft_entity_linking", False):
            return harmonized_entities, entity_map
        if len(harmonized_entities) < 2:
            return harmonized_entities, entity_map

        generic_types: set[str] = {"concept", "entity", "unknown", "other", ""}
        candidate_names: dict[str | None, list[str]] = {
            entity.get("uuid"): self._entity_candidate_names(entity)
            for entity in harmonized_entities
        }
        alias_keys: dict[str | None, set[str]] = {
            entity.get("uuid"): {
                key
                for name in candidate_names.get(entity.get("uuid"), [])
                for key in self._soft_entity_alias_keys(name)
            }
            for entity in harmonized_entities
        }
        embeddings: dict[str | None, list[float] | DecodedJSONValue] = {
            entity.get("uuid"): entity.get("embedding")
            for entity in harmonized_entities
        }
        missing_embeddings: list[NodeT] = [
            entity
            for entity in harmonized_entities
            if entity.get("uuid") and embeddings.get(entity.get("uuid")) is None
        ]
        if missing_embeddings and self.embedding_function:
            try:
                generated: list[list[float]] = self.embedding_function.embed_documents(
                    [str(entity.get("id") or "") for entity in missing_embeddings],
                )
                for entity, embedding in zip(missing_embeddings, generated):
                    entity_uuid: str | None = entity.get("uuid")
                    if entity_uuid:
                        embeddings[entity_uuid] = embedding
                        entity["embedding"] = embedding
            except Exception as exc:
                logging.debug("Soft entity linking embeddings unavailable: %s", exc)

        entities: list[NodeT] = [
            entity for entity in harmonized_entities if entity.get("uuid")
        ]
        parent: dict[str | None, str | None] = {
            entity["uuid"]: entity["uuid"] for entity in entities
        }

        def _find(item: str | None) -> str | None:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        def _union(left_uuid: str | None, right_uuid: str | None) -> None:
            left_root: str | None = _find(left_uuid)
            right_root: str | None = _find(right_uuid)
            if left_root != right_root:
                parent[left_root] = right_root

        for idx, left_entity in enumerate(entities):
            left_uuid: str | None = left_entity["uuid"]
            left_type: str = str(left_entity.get("type") or "").strip().lower()
            for right_entity in entities[idx + 1 :]:
                right_uuid: str | None = right_entity["uuid"]
                right_type: str = str(right_entity.get("type") or "").strip().lower()

                alias_overlap: set[str] = alias_keys.get(
                    left_uuid,
                    set(),
                ) & alias_keys.get(right_uuid, set())
                lexical_guard: bool = bool(alias_overlap)
                if not lexical_guard:
                    lexical_guard = any(
                        self._names_pass_synonym_guard(left_name, right_name)
                        for left_name in candidate_names.get(left_uuid, [])
                        for right_name in candidate_names.get(right_uuid, [])
                    )

                if not lexical_guard:
                    similarity: float = self._cosine_similarity(
                        embeddings.get(left_uuid),
                        embeddings.get(right_uuid),
                    )
                    if similarity < getattr(
                        self,
                        "soft_entity_similarity_threshold",
                        0.88,
                    ):
                        continue
                elif alias_overlap:
                    similarity = 1.0
                else:
                    similarity = self._cosine_similarity(
                        embeddings.get(left_uuid),
                        embeddings.get(right_uuid),
                    )

                if (
                    left_type not in generic_types
                    and right_type not in generic_types
                    and left_type != right_type
                    and not alias_overlap
                ):
                    continue

                if (
                    not alias_overlap
                    and similarity > 0.0
                    and similarity
                    < getattr(self, "soft_entity_similarity_threshold", 0.88)
                ):
                    continue

                _union(left_uuid, right_uuid)

        grouped: dict[str | None, list[NodeT]] = defaultdict(list)
        for entity in entities:
            grouped[_find(entity["uuid"])].append(entity)

        if not any(len(group) > 1 for group in grouped.values()):
            return harmonized_entities, entity_map

        merged_entities: list[NodeT] = []
        remapped_by_uuid: dict[str | None, NodeT] = {}
        merge_count: int = 0
        for group in grouped.values():
            if len(group) == 1:
                merged_entities.append(group[0])
                remapped_by_uuid[group[0]["uuid"]] = group[0]
                continue
            representative: NodeT = self._merge_entity_cluster(group)
            merged_entities.append(representative)
            for entity in group:
                remapped_by_uuid[entity["uuid"]] = representative
            merge_count += len(group) - 1

        updated_entity_map: dict[str | tuple[str, str], NodeT] = {
            key: remapped_by_uuid.get(value.get("uuid"), value)
            for key, value in entity_map.items()
        }
        for entity in merged_entities:
            for candidate in self._entity_candidate_names(entity):
                updated_entity_map.setdefault(candidate, entity)
                for variant in self._entity_name_variants(candidate):
                    updated_entity_map.setdefault(variant, entity)
                for scope_key in (entity.get("properties") or {}).get(
                    "source_scope_keys",
                ) or []:
                    if isinstance(scope_key, str) and scope_key.strip():
                        updated_entity_map.setdefault((scope_key, candidate), entity)

        logging.info(
            "Soft entity linking merged %d entities into %d representatives",
            merge_count,
            len(merged_entities),
        )
        return merged_entities, updated_entity_map

    @overload
    def _harmonize_entities(
        self,
        all_entities: Sequence[HarmonizableEntity],
        return_id_map: Literal[False] = False,
    ) -> list[NormalizedNode]: ...

    @overload
    def _harmonize_entities(
        self,
        all_entities: Sequence[HarmonizableEntity],
        return_id_map: Literal[True],
    ) -> tuple[list[NormalizedNode], NormalizedEntityLookup]: ...

    def _harmonize_entities(
        self,
        all_entities: Sequence[HarmonizableEntity],
        return_id_map: bool = False,
    ) -> list[NormalizedNode] | tuple[list[NormalizedNode], NormalizedEntityLookup]:
        """
        Harmonize entities across chunks to avoid duplicates using improved normalization
        """
        logging.info(f"Starting harmonization of {len(all_entities)} raw entities")

        # Step 0: Drop generic hub-entity names before any deduplication or
        # grouping.  This prevents high-frequency vague terms from becoming
        # hub nodes that fan out to hundreds of irrelevant chunks at retrieval
        # time.  The filter is conservative: it only removes names that are
        # explicitly blocklisted or shorter than the minimum length threshold.
        filtered_entities: list[HarmonizableEntity] = [
            e
            for e in all_entities
            if _is_valid_entity_name(e.get("id", "") or e.get("name", ""))
        ]
        if len(filtered_entities) < len(all_entities):
            logging.info(
                "Hub-entity filter removed %d/%d entities (generic names or too short)",
                len(all_entities) - len(filtered_entities),
                len(all_entities),
            )
        all_entities = filtered_entities

        # Step 1: Build grouping by normalized text, then refine to avoid cross-type collapse.
        #
        # Why text-first, not (text, type):
        #   Using (text, type) directly splits "Prostate Cancer" typed as Disease in chunk 1
        #   and Concept in chunk 2 into two nodes — that is LLM drift, not a real distinction.
        #   Grouping by text first and picking the most specific type handles this correctly.
        #
        # Why we then split by specific type:
        #   Pure text grouping collapses genuinely different entities that happen to share a
        #   surface form — e.g. "depression" (Disease) vs "depression" (GeologicalFeature).
        #   If a text group contains more than one *distinct specific type*, we split it into
        #   per-specific-type buckets; generic-typed occurrences (Concept/Entity/Unknown/Other)
        #   are assigned to the dominant (largest) specific-type bucket so that LLM drift
        #   toward generic labels still merges into the right node rather than floating free.
        _generic_types: set[str] = {"Concept", "Entity", "Unknown", "Other"}

        text_groups: dict[str, list[HarmonizableEntity]] = defaultdict(list)
        for entity in all_entities:
            normalized_text: str = self._normalize_entity_text(entity["id"])
            text_groups[normalized_text].append(entity)

        # Refine:
        #   1. for bundle-style multihop datasets, preserve article-local title
        #      entities (same surface form, different passage/article scope)
        #   2. split remaining groups that contain multiple distinct specific types
        # dominant_for_surface records which specific type is the largest bucket for
        # each surface form that was split — used below to make entity_map writes
        # deterministic (dominant type always wins, regardless of iteration order).
        entity_groups: dict[EntityGroupKey, list[HarmonizableEntity]] = defaultdict(
            list,
        )
        dominant_for_surface: dict[str, str] = {}  # norm_text → dominant specific type

        for norm_text, entities in text_groups.items():
            pending_groups: list[
                tuple[str | tuple[str, str, str], list[HarmonizableEntity]]
            ] = [(norm_text, list(entities))]
            title_matched_by_scope: dict[str, list[HarmonizableEntity]] = defaultdict(
                list,
            )
            residual_entities: list[HarmonizableEntity] = []

            for entity in entities:
                properties: GraphProperties = dict[AugmentedPropertyKey, object](
                    entity.get("properties") or {},
                )
                dataset: str = str(properties.get("dataset") or "").strip().lower()
                scope_key: str = str(properties.get("source_scope_key") or "").strip()
                if (
                    dataset in self._TITLE_AWARE_BUNDLE_DATASETS
                    and scope_key
                    and self._entity_matches_own_source_title(norm_text, entity)
                ):
                    title_matched_by_scope[scope_key].append(entity)
                else:
                    residual_entities.append(entity)

            if len(title_matched_by_scope) > 1:
                pending_groups = [
                    ((norm_text, "__title_scope__", scope_key), scoped_entities)
                    for scope_key, scoped_entities in sorted(
                        title_matched_by_scope.items(),
                    )
                ]
                if residual_entities:
                    pending_groups.append((norm_text, residual_entities))
                logging.info(
                    "Surface form '%s' split by title/article scope across %d passage scopes",
                    norm_text,
                    len(title_matched_by_scope),
                )

            for group_key, group_entities in pending_groups:
                specific_types: set[str | None] = {
                    e.get("type")
                    for e in group_entities
                    if e.get("type") not in _generic_types
                }
                if len(specific_types) <= 1:
                    entity_groups[group_key].extend(group_entities)
                    continue

                # Multiple distinct specific types → split, generics go to largest bucket.
                type_buckets: dict[str, list[HarmonizableEntity]] = defaultdict(list)
                generics: list[HarmonizableEntity] = []
                for e in group_entities:
                    if e.get("type") in _generic_types:
                        generics.append(e)
                    else:
                        type_buckets[e["type"]].append(e)
                dominant: str = max(type_buckets, key=lambda t: len(type_buckets[t]))
                if generics:
                    type_buckets[dominant].extend(generics)
                dominant_for_surface[str(group_key)] = dominant
                logging.info(
                    "Surface form '%s' has %d distinct specific types %s; "
                    "relationships will resolve to dominant type '%s' (%d occurrences). "
                    "Add source/target type fields to relationship dicts for per-type resolution.",
                    norm_text,
                    len(type_buckets),
                    sorted(type_buckets),
                    dominant,
                    len(type_buckets[dominant]),
                )
                for stype, bucket in type_buckets.items():
                    entity_groups[(group_key, stype)].extend(bucket)

        # Step 2: Log entity distribution before harmonization
        type_distribution: Counter[str] = Counter()
        for entities in entity_groups.values():
            for entity in entities:
                type_distribution[entity["type"]] += 1

        logging.info(
            f"Entity type distribution before harmonization: {dict(type_distribution)}",
        )

        # Step 3: Create harmonized entities
        harmonized_entities: list[NormalizedNode] = []
        entity_map: NormalizedEntityLookup = {}  # For mapping original IDs to harmonized versions
        total_duplicates_removed: int = 0

        for normalized_key, entities in entity_groups.items():
            if not entities:
                continue

            dominant_lookup_key: str | tuple[str, str, str] = (
                normalized_key[0]
                if isinstance(normalized_key, tuple) and len(normalized_key) == 2
                else normalized_key
            )
            title_scope_key: str | None = None
            if (
                isinstance(normalized_key, tuple)
                and len(normalized_key) >= 3
                and normalized_key[1] == "__title_scope__"
            ):
                title_scope_key = str(normalized_key[2])
            elif (
                isinstance(normalized_key, tuple)
                and len(normalized_key) == 2
                and isinstance(normalized_key[0], tuple)
                and len(normalized_key[0]) >= 3
                and normalized_key[0][1] == "__title_scope__"
            ):
                title_scope_key = str(normalized_key[0][2])

            # --- Deterministic deduplication rules ---
            # Rule 1: prefer specific ontology types over generic fallbacks (Concept, Entity, Unknown)
            # Rule 2: among equally specific types, prefer the longest description
            # Rule 3: if still tied, prefer the longest entity name (most fully-qualified)
            # (_generic_types defined at Step 1 above)

            def _type_specificity(e: HarmonizableEntity) -> int:
                return 0 if e.get("type", "Concept") in _generic_types else 1

            def _desc_len(e: HarmonizableEntity) -> int:
                return len(e.get("properties", {}).get("description") or "")

            def _name_len(e: HarmonizableEntity) -> int:
                return len(e.get("id") or "")

            representative_entity: HarmonizationCandidate = cast(
                HarmonizationCandidate,
                max(
                    entities,
                    key=lambda e: (_type_specificity(e), _desc_len(e), _name_len(e)),
                ).copy(),
            )

            mapped_entities: list[HarmonizationCandidate] = cast(
                "list[HarmonizationCandidate]",
                entities,
            )

            # Merge information from all occurrences
            all_names: set[str] = set()
            all_descriptions: set[str] = set()
            all_anchor_spans: list[AnchorSpan] = []
            all_source_scope_keys: set[str] = set()
            all_source_titles: set[str] = set()
            all_question_ids: set[str] = set()
            all_passage_indexes: set[int] = set()

            for mapped_entity in mapped_entities:
                # Rule 3: accumulate all surface forms as synonyms
                all_names.add(mapped_entity["id"])
                desc: object = mapped_entity.get("properties", {}).get("description")
                if desc:
                    all_descriptions.add(desc if isinstance(desc, str) else str(desc))
                all_anchor_spans = self._merge_anchor_spans(
                    all_anchor_spans,
                    (mapped_entity.get("properties") or {}).get("anchor_spans"),
                )
                props: GraphProperties = dict[AugmentedPropertyKey, object](
                    mapped_entity.get("properties") or {},
                )
                property_scope_key: object = props.get("source_scope_key")
                if property_scope_key:
                    all_source_scope_keys.add(str(property_scope_key))
                source_title: object = props.get("source_title")
                if source_title:
                    all_source_titles.add(str(source_title))
                question_id: object = props.get("question_id")
                if question_id is not None:
                    all_question_ids.add(str(question_id))
                passage_index: object = props.get("passage_index")
                if passage_index is not None:
                    all_passage_indexes.add(int(passage_index))

                # Keep the best embedding (prefer any available embedding)
                if not representative_entity.get("embedding") and mapped_entity.get(
                    "embedding",
                ):
                    representative_entity["embedding"] = mapped_entity["embedding"]

            # Merge all non-description properties from variants into representative
            for mapped_entity in mapped_entities:
                if mapped_entity.get("properties"):
                    for k, v in mapped_entity["properties"].items():
                        if k not in (
                            "description",
                        ):  # description already resolved above
                            representative_entity.setdefault(
                                "properties",
                                {},
                            ).setdefault(k, v)

            # Update representative mapped_entity with merged information
            representative_properties: GraphProperties = (
                representative_entity.setdefault(
                    "properties",
                    {},
                )
            )
            representative_properties.setdefault(
                "original_ids",
                [representative_entity["id"]],
            )
            if all_anchor_spans:
                representative_properties["anchor_spans"] = all_anchor_spans
                representative_properties["anchor_mention_count"] = len(
                    all_anchor_spans,
                )
            if all_source_scope_keys:
                representative_properties["source_scope_keys"] = sorted(
                    all_source_scope_keys,
                )
            if all_source_titles:
                representative_properties["source_titles"] = sorted(
                    all_source_titles,
                )
            if all_question_ids:
                representative_properties["question_ids"] = sorted(
                    all_question_ids,
                )
            if all_passage_indexes:
                representative_properties["passage_indexes"] = sorted(
                    all_passage_indexes,
                )
            if title_scope_key:
                representative_properties["title_entity_scoped"] = True
                representative_properties["title_scope_key"] = title_scope_key
                representative_properties["disambiguation_scope"] = title_scope_key
            if len(all_names) > 1 or len(all_descriptions) > 1:
                representative_properties["all_names"] = sorted(all_names)
                representative_properties["all_descriptions"] = sorted(
                    all_descriptions,
                    key=len,
                    reverse=True,
                )
                representative_properties["original_ids"] = sorted(all_names)
                total_duplicates_removed += len(entities) - 1

            # Generate deterministic UUID
            entity_uuid: str = self._generate_entity_id(representative_entity)
            representative_entity["uuid"] = entity_uuid

            normalized_entity: NormalizedNode = cast(
                NormalizedNode,
                representative_entity,
            )
            harmonized_entities.append(normalized_entity)

            # Map all original name variations to the harmonized mapped_entity.
            # For split surface forms, entity_map[surface_form] must always point to the
            # dominant-type representative so relationship resolution is deterministic
            # regardless of which order entity_groups iterates the split buckets.
            # The dominant type was pre-computed in the split phase above.
            for mapped_entity in mapped_entities:
                _norm: str = self._normalize_entity_text(mapped_entity["id"])
                is_dominant: bool = (
                    dominant_for_surface.get(str(dominant_lookup_key))
                    == representative_entity.get("type")
                    if str(dominant_lookup_key) in dominant_for_surface
                    else True  # not a split surface form — always write
                )
                if is_dominant or mapped_entity["id"] not in entity_map:
                    entity_map[mapped_entity["id"]] = normalized_entity
                scope_key = str(
                    (mapped_entity.get("properties") or {}).get("source_scope_key")
                    or "",
                ).strip()
                if scope_key:
                    entity_map[(scope_key, mapped_entity["id"])] = normalized_entity

        logging.info(
            f"Harmonization complete: {len(harmonized_entities)} entities (removed {total_duplicates_removed} duplicates)",
        )

        # Log final distribution
        final_type_distribution: Counter[str] = Counter(
            e["type"] for e in harmonized_entities
        )
        logging.info(
            f"Entity type distribution after harmonization: {dict(final_type_distribution)}",
        )

        harmonized_entities, entity_map = self._apply_soft_entity_linking(
            harmonized_entities,
            entity_map,
        )
        harmonized_entities, entity_map = self._apply_optional_umls_linking(
            harmonized_entities,
            entity_map,
        )

        if return_id_map:
            return (
                harmonized_entities,
                entity_map,
            )  # entity_map: original_id → representative
        return harmonized_entities

    def _harmonize_relationships(
        self,
        all_relationships: "Sequence[HarmonizableRelationship]",
        entity_map: "Mapping[EntityLookupKey, EntityNode]",
    ) -> "list[NormalizedRelationship]":
        """
        Harmonize relationships across chunks and map to UUID-based entity IDs.

        entity_map may be either:
          - original_id → representative entity  (from _harmonize_entities with return_id_map=True)
          - uuid → entity  (legacy call sites)
        In both cases we build original_to_uuid by walking .values().
        """
        harmonized_relationships: list[NormalizedRelationship] = []
        seen_relationships: dict[str, NormalizedRelationship] = {}

        # Build variant_name → uuid from entity_map.
        # entity_map keys are ALL original variant IDs (every surface form seen during
        # extraction), values are the representative entities with uuid set.
        # Iterating .items() — not .values() — ensures every variant name is covered,
        # so relationships that used a non-canonical spelling are not silently dropped.
        original_to_uuid: dict[str, str | None] = {}
        scoped_to_uuid: dict[tuple[str, str], str | None] = {}
        candidate_names_by_uuid: dict[str | None, set[str]] = defaultdict(set)
        entity_type_by_uuid: dict[str | None, str] = {}
        for variant_name, representative in entity_map.items():
            if "uuid" not in representative:
                logging.warning(
                    f"Entity '{representative.get('id', '?')}' missing uuid in entity_map — skipping in relationship mapping",
                )
                continue
            uuid_value: str | None = representative["uuid"]
            if representative.get("type"):
                entity_type_by_uuid.setdefault(
                    uuid_value,
                    cast(str, representative.get("type")),
                )
            if (
                isinstance(variant_name, tuple)
                and len(variant_name) == 2
                and all(isinstance(v, str) for v in variant_name)
            ):
                scope_key, scoped_name = variant_name
                for candidate in self._entity_name_variants(scoped_name):
                    scoped_to_uuid.setdefault((scope_key, candidate), uuid_value)
                    scoped_to_uuid.setdefault(
                        (scope_key, candidate.lower()),
                        uuid_value,
                    )
                    candidate_names_by_uuid[uuid_value].add(candidate)
                continue
            for candidate in (
                self._entity_name_variants(variant_name)
                if isinstance(variant_name, str)
                else []
            ):
                original_to_uuid.setdefault(candidate, uuid_value)
                original_to_uuid.setdefault(candidate.lower(), uuid_value)
                candidate_names_by_uuid[uuid_value].add(candidate)
            for candidate in self._entity_candidate_names(representative):
                for variant in self._entity_name_variants(candidate):
                    original_to_uuid.setdefault(variant, uuid_value)
                    original_to_uuid.setdefault(variant.lower(), uuid_value)
                    candidate_names_by_uuid[uuid_value].add(variant)

        def _lookup_uuid(
            name: object,
            preferred_scope: str | None = None,
        ) -> str | None:
            """Try increasingly fuzzy lookups before giving up."""
            if not isinstance(name, str) or not name.strip():
                return None
            if preferred_scope:
                for candidate in self._entity_name_variants(name):
                    scoped: str | None = scoped_to_uuid.get(
                        (preferred_scope, candidate),
                    ) or scoped_to_uuid.get((preferred_scope, candidate.lower()))
                    if scoped:
                        return scoped
            exact: str | None = (
                original_to_uuid.get(name)
                or original_to_uuid.get(name.lower())
                or original_to_uuid.get(name.replace("_", " "))
                or original_to_uuid.get(name.replace("_", " ").lower())
                or original_to_uuid.get(name.replace(" ", "_"))
                or original_to_uuid.get(name.replace(" ", "_").lower())
                or original_to_uuid.get(self._normalize_entity_text(name))
            )
            if exact:
                return exact

            raw_norm: str = self._normalize_entity_text(name)
            best_uuid: str | None = None
            best_score: float = 0.0
            second_best: float = 0.0

            for uuid_value, candidates in candidate_names_by_uuid.items():
                entity_best: float = 0.0
                for candidate in candidates:
                    cand_norm: str = self._normalize_entity_text(candidate)
                    if not cand_norm or not raw_norm:
                        continue
                    if cand_norm == raw_norm:
                        entity_best = 1.0
                        break
                    if raw_norm in cand_norm or cand_norm in raw_norm:
                        if min(len(raw_norm), len(cand_norm)) >= 4:
                            entity_best = max(entity_best, 0.94)
                    else:
                        entity_best = max(
                            entity_best,
                            difflib.SequenceMatcher(None, raw_norm, cand_norm).ratio(),
                        )

                if entity_best > best_score:
                    second_best = best_score
                    best_score = entity_best
                    best_uuid = uuid_value
                elif entity_best > second_best:
                    second_best = entity_best

            if best_uuid and best_score >= 0.90 and (best_score - second_best) >= 0.03:
                return best_uuid
            return None

        dropped_unmapped: int = 0
        dropped_schema_mismatch: int = 0
        deduped_relationships: int = 0
        for rel in all_relationships:
            src: DecodedJSONValue = rel.get("source")
            tgt: DecodedJSONValue = rel.get("target")
            rel_props: GraphProperties = dict[AugmentedPropertyKey, object](
                rel.get("properties") or {},
            )
            preferred_scope: str | None = (
                str(rel_props.get("source_scope_key") or "").strip() or None
            )
            source_uuid: str | None = _lookup_uuid(src, preferred_scope=preferred_scope)
            target_uuid: str | None = _lookup_uuid(tgt, preferred_scope=preferred_scope)

            if not source_uuid or not target_uuid:
                dropped_unmapped += 1
                logging.warning(
                    "Dropping relationship — entity not found in map: '%s' -[%s]-> '%s'",
                    src,
                    rel.get("type", "?"),
                    tgt,
                )
                continue

            if source_uuid and target_uuid:
                # Create new relationship with UUID-based IDs
                uuid_rel: RawExtractionRelationship = cast(
                    RawExtractionRelationship,
                    rel.copy(),
                )
                rel_properties: GraphProperties = dict[AugmentedPropertyKey, object](
                    uuid_rel.get("properties") or {},
                )
                if isinstance(src, str):
                    rel_properties.setdefault("source_name", src)
                if isinstance(tgt, str):
                    rel_properties.setdefault("target_name", tgt)
                uuid_rel["properties"] = rel_properties
                uuid_rel["source"] = source_uuid
                uuid_rel["target"] = target_uuid

                # Include negated in the key: A-INHIBITS->B (negated=True) and
                # A-INHIBITS->B (negated=False) are opposite claims and must not collapse.
                _neg: bool = bool(uuid_rel.get("negated", False))
                condition: object = (
                    uuid_rel.get("condition") or rel_properties.get("condition") or ""
                )
                quantitative: object = (
                    uuid_rel.get("quantitative")
                    or rel_properties.get("quantitative")
                    or ""
                )
                rel_type_key: str | None = self._canonicalize_relationship_type(
                    uuid_rel.get("type", ""),
                    source_type=entity_type_by_uuid.get(source_uuid),
                    target_type=entity_type_by_uuid.get(target_uuid),
                )
                if not rel_type_key:
                    dropped_schema_mismatch += 1
                    logging.info(
                        "Dropping relationship with no schema-compatible type: '%s' -[%s]-> '%s'",
                        src,
                        rel.get("type", "?"),
                        tgt,
                    )
                    continue
                uuid_rel["type"] = rel_type_key
                rel_key: str = (
                    f"{source_uuid}:{rel_type_key}:{target_uuid}:{_neg}:"
                    f"{str(condition).strip().lower()}:{str(quantitative).strip().lower()}"
                )

                if rel_key not in seen_relationships:
                    normalized_relationship: NormalizedRelationship = cast(
                        NormalizedRelationship,
                        uuid_rel,
                    )
                    harmonized_relationships.append(normalized_relationship)
                    seen_relationships[rel_key] = normalized_relationship
                else:
                    deduped_relationships += 1
                    existing_rel: NormalizedRelationship = seen_relationships[rel_key]
                    existing_props: GraphProperties = dict[
                        AugmentedPropertyKey,
                        object,
                    ](
                        existing_rel.get("properties") or {},
                    )
                    new_props: GraphProperties = dict[AugmentedPropertyKey, object](
                        uuid_rel.get("properties") or {},
                    )
                    merged_anchor_grounding: dict[str, list[AnchorSpan]] = (
                        self._merge_anchor_grounding(
                            existing_props.get("anchor_grounding"),
                            new_props.get("anchor_grounding"),
                        )
                    )
                    if merged_anchor_grounding:
                        existing_props["anchor_grounding"] = merged_anchor_grounding
                        restoration: RestorationRecord = (
                            self._restoration_from_anchor_grounding(
                                merged_anchor_grounding,
                            )
                        )
                        existing_props["restoration_status"] = restoration["status"]
                        existing_props["restoration_verified"] = restoration["verified"]
                        existing_props["restoration_grounded_components"] = restoration[
                            "grounded_components"
                        ]
                        existing_props["restoration_grounded_count"] = restoration[
                            "grounded_count"
                        ]
                    for key, value in new_props.items():
                        existing_props.setdefault(key, value)
                    existing_rel["properties"] = existing_props
                    merged_provenance: list[int] = sorted(
                        {
                            int(pos)
                            for pos in (existing_rel.get("provenance_positions") or [])
                            + (uuid_rel.get("provenance_positions") or [])
                            if isinstance(pos, (int, float))
                        },
                    )
                    if merged_provenance:
                        existing_rel["provenance_positions"] = merged_provenance

        logging.info(
            "Relationship harmonization kept=%d dropped_unmapped=%d dropped_schema_mismatch=%d deduped=%d",
            len(harmonized_relationships),
            dropped_unmapped,
            dropped_schema_mismatch,
            deduped_relationships,
        )
        self._last_relationship_harmonization_stats = {
            "kept": len(harmonized_relationships),
            "dropped_unmapped": dropped_unmapped,
            "dropped_schema_mismatch": dropped_schema_mismatch,
            "deduped": deduped_relationships,
        }

        harmonized_relationships = self._mark_relationship_contradictions(
            harmonized_relationships,
        )

        return harmonized_relationships

    def _mark_relationship_contradictions(
        self,
        harmonized_relationships: "list[RelationshipT]",
    ) -> "list[RelationshipT]":
        """Flag relation pairs that appear with both positive and negated polarity."""
        grouped: dict[tuple[str, str, str, str, str], list[RelationshipT]] = (
            defaultdict(list)
        )
        for rel in harmonized_relationships:
            properties: GraphProperties = dict[AugmentedPropertyKey, object](
                rel.get("properties") or {},
            )
            condition: object = rel.get("condition")
            if condition is None:
                condition = properties.get("condition")
            quantitative: object = rel.get("quantitative")
            if quantitative is None:
                quantitative = properties.get("quantitative")
            key: tuple[str, str, str, str, str] = (
                str(rel.get("source") or ""),
                str(rel.get("type") or ""),
                str(rel.get("target") or ""),
                str(condition or "").strip().lower(),
                str(quantitative or "").strip().lower(),
            )
            grouped[key].append(rel)

        contradiction_groups: int = 0
        contradiction_edges: int = 0
        for rels in grouped.values():
            polarities: set[bool] = {bool(rel.get("negated", False)) for rel in rels}
            if len(polarities) < 2:
                continue
            contradiction_groups += 1
            contradiction_edges += len(rels)
            for rel in rels:
                rel_props: GraphProperties = dict[AugmentedPropertyKey, object](
                    rel.get("properties") or {},
                )
                rel_props["contradiction_detected"] = True
                rel["properties"] = rel_props
                rel["contradiction_detected"] = True

        self._last_relationship_contradiction_stats = {
            "contradiction_groups": contradiction_groups,
            "contradiction_edges": contradiction_edges,
        }
        return harmonized_relationships

    def _graph_components_from_kg(
        self,
        kg: KnowledgeGraphInput,
    ) -> list[GraphComponent]:
        """Return connected components over the entity-relation graph."""
        nodes: Sequence[EntityNode] = list(kg.get("nodes") or [])
        relationships: list[RelationshipRecord] = list(kg.get("relationships") or [])
        node_lookup: dict[str, EntityNode] = {
            str(node.get("id") or ""): node for node in nodes if node.get("id")
        }
        entity_ids: list[str] = [
            entity_id for entity_id in node_lookup.keys() if entity_id
        ]
        adjacency: dict[str, set[str]] = {entity_id: set() for entity_id in entity_ids}

        for rel in relationships:
            source_id: str = str(rel.get("source") or rel.get("from") or "").strip()
            target_id: str = str(rel.get("target") or rel.get("to") or "").strip()
            if source_id in adjacency and target_id in adjacency:
                adjacency[source_id].add(target_id)
                adjacency[target_id].add(source_id)

        visited: set[str] = set()
        components: list[GraphComponent] = []
        for entity_id in entity_ids:
            if entity_id in visited:
                continue
            stack: list[str] = [entity_id]
            component_ids: list[str] = []
            while stack:
                current: str = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component_ids.append(current)
                stack.extend(sorted(adjacency.get(current, set()) - visited))

            component_set: set[str] = set(component_ids)
            component_relationships: list[RelationshipRecord] = [
                rel
                for rel in relationships
                if str(rel.get("source") or rel.get("from") or "").strip()
                in component_set
                and str(rel.get("target") or rel.get("to") or "").strip()
                in component_set
            ]
            degree: Counter[str] = Counter()
            for rel in component_relationships:
                source_id = str(rel.get("source") or rel.get("from") or "").strip()
                target_id = str(rel.get("target") or rel.get("to") or "").strip()
                if source_id:
                    degree[source_id] += 1
                if target_id:
                    degree[target_id] += 1

            top_entities: list[str] = sorted(
                component_ids,
                key=lambda entity_uuid: (
                    -degree.get(entity_uuid, 0),
                    str(
                        (node_lookup.get(entity_uuid, {}).get("properties") or {}).get(
                            "name",
                        )
                        or entity_uuid,
                    ),
                ),
            )[: max(1, getattr(self, "max_summary_entities", 6))]
            top_relationships: list[RelationshipRecord] = sorted(
                component_relationships,
                key=lambda rel: (
                    -self._confidence_score(
                        (rel.get("properties") or {}).get("confidence"),
                    ),
                    str(rel.get("type") or ""),
                ),
            )[: max(1, getattr(self, "max_summary_relationships", 6))]
            components.append(
                {
                    "entity_ids": sorted(component_ids),
                    "relationships": top_relationships,
                    "entity_count": len(component_ids),
                    "relationship_count": len(component_relationships),
                    "degree": degree,
                    "top_entities": top_entities,
                },
            )

        return components

    def _build_graph_enrichment_records(
        self,
        kg: KnowledgeGraphInput,
    ) -> GraphEnrichment:
        """Create optional component summaries, claims, and fragmentation bridges."""
        if not (
            getattr(self, "enable_graph_summaries", False)
            or getattr(self, "enable_claim_extraction", False)
            or getattr(self, "enable_fragmentation_repair", False)
        ):
            return {}

        node_lookup: dict[str, EntityNode] = {
            str(node.get("id") or ""): node
            for node in (kg.get("nodes") or [])
            if node.get("id")
        }
        components: list[GraphComponent] = self._graph_components_from_kg(kg)
        include_component_summaries: bool = bool(
            getattr(self, "enable_graph_summaries", False),
        )
        component_summaries: list[ComponentSummary] = []
        all_claims: list[ClaimRecord] = []

        for component_index, component in enumerate(components):
            if include_component_summaries:
                summary_id: str = hashlib.sha1(
                    f"{kg.get('metadata', GraphMetadata()).get('kg_name', '')}:component:{component_index}".encode(),
                ).hexdigest()
                summary_text: str = self._heuristic_component_summary_text(
                    component,
                    node_lookup,
                )
                component_record: ComponentSummary = {
                    "id": summary_id,
                    "component_index": component_index,
                    "entity_ids": list(component.get("entity_ids") or []),
                    "top_entities": list(component.get("top_entities") or []),
                    "entity_count": int(component.get("entity_count") or 0),
                    "relationship_count": int(component.get("relationship_count") or 0),
                    "text": summary_text,
                }
                component_summaries.append(component_record)
            all_claims.extend(
                self._claim_records_from_component(
                    component,
                    node_lookup,
                    component_index=component_index,
                ),
            )

        graph_summary: GraphSummary | None = None
        if getattr(self, "enable_graph_summaries", False):
            preview: list[str] = [
                f"Component {record['component_index'] + 1}: {record['entity_count']} entities, {record['relationship_count']} relationships"
                for record in component_summaries[:3]
            ]
            graph_summary = {
                "id": hashlib.sha1(
                    f"{kg.get('metadata', GraphMetadata()).get('kg_name', '')}:graph_summary".encode(),
                ).hexdigest(),
                "text": (
                    f"Graph contains {len(component_summaries)} connected components. "
                    + " ".join(preview)
                ).strip(),
                "component_count": len(component_summaries),
            }

        return {
            "component_summaries": component_summaries,
            "claims": all_claims,
            "graph_summary": graph_summary,
            "fragmentation_bridges": self._propose_fragmentation_bridges(
                components,
                node_lookup,
            ),
        }

    @staticmethod
    def _attach_relationship_provenance(
        relationships: RelationshipRecoveryInput,
        chunk_positions: list[int],
    ) -> list[ExtractionRelationship]:
        """Attach local chunk provenance so triple verification can stay passage-local."""
        annotated: list[ExtractionRelationship] = []
        normalized_positions: list[int] = [
            int(pos) for pos in chunk_positions if isinstance(pos, (int, float))
        ]
        for rel in relationships or []:
            if not isinstance(rel, dict):
                continue
            rel_copy: ExtractionRelationship = cast(
                "ExtractionRelationship",
                dict(rel),
            )
            existing_positions: DecodedJSONValue | list[int] = (
                rel_copy.get("provenance_positions") or []
            )
            merged_positions: set[int] = {
                int(pos) for pos in existing_positions if isinstance(pos, (int, float))
            }
            merged_positions.update(normalized_positions)
            rel_copy["provenance_positions"] = sorted(merged_positions)
            annotated.append(rel_copy)
        return annotated

    @staticmethod
    def _relationship_local_provenance(
        rel: "RelationshipRecord",
        chunks: Sequence[GraphChunkInput],
    ) -> RelationshipProvenance:
        """
        Resolve question-local provenance for a relationship from chunk positions.

        Older KGs only expose chunk positions on relationships. Newer builds also
        stamp question and passage provenance onto the edge so question-scoped
        retrieval can be enforced exactly at query time.
        """
        positions: set[int] = {
            int(pos)
            for pos in (rel.get("provenance_positions") or [])
            if isinstance(pos, (int, float))
        }
        if not positions:
            return {
                "provenance_positions": [],
                "question_ids": [],
                "passage_keys": [],
            }

        question_ids: set[str] = set()
        passage_keys: set[str] = set()
        for chunk in chunks or []:
            pos: int | None = chunk.get("position")
            if not isinstance(pos, (int, float)) or int(pos) not in positions:
                continue
            qid: str | None = chunk.get("question_id")
            if qid is not None and str(qid).strip():
                question_ids.add(str(qid))
            passage_key: str = (
                f"{chunk.get('question_id', '')}::p{chunk.get('passage_index', -1)}"
            )
            passage_keys.add(passage_key)

        return {
            "provenance_positions": sorted(positions),
            "question_ids": sorted(question_ids),
            "passage_keys": sorted(passage_keys),
        }

    def _compute_node_specificity_weights(
        self,
        graph: QueryGraph,
        kg_name: str | None,
    ) -> None:
        """Compute HippoRAG-style node specificity within the active KG scope."""
        graph.query(
            """
            MATCH (c:Chunk)-[:PART_OF]->(d:Document)
            WHERE $kg_name IS NULL OR d.kgName = $kg_name
            MATCH (c)-[:HAS_ENTITY]->(e:__Entity__)
            WITH e, count(DISTINCT c) AS passage_count
            SET e.passage_count = passage_count,
                e.node_specificity = CASE
                    WHEN passage_count > 0 THEN 1.0 / toFloat(passage_count)
                    ELSE 1.0
                END
            """,
            {"kg_name": kg_name},
        )

    def generate_knowledge_graph(
        self,
        text: str,
        llm: "GenerationProvider | None",
        file_name: str | None = None,
        model_name: str = "openai/gpt-oss-120b:free",
        max_chunks: int | None = None,
        kg_name: str | None = None,
        doc_metadata: "Mapping[str, object] | None" = None,
        doc_hash: str | None = None,
    ) -> GeneratedKnowledgeGraph:
        """
        Generate knowledge graph from text with ontology-guided entity extraction

        Args:
            text: Input text to process
            llm: LLM provider instance
            file_name: Optional filename for storage
            model_name: LLM model name
            max_chunks: Maximum number of chunks to process (for large documents)
        """
        logging.info("Starting ontology-guided knowledge graph generation")

        # Determine if ontology is available
        has_ontology: bool = bool(self.ontology_classes) or bool(
            self.ontology_relationships,
        )
        extraction_method: str = (
            "ontology_guided_llm" if has_ontology else "natural_llm"
        )
        logging.info(f"Extraction method: {extraction_method}")

        # Step 1b: Detect section headers once across the full document.
        # Each chunk is tagged with its section (e.g. "Methods", "Results") so
        # the extraction LLM can interpret claims in the correct context.
        section_headers: list[tuple[int, str]] = self._detect_section_headers(text)
        logging.info(
            "Detected %d section headers: %s",
            len(section_headers),
            [name for _, name in section_headers],
        )

        # Step 1: Chunk the text, keeping detected document sections separate when
        # possible so extraction never straddles unrelated headers.
        chunks: list[TextChunk] = self._chunk_text_with_section_boundaries(
            text,
            section_headers=section_headers,
        )
        logging.info(f"Created {len(chunks)} chunks")

        # Limit chunks if specified (for very large documents)
        if max_chunks and len(chunks) > max_chunks:
            logging.warning(
                f"Limiting processing to {max_chunks} chunks out of {len(chunks)} total",
            )
            chunks = chunks[:max_chunks]

        # Step 2: Extract entities and relationships from each chunk
        all_entities: list[ExtractionEntity] = []
        all_relationships: list[ExtractionRelationship] = []
        processed_chunks: int = 0
        failed_chunks: int = 0
        entities_per_chunk: dict[
            int,
            Sequence[ExtractionEntity],
        ] = {}  # chunk index → entities (for cross-chunk pass)

        for i, chunk in enumerate(chunks):
            try:
                logging.info(f"Processing chunk {i + 1}/{len(chunks)}")

                # --- Context enrichment ---
                # 1. Section header for this chunk
                chunk_section: str | None = self._get_section_for_position(
                    chunk.get("start_pos", 0),
                    section_headers,
                )
                chunk["section_name"] = chunk_section or "Preamble"

                # 2. Qualifier sentences from the previous chunk
                qualifier_ctx: str = ""
                if i > 0:
                    qualifier_ctx = self._extract_qualifier_sentences(
                        chunks[i - 1]["text"],
                    )

                previous_chunk_indexes: list[int] = list(range(max(0, i - 2), i))
                previous_entities: list[ExtractionEntity] = [
                    entity
                    for prev_idx in previous_chunk_indexes
                    for entity in entities_per_chunk.get(prev_idx, [])
                ]
                previous_texts: list[str] = [
                    chunks[prev_idx]["text"] for prev_idx in previous_chunk_indexes
                ]
                extraction_text: str = self._prepare_chunk_text_for_extraction(
                    chunk["text"],
                    previous_entities=previous_entities,
                    previous_texts=previous_texts,
                    llm=llm,
                    model_name=model_name,
                )

                chunk_kg: ExtractionResult = (
                    self._extract_entities_and_relationships_with_llm(
                        extraction_text,
                        llm,
                        model_name,
                        context_header=qualifier_ctx or None,
                        section_header=chunk_section,
                    )
                )
                chunk_kg = self._ground_chunk_extraction(chunk_kg, chunk)

                # Ensure chunk_kg is a dictionary with expected keys
                try:
                    if isinstance(chunk_kg, dict) and (
                        chunk_kg.get("entities", [])
                        or chunk_kg.get("relationships", [])
                    ):
                        chunk_entities: Sequence[ExtractionEntity] = chunk_kg.get(
                            "entities",
                            [],
                        )
                        all_entities.extend(chunk_entities)
                        all_relationships.extend(
                            self._attach_relationship_provenance(
                                chunk_kg.get("relationships", []),
                                [chunk.get("position", i)],
                            ),
                        )
                        entities_per_chunk[i] = chunk_entities
                        processed_chunks += 1
                        logging.info(
                            f"✓ Chunk {i + 1} processed: {len(chunk_entities)} entities, {len(chunk_kg.get('relationships', []))} relationships",
                        )
                    else:
                        logging.warning(
                            f"⚠ Chunk {i + 1} returned invalid or empty format: {type(chunk_kg)}",
                        )
                        failed_chunks += 1
                except Exception as inner_e:
                    logging.error(f"❌ Error processing chunk result: {inner_e}")
                    failed_chunks += 1

            except Exception as e:
                logging.error(f"❌ Failed to process chunk {i + 1}: {e}")
                import traceback

                logging.error(f"Traceback: {traceback.format_exc()}")
                failed_chunks += 1
                continue

            # Add small delay between chunks only when an external LLM is in use
            if (
                llm is not None and i < len(chunks) - 1
            ):  # Don't delay after the last chunk
                time.sleep(1.0)  # 1 second delay between API calls

        logging.info(
            f"Processing complete: {processed_chunks} successful, {failed_chunks} failed",
        )

        # Step 2b: Cross-chunk relationship extraction across wider adjacent windows.
        # The per-chunk LLM pass can only see entities within one chunk at a time, so relationships
        # that span chunk boundaries are otherwise lost. We recover them with a relation-only pass
        # over adjacent windows (default size 2-3), which improves recall without re-running
        # full entity extraction.
        if llm is not None and len(chunks) > 1:
            chunk_positions: dict[int, list[int]] = {
                i: [int(chunks[i].get("position", i))] for i in range(len(chunks))
            }
            all_relationships.extend(
                self._extract_relationships_for_segment_windows(
                    [chunk.get("text", "") for chunk in chunks],
                    entities_per_chunk,
                    chunk_positions,
                    llm,
                    model_name,
                    max_window_size=getattr(self, "cross_chunk_relation_window", 3),
                    scope_label="Cross-chunk",
                ),
            )

            all_relationships.extend(
                self._recover_cross_section_relationships(
                    text=text,
                    chunks=chunks,
                    entities_per_chunk=entities_per_chunk,
                    section_headers=section_headers,
                    llm=llm,
                    model_name=model_name,
                    scope_label="Cross-section",
                ),
            )

        if processed_chunks == 0 and failed_chunks > 0:
            raise RuntimeError(
                f"KG extraction failed: all {failed_chunks} chunk(s) returned errors. "
                "Check LLM connectivity and rate limits.",
            )

        # Step 3: Harmonize entities and relationships.
        # _harmonize_entities returns the representative entities AND builds the
        # full original-ID → representative mapping internally (stored on the
        # entities as entity['_all_ids'] is NOT available, so we rebuild it here).
        # We need every original variant ID to map to its representative so that
        # relationships whose source/target used a non-canonical name aren't dropped.
        harmonized_entities, id_to_representative = self._harmonize_entities(
            all_entities,
            return_id_map=True,
        )
        harmonized_relationships: list[NormalizedRelationship] = (
            self._harmonize_relationships(all_relationships, id_to_representative)
        )
        harmonized_entities, harmonized_relationships = (
            self._coerce_harmonized_entities_to_schema(
                harmonized_entities,
                harmonized_relationships,
            )
        )
        harmonized_relationships = self._mark_relationship_contradictions(
            harmonized_relationships,
        )

        logging.info(
            f"Harmonized to {len(harmonized_entities)} entities and {len(harmonized_relationships)} relationships",
        )

        # Step 4: Format the final knowledge graph
        # Use UUID-based IDs to prevent duplicates
        kg_prefix: str = f"{kg_name}_" if kg_name else ""

        kg: GeneratedKnowledgeGraph = {
            "nodes": [
                {
                    "id": f"{kg_prefix}{entity['uuid']}",
                    "label": entity["type"],
                    "properties": {
                        "name": entity["id"],
                        "type": entity["type"],
                        "original_id": entity["id"],  # Keep original ID for reference
                        **entity.get("properties", {}),
                    },
                    "embedding": entity.get("embedding"),
                    "color": self._get_node_color(entity["type"]),
                    "size": 30,
                    "font": {"size": 14, "color": "#333333"},
                    "title": f"Entity: {entity['id']}\nType: {entity['type']}\nKG: {kg_name or 'default'}\nClick for details",
                }
                for entity in harmonized_entities
            ],
            "relationships": [
                {
                    "id": f"{kg_prefix}rel_{rel['source']}_{rel['type']}_{rel['target']}_{idx}",
                    "from": f"{kg_prefix}{rel['source']}",
                    "to": f"{kg_prefix}{rel['target']}",
                    "source": f"{kg_prefix}{rel['source']}",
                    "target": f"{kg_prefix}{rel['target']}",
                    "type": rel["type"],
                    "label": rel["type"],
                    "negated": rel.get("negated", False),
                    "properties": rel.get("properties", {}),
                    "arrows": "to",
                    "color": {"color": "#444444"},
                    "font": {"size": 12, "align": "middle"},
                }
                for idx, rel in enumerate(harmonized_relationships)
            ],
            "chunks": chunks,
            "metadata": {
                "total_chunks": len(chunks),
                "total_entities": len(harmonized_entities),
                "total_relationships": len(harmonized_relationships),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": type(self.embedding_function).__name__,
                "embedding_dimension": self.embedding_dimension,
                "ontology_classes": len(self.ontology_classes),
                "ontology_relationships": len(self.ontology_relationships),
                "extraction_method": "ontology_guided_llm"
                if has_ontology
                else "natural_llm",
                "kg_name": kg_name,
                "provider": provider,
                "model": model_name,
                "max_chunks_setting": max_chunks,
                "created_at": datetime.now().isoformat(),
                "visualization_ready": True,
                "file_name": file_name,
                "doc_hash": doc_hash,
                "schema_card": self._build_schema_card(),
            },
        }
        kg["metadata"].update(
            {
                "schema_enforcement_dropped_entities": self._last_schema_enforcement_stats.get(
                    "dropped_entities",
                    0,
                ),
                "schema_enforcement_dropped_relationships": self._last_schema_enforcement_stats.get(
                    "dropped_relationships",
                    0,
                ),
                "harmonization_relationships_dropped_unmapped": self._last_relationship_harmonization_stats.get(
                    "dropped_unmapped",
                    0,
                ),
                "harmonization_relationships_dropped_schema_mismatch": self._last_relationship_harmonization_stats.get(
                    "dropped_schema_mismatch",
                    0,
                ),
                "harmonization_relationships_deduped": self._last_relationship_harmonization_stats.get(
                    "deduped",
                    0,
                ),
                "harmonization_relationship_contradiction_groups": self._last_relationship_contradiction_stats.get(
                    "contradiction_groups",
                    0,
                ),
                "harmonization_relationship_contradiction_edges": self._last_relationship_contradiction_stats.get(
                    "contradiction_edges",
                    0,
                ),
                "anchor_grounded_entities": sum(
                    1
                    for entity in harmonized_entities
                    if (entity.get("properties") or {}).get("anchor_spans")
                ),
                "restoration_full_relationships": sum(
                    1
                    for rel in harmonized_relationships
                    if (
                        (rel.get("properties") or {}).get("restoration_status")
                        == "full"
                    )
                ),
                "restoration_partial_relationships": sum(
                    1
                    for rel in harmonized_relationships
                    if (
                        (rel.get("properties") or {}).get("restoration_status")
                        == "partial"
                    )
                ),
            },
        )
        enrichment: GraphEnrichment = self._build_graph_enrichment_records(kg)
        if enrichment:
            kg["metadata"]["graph_enrichment"] = enrichment
            kg["metadata"]["component_summary_count"] = len(
                enrichment.get("component_summaries") or [],
            )
            kg["metadata"]["claim_count"] = len(enrichment.get("claims") or [])
            kg["metadata"]["fragmentation_bridge_count"] = len(
                enrichment.get("fragmentation_bridges") or [],
            )

        # Attach any document-level metadata from the source (e.g. CSV columns)
        if doc_metadata:
            kg["metadata"]["doc_metadata"] = doc_metadata

        # Step 5: Store in Neo4j if requested
        if file_name:
            success: bool = self.store_knowledge_graph_with_embeddings(
                kg,
                file_name,
                doc_metadata=doc_metadata,
                doc_hash=doc_hash,
                llm=llm,
                model_name=model_name,
            )
            kg["metadata"]["stored_in_neo4j"] = success

            # Step 5b: HippoRAG-style synonym merging.
            # After embeddings are in Neo4j, cluster near-duplicate entity nodes
            # (e.g. "TBK1" / "TBK1 kinase") and merge them so entity-first search
            # can find all surface-form variants via a single canonical node.
            if success:
                try:
                    graph_for_merge: QueryGraph = self._create_neo4j_connection()
                    merges: int = self.merge_synonym_entities(
                        graph_for_merge,
                        kg_name=kg_name,
                    )
                    kg["metadata"]["synonym_merges"] = merges
                    logging.info(f"Synonym merging complete: {merges} pairs merged")
                except Exception as syn_err:
                    logging.warning(f"Synonym merging failed (non-fatal): {syn_err}")

                # Compute node specificity (HippoRAG-style IDF weight):
                # s(e) = 1 / |passages containing e|.  Stored on each entity node
                # so that retrieval can down-weight ubiquitous hub entities as seeds.
                try:
                    graph_for_merge = self._create_neo4j_connection()
                    self._compute_node_specificity_weights(graph_for_merge, kg_name)
                    logging.info(
                        "Node specificity weights computed for kg_name=%s",
                        kg_name,
                    )
                except Exception as spec_err:
                    logging.warning(
                        "Node specificity computation failed (non-fatal): %s",
                        spec_err,
                    )

        return kg

    def generate_knowledge_graph_from_passages(
        self,
        passages: "Sequence[ContextPassageInput]",
        llm: "GenerationProvider | None",
        file_name: str | None = None,
        model_name: str = "openai/gpt-oss-120b:free",
        kg_name: str | None = None,
        doc_metadata: "Mapping[str, object] | None" = None,
        doc_hash: str | None = None,
    ) -> GeneratedKnowledgeGraph:
        """Generate a KG from a list of ContextPassage objects.

        Unlike generate_knowledge_graph, this method never concatenates passages
        from different records before chunking.  Each passage is chunked
        independently; sub-splitting only occurs when a single passage exceeds
        chunk_size.  Cross-chunk relationship extraction is scoped within each
        passage so the LLM never sees entity pairs that are only co-located
        because two unrelated passages were glued together.

        Harmonisation and Neo4j storage happen once after all passages are
        processed, so the result is equivalent in structure to the single-call
        path but with clean passage-level extraction boundaries.
        """
        logging.info("Starting passage-aware KG generation: %d passages", len(passages))
        has_ontology: bool = bool(self.ontology_classes) or bool(
            self.ontology_relationships,
        )
        extraction_method: str = (
            "ontology_guided_llm" if has_ontology else "natural_llm"
        )

        all_chunks: list[TextChunk] = []
        all_entities: list[ExtractionEntity] = []
        all_relationships: list[ExtractionRelationship] = []
        passage_entities: dict[int, Sequence[ExtractionEntity]] = {}
        passage_positions: dict[int, list[int]] = {}

        global_chunk_offset: int = (
            0  # running count for position uniqueness across passages
        )

        for p_idx, passage in enumerate(passages):
            passage_text: str = passage.text
            source_label: str = (
                f"{passage.dataset}/{passage.question_id}/p{passage.passage_index}"
            )
            source_title: object = getattr(
                passage,
                "source_title",
                None,
            ) or self._infer_passage_title(passage_text, passage.dataset)
            source_scope_key: str | None = self._build_passage_scope_key(
                dataset=passage.dataset,
                question_id=passage.question_id,
                passage_index=passage.passage_index,
            )

            # Shift position fields so they are globally unique across passages.
            section_headers: list[tuple[int, str]] = self._detect_section_headers(
                passage_text,
            )
            p_chunks: list[TextChunk] = self._chunk_text_with_section_boundaries(
                passage_text,
                section_headers=section_headers,
            )
            for local_i, ch in enumerate(p_chunks):
                ch["position"] = global_chunk_offset + local_i
                ch["source"] = source_label
                ch["dataset"] = passage.dataset
                ch["question_id"] = passage.question_id
                ch["passage_index"] = passage.passage_index
                ch["chunk_local_index"] = ch.get("chunk_id", local_i)
                ch["source_title"] = source_title
                ch["source_scope_key"] = source_scope_key

            entities_this_passage: dict[int, Sequence[ExtractionEntity]] = {}
            processed: int = 0
            failed: int = 0

            for local_i, chunk in enumerate(p_chunks):
                global_i: int = global_chunk_offset + local_i
                try:
                    chunk_section: str | None = self._get_section_for_position(
                        chunk.get("start_pos", 0),
                        section_headers,
                    )
                    chunk["section_name"] = chunk_section or "Preamble"
                    qualifier_ctx: str = ""
                    if local_i > 0:
                        qualifier_ctx = self._extract_qualifier_sentences(
                            p_chunks[local_i - 1]["text"],
                        )
                    previous_local_indexes: list[int] = list(
                        range(max(0, local_i - 2), local_i),
                    )
                    previous_entities: list[ExtractionEntity] = [
                        entity
                        for prev_idx in previous_local_indexes
                        for entity in entities_this_passage.get(prev_idx, [])
                    ]
                    previous_texts: list[str] = [
                        p_chunks[prev_idx]["text"]
                        for prev_idx in previous_local_indexes
                    ]
                    extraction_text: str = self._prepare_chunk_text_for_extraction(
                        chunk["text"],
                        previous_entities=previous_entities,
                        previous_texts=previous_texts,
                        llm=llm,
                        model_name=model_name,
                    )

                    chunk_kg: ExtractionResult = (
                        self._extract_entities_and_relationships_with_llm(
                            extraction_text,
                            llm,
                            model_name,
                            context_header=qualifier_ctx or None,
                            section_header=chunk_section,
                        )
                    )
                    chunk_kg = self._ground_chunk_extraction(chunk_kg, chunk)
                    if isinstance(chunk_kg, dict) and (
                        chunk_kg.get("entities") or chunk_kg.get("relationships")
                    ):
                        chunk_entities: Sequence[ExtractionEntity] = chunk_kg.get(
                            "entities",
                            [],
                        )
                        all_entities.extend(chunk_entities)
                        all_relationships.extend(
                            self._attach_relationship_provenance(
                                chunk_kg.get("relationships", []),
                                [chunk.get("position", global_i)],
                            ),
                        )
                        entities_this_passage[local_i] = chunk_entities
                        processed += 1
                        logging.info(
                            "Passage %d/%d chunk %d/%d: %d entities, %d rels [%s]",
                            p_idx + 1,
                            len(passages),
                            local_i + 1,
                            len(p_chunks),
                            len(chunk_entities),
                            len(chunk_kg.get("relationships", [])),
                            source_label,
                        )
                    else:
                        failed += 1
                except Exception as e:
                    logging.error(
                        "Passage %d chunk %d failed: %s",
                        p_idx + 1,
                        local_i + 1,
                        e,
                    )
                    failed += 1
                    continue

                if llm is not None and local_i < len(p_chunks) - 1:
                    time.sleep(1.0)

            # Cross-chunk relationship extraction — scoped within this passage only.
            if llm is not None and len(p_chunks) > 1:
                chunk_positions_this_passage: dict[int, list[int]] = {
                    local_i: [
                        int(
                            p_chunks[local_i].get(
                                "position",
                                global_chunk_offset + local_i,
                            ),
                        ),
                    ]
                    for local_i in range(len(p_chunks))
                }
                all_relationships.extend(
                    self._extract_relationships_for_segment_windows(
                        [chunk.get("text", "") for chunk in p_chunks],
                        entities_this_passage,
                        chunk_positions_this_passage,
                        llm,
                        model_name,
                        max_window_size=getattr(self, "cross_chunk_relation_window", 3),
                        scope_label=f"Cross-chunk[{source_label}]",
                        relationship_scope_metadata={
                            "dataset": passage.dataset,
                            "question_id": str(passage.question_id),
                            "passage_index": int(passage.passage_index),
                            "source": source_label,
                            "source_title": source_title,
                            "source_scope_key": source_scope_key,
                        },
                    ),
                )
                all_relationships.extend(
                    self._recover_cross_section_relationships(
                        text=passage_text,
                        chunks=p_chunks,
                        entities_per_chunk=entities_this_passage,
                        section_headers=section_headers,
                        llm=llm,
                        model_name=model_name,
                        scope_label=f"Cross-section[{source_label}]",
                    ),
                )

            passage_entities[p_idx] = [
                entity
                for entity_list in entities_this_passage.values()
                for entity in entity_list
            ]
            passage_positions[p_idx] = [
                int(chunk.get("position", global_chunk_offset + local_i))
                for local_i, chunk in enumerate(p_chunks)
            ]

            all_chunks.extend(p_chunks)
            global_chunk_offset += len(p_chunks)

            logging.info(
                "Passage %d/%d complete: processed=%d failed=%d [%s]",
                p_idx + 1,
                len(passages),
                processed,
                failed,
                source_label,
            )

        logging.info(
            "All passages processed: %d chunks, %d raw entities, %d raw relationships",
            len(all_chunks),
            len(all_entities),
            len(all_relationships),
        )

        if not all_entities and not all_relationships:
            raise RuntimeError(
                "Passage-aware KG extraction yielded no entities or relationships. "
                "Check LLM connectivity and passage content.",
            )

        if (
            llm is not None
            and len(passages) > 1
            and getattr(self, "enable_cross_passage_relation_recovery", True)
        ):
            cross_passage_groups: list[PassageGroup[ExtractionEntity]] = (
                self._group_passages_for_cross_passage_recovery(
                    passages,
                    passage_entities,
                    passage_positions,
                )
            )
            for group in cross_passage_groups:
                if len(group["texts"]) < 2:
                    continue
                all_relationships.extend(
                    self._extract_relationships_for_segment_windows(
                        group["texts"],
                        group["entities"],
                        group["positions"],
                        llm,
                        model_name,
                        max_window_size=getattr(
                            self,
                            "cross_passage_relation_window",
                            2,
                        ),
                        scope_label=group["source_label"],
                    ),
                )

        # Harmonise globally across all passages.
        harmonized_entities, id_to_representative = self._harmonize_entities(
            all_entities,
            return_id_map=True,
        )
        harmonized_relationships: list[NormalizedRelationship] = (
            self._harmonize_relationships(all_relationships, id_to_representative)
        )
        harmonized_entities, harmonized_relationships = (
            self._coerce_harmonized_entities_to_schema(
                harmonized_entities,
                harmonized_relationships,
            )
        )
        harmonized_relationships = self._mark_relationship_contradictions(
            harmonized_relationships,
        )
        logging.info(
            "Harmonized: %d entities, %d relationships",
            len(harmonized_entities),
            len(harmonized_relationships),
        )

        kg_prefix: str = f"{kg_name}_" if kg_name else ""
        kg: GeneratedKnowledgeGraph = {
            "nodes": [
                {
                    "id": f"{kg_prefix}{entity['uuid']}",
                    "label": entity["type"],
                    "properties": {
                        "name": entity["id"],
                        "type": entity["type"],
                        "original_id": entity["id"],
                        **entity.get("properties", {}),
                    },
                    "embedding": entity.get("embedding"),
                    "color": self._get_node_color(entity["type"]),
                    "size": 30,
                    "font": {"size": 14, "color": "#333333"},
                    "title": (
                        f"Entity: {entity['id']}\nType: {entity['type']}"
                        f"\nKG: {kg_name or 'default'}\nClick for details"
                    ),
                }
                for entity in harmonized_entities
            ],
            "relationships": [
                {
                    "id": f"{kg_prefix}rel_{rel['source']}_{rel['type']}_{rel['target']}_{idx}",
                    "from": f"{kg_prefix}{rel['source']}",
                    "to": f"{kg_prefix}{rel['target']}",
                    "source": f"{kg_prefix}{rel['source']}",
                    "target": f"{kg_prefix}{rel['target']}",
                    "type": rel["type"],
                    "label": rel["type"],
                    "negated": rel.get("negated", False),
                    "properties": rel.get("properties", {}),
                    "provenance_positions": rel.get("provenance_positions") or [],
                    "arrows": "to",
                    "color": {"color": "#444444"},
                    "font": {"size": 12, "align": "middle"},
                }
                for idx, rel in enumerate(harmonized_relationships)
            ],
            "chunks": all_chunks,
            "metadata": {
                "total_chunks": len(all_chunks),
                "total_passages": len(passages),
                "total_entities": len(harmonized_entities),
                "total_relationships": len(harmonized_relationships),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": type(self.embedding_function).__name__,
                "embedding_dimension": self.embedding_dimension,
                "ontology_classes": len(self.ontology_classes),
                "ontology_relationships": len(self.ontology_relationships),
                "extraction_method": extraction_method,
                "kg_name": kg_name,
                "created_at": datetime.now().isoformat(),
                "visualization_ready": True,
                "file_name": file_name,
                "doc_hash": doc_hash,
            },
        }
        kg["metadata"].update(
            {
                "schema_enforcement_dropped_entities": self._last_schema_enforcement_stats.get(
                    "dropped_entities",
                    0,
                ),
                "schema_enforcement_dropped_relationships": self._last_schema_enforcement_stats.get(
                    "dropped_relationships",
                    0,
                ),
                "harmonization_relationships_dropped_unmapped": self._last_relationship_harmonization_stats.get(
                    "dropped_unmapped",
                    0,
                ),
                "harmonization_relationships_dropped_schema_mismatch": self._last_relationship_harmonization_stats.get(
                    "dropped_schema_mismatch",
                    0,
                ),
                "harmonization_relationships_deduped": self._last_relationship_harmonization_stats.get(
                    "deduped",
                    0,
                ),
                "harmonization_relationship_contradiction_groups": self._last_relationship_contradiction_stats.get(
                    "contradiction_groups",
                    0,
                ),
                "harmonization_relationship_contradiction_edges": self._last_relationship_contradiction_stats.get(
                    "contradiction_edges",
                    0,
                ),
                "anchor_grounded_entities": sum(
                    1
                    for entity in harmonized_entities
                    if (entity.get("properties") or {}).get("anchor_spans")
                ),
                "restoration_full_relationships": sum(
                    1
                    for rel in harmonized_relationships
                    if (
                        (rel.get("properties") or {}).get("restoration_status")
                        == "full"
                    )
                ),
                "restoration_partial_relationships": sum(
                    1
                    for rel in harmonized_relationships
                    if (
                        (rel.get("properties") or {}).get("restoration_status")
                        == "partial"
                    )
                ),
            },
        )
        enrichment: GraphEnrichment = self._build_graph_enrichment_records(kg)
        if enrichment:
            kg["metadata"]["graph_enrichment"] = enrichment
            kg["metadata"]["component_summary_count"] = len(
                enrichment.get("component_summaries") or [],
            )
            kg["metadata"]["claim_count"] = len(enrichment.get("claims") or [])
            kg["metadata"]["fragmentation_bridge_count"] = len(
                enrichment.get("fragmentation_bridges") or [],
            )

        if doc_metadata:
            kg["metadata"]["doc_metadata"] = doc_metadata

        if file_name:
            success: bool = self.store_knowledge_graph_with_embeddings(
                kg,
                file_name,
                doc_metadata=doc_metadata,
                doc_hash=doc_hash,
                llm=llm,
                model_name=model_name,
            )
            kg["metadata"]["stored_in_neo4j"] = success
            if success:
                try:
                    graph_for_merge: QueryGraph = self._create_neo4j_connection()
                    merges: int = self.merge_synonym_entities(
                        graph_for_merge,
                        kg_name=kg_name,
                    )
                    kg["metadata"]["synonym_merges"] = merges
                    logging.info("Synonym merging complete: %d pairs merged", merges)
                except Exception as syn_err:
                    logging.warning("Synonym merging failed (non-fatal): %s", syn_err)

                try:
                    graph_for_merge = self._create_neo4j_connection()
                    self._compute_node_specificity_weights(graph_for_merge, kg_name)
                    logging.info(
                        "Node specificity weights computed for kg_name=%s",
                        kg_name,
                    )
                except Exception as spec_err:
                    logging.warning(
                        "Node specificity computation failed (non-fatal): %s",
                        spec_err,
                    )

        return kg

    def _get_node_color(self, entity_type: str) -> str:
        """Get color for node based on entity type"""
        color_map: dict[str, str] = {
            "Disease": "#ff6666",
            "Treatment": "#66ff66",
            "Drug": "#66ffff",
            "Symptom": "#ffff66",
            "Patient": "#ff9999",
            "Physician": "#ffcc99",
            "Hospital": "#99ccff",
            "MedicalProcedure": "#cc99ff",
            "MedicalDevice": "#ff99cc",
            "Anatomy": "#99ff99",
            "Concept": "#a6cee3",
        }
        return color_map.get(entity_type, "#a6cee3")

    def get_rag_context(self, query: str, top_k: int = 5) -> CreatorRagContext:
        """
        Get RAG context for a query using vector similarity search
        """
        try:
            graph: QueryGraph = self._create_neo4j_connection()

            # Generate query embedding
            query_embedding: list[float] = self.embedding_function.embed_query(query)

            # Vector search query
            search_query: str = """
            CALL db.index.vector.queryNodes('vector', $top_k, $query_vector)
            YIELD node AS chunk, score
            MATCH (chunk)-[:PART_OF]->(d:Document)
            OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:__Entity__)
            WITH chunk, score, d, collect(e) AS entities
            RETURN
                chunk.text AS text,
                chunk.id AS chunk_id,
                score,
                d.fileName AS document,
                [entity IN entities | {id: entity.id, type: entity.type, description: entity.description}] AS entities
            ORDER BY score DESC
            """

            results: Sequence[CreatorContextChunk] = cast(
                "Sequence[CreatorContextChunk]",
                graph.query(
                    search_query,
                    {"top_k": top_k, "query_vector": query_embedding},
                ),
            )

            context: _CreatorContextAccumulator = {
                "query": query,
                "chunks": [],
                "entities": set(),
                "documents": set(),
            }

            for result in results:
                context["chunks"].append(
                    {
                        "text": result["text"],
                        "chunk_id": result["chunk_id"],
                        "score": result["score"],
                        "document": result["document"],
                        "entities": result["entities"],
                    },
                )

                context["documents"].add(result["document"])
                for entity in result["entities"]:
                    context["entities"].add(entity["id"])

            return {
                "query": context["query"],
                "chunks": context["chunks"],
                "entities": list(context["entities"]),
                "documents": list(context["documents"]),
            }

        except Exception as e:
            logging.error(f"Error getting RAG context: {e}")
            return {"query": query, "chunks": [], "entities": [], "documents": []}
