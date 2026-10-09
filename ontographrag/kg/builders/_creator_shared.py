"""Shared constants and helpers for the KG creator modules."""

import re
from typing import Generic, NotRequired, Protocol, TypeAlias, TypedDict, TypeVar

from typing_extensions import ReadOnly

# Minimum character length for an entity name (after stripping whitespace).
# Single-character or two-character tokens are almost always stop-words or
# artefacts; named identifiers below this threshold (e.g. "p53", "IL-6") are
# kept via the allowlist below.
_ENTITY_MIN_NAME_LENGTH: int = 3

# Lowercase terms that must not become standalone KG nodes regardless of how
# the LLM classifies them.  These are generic process/state words that appear
# in almost every biomedical chunk and, when extracted as entities, become hub
# nodes fanning out to hundreds of irrelevant chunks during retrieval.
_GENERIC_HUB_ENTITY_BLOCKLIST: frozenset[str] = frozenset(
    {
        "treatment",
        "treatments",
        "condition",
        "conditions",
        "outcome",
        "outcomes",
        "model",
        "models",
        "effect",
        "effects",
        "result",
        "results",
        "factor",
        "factors",
        "mechanism",
        "mechanisms",
        "response",
        "responses",
        "function",
        "functions",
        "role",
        "roles",
        "process",
        "processes",
        "level",
        "levels",
        "system",
        "systems",
        "study",
        "studies",
        "analysis",
        "analyses",
        "approach",
        "approaches",
        "method",
        "methods",
        "measure",
        "measures",
        "group",
        "groups",
        "sample",
        "samples",
        "data",
        "finding",
        "findings",
        "evidence",
        "activity",
        "activities",
        "expression",
        "expressions",
        "production",
        "change",
        "changes",
        "increase",
        "increases",
        "decrease",
        "decreases",
        "type",
        "types",
        "form",
        "forms",
        "stage",
        "stages",
        "state",
        "states",
        "case",
        "cases",
        "patient",
        "patients",
        "subject",
        "subjects",
        "participant",
        "participants",
        "control",
        "controls",
        "target",
        "targets",
        "interaction",
        "interactions",
        "pathway",
        "pathways",
        "network",
        "networks",
        "signal",
        "signals",
        "marker",
        "markers",
        "indicator",
        "indicators",
        "test",
        "tests",
        "assessment",
        "assessments",
        "evaluation",
        "evaluations",
        "trial",
        "trials",
        "experiment",
        "experiments",
        "observation",
        "observations",
        "report",
        "reports",
        "review",
        "reviews",
        "context",
        "information",
        "knowledge",
        "concept",
        "concepts",
        "feature",
        "features",
        "aspect",
        "aspects",
        "component",
        "components",
        "element",
        "elements",
        "structure",
        "structures",
        "property",
        "properties",
        "characteristic",
        "characteristics",
        "parameter",
        "parameters",
        "variable",
        "variables",
        "value",
        "values",
        "score",
        "scores",
        "rate",
        "rates",
        "ratio",
        "ratios",
        "index",
        "indices",
        "mean",
        "median",
        "range",
        "prevalence",
        "incidence",
        "frequency",
        "proportion",
        "percentage",
        "number",
        "amount",
        "quantity",
        "duration",
        "period",
        "time",
        "age",
        "size",
        "dose",
        "concentration",
        "threshold",
        "limit",
        "maximum",
        "minimum",
        # Bare determiners/pronouns that sometimes slip through
        "this",
        "that",
        "these",
        "those",
        "which",
        "what",
        "who",
        "when",
        "where",
        "how",
        "why",
        "other",
        "another",
        "same",
        "different",
        "various",
        "several",
        "many",
        "few",
        "all",
        "both",
        "each",
        "every",
        "some",
        "any",
        "no",
        "not",
        "yes",
    },
)


def _is_valid_entity_name(name: str) -> bool:
    """Return True when a name is specific enough to be a KG node.

    Rejects names that are:
    - shorter than _ENTITY_MIN_NAME_LENGTH (after stripping whitespace), unless
      they match the short-identifier pattern (digits/letters mixed, e.g. p53, IL-6)
    - in the generic hub-entity blocklist (case-insensitive, singular/plural)
    - bare numeric / punctuation fragments without any alphabetic referent
    """
    stripped: str = name.strip()
    if not stripped:
        return False
    normalized: str = stripped.lower()
    # Reject punctuation-only fragments or bare numeric tokens. Legitimate short
    # biomedical identifiers such as "p53" or "IL-6" include alphabetic
    # characters and therefore bypass this guard.
    if not re.search(r"[A-Za-z0-9]", stripped):
        return False
    if re.fullmatch(r"[\d\W_]+", stripped):
        return False
    # Blocklist check
    if normalized in _GENERIC_HUB_ENTITY_BLOCKLIST:
        return False
    # Length check: allow short named identifiers (alphanumeric + hyphens/dots)
    if len(stripped) < _ENTITY_MIN_NAME_LENGTH:
        # Allow if it looks like a named identifier: has a digit or a hyphen
        # (e.g. "p53" is 3 chars and passes length anyway; "IL" is 2 and fails
        # unless it has a hyphen such as "IL-6").
        import re as _re

        if not _re.search(r"[\d\-\.]", stripped):
            return False
    return True


def _env_flag(value: str | None, default: bool) -> bool:
    """Parse a boolean-like environment value safely."""
    if value is None:
        return default
    normalized: str = str(value).strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


AnchorOffset: TypeAlias = int | float | str | None


class AnchorSpanInput(TypedDict, total=False):
    text: ReadOnly[object]
    start: ReadOnly[AnchorOffset]
    end: ReadOnly[AnchorOffset]
    chunk_position: ReadOnly[AnchorOffset]
    local_start: ReadOnly[AnchorOffset]
    local_end: ReadOnly[AnchorOffset]


class AnchorSpan(TypedDict):
    text: ReadOnly[str]
    start: ReadOnly[AnchorOffset]
    end: ReadOnly[AnchorOffset]
    chunk_position: ReadOnly[AnchorOffset]
    local_start: ReadOnly[AnchorOffset]
    local_end: ReadOnly[AnchorOffset]


class ExactAnchorSpan(AnchorSpan):
    start: ReadOnly[int]
    end: ReadOnly[int]
    chunk_position: ReadOnly[int | None]
    local_start: ReadOnly[int]
    local_end: ReadOnly[int]


AnchorGrounding: TypeAlias = dict[str, list[AnchorSpan]]


class RestorationRecord(TypedDict):
    status: str
    verified: bool
    grounded_components: list[str]
    grounded_count: int
    anchor_grounding: NotRequired[AnchorGrounding]


class EntityAnchor(TypedDict):
    text: str
    type: str
    anchor_spans: list[ExactAnchorSpan]


class RelationAnchor(TypedDict):
    text: str
    type_hint: str
    anchor_spans: list[ExactAnchorSpan]


class AttributeAnchor(TypedDict):
    text: str
    role: str
    anchor_spans: list[ExactAnchorSpan]


class AnchorInventory(TypedDict):
    entity_anchors: list[EntityAnchor]
    relation_anchors: list[RelationAnchor]
    attribute_anchors: list[AttributeAnchor]


GroupEntity = TypeVar("GroupEntity")


class PassageGroup(TypedDict, Generic[GroupEntity]):
    group_key: tuple[str, str]
    texts: list[str]
    entities: dict[int, list[GroupEntity]]
    positions: dict[int, list[int]]
    source_label: str


# These aliases describe decoded JSON only, not normalized graph records or
# arbitrary application properties (which may contain non-JSON Python values).
DecodedJSONValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["DecodedJSONValue"]
    | dict[str, "DecodedJSONValue"]
)
DecodedJSONObject: TypeAlias = dict[str, DecodedJSONValue]


class RetrievalSubchunk(TypedDict):
    id: str
    text: str
    embedding: list[float] | None
    retrieval_local_index: int
    start_pos: int
    end_pos: int
    position: int
    source: str | None
    dataset: str | None
    question_id: str | None
    passage_index: int | None
    chunk_local_index: int
    parent_chunk_id: str


class ContextPassageInput(Protocol):
    """Core passage fields; optional provenance is accessed through getattr."""

    @property
    def text(self) -> str: ...

    @property
    def dataset(self) -> str: ...

    @property
    def question_id(self) -> str: ...

    @property
    def passage_index(self) -> int: ...
