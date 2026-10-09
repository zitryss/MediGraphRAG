"""Helpers for deciding whether an existing dataset KG can be safely reused."""

from typing import TYPE_CHECKING, NotRequired

from typing_extensions import ReadOnly, TypedDict

if TYPE_CHECKING:
    from experiments.dataset_adapters import JSONValue


class ExistingKGMetadata(TypedDict, total=False):
    """Unvalidated stored values; normalization remains in the comparison."""

    uses_global_corpus: ReadOnly[object]
    corpus_source: ReadOnly[object]
    question_context_role: ReadOnly[object]
    dataset_kg_scope: ReadOnly[object]
    content_hash: ReadOnly[object]
    kg_build_fingerprint_version: ReadOnly[object]
    kg_builder: ReadOnly[object]
    kg_chunk_size: ReadOnly[object]
    kg_chunk_overlap: ReadOnly[object]
    kg_extraction_provider: ReadOnly[object]
    kg_extraction_model: ReadOnly[object]
    kg_embedding_provider: ReadOnly[object]
    selection_key: ReadOnly[object]


class ExpectedKGMetadata(TypedDict, total=False):
    usesGlobalCorpus: ReadOnly[object]
    corpusSource: ReadOnly[object]
    questionContextRole: ReadOnly[object]
    datasetKgScope: ReadOnly[object]
    contentHash: ReadOnly[object]
    kgBuildFingerprintVersion: ReadOnly[object]
    kgBuilder: ReadOnly[object]
    kgChunkSize: ReadOnly[object]
    kgChunkOverlap: ReadOnly[object]
    kgExtractionProvider: ReadOnly[object]
    kgExtractionModel: ReadOnly[object]
    kgEmbeddingProvider: ReadOnly[object]
    selectionKey: ReadOnly[object]


class ProducedKGMetadata(ExpectedKGMetadata):
    """Expected fields emitted by the dataset build, not stored legacy values."""

    usesGlobalCorpus: ReadOnly[bool]
    corpusSource: ReadOnly[str]
    questionContextRole: ReadOnly[str]
    datasetKgScope: ReadOnly[str]
    contentHash: NotRequired[ReadOnly[str]]
    kgBuildFingerprintVersion: ReadOnly[int]
    kgBuilder: ReadOnly[str]
    kgChunkSize: ReadOnly[int]
    kgChunkOverlap: ReadOnly[int]
    kgExtractionProvider: ReadOnly[str]
    kgExtractionModel: ReadOnly[str]
    kgEmbeddingProvider: ReadOnly[str]
    selectionKey: ReadOnly["JSONValue"]


def assess_dataset_kg_compatibility(
    existing_meta: ExistingKGMetadata,
    expected_meta: ExpectedKGMetadata,
    *,
    evaluation_subset_scope: str = "evaluation_subset",
) -> tuple[bool, list[str]]:
    """Compare stored KG metadata against the current expected build contract."""
    if not existing_meta or not expected_meta:
        return True, []

    reasons: list[str] = []

    def _compare(
        existing_key: str,
        expected_key: str,
        label: str,
        *,
        normalizer: type[int] | type[bool] | None = None,
    ) -> None:
        existing_value: object = existing_meta.get(existing_key)
        expected_value: object = expected_meta.get(expected_key)
        if existing_value in (None, "") or expected_value in (None, ""):
            return
        if normalizer is not None:
            existing_value = normalizer(existing_value)
            expected_value = normalizer(expected_value)
        if existing_value != expected_value:
            reasons.append(
                f"{label} mismatch (stored={existing_value!r}, expected={expected_value!r})",
            )

    _compare(
        "uses_global_corpus",
        "usesGlobalCorpus",
        "usesGlobalCorpus",
        normalizer=bool,
    )
    _compare("corpus_source", "corpusSource", "corpusSource")
    _compare("question_context_role", "questionContextRole", "questionContextRole")
    _compare("dataset_kg_scope", "datasetKgScope", "datasetKgScope")
    _compare("content_hash", "contentHash", "contentHash")
    _compare(
        "kg_build_fingerprint_version",
        "kgBuildFingerprintVersion",
        "kgBuildFingerprintVersion",
        normalizer=int,
    )
    _compare("kg_builder", "kgBuilder", "kgBuilder")
    _compare("kg_chunk_size", "kgChunkSize", "kgChunkSize", normalizer=int)
    _compare("kg_chunk_overlap", "kgChunkOverlap", "kgChunkOverlap", normalizer=int)
    _compare("kg_extraction_provider", "kgExtractionProvider", "kgExtractionProvider")
    _compare("kg_extraction_model", "kgExtractionModel", "kgExtractionModel")
    _compare("kg_embedding_provider", "kgEmbeddingProvider", "kgEmbeddingProvider")

    if expected_meta.get("datasetKgScope") == evaluation_subset_scope and not bool(
        expected_meta.get("usesGlobalCorpus"),
    ):
        _compare("selection_key", "selectionKey", "selectionKey")

    return len(reasons) == 0, reasons
