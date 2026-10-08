"""
Regression tests for KG reuse compatibility checks.

Pure unit tests — no live Neo4j or dataset loading required.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.kg_reuse import (
    ExistingKGMetadata,
    ExpectedKGMetadata,
    assess_dataset_kg_compatibility,
)

DATASET_KG_SCOPE_EVALUATION_SUBSET: str = "evaluation_subset"
DATASET_KG_SCOPE_FULL_DATASET: str = "full_dataset"


class TestDatasetKGCompatibility:
    def test_shared_corpus_ignores_selection_key_mismatch(self) -> None:
        existing_meta: ExistingKGMetadata = {
            "uses_global_corpus": True,
            "dataset_kg_scope": DATASET_KG_SCOPE_EVALUATION_SUBSET,
            "selection_key": "old-selection",
            "content_hash": "same-hash",
        }
        expected_meta: ExpectedKGMetadata = {
            "usesGlobalCorpus": True,
            "datasetKgScope": DATASET_KG_SCOPE_EVALUATION_SUBSET,
            "selectionKey": "new-selection",
            "contentHash": "same-hash",
        }

        compatible: bool
        reasons: list[str]
        compatible, reasons = assess_dataset_kg_compatibility(
            existing_meta,
            expected_meta,
        )

        assert compatible is True
        assert reasons == []

    def test_subset_scoped_selection_key_mismatch_forces_rebuild(self) -> None:
        existing_meta: ExistingKGMetadata = {
            "uses_global_corpus": False,
            "dataset_kg_scope": DATASET_KG_SCOPE_EVALUATION_SUBSET,
            "selection_key": "subset-a",
        }
        expected_meta: ExpectedKGMetadata = {
            "usesGlobalCorpus": False,
            "datasetKgScope": DATASET_KG_SCOPE_EVALUATION_SUBSET,
            "selectionKey": "subset-b",
        }

        compatible: bool
        reasons: list[str]
        compatible, reasons = assess_dataset_kg_compatibility(
            existing_meta,
            expected_meta,
        )

        assert compatible is False
        assert any("selectionKey mismatch" in reason for reason in reasons)

    def test_content_hash_mismatch_forces_rebuild(self) -> None:
        existing_meta: ExistingKGMetadata = {
            "uses_global_corpus": False,
            "dataset_kg_scope": DATASET_KG_SCOPE_FULL_DATASET,
            "content_hash": "old-hash",
        }
        expected_meta: ExpectedKGMetadata = {
            "usesGlobalCorpus": False,
            "datasetKgScope": DATASET_KG_SCOPE_FULL_DATASET,
            "contentHash": "new-hash",
        }

        compatible: bool
        reasons: list[str]
        compatible, reasons = assess_dataset_kg_compatibility(
            existing_meta,
            expected_meta,
        )

        assert compatible is False
        assert any("contentHash mismatch" in reason for reason in reasons)

    def test_missing_metadata_does_not_force_rebuild(self) -> None:
        existing_meta: ExistingKGMetadata = {
            "uses_global_corpus": False,
            "dataset_kg_scope": DATASET_KG_SCOPE_EVALUATION_SUBSET,
        }
        expected_meta: ExpectedKGMetadata = {
            "usesGlobalCorpus": False,
            "datasetKgScope": DATASET_KG_SCOPE_EVALUATION_SUBSET,
            "selectionKey": "subset-a",
            "contentHash": "hash-a",
        }

        compatible: bool
        reasons: list[str]
        compatible, reasons = assess_dataset_kg_compatibility(
            existing_meta,
            expected_meta,
        )

        assert compatible is True
        assert reasons == []
