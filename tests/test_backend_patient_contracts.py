"""Patient extraction and bulk ingestion with real creator initialization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from ontographrag.kg.builders import ontology_guided_kg_creator as base
from ontographrag.kg.builders.enhanced_kg_creator import (
    PatientDemographics,
    PatientEntity,
    PatientExtraction,
    SocialHistory,
    UnifiedOntologyGuidedKGCreator,
)


@pytest.fixture
def creator(monkeypatch: pytest.MonkeyPatch) -> UnifiedOntologyGuidedKGCreator:
    embedding: Mock = Mock()
    embedding.embed_documents.side_effect = lambda texts: [
        [1.0, 0.0, 0.0] for _ in texts
    ]
    monkeypatch.setattr(base, "load_embedding_model", Mock(return_value=(embedding, 3)))
    return UnifiedOntologyGuidedKGCreator(
        strict_ontology=False, enable_soft_entity_linking=False
    )


def test_patient_demographics_and_hybrid_graph(
    creator: UnifiedOntologyGuidedKGCreator,
) -> None:
    text: str = """Name: Synthetic Person
Unit No: U1
Date of Birth: 1980-01-01
Admission Date: 2026-01-01
Discharge Date: 2026-01-02
Sex: F
Service: Unit
Attending: Doctor
PAST MEDICAL HISTORY:
1. Hypertension
2. No disease
CHIEF COMPLAINT:
- abdominal distension
HISTORY OF PRESENT ILLNESS:
ascites due to portal hypertension
DISCHARGE DIAGNOSIS:
ascites
MEDICATIONS ON ADMISSION:
1. Aspirin 10 mg
DISCHARGE MEDICATIONS:
1. Medication 5 mg
BRIEF HOSPITAL COURSE:
Furosemide 20 mg
PERTINENT RESULTS:
Sodium: 140
Alanine(ALT): 45
SOCIAL HISTORY:
former smoker quit alcohol and drug use
"""
    demographics: PatientDemographics = creator._extract_patient_demographics(text)
    assert demographics == {
        "name": "Synthetic Person",
        "unit_no": "U1",
        "date_of_birth": "1980-01-01",
        "admission_date": "2026-01-01",
        "discharge_date": "2026-01-02",
        "sex": "F",
        "service": "Unit",
        "attending": "Doctor",
    }
    result: PatientExtraction = (
        creator._extract_entities_and_relationships_for_patients(text)
    )
    types: set[str] = {e["type"] for e in result["entities"]}
    assert types == {
        "Patient",
        "MedicalCondition",
        "Medication",
        "LabTest",
        "SocialFactor",
    }
    assert {r["type"] for r in result["relationships"]} == {
        "HAS_CONDITION",
        "TAKES_MEDICATION",
        "HAS_LAB_RESULT",
        "HAS_SOCIAL_HISTORY",
    }
    patient: PatientEntity = next(
        e for e in result["entities"] if e["type"] == "Patient"
    )
    assert all(r["source"] == patient["id"] for r in result["relationships"])
    assert creator._extract_entities_and_relationships_for_patients("") == {
        "entities": [],
        "relationships": [],
    }
    assert creator._extract_section(text, "Absent") == ""


@pytest.mark.parametrize(
    "text,status",
    [
        ("current smoker alcohol drug", "current_smoker"),
        ("smoker", "unknown"),
        ("former smoker", "former_smoker"),
    ],
)
def test_social_history_states(
    creator: UnifiedOntologyGuidedKGCreator, text: str, status: str
) -> None:
    result: SocialHistory = creator._extract_social_history("Social History:\n" + text)
    assert result["smoking_status"] == status
    if "alcohol" in text:
        assert result["alcohol_use"] == "present"
        assert result["drug_use"] == "history"


def test_patient_graph_generation_chunk_limit(
    creator: UnifiedOntologyGuidedKGCreator,
) -> None:
    result = creator.generate_patient_knowledge_graph(
        "Name: Synthetic Person\nChief Complaint: prolonged headache",
        file_name="synthetic.txt",
        max_chunks=1,
    )
    assert len(result["chunks"]) == 1
    assert result["metadata"]["file_name"] == "synthetic.txt"
    assert result["metadata"]["total_entities"] == len(result["nodes"])
    assert any(n["type"] == "Patient" for n in result["nodes"])


def test_bulk_documents_partial_failure_and_batch_accounting(
    creator: UnifiedOntologyGuidedKGCreator,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path: Path = tmp_path / "docs.csv"
    path.write_text("id,text,source\na,alpha,A\nb,beta,B\nc,gamma,C\n")
    generate: Mock = Mock(
        side_effect=[
            {"nodes": [], "relationships": [], "chunks": [{"id": "a"}]},
            RuntimeError("failed document"),
            {"nodes": [], "relationships": [], "chunks": [{"id": "c"}]},
        ]
    )
    monkeypatch.setattr(creator, "generate_knowledge_graph", generate)
    result = creator.bulk_process_documents(
        str(path), "text", id_column="id", batch_size=2, kg_name="unit"
    )
    assert result["metadata"]["total_documents_processed"] == 3
    assert result["metadata"]["total_knowledge_graphs"] == 2
    assert result["metadata"]["bulk_processing_info"]["total_batches"] == 2
    assert result["knowledge_graph"]["chunks"] == [{"id": "a"}, {"id": "c"}]
    assert generate.call_args.kwargs["doc_metadata"] == {"source": "C"}
    assert generate.call_args.kwargs["kg_name"] == "unit"
    with pytest.raises(ValueError, match="CSV validation failed"):
        creator.bulk_process_documents(str(path), "absent")


def test_creator_numeric_configuration_fallbacks(
    creator: UnifiedOntologyGuidedKGCreator,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in [
        "RETRIEVAL_CHUNK_SIZE",
        "RETRIEVAL_CHUNK_OVERLAP",
        "KG_MIN_TRIPLE_CONFIDENCE",
        "KG_RELATIONSHIP_TYPE_SIMILARITY_THRESHOLD",
        "KG_LOW_CONFIDENCE_REVERIFY_THRESHOLD",
        "KG_SOFT_ENTITY_SIMILARITY_THRESHOLD",
        "KG_FRAGMENTATION_BRIDGE_SIMILARITY_THRESHOLD",
        "KG_MAX_FRAGMENTATION_BRIDGES",
        "KG_MAX_SUMMARY_ENTITIES",
        "KG_MAX_SUMMARY_RELATIONSHIPS",
    ]:
        monkeypatch.setenv(name, "invalid")
    configured: UnifiedOntologyGuidedKGCreator = UnifiedOntologyGuidedKGCreator()
    assert configured.retrieval_chunk_size == 256
    assert configured.retrieval_chunk_overlap == 64
    assert configured.min_triple_confidence == 0.15
    assert configured.max_summary_entities == 6
    with pytest.raises(FileNotFoundError):
        UnifiedOntologyGuidedKGCreator(ontology_path=str(tmp_path / "missing.json"))
