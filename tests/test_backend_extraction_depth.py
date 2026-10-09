"""Real schema import and extraction with deterministic LLM responses."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast
from unittest.mock import Mock

import pytest

from ontographrag.kg.builders import ontology_guided_kg_creator as module
from ontographrag.kg.builders._creator_shared import (
    AnchorInventory,
    AttributeAnchor,
    DecodedJSONObject,
    EntityAnchor,
    RelationAnchor,
)
from ontographrag.kg.builders._storage import Neo4jStoredValue, StorageMixin
from ontographrag.kg.builders.ontology_guided_kg_creator import (
    CreatorRagContext,
    ExtractionResult,
    GraphChunkInput,
)
from ontographrag.schemas.models import PropertyType
from tests.conftest import required_text_field
from tests.known_defects import UndefinedProviderDefect

if TYPE_CHECKING:
    from ontographrag.kg.builders._ontology import SchemaCard
    from ontographrag.kg.builders.ontology_guided_kg_creator import KnowledgeGraph
    from ontographrag.schemas.models import EntityType, OntologySchema


class ReverificationOptions(TypedDict):
    source_name: str
    target_name: str
    relationship_type: str
    verification_chunks: list[GraphChunkInput]
    llm: Mock
    model_name: str


@pytest.fixture
def creator(monkeypatch: pytest.MonkeyPatch) -> module.OntologyGuidedKGCreator:
    monkeypatch.setattr(module, "load_embedding_model", Mock(return_value=(Mock(), 3)))
    return module.OntologyGuidedKGCreator(
        strict_ontology=False,
        enable_soft_entity_linking=False,
    )


@pytest.fixture
def schema_file(tmp_path: Path) -> Path:
    path: Path = tmp_path / "schema.json"
    path.write_text(
        json.dumps(
            {
                "entity_types": [
                    {
                        "id": "Drug",
                        "description": "medicine",
                        "properties": [
                            {
                                "name": "code",
                                "type": "id",
                                "identifier": True,
                                "required": True,
                            },
                            {
                                "name": "status",
                                "type": "enum",
                                "values": ["active", "inactive"],
                            },
                            {},
                        ],
                    },
                    {"id": "Disease"},
                    "invalid",
                    {},
                ],
                "relationship_types": [
                    {
                        "id": "TREATS",
                        "domain": "Drug",
                        "range": "Disease",
                        "cardinality": "many-to-many",
                        "attributes": [
                            {"name": "dose", "type": "float", "unit": "mg"},
                            {},
                        ],
                    },
                    {},
                ],
            },
        ),
    )
    return path


def test_json_schema_load_and_hash_card(
    creator: module.OntologyGuidedKGCreator,
    schema_file: Path,
) -> None:
    creator.ontology_path = str(schema_file)
    creator._load_ontology(str(schema_file))
    schema: OntologySchema | None = creator._ontology_schema
    assert schema is not None
    assert [e.id for e in schema.entity_types] == ["Drug", "Disease"]
    assert schema.entity_types[0].properties[0].type == PropertyType.ID
    assert schema.relationship_types[0].attributes[0].unit == "mg"
    card: SchemaCard = creator._build_schema_card()
    assert (
        card["ontologyFileHash"] == hashlib.sha256(schema_file.read_bytes()).hexdigest()
    )
    assert card["classCount"] == 2 and card["relationshipCount"] == 1
    assert "relationshipTypes" in card
    assert "domain" in card["relationshipTypes"][0]
    assert card["relationshipTypes"][0]["domain"] == "Drug"
    assert creator._match_ontology_entity_type("diseas") == "Disease"
    assert creator._match_ontology_entity_type("unknown", allow_fuzzy=False) is None
    assert creator._coerce_entity_type_with_ontology("Drug") == "Drug"
    assert creator._coerce_entity_type_with_ontology("off_schema") is None
    assert "TREATS" in creator._build_ontology_few_shot_examples()
    prompt: str = creator._build_ontology_prompt_section(
        "Aspirin is a Drug treating Disease",
    )
    assert "Drug" in prompt and "dose" in prompt


def test_owl_typed_properties_and_relationship_attributes(
    creator: module.OntologyGuidedKGCreator,
    tmp_path: Path,
) -> None:
    path: Path = tmp_path / "schema.owl"
    path.write_text("""<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" xmlns:owl="http://www.w3.org/2002/07/owl#" xmlns:app="urn:test:">
    <owl:Class rdf:about="urn:test#Drug"><rdfs:label>Drug</rdfs:label><rdfs:comment>Medicine</rdfs:comment></owl:Class>
    <owl:Class rdf:about="urn:test#Disease"/><owl:Class/>
    <owl:DatatypeProperty rdf:about="urn:test#code"><rdfs:domain rdf:resource="urn:test#Drug"/><app:propertyType>id</app:propertyType><app:isIdentifier>true</app:isIdentifier></owl:DatatypeProperty>
    <owl:DatatypeProperty rdf:about="urn:test#mass"><rdfs:domain rdf:resource="urn:test#Drug"/><rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#float"/><app:unit>mg</app:unit></owl:DatatypeProperty>
    <owl:DatatypeProperty rdf:about="urn:test#dose"><app:relationshipAttributeOf>TREATS</app:relationshipAttributeOf><app:attributeType>float</app:attributeType></owl:DatatypeProperty>
    <owl:DatatypeProperty rdf:about="urn:test#orphan"/><owl:DatatypeProperty/>
    <owl:ObjectProperty rdf:about="urn:test#TREATS"><rdfs:domain rdf:resource="urn:test#Drug"/><rdfs:range rdf:resource="urn:test#Disease"/><app:cardinality>many-to-many</app:cardinality></owl:ObjectProperty><owl:ObjectProperty/>
    </rdf:RDF>""")
    creator._load_ontology(str(path))
    schema: OntologySchema | None = creator._ontology_schema
    assert schema is not None
    drug: EntityType = schema.entity_types[0]
    assert drug.description == "Medicine"
    assert [(p.name, p.type) for p in drug.properties] == [
        ("code", PropertyType.ID),
        ("mass", PropertyType.FLOAT),
    ]
    assert schema.relationship_types[0].attributes[0].name == "dose"
    assert (
        creator._canonicalize_relationship_type(
            "treats",
            source_type="Drug",
            target_type="Disease",
        )
        == "TREATS"
    )


@pytest.mark.parametrize(
    "response,expected",
    [
        ("YES supported", True),
        ("NO contradicted", False),
        ("uncertain", None),
        (RuntimeError("timeout"), None),
    ],
)
def test_reverification_caches_only_explicit_decisions(
    creator: module.OntologyGuidedKGCreator,
    response: str | RuntimeError,
    expected: bool | None,
) -> None:
    llm: Mock = Mock()
    if isinstance(response, Exception):
        llm.generate.side_effect = response
    else:
        llm.generate.return_value = response
    options: ReverificationOptions = {
        "source_name": "aspirin",
        "target_name": "pain",
        "relationship_type": "TREATS",
        "verification_chunks": [{"text": "Aspirin treats pain."}],
        "llm": llm,
        "model_name": "unit",
    }
    assert creator._reverify_low_confidence_triple(**options) is expected
    assert creator._reverify_low_confidence_triple(**options) is expected
    assert llm.generate.call_count == (1 if expected is not None else 2)
    assert "Aspirin treats pain." in llm.generate.call_args.args[0]
    empty_options: ReverificationOptions = {**options, "verification_chunks": []}
    assert creator._reverify_low_confidence_triple(**empty_options) is None


@pytest.mark.parametrize(
    "response",
    [
        "These treated cells survived for twenty days.",
        "short",
        "x" * 1000,
        RuntimeError("provider failed"),
    ],
)
def test_coreference_plausibility_gate(
    creator: module.OntologyGuidedKGCreator,
    response: str | RuntimeError,
) -> None:
    original: str = "These cells survived during treatment."
    llm: Mock = Mock()
    if isinstance(response, Exception):
        llm.generate.side_effect = response
    else:
        llm.generate.return_value = response
    result: str = creator._resolve_coreferences_llm(
        original,
        "Treated cells received aspirin.",
        llm,
        "unit",
    )
    assert result == (
        response
        if isinstance(response, str) and 20 <= len(response) <= len(original) * 3
        else original
    )
    count: int = llm.generate.call_count
    assert creator._resolve_coreferences_llm(original, " ", llm, "unit") == original
    assert llm.generate.call_count == count


@pytest.mark.parametrize("schema", [False, True])
def test_real_llm_extraction_and_grounded_postprocessing(
    creator: module.OntologyGuidedKGCreator,
    schema_file: Path,
    schema: bool,
) -> None:
    if schema:
        creator._load_ontology(str(schema_file))
    creator.enable_anchor_constrained_extraction = False
    payload: dict[str, list[dict[str, str | dict[str, str]]]] = {
        "entities": [
            {"id": "Aspirin", "type": "Drug", "properties": {"name": "Aspirin"}},
            {"id": "Pain", "type": "Disease", "properties": {"name": "Pain"}},
        ],
        "relationships": [
            {
                "source": "Aspirin",
                "target": "Pain",
                "type": "TREATS",
                "properties": {"anchor_text": "treats"},
            },
        ],
    }
    llm: Mock = Mock(generate=Mock(return_value=json.dumps(payload)))
    result: ExtractionResult = creator._extract_entities_and_relationships_with_llm(
        "Aspirin treats Pain.",
        llm,
        model_name="unit",
        context_header="Treated cells",
        section_header="Results",
    )
    assert {required_text_field(e, "id") for e in result["entities"]} == {
        "Aspirin",
        "Pain",
    }
    assert "relationships" in result
    assert "type" in result["relationships"][0]
    assert result["relationships"][0]["type"] == "TREATS"
    prompt: str = llm.generate.call_args_list[0].args[0]
    assert "Results" in prompt and "Treated cells" in prompt
    llm.generate.side_effect = TimeoutError("read operation timed out")
    assert creator._extract_entities_and_relationships_with_llm(
        "Aspirin treats Pain.",
        llm,
    ) == {"entities": [], "relationships": []}


@pytest.mark.parametrize(
    "response",
    [
        '```json\n{"value": "brace } and \\"quote\\""}\n```',
        'prefix {"value": 3} suffix',
        '{"entities":[{"id":"Aspirin",}], BROKEN}',
    ],
)
def test_json_recovery_preserves_valid_content(
    creator: module.OntologyGuidedKGCreator,
    response: str,
) -> None:
    result: DecodedJSONObject = creator._parse_llm_json_response(
        response,
        allow_partial_extraction=True,
    )
    assert result.get("value") is not None or result.get("entities") == [
        {"id": "Aspirin"},
    ]


@pytest.mark.xfail(
    strict=True,
    raises=UndefinedProviderDefect,
    reason="generate_knowledge_graph references undefined provider while assembling metadata",
)
def test_real_preview_build_completes_without_persistence(
    creator: module.OntologyGuidedKGCreator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_embedding_function(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(
        creator,
        "embedding_function",
        Mock(
            embed_query=Mock(return_value=[1.0, 0.0, 0.0]),
            embed_documents=Mock(side_effect=fake_embedding_function),
        ),
    )
    creator.enable_anchor_constrained_extraction = False
    llm: Mock = Mock(
        generate=Mock(
            return_value=json.dumps(
                {
                    "entities": [
                        {"id": "Aspirin", "type": "Drug"},
                        {"id": "Pain", "type": "Disease"},
                    ],
                    "relationships": [
                        {"source": "Aspirin", "target": "Pain", "type": "TREATS"},
                    ],
                },
            ),
        ),
    )
    try:
        result: KnowledgeGraph = creator.generate_knowledge_graph(
            "Aspirin treats Pain.",
            llm,
            kg_name="unit",
            max_chunks=1,
        )
    except NameError as exc:
        if exc.name == "provider" and str(exc) == "name 'provider' is not defined":
            raise UndefinedProviderDefect(str(exc)) from exc
        raise
    assert "metadata" in result
    assert "total_entities" in result["metadata"]
    assert result["metadata"]["total_entities"] == 2
    assert "stored_in_neo4j" not in result["metadata"]


def test_total_extraction_failure_cannot_claim_success(
    creator: module.OntologyGuidedKGCreator,
) -> None:
    creator.enable_anchor_constrained_extraction = False
    with pytest.raises(RuntimeError, match="all 1 chunk"):
        creator.generate_knowledge_graph(
            "Aspirin treats Pain.",
            Mock(generate=Mock(side_effect=TimeoutError("timeout"))),
            max_chunks=1,
        )


def test_schema_embedding_classification_and_fallback(
    creator: module.OntologyGuidedKGCreator,
    monkeypatch: pytest.MonkeyPatch,
    schema_file: Path,
) -> None:
    creator._load_ontology(str(schema_file))
    creator._ontology_class_embeddings = [("Drug", [1, 0]), ("Disease", [0, 1])]
    embedding: Mock = Mock(embed_query=Mock(return_value=[0, 1]))
    monkeypatch.setattr(creator, "embedding_function", embedding)
    assert creator._classify_entity_with_ontology("unfamiliar syndrome") == "Disease"
    embedding.embed_query.side_effect = RuntimeError("embedding unavailable")
    assert creator._classify_entity_with_ontology("Drug candidate") == "Drug"
    assert creator._classify_relationship_with_ontology("Drug", "Disease") == "TREATS"


def test_anchor_inventory_only_retains_exact_grounded_spans(
    creator: module.OntologyGuidedKGCreator,
    schema_file: Path,
) -> None:
    creator._load_ontology(str(schema_file))
    text: str = "Aspirin treats Pain at 5 mg."
    inventory: AnchorInventory = creator._normalize_anchor_inventory(
        {
            "entity_anchors": [
                {"text": "Aspirin", "type": "Drug"},
                {"text": "Aspirin", "type": "Drug"},
                {"text": "Hallucinated", "type": "Drug"},
                {"text": "Pain", "type": "Disease"},
                1,
            ],
            "relation_anchors": [
                "treats",
                {"text": "treats", "type_hint": "TREATS"},
                "imaginary",
                {},
            ],
            "attribute_anchors": [
                {"text": "5 mg", "role": "quantity"},
                {"text": "5 mg", "role": "quantity"},
                "imaginary",
                1,
            ],
        },
        chunk_text=text,
    )
    assert [a["text"] for a in inventory["entity_anchors"]] == ["Aspirin", "Pain"]
    assert (
        len(inventory["relation_anchors"]) == len(inventory["attribute_anchors"]) == 1
    )
    # TypedDict.values erases the finite union declared by AnchorInventory.
    for group in cast(
        Iterable[Sequence[EntityAnchor | RelationAnchor | AttributeAnchor]],
        inventory.values(),
    ):
        for anchor in group:
            for span in anchor["anchor_spans"]:
                assert text[span["start"] : span["end"]] == span["text"]


def test_storage_connection_and_vector_index_contract(
    creator: module.OntologyGuidedKGCreator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ontographrag.kg.builders import _storage

    graph: Mock = Mock(query=Mock(return_value=[]))
    factory: Mock = Mock(return_value=graph)
    monkeypatch.setattr(_storage, "Neo4jGraph", factory)
    for name in ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"]:
        monkeypatch.setenv(
            name,
            "before",
        )  # restoration covers legacy direct os.environ writes
    creator.neo4j_password = ""
    assert creator._create_neo4j_connection() is graph
    assert factory.call_args.kwargs["password"] == "before"
    creator._create_vector_indexes(graph)
    statements: list[str] = [call.args[0] for call in graph.query.call_args_list]
    assert any("unique_document_filename_kgname" in query for query in statements)
    vectors: list[str] = [
        query for query in statements if "CREATE VECTOR INDEX" in query
    ]
    assert len(vectors) == 3 and all(
        "`vector.dimensions`: 3" in query for query in vectors
    )
    graph.query.side_effect = RuntimeError("constraint denied")
    creator._create_vector_indexes(graph)  # optional index setup logs and fails soft


def test_creator_retrieval_maps_unique_entities_and_errors(
    creator: module.OntologyGuidedKGCreator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph: Mock = Mock(
        query=Mock(
            return_value=[
                {
                    "text": "Evidence",
                    "chunk_id": "c",
                    "score": 0.7,
                    "document": "doc",
                    "entities": [{"id": "Aspirin"}, {"id": "Aspirin"}],
                },
            ],
        ),
    )
    monkeypatch.setattr(creator, "_create_neo4j_connection", Mock(return_value=graph))
    monkeypatch.setattr(
        creator,
        "embedding_function",
        Mock(embed_query=Mock(return_value=[1, 0, 0])),
    )
    result: CreatorRagContext = creator.get_rag_context("aspirin", top_k=2)
    assert result["entities"] == ["Aspirin"] and result["documents"] == ["doc"]
    assert graph.query.call_args.args[1] == {"top_k": 2, "query_vector": [1, 0, 0]}
    graph.query.side_effect = RuntimeError("read failure")
    assert creator.get_rag_context("aspirin")["chunks"] == []


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Regional cancer institute", "Disease"),
        ("University hospital", "Hospital"),
        ("TBK1 protein", "Biomarker"),
        ("Gleason score", "ClinicalFinding"),
    ],
)
def test_schema_free_classification_retains_domain_priority(
    creator: module.OntologyGuidedKGCreator,
    text: str,
    expected: str,
) -> None:
    # Explicit precedence: disease terms outrank location terms in ambiguous labels.
    assert creator._classify_entity_with_ontology(text) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (True, True),
        (3, 3),
        (1.5, 1.5),
        ("text", "text"),
        ([], []),
        ([1, "text", False], [1, "text", False]),
        ((1, 2), [1, 2]),
        ({7}, [7]),
        ([[1]], "[[1]]"),
        ({"nested": [2]}, '{"nested": [2]}'),
        ([float("nan")], "[NaN]"),
        (float("inf"), "inf"),
        (Path("relative"), "relative"),
    ],
)
def test_neo4j_property_coercion_contract(
    value: object,
    expected: Neo4jStoredValue | None,
) -> None:
    assert StorageMixin._coerce_neo4j_property_value(value) == expected


def test_neo4j_property_sanitization_drops_null_and_nonfinite() -> None:
    values: dict[str, object] = {
        "null": None,
        "nan": float("nan"),
        "infinity": float("inf"),
        "kept": [1, "text"],
        "nested": {"value": True},
    }
    assert StorageMixin._sanitize_neo4j_properties(values) == {
        "kept": [1, "text"],
        "nested": '{"value": true}',
    }
