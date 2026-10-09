import ast
import json
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import NotRequired, Protocol, TypeAlias, TypedDict, cast

from ontographrag.api.runtime_helpers import JSONValue
from ontographrag.kg.builders._ontology import SchemaCard
from ontographrag.schemas.models import (
    DataBinding,
    EntityType,
    OntologySchema,
    PropertyType,
    RelationshipAttribute,
    RelationshipType,
)


class SchemaPropertyFixture(TypedDict):
    name: str
    type: str
    identifier: NotRequired[bool]


class SchemaClassFixture(TypedDict):
    id: str
    properties: list[SchemaPropertyFixture]


SchemaRelationshipFixture = TypedDict(
    "SchemaRelationshipFixture",
    {"id": str, "from": str, "to": str, "cardinality": str},
)


class SchemaFixture(TypedDict):
    classes: list[SchemaClassFixture]
    relationships: list[SchemaRelationshipFixture]


class _OwlHost(Protocol):
    _prop_type: Callable[[str], PropertyType]


class _SchemaHost(Protocol):
    ontology_path: str | None
    ontology_classes: list[dict[str, str]]
    ontology_relationships: list[dict[str, str]]
    _ontology_schema: OntologySchema | None


_ValidateSchema: TypeAlias = Callable[[bytes, str], list[str]]
_PropType: TypeAlias = Callable[[str], PropertyType]
_LoadOwl: TypeAlias = Callable[[_OwlHost, str], OntologySchema]
_BuildCard: TypeAlias = Callable[[_SchemaHost], SchemaCard]


def _load_validate_ontology_schema() -> _ValidateSchema:
    """Load validate_ontology_schema from app.py without importing the whole app."""
    app_path: Path = (
        Path(__file__).resolve().parents[1] / "ontographrag" / "api" / "app.py"
    )
    source: str = app_path.read_text(encoding="utf-8")
    module: ast.Module = ast.parse(source, filename=str(app_path))
    fn_node: ast.FunctionDef = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_ontology_schema"
    )
    # Source positions come from this exact parsed text.
    fn_source: str = cast(str, ast.get_source_segment(source, fn_node))
    namespace: dict[str, object] = {
        "json": json,
        "cast": cast,
        "JSONValue": JSONValue,
        "os": __import__("os"),
        "logger": logging.getLogger("test"),
    }
    exec(fn_source, namespace)
    # The selected AST node is the maintained validator; the alias mirrors its signature.
    return cast(_ValidateSchema, namespace["validate_ontology_schema"])


def _load_builder_methods() -> tuple[_PropType, _LoadOwl, _BuildCard]:
    """Load selected OntologyGuidedKGCreator methods without importing langchain."""
    builder_path: Path = (
        Path(__file__).resolve().parents[1]
        / "ontographrag"
        / "kg"
        / "builders"
        / "_ontology.py"
    )
    source: str = builder_path.read_text(encoding="utf-8")
    module: ast.Module = ast.parse(source, filename=str(builder_path))
    class_node: ast.ClassDef = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "OntologySchemaMixin"
    )
    wanted: dict[str, str] = {
        node.name: cast(str, ast.get_source_segment(source, node))
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_prop_type", "_load_ontology_owl", "_build_schema_card"}
    }
    namespace: dict[str, object] = {
        "ET": ET,
        "hashlib": __import__("hashlib"),
        "json": json,
        "os": __import__("os"),
        "datetime": __import__("datetime").datetime,
        "defaultdict": defaultdict,
        "OntologySchema": OntologySchema,
        "OntEntityType": EntityType,
        "OntRelType": RelationshipType,
        "DataBinding": DataBinding,
        "RelationshipAttribute": RelationshipAttribute,
        "PropertyType": PropertyType,
        "SchemaCard": SchemaCard,
        "Optional": __import__("typing").Optional,
        "List": __import__("typing").List,
        "Dict": __import__("typing").Dict,
    }
    exec(wanted["_prop_type"], namespace)
    exec(wanted["_load_ontology_owl"], namespace)
    exec(wanted["_build_schema_card"], namespace)
    return (
        cast(_PropType, namespace["_prop_type"]),
        cast(_LoadOwl, namespace["_load_ontology_owl"]),
        cast(_BuildCard, namespace["_build_schema_card"]),
    )


def test_compatible_relationships_ranks_full_then_partial_then_unconstrained() -> None:
    schema: OntologySchema = OntologySchema(
        entity_types=[
            EntityType(id="Drug", label="Drug"),
            EntityType(id="Disease", label="Disease"),
            EntityType(id="Patient", label="Patient"),
        ],
        relationship_types=[
            RelationshipType(id="generic", label="generic"),
            RelationshipType(id="treats_any", label="treatsAny", domain="Drug"),
            RelationshipType(
                id="treats",
                label="treats",
                domain="Drug",
                range="Disease",
            ),
        ],
    )

    rel_ids: list[str] = [
        r.id for r in schema.compatible_relationships("Drug", "Disease")
    ]

    assert rel_ids == ["treats", "treats_any", "generic"]


def test_prop_type_preserves_datetime_decimal_and_double() -> None:
    prop_type, _, _ = _load_builder_methods()

    assert prop_type("datetime") == PropertyType.DATETIME
    assert prop_type("decimal") == PropertyType.DECIMAL
    assert prop_type("double") == PropertyType.DOUBLE


def test_load_ontology_owl_parses_datatype_properties_and_rel_attributes(
    tmp_path: Path,
) -> None:
    owl: str = """<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
    xmlns:owl="http://www.w3.org/2002/07/owl#"
    xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
    xmlns:ont="http://example.org/ontology/test/">
  <owl:Class rdf:about="http://example.org/ontology/test/Patient">
    <rdfs:label>Patient</rdfs:label>
  </owl:Class>
  <owl:Class rdf:about="http://example.org/ontology/test/Encounter">
    <rdfs:label>Encounter</rdfs:label>
  </owl:Class>
  <owl:ObjectProperty rdf:about="http://example.org/ontology/test/hasEncounter">
    <rdfs:label>hasEncounter</rdfs:label>
    <rdfs:domain rdf:resource="http://example.org/ontology/test/Patient"/>
    <rdfs:range rdf:resource="http://example.org/ontology/test/Encounter"/>
    <ont:cardinality>one_to_many</ont:cardinality>
  </owl:ObjectProperty>
  <owl:DatatypeProperty rdf:about="http://example.org/ontology/test/patient_id">
    <rdfs:label>patientId</rdfs:label>
    <rdfs:domain rdf:resource="http://example.org/ontology/test/Patient"/>
    <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#string"/>
    <ont:isIdentifier>true</ont:isIdentifier>
  </owl:DatatypeProperty>
  <owl:DatatypeProperty rdf:about="http://example.org/ontology/test/status">
    <rdfs:label>status</rdfs:label>
    <rdfs:domain rdf:resource="http://example.org/ontology/test/Encounter"/>
    <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#string"/>
    <ont:propertyType>enum</ont:propertyType>
    <ont:enumValues>scheduled,completed</ont:enumValues>
  </owl:DatatypeProperty>
  <owl:DatatypeProperty rdf:about="http://example.org/ontology/test/hasEncounter_note">
    <rdfs:label>note</rdfs:label>
    <ont:relationshipAttributeOf>hasEncounter</ont:relationshipAttributeOf>
    <ont:attributeType>string</ont:attributeType>
  </owl:DatatypeProperty>
</rdf:RDF>
"""
    owl_path: Path = tmp_path / "test.owl"
    owl_path.write_text(owl, encoding="utf-8")

    prop_type, load_ontology_owl, _ = _load_builder_methods()

    class _StubCreator:
        def __init__(self) -> None:
            self._prop_type: _PropType

    creator: _StubCreator = _StubCreator()
    creator._prop_type = prop_type
    schema: OntologySchema = load_ontology_owl(creator, str(owl_path))

    patient: EntityType = next(et for et in schema.entity_types if et.id == "Patient")
    encounter: EntityType = next(
        et for et in schema.entity_types if et.id == "Encounter"
    )
    rel: RelationshipType = next(
        rt for rt in schema.relationship_types if rt.id == "hasEncounter"
    )

    assert any(p.name == "patientId" and p.identifier for p in patient.properties)
    assert any(
        p.name == "status" and p.type == PropertyType.ENUM for p in encounter.properties
    )
    assert rel.cardinality == "one_to_many"
    assert any(a.name == "note" for a in rel.attributes)


def test_build_schema_card_hash_changes_when_property_signature_changes() -> None:
    _, _, build_schema_card = _load_builder_methods()

    base_entity_types: list[EntityType] = [
        EntityType(
            id="Patient",
            label="Patient",
            properties=[
                DataBinding(
                    name="patientId",
                    type=PropertyType.STRING,
                    identifier=True,
                ),
            ],
        ),
        EntityType(id="Encounter", label="Encounter"),
    ]
    rel_types: list[RelationshipType] = [
        RelationshipType(
            id="hasEncounter",
            label="hasEncounter",
            domain="Patient",
            range="Encounter",
        ),
    ]

    schema_a: OntologySchema = OntologySchema(
        entity_types=base_entity_types,
        relationship_types=rel_types,
        source_format="json",
        source_path="ontology.json",
    )
    schema_b: OntologySchema = OntologySchema(
        entity_types=[
            EntityType(
                id="Patient",
                label="Patient",
                properties=[
                    DataBinding(
                        name="patientId",
                        type=PropertyType.INTEGER,
                        identifier=True,
                    ),
                ],
            ),
            EntityType(id="Encounter", label="Encounter"),
        ],
        relationship_types=rel_types,
        source_format="json",
        source_path="ontology.json",
    )

    class _StubCreator:
        ontology_path: str | None
        ontology_classes: list[dict[str, str]]
        ontology_relationships: list[dict[str, str]]
        _ontology_schema: OntologySchema | None

    creator: _StubCreator = _StubCreator()
    creator.ontology_path = None
    creator.ontology_classes = [
        {"id": "Patient", "label": "Patient"},
        {"id": "Encounter", "label": "Encounter"},
    ]
    creator.ontology_relationships = [
        {
            "id": "hasEncounter",
            "label": "hasEncounter",
            "domain": "Patient",
            "range": "Encounter",
        },
    ]

    creator._ontology_schema = schema_a
    card_a: SchemaCard = build_schema_card(creator)

    creator._ontology_schema = schema_b
    card_b: SchemaCard = build_schema_card(creator)

    assert card_a["schemaHash"] != card_b["schemaHash"]


def test_validate_ontology_schema_rejects_missing_identifier_property() -> None:
    validate_ontology_schema: _ValidateSchema = _load_validate_ontology_schema()
    ontology: SchemaFixture = {
        "classes": [
            {
                "id": "Patient",
                "properties": [
                    {"name": "name", "type": "string"},
                ],
            },
        ],
        "relationships": [],
    }

    errors: list[str] = validate_ontology_schema(
        json.dumps(ontology).encode("utf-8"),
        "ontology.json",
    )

    assert any("missing identifier property" in err for err in errors)


def test_validate_ontology_schema_rejects_invalid_cardinality() -> None:
    validate_ontology_schema: _ValidateSchema = _load_validate_ontology_schema()
    ontology: SchemaFixture = {
        "classes": [
            {
                "id": "Patient",
                "properties": [
                    {"name": "patientId", "type": "string", "identifier": True},
                ],
            },
            {
                "id": "Encounter",
                "properties": [
                    {"name": "encounterId", "type": "string", "identifier": True},
                ],
            },
        ],
        "relationships": [
            {
                "id": "hasEncounter",
                "from": "Patient",
                "to": "Encounter",
                "cardinality": "manyish",
            },
        ],
    }

    errors: list[str] = validate_ontology_schema(
        json.dumps(ontology).encode("utf-8"),
        "ontology.json",
    )

    assert any("invalid cardinality" in err for err in errors)
