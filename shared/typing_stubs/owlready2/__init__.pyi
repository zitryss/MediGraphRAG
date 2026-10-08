"""Only the get_ontology call consumed by the application is declared."""

from .namespace import Ontology as Ontology

def get_ontology(base_iri: str) -> Ontology: ...
