import logging
import os
from collections.abc import Mapping, Sequence
from typing import Protocol, TypeVar

from langchain_core.embeddings import Embeddings

LLM = TypeVar("LLM")


class GraphQuery(Protocol):
    """Minimum query capability consumed by retrieval and graph merging."""

    def query(
        self,
        query: str,
        params: dict[str, object] = ...,
    ) -> list[dict[str, object]]: ...


class MergeNode(Protocol):
    @property
    def id(self) -> str | int: ...

    @property
    def type(self) -> str: ...

    @property
    def properties(self) -> Mapping[str, object]: ...


class MergeRelationship(Protocol):
    @property
    def source(self) -> MergeNode: ...

    @property
    def target(self) -> MergeNode: ...

    @property
    def type(self) -> str: ...

    @property
    def properties(self) -> Mapping[str, object]: ...


class MergeDocument(Protocol):
    @property
    def nodes(self) -> Sequence[MergeNode]: ...

    @property
    def relationships(self) -> Sequence[MergeRelationship]: ...


def load_embedding_model(
    embedding_model_name: str | None = None,
) -> tuple["Embeddings", int]:
    """
    Load the embedding model based on the given name.
    If no name provided, uses environment variable EMBEDDING_PROVIDER.
    Returns a tuple of (embedding_function, embedding_dimension)
    """
    from ontographrag.kg.utils.common_functions import (
        load_embedding_model as _shared_loader,
    )

    # The explicit branch retains mypy's optional-name narrowing without a cast.
    provider_name: str = (
        embedding_model_name
        if embedding_model_name
        else os.getenv("EMBEDDING_PROVIDER", "huggingface")
    )
    embedding_function, embedding_dimension = _shared_loader(provider_name)
    logging.info(
        "Using %s embeddings with dimension %s",
        provider_name,
        embedding_dimension,
    )
    return embedding_function, embedding_dimension


def wrap_llm_with_model_name(llm: LLM) -> LLM:
    """
    Ensure an LLM-like object exposes a model_name attribute for legacy callers.
    """
    if not hasattr(llm, "model_name"):
        model_name: object = getattr(llm, "model", "unknown_model")
        if not isinstance(model_name, str):
            model_name = str(model_name)
        setattr(llm, "model_name", model_name)
    return llm


def _add_graph_documents_with_merge(
    graph: GraphQuery,
    graph_document_list: Sequence[MergeDocument],
) -> None:
    """
    Custom implementation of add_graph_documents using MERGE instead of CREATE
    to handle duplicate IDs gracefully
    """
    for doc in graph_document_list:
        # Process nodes
        for node in doc.nodes:
            # Use MERGE to create or update nodes
            node_query: str = f"""
            MERGE (n:{node.type} {{id: $id}})
            SET n += $properties
            """
            properties: dict[str, object] = {}
            if hasattr(node, "properties") and node.properties:
                properties.update(node.properties)
            # Ensure id is always set
            properties["id"] = node.id

            graph.query(node_query, {"id": node.id, "properties": properties})

        # Process relationships
        for rel in doc.relationships:
            # Use MERGE to create or update relationships
            # Sanitize relationship type to replace spaces with underscores for valid Cypher
            sanitized_rel_type: str = (
                rel.type.replace(" ", "_").replace("-", "_").upper()
            )

            rel_query: str = f"""
            MATCH (source {{id: $source_id}})
            MATCH (target {{id: $target_id}})
            MERGE (source)-[r:{sanitized_rel_type}]->(target)
            SET r += $properties
            """
            properties = {}
            if hasattr(rel, "properties") and rel.properties:
                properties.update(rel.properties)

            graph.query(
                rel_query,
                {
                    "source_id": rel.source.id,
                    "target_id": rel.target.id,
                    "properties": properties,
                },
            )
