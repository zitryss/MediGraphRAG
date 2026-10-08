"""Static conformance witnesses checked by both whole-project typing gates.

These functions are not runtime tests and never construct SDK clients. Returning
SDK values as the consumed protocols checks the real interface without casts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_community.graphs.graph_document import GraphDocument
    from langchain_neo4j import Neo4jGraph

    from shared.common_fn import GraphQuery, MergeDocument


def graph_merge_contracts(
    graph: Neo4jGraph,
    document: GraphDocument,
) -> tuple[GraphQuery, MergeDocument]:
    return graph, document
