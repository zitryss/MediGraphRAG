import json
import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

from dotenv import load_dotenv
from neo4j import GraphDatabase, basic_auth
from owlready2 import get_ontology
from PyPDF2 import PdfReader

if TYPE_CHECKING:
    from neo4j import Driver, Result, Session
    from neo4j.graph import Node, Relationship
    from owlready2.namespace import Ontology


# Load environment variables
load_dotenv()

logger: logging.Logger = logging.getLogger(__name__)


class OntologyCategory(TypedDict):
    name: str
    description: NotRequired[str]


class OntologyMapping(TypedDict):
    category: str
    terms: NotRequired[list[str]]


class LoaderOntology(TypedDict, total=False):
    categories: list[OntologyCategory]
    mappings: list[OntologyMapping]
    node_labels: list[str]
    relationship_types: list[str]


class PDFNode(TypedDict):
    id: str
    label: str
    properties: dict[str, str | int]


class PDFRelationship(TypedDict):
    id: str
    type: str
    start: str
    end: str
    properties: dict[str, str | float]


class LoaderErrorDetails(TypedDict):
    error_type: str
    error_message: str
    error_args: tuple[object, ...]


class LoaderFailure(TypedDict):
    status: Literal["error"]
    message: str
    details: NotRequired[LoaderErrorDetails]


class PDFLoadSuccess(TypedDict):
    status: Literal["success"]
    nodes: list[PDFNode]
    supernodes: list[PDFNode]
    relationships: list[PDFRelationship]
    source: str
    filename: str
    ontology: str | None


class ImportedNode(TypedDict):
    id: int
    labels: list[str]
    label: str
    properties: dict[str, object]


ImportedRelationship = TypedDict(
    "ImportedRelationship",
    {
        "id": int,
        "type": str,
        "from": int,
        "to": int,
        "properties": dict[str, object],
    },
)


class ImportedGraph(TypedDict):
    status: Literal["success"]
    nodes: list[ImportedNode]
    relationships: list[ImportedRelationship]
    source: Literal["neo4j", "neo4j_sample"]
    kg_label: str | None
    query: NotRequired[str]
    total_nodes_in_db: int
    total_relationships_in_db: int
    loaded_nodes: int
    loaded_relationships: int
    complete_import: NotRequired[bool]
    sample_mode: NotRequired[bool]
    sample_strategy: NotRequired[str]


class GraphLabels(TypedDict):
    status: Literal["success"]
    labels: list[str]


class FileSaveSuccess(TypedDict):
    status: Literal["success"]
    message: str
    file_path: str


class GraphSaveSuccess(TypedDict):
    status: Literal["success"]
    message: str
    clear_database: bool
    nodes_processed: int
    relationships_processed: int


class StorableNode(TypedDict):
    id: str | int
    label: NotRequired[str]
    properties: NotRequired[dict[str, object]]


StorableRelationship = TypedDict(
    "StorableRelationship",
    {
        "from": str | int,
        "to": str | int,
        "type": str,
        "id": NotRequired[str | int],
        "properties": NotRequired[dict[str, object]],
    },
)


class StorableGraph(TypedDict, total=False):
    nodes: list[StorableNode]
    supernodes: list[StorableNode]
    relationships: list[StorableRelationship]


class KGLoader:
    def __init__(self) -> None:
        # Handle both Docker and local environments
        default_uri: str = "bolt://localhost:7687"
        configured_uri: str = os.getenv("NEO4J_URI", default_uri)

        # If running in Docker, use the configured URI, otherwise use localhost
        if os.getenv("DOCKER_ENV") == "true":
            self.neo4j_uri: str = configured_uri
        else:
            # Replace docker service name with localhost for local development
            self.neo4j_uri = configured_uri.replace("neo4j:7687", "localhost:7687")

        self.neo4j_user: str = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")

        logger.info("KGLoader initialized with Neo4j URI: %s", self.neo4j_uri)
        self.last_import_dir: str | None = None

    def _load_ontology(self, ontology_path: str) -> LoaderOntology:
        """Load ontology from JSON or OWL file"""
        if ontology_path.endswith(".json"):
            with open(ontology_path, "r") as f:
                return json.load(f)
        elif ontology_path.endswith(".owl"):
            onto: Ontology = get_ontology(ontology_path).load()
            return {
                "node_labels": [cls.name for cls in onto.classes()],
                "relationship_types": [prop.name for prop in onto.object_properties()],
            }
        else:
            raise ValueError(f"Unsupported ontology format: {ontology_path}")

    def load_from_pdf(
        self,
        file_path: str,
        ontology_path: str | None = None,
    ) -> PDFLoadSuccess | LoaderFailure:
        """Extract text from PDF and structure as knowledge graph using optional ontology"""
        try:
            logger.info("Loading PDF: %s", file_path)
            self.last_import_dir = os.path.dirname(file_path)
            reader: PdfReader = PdfReader(file_path)
            text: str = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

            # Load ontology if provided
            ontology: LoaderOntology = {}
            if ontology_path:
                try:
                    ontology = self._load_ontology(ontology_path)
                    logger.info("Loaded ontology: %s", ontology_path)
                except Exception as e:
                    logger.warning("Error loading ontology %s: %s", ontology_path, e)

            # Create knowledge graph with ontology and supernodes
            nodes, supernodes, relationships = self._create_graph_with_ontology(
                text,
                ontology,
            )

            return {
                "status": "success",
                "nodes": nodes,
                "supernodes": supernodes,
                "relationships": relationships,
                "source": "pdf",
                "filename": os.path.basename(file_path),
                "ontology": os.path.basename(ontology_path) if ontology_path else None,
            }
        except Exception as e:
            # Capture full error details for debugging
            error_details: LoaderErrorDetails = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "error_args": e.args,
            }
            logger.exception("Error loading PDF: %s", error_details)
            return {
                "status": "error",
                "message": "Failed to load PDF",
                "details": error_details,
            }

    def list_kg_labels(
        self,
        uri: str,
        user: str,
        password: str,
    ) -> GraphLabels | LoaderFailure:
        """List available KG labels in Neo4j database"""
        try:
            driver: Driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
            driver.verify_connectivity()

            with driver.session(
                database=os.getenv("NEO4J_DATABASE", "neo4j"),
            ) as session:
                result: Result = session.run("CALL db.labels()")
                labels: list[str] = [record["label"] for record in result]
                return {"status": "success", "labels": labels}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def load_from_neo4j(
        self,
        uri: str,
        user: str,
        password: str,
        kg_label: str | None = None,
        query: str | None = None,
        limit: int | None = None,
        sample_mode: bool = False,
    ) -> ImportedGraph | LoaderFailure:
        """Fetch data from Neo4j database with enhanced options for complete KG import

        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            kg_label: Optional label to filter nodes
            query: Custom query (overrides other parameters)
            limit: Maximum number of nodes to retrieve (None = no limit)
            sample_mode: If True, loads a representative sample for large graphs
        """
        driver: Driver | None = None
        session: Session | None = None
        try:
            # Set last_import_dir to kg_storage directory for consistent export behavior
            kg_storage_dir: str = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "kg_storage"),
            )
            os.makedirs(kg_storage_dir, exist_ok=True)
            self.last_import_dir = kg_storage_dir

            driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
            driver.verify_connectivity()
            db_name: str = os.getenv("NEO4J_DATABASE", "neo4j")
            session = driver.session(database=db_name)

            # First, get database statistics
            stats_result: Result = session.run(
                "MATCH (n) RETURN count(n) as node_count",
            )
            total_nodes: int = (stats_result.single() or {"node_count": 0})[
                "node_count"
            ]

            stats_result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
            total_relationships: int = (stats_result.single() or {"rel_count": 0})[
                "rel_count"
            ]

            logger.info(
                "Neo4j database stats: %s nodes, %s relationships",
                total_nodes,
                total_relationships,
            )

            # Determine loading strategy based on graph size
            if sample_mode and total_nodes > 1000:
                # For very large graphs, use sampling strategy
                # Note: We need to close current session before returning
                session.close()
                return self._load_neo4j_sample_with_driver(
                    driver,
                    kg_label,
                    total_nodes,
                    total_relationships,
                )

            if query:
                # Custom query provided
                node_query: str = query
                # Try to extract relationships for custom queries
                rel_query: str = "MATCH ()-[r]->() RETURN r, startNode(r) as start, endNode(r) as end"
                if limit:
                    rel_query += f" LIMIT {limit}"
            elif kg_label:
                # Scope by kgName property on entity nodes — avoids depending on the
                # Document←Chunk→Entity chain which misses entities without chunk links.
                node_query = (
                    f"MATCH (e:__Entity__ {{kgName: '{kg_label}'}}) RETURN e as node"
                    f" UNION ALL"
                    f" MATCH (d:Document {{kgName: '{kg_label}'}}) RETURN d as node"
                )
                rel_query = (
                    f"MATCH (e1:__Entity__ {{kgName: '{kg_label}'}})-[r]->(e2:__Entity__ {{kgName: '{kg_label}'}})"
                    f" RETURN r, e1 as start, e2 as end"
                    f" UNION ALL"
                    f" MATCH (d:Document {{kgName: '{kg_label}'}})<-[r:PART_OF]-(c:Chunk)"
                    f" RETURN r, c as start, d as end"
                )
                if limit:
                    node_query += f" LIMIT {limit}"
                    rel_query += f" LIMIT {limit * 2}"
            else:
                # Load only __Entity__ nodes and their relationships — exclude
                # Document, Chunk, and Mention infrastructure nodes which are used
                # by the RAG system but should not appear in the visualization.
                node_query = "MATCH (n:__Entity__) RETURN n"
                rel_query = "MATCH (start:__Entity__)-[r]->(end:__Entity__) RETURN r, start, end"
                if limit:
                    node_query += f" LIMIT {limit}"
                    rel_query += f" LIMIT {limit}"

            logger.debug("Executing node query: %s", node_query)
            result: Result = session.run(node_query)
            nodes: list[ImportedNode] = []
            node_id_map: dict[
                int,
                int,
            ] = {}  # Map Neo4j internal IDs to our node indices

            for record in result:
                # Handle UNION query that returns "node" column (aliased from either n or d)
                node: Node | None = record.get("node")
                if node is None:
                    continue  # Skip if node column is missing

                properties: dict[str, object] = {}
                for key, value in dict(node).items():
                    if isinstance(
                        value,
                        (list, dict, str, int, float, bool, type(None)),
                    ):
                        properties[key] = value
                    else:
                        properties[key] = str(value)

                # Use Neo4j internal ID as our node ID for consistency
                node_data: ImportedNode = {
                    "id": node.id,
                    "labels": list(node.labels),
                    "label": list(node.labels)[0] if node.labels else "Node",
                    "properties": properties,
                }
                nodes.append(node_data)
                node_id_map[node.id] = node.id

            logger.info("Loaded %s nodes", len(nodes))

            # Get relationships
            logger.debug("Executing relationship query: %s", rel_query)
            rel_result: Result = session.run(rel_query)
            relationships: list[ImportedRelationship] = []

            for record in rel_result:
                rel: Relationship = record["r"]
                start_node: Node = record["start"]
                end_node: Node = record["end"]

                # Only include relationships where both nodes were loaded
                if start_node.id in node_id_map and end_node.id in node_id_map:
                    properties = {}
                    for key, value in dict(rel).items():
                        if isinstance(
                            value,
                            (list, dict, str, int, float, bool, type(None)),
                        ):
                            properties[key] = value
                        else:
                            properties[key] = str(value)

                    relationships.append(
                        {
                            "id": rel.id,
                            "type": rel.type,
                            "from": start_node.id,
                            "to": end_node.id,
                            "properties": properties,
                        },
                    )

            logger.info("Loaded %s relationships", len(relationships))

            return {
                "status": "success",
                "nodes": nodes,
                "relationships": relationships,
                "source": "neo4j",
                "kg_label": kg_label,
                "query": node_query,
                "total_nodes_in_db": total_nodes,
                "total_relationships_in_db": total_relationships,
                "loaded_nodes": len(nodes),
                "loaded_relationships": len(relationships),
                "complete_import": limit is None and not kg_label and not query,
            }
        except Exception as e:
            error_details: LoaderErrorDetails = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "error_args": e.args,
            }
            logger.exception("Error loading from Neo4j: %s", error_details)
            return {"status": "error", "message": str(e), "details": error_details}
        finally:
            # Ensure resources are cleaned up properly
            if session:
                session.close()
            if driver:
                driver.close()

    def _load_neo4j_sample_with_driver(
        self,
        driver: "Driver",
        kg_label: str | None,
        total_nodes: int,
        total_relationships: int,
    ) -> ImportedGraph | LoaderFailure:
        """Load a representative sample from a large Neo4j graph with proper session management"""
        session: Session | None = None
        try:
            session = driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j"))
            return self._load_neo4j_sample(
                session,
                kg_label,
                total_nodes,
                total_relationships,
            )
        except Exception as e:
            return {"status": "error", "message": f"Sample loading failed: {e!s}"}
        finally:
            if session:
                session.close()
            if driver:
                driver.close()

    def _load_neo4j_sample(
        self,
        session: "Session",
        kg_label: str | None,
        total_nodes: int,
        total_relationships: int,
    ) -> ImportedGraph | LoaderFailure:
        """Load a representative sample from a large Neo4j graph"""
        try:
            # Sample strategy: get high-degree nodes and their neighborhoods
            sample_size: int = min(
                500,
                total_nodes // 10,
            )  # Sample 10% or max 500 nodes

            if kg_label:
                # Sample nodes with highest degree (most connected)
                sample_query: str = f"""
                MATCH (n:`{kg_label}`)
                WITH n, size((n)--()) as degree
                ORDER BY degree DESC
                LIMIT {sample_size}
                RETURN n
                """

                rel_query: str = f"""
                MATCH (a:`{kg_label}`)-[r]->(b:`{kg_label}`)
                WITH a, r, b, size((a)--()) + size((b)--()) as combined_degree
                ORDER BY combined_degree DESC
                LIMIT {sample_size}
                RETURN r, a as start, b as end
                """
            else:
                sample_query = f"""
                MATCH (n)
                WITH n, size((n)--()) as degree
                ORDER BY degree DESC
                LIMIT {sample_size}
                RETURN n
                """

                rel_query = f"""
                MATCH ()-[r]->()
                WITH r, startNode(r) as start, endNode(r) as end, 
                     size((startNode(r))--()) + size((endNode(r))--()) as combined_degree
                ORDER BY combined_degree DESC
                LIMIT {sample_size}
                RETURN r, start, end
                """

            logger.info(
                "Loading sample of %s most connected nodes from %s total nodes",
                sample_size,
                total_nodes,
            )

            # Load sample nodes
            result: Result = session.run(sample_query)
            nodes: list[ImportedNode] = []
            node_id_map: dict[int, int] = {}

            for record in result:
                node: Node = record["n"]
                properties: dict[str, object] = {}
                for key, value in dict(node).items():
                    if isinstance(
                        value,
                        (list, dict, str, int, float, bool, type(None)),
                    ):
                        properties[key] = value
                    else:
                        properties[key] = str(value)

                node_data: ImportedNode = {
                    "id": node.id,
                    "labels": list(node.labels),
                    "label": list(node.labels)[0] if node.labels else "Node",
                    "properties": properties,
                }
                nodes.append(node_data)
                node_id_map[node.id] = node.id

            # Load sample relationships
            rel_result: Result = session.run(rel_query)
            relationships: list[ImportedRelationship] = []

            for record in rel_result:
                rel: Relationship = record["r"]
                start_node: Node = record["start"]
                end_node: Node = record["end"]

                if start_node.id in node_id_map and end_node.id in node_id_map:
                    properties = {}
                    for key, value in dict(rel).items():
                        if isinstance(
                            value,
                            (list, dict, str, int, float, bool, type(None)),
                        ):
                            properties[key] = value
                        else:
                            properties[key] = str(value)

                    relationships.append(
                        {
                            "id": rel.id,
                            "type": rel.type,
                            "from": start_node.id,
                            "to": end_node.id,
                            "properties": properties,
                        },
                    )

            return {
                "status": "success",
                "nodes": nodes,
                "relationships": relationships,
                "source": "neo4j_sample",
                "kg_label": kg_label,
                "total_nodes_in_db": total_nodes,
                "total_relationships_in_db": total_relationships,
                "loaded_nodes": len(nodes),
                "loaded_relationships": len(relationships),
                "sample_mode": True,
                "sample_strategy": "high_degree_nodes",
            }

        except Exception as e:
            return {"status": "error", "message": f"Sample loading failed: {e!s}"}

    def _create_graph_with_ontology(
        self,
        text: str,
        ontology: LoaderOntology,
    ) -> tuple[list[PDFNode], list[PDFNode], list[PDFRelationship]]:
        """Create knowledge graph with ontology harmonization and supernodes"""
        # Extract entities using ontology if available
        entities: list[PDFNode] = []
        supernodes: list[PDFNode] = []
        relationships: list[PDFRelationship] = []

        # Create supernodes from ontology categories
        supernode_map: dict[str, str] = {}
        if "categories" in ontology:
            for i, category in enumerate(ontology["categories"]):
                supernode: PDFNode = {
                    "id": f"supernode_{i}",
                    "label": category["name"],
                    "properties": {
                        "description": category.get("description", ""),
                        "type": "supernode",
                    },
                }
                supernodes.append(supernode)
                supernode_map[category["name"]] = supernode["id"]

        # Extract entities and map to ontology
        words: set[str] = set(text.split())
        entity_counter: int = 0
        for word in words:
            if len(word) < 4:  # Skip short words
                continue

            # Find matching ontology category
            supernode_id: str | None = None
            if "mappings" in ontology:
                for mapping in ontology["mappings"]:
                    if word.lower() in mapping.get("terms", []):
                        supernode_id = supernode_map.get(mapping["category"])
                        break

            # Create entity node
            entity: PDFNode = {
                "id": f"entity_{entity_counter}",
                "label": word.capitalize(),
                "properties": {
                    "text": word,
                    "frequency": text.count(word),
                    "type": "entity",
                },
            }
            entities.append(entity)

            # Link to supernode if found
            if supernode_id:
                relationships.append(
                    {
                        "id": f"rel_{entity_counter}",
                        "type": "MEMBER_OF",
                        "start": entity["id"],
                        "end": supernode_id,
                        "properties": {"source": "ontology"},
                    },
                )

            entity_counter += 1

        # Create relationships between entities
        for i in range(min(10, len(entities) - 1)):
            relationships.append(
                {
                    "id": f"rel_entity_{i}",
                    "type": "RELATED_TO",
                    "start": entities[i]["id"],
                    "end": entities[i + 1]["id"],
                    "properties": {"strength": 0.5},
                },
            )

        return entities, supernodes, relationships

    def save_to_neo4j(
        self,
        uri: str,
        user: str,
        password: str,
        graph_data: StorableGraph,
        clear_database: bool = False,
    ) -> GraphSaveSuccess | LoaderFailure:
        """Save knowledge graph to Neo4j database with supernode support and duplicate handling

        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            graph_data: Knowledge graph data to save
            clear_database: If True, clears the database before saving (default: False)
        """
        try:
            # Use password directly without escaping
            driver: Driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))

            with driver.session(
                database=os.getenv("NEO4J_DATABASE", "neo4j"),
            ) as session:
                # Clear existing data only if requested
                if clear_database:
                    session.run("MATCH (n) DETACH DELETE n")
                    logger.info("Cleared existing database data")

                node_map: dict[str | int, str] = {}

                # Create all nodes (entities and supernodes) using MERGE to avoid duplicates
                all_nodes: list[StorableNode] = graph_data.get(
                    "nodes",
                    [],
                ) + graph_data.get("supernodes", [])
                for node in all_nodes:
                    # Sanitize labels: replace spaces with underscores
                    labels: str = node.get("label", "Node").replace(" ", "_")
                    properties: dict[str, object] = {
                        k: v for k, v in node.get("properties", {}).items()
                    }

                    # Use MERGE based on a unique identifier
                    # If node has an 'id' property, use that, otherwise use a combination of properties
                    if "id" in properties:
                        unique_key: str = "id"
                        unique_value: object = properties["id"]
                    elif "name" in properties:
                        unique_key = "name"
                        unique_value = properties["name"]
                    else:
                        # Create a hash of all properties as a unique identifier
                        import hashlib

                        prop_str: str = json.dumps(properties, sort_keys=True)
                        unique_value = hashlib.md5(prop_str.encode()).hexdigest()[:8]
                        unique_key = "unique_hash"
                        properties["unique_hash"] = unique_value

                    # Use MERGE to create if doesn't exist, or match if it does
                    merge_clause: str = (
                        f"MERGE (n:{labels} {{{unique_key}: $unique_value}})"
                    )
                    set_clause: str = "SET n += $properties"

                    result: Result = session.run(
                        f"{merge_clause} {set_clause} RETURN elementId(n) as node_id",
                        unique_value=unique_value,
                        properties=properties,
                    )
                    node_id: str = result.single()["node_id"]
                    node_map[node["id"]] = (
                        node_id  # Map our KG node ID to Neo4j node ID
                    )

                # Create relationships using the node_map, also with MERGE to avoid duplicates
                for rel in graph_data.get("relationships", []):
                    # Use the original node IDs to look up Neo4j internal IDs
                    start_neo4j_id: str | None = node_map.get(rel["from"])
                    end_neo4j_id: str | None = node_map.get(rel["to"])

                    if start_neo4j_id is not None and end_neo4j_id is not None:
                        # MERGE relationships based on the relationship type and connected nodes
                        rel_properties: dict[str, object] = rel.get("properties", {})

                        # Add a relationship ID if provided
                        if "id" in rel:
                            rel_id_value: str | int = rel["id"]
                            rel_properties["rel_id"] = rel_id_value
                        else:
                            # Use a combination of node IDs and type as unique identifier
                            rel_id_value = f"{rel['from']}_{rel['to']}_{rel['type']}"
                            rel_properties["rel_id"] = rel_id_value

                        merge_clause = (
                            f"MERGE (a)-[r:{rel['type']} {{rel_id: $rel_id}}]->(b)"
                        )

                        session.run(
                            "MATCH (a), (b) WHERE elementId(a) = $start_neo4j_id AND elementId(b) = $end_neo4j_id "
                            + merge_clause
                            + " "
                            + ("SET r += $properties" if rel_properties else ""),
                            start_neo4j_id=start_neo4j_id,
                            end_neo4j_id=end_neo4j_id,
                            rel_id=rel_id_value,
                            properties=rel_properties,
                        )
                    else:
                        logger.warning(
                            "Could not find nodes for relationship %s (%s -> %s)",
                            rel.get("id", "unknown"),
                            rel["from"],
                            rel["to"],
                        )

                return {
                    "status": "success",
                    "message": "Knowledge graph saved to Neo4j (duplicates handled)",
                    "clear_database": clear_database,
                    "nodes_processed": len(all_nodes),
                    "relationships_processed": len(graph_data.get("relationships", [])),
                }

        except Exception as e:
            # Capture full error details for debugging
            error_details: LoaderErrorDetails = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "error_args": e.args,
            }
            logger.exception("Error saving to Neo4j: %s", error_details)
            return {
                "status": "error",
                "message": "Failed to save knowledge graph to Neo4j",
                "details": error_details,
            }

    def save_to_file(
        self,
        graph_data: Mapping[str, object],
        file_path: str,
    ) -> FileSaveSuccess | LoaderFailure:
        """Save knowledge graph to a JSON file

        Args:
            graph_data: Knowledge graph data to save
            file_path: Path or filename for the output JSON file
        """
        try:
            # If file_path is just a filename (no directory separators), save to kg_storage
            if os.path.dirname(file_path) == "":
                # Just a filename provided - save to kg_storage directory
                kg_storage_dir: str = os.path.join(os.getcwd(), "kg_storage")
                os.makedirs(kg_storage_dir, exist_ok=True)
                file_path = os.path.join(kg_storage_dir, file_path)
            else:
                # Full path provided - use it as is, but ensure directory exists
                dir_path: str = os.path.dirname(file_path)
                os.makedirs(dir_path, exist_ok=True)

            # Ensure .json extension
            if not file_path.endswith(".json"):
                file_path += ".json"

            logger.info("Saving KG to: %s", os.path.abspath(file_path))

            with open(file_path, "w") as f:
                json.dump(graph_data, f, indent=2)

            return {
                "status": "success",
                "message": f"Knowledge graph saved to {file_path}",
                "file_path": file_path,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader: KGLoader = KGLoader()
    logger.info("KGLoader ready for Neo4j URI: %s", loader.neo4j_uri)
