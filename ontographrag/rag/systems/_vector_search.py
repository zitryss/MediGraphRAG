import logging
import os
import re
from typing import TYPE_CHECKING, cast

from ontographrag.rag.systems._records import (
    ChunkProjection,
    CompleteRetrievedChunk,
    EntityProjection,
    FloatListConvertible,
    GraphExpansion,
    GraphRetrievalContext,
    GraphWorkingContext,
    QueryEncoder,
    RelationshipRow,
    RetrievedEntity,
    ScoredChunkProjection,
    ScoredRetrievedChunk,
    StoredGraphProperty,
)

if TYPE_CHECKING:
    import numpy as np
    from langchain_core.embeddings import Embeddings
    from numpy.typing import NDArray

    from shared.common_fn import GraphQuery
from ontographrag.rag.reranking import (
    late_interaction_enabled,
    late_interaction_rescore_chunks_for_query,
)


class VectorSearchMixin:
    """Dense, late-interaction, and text-keyword retrieval.

    Mixin for :class:`EnhancedRAGSystem`; method bodies are unchanged
    from the original monolithic implementation.
    """

    if TYPE_CHECKING:
        _late_interaction_corpus_cache: dict[
            tuple[str, tuple[str, ...], str, bool],
            list[CompleteRetrievedChunk],
        ]
        _GRAPH_TRAVERSAL_SEED_LIMIT: int
        _active_vector_index_name: str | None

        def _create_neo4j_connection(self) -> "GraphQuery": ...

        _embedding_cache: dict[str, list[float]]

        def _expand_entities_via_graph(
            self,
            graph: "GraphQuery",
            seed_entity_ids: list[str],
            kg_name: str | None = None,
            max_hops: int = 2,
            max_neighbors: int = 30,
            question_id: str | None = None,
            document_names: list[str] | None = None,
            allowed_rel_types: list[str] | None = None,
            excluded_entity_ids: set[str] | None = None,
        ) -> GraphExpansion: ...
        def _fetch_relationships_for_entity_ids(
            self,
            graph: "GraphQuery",
            entity_ids: list[str],
            *,
            kg_name: str | None = None,
            question_id: str | None = None,
            document_names: list[str] | None = None,
        ) -> list[RelationshipRow]: ...

        _vector_index_available: bool | None
        embedding_model: "Embeddings"

    @staticmethod
    def _is_pure_vector_search_method(search_method: str | None) -> bool:
        """Return True when the context already came from a pure vector-only route."""
        return str(search_method or "").strip().lower() in {
            "vector_similarity",
            "retrieval_span_similarity",
        }

    @staticmethod
    def _late_interaction_corpus_cap() -> int:
        return max(
            200,
            int(
                os.getenv("ONTOGRAPHRAG_LATE_INTERACTION_CORPUS_CAP", "5000").strip()
                or 5000,
            ),
        )

    def _first_stage_late_interaction_enabled(self) -> bool:
        explicit: str = (
            str(os.getenv("ONTOGRAPHRAG_FIRST_STAGE_LATE_INTERACTION", ""))
            .strip()
            .lower()
        )
        if explicit:
            return explicit in {"1", "true", "yes", "on"}
        return late_interaction_enabled()

    @staticmethod
    def _late_interaction_scope_key(
        *,
        kg_name: str | None,
        document_names: list[str] | None,
        question_id: str | None,
        include_entities: bool,
    ) -> tuple[str, tuple[str, ...], str, bool]:
        return (
            kg_name or "",
            tuple(sorted(document_names or [])),
            question_id or "",
            bool(include_entities),
        )

    def _generate_query_embedding(self, query: str) -> list[float]:
        """Generate query embedding, caching results to avoid redundant calls."""
        if query in self._embedding_cache:
            return self._embedding_cache[query]

        if hasattr(self.embedding_model, "embed_query"):
            embedding: list[float] = self.embedding_model.embed_query(query)
        elif hasattr(self.embedding_model, "encode"):
            raw: NDArray[np.float32] | list[float] = cast(
                QueryEncoder,
                self.embedding_model,
            ).encode(query, convert_to_numpy=True)
            embedding = (
                cast(FloatListConvertible, raw).tolist()
                if hasattr(raw, "tolist")
                else list(raw)
            )
        else:
            raise ValueError(
                "Unsupported embedding model interface. Expected embed_query or encode.",
            )

        # Cap cache size to avoid unbounded memory growth.
        if len(self._embedding_cache) >= 2048:
            self._embedding_cache.pop(next(iter(self._embedding_cache)))
        self._embedding_cache[query] = embedding
        return embedding

    def _late_interaction_corpus_rows(
        self,
        graph: "GraphQuery",
        *,
        kg_name: str | None,
        document_names: list[str] | None,
        question_id: str | None,
        include_entities: bool,
    ) -> list[CompleteRetrievedChunk]:
        cache: (
            dict[tuple[str, tuple[str, ...], str, bool], list[CompleteRetrievedChunk]]
            | None
        ) = getattr(self, "_late_interaction_corpus_cache", None)
        if cache is None:
            cache = {}
            self._late_interaction_corpus_cache = cache

        scope_key: tuple[str, tuple[str, ...], str, bool] = (
            self._late_interaction_scope_key(
                kg_name=kg_name,
                document_names=document_names,
                question_id=question_id,
                include_entities=include_entities,
            )
        )
        if scope_key in cache:
            return [cast(CompleteRetrievedChunk, dict(row)) for row in cache[scope_key]]

        params: dict[str, object] = {"corpus_cap": self._late_interaction_corpus_cap()}
        filters: list[str] = []
        if kg_name:
            filters.append("d.kgName = $kg_name")
            params["kg_name"] = kg_name
        if document_names:
            filters.append("d.fileName IN $document_names")
            params["document_names"] = list(document_names)
        if question_id:
            filters.append(
                "coalesce(retrieval.questionId, chunk.questionId) = $question_id",
            )
            params["question_id"] = question_id
        where_clause: str = "WHERE " + " AND ".join(filters) if filters else ""

        entity_match: str = ""
        with_clause: str = "WITH retrieval, chunk, d"
        entity_return: str = "[] AS entities"
        if include_entities:
            entity_match = "OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(entity:__Entity__)"
            with_clause = (
                "WITH retrieval, chunk, d, collect(DISTINCT entity) AS chunk_entities"
            )
            entity_return = """[entity IN chunk_entities WHERE entity IS NOT NULL | {
                id: coalesce(entity.id, entity.name),
                element_id: elementId(entity),
                type: coalesce(entity.type, 'Entity'),
                description: coalesce(entity.name, '')
            }] AS entities"""

        retrieval_query: str = f"""
        MATCH (retrieval:RetrievalChunk)-[:RETRIEVES_FROM]->(chunk:Chunk)-[:PART_OF]->(d:Document)
        {where_clause}
        {entity_match}
        {with_clause}
        RETURN
            retrieval.text AS text,
            retrieval.id AS chunk_id,
            elementId(retrieval) AS chunk_element_id,
            chunk.id AS parent_chunk_id,
            elementId(chunk) AS parent_chunk_element_id,
            coalesce(retrieval.questionId, chunk.questionId) AS question_id,
            coalesce(retrieval.passageIndex, chunk.passageIndex) AS passage_index,
            coalesce(retrieval.chunkLocalIndex, chunk.chunkLocalIndex) AS chunk_local_index,
            retrieval.retrievalLocalIndex AS retrieval_local_index,
            chunk.position AS position,
            chunk.source AS source,
            d.fileName AS document,
            d.kgName AS kg_name,
            {entity_return}
        ORDER BY d.fileName ASC,
                 coalesce(retrieval.questionId, ''),
                 coalesce(retrieval.passageIndex, -1),
                 coalesce(retrieval.chunkLocalIndex, chunk.chunkLocalIndex, chunk.position),
                 chunk.position ASC
        LIMIT $corpus_cap
        """
        rows: list[ChunkProjection] = (
            cast(list[ChunkProjection], graph.query(retrieval_query, params)) or []
        )

        if not rows:
            chunk_filters: list[str] = []
            if kg_name:
                chunk_filters.append("d.kgName = $kg_name")
            if document_names:
                chunk_filters.append("d.fileName IN $document_names")
            if question_id:
                chunk_filters.append("chunk.questionId = $question_id")
            chunk_where: str = (
                "WHERE " + " AND ".join(chunk_filters) if chunk_filters else ""
            )

            entity_match = ""
            with_clause = "WITH chunk, d"
            entity_return = "[] AS entities"
            if include_entities:
                entity_match = (
                    "OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(entity:__Entity__)"
                )
                with_clause = (
                    "WITH chunk, d, collect(DISTINCT entity) AS chunk_entities"
                )
                entity_return = """[entity IN chunk_entities WHERE entity IS NOT NULL | {
                    id: coalesce(entity.id, entity.name),
                    element_id: elementId(entity),
                    type: coalesce(entity.type, 'Entity'),
                    description: coalesce(entity.name, '')
                }] AS entities"""

            chunk_query: str = f"""
            MATCH (chunk:Chunk)-[:PART_OF]->(d:Document)
            {chunk_where}
            {entity_match}
            {with_clause}
            RETURN
                chunk.text AS text,
                chunk.id AS chunk_id,
                elementId(chunk) AS chunk_element_id,
                chunk.questionId AS question_id,
                chunk.passageIndex AS passage_index,
                chunk.chunkLocalIndex AS chunk_local_index,
                chunk.position AS position,
                chunk.source AS source,
                d.fileName AS document,
                d.kgName AS kg_name,
                {entity_return}
            ORDER BY d.fileName ASC,
                     coalesce(chunk.questionId, ''),
                     coalesce(chunk.passageIndex, -1),
                     coalesce(chunk.chunkLocalIndex, chunk.position),
                     chunk.position ASC
            LIMIT $corpus_cap
            """
            rows = cast(list[ChunkProjection], graph.query(chunk_query, params)) or []

        cache[scope_key] = [cast(CompleteRetrievedChunk, dict(row)) for row in rows]
        return [cast(CompleteRetrievedChunk, dict(row)) for row in rows]

    def _late_interaction_search(
        self,
        graph: "GraphQuery",
        query: str,
        document_names: list[str] | None = None,
        max_chunks: int = 20,
        kg_name: str | None = None,
        max_hops: int = 2,
        question_id: str | None = None,
    ) -> GraphRetrievalContext:
        try:
            scope_key: tuple[str, tuple[str, ...], str, bool] = (
                self._late_interaction_scope_key(
                    kg_name=kg_name,
                    document_names=document_names,
                    question_id=question_id,
                    include_entities=True,
                )
            )
            rows: list[CompleteRetrievedChunk] = self._late_interaction_corpus_rows(
                graph,
                kg_name=kg_name,
                document_names=document_names,
                question_id=question_id,
                include_entities=True,
            )
            if not rows:
                return {
                    "query": query,
                    "chunks": [],
                    "entities": {},
                    "relationships": [],
                    "graph_neighbors": {},
                    "traversal_paths": [],
                    "documents": [],
                    "total_score": 0,
                    "search_method": "late_interaction_unavailable",
                }

            candidate_rows: list[ScoredRetrievedChunk] = []
            for row in rows:
                entities: list[RetrievedEntity] | list[EntityProjection] = (
                    row.get("entities") or []
                )
                candidate_rows.append(
                    {
                        "text": row["text"],
                        "chunk_id": row["chunk_id"],
                        "chunk_element_id": row["chunk_element_id"],
                        "parent_chunk_id": row.get("parent_chunk_id"),
                        "parent_chunk_element_id": row.get("parent_chunk_element_id"),
                        "question_id": row.get("question_id"),
                        "passage_index": row.get("passage_index"),
                        "chunk_local_index": row.get("chunk_local_index"),
                        "retrieval_local_index": row.get("retrieval_local_index"),
                        "position": row.get("position"),
                        "source": row.get("source"),
                        "score": 0.0,
                        "document": row["document"],
                        "kg_name": row.get("kg_name"),
                        "entities": entities,
                        "linked_entity_ids": [
                            entity.get("id")
                            for entity in entities
                            if isinstance(entity, dict) and entity.get("id")
                        ],
                        "linked_entity_count": len(entities),
                    },
                )

            reranked_rows, li_meta = late_interaction_rescore_chunks_for_query(
                query,
                candidate_rows,
                max_chunks=max_chunks,
                replace_score=True,
                index_key=scope_key,
            )
            selected_rows: list[ScoredRetrievedChunk] = reranked_rows[:max_chunks]
            if not li_meta.get("applied"):
                return {
                    "query": query,
                    "chunks": [],
                    "entities": {},
                    "relationships": [],
                    "graph_neighbors": {},
                    "traversal_paths": [],
                    "documents": [],
                    "total_score": 0,
                    "search_method": "late_interaction_unavailable",
                    "late_interaction_stage": li_meta,
                }

            score_values: list[float] = [
                abs(
                    float(
                        row.get("late_interaction_score", row.get("score", 0.0)) or 0.0,
                    ),
                )
                for row in selected_rows
            ]
            if not selected_rows or max(score_values, default=0.0) <= 1e-9:
                return {
                    "query": query,
                    "chunks": [],
                    "entities": {},
                    "relationships": [],
                    "graph_neighbors": {},
                    "traversal_paths": [],
                    "documents": [],
                    "total_score": 0,
                    "search_method": "late_interaction_unavailable",
                    "late_interaction_stage": {
                        **li_meta,
                        "applied": False,
                        "reason": "no_score_signal",
                    },
                }

            context: GraphRetrievalContext = {
                "query": query,
                "chunks": selected_rows,
                "entities": {},
                "relationships": [],
                "graph_neighbors": {},
                "traversal_paths": [],
                "documents": list(
                    {
                        row.get("document")
                        for row in selected_rows
                        if row.get("document")
                    },
                ),
                "total_score": float(
                    sum(float(row.get("score", 0.0)) for row in selected_rows),
                ),
                "search_method": "late_interaction_similarity",
                "late_interaction_stage": li_meta,
            }

            all_entity_ids: list[str] = []
            for selected_row in selected_rows:
                for entity in selected_row.get("entities", []):
                    entity_id: str | None = entity.get("id")
                    if not entity_id:
                        continue
                    if entity_id not in context["entities"]:
                        context["entities"][entity_id] = {
                            "id": entity_id,
                            "element_id": entity.get("element_id"),
                            "type": entity.get("type"),
                            "description": entity.get("description"),
                            "mentioned_in_chunks": [selected_row["chunk_id"]],
                            "source": "late_interaction",
                        }
                        all_entity_ids.append(entity_id)
                    else:
                        context["entities"][entity_id].setdefault(
                            "mentioned_in_chunks",
                            [],
                        ).append(selected_row["chunk_id"])

            if all_entity_ids:
                expansion: GraphExpansion = self._expand_entities_via_graph(
                    graph,
                    all_entity_ids[: self._GRAPH_TRAVERSAL_SEED_LIMIT],
                    kg_name=kg_name,
                    max_hops=max_hops,
                    question_id=question_id,
                    document_names=document_names,
                )
                context["graph_neighbors"] = cast(
                    dict[str, RetrievedEntity],
                    expansion["neighbors"],
                )
                context["traversal_paths"] = expansion["paths"]
                rel_results: list[RelationshipRow] = (
                    self._fetch_relationships_for_entity_ids(
                        graph,
                        all_entity_ids[:40],
                        kg_name=kg_name,
                        question_id=question_id,
                        document_names=document_names,
                    )
                )
                for rel in rel_results:
                    rel_key: str = (
                        rel.get("relationship_element_id")
                        or f"{rel['source']}-{rel['relationship_type']}-{rel['target']}"
                        f"-negated={bool(rel.get('negated', False))}"
                    )
                    context["relationships"].append(
                        {
                            "key": rel_key,
                            "source": rel["source"],
                            "source_element_id": rel["source_element_id"],
                            "target": rel["target"],
                            "target_element_id": rel["target_element_id"],
                            "type": rel["relationship_type"],
                            "element_id": rel["relationship_element_id"],
                            "negated": bool(rel.get("negated", False)),
                            "condition": rel.get("condition"),
                            "quantitative": rel.get("quantitative"),
                            "confidence": rel.get("confidence"),
                            "question_ids": rel.get("question_ids") or [],
                            "passage_keys": rel.get("passage_keys") or [],
                            "provenance_positions": rel.get("provenance_positions")
                            or [],
                        },
                    )

            context["entity_count"] = len(context["entities"])
            context["relationship_count"] = len(context["relationships"])
            context.setdefault("grounding_quality", 0.0)
            context.setdefault("seed_entity_count", 0)
            return context
        except Exception as exc:
            logging.warning("Late-interaction search failed: %s", exc)
            return {
                "query": query,
                "chunks": [],
                "entities": {},
                "relationships": [],
                "graph_neighbors": {},
                "traversal_paths": [],
                "documents": [],
                "total_score": 0,
                "search_method": "late_interaction_error",
                "error": str(exc),
            }

    def _semantic_similarity_search(
        self,
        graph: "GraphQuery",
        query: str,
        document_names: list[str] | None = None,
        similarity_threshold: float = 0.08,
        max_chunks: int = 20,
        kg_name: str | None = None,
        max_hops: int = 2,
        question_id: str | None = None,
        allow_first_stage_late_interaction: bool = True,
    ) -> GraphRetrievalContext:
        if (
            allow_first_stage_late_interaction
            and self._first_stage_late_interaction_enabled()
        ):
            context: GraphRetrievalContext = self._late_interaction_search(
                graph,
                query,
                document_names=document_names,
                max_chunks=max_chunks,
                kg_name=kg_name,
                max_hops=max_hops,
                question_id=question_id,
            )
            if context.get("chunks"):
                return context
        return self._vector_similarity_search(
            graph,
            query,
            document_names=document_names,
            similarity_threshold=similarity_threshold,
            max_chunks=max_chunks,
            kg_name=kg_name,
            max_hops=max_hops,
            question_id=question_id,
        )

    def check_vector_index(self) -> bool:
        """
        Check if vector index exists and has correct dimensions. Result is cached after first call.
        """
        if self._vector_index_available is not None:
            return self._vector_index_available
        try:
            graph: GraphQuery = self._create_neo4j_connection()
            result: list[dict[str, object]] = graph.query(
                "SHOW INDEXES YIELD name, type, state, options WHERE type = 'VECTOR'",
            )
            if not result:
                logging.warning("No vector index found - falling back to text search")
                self._vector_index_available = False
                return False
            probe_embedding: list[float] = self._generate_query_embedding("test")
            online_indexes: dict[str, dict[str, object]] = {
                str(row.get("name")): row
                for row in result
                if str(row.get("state") or "").upper() in {"", "ONLINE"}
            }
            for candidate in ("retrieval_vector", "vector"):
                if candidate not in online_indexes:
                    continue
                try:
                    graph.query(
                        f"""
                        CALL db.index.vector.queryNodes('{candidate}', 1, $query_vector)
                        YIELD node, score
                        RETURN count(node) AS n
                        """,
                        {"query_vector": probe_embedding},
                    )
                    logging.info(
                        "Vector index probe succeeded for %s — result cached",
                        candidate,
                    )
                    self._active_vector_index_name = candidate
                    self._vector_index_available = True
                    return True
                except Exception as probe_error:
                    logging.warning(
                        "Vector index %s exists but probe failed: %s",
                        candidate,
                        probe_error,
                    )
            logging.warning(
                "No compatible vector index found - falling back to text search",
            )
            self._vector_index_available = False
            return False
        except Exception as e:
            logging.error(f"Error checking vector index: {e}")
            self._vector_index_available = False
            return False

    def _text_similarity_search(
        self,
        graph: "GraphQuery",
        query: str,
        document_names: list[str] | None = None,
        max_chunks: int = 20,
        kg_name: str | None = None,
        max_hops: int = 2,
        question_id: str | None = None,
    ) -> GraphRetrievalContext:
        """
        Text-based similarity search as fallback when vector search fails
        """
        try:
            logging.info(
                f"Using text similarity search as fallback (kg_name: {kg_name})",
            )

            # Build WHERE clause based on filters and text match
            where_conditions: list[str] = [
                "ANY(term IN $search_terms WHERE toLower(c.text) CONTAINS term)",
            ]
            params: dict[str, object] = {}

            if kg_name:
                where_conditions.append("d.kgName = $kg_name")
                params["kg_name"] = kg_name

            if question_id:
                where_conditions.append("c.questionId = $question_id")
                params["question_id"] = question_id

            if document_names:
                where_conditions.append("d.fileName IN $document_names")
                params["document_names"] = document_names

            # Build a single WHERE clause to avoid duplicate WHERE errors
            where_clause: str = "WHERE " + " AND ".join(where_conditions)

            # Text similarity search: score by number of query terms matched in chunk text
            search_query: str = f"""
            MATCH (c:Chunk)-[:PART_OF]->(d:Document)
            {where_clause}
            WITH c, d,
                 size([term IN $search_terms WHERE toLower(c.text) CONTAINS term]) AS matched_terms
            OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:__Entity__)
            WITH c, d, matched_terms, collect(DISTINCT e) AS chunk_entities
            RETURN
                c.text AS text,
                c.id AS chunk_id,
                elementId(c) AS chunk_element_id,
                c.questionId AS question_id,
                c.passageIndex AS passage_index,
                c.chunkLocalIndex AS chunk_local_index,
                toFloat(matched_terms) / size($search_terms) AS score,
                d.fileName AS document,
                d.kgName AS kg_name,
                [entity IN chunk_entities WHERE entity IS NOT NULL | {{
                    id: coalesce(entity.id, entity.name),
                    element_id: elementId(entity),
                    type: coalesce(entity.type, 'Entity'),
                    description: coalesce(entity.name, '')
                }}] AS entities
            ORDER BY score DESC
            LIMIT $max_chunks
            """

            # Tokenized fallback matching is more robust than a single 3-word phrase.
            raw_terms: list[str] = [
                t.lower() for t in re.findall(r"[A-Za-z0-9]+", query)
            ]
            search_terms: list[str] = [t for t in raw_terms if len(t) >= 3][:8]
            if not search_terms:
                search_terms = [query.lower().strip()]

            params.update({"search_terms": search_terms, "max_chunks": max_chunks})

            results: list[ScoredChunkProjection] = cast(
                list[ScoredChunkProjection],
                graph.query(search_query, params),
            )

            context: GraphWorkingContext = {
                "query": query,
                "chunks": [],
                "entities": {},
                "relationships": [],
                "documents": set(),
                "total_score": 0,
                "search_method": "text_similarity_fallback",
            }

            for result in results:
                _result_entities: list[EntityProjection] = result.get("entities") or []
                chunk_info: ScoredRetrievedChunk = {
                    "text": result["text"],
                    "chunk_id": result["chunk_id"],
                    "chunk_element_id": result["chunk_element_id"],
                    "question_id": result.get("question_id"),
                    "passage_index": result.get("passage_index"),
                    "chunk_local_index": result.get("chunk_local_index"),
                    "score": result["score"],
                    "document": result["document"],
                    "kg_name": result.get("kg_name"),
                    "entities": result.get("entities") or [],
                }
                context["chunks"].append(chunk_info)
                cast(set[StoredGraphProperty], context["documents"]).add(
                    result["document"],
                )
                context["total_score"] += result["score"]

                # Collect unique entities
                for entity in result.get("entities") or []:
                    entity_id: str = entity["id"]
                    if entity_id not in context["entities"]:
                        context["entities"][entity_id] = {
                            "id": entity_id,
                            "element_id": entity["element_id"],
                            "type": entity["type"],
                            "description": entity["description"],
                            "mentioned_in_chunks": [],
                        }
                    if result.get("chunk_id"):
                        context["entities"][entity_id]["mentioned_in_chunks"].append(
                            result["chunk_id"],
                        )

            # Multi-hop graph traversal from seed entities
            seed_ids: list[str] = list(context["entities"].keys())
            expansion: GraphExpansion = self._expand_entities_via_graph(
                graph,
                seed_ids,
                kg_name=kg_name,
                max_hops=max_hops,
                question_id=question_id,
                document_names=document_names,
            )
            context["graph_neighbors"] = cast(
                dict[str, RetrievedEntity],
                expansion["neighbors"],
            )
            context["traversal_paths"] = expansion["paths"]

            # Merge neighbor entities into the entity dict (tagged so they're
            # distinguishable from directly-retrieved seed entities)
            for nid, ninfo in expansion["neighbors"].items():
                if nid not in context["entities"]:
                    context["entities"][nid] = {
                        "id": nid,
                        "element_id": ninfo["element_id"],
                        "type": ninfo["type"],
                        "description": ninfo["name"],
                        "mentioned_in_chunks": [],
                        "source": "graph_traversal",
                        "min_hops": ninfo["min_hops"],
                    }

            # Relationships between ALL entities (seeds + neighbors)
            all_entity_ids: list[str] = list(context["entities"].keys())
            if all_entity_ids:
                relationship_results: list[RelationshipRow] = (
                    self._fetch_relationships_for_entity_ids(
                        graph,
                        all_entity_ids,
                        kg_name=kg_name,
                        question_id=question_id,
                        document_names=document_names,
                    )
                )
                for rel_result in relationship_results:
                    rel_key: str = (
                        rel_result.get("relationship_element_id")
                        or f"{rel_result['source']}-{rel_result['relationship_type']}-{rel_result['target']}"
                        f"-negated={bool(rel_result.get('negated', False))}"
                    )
                    if not any(
                        r.get("key") == rel_key for r in context["relationships"]
                    ):
                        context["relationships"].append(
                            {
                                "key": rel_key,
                                "source": rel_result["source"],
                                "source_element_id": rel_result["source_element_id"],
                                "target": rel_result["target"],
                                "target_element_id": rel_result["target_element_id"],
                                "type": rel_result["relationship_type"],
                                "element_id": rel_result["relationship_element_id"],
                                "negated": bool(rel_result.get("negated", False)),
                                "condition": rel_result.get("condition"),
                                "quantitative": rel_result.get("quantitative"),
                                "confidence": rel_result.get("confidence"),
                                "question_ids": rel_result.get("question_ids") or [],
                                "passage_keys": rel_result.get("passage_keys") or [],
                                "provenance_positions": rel_result.get(
                                    "provenance_positions",
                                )
                                or [],
                            },
                        )

            context["documents"] = list(context["documents"])
            context["entity_count"] = len(context["entities"])
            context["relationship_count"] = len(context["relationships"])

            return cast(GraphRetrievalContext, context)

        except Exception as e:
            logging.error(f"Error in text similarity search: {e}")
            return {
                "query": query,
                "chunks": [],
                "entities": {},
                "relationships": [],
                "graph_neighbors": {},
                "traversal_paths": [],
                "documents": [],
                "total_score": 0,
                "search_method": "text_similarity_fallback",
                "error": str(e),
            }

    def _vector_similarity_search(
        self,
        graph: "GraphQuery",
        query: str,
        document_names: list[str] | None = None,
        similarity_threshold: float = 0.08,
        max_chunks: int = 20,
        kg_name: str | None = None,
        max_hops: int = 2,
        question_id: str | None = None,
    ) -> GraphRetrievalContext:
        """
        Vector similarity search using real sentence transformer embeddings.

        Args:
            graph: Neo4j graph connection
            query: The search query
            document_names: Optional list of document names to filter by
            similarity_threshold: Minimum similarity score for retrieval
            max_chunks: Maximum number of chunks to retrieve
            kg_name: Optional KG name to filter retrieval to a specific named KG
        """
        try:
            logging.info(
                f"Using vector similarity search with real embeddings (kg_name: {kg_name})",
            )

            query_embedding: list[float] = self._generate_query_embedding(query)
            logging.info(
                f"Generated query embedding with shape: {len(query_embedding)}",
            )
            if not self.check_vector_index():
                return {
                    "query": query,
                    "chunks": [],
                    "entities": {},
                    "relationships": [],
                    "graph_neighbors": {},
                    "traversal_paths": [],
                    "documents": [],
                    "total_score": 0,
                    "search_method": "vector_similarity_unavailable",
                }
            vector_index_name: str = self._active_vector_index_name or "vector"

            # Build the vector search query with optional filtering by kg_name and/or document_names
            # Relationship is: Chunk -[PART_OF]-> Document
            # We need to filter by either document_names or kg_name (or both)

            # Build one WHERE clause that always includes score threshold
            where_conditions: list[str] = ["score >= $similarity_threshold"]
            params: dict[str, object] = {
                "query_vector": query_embedding,
                "similarity_threshold": similarity_threshold,
            }

            if kg_name:
                where_conditions.append("d.kgName = $kg_name")
                params["kg_name"] = kg_name

            if document_names:
                where_conditions.append("d.fileName IN $document_names")
                params["document_names"] = document_names

            if question_id:
                where_conditions.append("chunk.questionId = $question_id")
                params["question_id"] = question_id

            where_clause: str = "WHERE " + " AND ".join(where_conditions)

            # When filtering by kg_name or document_names the vector index is
            # queried globally first, then filtered.  Overfetch so that the
            # target KG's chunks are not accidentally excluded because they
            # didn't rank in the global top-N.
            # Overfetch so KG-filtered chunks aren't excluded by global top-N,
            # but cap at 500 to avoid timeouts on large statistical queries.
            retrieval_count: int = (
                min(max_chunks * 20, 500)
                if (kg_name or document_names or question_id)
                else max_chunks
            )

            if vector_index_name == "retrieval_vector":
                if question_id:
                    where_clause = where_clause.replace(
                        "chunk.questionId = $question_id",
                        "coalesce(retrieval.questionId, chunk.questionId) = $question_id",
                    )
                search_query: str = f"""
                CALL db.index.vector.queryNodes('{vector_index_name}', {retrieval_count}, $query_vector)
                YIELD node AS retrieval, score
                MATCH (retrieval:RetrievalChunk)-[:RETRIEVES_FROM]->(chunk:Chunk)-[:PART_OF]->(d:Document)
                {where_clause}
                OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:__Entity__)
                WITH retrieval, chunk, score, d,
                     collect(DISTINCT e) AS chunk_entities
                RETURN
                    retrieval.text AS text,
                    retrieval.id AS chunk_id,
                    elementId(retrieval) AS chunk_element_id,
                    chunk.id AS parent_chunk_id,
                    elementId(chunk) AS parent_chunk_element_id,
                    coalesce(retrieval.questionId, chunk.questionId) AS question_id,
                    coalesce(retrieval.passageIndex, chunk.passageIndex) AS passage_index,
                    coalesce(retrieval.chunkLocalIndex, chunk.chunkLocalIndex) AS chunk_local_index,
                    retrieval.retrievalLocalIndex AS retrieval_local_index,
                    chunk.position AS position,
                    chunk.source AS source,
                    score,
                    d.fileName AS document,
                    d.kgName AS kg_name,
                    [entity IN chunk_entities WHERE entity IS NOT NULL | {{
                        id: coalesce(entity.id, entity.name),
                        element_id: elementId(entity),
                        type: coalesce(entity.type, 'Entity'),
                        description: coalesce(entity.name, '')
                    }}] AS entities
                ORDER BY score DESC
                LIMIT $max_chunks
                """
            else:
                search_query = f"""
                CALL db.index.vector.queryNodes('{vector_index_name}', {retrieval_count}, $query_vector)
                YIELD node AS chunk, score
                MATCH (chunk)-[:PART_OF]->(d:Document)
                {where_clause}
                OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:__Entity__)
                WITH chunk, score, d,
                     collect(DISTINCT e) AS chunk_entities
                RETURN
                    chunk.text AS text,
                    chunk.id AS chunk_id,
                    elementId(chunk) AS chunk_element_id,
                    chunk.questionId AS question_id,
                    chunk.passageIndex AS passage_index,
                    chunk.chunkLocalIndex AS chunk_local_index,
                    chunk.position AS position,
                    chunk.source AS source,
                    score,
                    d.fileName AS document,
                    d.kgName AS kg_name,
                    [entity IN chunk_entities WHERE entity IS NOT NULL | {{
                        id: coalesce(entity.id, entity.name),
                        element_id: elementId(entity),
                        type: coalesce(entity.type, 'Entity'),
                        description: coalesce(entity.name, '')
                    }}] AS entities
                ORDER BY score DESC
                LIMIT $max_chunks
                """
            params["max_chunks"] = max_chunks

            results: list[ScoredChunkProjection] = cast(
                list[ScoredChunkProjection],
                graph.query(search_query, params),
            )
            if vector_index_name == "retrieval_vector" and question_id and not results:
                logging.info(
                    "No question-scoped retrieval spans found for kg=%s question_id=%s; "
                    "falling back to parent Chunk vector index.",
                    kg_name,
                    question_id,
                )
                fallback_where_clause: str = where_clause.replace(
                    "coalesce(retrieval.questionId, chunk.questionId) = $question_id",
                    "chunk.questionId = $question_id",
                )
                fallback_query: str = f"""
                CALL db.index.vector.queryNodes('vector', {retrieval_count}, $query_vector)
                YIELD node AS chunk, score
                MATCH (chunk)-[:PART_OF]->(d:Document)
                {fallback_where_clause}
                OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:__Entity__)
                WITH chunk, score, d,
                     collect(DISTINCT e) AS chunk_entities
                RETURN
                    chunk.text AS text,
                    chunk.id AS chunk_id,
                    elementId(chunk) AS chunk_element_id,
                    chunk.questionId AS question_id,
                    chunk.passageIndex AS passage_index,
                    chunk.chunkLocalIndex AS chunk_local_index,
                    chunk.position AS position,
                    chunk.source AS source,
                    score,
                    d.fileName AS document,
                    d.kgName AS kg_name,
                    [entity IN chunk_entities WHERE entity IS NOT NULL | {{
                        id: coalesce(entity.id, entity.name),
                        element_id: elementId(entity),
                        type: coalesce(entity.type, 'Entity'),
                        description: coalesce(entity.name, '')
                    }}] AS entities
                ORDER BY score DESC
                LIMIT $max_chunks
                """
                try:
                    fallback_results: list[ScoredChunkProjection] = (
                        cast(
                            list[ScoredChunkProjection],
                            graph.query(fallback_query, params),
                        )
                        or []
                    )
                except Exception as fallback_exc:
                    logging.warning(
                        "Chunk-vector fallback failed for kg=%s question_id=%s: %s",
                        kg_name,
                        question_id,
                        fallback_exc,
                    )
                else:
                    if fallback_results:
                        results = fallback_results
                        vector_index_name = "vector"

            context: GraphWorkingContext = {
                "query": query,
                "chunks": [],
                "entities": {},
                "relationships": [],
                "graph_neighbors": {},
                "traversal_paths": [],
                "documents": set(),
                "total_score": 0,
                "search_method": (
                    "retrieval_span_similarity"
                    if vector_index_name == "retrieval_vector"
                    else "vector_similarity"
                ),
            }

            for result in results:
                chunk_info: ScoredRetrievedChunk = {
                    "text": result["text"],
                    "chunk_id": result["chunk_id"],
                    "chunk_element_id": result["chunk_element_id"],
                    "parent_chunk_id": result.get("parent_chunk_id"),
                    "parent_chunk_element_id": result.get("parent_chunk_element_id"),
                    "question_id": result.get("question_id"),
                    "passage_index": result.get("passage_index"),
                    "chunk_local_index": result.get("chunk_local_index"),
                    "retrieval_local_index": result.get("retrieval_local_index"),
                    "position": result.get("position"),
                    "source": result.get("source"),
                    "score": result["score"],
                    "document": result["document"],
                    "kg_name": result.get("kg_name"),
                    "entities": result.get("entities") or [],
                    "linked_entity_ids": [
                        entity.get("id")
                        for entity in (result.get("entities") or [])
                        if entity.get("id")
                    ],
                    "linked_entity_count": len(result.get("entities") or []),
                }
                context["chunks"].append(chunk_info)
                cast(set[StoredGraphProperty], context["documents"]).add(
                    result["document"],
                )
                context["total_score"] += result["score"]

                # Collect unique entities
                for entity in result.get("entities") or []:
                    entity_id: str = entity["id"]
                    if entity_id not in context["entities"]:
                        context["entities"][entity_id] = {
                            "id": entity_id,
                            "element_id": entity["element_id"],
                            "type": entity["type"],
                            "description": entity["description"],
                            "mentioned_in_chunks": [],
                        }
                    if result.get("chunk_id"):
                        context["entities"][entity_id]["mentioned_in_chunks"].append(
                            result["chunk_id"],
                        )

            # Multi-hop graph traversal from seed entities found in chunks
            seed_ids: list[str] = list(context["entities"].keys())
            expansion: GraphExpansion = self._expand_entities_via_graph(
                graph,
                seed_ids,
                kg_name=kg_name,
                max_hops=max_hops,
                question_id=question_id,
                document_names=document_names,
            )
            context["graph_neighbors"] = cast(
                dict[str, RetrievedEntity],
                expansion["neighbors"],
            )
            context["traversal_paths"] = expansion["paths"]

            # Merge neighbor entities into the entity dict
            for nid, ninfo in expansion["neighbors"].items():
                if nid not in context["entities"]:
                    context["entities"][nid] = {
                        "id": nid,
                        "element_id": ninfo["element_id"],
                        "type": ninfo["type"],
                        "description": ninfo["name"],
                        "mentioned_in_chunks": [],
                        "source": "graph_traversal",
                        "min_hops": ninfo["min_hops"],
                    }

            # Relationships between ALL entities (seeds + neighbors)
            all_entity_ids: list[str] = list(context["entities"].keys())
            if all_entity_ids:
                relationship_results: list[RelationshipRow] = (
                    self._fetch_relationships_for_entity_ids(
                        graph,
                        all_entity_ids,
                        kg_name=kg_name,
                        question_id=question_id,
                        document_names=document_names,
                    )
                )
                for rel_result in relationship_results:
                    rel_key: str = (
                        rel_result.get("relationship_element_id")
                        or f"{rel_result['source']}-{rel_result['relationship_type']}-{rel_result['target']}"
                        f"-negated={bool(rel_result.get('negated', False))}"
                    )
                    if not any(
                        r.get("key") == rel_key for r in context["relationships"]
                    ):
                        context["relationships"].append(
                            {
                                "key": rel_key,
                                "source": rel_result["source"],
                                "source_element_id": rel_result["source_element_id"],
                                "target": rel_result["target"],
                                "target_element_id": rel_result["target_element_id"],
                                "type": rel_result["relationship_type"],
                                "element_id": rel_result["relationship_element_id"],
                                "negated": bool(rel_result.get("negated", False)),
                                "condition": rel_result.get("condition"),
                                "quantitative": rel_result.get("quantitative"),
                                "confidence": rel_result.get("confidence"),
                                "question_ids": rel_result.get("question_ids") or [],
                                "passage_keys": rel_result.get("passage_keys") or [],
                                "provenance_positions": rel_result.get(
                                    "provenance_positions",
                                )
                                or [],
                            },
                        )

            context["documents"] = list(context["documents"])
            context["entity_count"] = len(context["entities"])
            context["relationship_count"] = len(context["relationships"])

            return cast(GraphRetrievalContext, context)

        except Exception as e:
            logging.error(f"Error in vector similarity search: {e}")
            # Don't return error — let caller fall back to text search
            raise Exception(f"Vector search failed: {e!s}")
