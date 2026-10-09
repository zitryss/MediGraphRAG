import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast, overload

from ontographrag.rag.systems._records import (
    ChunkProjection,
    ChunkT,
    ContextT,
    CountedGraphRetrievalContext,
    EntityProjection,
    EvidencePath,
    FusedChunkRecord,
    GraphPrimitive,
    GraphRetrievalContext,
    PartialRetrievalContext,
    ProducedGraphRetrievalContext,
    RetrievalContext,
    RetrievalProvenance,
    RetrievedChunk,
    RetrievedEntity,
    RetrievedRelationship,
    ScoredCompleteRetrievalContext,
    ScoredRetrievedChunk,
    StoredGraphProperty,
    StoredGraphValue,
    StrongContextT,
    TraversalPath,
)

if TYPE_CHECKING:
    from shared.common_fn import GraphQuery
from ontographrag.rag.graph_state import summarize_context_graph_state
from ontographrag.rag.retrieval_sampling import (
    compute_candidate_limit,
    select_ranked_subset,
)


class ContextAssemblyMixin:
    """Chunk selection, context merging, fusion, and prompt formatting.

    Mixin for :class:`EnhancedRAGSystem`; method bodies are unchanged
    from the original monolithic implementation.
    """

    if TYPE_CHECKING:
        _PASSAGE_ONLY_ANSWER_KGS: frozenset[str]
        _WEAK_GRAPH_GROUNDING_THRESHOLD: float
        _QUERY_FUSION_RRF_K: int
        _MAX_CHUNKS_PER_PASSAGE: int

        @staticmethod
        def _comparison_branches(query: str) -> list[str]: ...

        @classmethod
        def _comparison_branch_match_count(
            cls, query: str, chunk: RetrievedChunk
        ) -> int: ...

        @classmethod
        def _lexical_query_overlap_count(
            cls, query: str, chunk: RetrievedChunk
        ) -> int: ...

        @classmethod
        def _comparison_branch_coverage(
            cls, query: str, chunks: Sequence[RetrievedChunk]
        ) -> int: ...

    @staticmethod
    def _chunk_identity(chunk: RetrievedChunk) -> str:
        return (
            chunk.get("chunk_id")
            or chunk.get("chunk_element_id")
            or str(chunk.get("text", ""))[:80]
        )

    @staticmethod
    def _relationship_identity(rel: RetrievedRelationship) -> str:
        return (
            rel.get("key")
            or rel.get("element_id")
            or (
                f"{rel.get('source')}-{rel.get('type')}-{rel.get('target')}"
                f"-negated={bool(rel.get('negated', False))}"
                f"-condition={str(rel.get('condition') or '').strip()}"
                f"-quantitative={str(rel.get('quantitative') or '').strip()}"
            )
        )

    @staticmethod
    def _format_relationship_label(rel: RetrievedRelationship | None) -> str:
        if not isinstance(rel, dict):
            return "RELATED"
        label: str = str(rel.get("type") or "RELATED").strip() or "RELATED"
        if rel.get("negated"):
            label = f"NOT {label}"
        condition: str = str(rel.get("condition") or "").strip()
        if condition:
            label = f"{label} [{condition}]"
        quantitative: str = str(rel.get("quantitative") or "").strip()
        if quantitative and quantitative.lower() not in condition.lower():
            label = f"{label} ({quantitative})"
        return label

    @classmethod
    def _should_use_passage_only_answer_prompt(cls, kg_name: str | None) -> bool:
        return str(kg_name or "").strip().lower() in cls._PASSAGE_ONLY_ANSWER_KGS

    @classmethod
    def _sort_chunks_for_query(
        cls, query: str, chunks: Sequence[ChunkT]
    ) -> list[ChunkT]:
        branches: list[str] = cls._comparison_branches(query)
        return sorted(
            chunks,
            key=lambda c: (
                -cls._comparison_branch_match_count(query, c) if branches else 0,
                -cls._lexical_query_overlap_count(query, c),
                -int(cast(RetrievedChunk, c).get("linked_entity_count", 0)),
                -float(cast(RetrievedChunk, c).get("score", 0.0)),
                1 if cast(RetrievedChunk, c).get("adjacent") else 0,
                int(cast(RetrievedChunk, c).get("position") or 0),
            ),
        )

    @classmethod
    def _retained_entity_ids_from_chunks(cls, chunks: list[RetrievedChunk]) -> set[str]:
        retained_eids: set[str] = set()
        eid: StoredGraphProperty
        for chunk in chunks:
            for eid in chunk.get("linked_entity_ids") or []:
                if eid:
                    retained_eids.add(str(eid))
            for entity in chunk.get("entities") or []:
                eid = entity.get("id") if isinstance(entity, dict) else None
                if eid:
                    retained_eids.add(str(eid))
        return retained_eids

    @staticmethod
    def _selected_chunk_local_scope(
        chunks: list[RetrievedChunk],
    ) -> tuple[set[int], set[str], set[str]]:
        positions: set[int] = set()
        question_ids: set[str] = set()
        passage_keys: set[str] = set()
        for chunk in chunks:
            pos: int | None = chunk.get("position")
            if isinstance(pos, (int, float)):
                positions.add(int(pos))
            qid: str | None = chunk.get("question_id")
            if qid is not None and str(qid).strip():
                qid_str: str = str(qid)
                question_ids.add(qid_str)
                passage_keys.add(f"{qid_str}::p{chunk.get('passage_index', -1)}")
        return positions, question_ids, passage_keys

    @staticmethod
    def _relationship_supported_by_selected_chunks(
        rel: RetrievalProvenance,
        *,
        selected_positions: set[int],
        selected_question_ids: set[str],
        selected_passage_keys: set[str],
    ) -> bool:
        rel_positions: set[int] = {
            int(pos)
            for pos in (rel.get("provenance_positions") or [])
            if isinstance(pos, (int, float))
        }
        if rel_positions:
            return bool(rel_positions & selected_positions)

        rel_passage_keys: set[str] = {
            str(key) for key in (rel.get("passage_keys") or []) if str(key).strip()
        }
        if rel_passage_keys:
            return bool(rel_passage_keys & selected_passage_keys)

        rel_question_ids: set[str] = {
            str(qid) for qid in (rel.get("question_ids") or []) if str(qid).strip()
        }
        if rel_question_ids:
            return bool(rel_question_ids & selected_question_ids)

        return True

    @classmethod
    def _path_supported_by_selected_chunks(
        cls,
        path_entry: TraversalPath,
        *,
        selected_positions: set[int],
        selected_question_ids: set[str],
        selected_passage_keys: set[str],
    ) -> bool:
        return cls._relationship_supported_by_selected_chunks(
            path_entry,
            selected_positions=selected_positions,
            selected_question_ids=selected_question_ids,
            selected_passage_keys=selected_passage_keys,
        )

    @overload
    @classmethod
    def _apply_final_chunk_selection(
        cls,
        *,
        query: str,
        context: GraphRetrievalContext | ProducedGraphRetrievalContext,
        max_chunks: int,
        retrieval_temperature: float,
        retrieval_shortlist_factor: int,
        retrieval_sample_id: int,
        kg_name: str | None,
    ) -> ProducedGraphRetrievalContext: ...

    @overload
    @classmethod
    def _apply_final_chunk_selection(
        cls,
        *,
        query: str,
        context: RetrievalContext,
        max_chunks: int,
        retrieval_temperature: float,
        retrieval_shortlist_factor: int,
        retrieval_sample_id: int,
        kg_name: str | None,
    ) -> CountedGraphRetrievalContext | ProducedGraphRetrievalContext: ...

    @classmethod
    def _apply_final_chunk_selection(
        cls,
        *,
        query: str,
        context: RetrievalContext,
        max_chunks: int,
        retrieval_temperature: float,
        retrieval_shortlist_factor: int,
        retrieval_sample_id: int,
        kg_name: str | None,
    ) -> CountedGraphRetrievalContext | ProducedGraphRetrievalContext:
        chunks: list[RetrievedChunk] = cls._sort_chunks_for_query(
            query, list(context.get("chunks", []))
        )
        selected_chunks: list[RetrievedChunk] = select_ranked_subset(
            chunks,
            max_items=max_chunks,
            retrieval_temperature=retrieval_temperature,
            shortlist_factor=retrieval_shortlist_factor,
            sample_id=retrieval_sample_id,
            seed_parts=("kg", kg_name, query),
            score_getter=lambda chunk: float(chunk.get("score", 0.0)),
        )

        pruned: PartialRetrievalContext = cast(PartialRetrievalContext, dict(context))
        pruned["chunks"] = selected_chunks
        retained_eids: set[str] = cls._retained_entity_ids_from_chunks(selected_chunks)

        entities: dict[str, RetrievedEntity] = context.get("entities", {}) or {}
        graph_neighbors: dict[str, RetrievedEntity] = (
            context.get("graph_neighbors", {}) or {}
        )
        relationships: list[RetrievedRelationship] = (
            context.get("relationships", []) or []
        )
        traversal_paths: list[TraversalPath] = context.get("traversal_paths", []) or []

        if retained_eids:
            selected_positions, selected_question_ids, selected_passage_keys = (
                cls._selected_chunk_local_scope(selected_chunks)
            )
            pruned["entities"] = {
                entity_id: info
                for entity_id, info in entities.items()
                if entity_id in retained_eids
            }
            pruned["graph_neighbors"] = {
                entity_id: info
                for entity_id, info in graph_neighbors.items()
                if entity_id in retained_eids
            }
            pruned["relationships"] = [
                rel
                for rel in relationships
                if rel.get("source") in retained_eids
                and rel.get("target") in retained_eids
                and cls._relationship_supported_by_selected_chunks(
                    rel,
                    selected_positions=selected_positions,
                    selected_question_ids=selected_question_ids,
                    selected_passage_keys=selected_passage_keys,
                )
            ]
            filtered_paths: list[TraversalPath] = []
            seen_paths: set[str] = set()
            for path_entry in traversal_paths:
                path_str: str = path_entry.get("path", "")
                if path_str in seen_paths:
                    continue
                node_ids: list[str] | None = path_entry.get("node_ids")
                if node_ids and not all(
                    node_id in retained_eids for node_id in node_ids
                ):
                    continue
                if not cls._path_supported_by_selected_chunks(
                    path_entry,
                    selected_positions=selected_positions,
                    selected_question_ids=selected_question_ids,
                    selected_passage_keys=selected_passage_keys,
                ):
                    continue
                seen_paths.add(path_str)
                filtered_paths.append(path_entry)
            pruned["traversal_paths"] = filtered_paths
        else:
            pruned["entities"] = entities
            pruned["graph_neighbors"] = graph_neighbors
            pruned["relationships"] = relationships
            pruned["traversal_paths"] = traversal_paths

        pruned["documents"] = list(
            {
                chunk.get("document")
                for chunk in selected_chunks
                if chunk.get("document")
            }
        )
        pruned["total_score"] = float(
            sum(float(chunk.get("score", 0.0)) for chunk in selected_chunks)
        )
        pruned["entity_count"] = len(pruned.get("entities", {}))
        pruned["relationship_count"] = len(pruned.get("relationships", []))
        pruned["retrieval_sampling"] = {
            "temperature": float(retrieval_temperature or 0.0),
            "shortlist_factor": int(retrieval_shortlist_factor or 1),
            "sample_id": int(retrieval_sample_id or 0),
            "candidate_limit": int(
                compute_candidate_limit(
                    max_chunks,
                    retrieval_temperature,
                    retrieval_shortlist_factor,
                )
            ),
        }
        pruned["graph_state"] = summarize_context_graph_state(pruned)
        return cast(CountedGraphRetrievalContext, pruned)

    @classmethod
    def _graph_context_is_meaningful(
        cls, query: str, context: RetrievalContext | None
    ) -> bool:
        """
        Decide whether the graph-centric retrieval signal is strong enough to stay primary.

        The goal is not to prove correctness, only to detect obviously weak graph
        retrieval so we can switch to a vector-first hybrid context instead of
        over-trusting whatever nodes/paths were found.
        """
        if not context or not context.get("chunks"):
            return False

        grounding: float | None = context.get("grounding_quality")
        has_structure: bool = bool(context.get("relationships")) or bool(
            context.get("traversal_paths")
        )
        if (
            grounding is not None
            and grounding < cls._WEAK_GRAPH_GROUNDING_THRESHOLD
            and not has_structure
        ):
            return False

        branches: list[str] = cls._comparison_branches(query)
        if branches:
            return cls._comparison_branch_coverage(
                query, context.get("chunks", [])
            ) >= len(branches)

        return has_structure or (
            grounding is not None and grounding >= cls._WEAK_GRAPH_GROUNDING_THRESHOLD
        )

    @classmethod
    def _append_adjacent_chunks_to_context(
        cls,
        graph: "GraphQuery",
        context: ContextT,
        *,
        kg_name: str | None,
        question_id: str | None,
        document_names: list[str] | None,
        max_adjacent: int,
    ) -> ContextT:
        """Append position-adjacent chunks so answer spans split across boundaries survive.

        Vanilla retrieval already recovers adjacent chunks from the same local passage.
        Apply the same recovery here so graph-first and vector-first KG retrieval paths
        do not lose answer-bearing sentences simply because the retriever hit a nearby span.
        """
        context_view: RetrievalContext = context
        chunks: list[RetrievedChunk] = list(context_view.get("chunks") or [])
        if not chunks:
            return context

        seed_element_ids: list[str | None] = [
            chunk.get("parent_chunk_element_id") or chunk.get("chunk_element_id")
            for chunk in chunks
            if chunk.get("parent_chunk_element_id") or chunk.get("chunk_element_id")
        ]
        if not seed_element_ids:
            return context

        seen_ids: set[str] = {
            str(chunk_id)
            for chunk in chunks
            for chunk_id in [chunk.get("chunk_element_id")]
            if chunk_id
        }
        if not seen_ids:
            return context

        params: dict[str, object] = {
            "element_ids": seed_element_ids,
            "max_adjacent": max(1, int(max_adjacent or 1)),
        }
        scope_conditions: list[str] = []
        if kg_name:
            scope_conditions.append("d.kgName = $kg_name")
            params["kg_name"] = kg_name
        if document_names:
            scope_conditions.append("d.fileName IN $document_names")
            params["document_names"] = list(document_names)
        if question_id:
            scope_conditions.append("seed.questionId = $question_id")
            scope_conditions.append("adj.questionId = $question_id")
            params["question_id"] = question_id
        scope_clause: str = ""
        if scope_conditions:
            scope_clause = "\n                  AND " + "\n                  AND ".join(
                scope_conditions
            )

        adj_query: str = f"""
        UNWIND $element_ids AS eid
        MATCH (seed:Chunk)-[:PART_OF]->(d:Document)
        WHERE elementId(seed) = eid
        MATCH (adj:Chunk)-[:PART_OF]->(d)
        WHERE ((
            seed.questionId IS NOT NULL
            AND adj.questionId = seed.questionId
            AND coalesce(adj.passageIndex, -1) = coalesce(seed.passageIndex, -1)
            AND abs(
                coalesce(adj.chunkLocalIndex, adj.position)
                - coalesce(seed.chunkLocalIndex, seed.position)
            ) = 1
        ) OR (
            seed.questionId IS NULL
            AND adj.questionId IS NULL
            AND abs(adj.position - seed.position) = 1
        )){scope_clause}
        OPTIONAL MATCH (adj)-[:HAS_ENTITY]->(entity:__Entity__)
        WITH adj, d, collect(DISTINCT entity) AS chunk_entities
        RETURN DISTINCT
            adj.text AS text,
            adj.id AS chunk_id,
            elementId(adj) AS chunk_element_id,
            adj.position AS position,
            adj.source AS source,
            adj.questionId AS question_id,
            adj.passageIndex AS passage_index,
            adj.chunkLocalIndex AS chunk_local_index,
            0.0 AS score,
            d.fileName AS document,
            d.kgName AS kg_name,
            [entity IN chunk_entities WHERE entity IS NOT NULL | {{
                id: coalesce(entity.id, entity.name),
                element_id: elementId(entity),
                type: coalesce(entity.type, 'Entity'),
                description: coalesce(entity.name, '')
            }}] AS entities
        ORDER BY d.fileName ASC,
                 coalesce(adj.questionId, ''),
                 coalesce(adj.passageIndex, -1),
                 coalesce(adj.chunkLocalIndex, adj.position),
                 adj.position ASC
        LIMIT $max_adjacent
        """
        try:
            adj_results: list[ChunkProjection] = (
                cast(list[ChunkProjection], graph.query(adj_query, params)) or []
            )
        except Exception as adj_err:
            logging.debug(
                "Enhanced adjacent chunk expansion failed (non-fatal): %s", adj_err
            )
            return context

        if not adj_results:
            return context

        updated: PartialRetrievalContext = cast(PartialRetrievalContext, dict(context))
        updated_chunks: list[RetrievedChunk] = list(chunks)
        documents: set[StoredGraphProperty] = set(context_view.get("documents") or [])
        for result in adj_results:
            chunk_element_id: str | None = result.get("chunk_element_id")
            if not chunk_element_id or str(chunk_element_id) in seen_ids:
                continue
            entities: list[EntityProjection] = result.get("entities") or []
            adjacent_chunk: ScoredRetrievedChunk = {
                "text": result["text"],
                "chunk_id": result["chunk_id"],
                "chunk_element_id": chunk_element_id,
                "position": result.get("position"),
                "source": result.get("source"),
                "question_id": result.get("question_id"),
                "passage_index": result.get("passage_index"),
                "chunk_local_index": result.get("chunk_local_index"),
                "score": 0.0,
                "document": result["document"],
                "kg_name": result.get("kg_name"),
                "entities": entities,
                "linked_entity_ids": [
                    entity.get("id")
                    for entity in entities
                    if isinstance(entity, dict) and entity.get("id")
                ],
                "linked_entity_count": len(entities),
                "adjacent": True,
            }
            updated_chunks.append(adjacent_chunk)
            documents.add(result["document"])
            seen_ids.add(str(chunk_element_id))

        updated["chunks"] = updated_chunks
        updated["documents"] = list(documents)
        return cast(ContextT, updated)

    @overload
    @classmethod
    def _merge_retrieval_contexts(
        cls,
        primary: StrongContextT,
        secondary: GraphRetrievalContext
        | ProducedGraphRetrievalContext
        | ScoredCompleteRetrievalContext
        | None,
        *,
        max_chunks: int,
        search_method: str,
        secondary_limit: int | None = None,
        min_secondary_score: float | None = None,
    ) -> StrongContextT: ...

    @overload
    @classmethod
    def _merge_retrieval_contexts(
        cls,
        primary: RetrievalContext,
        secondary: RetrievalContext | None,
        *,
        max_chunks: int,
        search_method: str,
        secondary_limit: int | None = None,
        min_secondary_score: float | None = None,
    ) -> RetrievalContext: ...

    @classmethod
    def _merge_retrieval_contexts(
        cls,
        primary: RetrievalContext,
        secondary: RetrievalContext | None,
        *,
        max_chunks: int,
        search_method: str,
        secondary_limit: int | None = None,
        min_secondary_score: float | None = None,
    ) -> RetrievalContext:
        """Merge two retrieval contexts while keeping primary chunks dominant."""
        primary_view: RetrievalContext = primary
        if not secondary or not secondary.get("chunks"):
            merged: PartialRetrievalContext = cast(
                PartialRetrievalContext, dict(primary)
            )
            merged["search_method"] = search_method
            return merged

        merged = cast(PartialRetrievalContext, dict(primary))
        merged["chunks"] = [
            cast(RetrievedChunk, dict(chunk))
            for chunk in primary_view.get("chunks", [])
        ]
        merged["entities"] = dict(primary_view.get("entities", {}))
        merged["relationships"] = list(primary_view.get("relationships", []))
        merged["graph_neighbors"] = dict(primary_view.get("graph_neighbors", {}))
        merged["traversal_paths"] = list(primary_view.get("traversal_paths", []))

        existing_chunk_ids: set[str] = set()
        for chunk in merged["chunks"]:
            chunk_key: str = cls._chunk_identity(chunk)
            if chunk_key:
                existing_chunk_ids.add(str(chunk_key))
            parent_chunk_key: str | None = chunk.get("parent_chunk_id")
            if parent_chunk_key:
                existing_chunk_ids.add(str(parent_chunk_key))
        new_chunks: list[RetrievedChunk] = []
        for chunk in secondary.get("chunks", []):
            cid: str = cls._chunk_identity(chunk)
            parent_cid: str | None = chunk.get("parent_chunk_id")
            if cid in existing_chunk_ids or (
                parent_cid and str(parent_cid) in existing_chunk_ids
            ):
                continue
            existing_chunk_ids.add(str(cid))
            if parent_cid:
                existing_chunk_ids.add(str(parent_cid))
            candidate: RetrievedChunk = cast(RetrievedChunk, dict(chunk))
            if min_secondary_score is not None:
                candidate["score"] = max(
                    float(candidate.get("score", 0.0)), min_secondary_score
                )
            new_chunks.append(candidate)
            if secondary_limit is not None and len(new_chunks) >= secondary_limit:
                break

        merged["chunks"].extend(new_chunks)
        merged["chunks"] = cls._sort_chunks_for_query(
            primary_view.get("query", ""), merged["chunks"]
        )
        merged["chunks"] = merged["chunks"][:max_chunks]

        for entity_id, entity_info in secondary.get("entities", {}).items():
            merged["entities"].setdefault(entity_id, entity_info)
        for entity_id, entity_info in secondary.get("graph_neighbors", {}).items():
            merged["graph_neighbors"].setdefault(entity_id, entity_info)

        seen_rel_keys: set[str] = {
            cls._relationship_identity(rel) for rel in merged["relationships"]
        }
        for rel in secondary.get("relationships", []):
            rel_key: str = cls._relationship_identity(rel)
            if rel_key in seen_rel_keys:
                continue
            seen_rel_keys.add(rel_key)
            merged["relationships"].append(rel)

        seen_paths: set[str] = {
            path.get("path", "") for path in merged["traversal_paths"]
        }
        for path_entry in secondary.get("traversal_paths", []):
            path_str: str = path_entry.get("path", "")
            if not path_str or path_str in seen_paths:
                continue
            seen_paths.add(path_str)
            merged["traversal_paths"].append(path_entry)

        merged["documents"] = list(
            {
                *(primary_view.get("documents") or []),
                *(secondary.get("documents") or []),
                *(
                    chunk.get("document")
                    for chunk in merged["chunks"]
                    if chunk.get("document")
                ),
            }
        )
        merged["total_score"] = float(
            sum(float(chunk.get("score", 0.0)) for chunk in merged["chunks"])
        )
        merged["entity_count"] = len(merged["entities"])
        merged["relationship_count"] = len(merged["relationships"])
        merged["search_method"] = search_method
        # Always preserve the highest grounding_quality seen across both contexts
        # so that a weak-graph fallback to vector doesn't erase genuine entity grounding.
        gq: float = max(
            primary_view.get("grounding_quality") or 0.0,
            (secondary.get("grounding_quality") if secondary else None) or 0.0,
        )
        merged["grounding_quality"] = gq
        merged["seed_entity_count"] = max(
            int(primary_view.get("seed_entity_count") or 0),
            int((secondary or {}).get("seed_entity_count") or 0),
        )
        return merged

    @overload
    @classmethod
    def _fuse_contexts_with_rrf(
        cls,
        *,
        query: str,
        contexts: Sequence[ProducedGraphRetrievalContext],
        max_chunks: int,
        kg_name: str | None,
        retrieval_temperature: float,
        retrieval_shortlist_factor: int,
        retrieval_sample_id: int,
        search_method: str,
    ) -> ProducedGraphRetrievalContext | None: ...

    @overload
    @classmethod
    def _fuse_contexts_with_rrf(
        cls,
        *,
        query: str,
        contexts: Sequence[CountedGraphRetrievalContext],
        max_chunks: int,
        kg_name: str | None,
        retrieval_temperature: float,
        retrieval_shortlist_factor: int,
        retrieval_sample_id: int,
        search_method: str,
    ) -> CountedGraphRetrievalContext | ProducedGraphRetrievalContext | None: ...

    @overload
    @classmethod
    def _fuse_contexts_with_rrf(
        cls,
        *,
        query: str,
        contexts: Sequence[RetrievalContext],
        max_chunks: int,
        kg_name: str | None,
        retrieval_temperature: float,
        retrieval_shortlist_factor: int,
        retrieval_sample_id: int,
        search_method: str,
    ) -> RetrievalContext | None: ...

    @classmethod
    def _fuse_contexts_with_rrf(
        cls,
        *,
        query: str,
        contexts: Sequence[RetrievalContext],
        max_chunks: int,
        kg_name: str | None,
        retrieval_temperature: float,
        retrieval_shortlist_factor: int,
        retrieval_sample_id: int,
        search_method: str,
    ) -> RetrievalContext | None:
        usable_contexts: list[RetrievalContext] = [
            ctx for ctx in contexts if ctx and ctx.get("chunks")
        ]
        if not usable_contexts:
            return None
        if len(usable_contexts) == 1:
            return usable_contexts[0]

        merged: CountedGraphRetrievalContext = {
            "query": query,
            "chunks": [],
            "entities": {},
            "relationships": [],
            "graph_neighbors": {},
            "traversal_paths": [],
            "documents": [],
            "total_score": 0.0,
            "entity_count": 0,
            "relationship_count": 0,
            "search_method": search_method,
            "kg_name": kg_name,
            "grounding_quality": 0.0,
            "seed_entity_count": 0,
        }
        chunk_records: dict[str, FusedChunkRecord] = {}
        seen_rel_keys: set[str] = set()
        seen_paths: set[str] = set()

        for context in usable_contexts:
            ranked_chunks: list[RetrievedChunk] = cls._sort_chunks_for_query(
                query, list(context.get("chunks", []))
            )
            for rank, chunk in enumerate(ranked_chunks, 1):
                chunk_key: str = str(
                    chunk.get("parent_chunk_id")
                    or chunk.get("chunk_id")
                    or chunk.get("chunk_element_id")
                    or str(chunk.get("text", ""))[:120]
                )
                record: FusedChunkRecord | None = chunk_records.get(chunk_key)
                if record is None:
                    record = {
                        "chunk": cast(RetrievedChunk, dict(chunk)),
                        "rrf": 0.0,
                        "support_count": 0,
                        "best_score": float(chunk.get("score", 0.0)),
                    }
                    chunk_records[chunk_key] = record
                record["rrf"] += 1.0 / float(cls._QUERY_FUSION_RRF_K + rank)
                record["support_count"] += 1
                current_score: float = float(chunk.get("score", 0.0))
                if current_score >= record["best_score"]:
                    record["chunk"] = cast(RetrievedChunk, dict(chunk))
                    record["best_score"] = current_score

            for entity_id, entity_info in (context.get("entities") or {}).items():
                merged["entities"].setdefault(entity_id, entity_info)
            for entity_id, entity_info in (
                context.get("graph_neighbors") or {}
            ).items():
                merged["graph_neighbors"].setdefault(entity_id, entity_info)
            for rel in context.get("relationships", []) or []:
                rel_key: str = cls._relationship_identity(rel)
                if rel_key in seen_rel_keys:
                    continue
                seen_rel_keys.add(rel_key)
                merged["relationships"].append(rel)
            for path_entry in context.get("traversal_paths", []) or []:
                path_str: str = path_entry.get("path", "")
                if not path_str or path_str in seen_paths:
                    continue
                seen_paths.add(path_str)
                merged["traversal_paths"].append(path_entry)
            merged["grounding_quality"] = max(
                float(merged.get("grounding_quality") or 0.0),
                float(context.get("grounding_quality") or 0.0),
            )
            merged["seed_entity_count"] = max(
                int(merged.get("seed_entity_count") or 0),
                int(context.get("seed_entity_count") or 0),
            )

        fused_chunks: list[RetrievedChunk] = []
        for record in chunk_records.values():
            chunk = cast(RetrievedChunk, dict(record["chunk"]))
            chunk["rrf_score"] = record["rrf"]
            chunk["retrieval_support_count"] = record["support_count"]
            chunk["score"] = (
                record["rrf"]
                + 0.05 * min(record["support_count"], 3)
                + 0.01 * float(record["best_score"])
            )
            fused_chunks.append(chunk)

        merged["chunks"] = cls._sort_chunks_for_query(query, fused_chunks)
        merged["documents"] = list(
            {
                chunk.get("document")
                for chunk in merged["chunks"]
                if chunk.get("document")
            }
        )
        merged["total_score"] = float(
            sum(float(chunk.get("score", 0.0)) for chunk in merged["chunks"])
        )
        merged["entity_count"] = len(merged["entities"])
        merged["relationship_count"] = len(merged["relationships"])
        return cls._apply_final_chunk_selection(
            query=query,
            context=merged,
            max_chunks=max_chunks,
            retrieval_temperature=retrieval_temperature,
            retrieval_shortlist_factor=retrieval_shortlist_factor,
            retrieval_sample_id=retrieval_sample_id,
            kg_name=kg_name,
        )

    @staticmethod
    def _chunk_rank_key(chunk: RetrievedChunk) -> tuple[float, float, int]:
        return (
            -float(chunk.get("score", 0.0)),
            -int(chunk.get("linked_entity_count", 0)),
            int(chunk.get("position") or 0),
        )

    def _apply_per_passage_chunk_cap(self, chunks: list[ChunkT]) -> list[ChunkT]:
        """Keep the highest-ranked chunks from each source passage."""
        _passage_cap: int = self._MAX_CHUNKS_PER_PASSAGE
        if _passage_cap <= 0:
            return list(chunks)

        ranked_chunks: list[ChunkT] = sorted(chunks, key=self._chunk_rank_key)
        passage_counts: dict[str | int, int] = {}
        capped_chunks: list[ChunkT] = []
        pkey: str | int
        for chunk in ranked_chunks:
            chunk_view: RetrievedChunk = chunk
            qid_val: str = chunk_view.get("question_id") or ""
            pidx_val: int | None = chunk_view.get("passage_index")
            if pidx_val is not None and qid_val:
                pkey = f"{qid_val}::p{pidx_val}"
            else:
                pkey = chunk_view.get("chunk_id") or id(chunk)
            count: int = passage_counts.get(pkey, 0)
            if count < _passage_cap:
                capped_chunks.append(chunk)
                passage_counts[pkey] = count + 1
            else:
                logging.debug(
                    "Entity-first: capped passage %s at %d chunks",
                    pkey,
                    _passage_cap,
                )
        return capped_chunks or ranked_chunks

    def _build_evidence_block(self, context: RetrievalContext) -> str:
        """
        KG2RAG-style evidence organization.

        Groups graph traversal paths with the text passages that support them
        (overlap in linked entity IDs), then lists remaining passages as
        "Additional Evidence".  This makes multi-hop reasoning chains explicit
        to the LLM and reduces the chance of hallucinated bridge steps.
        """
        chunks: Sequence[RetrievedChunk] = context.get("chunks", [])
        traversal_paths: list[TraversalPath] = context.get("traversal_paths", [])
        relationships: list[RetrievedRelationship] = context.get("relationships", [])
        entities: dict[str, RetrievedEntity] = context.get("entities", {})

        def _chunk_passage_key(chunk: RetrievedChunk) -> str | None:
            qid: str | None = chunk.get("question_id")
            if qid is None or not str(qid).strip():
                return None
            return f"{qid}::p{chunk.get('passage_index', -1)}"

        def _chunk_supports_path(
            chunk: RetrievedChunk, path_info: EvidencePath
        ) -> bool:
            chunk_positions: set[int] = set()
            pos: int | None = chunk.get("position")
            if isinstance(pos, (int, float)):
                chunk_positions.add(int(pos))
            path_positions: set[int] = {
                int(p)
                for p in (path_info.get("provenance_positions") or [])
                if isinstance(p, (int, float))
            }
            if path_positions:
                return bool(chunk_positions & path_positions)

            chunk_passage_key: str | None = _chunk_passage_key(chunk)
            path_passage_keys: set[str] = {
                str(key)
                for key in (path_info.get("passage_keys") or [])
                if str(key).strip()
            }
            if path_passage_keys and chunk_passage_key is not None:
                return chunk_passage_key in path_passage_keys

            chunk_entity_ids: set[StoredGraphProperty] = set(
                chunk.get("linked_entity_ids") or []
            )
            path_nodes: set[str] = set(path_info.get("node_ids") or [])
            if not chunk_entity_ids or not path_nodes:
                return False

            overlap: set[StoredGraphProperty] = chunk_entity_ids & path_nodes
            # When provenance metadata is absent, require at least two path
            # entities for a chunk to count as chain support. This avoids
            # treating a passage that merely mentions one endpoint as evidence
            # for the whole relation/path.
            required_overlap: int = min(2, len(path_nodes))
            return len(overlap) >= required_overlap

        # Index chunks by their linked entities for fast lookup.
        chunk_by_id: dict[str | None, RetrievedChunk] = {
            c["chunk_id"]: c for c in chunks if c.get("chunk_id")
        }

        # Build a unified path list from both traversal paths and direct relationships.
        # Each entry: {path_str, node_ids, hops}
        all_paths: list[EvidencePath] = []
        for p in sorted(traversal_paths, key=lambda x: x.get("hops", 99)):
            all_paths.append(
                {
                    "path_str": p.get("path", ""),
                    "node_ids": set(p.get("node_ids") or []),
                    "hops": p.get("hops", 1),
                    "provenance_positions": list(p.get("provenance_positions") or []),
                    "passage_keys": list(p.get("passage_keys") or []),
                }
            )
        entity_name_lookup: dict[str, StoredGraphValue] = {
            eid: (info.get("name") or info.get("description") or eid)
            for eid, info in entities.items()
        }
        MAX_REL_PATHS: int = max(0, 25 - len(all_paths))
        for rel in relationships[:MAX_REL_PATHS]:
            src: str | list[GraphPrimitive] = entity_name_lookup.get(
                rel["source"], rel["source"]
            )[:50]
            tgt: str | list[GraphPrimitive] = entity_name_lookup.get(
                rel["target"], rel["target"]
            )[:50]
            rel_label: str = self._format_relationship_label(rel)
            all_paths.append(
                {
                    "path_str": f"{src} --{rel_label}--> {tgt}",
                    "node_ids": {rel["source"], rel["target"]},
                    "hops": 1,
                    "provenance_positions": list(rel.get("provenance_positions") or []),
                    "passage_keys": list(rel.get("passage_keys") or []),
                }
            )

        # Associate each chunk with up to two supporting chains so shared bridge
        # evidence can appear in multiple reasoning chains without flooding all of them.
        chunk_chain_count: dict[str | None, int] = {}
        _MAX_CHAINS_PER_CHUNK: int = 2
        chain_blocks: list[str] = []
        ungrounded_paths: list[str] = []  # paths with no supporting passage
        for chain_idx, path_info in enumerate(all_paths, 1):
            path_str: str = path_info["path_str"]
            if not path_str:
                continue
            supporting: list[RetrievedChunk] = []
            for cid, chunk in chunk_by_id.items():
                if chunk_chain_count.get(cid, 0) >= _MAX_CHAINS_PER_CHUNK:
                    continue
                if _chunk_supports_path(chunk, path_info):
                    supporting.append(chunk)
            if not supporting:
                # Collect passage-unsupported paths separately; they are emitted
                # in a clearly-labelled structural hints section so the LLM knows
                # not to treat them as confirmed textual evidence.
                ungrounded_paths.append(f"  {path_info['hops']} hop: {path_str}")
            else:
                lines: list[str] = [
                    (
                        f"Chain {chain_idx} ({path_info['hops']} hop{'s' if path_info['hops'] != 1 else ''}): "
                        f"{path_str}"
                    )
                ]
                for i, chunk in enumerate(supporting[:3], 1):
                    text: str = (chunk.get("text") or "").strip()[:400]
                    lines.append(f"  [P{i}] {text}")
                    chunk_chain_count[chunk["chunk_id"]] = (
                        chunk_chain_count.get(chunk["chunk_id"], 0) + 1
                    )
                chain_blocks.append("\n".join(lines))

        # Remaining chunks become "Additional Evidence".
        additional: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            if chunk_chain_count.get(chunk.get("chunk_id"), 0) == 0:
                text = (chunk.get("text") or "").strip()
                additional.append(f"[{i}] {text}")

        sections: list[str] = []
        if chain_blocks:
            sections.append("REASONING CHAINS:\n" + "\n\n".join(chain_blocks))
        if additional:
            sections.append("ADDITIONAL PASSAGES:\n" + "\n\n".join(additional))
        if ungrounded_paths:
            # Clearly labelled so the LLM treats these as weak structural hints
            # only, not as textual evidence.  The prompt instructs the LLM to
            # ignore this section if passage evidence is available.
            sections.append(
                "STRUCTURAL HINTS (graph paths — no passage retrieved; "
                "use only if no passage evidence addresses the question):\n"
                + "\n".join(ungrounded_paths)
            )
        if not sections:
            sections.append("(No evidence retrieved)")

        return "\n\n".join(sections)

    def _build_passages_only_evidence_block(self, context: RetrievalContext) -> str:
        chunks: Sequence[RetrievedChunk] = context.get("chunks", []) or []
        if not chunks:
            return "(No passages retrieved)"
        lines: list[str] = []
        for idx, chunk in enumerate(chunks, 1):
            text: str = (chunk.get("text") or "").strip()
            if text:
                lines.append(f"[{idx}] {text}")
        if not lines:
            return "(No passages retrieved)"
        return "PASSAGES:\n" + "\n\n".join(lines)

    def format_context_for_llm(self, context: RetrievalContext) -> tuple[str, str, str]:
        """
        Format the context for the LLM prompt.

        Returns:
            (evidence_block, formatted_entities, formatted_paths)

        evidence_block is the KG2RAG-style organized evidence string passed to
        {evidence_block} in the prompt template.  formatted_entities and
        formatted_paths are retained for backward-compatible callers.
        """
        evidence_block: str = self._build_evidence_block(context)

        # --- legacy: flat chunk list (for callers that use formatted_context directly) ---
        chunk_texts: list[str] = []
        for _i, chunk in enumerate(context["chunks"], 1):
            doc: StoredGraphValue = chunk.get("document", "unknown")
            chunk_text: str = f"[Source: {doc}]\n{chunk['text']}\n"
            chunk_texts.append(chunk_text)
        _formatted_context: str = "\n".join(chunk_texts)

        # --- graph paths (flat, for backward compat) ---
        path_lines: list[str] = []
        traversal_paths: list[TraversalPath] = context.get("traversal_paths", [])
        if traversal_paths:
            sorted_paths: list[TraversalPath] = sorted(
                traversal_paths, key=lambda p: p.get("hops", 99)
            )
            path_lines.extend([f"  {p['path']}" for p in sorted_paths])
        entity_name_lookup: dict[str, StoredGraphValue] = {
            eid: (info.get("description") or eid)
            for eid, info in context.get("entities", {}).items()
        }
        MAX_RELATIONSHIPS: int = 25
        for rel in context.get("relationships", [])[:MAX_RELATIONSHIPS]:
            src_name: str | list[GraphPrimitive] = entity_name_lookup.get(
                rel["source"], rel["source"]
            )[:50]
            tgt_name: str | list[GraphPrimitive] = entity_name_lookup.get(
                rel["target"], rel["target"]
            )[:50]
            rel_type: str = self._format_relationship_label(rel)
            path_lines.append(f"  {src_name} --{rel_type}--> {tgt_name}")
        formatted_paths: str = (
            "\n".join(path_lines) if path_lines else "(No graph relationships found)"
        )

        # --- entities ---
        entity_texts: list[str] = []
        seed_entities: dict[str, RetrievedEntity] = {
            eid: info
            for eid, info in context["entities"].items()
            if info.get("source") != "graph_traversal"
        }
        neighbor_entities: dict[str, RetrievedEntity] = {
            eid: info
            for eid, info in context["entities"].items()
            if info.get("source") == "graph_traversal"
        }

        MAX_SEED_ENTITIES: int = 15
        MAX_NEIGHBOR_ENTITIES: int = 10

        if seed_entities:
            entity_texts.append("Seed entities (directly from retrieved chunks):")
            for entity_id, info in list(seed_entities.items())[:MAX_SEED_ENTITIES]:
                eid: StoredGraphValue = info.get("id") or entity_id or "Unknown"
                etype: str = info.get("type") or "Unknown"
                line: str = f"  - {eid} | Type: {etype}"
                desc: StoredGraphProperty = info.get("description")
                if desc and desc != eid:
                    line += f" | {desc}"
                entity_texts.append(line)
            if len(seed_entities) > MAX_SEED_ENTITIES:
                entity_texts.append(
                    f"  ... ({len(seed_entities) - MAX_SEED_ENTITIES} more omitted)"
                )

        if neighbor_entities:
            shown: list[tuple[str, RetrievedEntity]] = list(neighbor_entities.items())[
                :MAX_NEIGHBOR_ENTITIES
            ]
            entity_texts.append(
                f"\nGraph-traversal neighbors ({len(neighbor_entities)} discovered, showing top {len(shown)}):"
            )
            for entity_id, info in shown:
                eid = info.get("id") or entity_id or "Unknown"
                etype = info.get("type") or "Unknown"
                hops: int | str = info.get("min_hops", "?")
                line = f"  - {eid} | Type: {etype} | {hops}-hop neighbor"
                desc = info.get("description")
                if desc and desc != eid:
                    line += f" | {desc}"
                entity_texts.append(line)

        formatted_entities: str = (
            "\n".join(entity_texts) if entity_texts else "(No entities found)"
        )

        return evidence_block, formatted_entities, formatted_paths
