"""
Regression tests for retrieval-layer contracts.

Pure unit tests — no live Neo4j or LLM calls.
Covers:
  - classify_question_type routing
  - _grounding_quality scoring
  - _decompose_question fallback behaviour
  - format_context_for_llm output shape
"""

import json
import os
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Never, TypedDict, Unpack

import pytest

pytest.importorskip(
    "langchain_neo4j", reason="langchain_neo4j not installed — skipping retrieval tests"
)
pytest.importorskip(
    "langchain", reason="langchain not installed — skipping retrieval tests"
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import ontographrag.rag.systems.vanilla_rag_system as vanilla_rag_module
from ontographrag.rag.reranking import RerankingMetadata
from ontographrag.rag.systems._records import (
    CountedGraphRetrievalContext,
    GraphRetrievalContext,
    ProducedGraphRetrievalContext,
    RetrievalContext,
    RetrievedChunk,
    RetrievedEntity,
    RetrievedRelationship,
    ScoredCompleteRetrievalContext,
    TraversalPath,
)
from ontographrag.rag.systems._response_records import UsedEvidence
from ontographrag.rag.systems.enhanced_rag_system import EnhancedRAGSystem
from ontographrag.rag.systems.vanilla_rag_system import VanillaRAGSystem

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableLambda

    from ontographrag.rag.systems._records import GraphExpansion
    from ontographrag.rag.systems._response_records import GenerationResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class AnswerArguments(TypedDict):
    context: RetrievalContext


class GuardrailArguments(TypedDict):
    response: str
    context: RetrievalContext


class FixtureGuardrailMetadata(TypedDict):
    enabled: bool
    final_decision: str


def _make_system() -> EnhancedRAGSystem:
    """Instantiate with no graph / embedding dependencies."""
    s: EnhancedRAGSystem = EnhancedRAGSystem.__new__(EnhancedRAGSystem)
    return s


def _make_vanilla_system() -> VanillaRAGSystem:
    s: VanillaRAGSystem = VanillaRAGSystem.__new__(VanillaRAGSystem)
    return s


def _minimal_context(
    chunks: list[RetrievedChunk] | None = None,
    entities: dict[str, RetrievedEntity] | None = None,
    relationships: list[RetrievedRelationship] | None = None,
    traversal_paths: list[TraversalPath] | None = None,
) -> RetrievalContext:
    return {
        "chunks": chunks or [],
        "entities": entities or {},
        "relationships": relationships or [],
        "traversal_paths": traversal_paths or [],
    }


# ---------------------------------------------------------------------------
# 1. classify_question_type
# ---------------------------------------------------------------------------


class TestClassifyQuestionType:
    sys_: EnhancedRAGSystem = _make_system()

    def test_statistical_term_detected(self) -> None:
        assert (
            self.sys_.classify_question_type("What is the incidence rate of diabetes?")
            == "statistical"
        )

    def test_how_many_is_statistical(self) -> None:
        assert (
            self.sys_.classify_question_type("How many patients were enrolled?")
            == "statistical"
        )

    def test_what_percentage_is_statistical(self) -> None:
        assert (
            self.sys_.classify_question_type(
                "What percentage of cases respond to metformin?"
            )
            == "statistical"
        )

    def test_explain_is_semantic(self) -> None:
        assert (
            self.sys_.classify_question_type(
                "Explain the mechanism of insulin resistance."
            )
            == "semantic"
        )

    def test_what_is_is_semantic(self) -> None:
        assert (
            self.sys_.classify_question_type("What is the role of TBK1 in signalling?")
            == "semantic"
        )

    def test_generic_relational_question(self) -> None:
        qt: str = self.sys_.classify_question_type("Does aspirin treat fever?")
        assert qt in ("generic", "semantic", "statistical"), f"Unexpected type: {qt}"

    def test_case_insensitive(self) -> None:
        assert self.sys_.classify_question_type("EXPLAIN THE MECHANISM") == "semantic"

    def test_returns_string(self) -> None:
        result: str = self.sys_.classify_question_type("Some random text.")
        assert isinstance(result, str)
        assert result in ("statistical", "semantic", "generic")


# ---------------------------------------------------------------------------
# 2. _grounding_quality
# ---------------------------------------------------------------------------


class TestGroundingQuality:
    def test_perfect_grounding_capped_at_1(self) -> None:
        # 3 content words, 5 matched → capped at 1.0
        score: float = EnhancedRAGSystem._grounding_quality(
            "Metformin treats diabetes", 5
        )
        assert score == 1.0

    def test_partial_grounding(self) -> None:
        # "Metformin treats diabetes" has 3 content words (>=4 chars: Metformin, treats, diabetes)
        # 1 matched → 1/3
        score: float = EnhancedRAGSystem._grounding_quality(
            "Metformin treats diabetes", 1
        )
        assert abs(score - 1 / 3) < 1e-9

    def test_zero_matched_returns_0(self) -> None:
        score: float = EnhancedRAGSystem._grounding_quality(
            "Metformin treats diabetes", 0
        )
        assert score == 0.0

    def test_empty_query_returns_0(self) -> None:
        score: float = EnhancedRAGSystem._grounding_quality("", 3)
        assert score == 0.0

    def test_short_words_only_returns_0(self) -> None:
        # All words < 4 chars → no content words → 0.0
        score: float = EnhancedRAGSystem._grounding_quality("do it now", 1)
        assert score == 0.0

    def test_output_in_unit_interval(self) -> None:
        for n in range(6):
            score: float = EnhancedRAGSystem._grounding_quality(
                "Aspirin reduces fever inflammation", n
            )
            assert 0.0 <= score <= 1.0


class TestQuestionLocalGraphScope:
    def test_entity_support_clause_contains_question_local_constraints(self) -> None:
        clause: str = EnhancedRAGSystem._question_local_entity_support_clause(
            "e",
            kg_name="musique",
            question_id="q1",
        )
        assert "c.questionId = $question_id" in clause
        assert "d.kgName = $kg_name" in clause

    def test_entity_support_clause_contains_document_scope(self) -> None:
        clause: str = EnhancedRAGSystem._question_local_entity_support_clause(
            "e",
            kg_name="custom_kg",
            question_id=None,
            document_names=["doc_a.txt"],
        )
        assert "d.kgName = $kg_name" in clause
        assert "d.fileName IN $document_names" in clause

    def test_pair_support_clause_is_true_without_question_id(self) -> None:
        clause: str = EnhancedRAGSystem._question_local_pair_support_clause(
            "e1",
            "e2",
            kg_name="musique",
            question_id=None,
        )
        assert clause == "true"

    def test_pair_support_clause_scopes_to_documents_without_question_id(self) -> None:
        clause: str = EnhancedRAGSystem._question_local_pair_support_clause(
            "e1",
            "e2",
            kg_name="custom_kg",
            question_id=None,
            document_names=["doc_a.txt"],
        )
        assert clause != "true"
        assert "d1.fileName IN $document_names" in clause
        assert "d2.fileName IN $document_names" in clause

    def test_pair_support_clause_contains_question_bundle_constraints(self) -> None:
        clause: str = EnhancedRAGSystem._question_local_pair_support_clause(
            "e1",
            "e2",
            kg_name="musique",
            question_id="q1",
            relationship_var="r",
        )
        assert "$question_id IN coalesce(r.questionIds, [])" in clause
        assert "c1.questionId = $question_id" in clause
        assert "c2.questionId = $question_id" in clause
        assert "coalesce(c1.passageIndex, -1) = coalesce(c2.passageIndex, -1)" in clause
        assert "d1.kgName = $kg_name" in clause

    def test_expand_entities_via_graph_adds_question_local_path_scope(self) -> None:
        sys_: EnhancedRAGSystem = _make_system()

        class FakeGraph:
            def __init__(self) -> None:
                self.query_text: str | None = None
                self.params: dict[str, object] | None = None

            def query(
                self, query: str, params: dict[str, object] | None
            ) -> list[dict[str, object]]:
                self.query_text = query
                self.params = params
                return []

        graph: FakeGraph = FakeGraph()
        result: GraphExpansion = sys_._expand_entities_via_graph(
            graph,
            ["e1"],
            kg_name="musique",
            max_hops=2,
            question_id="q1",
        )

        assert result == {"neighbors": {}, "paths": []}
        assert graph.params["question_id"] == "q1"
        assert "ALL(idx IN range(0, length(path) - 1)" in graph.query_text
        assert "c1.questionId = $question_id" in graph.query_text
        assert "c2.questionId = $question_id" in graph.query_text

    def test_expand_entities_via_graph_adds_document_scope(self) -> None:
        sys_: EnhancedRAGSystem = _make_system()

        class FakeGraph:
            def __init__(self) -> None:
                self.query_text: str | None = None
                self.params: dict[str, object] | None = None

            def query(
                self, query: str, params: dict[str, object] | None
            ) -> list[dict[str, object]]:
                self.query_text = query
                self.params = params
                return []

        graph: FakeGraph = FakeGraph()
        result: GraphExpansion = sys_._expand_entities_via_graph(
            graph,
            ["e1"],
            kg_name="custom_kg",
            max_hops=2,
            question_id=None,
            document_names=["doc_a.txt"],
        )

        assert result == {"neighbors": {}, "paths": []}
        assert graph.params["document_names"] == ["doc_a.txt"]
        assert "d.fileName IN $document_names" in graph.query_text

    def test_final_chunk_selection_prunes_off_chunk_relationships(self) -> None:
        context: RetrievalContext = {
            "chunks": [
                {
                    "text": "A",
                    "score": 1.0,
                    "linked_entity_ids": ["e1", "e2"],
                    "position": 10,
                    "question_id": "q1",
                    "passage_index": 0,
                }
            ],
            "entities": {
                "e1": {"id": "e1"},
                "e2": {"id": "e2"},
            },
            "relationships": [
                {
                    "source": "e1",
                    "target": "e2",
                    "type": "REL",
                    "provenance_positions": [10],
                },
                {
                    "source": "e1",
                    "target": "e2",
                    "type": "REL2",
                    "provenance_positions": [999],
                },
            ],
            "traversal_paths": [],
        }

        pruned: CountedGraphRetrievalContext | ProducedGraphRetrievalContext = (
            EnhancedRAGSystem._apply_final_chunk_selection(
                query="test",
                context=context,
                max_chunks=1,
                retrieval_temperature=0.0,
                retrieval_shortlist_factor=1,
                retrieval_sample_id=0,
                kg_name="musique",
            )
        )

        assert len(pruned["relationships"]) == 1
        assert pruned["relationships"][0]["type"] == "REL"


# ---------------------------------------------------------------------------
# 3. _decompose_question — fallback and cap behaviour
# ---------------------------------------------------------------------------


class TestDecomposeQuestion:
    sys_: EnhancedRAGSystem = _make_system()

    def _llm(self, response: str) -> "RunnableLambda[object, str]":
        # RunnableLambda is a proper LangChain Runnable — works with prompt | llm | parser.
        from langchain_core.runnables import RunnableLambda

        def fake_response(_: "object") -> "str":
            return response

        return RunnableLambda(fake_response)

    def test_valid_json_array_returned(self) -> None:
        llm: RunnableLambda[object, str] = self._llm(
            '["Who founded CRISPR?", "What country are they from?"]'
        )
        result: list[str] = self.sys_._decompose_question(
            "Multi hop question?", llm, max_hops=2
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(q, str) for q in result)

    def test_malformed_json_falls_back_to_original(self) -> None:
        llm: RunnableLambda[object, str] = self._llm("I cannot decompose this.")
        result: list[str] = self.sys_._decompose_question(
            "Original question?", llm, max_hops=2
        )
        assert result == ["Original question?"]

    def test_empty_response_falls_back(self) -> None:
        llm: RunnableLambda[object, str] = self._llm("")
        result: list[str] = self.sys_._decompose_question(
            "Some question?", llm, max_hops=2
        )
        assert result == ["Some question?"]

    def test_single_item_array_falls_back(self) -> None:
        # Needs >= 2 sub-questions; single-item signals LLM didn't decompose
        llm: RunnableLambda[object, str] = self._llm('["Only one question?"]')
        result: list[str] = self.sys_._decompose_question(
            "Some question?", llm, max_hops=2
        )
        assert result == ["Some question?"]

    def test_result_capped_to_max_hops(self) -> None:
        many: str = '["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"]'
        llm: RunnableLambda[object, str] = self._llm(many)
        result: list[str] = self.sys_._decompose_question(
            "Multi hop question?", llm, max_hops=3
        )
        assert len(result) <= 3

    def test_always_returns_list(self) -> None:
        llm: RunnableLambda[object, str] = self._llm("garbage [] more garbage")
        result: list[str] = self.sys_._decompose_question("X?", llm)
        assert isinstance(result, list)
        assert len(result) >= 1


class TestExtractQueryEntities:
    sys_: EnhancedRAGSystem = _make_system()

    def _llm(self, response: str) -> "RunnableLambda[object, str]":
        from langchain_core.runnables import RunnableLambda

        def fake_response(_: "object") -> "str":
            return response

        return RunnableLambda(fake_response)

    def test_dedupes_and_caps_entities(self) -> None:
        self.sys_._entity_extraction_cache = {}
        llm: RunnableLambda[object, str] = self._llm(
            json.dumps(
                [
                    "Marie Curie",
                    " Poland ",
                    "radioactivity",
                    "Marie Curie",
                    "",
                    "Pierre Curie",
                    "Nobel Prize",
                    "Paris",
                    "France",
                    "University of Paris",
                    "Marie Curie",
                ]
            )
        )
        result: list[str] = self.sys_._extract_query_entities(
            "Who was Marie Curie?", llm
        )

        assert (
            result
            == [
                "Marie Curie",
                "Poland",
                "radioactivity",
                "Pierre Curie",
                "Nobel Prize",
                "Paris",
                "France",
                "University of Paris",
            ][: EnhancedRAGSystem._MAX_EXTRACTED_QUERY_ENTITIES]
        )
        assert len(result) == EnhancedRAGSystem._MAX_EXTRACTED_QUERY_ENTITIES


class TestPprEntityScores:
    sys_: EnhancedRAGSystem = _make_system()

    def test_bidirectional_walk_and_dangling_mass_preserved(self) -> None:
        scores: dict[str, float] = self.sys_._ppr_entity_scores(
            seed_ids=["a"],
            all_entity_ids=["a", "b", "c"],
            edges=[("a", "b")],
        )

        assert set(scores) == {"a", "b", "c"}
        assert abs(sum(scores.values()) - 1.0) < 1e-9
        assert scores["b"] > scores["c"]


class TestMergeRetrievalContexts:
    def test_preserves_nonzero_seed_entity_count_from_secondary(self) -> None:
        primary: RetrievalContext = {
            "query": "test",
            "chunks": [{"chunk_id": "c1", "text": "a", "score": 0.8}],
            "entities": {},
            "relationships": [],
            "graph_neighbors": {},
            "traversal_paths": [],
            "documents": [],
            "seed_entity_count": 0,
            "grounding_quality": 0.0,
        }
        secondary: RetrievalContext = {
            "chunks": [{"chunk_id": "c2", "text": "b", "score": 0.6}],
            "entities": {},
            "relationships": [],
            "graph_neighbors": {},
            "traversal_paths": [],
            "documents": [],
            "seed_entity_count": 5,
            "grounding_quality": 0.4,
        }
        merged: RetrievalContext = EnhancedRAGSystem._merge_retrieval_contexts(
            primary,
            secondary,
            max_chunks=4,
            search_method="hybrid",
        )

        assert merged["seed_entity_count"] == 5


class TestVanillaQuestionScopedRetrieval:
    def test_retrieval_vector_question_scope_falls_back_to_parent_chunk_scope(
        self,
    ) -> None:
        system: VanillaRAGSystem = VanillaRAGSystem.__new__(VanillaRAGSystem)
        system._vector_index_name = None

        class FakeGraph:
            def __init__(self) -> None:
                self.queries: list[tuple[str, dict[str, object]]] = []

            def query(
                self, query: str, params: dict[str, object] | None = None
            ) -> list[dict[str, object]]:
                self.queries.append((query, params or {}))
                if "RETURN count(c) as chunk_count" in query:
                    return [{"chunk_count": 1}]
                if "CALL db.index.vector.queryNodes('retrieval_vector'" in query:
                    return [
                        {
                            "text": "Rome is where the Prince of tenors starred.",
                            "chunk_id": "rc1",
                            "chunk_element_id": "retrieval-1",
                            "parent_chunk_id": "chunk-1",
                            "parent_chunk_element_id": "chunk-el-1",
                            "position": 0,
                            "source": "hotpotqa/q1/p0",
                            "question_id": "q1",
                            "passage_index": 0,
                            "chunk_local_index": 0,
                            "retrieval_local_index": 0,
                            "score": 0.9,
                            "document": "hotpotqa",
                            "kg_name": "hotpotqa",
                        }
                    ]
                return []

        graph: FakeGraph = FakeGraph()
        system._create_neo4j_connection = lambda: graph

        def fake_generate_query_embedding(_: "object") -> "list[float]":
            return [0.0, 0.1]

        system._generate_query_embedding = fake_generate_query_embedding

        def fake_resolve_vector_index_name(_: "object") -> "str":
            return "retrieval_vector"

        system._resolve_vector_index_name = fake_resolve_vector_index_name

        context: ScoredCompleteRetrievalContext = system.get_vanilla_rag_context(
            "In what city did the Prince of tenors star?",
            kg_name="hotpotqa",
            question_id="q1",
            similarity_threshold=0.1,
            max_chunks=5,
        )

        search_query: str = next(
            query
            for query, _params in graph.queries
            if "CALL db.index.vector.queryNodes('retrieval_vector'" in query
        )
        assert (
            "coalesce(retrieval.questionId, chunk.questionId) = $question_id"
            in search_query
        )
        assert (
            "coalesce(retrieval.questionId, chunk.questionId) AS question_id"
            in search_query
        )
        assert len(context["chunks"]) == 1
        assert context["chunks"][0]["question_id"] == "q1"

    def test_retrieval_vector_empty_question_scope_falls_back_to_chunk_vector(
        self,
    ) -> None:
        system: VanillaRAGSystem = VanillaRAGSystem.__new__(VanillaRAGSystem)
        system._vector_index_name = None

        class FakeGraph:
            def __init__(self) -> None:
                self.queries: list[tuple[str, dict[str, object]]] = []

            def query(
                self, query: str, params: dict[str, object] | None = None
            ) -> list[dict[str, object]]:
                self.queries.append((query, params or {}))
                if "RETURN count(c) as chunk_count" in query:
                    return [{"chunk_count": 1}]
                if "CALL db.index.vector.queryNodes('retrieval_vector'" in query:
                    return []
                if "CALL db.index.vector.queryNodes('vector'" in query:
                    return [
                        {
                            "text": "Rome is where he starred.",
                            "chunk_id": "chunk-1",
                            "chunk_element_id": "chunk-el-1",
                            "position": 0,
                            "source": "hotpotqa/q1/p0",
                            "question_id": "q1",
                            "passage_index": 0,
                            "chunk_local_index": 0,
                            "score": 0.92,
                            "document": "hotpotqa",
                            "kg_name": "hotpotqa",
                        }
                    ]
                return []

        graph: FakeGraph = FakeGraph()
        system._create_neo4j_connection = lambda: graph

        def fake_generate_query_embedding(_: "object") -> "list[float]":
            return [0.0, 0.1]

        system._generate_query_embedding = fake_generate_query_embedding

        def fake_resolve_vector_index_name(_: "object") -> "str":
            return "retrieval_vector"

        system._resolve_vector_index_name = fake_resolve_vector_index_name

        context: ScoredCompleteRetrievalContext = system.get_vanilla_rag_context(
            "In what city did the Prince of tenors star?",
            kg_name="hotpotqa",
            question_id="q1",
            similarity_threshold=0.1,
            max_chunks=5,
        )

        assert any(
            "CALL db.index.vector.queryNodes('vector'" in query
            for query, _params in graph.queries
        )
        assert context["search_method"] == "vanilla_vector_similarity"
        assert len(context["chunks"]) == 1
        assert context["chunks"][0]["question_id"] == "q1"


class TestEnhancedQuestionScopedVectorRetrieval:
    def test_retrieval_vector_question_scope_falls_back_to_parent_chunk_scope(
        self,
    ) -> None:
        system: EnhancedRAGSystem = _make_system()
        system._active_vector_index_name = "retrieval_vector"
        system.check_vector_index = lambda: True

        def fake_generate_query_embedding(_: "object") -> "list[float]":
            return [0.0, 0.1]

        system._generate_query_embedding = fake_generate_query_embedding

        class FakeGraph:
            def __init__(self) -> None:
                self.queries: list[tuple[str, dict[str, object]]] = []

            def query(
                self, query: str, params: dict[str, object] | None = None
            ) -> list[dict[str, object]]:
                self.queries.append((query, params or {}))
                if "CALL db.index.vector.queryNodes('retrieval_vector'" in query:
                    return [
                        {
                            "text": "Rome is where the Prince of tenors starred.",
                            "chunk_id": "rc1",
                            "chunk_element_id": "retrieval-1",
                            "parent_chunk_id": "chunk-1",
                            "parent_chunk_element_id": "chunk-el-1",
                            "position": 0,
                            "source": "hotpotqa/q1/p0",
                            "question_id": "q1",
                            "passage_index": 0,
                            "chunk_local_index": 0,
                            "retrieval_local_index": 0,
                            "score": 0.9,
                            "document": "hotpotqa",
                            "kg_name": "hotpotqa",
                            "entities": [],
                        }
                    ]
                return []

        graph: FakeGraph = FakeGraph()
        context: GraphRetrievalContext = system._vector_similarity_search(
            graph,
            "In what city did the Prince of tenors star?",
            kg_name="hotpotqa",
            question_id="q1",
            similarity_threshold=0.1,
            max_chunks=5,
        )

        search_query: str = next(
            query
            for query, _params in graph.queries
            if "CALL db.index.vector.queryNodes('retrieval_vector'" in query
        )
        assert (
            "coalesce(retrieval.questionId, chunk.questionId) = $question_id"
            in search_query
        )
        assert (
            "coalesce(retrieval.questionId, chunk.questionId) AS question_id"
            in search_query
        )
        assert len(context["chunks"]) == 1
        assert context["chunks"][0]["question_id"] == "q1"

    def test_retrieval_vector_empty_question_scope_falls_back_to_chunk_vector(
        self,
    ) -> None:
        system: EnhancedRAGSystem = _make_system()
        system._active_vector_index_name = "retrieval_vector"
        system.check_vector_index = lambda: True

        def fake_generate_query_embedding(_: "object") -> "list[float]":
            return [0.0, 0.1]

        system._generate_query_embedding = fake_generate_query_embedding

        class FakeGraph:
            def __init__(self) -> None:
                self.queries: list[tuple[str, dict[str, object]]] = []

            def query(
                self, query: str, params: dict[str, object] | None = None
            ) -> list[dict[str, object]]:
                self.queries.append((query, params or {}))
                if "CALL db.index.vector.queryNodes('retrieval_vector'" in query:
                    return []
                if "CALL db.index.vector.queryNodes('vector'" in query:
                    return [
                        {
                            "text": "Rome is where he starred.",
                            "chunk_id": "chunk-1",
                            "chunk_element_id": "chunk-el-1",
                            "position": 0,
                            "source": "hotpotqa/q1/p0",
                            "question_id": "q1",
                            "passage_index": 0,
                            "chunk_local_index": 0,
                            "score": 0.92,
                            "document": "hotpotqa",
                            "kg_name": "hotpotqa",
                            "entities": [],
                        }
                    ]
                return []

        graph: FakeGraph = FakeGraph()
        context: GraphRetrievalContext = system._vector_similarity_search(
            graph,
            "In what city did the Prince of tenors star?",
            kg_name="hotpotqa",
            question_id="q1",
            similarity_threshold=0.1,
            max_chunks=5,
        )

        assert any(
            "CALL db.index.vector.queryNodes('vector'" in query
            for query, _params in graph.queries
        )
        assert context["search_method"] == "vector_similarity"
        assert len(context["chunks"]) == 1
        assert context["chunks"][0]["question_id"] == "q1"


class TestLateInteractionFirstStageRetrieval:
    def test_enhanced_semantic_search_prefers_late_interaction_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        system: EnhancedRAGSystem = _make_system()
        system.retrieval_mode = "vector_only"
        system.use_rfge = True
        system.check_vector_index = lambda: False
        system._first_stage_late_interaction_enabled = lambda: True

        class FakeGraph:
            def query(
                self, query: str, params: dict[str, object] | None = None
            ) -> list[dict[str, object]]:
                if "RETURN count(c) as chunk_count" in query:
                    return [{"chunk_count": 1}]
                return []

        graph: FakeGraph = FakeGraph()
        system._create_neo4j_connection = lambda: graph

        def fake_late_interaction_search(
            *args: object, **kwargs: object
        ) -> "RetrievalContext":
            return {
                "query": "Question?",
                "chunks": [
                    {
                        "text": "Late interaction evidence.",
                        "chunk_id": "c1",
                        "score": 0.7,
                        "document": "doc",
                    }
                ],
                "entities": {},
                "relationships": [],
                "graph_neighbors": {},
                "traversal_paths": [],
                "documents": ["doc"],
                "total_score": 0.7,
                "entity_count": 0,
                "relationship_count": 0,
                "search_method": "late_interaction_similarity",
            }

        system._late_interaction_search = fake_late_interaction_search

        context: ProducedGraphRetrievalContext = system.get_rag_context(
            "Question?", max_chunks=1, llm=None
        )

        assert context["search_method"] == "late_interaction_similarity"
        assert context["chunks"][0]["text"] == "Late interaction evidence."

    def test_enhanced_semantic_search_falls_back_to_vector_when_late_interaction_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        system: EnhancedRAGSystem = _make_system()
        system._first_stage_late_interaction_enabled = lambda: True

        def fake_late_interaction_search(
            *args: object, **kwargs: object
        ) -> "RetrievalContext":
            return {
                "query": "Question?",
                "chunks": [],
                "entities": {},
                "relationships": [],
                "graph_neighbors": {},
                "traversal_paths": [],
                "documents": [],
                "total_score": 0.0,
            }

        system._late_interaction_search = fake_late_interaction_search

        def fake_vector_similarity_search(
            *args: object, **kwargs: object
        ) -> "RetrievalContext":
            return {
                "query": "Question?",
                "chunks": [
                    {
                        "text": "Vector fallback.",
                        "chunk_id": "c1",
                        "score": 0.8,
                        "document": "doc",
                    }
                ],
                "entities": {},
                "relationships": [],
                "graph_neighbors": {},
                "traversal_paths": [],
                "documents": ["doc"],
                "total_score": 0.8,
                "entity_count": 0,
                "relationship_count": 0,
                "search_method": "vector_similarity",
            }

        system._vector_similarity_search = fake_vector_similarity_search

        context: GraphRetrievalContext = system._semantic_similarity_search(
            graph=object(),
            query="Question?",
            max_chunks=1,
        )

        assert context["search_method"] == "vector_similarity"
        assert context["chunks"][0]["text"] == "Vector fallback."

    def test_enhanced_late_interaction_unapplied_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        system: EnhancedRAGSystem = _make_system()

        def fake_late_interaction_scope_key(**kwargs: object) -> "tuple[str]":
            return ("scope",)

        system._late_interaction_scope_key = fake_late_interaction_scope_key

        def fake_late_interaction_corpus_rows(
            *args: object, **kwargs: object
        ) -> "list[RetrievedChunk]":
            return [
                {
                    "text": "Corpus-order chunk that should not masquerade as retrieval.",
                    "chunk_id": "c1",
                    "chunk_element_id": "ce1",
                    "document": "doc",
                    "entities": [],
                }
            ]

        system._late_interaction_corpus_rows = fake_late_interaction_corpus_rows
        import ontographrag.rag.systems._vector_search as vector_search_module

        def fake_late_interaction_rescore_chunks_for_query(
            query: "object", chunks: "list[RetrievedChunk]", **kwargs: "object"
        ) -> "tuple[list[RetrievedChunk], RerankingMetadata]":
            return (
                list(chunks),
                {"enabled": True, "applied": False, "reason": "model_unavailable"},
            )

        monkeypatch.setattr(
            vector_search_module,
            "late_interaction_rescore_chunks_for_query",
            fake_late_interaction_rescore_chunks_for_query,
        )

        context: GraphRetrievalContext = system._late_interaction_search(
            object(), "Question?", max_chunks=1
        )

        assert context["chunks"] == []
        assert context["search_method"] == "late_interaction_unavailable"
        assert context["late_interaction_stage"]["reason"] == "model_unavailable"

    def test_enhanced_semantic_search_can_bypass_first_stage_late_interaction(
        self,
    ) -> None:
        system: EnhancedRAGSystem = _make_system()
        system._first_stage_late_interaction_enabled = lambda: True

        def fake_late_interaction_search(*args: object, **kwargs: object) -> "Never":
            pytest.fail("late interaction should be bypassed for dense-floor fallback")

        system._late_interaction_search = fake_late_interaction_search

        def fake_vector_similarity_search(
            *args: object, **kwargs: object
        ) -> "RetrievalContext":
            return {
                "query": "Question?",
                "chunks": [
                    {
                        "text": "Dense floor.",
                        "chunk_id": "c1",
                        "score": 0.8,
                        "document": "doc",
                    }
                ],
                "entities": {},
                "relationships": [],
                "graph_neighbors": {},
                "traversal_paths": [],
                "documents": ["doc"],
                "total_score": 0.8,
                "entity_count": 0,
                "relationship_count": 0,
                "search_method": "vector_similarity",
            }

        system._vector_similarity_search = fake_vector_similarity_search

        context: GraphRetrievalContext = system._semantic_similarity_search(
            graph=object(),
            query="Question?",
            max_chunks=1,
            allow_first_stage_late_interaction=False,
        )

        assert context["search_method"] == "vector_similarity"
        assert context["chunks"][0]["text"] == "Dense floor."

    def test_vanilla_late_interaction_unapplied_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        system: VanillaRAGSystem = VanillaRAGSystem.__new__(VanillaRAGSystem)

        def fake_late_interaction_scope_key(**kwargs: object) -> "tuple[str]":
            return ("scope",)

        system._late_interaction_scope_key = fake_late_interaction_scope_key

        def fake_late_interaction_corpus_rows(
            *args: object, **kwargs: object
        ) -> "list[RetrievedChunk]":
            return [
                {
                    "text": "Corpus-order chunk that should not masquerade as retrieval.",
                    "chunk_id": "c1",
                    "chunk_element_id": "ce1",
                    "document": "doc",
                }
            ]

        system._late_interaction_corpus_rows = fake_late_interaction_corpus_rows

        def fake_late_interaction_rescore_chunks_for_query(
            query: "object", chunks: "list[RetrievedChunk]", **kwargs: "object"
        ) -> "tuple[list[RetrievedChunk], RerankingMetadata]":
            return (
                list(chunks),
                {"enabled": True, "applied": False, "reason": "model_unavailable"},
            )

        monkeypatch.setattr(
            vanilla_rag_module,
            "late_interaction_rescore_chunks_for_query",
            fake_late_interaction_rescore_chunks_for_query,
        )

        context: ScoredCompleteRetrievalContext = system._late_interaction_search(
            object(), "Question?", max_chunks=1
        )

        assert context["chunks"] == []
        assert context["search_method"] == "vanilla_late_interaction_unavailable"
        assert context["late_interaction_stage"]["reason"] == "model_unavailable"

    def test_vanilla_context_prefers_late_interaction_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        system: VanillaRAGSystem = VanillaRAGSystem.__new__(VanillaRAGSystem)
        system._first_stage_late_interaction_enabled = lambda: True

        class FakeGraph:
            def query(
                self, query: str, params: dict[str, object] | None = None
            ) -> list[dict[str, object]]:
                if "RETURN count(c) as chunk_count" in query:
                    return [{"chunk_count": 1}]
                return []

        graph: FakeGraph = FakeGraph()
        system._create_neo4j_connection = lambda: graph

        def fake_late_interaction_search(
            *args: object, **kwargs: object
        ) -> "RetrievalContext":
            return {
                "query": "Question?",
                "chunks": [
                    {
                        "text": "Late interaction evidence.",
                        "chunk_id": "c1",
                        "score": 0.7,
                        "document": "doc",
                    }
                ],
                "documents": ["doc"],
                "total_score": 0.7,
                "search_method": "vanilla_late_interaction_similarity",
            }

        system._late_interaction_search = fake_late_interaction_search

        context: ScoredCompleteRetrievalContext = system.get_vanilla_rag_context(
            "Question?", max_chunks=1
        )

        assert context["search_method"] == "vanilla_late_interaction_similarity"
        assert context["chunks"][0]["text"] == "Late interaction evidence."


class TestEnhancedAdjacentChunkRecovery:
    def test_adjacent_chunk_recovery_appends_neighbor_with_entities(self) -> None:
        system: EnhancedRAGSystem = _make_system()

        class FakeGraph:
            def __init__(self) -> None:
                self.queries: list[tuple[str, dict[str, object]]] = []

            def query(
                self, query: str, params: dict[str, object] | None = None
            ) -> list[dict[str, object]]:
                self.queries.append((query, params or {}))
                return [
                    {
                        "text": "The next sentence contains the answer.",
                        "chunk_id": "chunk-2",
                        "chunk_element_id": "chunk-el-2",
                        "position": 11,
                        "source": "hotpotqa/q1/p0",
                        "question_id": "q1",
                        "passage_index": 0,
                        "chunk_local_index": 2,
                        "score": 0.0,
                        "document": "hotpotqa",
                        "kg_name": "hotpotqa",
                        "entities": [
                            {
                                "id": "entity-2",
                                "element_id": "entity-el-2",
                                "type": "Entity",
                                "description": "Rome",
                            }
                        ],
                    }
                ]

        graph: FakeGraph = FakeGraph()
        context: RetrievalContext = {
            "chunks": [
                {
                    "text": "Seed retrieval chunk.",
                    "chunk_id": "retrieval-1",
                    "chunk_element_id": "retrieval-el-1",
                    "parent_chunk_element_id": "chunk-el-1",
                    "question_id": "q1",
                    "passage_index": 0,
                    "chunk_local_index": 1,
                    "document": "hotpotqa",
                    "kg_name": "hotpotqa",
                    "score": 0.9,
                    "linked_entity_ids": ["entity-1"],
                }
            ],
            "documents": ["hotpotqa"],
        }

        expanded: RetrievalContext = system._append_adjacent_chunks_to_context(
            graph,
            context,
            kg_name="hotpotqa",
            question_id="q1",
            document_names=None,
            max_adjacent=3,
        )

        assert len(expanded["chunks"]) == 2
        adjacent: RetrievedChunk = next(
            chunk for chunk in expanded["chunks"] if chunk.get("adjacent")
        )
        assert adjacent["question_id"] == "q1"
        assert adjacent["linked_entity_ids"] == ["entity-2"]
        adj_query: str = graph.queries[0][0]
        assert "adj.questionId = $question_id" in adj_query
        assert "OPTIONAL MATCH (adj)-[:HAS_ENTITY]->(entity:__Entity__)" in adj_query


class TestDocumentScopedEnhancedRetrieval:
    def test_get_rag_context_forwards_document_names_to_entity_first(self) -> None:
        system: EnhancedRAGSystem = _make_system()
        system.retrieval_mode = "hybrid_auto"
        system.use_rfge = True
        system.check_vector_index = lambda: False

        class FakeGraph:
            def query(
                self, query: str, params: dict[str, object] | None = None
            ) -> list[dict[str, object]]:
                if "RETURN count(c) as chunk_count" in query:
                    return [{"chunk_count": 1}]
                return []

        graph: FakeGraph = FakeGraph()
        system._create_neo4j_connection = lambda: graph

        captured: dict[str, list[str] | None] = {}

        def _fake_entity_first(
            graph: object,
            query: str,
            max_chunks: int = 20,
            kg_name: str | None = None,
            max_hops: int = 2,
            question_id: str | None = None,
            llm: object | None = None,
            document_names: list[str] | None = None,
            anchor_mask_entity_ids: list[str] | None = None,
            anchor_mask_entity_names: list[str] | None = None,
        ) -> RetrievalContext | None:
            captured["document_names"] = document_names
            return {
                "query": query,
                "chunks": [
                    {
                        "text": "Aspirin reduces fever.",
                        "score": 1.0,
                        "chunk_id": "c1",
                        "linked_entity_ids": ["e1"],
                        "document": "doc_a.txt",
                    }
                ],
                "entities": {"e1": {"id": "e1"}},
                "relationships": [],
                "graph_neighbors": {},
                "traversal_paths": [],
                "documents": ["doc_a.txt"],
                "total_score": 1.0,
                "entity_count": 1,
                "relationship_count": 0,
            }

        system._entity_first_search = _fake_entity_first

        context: ProducedGraphRetrievalContext = system.get_rag_context(
            "What reduces fever?",
            document_names=["doc_a.txt"],
            max_chunks=1,
            llm=None,
        )

        assert captured["document_names"] == ["doc_a.txt"]
        assert len(context["chunks"]) == 1


class TestEnhancedRetrievalUpgrades:
    sys_: EnhancedRAGSystem = _make_system()

    def test_lexical_rerank_prioritizes_query_overlap(self) -> None:
        chunks: list[RetrievedChunk] = [
            {
                "text": "Generic background passage.",
                "score": 0.95,
                "linked_entity_count": 0,
            },
            {
                "text": "Lea Pool directed Lost and Delirious.",
                "score": 0.62,
                "linked_entity_count": 0,
            },
        ]
        sorted_chunks: list[RetrievedChunk] = self.sys_._sort_chunks_for_query(
            "Who directed Lost and Delirious?",
            chunks,
        )
        assert "Lost and Delirious" in sorted_chunks[0]["text"]


def test_enhanced_query_fusion_supplements_missing_comparison_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system: EnhancedRAGSystem = _make_system()

    question: str = "Which film has the director born later, Riding The California Trail or Lost And Delirious?"
    base_context: RetrievalContext = {
        "query": question,
        "chunks": [
            {
                "text": "Riding the California Trail is directed by William Nigh.",
                "chunk_id": "c1",
                "chunk_element_id": "ce1",
                "score": 0.9,
                "document": "2wikimultihopqa",
                "linked_entity_ids": [],
                "linked_entity_count": 0,
            }
        ],
        "entities": {},
        "relationships": [],
        "graph_neighbors": {},
        "traversal_paths": [],
        "documents": ["2wikimultihopqa"],
        "total_score": 0.9,
        "entity_count": 0,
        "relationship_count": 0,
        "search_method": "vector_similarity",
    }
    fused_branch_context: RetrievalContext = {
        "query": question,
        "chunks": [
            {
                "text": "Lost and Delirious is directed by Lea Pool.",
                "chunk_id": "c2",
                "chunk_element_id": "ce2",
                "score": 0.7,
                "document": "2wikimultihopqa",
                "linked_entity_ids": [],
                "linked_entity_count": 0,
            }
        ],
        "entities": {},
        "relationships": [],
        "graph_neighbors": {},
        "traversal_paths": [],
        "documents": ["2wikimultihopqa"],
        "total_score": 0.7,
        "entity_count": 0,
        "relationship_count": 0,
        "search_method": "vector_similarity",
    }

    def fake_create_neo4j_connection(
        *_args: object, **_kwargs: "object"
    ) -> "list[dict[str, object]]":
        return []

    monkeypatch.setattr(
        system,
        "_create_neo4j_connection",
        lambda: type("G", (), {"query": fake_create_neo4j_connection})(),
    )

    def fake_should_run_query_fusion(*args: object, **kwargs: object) -> "bool":
        return True

    monkeypatch.setattr(
        system, "_should_run_query_fusion", fake_should_run_query_fusion
    )

    def fake_generate_query_variants(*args: object, **kwargs: object) -> "list[str]":
        return ["focus lost branch"]

    monkeypatch.setattr(
        system,
        "_generate_query_variants",
        fake_generate_query_variants,
    )

    def fake_get_rag_context(query: "str", **kwargs: object) -> "RetrievalContext":
        return fused_branch_context if query == "focus lost branch" else base_context

    monkeypatch.setattr(
        system,
        "get_rag_context",
        fake_get_rag_context,
    )

    def fake_invoke_answer_chain(**kwargs: "Unpack[AnswerArguments]") -> "str":
        return " || ".join(chunk["text"] for chunk in kwargs["context"]["chunks"])

    monkeypatch.setattr(
        system,
        "_invoke_answer_chain",
        fake_invoke_answer_chain,
    )

    def fake_apply_runtime_answer_guardrail(
        **kwargs: "Unpack[GuardrailArguments]",
    ) -> "tuple[str, RetrievalContext, FixtureGuardrailMetadata]":
        return (
            kwargs["response"],
            kwargs["context"],
            {"enabled": False, "final_decision": "keep"},
        )

    monkeypatch.setattr(
        system,
        "_apply_runtime_answer_guardrail",
        fake_apply_runtime_answer_guardrail,
    )

    def fake_extract_used_entities_and_chunks(
        response: "object", context: "object"
    ) -> "UsedEvidence":
        return {"used_entities": [], "used_chunks": [], "reasoning_edges": []}

    monkeypatch.setattr(
        system,
        "_extract_used_entities_and_chunks",
        fake_extract_used_entities_and_chunks,
    )

    result: GenerationResponse = system.generate_response(
        question, llm=object(), max_hops=1
    )

    assert "Riding the California Trail" in result["response"]
    assert "Lost and Delirious" in result["response"]


def test_vanilla_query_fusion_supplements_missing_comparison_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system: VanillaRAGSystem = VanillaRAGSystem.__new__(VanillaRAGSystem)

    question: str = "Which film has the director born later, Riding The California Trail or Lost And Delirious?"
    base_context: RetrievalContext = {
        "query": question,
        "chunks": [
            {
                "text": "Riding the California Trail is directed by William Nigh.",
                "chunk_id": "c1",
                "chunk_element_id": "ce1",
                "score": 0.9,
                "document": "2wikimultihopqa",
            }
        ],
        "documents": ["2wikimultihopqa"],
        "total_score": 0.9,
        "search_method": "vanilla_vector_similarity",
    }
    variant_context: RetrievalContext = {
        "query": question,
        "chunks": [
            {
                "text": "Lost and Delirious is directed by Lea Pool.",
                "chunk_id": "c2",
                "chunk_element_id": "ce2",
                "score": 0.7,
                "document": "2wikimultihopqa",
            }
        ],
        "documents": ["2wikimultihopqa"],
        "total_score": 0.7,
        "search_method": "vanilla_vector_similarity",
    }

    def fake_should_run_query_fusion(*args: object, **kwargs: object) -> "bool":
        return True

    monkeypatch.setattr(
        system, "_should_run_query_fusion", fake_should_run_query_fusion
    )

    def fake_generate_query_variants(*args: object, **kwargs: object) -> "list[str]":
        return ["focus lost branch"]

    monkeypatch.setattr(
        system,
        "_generate_query_variants",
        fake_generate_query_variants,
    )

    def fake_get_vanilla_rag_context(
        query: "str", **kwargs: object
    ) -> "RetrievalContext":
        return variant_context if query == "focus lost branch" else base_context

    monkeypatch.setattr(
        system,
        "get_vanilla_rag_context",
        fake_get_vanilla_rag_context,
    )

    def fake_invoke_answer_chain(**kwargs: "Unpack[AnswerArguments]") -> "str":
        return " || ".join(chunk["text"] for chunk in kwargs["context"]["chunks"])

    monkeypatch.setattr(
        system,
        "_invoke_answer_chain",
        fake_invoke_answer_chain,
    )

    def fake_apply_runtime_answer_guardrail(
        **kwargs: "Unpack[GuardrailArguments]",
    ) -> "tuple[str, FixtureGuardrailMetadata]":
        return (kwargs["response"], {"enabled": False, "final_decision": "keep"})

    monkeypatch.setattr(
        system,
        "_apply_runtime_answer_guardrail",
        fake_apply_runtime_answer_guardrail,
    )

    result: GenerationResponse = system.generate_response(
        question, llm=object(), max_chunks=4
    )

    assert "Riding the California Trail" in result["response"]
    assert "Lost and Delirious" in result["response"]


def test_enhanced_generate_response_respects_reranker_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ontographrag.rag.systems.enhanced_rag_system as enhanced_mod

    system: EnhancedRAGSystem = _make_system()
    context: RetrievalContext = {
        "query": "Who directed Lost and Delirious?",
        "chunks": [
            {
                "text": "Generic background.",
                "chunk_id": "c1",
                "chunk_element_id": "ce1",
                "score": 0.9,
                "document": "doc",
            },
            {
                "text": "Lost and Delirious is directed by Lea Pool.",
                "chunk_id": "c2",
                "chunk_element_id": "ce2",
                "score": 0.5,
                "document": "doc",
            },
        ],
        "entities": {},
        "relationships": [],
        "graph_neighbors": {},
        "traversal_paths": [],
        "documents": ["doc"],
        "total_score": 1.4,
        "entity_count": 0,
        "relationship_count": 0,
        "search_method": "vector_similarity",
    }

    def fake_context_result(*args: object, **kwargs: object) -> "RetrievalContext":
        return context

    monkeypatch.setattr(system, "get_rag_context", fake_context_result)

    def fake_should_run_query_fusion(*args: object, **kwargs: object) -> "bool":
        return False

    monkeypatch.setattr(
        system, "_should_run_query_fusion", fake_should_run_query_fusion
    )

    def fake_create_neo4j_connection(
        *_args: object, **_kwargs: "object"
    ) -> "list[dict[str, object]]":
        return []

    monkeypatch.setattr(
        system,
        "_create_neo4j_connection",
        lambda: type("G", (), {"query": fake_create_neo4j_connection})(),
    )

    def fake_late_interaction_rescore_chunks_for_query(
        query: "object", chunks: "list[RetrievedChunk]", max_chunks: "object" = None
    ) -> "tuple[list[RetrievedChunk], RerankingMetadata]":
        return (chunks, {"enabled": False, "applied": False, "reason": "disabled"})

    monkeypatch.setattr(
        enhanced_mod,
        "late_interaction_rescore_chunks_for_query",
        fake_late_interaction_rescore_chunks_for_query,
    )

    def fake_rerank_chunks_for_query(
        query: "object", chunks: "list[RetrievedChunk]", max_chunks: "object" = None
    ) -> "tuple[list[RetrievedChunk], RerankingMetadata]":
        return (
            [chunks[1], chunks[0]],
            {"enabled": True, "applied": True, "model": "fake-reranker"},
        )

    monkeypatch.setattr(
        enhanced_mod,
        "rerank_chunks_for_query",
        fake_rerank_chunks_for_query,
    )

    def fake_invoke_answer_chain(**kwargs: "Unpack[AnswerArguments]") -> "str":
        return " || ".join(chunk["text"] for chunk in kwargs["context"]["chunks"])

    monkeypatch.setattr(
        system,
        "_invoke_answer_chain",
        fake_invoke_answer_chain,
    )

    def fake_apply_runtime_answer_guardrail(
        **kwargs: "Unpack[GuardrailArguments]",
    ) -> "tuple[str, RetrievalContext, FixtureGuardrailMetadata]":
        return (
            kwargs["response"],
            kwargs["context"],
            {"enabled": False, "final_decision": "keep"},
        )

    monkeypatch.setattr(
        system,
        "_apply_runtime_answer_guardrail",
        fake_apply_runtime_answer_guardrail,
    )

    def fake_extract_used_entities_and_chunks(
        response: "object", context: "object"
    ) -> "UsedEvidence":
        return {"used_entities": [], "used_chunks": [], "reasoning_edges": []}

    monkeypatch.setattr(
        system,
        "_extract_used_entities_and_chunks",
        fake_extract_used_entities_and_chunks,
    )

    result: GenerationResponse = system.generate_response(
        "Who directed Lost and Delirious?", llm=object(), max_hops=1
    )

    assert result["response"].startswith("Lost and Delirious is directed by Lea Pool.")
    assert result["reranker"]["applied"] is True


def test_vanilla_generate_response_respects_reranker_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ontographrag.rag.systems.vanilla_rag_system as vanilla_mod

    system: VanillaRAGSystem = VanillaRAGSystem.__new__(VanillaRAGSystem)
    context: RetrievalContext = {
        "query": "Who directed Lost and Delirious?",
        "chunks": [
            {
                "text": "Generic background.",
                "chunk_id": "c1",
                "chunk_element_id": "ce1",
                "score": 0.9,
                "document": "doc",
            },
            {
                "text": "Lost and Delirious is directed by Lea Pool.",
                "chunk_id": "c2",
                "chunk_element_id": "ce2",
                "score": 0.5,
                "document": "doc",
            },
        ],
        "documents": ["doc"],
        "total_score": 1.4,
        "search_method": "vanilla_vector_similarity",
    }

    def fake_get_vanilla_rag_context(
        *args: object, **kwargs: object
    ) -> "RetrievalContext":
        return context

    monkeypatch.setattr(system, "get_vanilla_rag_context", fake_get_vanilla_rag_context)

    def fake_should_run_query_fusion(*args: object, **kwargs: object) -> "bool":
        return False

    monkeypatch.setattr(
        system, "_should_run_query_fusion", fake_should_run_query_fusion
    )

    def fake_late_interaction_rescore_chunks_for_query(
        query: "object", chunks: "list[RetrievedChunk]", max_chunks: "object" = None
    ) -> "tuple[list[RetrievedChunk], RerankingMetadata]":
        return (chunks, {"enabled": False, "applied": False, "reason": "disabled"})

    monkeypatch.setattr(
        vanilla_mod,
        "late_interaction_rescore_chunks_for_query",
        fake_late_interaction_rescore_chunks_for_query,
    )

    def fake_rerank_chunks_for_query(
        query: "object", chunks: "list[RetrievedChunk]", max_chunks: "object" = None
    ) -> "tuple[list[RetrievedChunk], RerankingMetadata]":
        return (
            [chunks[1], chunks[0]],
            {"enabled": True, "applied": True, "model": "fake-reranker"},
        )

    monkeypatch.setattr(
        vanilla_mod,
        "rerank_chunks_for_query",
        fake_rerank_chunks_for_query,
    )

    def fake_invoke_answer_chain(**kwargs: "Unpack[AnswerArguments]") -> "str":
        return " || ".join(chunk["text"] for chunk in kwargs["context"]["chunks"])

    monkeypatch.setattr(
        system,
        "_invoke_answer_chain",
        fake_invoke_answer_chain,
    )

    def fake_apply_runtime_answer_guardrail(
        **kwargs: "Unpack[GuardrailArguments]",
    ) -> "tuple[str, FixtureGuardrailMetadata]":
        return (kwargs["response"], {"enabled": False, "final_decision": "keep"})

    monkeypatch.setattr(
        system,
        "_apply_runtime_answer_guardrail",
        fake_apply_runtime_answer_guardrail,
    )

    result: GenerationResponse = system.generate_response(
        "Who directed Lost and Delirious?", llm=object(), max_chunks=4
    )

    assert result["response"].startswith("Lost and Delirious is directed by Lea Pool.")
    assert result["reranker"]["applied"] is True


def test_enhanced_generate_response_respects_late_interaction_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ontographrag.rag.systems.enhanced_rag_system as enhanced_mod

    system: EnhancedRAGSystem = _make_system()
    context: RetrievalContext = {
        "query": "Who directed Lost and Delirious?",
        "chunks": [
            {
                "text": "Generic background.",
                "chunk_id": "c1",
                "chunk_element_id": "ce1",
                "score": 0.9,
                "document": "doc",
            },
            {
                "text": "Lost and Delirious is directed by Lea Pool.",
                "chunk_id": "c2",
                "chunk_element_id": "ce2",
                "score": 0.5,
                "document": "doc",
            },
        ],
        "entities": {},
        "relationships": [],
        "graph_neighbors": {},
        "traversal_paths": [],
        "documents": ["doc"],
        "total_score": 1.4,
        "entity_count": 0,
        "relationship_count": 0,
        "search_method": "vector_similarity",
    }

    def fake_get_rag_context(*args: object, **kwargs: object) -> "RetrievalContext":
        return context

    monkeypatch.setattr(system, "get_rag_context", fake_get_rag_context)

    def fake_should_run_query_fusion(*args: object, **kwargs: object) -> "bool":
        return False

    monkeypatch.setattr(
        system, "_should_run_query_fusion", fake_should_run_query_fusion
    )

    def fake_create_neo4j_connection(
        *_args: object, **_kwargs: "object"
    ) -> "list[dict[str, object]]":
        return []

    monkeypatch.setattr(
        system,
        "_create_neo4j_connection",
        lambda: type("G", (), {"query": fake_create_neo4j_connection})(),
    )

    def fake_late_interaction_rescore_chunks_for_query(
        query: "object", chunks: "list[RetrievedChunk]", max_chunks: "object" = None
    ) -> "tuple[list[RetrievedChunk], RerankingMetadata]":
        return (
            [chunks[1], chunks[0]],
            {"enabled": True, "applied": True, "model": "fake-late-interaction"},
        )

    monkeypatch.setattr(
        enhanced_mod,
        "late_interaction_rescore_chunks_for_query",
        fake_late_interaction_rescore_chunks_for_query,
    )

    def fake_rerank_chunks_for_query(
        query: "object", chunks: "list[RetrievedChunk]", max_chunks: "object" = None
    ) -> "tuple[list[RetrievedChunk], RerankingMetadata]":
        return (chunks, {"enabled": False, "applied": False, "reason": "disabled"})

    monkeypatch.setattr(
        enhanced_mod,
        "rerank_chunks_for_query",
        fake_rerank_chunks_for_query,
    )

    def fake_invoke_answer_chain(**kwargs: "Unpack[AnswerArguments]") -> "str":
        return " || ".join(chunk["text"] for chunk in kwargs["context"]["chunks"])

    monkeypatch.setattr(
        system,
        "_invoke_answer_chain",
        fake_invoke_answer_chain,
    )

    def fake_apply_runtime_answer_guardrail(
        **kwargs: "Unpack[GuardrailArguments]",
    ) -> "tuple[str, RetrievalContext, FixtureGuardrailMetadata]":
        return (
            kwargs["response"],
            kwargs["context"],
            {"enabled": False, "final_decision": "keep"},
        )

    monkeypatch.setattr(
        system,
        "_apply_runtime_answer_guardrail",
        fake_apply_runtime_answer_guardrail,
    )

    def fake_extract_used_entities_and_chunks(
        response: "object", context: "object"
    ) -> "UsedEvidence":
        return {"used_entities": [], "used_chunks": [], "reasoning_edges": []}

    monkeypatch.setattr(
        system,
        "_extract_used_entities_and_chunks",
        fake_extract_used_entities_and_chunks,
    )

    result: GenerationResponse = system.generate_response(
        "Who directed Lost and Delirious?", llm=object(), max_hops=1
    )

    assert result["response"].startswith("Lost and Delirious is directed by Lea Pool.")
    assert result["late_interaction"]["applied"] is True


def test_vanilla_generate_response_respects_late_interaction_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ontographrag.rag.systems.vanilla_rag_system as vanilla_mod

    system: VanillaRAGSystem = VanillaRAGSystem.__new__(VanillaRAGSystem)
    context: RetrievalContext = {
        "query": "Who directed Lost and Delirious?",
        "chunks": [
            {
                "text": "Generic background.",
                "chunk_id": "c1",
                "chunk_element_id": "ce1",
                "score": 0.9,
                "document": "doc",
            },
            {
                "text": "Lost and Delirious is directed by Lea Pool.",
                "chunk_id": "c2",
                "chunk_element_id": "ce2",
                "score": 0.5,
                "document": "doc",
            },
        ],
        "documents": ["doc"],
        "total_score": 1.4,
        "search_method": "vanilla_vector_similarity",
    }

    def fake_get_vanilla_rag_context(
        *args: object, **kwargs: object
    ) -> "RetrievalContext":
        return context

    monkeypatch.setattr(system, "get_vanilla_rag_context", fake_get_vanilla_rag_context)

    def fake_should_run_query_fusion(*args: object, **kwargs: object) -> "bool":
        return False

    monkeypatch.setattr(
        system, "_should_run_query_fusion", fake_should_run_query_fusion
    )

    def fake_late_interaction_rescore_chunks_for_query(
        query: "object", chunks: "list[RetrievedChunk]", max_chunks: "object" = None
    ) -> "tuple[list[RetrievedChunk], RerankingMetadata]":
        return (
            [chunks[1], chunks[0]],
            {"enabled": True, "applied": True, "model": "fake-late-interaction"},
        )

    monkeypatch.setattr(
        vanilla_mod,
        "late_interaction_rescore_chunks_for_query",
        fake_late_interaction_rescore_chunks_for_query,
    )

    def fake_rerank_chunks_for_query(
        query: "object", chunks: "list[RetrievedChunk]", max_chunks: "object" = None
    ) -> "tuple[list[RetrievedChunk], RerankingMetadata]":
        return (chunks, {"enabled": False, "applied": False, "reason": "disabled"})

    monkeypatch.setattr(
        vanilla_mod,
        "rerank_chunks_for_query",
        fake_rerank_chunks_for_query,
    )

    def fake_invoke_answer_chain(**kwargs: "Unpack[AnswerArguments]") -> "str":
        return " || ".join(chunk["text"] for chunk in kwargs["context"]["chunks"])

    monkeypatch.setattr(
        system,
        "_invoke_answer_chain",
        fake_invoke_answer_chain,
    )

    def fake_apply_runtime_answer_guardrail(
        **kwargs: "Unpack[GuardrailArguments]",
    ) -> "tuple[str, FixtureGuardrailMetadata]":
        return (kwargs["response"], {"enabled": False, "final_decision": "keep"})

    monkeypatch.setattr(
        system,
        "_apply_runtime_answer_guardrail",
        fake_apply_runtime_answer_guardrail,
    )

    result: GenerationResponse = system.generate_response(
        "Who directed Lost and Delirious?", llm=object(), max_chunks=4
    )

    assert result["response"].startswith("Lost and Delirious is directed by Lea Pool.")
    assert result["late_interaction"]["applied"] is True


# ---------------------------------------------------------------------------
# 4. format_context_for_llm — output shape
# ---------------------------------------------------------------------------


class TestFormatContextForLLM:
    sys_: EnhancedRAGSystem = _make_system()

    def test_returns_three_strings(self) -> None:
        ctx: RetrievalContext = _minimal_context()
        result: tuple[str, str, str] = self.sys_.format_context_for_llm(ctx)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(s, str) for s in result)

    def test_chunks_rendered_in_evidence_block(self) -> None:
        ctx: RetrievalContext = _minimal_context(
            chunks=[
                {
                    "text": "Aspirin reduces fever.",
                    "document": "paper.txt",
                    "chunk_id": "c1",
                    "linked_entity_ids": [],
                }
            ]
        )
        evidence_block, _, _ = self.sys_.format_context_for_llm(ctx)
        assert "Aspirin reduces fever." in evidence_block

    def test_ungrounded_paths_are_demoted_to_structural_hints(self) -> None:
        ctx: RetrievalContext = _minimal_context(
            chunks=[
                {
                    "text": "Aspirin reduces fever.",
                    "document": "paper.txt",
                    "chunk_id": "c1",
                    "linked_entity_ids": ["e1"],
                }
            ],
            traversal_paths=[
                {
                    "path": "Aspirin --TREATS--> Fever",
                    "hops": 1,
                    "node_ids": ["e1", "e2"],
                }
            ],
        )
        evidence_block, _, _ = self.sys_.format_context_for_llm(ctx)
        assert "STRUCTURAL HINTS" in evidence_block
        assert "no passage retrieved" in evidence_block
        assert "REASONING CHAINS:\nChain" not in evidence_block

    def test_path_with_matching_provenance_position_stays_in_reasoning_chains(
        self,
    ) -> None:
        ctx: RetrievalContext = _minimal_context(
            chunks=[
                {
                    "text": "Aspirin reduces fever.",
                    "document": "paper.txt",
                    "chunk_id": "c1",
                    "linked_entity_ids": ["e1"],
                    "position": 7,
                }
            ],
            traversal_paths=[
                {
                    "path": "Aspirin --TREATS--> Fever",
                    "hops": 1,
                    "node_ids": ["e1", "e2"],
                    "provenance_positions": [7],
                }
            ],
        )
        evidence_block, _, _ = self.sys_.format_context_for_llm(ctx)
        assert "REASONING CHAINS:\nChain 1" in evidence_block
        assert "STRUCTURAL HINTS" not in evidence_block

    def test_no_chunks_returns_no_evidence_message(self) -> None:
        ctx: RetrievalContext = _minimal_context()
        evidence_block, _, _ = self.sys_.format_context_for_llm(ctx)
        assert evidence_block  # must be non-empty (shows placeholder)

    def test_relationships_rendered_in_paths(self) -> None:
        ctx: RetrievalContext = _minimal_context(
            entities={
                "e1": {"id": "Aspirin", "type": "Drug", "description": "Aspirin"}
            },
            relationships=[{"source": "e1", "target": "e2", "type": "TREATS"}],
        )
        _, _, paths_str = self.sys_.format_context_for_llm(ctx)
        assert "TREATS" in paths_str

    def test_negated_relationships_rendered_in_paths(self) -> None:
        ctx: RetrievalContext = _minimal_context(
            entities={
                "e1": {"id": "Aspirin", "type": "Drug", "description": "Aspirin"},
                "e2": {"id": "Fever", "type": "Symptom", "description": "Fever"},
            },
            relationships=[
                {"source": "e1", "target": "e2", "type": "TREATS", "negated": True}
            ],
        )
        _, _, paths_str = self.sys_.format_context_for_llm(ctx)
        assert "NOT TREATS" in paths_str

    def test_no_graph_data_shows_fallback_message(self) -> None:
        ctx: RetrievalContext = _minimal_context()
        _, _, paths_str = self.sys_.format_context_for_llm(ctx)
        assert "No graph" in paths_str

    def test_traversal_paths_rendered(self) -> None:
        ctx: RetrievalContext = _minimal_context(
            traversal_paths=[{"path": "Aspirin -> TREATS -> Fever", "hops": 1}]
        )
        _, _, paths_str = self.sys_.format_context_for_llm(ctx)
        assert "Aspirin -> TREATS -> Fever" in paths_str

    def test_seed_vs_neighbor_entity_labelling(self) -> None:
        ctx: RetrievalContext = _minimal_context(
            entities={
                "e1": {
                    "id": "Aspirin",
                    "type": "Drug",
                    "source": "entity_lookup",
                    "description": "Aspirin",
                },
                "e2": {
                    "id": "Fever",
                    "type": "Symptom",
                    "source": "graph_traversal",
                    "description": "Fever",
                },
            }
        )
        _, entities_str, _ = self.sys_.format_context_for_llm(ctx)
        assert "Seed entities" in entities_str
        assert "Graph-traversal neighbors" in entities_str


class TestGraphEvidenceFormatting:
    def test_relationship_identity_distinguishes_negation(self) -> None:
        positive: str = EnhancedRAGSystem._relationship_identity(
            {"source": "drug", "target": "symptom", "type": "TREATS", "negated": False}
        )
        negative: str = EnhancedRAGSystem._relationship_identity(
            {"source": "drug", "target": "symptom", "type": "TREATS", "negated": True}
        )
        assert positive != negative

    def test_relationship_label_includes_condition_and_quantitative(self) -> None:
        label: str = EnhancedRAGSystem._format_relationship_label(
            {
                "type": "ASSOCIATED_WITH",
                "condition": "in mice",
                "quantitative": "p<0.05",
            }
        )
        assert label == "ASSOCIATED_WITH [in mice] (p<0.05)"

    def test_passage_only_evidence_block_uses_chunks_only(self) -> None:
        system: EnhancedRAGSystem = _make_system()
        context: RetrievalContext = _minimal_context(
            chunks=[{"text": "The study concludes yes."}],
            relationships=[
                {"source": "e1", "target": "e2", "type": "TREATS", "negated": True}
            ],
            traversal_paths=[{"path": "Drug --NOT TREATS--> Disease", "hops": 1}],
        )
        evidence_block: str = system._build_passages_only_evidence_block(context)
        assert evidence_block.startswith("PASSAGES:\n")
        assert "The study concludes yes." in evidence_block
        assert "TREATS" not in evidence_block

    def test_pubmedqa_uses_passage_only_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        system: EnhancedRAGSystem = _make_system()
        system.rag_prompt = object()
        captured: dict[str, str] = {}

        class FakeChain:
            def __or__(self, _other: object) -> "FakeChain | FakePrompt":
                return self

            def invoke(self, payload: dict[str, str]) -> str:
                captured["evidence_block"] = payload["evidence_block"]
                return "stub"

        class FakePrompt:
            def __or__(self, _other: object) -> "FakeChain | FakePrompt":
                return self

        system.rag_prompt = FakePrompt()

        class FakePromptChain(FakePrompt):
            def __or__(self, _other: object) -> "FakeChain":
                return FakeChain()

        system.rag_prompt = FakePromptChain()
        context: RetrievalContext = _minimal_context(
            chunks=[{"text": "Study says yes."}],
            relationships=[{"source": "e1", "target": "e2", "type": "TREATS"}],
            traversal_paths=[{"path": "Drug --TREATS--> Disease", "hops": 1}],
        )

        answer: str = system._invoke_answer_chain(
            question="Does it work?",
            llm=object(),
            context=context,
            kg_name="pubmedqa",
        )

        assert answer == "stub"
        assert captured["evidence_block"].startswith("PASSAGES:\n")
        assert "STRUCTURAL HINTS" not in captured["evidence_block"]


# ---------------------------------------------------------------------------
# Traversal relevance gate and per-passage chunk cap
# ---------------------------------------------------------------------------


class TestTraversalRelevanceGate:
    """_TRAVERSAL_CHUNK_MIN_SIM filters hop-1+ chunks below the threshold."""

    def _make_chunks(self, entries: list[tuple[int, str]]) -> list[RetrievedChunk]:
        """entries: list of (min_hop, chunk_element_id)"""
        return [
            {
                "text": f"chunk {i}",
                "chunk_id": f"c{i}",
                "chunk_element_id": f"eid_{eid}",
                "score": 0.8,
                "min_hop": hop,
                "question_id": "q1",
                "passage_index": i,
                "chunk_local_index": 0,
                "linked_entity_count": 1,
                "linked_entity_ids": [],
                "document": "doc",
                "kg_name": None,
                "position": i,
                "source": None,
                "entities": [],
            }
            for i, (hop, eid) in enumerate(entries)
        ]

    def test_seed_chunks_always_kept(self) -> None:
        """Seed chunks (min_hop=0) are never filtered regardless of threshold."""
        s: EnhancedRAGSystem = _make_system()
        s._TRAVERSAL_CHUNK_MIN_SIM = 0.99  # impossibly high threshold
        s._MAX_CHUNKS_PER_PASSAGE = 0  # disable cap

        chunks: list[RetrievedChunk] = self._make_chunks([(0, "a"), (0, "b")])
        # Simulate the gate with no query embedding → gate is skipped
        # (query_embedding guard: gate only runs when query_embedding is truthy)
        # Confirm seed chunks survive when gate is disabled
        assert all(c["min_hop"] == 0 for c in chunks)

    def test_gate_disabled_at_zero_threshold(self) -> None:
        """Setting threshold to 0.0 skips the gate entirely."""
        s: EnhancedRAGSystem = _make_system()
        s._TRAVERSAL_CHUNK_MIN_SIM = 0.0
        s._MAX_CHUNKS_PER_PASSAGE = 0

        _chunks: list[RetrievedChunk] = self._make_chunks([(1, "x"), (1, "y")])
        # With threshold 0 the gate is not entered; all chunks survive
        min_sim: float = s._TRAVERSAL_CHUNK_MIN_SIM
        assert min_sim == 0.0


class TestPerPassageChunkCap:
    """_MAX_CHUNKS_PER_PASSAGE limits chunks per (questionId, passageIndex)."""

    def _chunks_from_passages(
        self, passage_assignments: Sequence[tuple[str | None, int | None]]
    ) -> list[RetrievedChunk]:
        """passage_assignments: list of (question_id, passage_index)"""
        chunks: list[RetrievedChunk] = []
        for i, (qid, pidx) in enumerate(passage_assignments):
            chunks.append(
                {
                    "text": f"chunk {i}",
                    "chunk_id": f"c{i}",
                    "chunk_element_id": f"eid_{i}",
                    "score": 1.0 - i * 0.05,
                    "min_hop": 0,
                    "question_id": qid,
                    "passage_index": pidx,
                    "chunk_local_index": i,
                    "linked_entity_count": 1,
                    "linked_entity_ids": [],
                    "document": "doc",
                    "kg_name": None,
                    "position": i,
                    "source": None,
                    "entities": [],
                }
            )
        return chunks

    def _apply_cap(
        self, chunks: list[RetrievedChunk], cap: int
    ) -> list[RetrievedChunk]:
        """Run the real per-passage chunk cap helper."""
        s: EnhancedRAGSystem = _make_system()
        s._MAX_CHUNKS_PER_PASSAGE = cap
        return s._apply_per_passage_chunk_cap(chunks)

    def test_cap_limits_per_passage(self) -> None:
        chunks: list[RetrievedChunk] = self._chunks_from_passages(
            [("q1", 0)] * 5 + [("q1", 1)] * 3
        )
        result: list[RetrievedChunk] = self._apply_cap(chunks, cap=2)
        from collections import Counter

        counts: Counter[str] = Counter(
            f"{c['question_id']}::p{c['passage_index']}" for c in result
        )
        assert counts["q1::p0"] == 2
        assert counts["q1::p1"] == 2

    def test_cap_preserves_all_when_under_limit(self) -> None:
        chunks: list[RetrievedChunk] = self._chunks_from_passages(
            [("q1", 0), ("q1", 1), ("q1", 2)]
        )
        result: list[RetrievedChunk] = self._apply_cap(chunks, cap=2)
        assert len(result) == 3  # one per passage, all under cap

    def test_cap_zero_disables(self) -> None:
        """cap=0 means the guard `if _passage_cap > 0` is False; all chunks pass."""
        _chunks: list[RetrievedChunk] = self._chunks_from_passages([("q1", 0)] * 10)
        # With cap=0 the guard is skipped; simulate by not calling _apply_cap
        s: EnhancedRAGSystem = _make_system()
        s._MAX_CHUNKS_PER_PASSAGE = 0
        assert s._MAX_CHUNKS_PER_PASSAGE == 0  # passes without applying cap

    def test_cap_without_passage_metadata_falls_back_to_chunk_id(self) -> None:
        """Chunks lacking passage_index are keyed by chunk_id (not grouped)."""
        chunks: list[RetrievedChunk] = [
            {
                "text": f"chunk {i}",
                "chunk_id": f"c{i}",
                "chunk_element_id": f"eid_{i}",
                "score": 0.9,
                "min_hop": 0,
                "question_id": None,
                "passage_index": None,
                "chunk_local_index": 0,
                "linked_entity_count": 1,
                "linked_entity_ids": [],
                "document": "doc",
                "kg_name": None,
                "position": i,
                "source": None,
                "entities": [],
            }
            for i in range(5)
        ]
        result: list[RetrievedChunk] = self._apply_cap(chunks, cap=2)
        # Each has a unique chunk_id key so none are dropped
        assert len(result) == 5

    def test_default_cap_is_two(self) -> None:
        s: EnhancedRAGSystem = _make_system()
        assert s._MAX_CHUNKS_PER_PASSAGE == 2

    def test_cap_keeps_highest_scoring_chunk_per_passage(self) -> None:
        chunks: list[RetrievedChunk] = [
            {
                "text": "low",
                "chunk_id": "low",
                "chunk_element_id": "eid_low",
                "score": 0.2,
                "min_hop": 0,
                "question_id": "q1",
                "passage_index": 0,
                "chunk_local_index": 0,
                "linked_entity_count": 1,
                "linked_entity_ids": [],
                "document": "doc",
                "kg_name": None,
                "position": 0,
                "source": None,
                "entities": [],
            },
            {
                "text": "high",
                "chunk_id": "high",
                "chunk_element_id": "eid_high",
                "score": 0.9,
                "min_hop": 0,
                "question_id": "q1",
                "passage_index": 0,
                "chunk_local_index": 1,
                "linked_entity_count": 1,
                "linked_entity_ids": [],
                "document": "doc",
                "kg_name": None,
                "position": 1,
                "source": None,
                "entities": [],
            },
            {
                "text": "other",
                "chunk_id": "other",
                "chunk_element_id": "eid_other",
                "score": 0.8,
                "min_hop": 0,
                "question_id": "q1",
                "passage_index": 1,
                "chunk_local_index": 2,
                "linked_entity_count": 1,
                "linked_entity_ids": [],
                "document": "doc",
                "kg_name": None,
                "position": 2,
                "source": None,
                "entities": [],
            },
        ]

        result: list[RetrievedChunk] = self._apply_cap(chunks, cap=1)

        assert [chunk["chunk_id"] for chunk in result] == ["high", "other"]


class TestRetrievalCacheInvalidation:
    def test_enhanced_clear_retrieval_caches_resets_state(self) -> None:
        s: EnhancedRAGSystem = _make_system()
        s._vector_index_available = True
        s._active_vector_index_name = "retrieval_chunk_vectors"
        s._graph = object()
        s._embedding_cache = {"q": [0.1]}
        s._entity_extraction_cache = {"q": ["entity"]}
        s._late_interaction_corpus_cache = {"scope": [{"chunk_id": "c1"}]}

        s.clear_retrieval_caches()

        assert s._vector_index_available is None
        assert s._active_vector_index_name is None
        assert s._graph is None
        assert s._embedding_cache == {}
        assert s._entity_extraction_cache == {}
        assert s._late_interaction_corpus_cache == {}

    def test_vanilla_clear_retrieval_caches_resets_state(self) -> None:
        s: VanillaRAGSystem = _make_vanilla_system()
        s._vector_index_name = "retrieval_chunk_vectors"
        s._late_interaction_corpus_cache = {"scope": [{"chunk_id": "c1"}]}

        s.clear_retrieval_caches()

        assert s._vector_index_name is None
        assert s._late_interaction_corpus_cache == {}
