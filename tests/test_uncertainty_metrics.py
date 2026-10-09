import sys
import types
from types import TracebackType
from typing import TYPE_CHECKING, ClassVar, Literal, Self, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from typing_extensions import TypedDict

from experiments import uncertainty_metrics as um

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from experiments.uncertainty_metrics import GraphPathSupportDetail


class QueryParameters(TypedDict, total=False):
    q_ids: list[str]
    a_id: str
    a_ids: list[str]
    question_id: str
    kg_name: str
    question_lower: str
    answer_lower: str
    q_id: str
    min_name_length: int
    limit: int


class PairQueryParameters(TypedDict):
    q_ids: list[str]
    a_id: str


class PathQueryParameters(TypedDict):
    q_ids: list[str]
    a_ids: list[str]


class QueryRow(TypedDict, total=False):
    id: str
    name: str
    aliases: list[str]
    original_ids: list[str]
    synonyms: list[str]
    head: str
    rel: str
    tail: str
    reachable_id: str
    a_id: str
    rel_ids: list[int]
    rel_type: str


class CountRow(TypedDict):
    n: int


class UncertaintyDetail(TypedDict, total=False):
    vanilla_correct: bool
    kg_correct: bool
    vanilla_generation_failed: bool
    kg_generation_failed: bool
    vanilla_graph_path_support: float
    kg_graph_path_support: float
    vanilla_graph_path_support_null_reason: str
    kg_graph_path_support_null_reason: str
    vanilla_support_entailment_uncertainty: float
    kg_support_entailment_uncertainty: float
    vanilla_evidence_conflict_uncertainty: float
    kg_evidence_conflict_uncertainty: float


class _FakeSentenceTransformer:
    init_calls: ClassVar[list[str]] = []

    def __init__(self, model_name: str) -> None:
        self.model_name: str = model_name
        self.__class__.init_calls.append(model_name)

    def encode(
        self,
        texts: str | list[str],
        show_progress_bar: bool = False,
        normalize_embeddings: bool = False,
    ) -> NDArray[np.float64]:
        if isinstance(texts, str):
            texts = [texts]
        vecs: NDArray[np.float64] = np.ones((len(texts), 3), dtype=float)
        if normalize_embeddings:
            norms: NDArray[np.float64] = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / norms
        return vecs


def test_sentence_transformer_helper_reuses_normalized_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module: types.SimpleNamespace = types.SimpleNamespace(
        SentenceTransformer=_FakeSentenceTransformer,
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(um, "_SENTENCE_TRANSFORMER_CACHE", {})
    _FakeSentenceTransformer.init_calls.clear()

    model_a: SentenceTransformer = um._get_or_load_sentence_transformer(
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    model_b: SentenceTransformer = um._get_or_load_sentence_transformer(
        "all-MiniLM-L6-v2",
    )

    assert model_a is model_b
    assert _FakeSentenceTransformer.init_calls == ["all-MiniLM-L6-v2"]


def test_evidence_vn_entropy_query_checks_tail_reachability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_queries: list[str] = []

    class FakeSession:
        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> Literal[False]:
            return False

        def run(
            self,
            query: str,
            params: QueryParameters | None = None,
        ) -> list[QueryRow]:
            captured_queries.append(query)
            if "RETURN DISTINCT e.id AS id" in query:
                return [{"id": "q1"}]
            return [{"head": "Head", "rel": "RELATED_TO", "tail": "Tail"}]

    class FakeDriver:
        def session(self, database: str | None = None) -> FakeSession:
            return FakeSession()

        def close(self) -> None:
            return None

    class FakeGraphDatabase:
        @staticmethod
        def driver(uri: str, auth: tuple[str, str] | None = None) -> FakeDriver:
            return FakeDriver()

    fake_sentence_module: types.SimpleNamespace = types.SimpleNamespace(
        SentenceTransformer=_FakeSentenceTransformer,
    )
    fake_neo4j_module: types.SimpleNamespace = types.SimpleNamespace(
        GraphDatabase=FakeGraphDatabase,
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_module)
    monkeypatch.setitem(sys.modules, "neo4j", fake_neo4j_module)
    monkeypatch.setattr(um, "_SENTENCE_TRANSFORMER_CACHE", {})
    _FakeSentenceTransformer.init_calls.clear()

    score: float = um.compute_evidence_vn_entropy(
        question="Which county is Hughesville in?",
        neo4j_uri="bolt://unused",
        neo4j_user="neo4j",
        neo4j_password="test",
        max_hops=4,
        n_triples=5,
    )

    assert 0.0 <= score <= 1.0
    triple_query: str = captured_queries[1]
    assert "MATCH path = (q_e:__Entity__)-[*1..4]-(t)" in triple_query


def test_graph_path_support_query_applies_question_local_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_queries: list[str] = []

    class FakeSession:
        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> Literal[False]:
            return False

        def run(
            self,
            query: str,
            params: QueryParameters | None = None,
            timeout: float | None = None,
        ) -> list[QueryRow]:
            captured_queries.append(query)
            if "RETURN DISTINCT e.id AS id," in query:
                return [
                    {
                        "id": "q1",
                        "name": "dune",
                        "aliases": [],
                        "original_ids": [],
                        "synonyms": [],
                    },
                    {
                        "id": "a1",
                        "name": "frank herbert",
                        "aliases": [],
                        "original_ids": [],
                        "synonyms": [],
                    },
                ]
            return [{"reachable_id": "a1"}]

    class FakeDriver:
        def session(self, database: str | None = None) -> FakeSession:
            return FakeSession()

        def close(self) -> None:
            return None

    class FakeGraphDatabase:
        @staticmethod
        def driver(uri: str, auth: tuple[str, str] | None = None) -> FakeDriver:
            return FakeDriver()

    fake_neo4j_module: types.SimpleNamespace = types.SimpleNamespace(
        GraphDatabase=FakeGraphDatabase,
    )
    monkeypatch.setitem(sys.modules, "neo4j", fake_neo4j_module)

    score: float = um.compute_graph_path_support(
        question="Who wrote Dune?",
        answer="Frank Herbert",
        neo4j_uri="bolt://unused",
        neo4j_user="neo4j",
        neo4j_password="test",
        kg_name="musique",
        question_id="q1",
        max_hops=3,
    )

    assert score == 0.0
    assert "c.questionId = $question_id" in captured_queries[0]
    assert (
        "$question_id IN coalesce(relationships(p)[idx].questionIds, [])"
        in captured_queries[1]
    )


def test_graph_path_support_matches_aliases_without_abstaining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSession:
        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> Literal[False]:
            return False

        def run(
            self,
            query: str,
            params: QueryParameters | None = None,
            timeout: float | None = None,
        ) -> list[QueryRow]:
            if "RETURN DISTINCT e.id AS id," in query:
                return [
                    {
                        "id": "q1",
                        "name": "acetylsalicylic acid",
                        "aliases": ["aspirin"],
                        "original_ids": [],
                        "synonyms": [],
                    },
                    {
                        "id": "a1",
                        "name": "bleeding risk",
                        "aliases": [],
                        "original_ids": [],
                        "synonyms": [],
                    },
                ]
            return [{"reachable_id": "a1"}]

    class FakeDriver:
        def session(self, database: str | None = None) -> FakeSession:
            return FakeSession()

        def close(self) -> None:
            return None

    class FakeGraphDatabase:
        @staticmethod
        def driver(uri: str, auth: tuple[str, str] | None = None) -> FakeDriver:
            return FakeDriver()

    monkeypatch.setitem(
        sys.modules,
        "neo4j",
        types.SimpleNamespace(GraphDatabase=FakeGraphDatabase),
    )

    detail: GraphPathSupportDetail = um.compute_graph_path_support_detailed(
        question="Should aspirin be stopped before surgery?",
        answer="Bleeding risk",
        neo4j_uri="bolt://unused",
        neo4j_user="neo4j",
        neo4j_password="test",
    )

    assert detail["null_reason"] is None
    assert detail["score"] == 0.0


def test_graph_path_support_uses_non_self_question_entities_for_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSession:
        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> Literal[False]:
            return False

        def run(
            self,
            query: str,
            params: QueryParameters | None = None,
            timeout: float | None = None,
        ) -> list[QueryRow]:
            if "RETURN DISTINCT e.id AS id," in query:
                return [
                    {
                        "id": "q1",
                        "name": "william nigh",
                        "aliases": [],
                        "original_ids": [],
                        "synonyms": [],
                    },
                    {
                        "id": "a1",
                        "name": "the ape",
                        "aliases": [],
                        "original_ids": [],
                        "synonyms": [],
                    },
                ]
            # The production per-pair query always supplies q_ids and a_id.
            pair_params: PairQueryParameters = cast(PairQueryParameters, params)
            assert pair_params["q_ids"] == ["q1"]
            assert pair_params["a_id"] == "a1"
            return [{"reachable_id": "a1"}]

    class FakeDriver:
        def session(self, database: str | None = None) -> FakeSession:
            return FakeSession()

        def close(self) -> None:
            return None

    class FakeGraphDatabase:
        @staticmethod
        def driver(uri: str, auth: tuple[str, str] | None = None) -> FakeDriver:
            return FakeDriver()

    monkeypatch.setitem(
        sys.modules,
        "neo4j",
        types.SimpleNamespace(GraphDatabase=FakeGraphDatabase),
    )

    detail: GraphPathSupportDetail = um.compute_graph_path_support_detailed(
        question="Did William Nigh direct The Ape?",
        answer="The Ape",
        neo4j_uri="bolt://unused",
        neo4j_user="neo4j",
        neo4j_password="test",
    )

    assert detail["null_reason"] is None
    assert detail["score"] == 0.0


def test_sps_uses_non_self_question_entities_for_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSession:
        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> Literal[False]:
            return False

        def run(
            self,
            query: str,
            params: QueryParameters | None = None,
            timeout: float | None = None,
        ) -> list[QueryRow]:
            if "RETURN DISTINCT e.id AS id," in query:
                return [
                    {
                        "id": "q1",
                        "name": "william nigh",
                        "aliases": [],
                        "original_ids": [],
                        "synonyms": [],
                    },
                    {
                        "id": "a1",
                        "name": "the ape",
                        "aliases": [],
                        "original_ids": [],
                        "synonyms": [],
                    },
                ]
            # The production path query always supplies q_ids and a_ids.
            path_params: PathQueryParameters = cast(PathQueryParameters, params)
            assert path_params["q_ids"] == ["q1"]
            assert path_params["a_ids"] == ["a1"]
            return [{"a_id": "a1", "rel_ids": [11, 12]}]

    class FakeDriver:
        def session(self, database: str | None = None) -> FakeSession:
            return FakeSession()

        def close(self) -> None:
            return None

    class FakeGraphDatabase:
        @staticmethod
        def driver(uri: str, auth: tuple[str, str] | None = None) -> FakeDriver:
            return FakeDriver()

    monkeypatch.setitem(
        sys.modules,
        "neo4j",
        types.SimpleNamespace(GraphDatabase=FakeGraphDatabase),
    )

    detail: GraphPathSupportDetail = (
        um.compute_subgraph_perturbation_stability_detailed(
            question="Did William Nigh direct The Ape?",
            answer="The Ape",
            neo4j_uri="bolt://unused",
            neo4j_user="neo4j",
            neo4j_password="test",
            n_perturbations=2,
        )
    )

    assert detail["null_reason"] is None
    assert 0.0 <= detail["score"] <= 1.0


def test_competing_answer_alternatives_counts_reverse_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_queries: list[str] = []

    class FakeSingleResult:
        def __init__(self, row: CountRow) -> None:
            self._row: CountRow = row

        def single(self) -> CountRow:
            return self._row

    class FakeSession:
        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> Literal[False]:
            return False

        def run(
            self,
            query: str,
            params: QueryParameters | None = None,
        ) -> list[QueryRow] | FakeSingleResult:
            captured_queries.append(query)
            if "$question_lower CONTAINS toLower(e.name)" in query:
                return [{"id": "q1", "name": "paris"}]
            if "$answer_lower CONTAINS toLower(e.name)" in query:
                return [{"id": "a1", "name": "france"}]
            if "MATCH (q_e:__Entity__)-[r]->(a_e:__Entity__)" in query:
                return []
            if "MATCH (a_e:__Entity__)-[r]->(q_e:__Entity__)" in query:
                return [{"rel_type": "CAPITAL_OF"}]
            if "MATCH (alt:__Entity__)-[r:CAPITAL_OF]->(q_e:__Entity__)" in query:
                return FakeSingleResult({"n": 3})
            raise AssertionError(f"Unexpected query: {query}")

    class FakeDriver:
        def session(self, database: str | None = None) -> FakeSession:
            return FakeSession()

        def close(self) -> None:
            return None

    class FakeGraphDatabase:
        @staticmethod
        def driver(uri: str, auth: tuple[str, str] | None = None) -> FakeDriver:
            return FakeDriver()

    fake_neo4j_module: types.SimpleNamespace = types.SimpleNamespace(
        GraphDatabase=FakeGraphDatabase,
    )
    monkeypatch.setitem(sys.modules, "neo4j", fake_neo4j_module)

    score: float = um.compute_competing_answer_alternatives(
        question="What country is Paris the capital of?",
        answer="France",
        neo4j_uri="bolt://unused",
        neo4j_user="neo4j",
        neo4j_password="test",
    )

    assert score == 0.75
    assert any(
        "MATCH (alt:__Entity__)-[r:CAPITAL_OF]->(q_e:__Entity__)" in query
        for query in captured_queries
    )


def test_compute_auroc_aurec_excludes_generation_failures_by_default() -> None:
    details: list[UncertaintyDetail] = [
        {
            "vanilla_correct": True,
            "vanilla_generation_failed": False,
            "vanilla_graph_path_support": 0.1,
            "kg_correct": True,
            "kg_generation_failed": False,
            "kg_graph_path_support": 0.1,
        },
        {
            "vanilla_correct": True,
            "vanilla_generation_failed": False,
            "vanilla_graph_path_support": 0.2,
            "kg_correct": True,
            "kg_generation_failed": False,
            "kg_graph_path_support": 0.2,
        },
        {
            "vanilla_correct": False,
            "vanilla_generation_failed": False,
            "vanilla_graph_path_support": 0.9,
            "kg_correct": False,
            "kg_generation_failed": False,
            "kg_graph_path_support": 0.9,
        },
        {
            "vanilla_correct": False,
            "vanilla_generation_failed": True,
            "vanilla_graph_path_support": 0.0,
            "kg_correct": False,
            "kg_generation_failed": True,
            "kg_graph_path_support": 0.0,
        },
    ]

    clean_only: dict[str, dict[str, float]] = um.compute_auroc_aurec(
        details,
        metric_names=["graph_path_support"],
    )
    all_rows: dict[str, dict[str, float]] = um.compute_auroc_aurec(
        details,
        metric_names=["graph_path_support"],
        exclude_generation_failures=False,
    )

    assert clean_only["vanilla_rag"]["graph_path_support_auroc"] == 1.0
    assert clean_only["kg_rag"]["graph_path_support_auroc"] == 1.0
    assert all_rows["vanilla_rag"]["graph_path_support_auroc"] < 1.0
    assert all_rows["kg_rag"]["graph_path_support_auroc"] < 1.0
    assert (
        clean_only["vanilla_rag"]["graph_path_support_aurec"]
        < all_rows["vanilla_rag"]["graph_path_support_aurec"]
    )


def test_compute_auroc_aurec_default_metric_list_includes_grounding_metrics() -> None:
    details: list[UncertaintyDetail] = [
        {
            "vanilla_correct": True,
            "vanilla_generation_failed": False,
            "vanilla_support_entailment_uncertainty": 0.1,
            "vanilla_evidence_conflict_uncertainty": 0.0,
            "kg_correct": True,
            "kg_generation_failed": False,
            "kg_support_entailment_uncertainty": 0.1,
            "kg_evidence_conflict_uncertainty": 0.0,
        },
        {
            "vanilla_correct": False,
            "vanilla_generation_failed": False,
            "vanilla_support_entailment_uncertainty": 0.9,
            "vanilla_evidence_conflict_uncertainty": 0.7,
            "kg_correct": False,
            "kg_generation_failed": False,
            "kg_support_entailment_uncertainty": 0.9,
            "kg_evidence_conflict_uncertainty": 0.7,
        },
    ]

    metrics: dict[str, dict[str, float]] = um.compute_auroc_aurec(details)

    assert "support_entailment_uncertainty_auroc" in metrics["vanilla_rag"]
    assert "evidence_conflict_uncertainty_auroc" in metrics["kg_rag"]


def test_compute_auroc_aurec_reports_all_observed_gps_null_reasons() -> None:
    details: list[UncertaintyDetail] = [
        {
            "vanilla_correct": True,
            "vanilla_generation_failed": False,
            "vanilla_graph_path_support": 0.5,
            "vanilla_graph_path_support_null_reason": "no_input",
            "kg_correct": True,
            "kg_generation_failed": False,
            "kg_graph_path_support": 0.5,
            "kg_graph_path_support_null_reason": "neo4j_unavailable",
        },
        {
            "vanilla_correct": False,
            "vanilla_generation_failed": False,
            "vanilla_graph_path_support": 0.1,
            "vanilla_graph_path_support_null_reason": "",
            "kg_correct": False,
            "kg_generation_failed": False,
            "kg_graph_path_support": 0.1,
            "kg_graph_path_support_null_reason": "",
        },
        {
            "vanilla_correct": True,
            "vanilla_generation_failed": False,
            "vanilla_graph_path_support": 0.2,
            "vanilla_graph_path_support_null_reason": "",
            "kg_correct": True,
            "kg_generation_failed": False,
            "kg_graph_path_support": 0.2,
            "kg_graph_path_support_null_reason": "",
        },
        {
            "vanilla_correct": False,
            "vanilla_generation_failed": False,
            "vanilla_graph_path_support": 0.9,
            "vanilla_graph_path_support_null_reason": "",
            "kg_correct": False,
            "kg_generation_failed": False,
            "kg_graph_path_support": 0.9,
            "kg_graph_path_support_null_reason": "",
        },
    ]

    metrics: dict[str, dict[str, float]] = um.compute_auroc_aurec(
        details,
        metric_names=["graph_path_support"],
    )

    assert metrics["vanilla_rag"]["graph_path_support_null_rate"] == 0.25
    assert metrics["vanilla_rag"]["graph_path_support_null_no_input"] == 0.25
    assert metrics["kg_rag"]["graph_path_support_null_rate"] == 0.25
    assert metrics["kg_rag"]["graph_path_support_null_neo4j_unavailable"] == 0.25
