import pytest

from experiments import dataset_adapters as dataset_adapters_module
from experiments.dataset_adapters import (
    BioASQRecord,
    ContextPassage,
    DatasetCorpusProfile,
    GoldRecord,
    HopCountRecord,
    HotpotQARecord,
    InferenceRecord,
    MedHopRecord,
    MedQARecord,
    MultiHopRAGRecord,
    RealMedQARecord,
    TaskType,
    TwoWikiRecord,
    adapt_2wikimultihopqa,
    adapt_bioasq,
    adapt_hotpotqa,
    adapt_medhop,
    adapt_medqa,
    adapt_multihoprag,
    adapt_realmedqa,
    build_global_corpus_passages,
    build_passage_corpus,
    get_dataset_corpus_profile,
    infer_hop_count_from_raw,
)


class MedHopFixture(MedHopRecord):
    query: str
    supports: list[str]


class MedQAFixture(MedQARecord):
    options: dict[str, str]


class MultiHopRAGFixture(MultiHopRAGRecord):
    query: str
    question_type: str


class TwoWikiFixture(TwoWikiRecord):
    evidences: list[list[str]]


class RealMedQAFixture(RealMedQARecord):
    Question: str
    Recommendation: str
    Generator: str


def test_adapt_medhop_maps_supports_and_candidates() -> None:
    raw: MedHopFixture = {
        "id": "medhop_1",
        "query": "Which protein mediates the interaction between drug A and drug B?",
        "answer": "Protein X",
        "candidates": ["Protein Y", "Protein X", "Protein Z"],
        "supports": [
            "Drug A activates Protein X in macrophages.",
            "Protein X is inhibited by Drug B in vivo.",
        ],
    }

    inf: InferenceRecord
    gold: GoldRecord
    inf, gold = adapt_medhop(raw)

    assert inf.id == "medhop_1"
    assert inf.dataset == "medhop"
    assert inf.question == raw["query"]
    assert inf.contexts == raw["supports"]
    assert inf.task_type == TaskType.MCQ.value
    assert inf.options == {"A": "Protein Y", "B": "Protein X", "C": "Protein Z"}

    assert gold.id == "medhop_1"
    assert gold.short_answer == "protein x"
    assert gold.long_answer is None
    assert gold.aliases is None


def test_adapt_medqa_handles_letter_answer_idx() -> None:
    raw: MedQAFixture = {
        "question": "Which action is correct?",
        "options": {
            "A": "Option A",
            "B": "Option B",
            "C": "Option C",
            "D": "Option D",
        },
        "answer_idx": "B",
        "answer": "Option B",
    }

    inf: InferenceRecord
    gold: GoldRecord
    inf, gold = adapt_medqa(raw, q_id="medqa_1")

    assert inf.id == "medqa_1"
    assert inf.dataset == "medqa"
    assert inf.task_type == TaskType.MCQ.value
    assert inf.contexts == []
    assert inf.options == raw["options"]
    assert gold.short_answer == "option b"


def test_adapt_multihoprag_maps_query_answer_and_evidence() -> None:
    raw: MultiHopRAGFixture = {
        "query": "Did the Reuters piece come before the AP piece?",
        "answer": "yes",
        "question_type": "temporal_query",
        "evidence_list": [
            {
                "title": "Reuters sample",
                "source": "Reuters",
                "category": "business",
                "published_at": "2023-10-01T12:00:00+00:00",
                "fact": "Reuters reported the event first.",
            },
            {
                "title": "AP sample",
                "source": "Associated Press",
                "category": "business",
                "published_at": "2023-10-02T12:00:00+00:00",
                "fact": "AP reported the event later.",
            },
        ],
    }

    inf: InferenceRecord
    gold: GoldRecord
    inf, gold = adapt_multihoprag(raw, q_id="mhr_1")

    assert inf.id == "mhr_1"
    assert inf.dataset == "multihoprag"
    assert inf.question == raw["query"]
    assert inf.task_type == TaskType.BINARY.value
    assert inf.options == {"A": "yes", "B": "no"}
    assert len(inf.contexts) == 2
    assert "Title: Reuters sample" in inf.contexts[0]
    assert "Source: Reuters" in inf.contexts[0]
    assert "Fact: Reuters reported the event first." in inf.contexts[0]

    assert gold.id == "mhr_1"
    assert gold.short_answer == "yes"
    assert gold.long_answer is None
    assert gold.aliases is None


def test_adapt_bioasq_factoid_preserves_aliases() -> None:
    raw: BioASQRecord = {
        "id": "bioasq_1",
        "body": "What drug is also known as acetylsalicylic acid?",
        "type": "factoid",
        "exact_answer": [["Aspirin", "acetylsalicylic acid"]],
        "ideal_answer": "Aspirin is also called acetylsalicylic acid.",
        "snippets": [{"text": "Aspirin, or acetylsalicylic acid, is commonly used."}],
    }

    inf: InferenceRecord
    gold: GoldRecord
    inf, gold = adapt_bioasq(raw)

    assert inf.task_type == TaskType.FREE_TEXT.value
    assert gold.short_answer == "aspirin"
    assert gold.aliases == ["acetylsalicylic acid"]


def test_adapt_2wiki_does_not_infer_aliases_from_evidences() -> None:
    raw: TwoWikiFixture = {
        "_id": "2wiki_1",
        "question": "Who is the mother of the director of Polish-Russian War?",
        "answer": "Małgorzata Braunek",
        "type": "compositional",
        "context": [["Polish-Russian War", ["Sentence one."]]],
        "evidences": [
            ["Polish-Russian War", "director", "Xawery Żuławski"],
            ["Xawery Żuławski", "mother", "Małgorzata Braunek"],
        ],
    }

    gold: GoldRecord
    _, gold = adapt_2wikimultihopqa(raw)
    assert gold.aliases is None


def test_bioasq_profile_marks_question_contexts_as_gold_evidence() -> None:
    profile: DatasetCorpusProfile = get_dataset_corpus_profile("bioasq")

    assert profile["question_context_role"] == "gold_evidence"
    assert profile["requires_shared_corpus_for_fair_retrieval"] is True


def test_adapt_realmedqa_uses_question_only_and_recommendation_as_gold() -> None:
    raw: RealMedQAFixture = {
        "Question": "How should beta-blockers be titrated in heart failure?",
        "Recommendation": "Introduce beta-blockers in a start low, go slow manner and monitor after each titration.",
        "Generator": "LLM",
        "Plausible": "Completely",
        "Answered": "Completely",
    }

    inf: InferenceRecord
    gold: GoldRecord
    inf, gold = adapt_realmedqa(raw, q_id="realmedqa_1")

    assert inf.id == "realmedqa_1"
    assert inf.dataset == "realmedqa"
    assert inf.question == raw["Question"]
    assert inf.contexts == []
    assert inf.task_type == TaskType.FREE_TEXT.value

    assert gold.short_answer == raw["Recommendation"].lower()
    assert gold.long_answer == raw["Recommendation"]


def test_realmedqa_profile_requires_shared_corpus() -> None:
    profile: DatasetCorpusProfile = get_dataset_corpus_profile("realmedqa")

    assert profile["question_context_role"] == "no_context"
    assert profile["requires_shared_corpus_for_fair_retrieval"] is True


def test_multihoprag_uses_shared_corpus_for_fair_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile: DatasetCorpusProfile = get_dataset_corpus_profile("multihoprag")

    def fake_loader(paths: list[str]) -> list[dict[str, str]]:
        return [
            {
                "title": "Reuters sample",
                "source": "Reuters",
                "author": "Reporter",
                "category": "business",
                "published_at": "2023-10-01T12:00:00+00:00",
                "body": "Reuters reported the event first.",
            },
        ]

    patch: pytest.MonkeyPatch
    with monkeypatch.context() as patch:
        patch.setattr(
            dataset_adapters_module,
            "_load_optional_json_records",
            fake_loader,
        )
        passages: list[ContextPassage] | None = build_global_corpus_passages(
            "multihoprag",
        )

    assert profile["question_context_role"] == "gold_evidence"
    assert profile["requires_shared_corpus_for_fair_retrieval"] is True
    assert passages is not None
    assert len(passages) > 0
    assert "Title: Reuters sample" in passages[0].text


def test_multihoprag_shared_corpus_absent_without_local_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dataset_adapters_module,
        "_load_optional_json_records",
        _missing_records,
    )
    passages: list[ContextPassage] | None = build_global_corpus_passages("multihoprag")
    assert passages is None


def test_bioasq_shared_corpus_absent_without_local_abstract_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dataset_adapters_module,
        "_load_optional_json_records",
        _missing_records,
    )
    passages: list[ContextPassage] | None = build_global_corpus_passages("bioasq")
    assert passages is None


def test_realmedqa_shared_corpus_uses_verified_ideal_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_realmedqa_records(candidate_paths: list[str]) -> list[dict[str, str]]:
        return [
            {
                "Question": "Q1",
                "Recommendation": "R1",
                "_normalized_plausible": "completely",
                "_normalized_answered": "completely",
                "_fallback_id": "realmedqa_a",
            },
            {
                "Question": "Q2",
                "Recommendation": "R2",
                "_normalized_plausible": "partially",
                "_normalized_answered": "completely",
                "_fallback_id": "realmedqa_b",
            },
        ]

    monkeypatch.setattr(
        dataset_adapters_module,
        "_load_realmedqa_records",
        fake_load_realmedqa_records,
    )
    passages: list[ContextPassage] | None = build_global_corpus_passages("realmedqa")

    assert passages is not None
    assert len(passages) == 1
    assert passages[0].dataset == "realmedqa"
    assert passages[0].text == "Recommendation:\nR1"


def test_infer_hop_count_from_raw_uses_question_decomposition_when_present() -> None:
    raw: HopCountRecord = {
        "question_decomposition": [
            {"q": "step 1"},
            {"q": "step 2"},
            {"q": "step 3"},
        ],
    }

    assert infer_hop_count_from_raw("musique", "3hop__example", raw) == 3


def test_infer_hop_count_from_raw_uses_dataset_specific_evidence_fields() -> None:
    multihoprag_raw: HopCountRecord = {
        "evidence_list": [
            {"fact": "A"},
            {"fact": "B"},
            {"fact": "C"},
        ],
    }
    twowiki_raw: HopCountRecord = {
        "evidences": [
            ["A", "rel", "B"],
            ["B", "rel", "C"],
        ],
    }
    hotpot_raw: HopCountRecord = {
        "supporting_facts": [
            ["Title A", 0],
            ["Title B", 1],
        ],
    }

    assert infer_hop_count_from_raw("multihoprag", "mhr_1", multihoprag_raw) == 3
    assert infer_hop_count_from_raw("2wikimultihopqa", "2wiki_1", twowiki_raw) == 2
    assert infer_hop_count_from_raw("hotpotqa", "hotpot_1", hotpot_raw) == 2


def test_infer_hop_count_from_raw_falls_back_to_id_prefix() -> None:
    assert infer_hop_count_from_raw("musique", "4hop1__abc", {}) == 4


def test_build_passage_corpus_preserves_question_local_duplicates_when_requested() -> (
    None
):
    records: list[InferenceRecord] = [
        InferenceRecord(
            id="q1",
            dataset="musique",
            question="Q1",
            contexts=["Shared passage", "Local passage 1"],
            task_type=TaskType.FREE_TEXT.value,
        ),
        InferenceRecord(
            id="q2",
            dataset="musique",
            question="Q2",
            contexts=["Shared passage", "Local passage 2"],
            task_type=TaskType.FREE_TEXT.value,
        ),
    ]

    deduped: list[ContextPassage] = build_passage_corpus(records)
    question_scoped: list[ContextPassage] = build_passage_corpus(
        records,
        dedupe_across_questions=False,
    )

    assert len(deduped) == 3
    assert len(question_scoped) == 4
    assert [p.question_id for p in question_scoped if p.text == "Shared passage"] == [
        "q1",
        "q2",
    ]


def test_build_passage_corpus_preserves_structured_titles_with_periods() -> None:
    raw: HotpotQARecord = {
        "_id": "hp1",
        "question": "Which film starred James Bond?",
        "answer": "dr. no",
        "type": "bridge",
        "context": [
            ["Dr. No", ["James Bond appears in the film."]],
        ],
    }

    inf: InferenceRecord
    inf, _ = adapt_hotpotqa(raw)
    passages: list[ContextPassage] = build_passage_corpus(
        [inf],
        dedupe_across_questions=False,
    )

    assert len(passages) == 1
    assert passages[0].text.startswith("Dr. No. James Bond")
    assert passages[0].source_title == "Dr. No"


def _missing_records(candidate_paths: list[str]) -> None:
    return None
