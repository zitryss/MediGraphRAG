import json
from collections.abc import Callable, Hashable
from pathlib import Path

import pytest

from experiments import dataset_adapters as dataset_adapters_module
from experiments.dataset_adapters import (
    BioASQFactoidRecord,
    BioASQListRecord,
    BioASQQuestion,
    BioASQRecord,
    BioASQSummaryRecord,
    BioASQYesNoRecord,
    ContextPassage,
    DatasetCorpusProfile,
    GoldRecord,
    HopCountRecord,
    HotpotQARecord,
    InferenceRecord,
    JSONValue,
    MedHopRecord,
    MedQARecord,
    MMLUProducedRecord,
    MultiHopRAGRecord,
    MuSiQueRecord,
    NormalizedRealMedQARecord,
    RawDataset,
    RealMedQARecord,
    TaskType,
    TwoWikiRecord,
    adapt_2wikimultihopqa,
    adapt_bioasq,
    adapt_hotpotqa,
    adapt_medhop,
    adapt_medqa,
    adapt_multihoprag,
    adapt_musique,
    adapt_realmedqa,
    build_global_corpus_passages,
    build_passage_corpus,
    get_dataset_corpus_profile,
    infer_hop_count_from_raw,
    load_raw_dataset,
    normalize_dataset,
)

MALFORMED_DATASET_CASES: tuple[tuple[str, str, JSONValue, type[Exception]], ...] = (
    (
        "bioasq",
        "MIRAGE/rawdata/bioasq/Task10BGoldenEnriched/10B1_golden.json",
        None,
        TypeError,
    ),
    (
        "bioasq",
        "MIRAGE/rawdata/bioasq/Task10BGoldenEnriched/10B1_golden.json",
        {"questions": [{"id": []}]},
        TypeError,
    ),
    (
        "medqa",
        "MIRAGE/rawdata/medqa/data_clean/questions/US/4_options/phrases_no_exclude_test.jsonl",
        17,
        AttributeError,
    ),
    (
        "medqa",
        "MIRAGE/rawdata/medqa/data_clean/questions/US/4_options/phrases_no_exclude_test.jsonl",
        {"id": []},
        TypeError,
    ),
    (
        "medmcqa",
        "MIRAGE/rawdata/medmcqa/data/test.json",
        [None],
        TypeError,
    ),
    (
        "medmcqa",
        "MIRAGE/rawdata/medmcqa/data/test.json",
        [{"id": []}],
        TypeError,
    ),
    (
        "hotpotqa",
        "MIRAGE/rawdata/hotpotqa/hotpot_dev_fullwiki_v1.json",
        {"data": 17},
        TypeError,
    ),
    (
        "hotpotqa",
        "MIRAGE/rawdata/hotpotqa/hotpot_dev_fullwiki_v1.json",
        [17],
        AttributeError,
    ),
    (
        "musique",
        "MIRAGE/rawdata/musique/dev.jsonl",
        17,
        AttributeError,
    ),
    (
        "musique",
        "MIRAGE/rawdata/musique/dev.json",
        {"questions": [17]},
        AttributeError,
    ),
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


@pytest.mark.parametrize(
    ("raw", "short_answer", "aliases", "task_type"),
    [
        ({"type": "yesno", "exact_answer": "YES"}, "yes", None, "binary"),
        (
            {"type": "factoid", "exact_answer": [["Aspirin", "ASA"], ["Drug"]]},
            "aspirin",
            ["asa", "drug"],
            "free_text",
        ),
        (
            {"type": "factoid", "exact_answer": ["Aspirin", "ASA"]},
            "aspirin",
            ["asa"],
            "free_text",
        ),
        (
            {"type": "factoid", "exact_answer": "ABC"},
            "a",
            ["b", "c"],
            "free_text",
        ),
        ({"type": "list", "exact_answer": [["Aspirin"]]}, "", None, "free_text"),
        ({"type": "summary"}, "", None, "free_text"),
    ],
)
def test_bioasq_supported_kind_shapes(
    raw: BioASQQuestion,
    short_answer: str,
    aliases: list[str] | None,
    task_type: str,
) -> None:
    inference: InferenceRecord
    gold: GoldRecord
    inference, gold = adapt_bioasq(raw)
    assert inference.task_type == task_type
    assert gold.short_answer == short_answer
    assert gold.aliases == aliases


def test_bioasq_kind_contracts_and_list_ideal_answer() -> None:
    yesno: BioASQYesNoRecord = {"type": "yesno", "exact_answer": "no"}
    factoid: BioASQFactoidRecord = {"type": "factoid", "exact_answer": ["Drug"]}
    listing: BioASQListRecord = {"type": "list", "exact_answer": [["Drug"]]}
    summary: BioASQSummaryRecord = {
        "type": "summary",
        "ideal_answer": ["First answer.", "Second answer."],
    }
    records: list[BioASQQuestion] = [yesno, factoid, listing, summary]
    for raw in records:
        inference: InferenceRecord
        gold: GoldRecord
        inference, gold = adapt_bioasq(raw)
        assert inference.dataset == "bioasq"
        assert gold.long_answer == raw.get("ideal_answer", "")
        assert gold.to_dict()["long_answer"] == raw.get("ideal_answer", "")
    inference, gold = adapt_bioasq(summary)
    assert gold.long_answer is summary["ideal_answer"]
    assert gold.to_dict()["long_answer"] is not summary["ideal_answer"]


def test_bioasq_malformed_yesno_retains_attribute_error() -> None:
    raw: BioASQRecord = {"type": "yesno", "exact_answer": ["yes"]}
    with pytest.raises(AttributeError, match="lower"):
        adapt_bioasq(raw)


def test_bioasq_empty_yesno_sequence_retains_empty_answer() -> None:
    raw: BioASQRecord = {"type": "yesno", "exact_answer": []}
    inference: InferenceRecord
    gold: GoldRecord
    inference, gold = adapt_bioasq(raw)
    assert inference.task_type == "binary"
    assert gold.short_answer == ""


@pytest.mark.parametrize(
    "contents",
    [
        """[{"title": 7, "body": ["raw", null]}, 42, null]""",
        """{"records": [{"title": 7, "body": ["raw", null]}, 42, null]}""",
    ],
)
def test_shared_corpus_filters_nonrecords_without_coercing_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: str,
) -> None:
    corpus: Path = tmp_path / "MIRAGE" / "rawdata" / "multihoprag"
    corpus.mkdir(parents=True)
    (corpus / "corpus.json").write_text(contents)
    monkeypatch.chdir(tmp_path)
    passages: list[ContextPassage] | None = build_global_corpus_passages("multihoprag")
    assert passages is not None
    assert len(passages) == 1
    assert "Title: 7" in passages[0].text
    assert "['raw', None]" in passages[0].text


@pytest.mark.parametrize("question_id", [17, 2.5, ("part", 3)])
def test_adapter_explicit_hashable_ids_are_only_stringified(
    question_id: Hashable,
) -> None:
    raw: BioASQYesNoRecord = {"type": "yesno", "exact_answer": "yes"}
    inference: InferenceRecord
    gold: GoldRecord
    inference, gold = adapt_bioasq(raw, q_id=question_id)
    assert inference.id == str(question_id)
    assert gold.id == str(question_id)


def _write_raw_dataset(tmp_path: Path, relative_path: str, payload: JSONValue) -> None:
    path: Path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.mark.parametrize(
    ("dataset", "relative_path", "payload", "expected"),
    [
        (
            "bioasq",
            "MIRAGE/rawdata/bioasq/Task10BGoldenEnriched/10B1_golden.json",
            {"questions": [{"id": "b", "body": "Question"}]},
            {"b": {"id": "b", "body": "Question"}},
        ),
        (
            "medqa",
            "MIRAGE/rawdata/medqa/data_clean/questions/US/4_options/phrases_no_exclude_test.jsonl",
            {"id": "m", "question": "Question"},
            {"m": {"id": "m", "question": "Question"}},
        ),
        (
            "medmcqa",
            "MIRAGE/rawdata/medmcqa/data/test.json",
            [{"id": "m", "question": "Question"}],
            {"m": {"id": "m", "question": "Question"}},
        ),
        (
            "hotpotqa",
            "MIRAGE/rawdata/hotpotqa/hotpot_dev_fullwiki_v1.json",
            [{"_id": "h", "question": "Question"}],
            {"h": {"_id": "h", "question": "Question"}},
        ),
        (
            "musique",
            "MIRAGE/rawdata/musique/dev.jsonl",
            {"id": "m", "question": "Question"},
            {"m": {"id": "m", "question": "Question"}},
        ),
        (
            "musique",
            "MIRAGE/rawdata/musique/dev.json",
            {"questions": [{"id": "m", "question": "Question"}]},
            {"m": {"id": "m", "question": "Question"}},
        ),
        (
            "multihoprag",
            "MIRAGE/rawdata/multihoprag/MultiHopRAG.json",
            {"data": [{"id": "m", "query": "Question"}]},
            {"m": {"id": "m", "query": "Question"}},
        ),
        ("pubmedqa", "MIRAGE/rawdata/pubmedqa/data/test_set.json", 17, 17),
        ("medmcqa", "MIRAGE/rawdata/medmcqa/data/test.json", 17, 17),
    ],
)
def test_raw_dataset_successful_shapes_are_not_falsely_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: str,
    relative_path: str,
    payload: JSONValue,
    expected: JSONValue,
) -> None:
    _write_raw_dataset(tmp_path, relative_path, payload)
    monkeypatch.chdir(tmp_path)
    loaded: RawDataset = load_raw_dataset(dataset)
    assert loaded == expected


@pytest.mark.parametrize(
    ("dataset", "relative_path", "payload", "error"),
    MALFORMED_DATASET_CASES,
)
def test_raw_dataset_shape_errors_are_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: str,
    relative_path: str,
    payload: JSONValue,
    error: type[Exception],
) -> None:
    _write_raw_dataset(tmp_path, relative_path, payload)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(error):
        load_raw_dataset(dataset)


def test_mmlu_csv_producer_does_not_claim_cells_are_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path: Path = tmp_path / "MIRAGE/rawdata/mmlu/data/test/anatomy_test.csv"
    path.parent.mkdir(parents=True)
    path.write_text("1,2,3,4,5,6\n")
    monkeypatch.chdir(tmp_path)
    loaded: dict[str, MMLUProducedRecord] = load_raw_dataset("mmlu")
    assert loaded["anatomy_0"]["question"] == 1
    assert loaded["anatomy_0"]["answer"] == 6


def test_realmedqa_producer_has_required_normalized_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_raw_dataset(
        tmp_path,
        "MIRAGE/rawdata/realmedqa/RealMedQA.json",
        [
            {
                "id": "record",
                "Question": "Question",
                "Recommendation": "Answer",
                "Plausible": "Completely",
                "Answered": "Completely",
            },
        ],
    )
    monkeypatch.chdir(tmp_path)
    loaded: dict[str, NormalizedRealMedQARecord] = load_raw_dataset("realmedqa")
    assert loaded["record"]["_normalized_plausible"] == "completely"
    assert loaded["record"]["_normalized_answered"] == "completely"
    assert loaded["record"]["_fallback_id"].startswith("realmedqa_")


def test_normalization_retains_nonmapping_root_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_raw_dataset(tmp_path, "MIRAGE/rawdata/pubmedqa/data/test_set.json", [])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(AttributeError, match="items"):
        normalize_dataset("pubmedqa")


def test_normalization_retains_existing_unvalidated_field_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_raw_dataset(
        tmp_path,
        "MIRAGE/rawdata/pubmedqa/data/test_set.json",
        {"record": {"QUESTION": 17, "CONTEXTS": [], "final_decision": "yes"}},
    )
    monkeypatch.chdir(tmp_path)
    inference: list[InferenceRecord]
    gold: list[GoldRecord]
    inference, gold = normalize_dataset("pubmedqa")
    assert len(inference) == len(gold) == 1
    assert repr(inference[0].question) == "17"


def test_bioasq_raw_factoid_preserves_existing_mixed_alias_rules() -> None:
    raw: BioASQRecord = {
        "type": "factoid",
        "exact_answer": [["Aspirin", 17], 42, "ASA"],
    }
    inference: InferenceRecord
    gold: GoldRecord
    inference, gold = adapt_bioasq(raw)
    assert inference.task_type == "free_text"
    assert gold.short_answer == "aspirin"
    assert gold.aliases == ["17", "asa"]


def test_bioasq_raw_factoid_nontext_primary_retains_error() -> None:
    raw: BioASQRecord = {"type": "factoid", "exact_answer": [[17]]}
    with pytest.raises(AttributeError, match="lower"):
        adapt_bioasq(raw)


def test_musique_guarded_raw_containers_keep_open_values() -> None:
    raw: MuSiQueRecord = {
        "context": [
            ["Title", [17, None]],
            {"title": 12, "text": False},
            [],
        ],
    }
    inference: InferenceRecord
    gold: GoldRecord
    inference, gold = adapt_musique(raw)
    assert inference.contexts == ["Title. 17 None", "12. False", "[]"]
    assert gold.short_answer == ""


def test_musique_guarded_pair_does_not_validate_its_body() -> None:
    raw: MuSiQueRecord = {"context": [["Title", 17]]}
    with pytest.raises(TypeError, match="iterable"):
        adapt_musique(raw)


def test_hop_inference_keeps_nonmapping_id_fallback() -> None:
    assert infer_hop_count_from_raw("musique", "3hop1__id", None) == 3
    assert infer_hop_count_from_raw("musique", "3hop1__id", [1, 2]) == 3


@pytest.mark.parametrize("adapter", [adapt_hotpotqa, adapt_2wikimultihopqa])
def test_heterogeneous_passage_pair_retains_nontext_title_error(
    adapter: Callable[[TwoWikiRecord], tuple[InferenceRecord, GoldRecord]],
) -> None:
    raw: TwoWikiRecord = {"context": [[["not-a-text-title"], ["Sentence"]]]}
    with pytest.raises(TypeError, match="concatenate"):
        adapter(raw)
