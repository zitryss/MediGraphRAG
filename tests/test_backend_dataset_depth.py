"""Dataset release-shape and gold/inference separation contracts."""

from __future__ import annotations

import pytest

from experiments import dataset_adapters as adapters
from experiments.dataset_adapters import GoldRecord, InferenceRecord


@pytest.mark.parametrize(
    "answer_idx,explicit,expected",
    [("1", "", "beta"), ("b", "", "beta"), (99, "", ""), (0, "Explicit", "explicit")],
)
def test_medqa_release_answer_variants(
    answer_idx: str | int,
    explicit: str,
    expected: str,
) -> None:
    inference: InferenceRecord
    gold: GoldRecord
    inference, gold = adapters.adapt_medqa(
        {
            "id": "q",
            "question": "Which?",
            "options": {"0": "Alpha", "1": "Beta"},
            "answer_idx": answer_idx,
            "answer": explicit,
        },
    )
    assert gold.short_answer == expected
    assert inference.options == {"A": "Alpha", "B": "Beta"}
    assert inference.contexts == [] and not hasattr(inference, "short_answer")


@pytest.mark.parametrize(
    "raw,expected,aliases",
    [
        (
            [["Aspirin", "ASA"], ["Acetylsalicylic acid"]],
            "aspirin",
            ["asa", "acetylsalicylic acid"],
        ),
        (["Aspirin", "ASA"], "aspirin", ["asa"]),
        ([], "", None),
    ],
)
def test_bioasq_factoid_aliases_stay_in_gold(
    raw: list[str] | list[list[str]],
    expected: str,
    aliases: list[str] | None,
) -> None:
    inference: InferenceRecord
    gold: GoldRecord
    inference, gold = adapters.adapt_bioasq(
        {
            "id": "q",
            "body": "What?",
            "type": "factoid",
            "exact_answer": raw,
            "snippets": [{"text": "retrieval evidence"}, {}],
        },
    )
    assert gold.short_answer == expected and gold.aliases == aliases
    assert inference.contexts == ["retrieval evidence"]
    assert not hasattr(inference, "aliases")


def test_musique_paragraph_and_legacy_context_variants() -> None:
    record: InferenceRecord
    gold: GoldRecord
    record, gold = adapters.adapt_musique(
        {
            "id": "q",
            "question": " Where? ",
            "answer": "City",
            "answer_aliases": ["City", "Alias", ""],
            "paragraphs": [
                {"title": "Title", "paragraph_text": "Body"},
                {"title": "Existing", "paragraph_text": "Existing title body"},
                "plain",
                {},
            ],
        },
    )
    assert record.contexts == ["Title. Body", "Existing title body", "plain"]
    assert record.context_titles == ["Title", "Existing", None]
    assert gold.aliases == ["alias"]
    legacy: InferenceRecord
    legacy, _ = adapters.adapt_musique(
        {
            "context": [
                ["Title", ["one", "two"]],
                {"title": "Other", "text": "body"},
                "standalone",
                "",
            ],
        },
    )
    assert legacy.contexts == ["Title. one two", "Other. body", "standalone"]


def test_medhop_candidates_and_multihoprag_evidence_do_not_include_gold() -> None:
    record: InferenceRecord
    gold: GoldRecord
    record, gold = adapters.adapt_medhop(
        {
            "id": "q",
            "query": "Relation?",
            "answer": "DrugB",
            "supports": ["Source evidence", " "],
            "candidates": ["DrugA", "DrugB", ""],
        },
    )
    assert record.options == {"A": "DrugA", "B": "DrugB"}
    assert record.contexts == ["Source evidence"] and gold.short_answer == "drugb"
    binary: InferenceRecord
    answer: GoldRecord
    binary, answer = adapters.adapt_multihoprag(
        {
            "id": "q",
            "query": "True?",
            "answer": "Yes",
            "evidence_list": [
                None,
                {"title": "Source", "fact": "Local evidence", "source": "paper"},
            ],
        },
    )
    assert binary.options == {"A": "yes", "B": "no"}
    assert "Local evidence" in binary.contexts[0]
    assert answer.short_answer == "yes"
