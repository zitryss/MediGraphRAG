import pytest

pytest.importorskip("langchain_core", reason="langchain_core not installed")

from langchain_core.runnables import RunnableLambda

from ontographrag.rag.answer_guardrails import (
    GuardrailVerdict,
    evaluate_runtime_answer_guardrail,
    parse_guardrail_verdict,
    render_guardrail_context,
)


def test_parse_guardrail_verdict_from_json() -> None:
    verdict: GuardrailVerdict = parse_guardrail_verdict(
        '{"decision":"retry","answers_question":false,"supported_by_context":true,"reason":"off target"}'
    )
    assert verdict["decision"] == "retry"
    assert verdict["answers_question"] is False
    assert verdict["supported_by_context"] is True
    assert verdict["reason"] == "off target"


def test_parse_guardrail_verdict_from_embedded_json() -> None:
    verdict: GuardrailVerdict = parse_guardrail_verdict(
        'verdict: {"decision":"abstain","answers_question":false,"supported_by_context":false,"reason":"insufficient evidence"}'
    )
    assert verdict["decision"] == "abstain"
    assert verdict["answers_question"] is False
    assert verdict["supported_by_context"] is False


def test_render_guardrail_context_limits_chunks() -> None:
    chunks: list[dict[str, str]] = [
        {"document": "a.txt", "text": "first chunk"},
        {"document": "b.txt", "text": "second chunk"},
        {"document": "c.txt", "text": "third chunk"},
    ]
    rendered: str = render_guardrail_context(chunks, max_chunks=2)
    assert "[1] a.txt: first chunk" in rendered
    assert "[2] b.txt: second chunk" in rendered
    assert "third chunk" not in rendered


def test_parse_guardrail_verdict_ambiguous_defaults_to_abstain() -> None:
    # When LLM output contains neither "keep", "retry", nor "abstain", the
    # conservative default must be abstain (not keep).
    verdict: GuardrailVerdict = parse_guardrail_verdict("I'm not sure about this one.")
    assert verdict["decision"] == "abstain"
    assert verdict["answers_question"] is False
    assert verdict["supported_by_context"] is False


def test_parse_guardrail_verdict_keep_keyword_detected() -> None:
    # Prose containing "keep" should resolve to keep when no JSON is present.
    verdict: GuardrailVerdict = parse_guardrail_verdict("You should keep this answer.")
    assert verdict["decision"] == "keep"


def test_evaluate_runtime_answer_guardrail_uses_llm_json() -> None:
    llm: RunnableLambda[object, str] = RunnableLambda(
        lambda _: (
            '{"decision":"retry","answers_question":false,"supported_by_context":true,"reason":"off target"}'
        )
    )
    verdict: GuardrailVerdict = evaluate_runtime_answer_guardrail(
        question="Who directed Lost and Delirious?",
        answer="Léa Pool",
        chunks=[
            {
                "document": "2wikimultihopqa",
                "text": "Lost and Delirious is a 2001 Canadian drama film directed by Léa Pool.",
            }
        ],
        llm=llm,
    )
    assert verdict["decision"] == "retry"
    assert verdict["answers_question"] is False
    assert verdict["supported_by_context"] is True
    assert verdict["reason"] == "off target"


def test_runtime_guardrail_preserves_retry_for_nontext_model_output() -> None:
    llm: RunnableLambda[object, object] = RunnableLambda(lambda _: object())
    verdict: GuardrailVerdict = evaluate_runtime_answer_guardrail(
        question="Question?",
        answer="Answer",
        chunks=[],
        llm=llm,
    )
    assert verdict["decision"] == "retry"
    assert verdict["reason"].startswith("guardrail_error:")
    assert verdict["raw"] == ""


@pytest.mark.parametrize("raw", ["[]", "null", "1", "true", '"keep"'])
def test_parse_guardrail_verdict_preserves_nonobject_json_failure(raw: str) -> None:
    with pytest.raises(AttributeError, match="get"):
        parse_guardrail_verdict(raw)
