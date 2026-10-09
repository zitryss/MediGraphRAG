from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableLambda

from ontographrag.rag.answer_guardrails import (
    GuardrailVerdict,
    evaluate_runtime_answer_guardrail,
)


def test_guardrail_preserves_retry_for_non_message_runnable_output() -> None:
    def invalid_output(prompt: PromptValue) -> object:
        return 17

    llm: Runnable[PromptValue, object] = RunnableLambda(invalid_output)
    verdict: GuardrailVerdict = evaluate_runtime_answer_guardrail(
        question="Which finding?",
        answer="A finding.",
        chunks=[],
        llm=llm,
    )

    assert verdict["decision"] == "retry"
    assert verdict["answers_question"] is False
    assert verdict["supported_by_context"] is False
    assert verdict["reason"].startswith("guardrail_error:")
    assert "validation error" in verdict["reason"]
    assert verdict["raw"] == ""
