import json
import logging
import re
from collections.abc import Mapping, Sequence
from typing import TypeAlias

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from typing_extensions import TypedDict

GuardrailJSON: TypeAlias = (
    str | int | float | bool | None | list["GuardrailJSON"] | dict[str, "GuardrailJSON"]
)


class GuardrailVerdict(TypedDict):
    decision: str
    answers_question: bool
    supported_by_context: bool
    reason: str
    raw: str


RUNTIME_GUARDRAIL_ABSTENTION: str = "I’m not confident the retrieved evidence supports a reliable answer to that question."


def render_guardrail_context(
    chunks: Sequence[Mapping[str, object]],
    max_chunks: int = 8,
    max_chars: int = 320,
) -> str:
    rendered: list[str] = []
    for idx, chunk in enumerate(chunks[:max_chunks], 1):
        text: str = " ".join(str(chunk.get("text", "")).split())
        text = text[:max_chars]
        document: str = str(chunk.get("document") or chunk.get("doc_name") or "unknown")
        rendered.append(f"[{idx}] {document}: {text}")
    return "\n".join(rendered)


def parse_guardrail_verdict(raw: str) -> GuardrailVerdict:
    text: str = str(raw or "").strip()
    # The default decoder accepts scalar and array JSON too. Keep that existing
    # failure surface visible; an object annotation would falsely validate it.
    payload: GuardrailJSON = {}

    if text:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            match: re.Match[str] | None = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group(0))
                except json.JSONDecodeError:
                    payload = {}

    decision: str = str(payload.get("decision") or "").strip().lower()
    if decision not in {"keep", "retry", "abstain"}:
        lowered: str = text.lower()
        if "abstain" in lowered:
            decision = "abstain"
        elif "retry" in lowered:
            decision = "retry"
        elif "keep" in lowered:
            decision = "keep"
        else:
            # Ambiguous LLM response — default to abstain rather than keep to
            # avoid surfacing unsupported answers.
            decision = "abstain"

    def _boolish(value: object, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered: str = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
        return default

    answers_question: bool = _boolish(
        payload.get("answers_question"),
        decision == "keep",
    )
    supported_by_context: bool = _boolish(
        payload.get("supported_by_context"),
        decision == "keep",
    )
    reason: str = str(payload.get("reason") or text or "").strip()

    return {
        "decision": decision,
        "answers_question": answers_question,
        "supported_by_context": supported_by_context,
        "reason": reason[:500],
        "raw": text,
    }


def evaluate_runtime_answer_guardrail(
    *,
    question: str,
    answer: str,
    chunks: Sequence[Mapping[str, object]],
    llm: Runnable[PromptValue, object],
) -> GuardrailVerdict:
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a strict runtime answer verifier for retrieval-augmented question answering.\n"
                    "Given a question, a candidate answer, and retrieved evidence chunks, decide whether the answer should be kept,\n"
                    "retried in a safer retrieval mode, or replaced with an abstention.\n\n"
                    "Decision rules:\n"
                    "- keep: the answer directly addresses the question and is supported by the evidence.\n"
                    "- retry: the answer is off-target, incomplete, or mismatched, but the evidence appears to contain useful information.\n"
                    "- abstain: the evidence is insufficient or the answer is unsupported.\n\n"
                    "Return ONLY valid JSON with this schema:\n"
                    '{{"decision":"keep|retry|abstain","answers_question":true|false,'
                    '"supported_by_context":true|false,"reason":"short reason"}}'
                ),
            ),
            (
                "human",
                (
                    "Question: {question}\n\n"
                    "Candidate answer: {answer}\n\n"
                    "Evidence:\n{evidence}"
                ),
            ),
        ],
    )

    evidence: str = render_guardrail_context(chunks)
    try:
        chain: Runnable[dict[str, str], str] = prompt | llm | StrOutputParser()
        raw: str = chain.invoke(
            {
                "question": question,
                "answer": answer,
                "evidence": evidence or "[no retrieved evidence]",
            },
        )
        return parse_guardrail_verdict(raw)
    except Exception as exc:
        logging.warning("Runtime answer guardrail failed: %s", exc)
        return {
            "decision": "retry",
            "answers_question": False,
            "supported_by_context": False,
            "reason": f"guardrail_error:{exc}",
            "raw": "",
        }
