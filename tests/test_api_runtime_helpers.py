import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ontographrag.api.runtime_helpers import (
    ReadinessReport,
    SupportGuardrailVerdict,
    build_readiness_report,
    build_support_guardrail_verdict,
    configured_provider_names_from_env,
    filesystem_write_probe_ok,
    guardrail_forces_abstention,
    parse_request_timeout_seconds,
)


def test_parse_request_timeout_seconds_clamps_values() -> None:
    assert parse_request_timeout_seconds("not-a-number") == 120
    assert parse_request_timeout_seconds("5") == 15
    assert parse_request_timeout_seconds("900") == 300
    assert parse_request_timeout_seconds("60") == 60


def test_configured_provider_names_from_env_detects_keys() -> None:
    providers: list[str] = configured_provider_names_from_env(
        {
            "OPENAI_API_KEY": "x",
            "OPENROUTER_API_KEY": "y",
            "OLLAMA_HOST": "http://localhost:11434",
        }
    )
    assert providers == ["openai", "openrouter", "ollama"]


def test_guardrail_forces_abstention_only_on_non_keep_verdicts() -> None:
    assert (
        guardrail_forces_abstention({"enabled": True, "final_decision": "abstain"})
        is True
    )
    assert (
        guardrail_forces_abstention({"enabled": True, "final_decision": "retry_keep"})
        is False
    )
    assert (
        guardrail_forces_abstention({"enabled": False, "final_decision": "abstain"})
        is False
    )
    assert guardrail_forces_abstention(None) is False


def test_support_guardrail_abstains_when_two_signals_are_weak() -> None:
    verdict: SupportGuardrailVerdict = build_support_guardrail_verdict(
        enabled=True,
        mode="retry_then_abstain",
        structural_support=0.21,
        grounding_support=0.44,
        confidence=0.77,
    )

    assert verdict["final_decision"] == "abstain"
    assert verdict["would_abstain"] is True
    assert verdict["weak_signal_count"] == 2


def test_support_guardrail_keeps_when_only_one_signal_is_weak() -> None:
    verdict: SupportGuardrailVerdict = build_support_guardrail_verdict(
        enabled=True,
        mode="retry_then_abstain",
        structural_support=0.62,
        grounding_support=0.44,
        confidence=0.81,
    )

    assert verdict["final_decision"] == "keep"
    assert verdict["would_abstain"] is False


def test_support_guardrail_disabled_exposes_would_abstain_without_suppressing() -> None:
    verdict: SupportGuardrailVerdict = build_support_guardrail_verdict(
        enabled=False,
        mode="abstain_only",
        structural_support=0.12,
        grounding_support=0.31,
        confidence=0.18,
    )

    assert verdict["enabled"] is False
    assert verdict["would_abstain"] is True
    assert verdict["final_decision"] == "keep"


def test_filesystem_write_probe_ok(tmp_path: Path) -> None:
    ok: bool
    detail: str
    ok, detail = filesystem_write_probe_ok(tmp_path)
    assert ok is True
    assert "Can write to" in detail


def test_build_readiness_report_flags_missing_dependencies() -> None:
    report: ReadinessReport = build_readiness_report(
        neo4j_ready=True,
        rag_runtime_ready=False,
        write_probe_ok=True,
        write_probe_detail="ok",
        configured_providers=[],
    )
    assert report["ready"] is False
    assert report["status"] == "not_ready"


class FloatConvertible:
    def __float__(self) -> float:
        return 0.8


def test_timeout_conversion_fallback_and_numeric_clamps() -> None:
    assert parse_request_timeout_seconds(None) == 120
    assert parse_request_timeout_seconds(object()) == 120
    assert parse_request_timeout_seconds("-20") == 15
    assert parse_request_timeout_seconds("100000") == 300


def test_guardrail_preserves_numeric_conversion_and_unavailable_signals() -> None:
    verdict: SupportGuardrailVerdict = build_support_guardrail_verdict(
        enabled=True,
        mode="abstain_only",
        structural_support="0.8",
        grounding_support=FloatConvertible(),
        confidence=0.8,
    )
    assert verdict["final_decision"] == "keep"
    assert verdict["weak_signal_count"] == 0
    for unavailable in (None, object(), "not-a-number", float("nan")):
        verdict = build_support_guardrail_verdict(
            enabled=True,
            mode="abstain_only",
            structural_support=unavailable,
            grounding_support=unavailable,
            confidence=unavailable,
        )
        assert verdict["weak_signal_count"] == 0
        assert verdict["final_decision"] == "keep"
