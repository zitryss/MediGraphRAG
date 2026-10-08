"""Small production-facing helpers for API runtime behavior."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NotRequired

from typing_extensions import TypedDict


class WeakSupportSignal(TypedDict):
    name: str
    value: float
    threshold: float


class SupportGuardrailVerdict(TypedDict):
    enabled: bool
    mode: str
    strategy: str
    final_decision: str
    would_abstain: bool
    retried: bool
    available_signal_count: int
    required_weak_signals: int
    weak_signal_count: int
    weak_signals: list[WeakSupportSignal]
    metric_values: dict[str, float | None]
    thresholds: dict[str, float]
    reason: str
    supported_by_context: bool


class ReadinessCheck(TypedDict):
    name: str
    ok: bool
    detail: NotRequired[str | list[str]]


class ReadinessReport(TypedDict):
    status: str
    ready: bool
    checks: list[ReadinessCheck]


def parse_request_timeout_seconds(
    raw_value: Any,
    *,
    default: int = 120,
    minimum: int = 15,
    maximum: int = 300,
) -> int:
    """Parse and clamp a request timeout from config."""
    try:
        value: int = int(raw_value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def configured_provider_names_from_env(env: Mapping[str, str]) -> list[str]:
    """Return model providers that appear configured from environment variables."""
    providers: list[str] = []
    if env.get("OPENAI_API_KEY"):
        providers.append("openai")
    if env.get("OPENROUTER_API_KEY"):
        providers.append("openrouter")
    if env.get("GEMINI_API_KEY"):
        providers.append("gemini")
    if env.get("DEEPSEEK_API_KEY"):
        providers.append("deepseek")
    if env.get("OLLAMA_HOST") or env.get("OLLAMA_BASE_URL"):
        providers.append("ollama")
    return providers


def guardrail_forces_abstention(guardrail: Mapping[str, object] | None) -> bool:
    """True when a runtime guardrail verdict means the response should be treated as abstained."""
    if not guardrail:
        return False
    final_decision: str = str(guardrail.get("final_decision", "")).strip().lower()
    return bool(guardrail.get("enabled")) and final_decision not in {
        "keep",
        "retry_keep",
    }


def build_support_guardrail_verdict(
    *,
    enabled: bool,
    mode: str,
    structural_support: Any,
    grounding_support: Any,
    confidence: Any,
    structural_threshold: float = 0.40,
    grounding_threshold: float = 0.50,
    confidence_threshold: float = 0.40,
    min_weak_signals: int = 2,
) -> SupportGuardrailVerdict:
    """Build a deterministic support-based guardrail decision from already-computed metrics."""

    def _metric_value(value: Any) -> float | None:
        try:
            numeric: float = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric):
            return None
        return max(0.0, min(1.0, numeric))

    metric_values: dict[str, float | None] = {
        "structural_support": _metric_value(structural_support),
        "grounding_support": _metric_value(grounding_support),
        "confidence": _metric_value(confidence),
    }
    thresholds: dict[str, float] = {
        "structural_support": float(structural_threshold),
        "grounding_support": float(grounding_threshold),
        "confidence": float(confidence_threshold),
    }

    weak_signals: list[WeakSupportSignal] = [
        {
            "name": name,
            "value": value,
            "threshold": thresholds[name],
        }
        for name, value in metric_values.items()
        if value is not None and value < thresholds[name]
    ]
    available_signal_count: int = sum(
        value is not None for value in metric_values.values()
    )
    weak_signal_count: int = len(weak_signals)
    required_weak_signals: int = max(1, int(min_weak_signals))
    would_abstain: bool = (
        available_signal_count >= required_weak_signals
        and weak_signal_count >= required_weak_signals
    )

    reason: str
    if would_abstain:
        reason = "Weak support signals: " + ", ".join(
            f"{signal['name']}={signal['value']:.3f}<{signal['threshold']:.3f}"
            for signal in weak_signals
        )
    else:
        reason = "Support signals remain above abstention thresholds."

    return {
        "enabled": bool(enabled),
        "mode": str(mode or "").strip().lower() or "abstain_only",
        "strategy": "support_threshold_gate",
        "final_decision": "abstain" if enabled and would_abstain else "keep",
        "would_abstain": would_abstain,
        "retried": False,
        "available_signal_count": available_signal_count,
        "required_weak_signals": required_weak_signals,
        "weak_signal_count": weak_signal_count,
        "weak_signals": weak_signals,
        "metric_values": metric_values,
        "thresholds": thresholds,
        "reason": reason,
        "supported_by_context": not any(
            signal["name"] in {"structural_support", "grounding_support"}
            for signal in weak_signals
        ),
    }


def filesystem_write_probe_ok(probe_root: Path) -> tuple[bool, str]:
    """Check whether the process can create and delete a small file under probe_root."""
    try:
        probe_root.mkdir(parents=True, exist_ok=True)
        probe_file: Path = probe_root / ".ontographrag_write_probe"
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink(missing_ok=True)
        return True, f"Can write to {probe_root}"
    except Exception as e:  # pragma: no cover - platform-specific exception shapes
        return False, str(e)


def build_readiness_report(
    *,
    neo4j_ready: bool,
    rag_runtime_ready: bool,
    write_probe_ok: bool,
    write_probe_detail: str,
    configured_providers: Sequence[str],
) -> ReadinessReport:
    """Create a compact readiness payload for fast infrastructure checks."""
    checks: list[ReadinessCheck] = [
        {"name": "neo4j", "ok": bool(neo4j_ready)},
        {"name": "rag_runtime", "ok": bool(rag_runtime_ready)},
        {
            "name": "write_permissions",
            "ok": bool(write_probe_ok),
            "detail": write_probe_detail,
        },
        {
            "name": "model_provider",
            "ok": len(configured_providers) > 0,
            "detail": list(configured_providers),
        },
    ]
    ready: bool = all(check["ok"] for check in checks)
    return {
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "checks": checks,
    }
