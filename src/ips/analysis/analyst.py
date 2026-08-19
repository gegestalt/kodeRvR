"""Grounded Markdown analyst for notebook evidence.

The fallback is deterministic and requires no API.  An optional external
callable may be supplied, but its numeric claims are rejected when they are not
present in the structured evidence supplied by the experiment.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import re
from typing import Any


def _numbers(value: Any) -> set[float]:
    if isinstance(value, Mapping):
        return {number for item in value.values() for number in _numbers(item)}
    if isinstance(value, (list, tuple)):
        return {number for item in value for number in _numbers(item)}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {float(value)}
    return set()


def _unsupported_numbers(text: str, evidence: Mapping[str, Any]) -> list[float]:
    allowed = _numbers(evidence) | {0.0, 1.0, 100.0}
    mentioned = [float(token) for token in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", text)]
    return [number for number in mentioned if not any(abs(number - item) <= max(1e-9, abs(item) * 5e-4) for item in allowed)]


class EvidenceAnalyst:
    """Explain structured evidence without acting as an IPS controller."""

    def __init__(self, llm_callable: Callable[[dict[str, Any]], str] | None = None) -> None:
        self.llm_callable = llm_callable

    def explain(self, evidence: Mapping[str, Any]) -> str:
        if not evidence:
            raise ValueError("analyst requires structured evidence")
        payload = json.loads(json.dumps(evidence, default=str))
        if self.llm_callable is not None:
            candidate = self.llm_callable(payload)
            if not isinstance(candidate, str) or not candidate.strip():
                raise ValueError("LLM analyst returned no Markdown")
            unsupported = _unsupported_numbers(candidate, payload)
            if unsupported:
                raise ValueError(f"LLM analyst introduced unsupported numbers: {unsupported}")
            return candidate
        return self._fallback(payload)

    @staticmethod
    def _fallback(evidence: dict[str, Any]) -> str:
        detector = evidence.get("detector", {})
        policy = evidence.get("policy", {})
        gates = evidence.get("gates", {})
        best_detector = detector.get("best_model", "not established")
        detector_score = detector.get("pr_auc")
        best_policy = policy.get("best_policy", "not established")
        evidence_grade = policy.get("evidence_grade", "not supplied")
        failed = [name for name, state in gates.items() if str(state).upper() in {"FAIL", "BLOCKED"}]
        score_text = f" with PR-AUC **{float(detector_score):.4f}**" if detector_score is not None else ""
        failed_text = ", ".join(failed) if failed else "none in the supplied gate set"
        return f"""### Security analyst interpretation

**What happened?** The strongest supplied detector is **{best_detector}**{score_text}. The current policy candidate is **{best_policy}**.

**What supports this?** This statement is generated only from the structured experiment artifact passed to the analyst. Policy evidence is graded **{evidence_grade}**.

**What failed or remains uncertain?** Failed or blocked gates: **{failed_text}**.

**Operational risk.** Detector observations and counterfactual policy consequences must not be presented as real intervention outcomes.

**Recommended next experiment.** Resolve the failed gates using validation-only choices, then collect observed shadow-mode/cyber-range intervention transitions before deployment claims.
"""
