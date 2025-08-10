from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class FusionResult:
    risk: float
    label: str
    rationale: str
    tags: List[str]


def fuse_scores(spoof: float, intent: float, heuristics: float, tags: List[str]) -> FusionResult:
    # Heavier weight on intent for demo priorities
    risk = 0.3 * spoof + 0.6 * intent + 0.1 * heuristics
    if risk < 0.35:
        label = "SAFE"
    elif risk < 0.65:
        label = "SUSPICIOUS"
    else:
        label = "SCAM"
    rationale_parts: List[str] = []
    if intent > 0.0:
        rationale_parts.append(f"intent={intent:.2f}")
    if spoof > 0.0:
        rationale_parts.append(f"spoof={spoof:.2f}")
    if heuristics > 0.0:
        rationale_parts.append(f"heuristics={heuristics:.2f}")
    rationale = ", ".join(rationale_parts) if rationale_parts else "baseline"
    return FusionResult(risk=float(risk), label=label, rationale=rationale, tags=tags)


