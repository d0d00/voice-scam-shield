from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    from openai import OpenAI  # type: ignore
except Exception:  # optional dependency
    OpenAI = None  # type: ignore


INTENT_KEYWORDS = {
    "CREDENTIAL_REQUEST": [
        # EN
        "password", "login", "account details", "verify your identity",
        # ES
        "contraseña", "iniciar sesión", "verificar su identidad",
        # FR
        "mot de passe", "identifiant", "vérifier votre identité",
    ],
    "OTP_REQUEST": [
        # EN
        "one-time code", "verification code", "OTP",
        # ES
        "código", "codigo", "verificación",
        # FR
        "code", "vérification",
    ],
    "PAYMENT": [
        # EN
        "gift card", "wire transfer", "bitcoin", "credit card", "payment",
        # ES
        "tarjeta regalo", "transferencia", "bitcoin", "pago",
        # FR
        "carte cadeau", "virement", "bitcoin", "paiement",
    ],
    "LINK": [
        # EN
        "click the link", "follow this link", "open this link",
        # ES
        "haga clic en el enlace", "siga este enlace",
        # FR
        "cliquez sur le lien", "suivez ce lien",
    ],
}


@dataclass
class IntentResult:
    score: float
    tags: List[str]
    rationale: str


def _llm_refine_intent(text: str, api_key: Optional[str]) -> Optional[IntentResult]:
    if not api_key or not text or OpenAI is None:
        return None
    try:
        client = OpenAI(api_key=api_key)
        prompt = (
            "Classify potential scam intent from the following transcript in EN/ES/FR.\n"
            "Return JSON with fields: score (0..1), tags (array from {CREDENTIAL_REQUEST,OTP_REQUEST,PAYMENT,LINK}), rationale (short).\n"
            f"Transcript: {text}\n"
        )
        # Using Responses API to keep payload small and deterministic
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
        content = resp.output_text  # type: ignore[attr-defined]
        import json  # local import to avoid top-level dep

        data = json.loads(content)
        score = float(max(0.0, min(1.0, data.get("score", 0.0))))
        tags = [t for t in data.get("tags", []) if isinstance(t, str)]
        rationale = str(data.get("rationale", ""))
        return IntentResult(score=score, tags=tags, rationale=rationale or "llm")
    except Exception:
        return None


def score_intent(transcript_fragment: str, api_key: Optional[str] = None) -> IntentResult:
    if not transcript_fragment:
        return IntentResult(0.0, [], "no speech")

    text = transcript_fragment.lower()
    hits: List[str] = []
    score = 0.0

    for tag, keys in INTENT_KEYWORDS.items():
        matched = any(k in text for k in keys)
        if matched:
            hits.append(tag)
            # Heuristic per-tag contribution
            if tag in ("CREDENTIAL_REQUEST", "OTP_REQUEST"):
                score += 0.5
            elif tag in ("PAYMENT", "LINK"):
                score += 0.3

    score = max(0.0, min(1.0, score))
    rationale = ", ".join(hits) if hits else "no risky keywords"

    # Optionally refine with LLM if available
    refined = _llm_refine_intent(transcript_fragment, api_key)
    if refined is not None:
        # Merge heuristics with LLM refinement: max score and union tags
        merged_score = max(score, refined.score)
        merged_tags = sorted(list({*hits, *refined.tags}))
        merged_rationale = refined.rationale or rationale
        return IntentResult(score=merged_score, tags=merged_tags, rationale=merged_rationale)

    return IntentResult(score=score, tags=hits, rationale=rationale)


