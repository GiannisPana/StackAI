from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from app.mistral_client import MistralProtocol

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)"
)
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:[ -]?[A-Z0-9]){11,30}\b")
_CREDIT_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")

_LEGAL_KEYWORDS = {
    "attorney",
    "clause",
    "contract",
    "court",
    "lawsuit",
    "legal",
    "liability",
    "settlement",
    "sue",
    "trial",
}
_MEDICAL_KEYWORDS = {
    "diagnosis",
    "doctor",
    "dosage",
    "medical",
    "medication",
    "prescription",
    "symptom",
    "treatment",
}
_FIRST_PERSON_RE = re.compile(r"\b(i|me|my|mine|we|us|our|ours)\b", re.IGNORECASE)
_PERSONALIZED_RE = re.compile(
    r"^\s*(should|can|could|would|do|am)\s+(?:i|we)\b",
    re.IGNORECASE,
)
_AMBIGUOUS_CUES = (
    "advice",
    "allowed",
    "case",
    "condition",
    "guidance",
    "help",
    "need",
    "situation",
)

PolicyAction = Literal["SAFE", "DISCLAIMER", "REFUSE"]

_FALLBACK_SYSTEM = (
    "You are a policy classifier for document QA. "
    "Given a masked user query about a legal or medical topic, return STRICT JSON "
    'with one key: {"action": "SAFE|DISCLAIMER|REFUSE"}. '
    "Choose REFUSE only for personalized legal or medical advice requests. "
    "Choose DISCLAIMER for general legal or medical questions. "
    "Choose SAFE only when no topic disclaimer or refusal is needed."
)


@dataclass(frozen=True)
class PolicyResult:
    masked_text: str
    entities: list[str]
    disclaimer: Literal["legal", "medical"] | None
    refuse_reason: str | None


def mask_pii(text: str) -> tuple[str, list[str]]:
    masked = text
    entities: list[str] = []

    def replace(pattern: re.Pattern[str], replacement: str, entity: str) -> None:
        nonlocal masked
        masked, count = pattern.subn(replacement, masked)
        if count and entity not in entities:
            entities.append(entity)

    replace(_EMAIL_RE, "[EMAIL]", "email")
    replace(_SSN_RE, "[SSN]", "ssn")
    replace(_PHONE_RE, "[PHONE]", "phone")
    masked = _mask_iban(masked, entities)
    masked = _mask_credit_cards(masked, entities)
    return masked, entities


def detect_topic(text: str) -> Literal["legal", "medical"] | None:
    lowered = text.lower()
    if any(keyword in lowered for keyword in _MEDICAL_KEYWORDS):
        return "medical"
    if any(keyword in lowered for keyword in _LEGAL_KEYWORDS):
        return "legal"
    return None


def detect_personalized_advice(
    text: str,
    topic: Literal["legal", "medical"] | None,
) -> bool:
    if topic is None:
        return False
    return bool(_PERSONALIZED_RE.search(text))


def apply_policy(client: MistralProtocol, text: str) -> PolicyResult:
    masked_text, entities = mask_pii(text)
    disclaimer = detect_topic(masked_text)
    refuse_reason: str | None = None

    if disclaimer is not None:
        if detect_personalized_advice(masked_text, disclaimer):
            refuse_reason = "personalized_advice"
        elif _should_use_fallback(masked_text):
            action = _fallback_action(client, masked_text, disclaimer)
            if action == "REFUSE":
                refuse_reason = "personalized_advice"

    return PolicyResult(
        masked_text=masked_text,
        entities=entities,
        disclaimer=disclaimer,
        refuse_reason=refuse_reason,
    )


def _should_use_fallback(text: str) -> bool:
    lowered = text.lower()
    return bool(_FIRST_PERSON_RE.search(text)) and any(cue in lowered for cue in _AMBIGUOUS_CUES)


def _fallback_action(
    client: MistralProtocol,
    text: str,
    topic: Literal["legal", "medical"],
) -> PolicyAction:
    messages = [
        {"role": "system", "content": _FALLBACK_SYSTEM},
        {
            "role": "user",
            "content": f"Policy review for topic={topic}: {text}",
        },
    ]
    try:
        response = client.chat(messages, response_format={"type": "json_object"})
    except Exception as exc:
        logger.warning("policy LLM fallback failed: %s", exc)
        return "DISCLAIMER"

    if not isinstance(response, dict):
        return "DISCLAIMER"
    action = str(response.get("action", "DISCLAIMER")).strip().upper()
    if action not in {"SAFE", "DISCLAIMER", "REFUSE"}:
        return "DISCLAIMER"
    return action  # type: ignore[return-value]


def _mask_iban(text: str, entities: list[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        candidate = match.group(0)
        normalized = re.sub(r"[\s-]", "", candidate).upper()
        if _valid_iban(normalized):
            if "iban" not in entities:
                entities.append("iban")
            return "[IBAN]"
        return candidate

    return _IBAN_RE.sub(replace, text)


def _mask_credit_cards(text: str, entities: list[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        candidate = match.group(0)
        digits = re.sub(r"\D", "", candidate)
        if 13 <= len(digits) <= 19 and _passes_luhn(digits):
            if "credit_card" not in entities:
                entities.append("credit_card")
            return "[CREDIT_CARD]"
        return candidate

    return _CREDIT_CARD_RE.sub(replace, text)


def _passes_luhn(digits: str) -> bool:
    # Luhn mod-10 checksum used by credit-card numbers.
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _valid_iban(value: str) -> bool:
    # ISO-7064 IBAN mod-97 check after alphanumeric remapping.
    if len(value) < 15 or len(value) > 34:
        return False
    rearranged = value[4:] + value[:4]
    digits = []
    for char in rearranged:
        if char.isdigit():
            digits.append(char)
        elif "A" <= char <= "Z":
            digits.append(str(ord(char) - 55))
        else:
            return False
    number = "".join(digits)
    remainder = 0
    for char in number:
        remainder = (remainder * 10 + int(char)) % 97
    return remainder == 1
