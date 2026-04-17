from __future__ import annotations

from app.generation.policy import (
    PolicyResult,
    apply_policy,
    detect_personalized_advice,
    detect_topic,
    mask_pii,
)
from tests.fakes.mistral import FakeMistralClient


def test_email_masked():
    masked, entities = mask_pii("Contact me at jane.doe@example.com")
    assert masked == "Contact me at [EMAIL]"
    assert entities == ["email"]


def test_ssn_masked():
    masked, entities = mask_pii("Employee SSN 123-45-6789 is on file.")
    assert masked == "Employee SSN [SSN] is on file."
    assert entities == ["ssn"]


def test_phone_masked():
    masked, entities = mask_pii("Call me at (415) 555-2671 tomorrow.")
    assert masked == "Call me at [PHONE] tomorrow."
    assert entities == ["phone"]


def test_iban_masked():
    masked, entities = mask_pii("Wire funds to GB82 WEST 1234 5698 7654 32.")
    assert masked == "Wire funds to [IBAN]."
    assert entities == ["iban"]


def test_credit_card_luhn_valid_masked():
    masked, entities = mask_pii("Card number 4111 1111 1111 1111 was declined.")
    assert masked == "Card number [CREDIT_CARD] was declined."
    assert entities == ["credit_card"]


def test_credit_card_luhn_invalid_not_masked():
    masked, entities = mask_pii("Card number 4111 1111 1111 1112 was declined.")
    assert masked == "Card number 4111 1111 1111 1112 was declined."
    assert entities == []


def test_legal_keyword_sets_disclaimer():
    assert detect_topic("Can you explain this lawsuit settlement clause?") == "legal"


def test_medical_keyword_sets_disclaimer():
    assert detect_topic("What dosage does the prescription mention?") == "medical"


def test_personalized_advice_triggers_refuse():
    assert detect_personalized_advice("Should I sue my employer?", "legal") is True


def test_mask_is_idempotent():
    once, entities = mask_pii("Email [EMAIL] and SSN [SSN] already masked.")
    twice, entities_again = mask_pii(once)
    assert once == twice
    assert entities == []
    assert entities_again == []


def test_llm_fallback_called_when_ambiguous():
    fake = FakeMistralClient(dim=8)
    fake.register_chat(r"policy", {"action": "REFUSE"})

    result = apply_policy(fake, "My attorney situation is complicated and I need guidance.")

    assert fake.chat_call_count == 1
    assert result == PolicyResult(
        masked_text="My attorney situation is complicated and I need guidance.",
        entities=[],
        disclaimer="legal",
        refuse_reason="personalized_advice",
    )
