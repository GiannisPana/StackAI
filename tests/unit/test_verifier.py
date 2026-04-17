from __future__ import annotations

from app.api.schemas import Verification
from app.generation.verifier import parse_citation_tags, split_answer_sentences, verify_answer
from tests.fakes.mistral import FakeMistralClient


def test_sentence_splitter_handles_abbreviations():
    text = "Dr. Smith reviewed the handbook [1]. Benefits begin after six months [2]."
    assert split_answer_sentences(text) == [
        "Dr. Smith reviewed the handbook [1].",
        "Benefits begin after six months [2].",
    ]


def test_citation_tag_parser():
    assert parse_citation_tags("First [1]. Second [2][3]. Third [4, 5].") == [[1], [2, 3], [4, 5]]


def test_verifier_flags_uncited_factual_sentence():
    fake = FakeMistralClient(dim=8)
    fake.register_chat(r"entailment", {"0": True})

    out = verify_answer(
        fake,
        answer="Employees receive sixteen weeks [1]. Eligibility rules vary by team.",
        chunk_lookup={1: "Employees receive sixteen weeks of parental leave."},
    )

    assert out == Verification(all_supported=False, unsupported_sentences=[1])


def test_verifier_batches_all_pairs_into_one_call():
    fake = FakeMistralClient(dim=8)
    fake.register_chat(r"entailment", {"0": True, "1": False, "2": True})

    out = verify_answer(
        fake,
        answer="A supported sentence [1][2]. Another supported sentence [2].",
        chunk_lookup={
            1: "Sentence one support.",
            2: "Sentence one and two support.",
        },
    )

    assert fake.chat_call_count == 1
    assert out.all_supported is True
    assert out.unsupported_sentences == []


def test_verifier_ignores_leading_disclaimer():
    fake = FakeMistralClient(dim=8)
    fake.register_chat(r"entailment", {"0": True})

    answer = (
        "Note: this is general information, not legal advice. "
        "Consult a qualified attorney for advice specific to your situation.\n\n"
        "The policy grants sixteen weeks [1]."
    )
    out = verify_answer(
        fake,
        answer=answer,
        chunk_lookup={1: "The policy grants sixteen weeks of leave."},
    )

    assert out == Verification(all_supported=True, unsupported_sentences=[])
