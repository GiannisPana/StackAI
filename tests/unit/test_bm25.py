"""
Unit tests for the BM25 lexical search component.

These tests verify the behavior of the BM25 implementation, including tokenization,
IDF calculations (rare term scoring), length normalization, TF saturation,
and persistence (save/load).
"""

from __future__ import annotations

from app.retrieval.bm25 import BM25Index, tokenize
from app.storage.bm25_store import load_bm25, save_bm25


def test_tokenize_lowercase_strip_punct():
    """
    Verify that the tokenizer correctly lowercases text and removes punctuation.
    """
    # The tokenizer should normalize "Hello, World!" to ["hello", "world"]
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_stopwords_removed():
    """
    Verify that common stopwords are filtered out during tokenization.
    """
    tokens = tokenize("the quick brown fox")
    # "the" is a common stopword and should be removed
    assert "the" not in tokens
    # Other words should be preserved
    assert {"quick", "brown", "fox"} <= set(tokens)


def test_bm25_rare_term_scores_higher():
    """
    Verify that rarer terms contribute more to the BM25 score (Inverse Document Frequency).
    """
    index = BM25Index()
    # Document 0 has the rare term "rare"
    index.add(0, "common common common rare")
    # Documents 1 and 2 only have "common"
    index.add(1, "common common common common")
    index.add(2, "common common common common")
    index.finalize()

    # Searching for "rare" should return Document 0 as the top hit
    hits = index.top_k(["rare"], k=3)

    assert hits[0][0] == 0
    assert hits[0][1] > 0


def test_bm25_length_normalization():
    """
    Verify that BM25 applies length normalization.
    Shorter documents should score higher than longer ones for the same term frequency.
    """
    index = BM25Index()
    # "cat" appears once in a very short document
    index.add(0, "cat")
    # "cat" appears once in a longer document
    index.add(1, "cat dog bird fish snake lizard")
    index.finalize()

    hits = dict(index.top_k(["cat"], k=2))
    # Document 0 should have a higher score than Document 1
    assert hits[0] > hits[1]


def test_bm25_tf_saturation():
    """
    Verify that term frequency contribution saturates according to the k1 parameter.
    """
    index = BM25Index(k1=1.2)
    # Document 0 has one "cat"
    index.add(0, "cat")
    # Document 1 has ten "cat"s
    index.add(1, "cat cat cat cat cat cat cat cat cat cat")
    index.finalize()

    hits = dict(index.top_k(["cat"], k=2))
    # Even with 10x the frequency, the score should not be 10x higher due to saturation
    assert hits[1] / hits[0] < 3.0


def test_bm25_top_k_mask_filters():
    """
    Verify that the search can be restricted to a specific subset of documents using a mask.
    """
    index = BM25Index()
    index.add(0, "alpha")
    index.add(1, "alpha")
    index.add(2, "alpha")
    index.finalize()

    # Only documents in the mask {1, 2} should be returned
    hits = index.top_k(["alpha"], k=5, mask={1, 2})

    assert {row for row, _ in hits} == {1, 2}


def test_bm25_empty_query_returns_empty():
    """
    Verify that an empty query returns an empty list of results.
    """
    index = BM25Index()
    index.add(0, "alpha")
    index.finalize()

    assert index.top_k([], k=5) == []


def test_bm25_persist_roundtrip(tmp_path):
    """
    Verify that the BM25 index can be saved to and loaded from disk while preserving its state.
    """
    index = BM25Index()
    index.add(5, "alpha beta")
    index.add(7, "beta gamma")
    index.finalize()
    path = tmp_path / "bm25.json"

    # Save the index
    save_bm25(path, index)
    # Load it back
    loaded = load_bm25(path)

    # The scores should be identical
    assert index.top_k(["beta"], k=5) == loaded.top_k(["beta"], k=5)


def test_bm25_persist_roundtrip_preserves_tokenizer_version(tmp_path):
    index = BM25Index()
    index.add(1, "alpha beta")
    index.finalize()
    path = tmp_path / "bm25.json"
    save_bm25(path, index)
    loaded = load_bm25(path)
    from app.retrieval.bm25 import TOKENIZER_VERSION
    assert loaded.tokenizer_version == TOKENIZER_VERSION


def test_bm25_from_dict_defaults_v1_when_version_missing():
    index = BM25Index()
    index.add(1, "hello world")
    index.finalize()
    data = index.to_dict()
    del data["tokenizer_version"]
    restored = BM25Index.from_dict(data)
    assert restored.tokenizer_version == "v1"


# --- Tokenizer regression tests for I4 ---

def test_tokenize_currency_amount():
    assert tokenize("$25,000") == ["25,000"]


def test_tokenize_version_number():
    assert tokenize("v3.14") == ["v3.14"]


def test_tokenize_contraction_preserved():
    assert tokenize("don't") == ["don't"]


def test_tokenize_hyphen_splits_words():
    # Hyphens intentionally excluded from internal-separator set.
    assert tokenize("ID-00123") == ["id", "00123"]


def test_tokenize_hyphen_splits_phrase():
    # Hyphens split; stopwords (of, the) are then filtered.
    assert tokenize("state-of-the-art") == ["state", "art"]


# --- Retrieval-level regression for the motivating contract bug ---

def test_bm25_retrieves_dollar_threshold_chunks_over_generic_flow_down():
    """Regression for the contract-threshold miss.

    Without I4, ``$25,000`` tokenises to ``['25', '000']`` and a query for
    ``$25,000`` loses the exact-match signal. Chunks mentioning the specific
    threshold must outrank a generic flow-down clause that shares only the
    ``subconsultant`` keyword.
    """
    index = BM25Index()
    # Chunk 0: generic flow-down boilerplate (page 9 in the real contract).
    index.add(0, "Consultant shall flow down all applicable terms to each subconsultant.")
    # Chunk 1: $25,000 threshold (page 7/9 in the real contract).
    index.add(1, "Any subconsultant agreement exceeding $25,000 requires written approval.")
    # Chunk 2: $50,000 threshold (page 5 in the real contract).
    index.add(2, "Subconsultants with contracts above $50,000 require client consent.")
    index.finalize()

    hits = dict(index.top_k(tokenize("What is the $25,000 subconsultant threshold?"), k=3))

    assert 1 in hits, "chunk containing $25,000 must be retrieved"
    assert hits[1] > hits[0], (
        "chunk with the exact $25,000 threshold must outrank the generic "
        "flow-down clause that only shares the 'subconsultant' token"
    )
