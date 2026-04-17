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
