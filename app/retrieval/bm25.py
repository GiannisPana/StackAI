"""BM25 (Best Matching 25) ranking implementation.

This module provides a pure-Python implementation of the BM25 algorithm, which
is used for ranking document relevance based on term frequency and inverse
document frequency (TF-IDF) weighted by document length.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

# List of common English stopwords to exclude during tokenization
STOPWORDS = frozenset(
    """
a an and are as at be but by for from has have he her hers him his how i if in
into is it its me my not of on or our ours she so some such than that the their
them then there these they this those to too us was we were what when where
which who whom why will with you your yours
""".split()
)

# Regular expression to identify word-like tokens
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercased alphanumeric words, excluding stopwords.

    Args:
        text: The raw input string to tokenize.

    Returns:
        A list of filtered tokens.
    """
    tokens: list[str] = []
    for match in WORD_RE.finditer(text.lower()):
        word = match.group(0)
        if word not in STOPWORDS:
            tokens.append(word)
    return tokens


class BM25Index:
    """An index for efficient BM25 relevance scoring and retrieval.

    BM25 uses term frequency and document length to rank documents against a query.
    This implementation stores document lengths and inverted postings.

    Attributes:
        k1: Free parameter for term frequency saturation.
        b: Free parameter for document length normalization.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize the BM25 index with tuning parameters.

        Args:
            k1: Controls term frequency scaling (typical range [1.2, 2.0]).
            b: Controls degree of length normalization (typical range [0.5, 1.0]).
        """
        self.k1 = k1
        self.b = b
        self._doc_len: dict[int, int] = {}
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        self._avgdl = 0.0
        self._num_docs = 0
        self._finalized = False

    def add(self, row_id: int, text: str) -> None:
        """Add a document to the index.

        Args:
            row_id: Unique identifier for the document.
            text: Raw content of the document.
        """
        if row_id in self._doc_len:
            raise ValueError(f"BM25 row_id {row_id} already exists")
        tokens = tokenize(text)
        self._doc_len[row_id] = len(tokens)
        counts: dict[str, int] = defaultdict(int)
        for token in tokens:
            counts[token] += 1
        for term, tf in counts.items():
            # Inverted index: term -> {row_id: frequency}
            self._postings[term][row_id] = tf
        self._finalized = False

    def finalize(self) -> None:
        """Finalize the index statistics for retrieval.

        Computes average document length (avgdl) and total document count.
        """
        self._num_docs = len(self._doc_len)
        total_len = sum(self._doc_len.values())
        self._avgdl = (total_len / self._num_docs) if self._num_docs > 0 else 0.0
        self._finalized = True

    def _idf(self, term: str) -> float:
        """Calculate Inverse Document Frequency (IDF) for a term.

        Uses a BM25-style IDF variant and floors the result slightly above zero
        so ubiquitous terms still retain a minimal weight.

        Args:
            term: The token to calculate IDF for.

        Returns:
            The IDF score for the term.
        """
        df = len(self._postings.get(term, {}))
        if df == 0:
            return 0.0
        # Floor near-zero weights so terms present in every document still
        # contribute a tiny amount during ranking.
        return max(0.01, math.log(1 + (self._num_docs - df + 0.5) / (df + 0.5)))

    def top_k(
        self,
        query_tokens: list[str],
        k: int,
        mask: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Retrieve the top-K documents for a set of query tokens.

        Calculates scores using the standard BM25 formula:
        score = sum(IDF * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl)))

        Args:
            query_tokens: Pre-tokenized query words.
            k: Maximum number of results to return.
            mask: Optional set of allowed row IDs to consider.

        Returns:
            A list of (row_id, score) tuples, sorted by score descending.
        """
        if not query_tokens or not self._finalized or self._num_docs == 0:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in query_tokens:
            if term not in self._postings:
                continue
            idf = self._idf(term)
            for row_id, tf in self._postings[term].items():
                if mask is not None and row_id not in mask:
                    continue
                dl = self._doc_len[row_id]
                # Length normalization component: (1 - b + b * (dl / avgdl))
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
                # Final BM25 score accumulation for this term
                scores[row_id] += idf * (tf * (self.k1 + 1)) / denom

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda item: -item[1])
        return ranked[:k]

    def to_dict(self) -> dict:
        """Serialize the index to a dictionary.

        Returns:
            A dictionary containing all index state.
        """
        return {
            "k1": self.k1,
            "b": self.b,
            "num_docs": self._num_docs,
            "avgdl": self._avgdl,
            "doc_len": self._doc_len,
            "postings": {term: dict(posts) for term, posts in self._postings.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> BM25Index:
        """Deserialize an index from a dictionary.

        Args:
            data: Dictionary containing serialized index state.

        Returns:
            A restored BM25Index instance.
        """
        index = cls(k1=float(data["k1"]), b=float(data["b"]))
        index._doc_len = {int(key): int(value) for key, value in data["doc_len"].items()}
        index._postings = defaultdict(
            dict,
            {
                term: {int(row): int(tf) for row, tf in posts.items()}
                for term, posts in data["postings"].items()
            },
        )
        index._num_docs = int(data["num_docs"])
        index._avgdl = float(data["avgdl"])
        index._finalized = True
        return index
