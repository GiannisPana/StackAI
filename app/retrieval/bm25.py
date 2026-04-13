from __future__ import annotations

import math
import re
from collections import defaultdict

STOPWORDS = frozenset(
    """
a an and are as at be but by for from has have he her hers him his how i if in
into is it its me my not of on or our ours she so some such than that the their
them then there these they this those to too us was we were what when where
which who whom why will with you your yours
""".split()
)

WORD_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in WORD_RE.finditer(text.lower()):
        word = match.group(0)
        if word not in STOPWORDS:
            tokens.append(word)
    return tokens


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._doc_len: dict[int, int] = {}
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        self._avgdl = 0.0
        self._num_docs = 0
        self._finalized = False

    def add(self, row_id: int, text: str) -> None:
        tokens = tokenize(text)
        self._doc_len[row_id] = len(tokens)
        counts: dict[str, int] = defaultdict(int)
        for token in tokens:
            counts[token] += 1
        for term, tf in counts.items():
            self._postings[term][row_id] = tf
        self._finalized = False

    def finalize(self) -> None:
        self._num_docs = len(self._doc_len)
        total = sum(self._doc_len.values())
        self._avgdl = total / self._num_docs if self._num_docs else 0.0
        self._finalized = True

    def _idf(self, term: str) -> float:
        df = len(self._postings.get(term, {}))
        if df == 0:
            return 0.0
        return math.log(1 + (self._num_docs - df + 0.5) / (df + 0.5))

    def top_k(
        self,
        query_tokens: list[str],
        k: int,
        mask: set[int] | None = None,
    ) -> list[tuple[int, float]]:
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
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
                scores[row_id] += idf * (tf * (self.k1 + 1)) / denom

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda item: -item[1])
        return ranked[:k]

    def to_dict(self) -> dict:
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
