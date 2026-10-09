"""RETRIEVE stage: BM25 over sentence-window chunks."""

import math
from collections import Counter
from dataclasses import dataclass

from .kb import Chunk, Document, chunk_documents
from .text import tokenize


@dataclass
class Hit:
    chunk: Chunk
    score: float


class BM25Retriever:
    def __init__(self, docs: list[Document], k1: float = 1.2, b: float = 0.75):
        self.docs = docs
        self.chunks = chunk_documents(docs)
        self.k1, self.b = k1, b
        self.tf = [Counter(c.tokens) for c in self.chunks]
        self.avgdl = sum(len(c.tokens) for c in self.chunks) / max(1, len(self.chunks))
        chunk_df = Counter(t for tf in self.tf for t in tf)
        self.chunk_idf = {t: self._idf(df, len(self.chunks)) for t, df in chunk_df.items()}
        # Document-level IDF, used by the support check to weight question terms.
        self.doc_terms = {d.id: set(tokenize(d.title) + tokenize(d.text)) for d in docs}
        self.doc_df = Counter(t for terms in self.doc_terms.values() for t in terms)
        self.doc_idf = {t: self._idf(df, len(docs)) for t, df in self.doc_df.items()}
        self.max_term_weight = self._idf(1, len(docs))

    @staticmethod
    def _idf(df: int, n: int) -> float:
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def term_weight(self, term: str) -> float:
        # Terms the KB never mentions get the weight of the rarest known term:
        # they are strong evidence the question asks about something not covered.
        return self.doc_idf.get(term, self.max_term_weight)

    def search(self, query_terms: list[str], top_k: int = 5) -> list[Hit]:
        hits = []
        for chunk, tf in zip(self.chunks, self.tf):
            dl = len(chunk.tokens)
            score = 0.0
            for t in set(query_terms):
                f = tf.get(t, 0)
                if f:
                    score += self.chunk_idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if score > 0:
                hits.append(Hit(chunk, score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
