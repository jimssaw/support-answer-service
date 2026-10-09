"""CHECK_SUPPORT stage: decide whether retrieved evidence is strong enough to answer.

Rule (see README): a retrieved chunk *supports* the question when
  1. it covers >= MIN_COVERAGE of the question's content terms, weighted by IDF
     (terms unknown to the KB count with maximum weight), and
  2. every number in the question (e.g. "2 hours") appears in that chunk, and
  3. no question term is "owned" by a different document: a term that occurs in
     exactly one KB document must belong to the evidence document (otherwise the
     question mixes topics, e.g. "change my *email* address" vs. "Address changes").
The system answers only if at least one retrieved chunk supports the question.
"""

from dataclasses import dataclass

from .retriever import BM25Retriever, Hit

MIN_COVERAGE = 0.65


@dataclass
class Evidence:
    hit: Hit
    coverage: float
    numbers_ok: bool
    topic_ok: bool

    @property
    def supported(self) -> bool:
        return self.coverage >= MIN_COVERAGE and self.numbers_ok and self.topic_ok


@dataclass
class SupportResult:
    supported: bool
    score: float
    evidence: list[Evidence]  # supporting evidence, best first, one per document


def check_support(query_terms: list[str], hits: list[Hit], retriever: BM25Retriever,
                  max_docs: int = 2) -> SupportResult:
    terms = sorted(set(query_terms))
    if not terms or not hits:
        return SupportResult(False, 0.0, [])

    total = sum(retriever.term_weight(t) for t in terms)
    numbers = {t for t in terms if t.isdigit()}
    scored = []
    for hit in hits:
        present = set(hit.chunk.tokens)
        covered = sum(retriever.term_weight(t) for t in terms if t in present)
        doc_terms = retriever.doc_terms[hit.chunk.doc.id]
        foreign = [t for t in terms if retriever.doc_df.get(t) == 1 and t not in doc_terms]
        scored.append(Evidence(hit, covered / total, numbers <= present, not foreign))

    scored.sort(key=lambda e: (e.supported, e.coverage, e.hit.score), reverse=True)
    # Reported score: best coverage among chunks that pass the hard rules (2, 3),
    # so that support_score >= MIN_COVERAGE exactly when the system answers.
    score = max((e.coverage for e in scored if e.numbers_ok and e.topic_ok), default=0.0)

    supporting, seen = [], set()
    for ev in scored:
        if ev.supported and ev.hit.chunk.doc.id not in seen:
            seen.add(ev.hit.chunk.doc.id)
            supporting.append(ev)
    supporting = supporting[:max_docs]
    return SupportResult(bool(supporting), round(score, 3), supporting)
