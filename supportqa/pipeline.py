"""LOAD_KB -> RETRIEVE -> CHECK_SUPPORT -> ANSWER | ABSTAIN."""

from pathlib import Path

from .composers import RuleComposer, build_composer
from .kb import load_kb
from .retriever import BM25Retriever
from .support import check_support
from .text import tokenize

DEFAULT_KB = Path(__file__).resolve().parent.parent / "kb.json"

ABSTAIN_MESSAGE = (
    "I don't have enough information in the knowledge base to answer that reliably. "
    "Please contact human support for help with this request."
)


class SupportQA:
    def __init__(self, kb_path=DEFAULT_KB, mode: str = "rule", top_k: int = 5):
        self.kb_path = Path(kb_path)
        self.docs = load_kb(self.kb_path)                          # LOAD_KB
        self.retriever = BM25Retriever(self.docs)
        self.composer = build_composer(mode) if mode != "rule" else RuleComposer()
        self.top_k = top_k

    def ask(self, question: str) -> dict:
        question = (question or "").strip()
        terms = tokenize(question)
        hits = self.retriever.search(terms, self.top_k)              # RETRIEVE
        retrieved_ids = list(dict.fromkeys(h.chunk.doc.id for h in hits))
        support = check_support(terms, hits, self.retriever)         # CHECK_SUPPORT

        answer, citations = None, []
        if support.supported:                                        # ANSWER
            answer, citations = self.composer.compose(
                question, terms, support.evidence, self.retriever.term_weight)

        if not answer or not citations:                              # ABSTAIN
            return self._result(question, "abstain", ABSTAIN_MESSAGE, [], retrieved_ids, support.score)
        return self._result(question, "answer", answer, citations, retrieved_ids, support.score)

    @staticmethod
    def _result(question, decision, answer, citations, retrieved_ids, score):
        return {
            "question": question,
            "decision": decision,
            "answer": answer,
            "citations": citations,
            "debug": {"retrieved_ids": retrieved_ids, "support_score": score},
        }
