"""ANSWER stage: swappable answer composers.

Both composers only ever see the evidence that passed the support check.
`RuleComposer` is extractive (answers are verbatim KB sentences) and needs no
network. `LLMComposer` asks Claude to write a short answer from the same
passages, validates the result, and falls back to `RuleComposer` on any
problem (missing SDK/credentials, API error, refusal, bad citations, numbers
not present in the cited passages).
"""

import json
import os
import re
import sys

from .support import Evidence
from .text import tokenize


def select_sentences(ev: Evidence, query_terms: list[str], weight, max_sentences: int = 3) -> list[str]:
    """Pick the document sentences that best match the question (in document order).

    Question terms that also appear in the title only say which article this is,
    so they count for a quarter; the remaining terms decide which sentences hold
    the specific answer. If only title terms matched, the question is about the
    topic in general and the article's opening sentences are returned.
    """
    doc, terms = ev.hit.chunk.doc, set(query_terms)
    title_terms = terms & set(tokenize(doc.title))
    specific = terms - title_terms
    scores, specific_hit = [], False
    for s in doc.sentences:
        present = set(tokenize(s))
        specific_hit |= bool(specific & present)
        scores.append(sum(weight(t) for t in specific & present)
                      + 0.25 * sum(weight(t) for t in title_terms & present))
    if not specific_hit:
        return doc.sentences[:2]
    top = max(scores)
    keep = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = sorted(i for i in keep[:max_sentences] if scores[i] >= 0.5 * top)
    return [doc.sentences[i] for i in keep]


def make_citations(evidence: list[Evidence], snippets: dict[str, list[str]]) -> list[dict]:
    return [{"id": ev.hit.chunk.doc.id, "title": ev.hit.chunk.doc.title,
             "snippet": " ".join(snippets[ev.hit.chunk.doc.id])} for ev in evidence]


class RuleComposer:
    name = "rule"

    def compose(self, question, query_terms, evidence, weight):
        snippets = {ev.hit.chunk.doc.id: select_sentences(ev, query_terms, weight) for ev in evidence}
        parts = [f"{' '.join(snippets[ev.hit.chunk.doc.id])} [{ev.hit.chunk.doc.id}]" for ev in evidence]
        return " ".join(parts), make_citations(evidence, snippets)


class LLMComposer:
    name = "llm"
    SYSTEM = (
        "You are a customer-support assistant. Answer the user's question using ONLY the "
        "passages provided. Do not add steps, timings, policies or product behaviour that "
        "the passages do not state. Keep the answer to 1-3 sentences. If the passages do not "
        "actually answer the question, set decision to \"abstain\". List in citation_ids the "
        "ids of every passage your answer relies on."
    )
    SCHEMA = {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["answer", "abstain"]},
            "answer": {"type": "string"},
            "citation_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["decision", "answer", "citation_ids"],
        "additionalProperties": False,
    }

    def __init__(self, model: str | None = None):
        import anthropic  # optional dependency; ImportError triggers fallback in build_composer
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5")
        self.fallback = RuleComposer()

    def compose(self, question, query_terms, evidence, weight):
        try:
            return self._compose(question, query_terms, evidence, weight)
        except Exception as e:  # never let the LLM path break the service
            print(f"[llm] falling back to rule composer: {e}", file=sys.stderr)
            return self.fallback.compose(question, query_terms, evidence, weight)

    def _compose(self, question, query_terms, evidence, weight):
        by_id = {ev.hit.chunk.doc.id: ev for ev in evidence}
        passages = "\n\n".join(
            f'<passage id="{ev.hit.chunk.doc.id}" title="{ev.hit.chunk.doc.title}">\n'
            f"{ev.hit.chunk.doc.text}\n</passage>" for ev in evidence)
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=2000,
            system=self.SYSTEM,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": self.SCHEMA}},
            messages=[{"role": "user", "content": f"{passages}\n\nQuestion: {question}"}],
        )
        if response.stop_reason != "end_turn":
            raise RuntimeError(f"stop_reason={response.stop_reason}")
        data = json.loads(next(b.text for b in response.content if b.type == "text"))

        if data["decision"] == "abstain":
            return None, []  # LLM may only make the decision stricter
        cited = [i for i in dict.fromkeys(data["citation_ids"]) if i in by_id]
        if not cited or len(cited) != len(set(data["citation_ids"])):
            raise ValueError(f"invalid citations {data['citation_ids']}")
        cited_text = " ".join(by_id[i].hit.chunk.doc.text for i in cited)
        for num in re.findall(r"\d+", data["answer"]):
            if num not in cited_text:
                raise ValueError(f"answer contains unsupported number '{num}'")

        used = [by_id[i] for i in cited]
        snippets = {ev.hit.chunk.doc.id: select_sentences(ev, query_terms, weight) for ev in used}
        return data["answer"].strip(), make_citations(used, snippets)


def build_composer(mode: str):
    if mode == "llm":
        try:
            return LLMComposer()
        except Exception as e:
            print(f"[llm] unavailable ({e}); using rule-based composer", file=sys.stderr)
    return RuleComposer()
