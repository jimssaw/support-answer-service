"""LOAD_KB stage: read kb.json from disk, validate it and split it into chunks."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from .text import split_sentences, tokenize

# Sliding window of N consecutive sentences per chunk (stride 1).
WINDOW = 2


class KBError(ValueError):
    pass


@dataclass
class Document:
    id: str
    title: str
    text: str
    sentences: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    doc: Document
    sentence_idx: list[int]
    tokens: list[str]  # title tokens + window tokens, used for retrieval and support

    @property
    def text(self) -> str:
        return " ".join(self.doc.sentences[i] for i in self.sentence_idx)


def load_kb(path) -> list[Document]:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise KBError(f"knowledge base not found: {path}")
    except json.JSONDecodeError as e:
        raise KBError(f"knowledge base is not valid JSON ({path}): {e}")
    if not isinstance(raw, list) or not raw:
        raise KBError("knowledge base must be a non-empty JSON array")

    docs, seen = [], set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise KBError(f"entry {i} is not an object")
        for key in ("id", "title", "text"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise KBError(f"entry {i} is missing a non-empty string '{key}'")
        if entry["id"] in seen:
            raise KBError(f"duplicate id '{entry['id']}'")
        seen.add(entry["id"])
        docs.append(Document(entry["id"], entry["title"].strip(), entry["text"].strip(),
                             split_sentences(entry["text"])))
    return docs


def chunk_documents(docs: list[Document], window: int = WINDOW) -> list[Chunk]:
    chunks = []
    for doc in docs:
        title_tokens = tokenize(doc.title)
        n = len(doc.sentences)
        starts = range(max(1, n - window + 1))
        for start in starts:
            idx = list(range(start, min(start + window, n)))
            tokens = title_tokens + [t for i in idx for t in tokenize(doc.sentences[i])]
            chunks.append(Chunk(doc, idx, tokens))
    return chunks
