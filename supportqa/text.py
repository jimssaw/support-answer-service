"""Tokenisation, light stemming and sentence splitting (stdlib only)."""

import re

# Generic English stopwords plus question "filler" words that carry no topic
# information ("how long does it take", "can I", "do you"). Nothing here is
# specific to any knowledge base.
STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been
being before below between both but by can could did do does doing done down
during each either else ever every few for from further get gets getting got
had has have having he her here hers him his how i if in into is it its itself
just let me might more most must my myself need needs no nor not now of off on
once only or other our ours out over own please possible same she should so
some such than that the their theirs them then there these they this those
through to too under until up us very want wants was way we were what when
where which while who whom why will with would yet you your yours yourself
able allowed long much many often take takes taking tell know usually really
anyone someone something anything thing things kind sort ok okay hi hello hey
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'A-Z0-9])")


def stem(word: str) -> str:
    """Tiny suffix stripper so that close/closed/closing and change/changes match."""
    if word.isdigit() or len(word) <= 3:
        return word
    for suffix, repl in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("es", "e"), ("s", "")):
        if word.endswith(suffix) and not (suffix == "s" and word.endswith("ss")):
            candidate = word[: -len(suffix)] + repl
            if len(candidate) >= 3:
                word = candidate
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [stem(t) for t in tokens]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]
