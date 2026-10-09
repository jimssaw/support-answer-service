# Grounded Support-Answer Service

A small support bot that answers questions only from a local FAQ knowledge base (`kb.json`).
Every answer cites the passages it came from. When the knowledge base doesn't clearly
support an answer, the bot declines and points the user to human support.

- **No dependencies** for the core path: Python ≥ 3.10, standard library only.
- **Interfaces:** a CLI (`python app.py "..."`) and an HTTP API (`POST /ask`).
- **Two answer modes:** rule-based extractive answers (the default, offline) or Claude-written
  answers (`--mode llm`, optional). The LLM mode falls back to rule-based when it can't run.

## Setup

```bash
git clone <repo> && cd support-answer-service
python3 --version            # 3.10+
# optional, only for --mode llm:
pip install -r requirements-llm.txt && export ANTHROPIC_API_KEY=...
```

## Run

```bash
# CLI: question as an argument, or {"question": "..."} JSON on stdin
python app.py "How do I reset my password?"
echo '{"question": "What 2FA methods are supported?"}' | python app.py

# Use a different knowledge base (same schema)
python app.py --kb /path/to/other_kb.json "..."      # or: KB_PATH=/path/to/kb.json python app.py "..."

# HTTP API
python app.py --serve --port 8000
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "How long do withdrawal reviews take?"}'

# LLM-backed composition (falls back to rule-based without SDK/key)
python app.py --mode llm "How do I change my address?"     # or ANSWER_MODE=llm
```

Example output:

```json
{
  "question": "Can you guarantee my withdrawal will finish in 2 hours?",
  "decision": "abstain",
  "answer": "I don't have enough information in the knowledge base to answer that reliably. Please contact human support for help with this request.",
  "citations": [],
  "debug": { "retrieved_ids": ["doc_3", "doc_6"], "support_score": 0.0 }
}
```

## Validate

```bash
python validate.py          # unit tests + evaluation; exits 0 only if everything passes
python -m unittest -v       # tests only (pytest also works)
python eval.py              # evaluation metrics only
```

The tests (`tests/test_app.py`) check that:

- `kb.json` loads from disk, and malformed files are rejected with a clear error;
- answerable questions return `decision: "answer"` with ≥ 1 citation, and each citation's snippet
  is verbatim text from the cited document;
- unsupported questions (including the three examples in the brief) return `decision: "abstain"`;
- every response has exactly the required JSON keys;
- **the answers come from the file, not from the code**: with a different KB (pet grooming), the bot
  answers from it, and the sample questions now abstain. This is checked through the CLI with both
  `--kb` and `KB_PATH`;
- the HTTP endpoint works;
- the LLM layer: a valid LLM answer is used, while an invented timing or a citation to a passage that
  wasn't retrieved triggers the rule-based fallback. The LLM can't override an abstain decision, and
  `--mode llm` still runs without an API key. A fake client stands in for the API, so no key is needed.

`eval.py` runs `eval_questions.json` (6 answerable questions, 4 that should abstain) and reports
answer/abstain accuracy, citation presence rate, and how often the expected source was cited.
On the sample KB the result is 10/10, 100%, 6/6. For a replacement KB, pass
`--kb new_kb.json --questions new_questions.json`.

## How it works

```text
LOAD_KB ─▶ RETRIEVE ─▶ CHECK_SUPPORT ─▶ ANSWER  (composer: rule | llm)
                                     └▶ ABSTAIN
```

| Stage | Code | What happens |
|---|---|---|
| LOAD_KB | `supportqa/kb.py` | Reads and validates `kb.json` (non-empty array of `{id, title, text}`, unique ids), splits each document into sentences, then builds chunks. |
| RETRIEVE | `supportqa/retriever.py` | BM25 over the chunks. Question and chunks are tokenised the same way: lowercased, stopwords and question filler words removed ("how long", "can I"), light suffix stemming. Returns the top 5 chunks. |
| CHECK_SUPPORT | `supportqa/support.py` | Applies the rule below to the retrieved chunks only. |
| ANSWER / ABSTAIN | `supportqa/composers.py`, `supportqa/pipeline.py` | A composer runs only if support passed, and it only sees the supporting passages. Otherwise the fixed abstain message is returned. |

Retrieval always finishes before any answer is composed. The composer receives evidence objects
produced by the support check, so it has no other way to see the KB.

### Support rule (answer vs. abstain)

Let *T* be the question's content terms (after stopword removal and stemming). Each term is
weighted by its document-level IDF. A term the KB never uses gets the highest weight, because
it suggests the question is about something the KB doesn't cover.

A retrieved chunk **supports** the question only if all three conditions hold:

1. **Coverage ≥ 0.65.** The IDF-weighted share of *T* that appears in the chunk (title + sentence
   window) must be at least `MIN_COVERAGE = 0.65`.
2. **Numbers match.** Every number in the question must appear in the chunk. "Will it finish in
   **2** hours?" is not supported by "usually complete within 24 hours".
3. **No topic conflict.** If a question term appears in exactly one KB document, that document
   must be the evidence document. For example, "change my **email** address" partly matches
   *Address changes*, but "email" only appears in *Password reset*, so the question mixes topics
   and no single passage supports it.

The bot answers if at least one retrieved chunk supports the question, and otherwise abstains.
If two documents both pass, it cites both. `debug.support_score` is the best coverage among
chunks that pass rules 2 and 3, so `support_score ≥ 0.65` exactly when `decision == "answer"`.

How the rule handles the brief's abstain examples on the sample KB:

| Question | Why it abstains |
|---|---|
| Can you guarantee my withdrawal will finish in 2 hours? | "2" isn't in the evidence; "guarantee" and "finish" aren't covered by the withdrawals doc (coverage 0.40) |
| What tax advice do you provide for closed accounts? | "tax advice" belongs to *Unsupported request types* and "closed accounts" belongs to *Account closure*. No single passage covers the question (best coverage 0.60), and each passage has a topic conflict with the other. |
| Can support change my email address for me? | Topic conflict between "email" (*Password reset*) and "change … address" (*Address changes*) |

### Answer composition (swappable)

- **`rule`** (default): an extractive answer. From each supporting document it selects the
  sentences that best match the question, up to 3, kept in document order. Question terms that
  also appear in the title count for a quarter, because they only identify the article. If only
  title terms matched, the answer is the article's opening sentences. The answer is those
  sentences followed by `[doc_id]`, and each citation snippet is exactly the sentences used.
  Every answer is verbatim KB text, so the bot can't invent steps or timings.
- **`llm`**: sends the supporting passages to Claude (`claude-opus-5-5`, change it with
  `ANTHROPIC_MODEL`). Claude must return a JSON schema `{decision, answer, citation_ids}`. The code
  checks the result before using it:
  - citations must be a non-empty subset of the supporting passages;
  - every number in the answer must appear in the cited passages;
  - Claude may return `abstain`, but it can never turn an abstain into an answer.

  If the SDK is missing, there are no credentials, the API errors, the model refuses, or any check
  fails, the bot logs to stderr and uses the rule-based composer instead. The request also enables
  the API's server-side refusal fallback (`fallbacks: "default"`).

### Chunking choice

Chunks are **sliding windows of 2 consecutive sentences (stride 1), each with the document
title prepended**. A 1-sentence document becomes a single chunk.

- Support articles are short (2–4 sentences), so windows give little speed benefit. They help
  with precision: rules 1 and 2 need the question's terms to appear *together* in one window,
  not scattered across a whole article. Whole-document chunks would let "support … email …
  address" collect matches from three unrelated sentences of *Password reset* and pass the
  coverage check.
- Single-sentence chunks would be too strict. Answers often span two sentences ("Supported
  methods are …" / "… before 2FA can be removed"). The title is attached because it holds the
  topic words ("Withdrawals review") that body sentences often leave out.
- Tradeoff: a question whose answer needs three or more sentences that don't share terms may
  get lower coverage. The composer reduces this by choosing sentences from the whole supporting
  document, not just the matched window.

### Limitations

- Retrieval is lexical, so heavy paraphrase with no shared vocabulary ("how long is it good
  for?" vs. "expires") can cause an abstain. The design accepts that failure mode on purpose:
  a wrong abstain is better than an unsupported answer. To improve recall, add embeddings in
  `BM25Retriever.search`. The support check doesn't depend on how retrieval works.
- Rule 3 is deliberately strict. A question that mentions a word owned by another article
  ("reset my password in the *app*") abstains in a small KB.
- The KB is loaded once at startup in `--serve` mode, so restart the server after editing `kb.json`.
  The CLI reloads it on every call.
- No generated files are written. Every output is computed at query time from the KB on disk.

## Layout

```text
app.py                 CLI + HTTP API (POST /ask, GET /health)
supportqa/
  kb.py                LOAD_KB: load/validate kb.json, sentence-window chunking
  text.py              tokenizer, stopwords, stemmer, sentence splitter
  retriever.py         RETRIEVE: BM25
  support.py           CHECK_SUPPORT: the answer/abstain rule
  composers.py         ANSWER: RuleComposer, LLMComposer (+ validation & fallback)
  pipeline.py          SupportQA.ask(): wires the stages, builds the JSON response
kb.json                sample knowledge base
tests/test_app.py      unit/integration tests
eval.py, eval_questions.json   evaluation script + labelled questions
validate.py            runs tests + evaluation
```
