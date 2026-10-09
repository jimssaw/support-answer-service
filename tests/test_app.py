"""Run with:  python -m unittest -v   (or: pytest)"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import make_server  # noqa: E402
from supportqa import ABSTAIN_MESSAGE, SupportQA  # noqa: E402
from supportqa.composers import LLMComposer, RuleComposer  # noqa: E402
from supportqa.kb import KBError, load_kb  # noqa: E402

KB_PATH = ROOT / "kb.json"
ANSWERABLE = ["How do I reset my password?", "What 2FA methods are supported?",
              "How long do withdrawal reviews take?", "How do I change my address?"]
UNSUPPORTED = ["Can you guarantee my withdrawal will finish in 2 hours?",
               "What tax advice do you provide for closed accounts?",
               "Can support change my email address for me?",
               "How do I buy bitcoin?"]

ALT_KB = [
    {"id": "pet_1", "title": "Grooming appointments",
     "text": "Grooming appointments can be booked in the mobile app. Each appointment lasts 90 minutes. "
             "Cancellations are free up to 12 hours before the appointment."},
    {"id": "pet_2", "title": "Vaccination records",
     "text": "Dogs must have up-to-date rabies vaccination records before boarding. "
             "Records can be uploaded from the pet profile page."},
]


def assert_schema(tc, r):
    tc.assertEqual(set(r), {"question", "decision", "answer", "citations", "debug"})
    tc.assertIn(r["decision"], ("answer", "abstain"))
    tc.assertIsInstance(r["answer"], str)
    tc.assertTrue(r["answer"])
    for c in r["citations"]:
        tc.assertEqual(set(c), {"id", "title", "snippet"})
        tc.assertTrue(all(isinstance(c[k], str) for k in c))
    tc.assertEqual(set(r["debug"]), {"retrieved_ids", "support_score"})
    tc.assertIsInstance(r["debug"]["retrieved_ids"], list)
    tc.assertIsInstance(r["debug"]["support_score"], float)


def write_kb(data) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    return Path(f.name)


class TestKnowledgeBase(unittest.TestCase):
    def test_loads_kb_json_from_disk(self):
        on_disk = json.loads(KB_PATH.read_text())
        docs = load_kb(KB_PATH)
        self.assertEqual([d.id for d in docs], [e["id"] for e in on_disk])
        self.assertTrue(all(d.sentences for d in docs))

    def test_invalid_kb_is_rejected(self):
        for bad in ([], [{"id": "x", "title": "t"}], {"id": "x"}):
            with self.assertRaises(KBError):
                load_kb(write_kb(bad))
        with self.assertRaises(KBError):
            load_kb(ROOT / "does_not_exist.json")


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qa = SupportQA(KB_PATH)
        cls.kb = {d["id"]: d for d in json.loads(KB_PATH.read_text())}

    def test_answerable_questions_answer_with_grounded_citations(self):
        for q in ANSWERABLE:
            with self.subTest(q=q):
                r = self.qa.ask(q)
                assert_schema(self, r)
                self.assertEqual(r["decision"], "answer")
                self.assertGreaterEqual(len(r["citations"]), 1)
                for c in r["citations"]:
                    self.assertIn(c["id"], r["debug"]["retrieved_ids"])
                    self.assertEqual(c["title"], self.kb[c["id"]]["title"])
                    self.assertIn(c["snippet"], self.kb[c["id"]]["text"])
                    # rule-based answers are composed only of cited snippets
                    self.assertIn(c["snippet"], r["answer"])

    def test_expected_source_is_cited(self):
        r = self.qa.ask("How do I reset my password?")
        self.assertEqual(r["citations"][0]["id"], "doc_1")
        self.assertIn("Forgot password", r["answer"])

    def test_unsupported_questions_abstain(self):
        for q in UNSUPPORTED + ["", "   ", "how do I?"]:
            with self.subTest(q=q):
                r = self.qa.ask(q)
                assert_schema(self, r)
                self.assertEqual(r["decision"], "abstain")
                self.assertEqual(r["citations"], [])
                self.assertEqual(r["answer"], ABSTAIN_MESSAGE)
                self.assertIn("support", r["answer"].lower())

    def test_support_score_matches_decision(self):
        from supportqa.support import MIN_COVERAGE
        for q in ANSWERABLE + UNSUPPORTED:
            r = self.qa.ask(q)
            self.assertEqual(r["decision"] == "answer", r["debug"]["support_score"] >= MIN_COVERAGE, q)


class TestReplacementKB(unittest.TestCase):
    """Proves answers come from the file on disk, not embedded constants."""

    def setUp(self):
        self.path = write_kb(ALT_KB)
        self.qa = SupportQA(self.path)

    def test_answers_from_replacement_kb(self):
        r = self.qa.ask("How long does a grooming appointment last?")
        self.assertEqual(r["decision"], "answer")
        self.assertEqual(r["citations"][0]["id"], "pet_1")
        self.assertIn("90 minutes", r["answer"])

    def test_sample_questions_now_abstain(self):
        r = self.qa.ask("How do I reset my password?")
        self.assertEqual(r["decision"], "abstain")
        self.assertNotIn("doc_1", json.dumps(r))

    def test_numbers_must_be_supported(self):
        self.assertEqual(self.qa.ask("Can I cancel 2 hours before my grooming appointment?")["decision"],
                         "abstain")

    def test_cli_reads_kb_path(self):
        out = subprocess.run([sys.executable, str(ROOT / "app.py"), "--kb", str(self.path),
                              "Where do I upload vaccination records?"],
                             capture_output=True, text=True, check=True)
        r = json.loads(out.stdout)
        assert_schema(self, r)
        self.assertEqual(r["decision"], "answer")
        self.assertEqual(r["citations"][0]["id"], "pet_2")

    def test_cli_env_var_and_stdin_json(self):
        env = {**os.environ, "KB_PATH": str(self.path)}
        out = subprocess.run([sys.executable, str(ROOT / "app.py")], env=env,
                             input=json.dumps({"question": "Where do I upload vaccination records?"}),
                             capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(out.stdout)["citations"][0]["id"], "pet_2")


class TestHTTP(unittest.TestCase):
    def test_post_ask(self):
        server = make_server(SupportQA(KB_PATH), "127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/ask"
            for q, decision in ((ANSWERABLE[0], "answer"), (UNSUPPORTED[0], "abstain")):
                req = urllib.request.Request(url, data=json.dumps({"question": q}).encode(),
                                             headers={"Content-Type": "application/json"})
                r = json.loads(urllib.request.urlopen(req).read())
                assert_schema(self, r)
                self.assertEqual(r["decision"], decision)
        finally:
            server.shutdown()
            server.server_close()


class FakeClient:
    """Stands in for anthropic.Anthropic so the LLM path is tested offline."""

    def __init__(self, payload):
        text = SimpleNamespace(type="text", text=json.dumps(payload))
        response = SimpleNamespace(stop_reason="end_turn", content=[text])
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response))


class TestLLMComposer(unittest.TestCase):
    def ask_with(self, payload, question="How do I reset my password?"):
        qa = SupportQA(KB_PATH)
        composer = LLMComposer.__new__(LLMComposer)
        composer.client, composer.model, composer.fallback = FakeClient(payload), "test", RuleComposer()
        qa.composer = composer
        return qa.ask(question)

    def test_valid_llm_answer_is_used(self):
        r = self.ask_with({"decision": "answer", "answer": "Click 'Forgot password' on the login page.",
                           "citation_ids": ["doc_1"]})
        self.assertEqual(r["answer"], "Click 'Forgot password' on the login page.")
        self.assertEqual(r["citations"][0]["id"], "doc_1")

    def test_invented_timing_falls_back_to_rules(self):
        r = self.ask_with({"decision": "answer", "answer": "The link is valid for 2 hours.",
                           "citation_ids": ["doc_1"]})
        self.assertNotIn("2 hours", r["answer"])
        self.assertEqual(r["decision"], "answer")

    def test_unretrieved_citation_falls_back_to_rules(self):
        r = self.ask_with({"decision": "answer", "answer": "Contact support.", "citation_ids": ["doc_9"]})
        self.assertEqual([c["id"] for c in r["citations"]], ["doc_1"])

    def test_llm_can_only_make_decision_stricter(self):
        self.assertEqual(self.ask_with({"decision": "abstain", "answer": "", "citation_ids": []})["decision"],
                         "abstain")
        # Unsupported question: the LLM is never consulted, even if it would answer.
        r = self.ask_with({"decision": "answer", "answer": "Yes, guaranteed.", "citation_ids": ["doc_3"]},
                          question=UNSUPPORTED[0])
        self.assertEqual(r["decision"], "abstain")

    def test_llm_mode_without_sdk_or_key_still_runs(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC")}
        env["ANTHROPIC_API_KEY"] = ""
        out = subprocess.run([sys.executable, str(ROOT / "app.py"), "--mode", "llm", ANSWERABLE[0]],
                             env=env, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(out.stdout)["decision"], "answer")


if __name__ == "__main__":
    unittest.main()
