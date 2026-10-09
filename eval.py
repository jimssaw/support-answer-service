"""Small evaluation: answer/abstain accuracy, citation presence and citation correctness.

    python eval.py                       # sample KB + eval_questions.json
    python eval.py --kb other.json --questions other_questions.json --mode llm
"""

import argparse
import json
import sys
from pathlib import Path

from supportqa import DEFAULT_KB, SupportQA


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--kb", default=DEFAULT_KB)
    p.add_argument("--questions", default=Path(__file__).resolve().parent / "eval_questions.json")
    p.add_argument("--mode", choices=["rule", "llm"], default="rule")
    args = p.parse_args(argv)

    qa = SupportQA(args.kb, mode=args.mode)
    cases = json.load(open(args.questions, encoding="utf-8"))
    correct = answered = with_citations = right_source = expected_answers = 0

    for case in cases:
        r = qa.ask(case["question"])
        ok = r["decision"] == case["expected"]
        correct += ok
        cited = [c["id"] for c in r["citations"]]
        if r["decision"] == "answer":
            answered += 1
            with_citations += bool(cited)
        if case["expected"] == "answer" and case.get("expected_ids"):
            expected_answers += 1
            right_source += ok and bool(set(cited) & set(case["expected_ids"]))
        print(f"[{'PASS' if ok else 'FAIL'}] {r['decision']:7} expected={case['expected']:7} "
              f"score={r['debug']['support_score']:.2f} cited={cited}  {case['question']}")

    n = len(cases)
    print("\nMetrics")
    print(f"  decision accuracy      : {correct}/{n} = {correct / n:.0%}")
    print(f"  citation presence rate : {with_citations}/{answered} answered" +
          (f" = {with_citations / answered:.0%}" if answered else ""))
    if expected_answers:
        print(f"  correct source cited   : {right_source}/{expected_answers} = {right_source / expected_answers:.0%}")
    return 0 if correct == n else 1


if __name__ == "__main__":
    sys.exit(main())
