"""One-shot validation: unit tests + evaluation metrics. Exit code 0 means everything passed."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT)
print("\n=== Evaluation (eval_questions.json) ===", flush=True)
evaluation = subprocess.run([sys.executable, "eval.py"], cwd=ROOT)
ok = tests.returncode == 0 and evaluation.returncode == 0
print("\nVALIDATION", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
