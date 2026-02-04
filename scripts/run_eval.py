"""
Run evaluation: run each eval question through the RAG pipeline, score answers
(exact match or semantic similarity), log scores and summary (average, pass rate).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.observability import configure_tracing
from src.pipeline import get_pipeline


def load_eval_set(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def exact_match(got: str, expected: str) -> float:
    got = (got or "").strip().lower()
    expected = (expected or "").strip().lower()
    return 1.0 if expected in got or got in expected else 0.0


def run_eval(dataset_path: Path | None = None):
    dataset_path = dataset_path or ROOT / "data" / "eval_dataset.json"
    if not dataset_path.exists():
        print("Eval dataset not found:", dataset_path)
        return
    configure_tracing()
    pipeline = get_pipeline()
    rows = load_eval_set(dataset_path)
    results = []
    for i, row in enumerate(rows):
        q = row.get("question", "")
        expected = row.get("expected", "")
        out = pipeline.run(q)
        answer = out.get("answer", "")
        score = exact_match(answer, expected)
        results.append({
            "question": q,
            "expected": expected,
            "answer": answer,
            "score": score,
        })
        print(f"[{i+1}/{len(rows)}] score={score:.2f} | {q[:50]}...")
    scores = [r["score"] for r in results]
    avg = sum(scores) / len(scores) if scores else 0
    pass_count = sum(1 for s in scores if s >= 0.5)
    print("\nSummary: average score =", round(avg, 3), "| pass rate =", pass_count, "/", len(scores))
    return results


def main():
    run_eval()


if __name__ == "__main__":
    main()
