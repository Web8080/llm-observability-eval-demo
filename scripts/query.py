"""
Run a single RAG query from the command line.
Usage: python scripts/query.py "Your question here"
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.observability import configure_tracing
from src.pipeline import get_pipeline


def main():
    configure_tracing()
    if len(sys.argv) < 2:
        print("Usage: python scripts/query.py \"Your question\"")
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    pipeline = get_pipeline()
    out = pipeline.run(question)
    print("Answer:", out["answer"])
    print("Chunk IDs:", out["chunk_ids"])
    print("Latency (s):", round(out["latency_seconds"], 3))


if __name__ == "__main__":
    main()
