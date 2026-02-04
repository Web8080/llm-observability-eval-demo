"""
Load docs from docs/, chunk, embed with Azure OpenAI, and persist to Chroma.
Run once (or when docs change) before running queries or eval.
"""

import sys
from pathlib import Path

# project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.config import DOCS_DIR
from src.observability import configure_tracing
from src.retriever import build_vector_store


def main():
    configure_tracing()
    store = build_vector_store(docs_path=DOCS_DIR, use_persist=True)
    print("Ingestion done. Vector store persisted under data/chroma.")


if __name__ == "__main__":
    main()
