"""
RAG pipeline entry point: runnable that takes a question and returns answer plus
retrieved chunks. Logs each run for prompt monitoring (version, latency, tokens).
"""

import json
import time
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .graph import run_rag, create_graph
from .prompts import PROMPT_VERSION
from .retriever import build_vector_store, get_retriever, load_existing_store


def _chunk_ids(chunks) -> list[str]:
    ids = []
    for i, c in enumerate(chunks):
        doc_id = getattr(c, "metadata", {}).get("doc_id") or getattr(c, "id", None) or f"chunk_{i}"
        ids.append(str(doc_id))
    return ids


def _token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


def log_run(
    prompt_version: str,
    question: str,
    chunk_ids: list[str],
    answer: str,
    latency_seconds: float,
    input_tokens: int,
    output_tokens: int,
    log_path: Path | None = None,
):
    """Append one run to JSONL for prompt monitoring and quality tracking."""
    if log_path is None:
        log_path = PROJECT_ROOT / "data" / "runs.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "prompt_version": prompt_version,
        "question": question[:500],
        "chunk_ids": chunk_ids,
        "answer_preview": (answer or "")[:300],
        "latency_seconds": round(latency_seconds, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(row) + "\n")


class RAGPipeline:
    """Single runnable: question -> (answer, retrieved_chunks). Uses existing Chroma or builds from docs."""

    def __init__(self, docs_path: Path | None = None, persist_store: bool = True):
        self._store = load_existing_store()
        if self._store is None:
            self._store = build_vector_store(docs_path=docs_path, use_persist=persist_store)
        self._retriever = get_retriever(self._store, k=4)
        self._graph = create_graph(self._retriever)
        self._run_log_path = PROJECT_ROOT / "data" / "runs.jsonl"

    def run(self, question: str) -> dict[str, Any]:
        """Run RAG and return answer, chunks, and run metadata (latency, token estimates)."""
        t0 = time.perf_counter()
        answer, chunks = run_rag(self._graph, question)
        latency = time.perf_counter() - t0
        chunk_ids = _chunk_ids(chunks)
        context = "\n\n".join(getattr(c, "page_content", str(c)) for c in chunks)
        input_tokens = _token_estimate(question + context)
        output_tokens = _token_estimate(answer)
        log_run(
            PROMPT_VERSION,
            question,
            chunk_ids,
            answer,
            latency,
            input_tokens,
            output_tokens,
            self._run_log_path,
        )
        return {
            "answer": answer,
            "retrieved_chunks": chunks,
            "chunk_ids": chunk_ids,
            "latency_seconds": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "prompt_version": PROMPT_VERSION,
        }


def get_pipeline(docs_path: Path | None = None) -> RAGPipeline:
    """Build pipeline (loads docs if needed)."""
    return RAGPipeline(docs_path=docs_path)
