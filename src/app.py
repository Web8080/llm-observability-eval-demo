"""
FastAPI app for the RAG demo: one-page UI to ask questions and see answer + chunks.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

# Load env before importing pipeline (needs Azure/LangSmith)
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.observability import configure_tracing
from src.pipeline import get_pipeline

configure_tracing()
app = FastAPI(title="LLM Observability & Evaluation Demo", version="0.1.0")

# Lazy init so ingest can run first
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = get_pipeline()
    return _pipeline


HTML_INDEX = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG Q&A Demo</title>
  <style>
    :root { --bg: #0f0f12; --surface: #1a1a1f; --text: #e4e4e7; --muted: #71717a; --accent: #3b82f6; }
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 2rem; min-height: 100vh; }
    .container { max-width: 640px; margin: 0 auto; }
    h1 { font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem; }
    .sub { color: var(--muted); font-size: 0.875rem; margin-bottom: 1.5rem; }
    form { display: flex; flex-direction: column; gap: 0.75rem; }
    input[type="text"] { background: var(--surface); border: 1px solid #27272a; color: var(--text); padding: 0.75rem 1rem; border-radius: 8px; font-size: 1rem; }
    input[type="text"]:focus { outline: none; border-color: var(--accent); }
    button { background: var(--accent); color: #fff; border: none; padding: 0.75rem 1.25rem; border-radius: 8px; font-size: 1rem; cursor: pointer; }
    button:hover { opacity: 0.9; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .result { margin-top: 1.5rem; padding: 1rem; background: var(--surface); border-radius: 8px; border: 1px solid #27272a; }
    .result h2 { font-size: 0.875rem; color: var(--muted); margin: 0 0 0.5rem 0; }
    .answer { white-space: pre-wrap; margin-bottom: 1rem; }
    .meta { font-size: 0.75rem; color: var(--muted); }
    .chunks { margin-top: 1rem; }
    .chunk { font-size: 0.875rem; padding: 0.5rem 0; border-bottom: 1px solid #27272a; }
    .chunk:last-child { border-bottom: none; }
    .error { color: #f87171; }
  </style>
</head>
<body>
  <div class="container">
    <h1>RAG Q&A Demo</h1>
    <p class="sub">Ask a question over the sample docs. Answer and retrieved chunks appear below.</p>
    <form method="post" action="/query" id="f">
      <input type="text" name="question" placeholder="e.g. What does this project demonstrate?" required />
      <button type="submit" id="btn">Ask</button>
    </form>
    <div class="result" id="result"></div>
  </div>
  <script>
    document.getElementById('f').onsubmit = function() {
      document.getElementById('btn').disabled = true;
      document.getElementById('result').innerHTML = 'Running...';
    };
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_INDEX


@app.post("/query", response_class=HTMLResponse)
async def query(question: str = Form("")):
    question = (question or "").strip()
    if not question:
        return HTML_INDEX + '<div class="result error">Please enter a question.</div>'
    try:
        pipeline = _get_pipeline()
        out = pipeline.run(question)
        chunks_preview = []
        for i, c in enumerate(out.get("retrieved_chunks", [])[:5]):
            content = getattr(c, "page_content", str(c))[:200]
            chunks_preview.append(f'<div class="chunk">[{i+1}] {content}...</div>')
        chunks_html = "".join(chunks_preview) if chunks_preview else "<div class='chunk'>No chunks.</div>"
        meta = f"Latency: {out.get('latency_seconds', 0):.2f}s | Tokens (est): in {out.get('input_tokens', 0)} / out {out.get('output_tokens', 0)} | Prompt: {out.get('prompt_version', '')}"
        result = f"""
        <div class="result">
          <h2>Answer</h2>
          <div class="answer">{_escape(out.get("answer", ""))}</div>
          <div class="meta">{_escape(meta)}</div>
          <h2 class="chunks">Retrieved chunks</h2>
          <div class="chunks">{chunks_html}</div>
        </div>
        """
        return HTML_INDEX + result
    except Exception as e:
        return HTML_INDEX + f'<div class="result error">Error: {_escape(str(e))}</div>'


def _escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


@app.get("/health")
async def health():
    return {"status": "ok"}
