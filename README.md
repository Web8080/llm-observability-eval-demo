# LLM Observability & Evaluation Demo

RAG Q&A pipeline with LangGraph, Azure OpenAI, observability (LangSmith), evaluation, and prompt monitoring.

**GitHub repo description (paste in Settings > General > Description):** see [GITHUB_DESCRIPTION.txt](./GITHUB_DESCRIPTION.txt).

This project demonstrates (1) LLM observability via LangSmith tracing, (2) evaluation on a small Q&A dataset with correctness/similarity metrics, and (3) prompt monitoring with versioned prompts and per-run logs (latency, tokens) for quality tracking.

**Tech stack:** Python 3.11+, LangChain, LangGraph, Azure OpenAI, LangSmith.

---

## Purpose and Product Context

- **Use case:** Document Q&A over a small doc set. User asks questions; a RAG pipeline (LangChain + LangGraph) calls Azure OpenAI, logs every run (observability), scores answers (evaluation), and tracks prompt versions and quality (prompt monitoring).
- **Target users:** Engineers evaluating RAG + observability + evals in one place.
- **Non-goals:** Production-scale indexing, multi-tenant APIs, or custom model training.

---

## Architecture Overview

- **src/:** Core code. `retriever` loads/chunks docs, embeds with Azure OpenAI, stores in Chroma. `graph` defines the LangGraph (retrieve -> generate). `pipeline` exposes a runnable that returns answer + retrieved chunks and logs each run. `prompts` holds the versioned RAG template. `observability` sets up LangSmith or optional OTLP.
- **scripts/:** `ingest.py` (build vector store), `query.py` (single question), `run_eval.py` (run eval set and print scores).
- **data/:** Chroma persist dir, `eval_dataset.json`, and `runs.jsonl` (prompt monitoring log).
- **docs/:** Sample .txt (and optionally PDF) for RAG.

---

## Key Workflows

1. **Ingest:** Run `scripts/ingest.py` to load docs from `docs/`, chunk, embed, persist to `data/chroma`.
2. **Query:** Run `scripts/query.py "Your question"`; pipeline retrieves chunks, calls Azure OpenAI, returns answer and logs to `data/runs.jsonl`.
3. **Eval:** Run `scripts/run_eval.py`; each eval question goes through the pipeline, answers are scored (e.g. exact match), summary (average, pass rate) printed.

---

## Running the Demo

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment**
   - Copy `.env.example` to `.env`.
   - Set `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_PROJECT=llm-observability-demo` for LangSmith.
   - Set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and deployment names for chat and embeddings.

3. **Ingest documents**
   ```bash
   python scripts/ingest.py
   ```

4. **Run one query**
   ```bash
   python scripts/query.py "What does this project demonstrate?"
   ```

5. **Run evaluation**
   ```bash
   python scripts/run_eval.py
   ```

6. **Launch Web UI (local)**
   ```bash
   python scripts/serve.py
   ```
   Open http://127.0.0.1:8000 to ask questions in the browser.

---

## Deploy (live demo)

- **Render:** Connect this repo to [Render](https://render.com). Use the `render.yaml` blueprint or create a Web Service: build `pip install -r requirements.txt`, start `uvicorn src.app:app --host 0.0.0.0 --port $PORT`. Set all env vars from `.env.example` in the Render dashboard. Run ingest once (e.g. via a one-off job or a boot script that builds Chroma if missing) so the vector store exists; for a minimal live demo you can bake Chroma into the build or run ingest on first request (slower).
- **Vercel:** Suited for static/Next.js frontends. This app is FastAPI (stateful RAG + Chroma); deploy the API to Render or Railway and point a frontend at it, or use a single FastAPI-on-Render deployment as the live demo link.

---

## Configuration

| Variable | Purpose |
|----------|---------|
| `LANGCHAIN_API_KEY` | LangSmith API key for tracing |
| `LANGCHAIN_TRACING_V2` | Set to `true` to enable tracing |
| `LANGCHAIN_PROJECT` | LangSmith project name (e.g. `llm-observability-demo`) |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT_CHAT` | Chat model deployment (e.g. `gpt-4o`) |
| `AZURE_OPENAI_DEPLOYMENT_EMBEDDING` | Embedding model deployment (e.g. `text-embedding-3-small`) |

See `.env.example` for a full list.

---

## Failure Modes

- **No docs:** Ingest fails if `docs/` has no `.txt` or `.pdf`. Add at least one file (e.g. `docs/sample.txt`).
- **Missing .env:** Query/eval will fail without valid Azure OpenAI and (if tracing) LangSmith credentials.
- **Chroma missing:** If you run query or eval before ingest, the pipeline will try to build the store from `docs/`; if `docs/` is empty, that build fails.

---

## Debugging Tips

- Enable LangSmith and open the project to see trace IDs, latency, and token usage per run.
- Check `data/runs.jsonl` for prompt version, question, chunk IDs, latency, and token counts per run.
- Eval scores are printed per row and summarized at the end; adjust `data/eval_dataset.json` to add or change questions/expected answers.

---

## Explicit Non-Goals

- No production-scale vector DB or distributed ingestion.
- No Azure Application Insights or OpenTelemetry wiring in the main path (optional OTLP is environment-based and may require extra setup).
- Eval is a minimal scorer (exact match); LangSmith eval helpers can be wired in for richer metrics.

---

## Known Debt

- Token counts are estimates (char/4); real usage comes from LangSmith or API response metadata if needed.
- Eval uses a simple exact-match style check; semantic similarity or LLM-as-judge can be added.

---

## CV bullet

Built a RAG pipeline with LangGraph and Azure OpenAI, with LangSmith observability, LLM evaluation on a Q&A dataset, and versioned prompt monitoring.
