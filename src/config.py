"""
Environment-based config for Azure OpenAI and observability.
"""

import os
from pathlib import Path


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


# Azure OpenAI
AZURE_OPENAI_ENDPOINT = _env("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = _env("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = _env("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_DEPLOYMENT_CHAT = _env("AZURE_OPENAI_DEPLOYMENT_CHAT", "gpt-4o")
AZURE_OPENAI_DEPLOYMENT_EMBEDDING = _env("AZURE_OPENAI_DEPLOYMENT_EMBEDDING", "text-embedding-3-small")

# LangSmith: when LANGCHAIN_TRACING_V2=true, chains and LLM calls get trace IDs,
# latency, token usage, and prompt/completion visibility in the LangSmith UI.
LANGCHAIN_TRACING_V2 = _env("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1", "yes")
LANGCHAIN_PROJECT = _env("LANGCHAIN_PROJECT", "llm-observability-demo")

# Optional OTLP (when not using LangSmith)
OTEL_EXPORTER_OTLP_ENDPOINT = _env("OTEL_EXPORTER_OTLP_ENDPOINT")
OTEL_SERVICE_NAME = _env("OTEL_SERVICE_NAME", "llm-observability-demo")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "data" / "chroma"
