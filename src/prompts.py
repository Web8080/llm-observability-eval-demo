"""
RAG prompt template and version for monitoring.
Change PROMPT_VERSION when you iterate on the template so runs can be compared.
"""

PROMPT_VERSION = "v1"

RAG_SYSTEM = """You answer questions using only the provided context. If the context does not contain enough information, say so. Do not invent facts. Keep answers concise."""

RAG_USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""


def get_rag_prompt_context(context: str, question: str) -> dict:
    """Fill the RAG user template. Returns dict for LangChain format."""
    return {
        "context": context,
        "question": question,
    }
