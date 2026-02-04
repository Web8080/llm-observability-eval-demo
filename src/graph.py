"""
LangGraph RAG: retrieve -> build context -> generate with Azure OpenAI.
"""

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph

from .config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT_CHAT,
    AZURE_OPENAI_ENDPOINT,
)
from .prompts import RAG_SYSTEM, RAG_USER_TEMPLATE


class RAGState(TypedDict):
    question: str
    chunks: list
    context: str
    answer: str


def _format_doc(doc) -> str:
    content = getattr(doc, "page_content", str(doc))
    return content.strip()


def create_graph(retriever, llm=None):
    """Build a two-node graph: retrieve -> generate."""
    if llm is None:
        llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT_CHAT,
            temperature=0,
        )

    def retrieve(state: RAGState) -> dict:
        question = state["question"]
        chunks = retriever.invoke(question)
        context = "\n\n".join(_format_doc(c) for c in chunks)
        return {"chunks": chunks, "context": context}

    def generate(state: RAGState) -> dict:
        prompt = RAG_USER_TEMPLATE.format(
            context=state["context"],
            question=state["question"],
        )
        messages = [
            SystemMessage(content=RAG_SYSTEM),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        return {"answer": answer}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge("__start__", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def run_rag(compiled_graph, question: str) -> tuple[str, list]:
    """Run the graph for one question. Returns (answer, list of retrieved doc chunks)."""
    state = compiled_graph.invoke({"question": question})
    chunks = state.get("chunks") or []
    answer = state.get("answer") or ""
    return answer, list(chunks)
