"""
Document loading, chunking, embedding, and vector store.
Uses Azure OpenAI embeddings; store is Chroma with optional persistence.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT_EMBEDDING,
    AZURE_OPENAI_ENDPOINT,
    CHROMA_PERSIST_DIR,
    DOCS_DIR,
)


def get_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_EMBEDDING,
    )


def load_docs_from_directory(docs_path: Path) -> List:
    """Load .txt and .pdf from docs_path into LangChain documents."""
    docs = []
    if not docs_path.exists():
        return docs
    for ext, loader_cls in [(".txt", TextLoader), (".pdf", PyPDFLoader)]:
        for path in docs_path.rglob(f"*{ext}"):
            try:
                loader = loader_cls(str(path))
                docs.extend(loader.load())
            except Exception:
                pass
    return docs


def build_vector_store(
    docs_path: Path | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    use_persist: bool = True,
):
    """Load docs, chunk, embed, and return a Chroma vector store."""
    path = docs_path or DOCS_DIR
    raw_docs = load_docs_from_directory(path)
    if not raw_docs:
        raise ValueError(f"No .txt or .pdf files found under {path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(raw_docs)

    embeddings = get_embeddings()
    persist_dir = str(CHROMA_PERSIST_DIR) if use_persist else None
    if persist_dir:
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="rag_docs",
    )


def load_existing_store():
    """Load Chroma from persist dir if it exists; otherwise None."""
    if not CHROMA_PERSIST_DIR.exists():
        return None
    try:
        return Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=get_embeddings(),
            collection_name="rag_docs",
        )
    except Exception:
        return None


def get_retriever(store, k: int = 4):
    """Return a retriever that yields top-k chunks."""
    return store.as_retriever(search_kwargs={"k": k})
