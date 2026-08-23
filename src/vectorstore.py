"""Chunks -> embedded, persisted, searchable Chroma store.

Comparison against FAISS and the cosine-score sanity check: README.md.
"""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from config import settings


class VectorStoreService:
    """Owns one Chroma collection: create, reopen, add, and inspect it.

    Chroma's HNSW index defaults to squared L2, not cosine; Ollama embeddings
    are unit-normalized, where L2^2 == 2*(1-cosine_similarity), so ranking is
    unaffected either way but the reported distance isn't a real cosine
    distance unless collection_metadata sets it explicitly (see README).
    """

    def __init__(self, persist_directory: str = settings.PERSIST_DIR,
                 collection_name: str = settings.COLLECTION_NAME):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

    @property
    def exists(self) -> bool:
        return (Path(self.persist_directory) / "chroma.sqlite3").exists()

    @staticmethod
    def embeddings() -> OllamaEmbeddings:
        """Calls Ollama's POST /api/embed (batch), not the older /api/embeddings."""
        return OllamaEmbeddings(model=settings.EMBED_MODEL, base_url=settings.OLLAMA_URL)

    def build(self, chunks: list[Document]) -> Chroma:
        """Embed chunks into a new persistent collection."""
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings(),
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def load(self) -> Chroma:
        """Reopen an existing collection without re-embedding."""
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings(),
            persist_directory=self.persist_directory,
        )

    def ingest(self, chunks: list[Document]) -> Chroma:
        """Create the collection if it doesn't exist yet, else add to it."""
        if self.exists:
            store = self.load()
            store.add_documents(chunks)
            return store
        return self.build(chunks)

    def list_sources(self) -> dict[str, int]:
        """{source: chunk_count} for every distinct source, for `rag list`."""
        metadatas = self.load()._collection.get(include=["metadatas"])["metadatas"]
        counts: dict[str, int] = {}
        for metadata in metadatas:
            source = metadata.get("source", "<unknown>")
            counts[source] = counts.get(source, 0) + 1
        return counts


def _self_check():
    """Wiring only -- no live Ollama/Chroma call (see README for a live index)."""
    service = VectorStoreService()
    assert service.persist_directory == settings.PERSIST_DIR
    assert service.collection_name == settings.COLLECTION_NAME
    assert isinstance(service.exists, bool)
    print("ok (wiring checks only; run vectorstore.py <pdf> for a live index)")


if __name__ == "__main__":
    import sys
    import warnings

    from chunking import DocumentBuilder, RecursiveChunker
    from ingest import PDFLoader

    if len(sys.argv) > 1:
        warnings.filterwarnings("ignore")
        docs = DocumentBuilder.from_records(PDFLoader().load(sys.argv[1]))
        chunks = RecursiveChunker().split(docs)
        service = VectorStoreService()
        store = service.build(chunks)
        print(f"indexed {sys.argv[1]} -> {service.persist_directory}")
        for r in store.similarity_search(sys.argv[2] if len(sys.argv) > 2 else "python", k=3):
            print("---", r.metadata, "---")
            print(r.page_content[:160])
    else:
        _self_check()
