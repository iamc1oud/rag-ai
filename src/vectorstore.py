"""Chunks (chunking.py) -> embedded, persisted, searchable Chroma store.

Comparison against FAISS and the cosine-score sanity check: README.md
("Vector store" section).
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from config import settings

# Default on-disk location for the persistent Chroma collection. Configured
# once in config.py (issue #7), aliased here so existing imports keep working.
PERSIST_DIR = settings.PERSIST_DIR
COLLECTION_NAME = settings.COLLECTION_NAME


def get_embeddings() -> OllamaEmbeddings:
    """OllamaEmbeddings, talking to the configured Ollama host.

    Under the hood this calls Ollama's POST /api/embed (the batch endpoint --
    embed_documents sends the whole list of chunk texts in one request), not
    the older single-text /api/embeddings. See README for how that was
    confirmed (langchain_ollama -> ollama-python client source).
    """
    return OllamaEmbeddings(model=settings.EMBED_MODEL, base_url=settings.OLLAMA_URL)


def build_vectorstore(
    chunks: list[Document],
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Embed chunks and write them into a persistent Chroma collection.

    Chroma persists automatically on every write when given a persist_directory
    -- there is no separate .persist() call in current langchain-chroma/chromadb
    (that method existed in older Chroma releases and was removed once the
    client became durable-by-default). from_documents() below both creates the
    collection on disk and embeds+adds the chunks in one call.
    """
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=persist_directory,
        # Chroma's HNSW index defaults to squared L2, not cosine. Ollama's
        # embeddings are unit-normalized (confirmed: norm ~1.0), where
        # L2^2 == 2*(1-cosine_similarity) -- so results are the same *ranking*
        # either way, but the reported distance isn't directly the cosine
        # distance unless this is set. See README's cosine sanity check.
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_vectorstore(
    persist_directory: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Reopen a Chroma collection already written to disk -- no re-embedding.

    Chroma reads the collection back from persist_directory; the embedding
    function is still needed at query time (to embed the query itself), not
    to reload the stored vectors.
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=persist_directory,
    )


def add_documents(store: Chroma, chunks: list[Document]) -> list[str]:
    """Embed and add more chunks to an already-open store. Persists immediately."""
    return store.add_documents(chunks)


def list_sources(store: Chroma) -> dict[str, int]:
    """{source: chunk_count} for every distinct source in the store, for `rag list`."""
    metadatas = store._collection.get(include=["metadatas"])["metadatas"]
    counts: dict[str, int] = {}
    for metadata in metadatas:
        source = metadata.get("source", "<unknown>")
        counts[source] = counts.get(source, 0) + 1
    return counts


def store_exists(persist_directory: str = PERSIST_DIR) -> bool:
    """Whether a Chroma collection has already been written to persist_directory.

    Used to decide ingest -> create-new vs. add-to-existing, and to give a
    clear "nothing ingested yet" error from `rag ask`/`rag list` instead of a
    confusing empty-result or Chroma internal error.
    """
    from pathlib import Path

    return (Path(persist_directory) / "chroma.sqlite3").exists()


def index_pdf(path: str, persist_directory: str = PERSIST_DIR) -> Chroma:
    """ingest -> chunk -> embed -> persist, in one call."""
    import warnings

    from chunking import split_recursive, to_documents
    from ingest import load_pdf

    warnings.filterwarnings("ignore")
    docs = to_documents(load_pdf(path))
    chunks = split_recursive(docs)
    return build_vectorstore(chunks, persist_directory=persist_directory)


def _self_check():
    """No Ollama/Chroma round trip here -- that needs a live server and is
    covered by index_pdf() run manually (see README's "Sanity-checking a
    similarity score" section for the actual embedding-backed verification).
    This just checks the module wires the right calls without a live model.
    """
    import inspect

    # confirm Chroma.from_documents is what wires embedding + persistence in
    # one call, i.e. build_vectorstore doesn't skip persistence
    assert "persist_directory" in inspect.signature(build_vectorstore).parameters
    assert "persist_directory" in inspect.signature(load_vectorstore).parameters
    # same collection/persist_directory defaults on both sides, or reload breaks
    assert inspect.signature(build_vectorstore).parameters["persist_directory"].default == PERSIST_DIR
    assert inspect.signature(load_vectorstore).parameters["persist_directory"].default == PERSIST_DIR
    print("ok (wiring checks only; run vectorstore.py <pdf> for a live index)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        store = index_pdf(sys.argv[1])
        print(f"indexed {sys.argv[1]} -> {PERSIST_DIR}")
        results = store.similarity_search(sys.argv[2] if len(sys.argv) > 2 else "python", k=3)
        for r in results:
            print("---", r.metadata, "---")
            print(r.page_content[:160])
    else:
        _self_check()
