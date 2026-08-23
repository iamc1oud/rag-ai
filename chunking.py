"""Page records (ingest.py) -> retrieval-sized Document chunks.

Comparison of splitters and the chosen chunk_size/overlap: README.md.
"""

from __future__ import annotations

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from config import settings

# Chosen defaults -- reasoning in README.md ("Chunk size and overlap"). The
# char-based ones live in config.py (settings.CHUNK_SIZE/CHUNK_OVERLAP) since
# issue #7 wants chunk_size configured in one place, not as a module constant
# here; aliased locally so existing callers/imports don't break.
CHUNK_SIZE_CHARS = settings.CHUNK_SIZE
CHUNK_OVERLAP_CHARS = settings.CHUNK_OVERLAP
CHUNK_SIZE_TOKENS = 250
CHUNK_OVERLAP_TOKENS = 40

# cl100k_base is the GPT-3.5/4 tokenizer. Our chat/embedding models are served
# by Ollama and tokenize differently, but there is no public tiktoken encoding
# for them; cl100k_base is used as a stand-in token counter so chunk_size means
# "roughly how many tokens" instead of raw characters. See README for why this
# matters even though it isn't the exact tokenizer in the loop.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def to_documents(records: list[dict]) -> list[Document]:
    """ingest.load_pdf() records -> Documents, dropping pages with no real text.

    metadata carries exactly what issue #10 (citation tracking) needs.
    """
    return [
        Document(
            page_content=r["text"],
            metadata={"source": r["source_file"], "page": r["page_number"]},
        )
        for r in records
        if r["text"].strip() and not r["is_scanned"]
    ]


def split_recursive(
    docs: list[Document],
    chunk_size: int = CHUNK_SIZE_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[Document]:
    """Default splitter: paragraph -> line -> sentence -> word -> character.

    Each separator is tried in order; a split only falls through to the next,
    finer one if the chunk is still over chunk_size. ". " is added to the
    default separator list so it prefers a sentence boundary over a mid-sentence
    word wrap before resorting to a hard character cut.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def split_naive(
    docs: list[Document],
    chunk_size: int = CHUNK_SIZE_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[Document]:
    """CharacterTextSplitter with separator="": a hard cut every chunk_size
    characters, blind to word or sentence boundaries. This is the baseline
    the comparison in README is against, not something to use for real chunks.
    """
    splitter = CharacterTextSplitter(
        separator="", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def split_by_tokens(
    docs: list[Document],
    chunk_size: int = CHUNK_SIZE_TOKENS,
    chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[Document]:
    """Recursive splitter, but chunk_size/overlap count tokens, not characters.

    Embedding and context-window limits are token-based, so a document near
    the character-based chunk_size can still silently blow a token budget on
    dense text (code, non-English, LaTeX). length_function makes the size
    check exact instead of an approximation.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def chunk_pdf(path: str, chunk_size: int = CHUNK_SIZE_CHARS,
              chunk_overlap: int = CHUNK_OVERLAP_CHARS) -> list[Document]:
    """ingest -> chunk in one call, using the chosen default splitter."""
    from ingest import load_pdf

    docs = to_documents(load_pdf(path))
    return split_recursive(docs, chunk_size, chunk_overlap)


def _stats(chunks: list[Document]) -> dict:
    sizes = [len(c.page_content) for c in chunks]
    return {
        "count": len(chunks),
        "avg_chars": round(sum(sizes) / len(sizes)) if sizes else 0,
        "min_chars": min(sizes) if sizes else 0,
        "max_chars": max(sizes) if sizes else 0,
    }
