"""PageRecord -> retrieval-sized Document chunks. Splitter comparison: README.md."""

from __future__ import annotations

from abc import ABC, abstractmethod

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from config import settings
from ingest import PageRecord


class DocumentBuilder:
    @staticmethod
    def from_records(records: list[PageRecord]) -> list[Document]:
        """Drops scanned/empty pages. metadata carries the citation key (source, page)."""
        return [
            Document(page_content=r.text, metadata={"source": r.source, "page": r.page})
            for r in records
            if r.text.strip() and not r.is_scanned
        ]


class TextChunker(ABC):
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE,
                 chunk_overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split(self, docs: list[Document]) -> list[Document]: ...


class RecursiveChunker(TextChunker):
    """Default: paragraph -> line -> sentence -> word -> character.

    Each separator is tried in order, falling through to a finer one only if
    the chunk is still over chunk_size. ". " is included so it prefers a
    sentence boundary over a mid-sentence cut.
    """

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def split(self, docs: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
            separators=self.SEPARATORS,
        )
        return splitter.split_documents(docs)


class NaiveChunker(TextChunker):
    """Hard cut every chunk_size characters. Comparison baseline, not for real use."""

    def split(self, docs: list[Document]) -> list[Document]:
        splitter = CharacterTextSplitter(
            separator="", chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(docs)


class TokenChunker(RecursiveChunker):
    """Recursive splitter sized in tokens, not characters -- embedding/context
    limits are token-based, and character count is a poor proxy on dense text.
    """

    ENCODING = tiktoken.get_encoding("cl100k_base")

    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 40):
        super().__init__(chunk_size, chunk_overlap)

    @classmethod
    def count_tokens(cls, text: str) -> int:
        return len(cls.ENCODING.encode(text))

    def split(self, docs: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
            length_function=self.count_tokens, separators=self.SEPARATORS,
        )
        return splitter.split_documents(docs)


def _stats(chunks: list[Document]) -> dict:
    sizes = [len(c.page_content) for c in chunks]
    return {
        "count": len(chunks),
        "avg_chars": round(sum(sizes) / len(sizes)) if sizes else 0,
        "min_chars": min(sizes) if sizes else 0,
        "max_chars": max(sizes) if sizes else 0,
    }
