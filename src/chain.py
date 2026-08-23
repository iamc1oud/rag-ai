"""Retriever + ChatOllama -> grounded, cited, streamed answers (LCEL).

Threshold pick and the 3 manually-verified test questions: README.md.
"""

from __future__ import annotations

from collections.abc import Iterator

from langchain_community.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import settings


class RagChain:
    """Threshold-gated retrieval + generation over one vector store.

    Below RELEVANCE_THRESHOLD, the top retrieved chunk isn't actually about
    the question -- generation is skipped rather than let the model improvise
    on unrelated context. 0.45 sits in the measured gap between on-topic
    (0.62-0.70) and off-topic (0.21-0.35) scores on this corpus (README);
    it's a property of the embedding model and corpus, not universal.
    """

    NO_CONTEXT_MESSAGE = "I don't know -- no relevant context found for that question."

    SYSTEM_PROMPT = """You are a RAG assistant. Answer the question using ONLY the \
numbered context below -- never use outside knowledge, even if you know the answer.

Every context block is labeled with a number, e.g. "[1]". When you use \
information from a block, cite its number inline right after the sentence \
that uses it, like "...as shown in the example [1]." Cite every block you \
draw on; a sentence can cite more than one, e.g. "[1][2]".

If the context does not contain enough information to answer the question, \
respond with exactly: I don't know -- no relevant context found for that question.

Context:
{context}"""

    def __init__(self, store: VectorStore, chat: ChatOllama | None = None,
                 k: int = settings.K, threshold: float = settings.SCORE_THRESHOLD):
        self.store = store
        self.chat = chat or ChatOllama(
            model=settings.CHAT_MODEL, base_url=settings.OLLAMA_URL, temperature=0
        )
        self.k = k
        self.threshold = threshold
        prompt = ChatPromptTemplate.from_messages(
            [("system", self.SYSTEM_PROMPT), ("human", "{question}")]
        )
        self._chain = prompt | self.chat | StrOutputParser()

    @staticmethod
    def format_docs(docs: list[Document]) -> str:
        """Numbered context blocks the prompt asks the model to cite by number."""
        return "\n\n".join(
            f"[{i}] (source: {doc.metadata.get('source')}, page {doc.metadata.get('page')})\n"
            f"{doc.page_content}"
            for i, doc in enumerate(docs, start=1)
        )

    @staticmethod
    def format_citations(docs: list[Document]) -> str:
        lines = [
            f"[{i}] {doc.metadata.get('source')}, page {doc.metadata.get('page')}"
            for i, doc in enumerate(docs, start=1)
        ]
        return "Sources:\n" + "\n".join(lines)

    def generate(self, question: str, docs: list[Document]) -> str:
        """Generate over a caller-chosen set of docs, bypassing the threshold gate."""
        return self._chain.invoke({"context": self.format_docs(docs), "question": question})

    def answer(self, question: str, k: int | None = None,
               threshold: float | None = None) -> Iterator[str]:
        """Retrieve -> threshold-gate -> stream a grounded, cited answer.

        Yields NO_CONTEXT_MESSAGE and returns, without calling the chat model,
        if the top result is below threshold.
        """
        k = k if k is not None else self.k
        threshold = threshold if threshold is not None else self.threshold

        scored = self.store.similarity_search_with_relevance_scores(question, k=k)
        if not scored or scored[0][1] < threshold:
            yield self.NO_CONTEXT_MESSAGE
            return

        docs = [doc for doc, _score in scored]
        context = self.format_docs(docs)
        for token in self._chain.stream({"context": context, "question": question}):
            yield token
        yield "\n\n" + self.format_citations(docs)

    def ask(self, question: str, **kwargs) -> str:
        """Non-streaming convenience wrapper: collect answer() into one string."""
        return "".join(self.answer(question, **kwargs))


def _self_check():
    """Wiring only -- no live Ollama/Chroma call. See README for the 3
    manually-verified live questions (2 grounded+cited, 1 out-of-scope)."""
    docs = [
        Document(page_content="Python was first released in 1991.",
                 metadata={"source": "book.pdf", "page": 5}),
        Document(page_content="CPython is the reference implementation.",
                 metadata={"source": "book.pdf", "page": 6}),
    ]
    context = RagChain.format_docs(docs)
    assert "[1] (source: book.pdf, page 5)" in context
    assert "[2] (source: book.pdf, page 6)" in context
    assert "Python was first released in 1991." in context

    citations = RagChain.format_citations(docs)
    assert citations == "Sources:\n[1] book.pdf, page 5\n[2] book.pdf, page 6"

    class FakeStore:
        def similarity_search_with_relevance_scores(self, question, k):
            return [(docs[0], 0.1)]  # below threshold

    out = RagChain(FakeStore()).ask("irrelevant question")
    assert out == RagChain.NO_CONTEXT_MESSAGE, out

    print("ok (wiring checks only; run chain.py <question> for a live answer)")


if __name__ == "__main__":
    import sys
    import warnings

    warnings.filterwarnings("ignore")

    if len(sys.argv) > 1:
        from vectorstore import VectorStoreService

        question = " ".join(sys.argv[1:])
        store = VectorStoreService().load()
        for chunk in RagChain(store).answer(question):
            print(chunk, end="", flush=True)
        print()
    else:
        _self_check()
