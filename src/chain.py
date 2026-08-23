"""Retriever + ChatOllama -> grounded, cited, streamed answers (LCEL).

Threshold pick and the 3 manually-verified test questions: README.md
("RAG chain" section).
"""

from __future__ import annotations
from langchain_community.vectorstores import VectorStore

from collections.abc import Iterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import settings

# Below this cosine similarity, the top retrieved chunk isn't actually about
# the question -- skip generation rather than let the model improvise on
# unrelated context. Picked from measured scores on this corpus (README):
# on-topic questions scored 0.62-0.70, off-topic 0.21-0.35. 0.45 sits in the
# gap with margin on both sides; it is a property of this embedding model and
# corpus, not a universal constant. Lives in config.py (settings.SCORE_THRESHOLD)
# per issue #7; aliased here so existing imports keep working.
RELEVANCE_THRESHOLD = settings.SCORE_THRESHOLD

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


def get_chat() -> ChatOllama:
    return ChatOllama(model=settings.CHAT_MODEL, base_url=settings.OLLAMA_URL, temperature=0)


def get_retriever(store, k: int = settings.K):
    return store.as_retriever(search_kwargs={"k": k})


def format_docs(docs: list[Document]) -> str:
    """Numbered context blocks the prompt asks the model to cite by number."""
    return "\n\n".join(
        f"[{i}] (source: {doc.metadata.get('source')}, page {doc.metadata.get('page')})\n"
        f"{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )


def format_citations(docs: list[Document]) -> str:
    """Sources list keyed by the same [n] numbers used in format_docs/the prompt."""
    lines = [
        f"[{i}] {doc.metadata.get('source')}, page {doc.metadata.get('page')}"
        for i, doc in enumerate(docs, start=1)
    ]
    return "Sources:\n" + "\n".join(lines)


def build_chain(chat: ChatOllama | None = None):
    """retriever-independent LCEL chain: formatted context + question -> answer.

    Retrieval and the relevance threshold are handled by answer_question()
    below, not inside this chain -- the threshold has to gate whether the LLM
    is called *at all*, which an LCEL step run after retrieval can't cheaply
    veto once it's already wired into the same pipeline.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{question}")]
    )
    return prompt | (chat or get_chat()) | StrOutputParser()


def answer_question(
    store: VectorStore,
    question: str,
    k: int = settings.K,
    threshold: float = RELEVANCE_THRESHOLD,
    chat: ChatOllama | None = None,
) -> Iterator[str]:
    """Retrieve -> threshold-gate -> stream a grounded, cited answer.

    Yields the answer text in chunks (chain.stream), then one final chunk
    with the citation list. Yields NO_CONTEXT_MESSAGE and returns, without
    calling the chat model, if the top result is below `threshold` -- avoids
    generating a fabricated answer over irrelevant context.
    """
    scored = store.similarity_search_with_relevance_scores(question, k=k)
    if not scored or scored[0][1] < threshold:
        yield NO_CONTEXT_MESSAGE
        return

    docs = [doc for doc, _score in scored]
    chain = build_chain(chat)
    context = format_docs(docs)

    for token in chain.stream({"context": context, "question": question}):
        yield token

    yield "\n\n" + format_citations(docs)


def ask(store, question: str, **kwargs) -> str:
    """Non-streaming convenience wrapper: collect answer_question() into one string."""
    return "".join(answer_question(store, question, **kwargs))


def _self_check():
    """Wiring only -- no live Ollama/Chroma call. See README for the 3
    manually-verified live questions (2 grounded+cited, 1 out-of-scope)."""
    docs = [
        Document(page_content="Python was first released in 1991.",
                 metadata={"source": "book.pdf", "page": 5}),
        Document(page_content="CPython is the reference implementation.",
                 metadata={"source": "book.pdf", "page": 6}),
    ]
    context = format_docs(docs)
    assert "[1] (source: book.pdf, page 5)" in context
    assert "[2] (source: book.pdf, page 6)" in context
    assert "Python was first released in 1991." in context

    citations = format_citations(docs)
    assert citations == "Sources:\n[1] book.pdf, page 5\n[2] book.pdf, page 6"

    class FakeStore:
        def similarity_search_with_relevance_scores(self, question, k):
            return [(docs[0], 0.1)]  # below RELEVANCE_THRESHOLD

    out = ask(FakeStore(), "irrelevant question")
    assert out == NO_CONTEXT_MESSAGE, out

    print("ok (wiring checks only; run chain.py <question> for a live answer)")


if __name__ == "__main__":
    import sys
    import warnings

    warnings.filterwarnings("ignore")

    if len(sys.argv) > 1:
        from vectorstore import load_vectorstore

        question = " ".join(sys.argv[1:])
        store = load_vectorstore()
        for chunk in answer_question(store, question):
            print(chunk, end="", flush=True)
        print()
    else:
        _self_check()
