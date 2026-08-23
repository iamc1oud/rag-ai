"""rag: the entry point wiring issues #1-#6 into one tool.

    rag ingest <path-or-url>   load -> chunk -> embed -> store
    rag ask "<question>"      retrieve -> generate -> print streamed, cited answer
    rag list                  show ingested sources in the vectorstore

Config (Ollama host, model names, chunk_size, k, score threshold): config.py,
one place -- not scattered as constants across ingest/chunking/vectorstore/chain
(each of those now imports its default from settings; see the comments there).

Example session: README.md ("CLI" section).
"""

from __future__ import annotations

import sys
import warnings

import click

from config import settings


def _fail(message: str) -> None:
    """Print a one-line, non-traceback error and exit(1). For failures the
    user causes or can't do anything about from inside this process (Ollama
    down, bad path, empty store) -- not for bugs, which should still raise."""
    click.echo(f"Error: {message}", err=True)
    sys.exit(1)


def _is_url(path_or_url: str) -> bool:
    return path_or_url.startswith("http://") or path_or_url.startswith("https://")


def _check_ollama_reachable() -> None:
    """Fail fast with a clear message instead of a raw ConnectionError deep
    in an embedding or chat call several steps into ingest/ask."""
    import httpx2

    try:
        httpx2.get(settings.OLLAMA_URL, timeout=3.0)
    except httpx2.HTTPError:
        _fail(
            f"Can't reach Ollama at {settings.OLLAMA_URL}. "
            f"Is it running? Try: ollama serve"
        )


@click.group()
def cli():
    """rag: ingest documents and ask grounded, cited questions about them."""


@cli.command()
@click.argument("path_or_url")
@click.option("--chunk-size", default=settings.CHUNK_SIZE, show_default=True)
@click.option("--chunk-overlap", default=settings.CHUNK_OVERLAP, show_default=True)
def ingest(path_or_url: str, chunk_size: int, chunk_overlap: int):
    """Load PATH_OR_URL (a PDF file or a http(s) URL), chunk it, embed it, store it."""
    import warnings as _w

    from pathlib import Path

    from chunking import split_recursive, to_documents
    from ingest import load_pdf, load_url
    from vectorstore import add_documents, build_vectorstore, load_vectorstore, store_exists

    _check_ollama_reachable()

    if _is_url(path_or_url):
        try:
            records = load_url(path_or_url)
        except ConnectionError as e:
            _fail(str(e))
    else:
        path = Path(path_or_url)
        if not path.exists():
            _fail(f"No such file: {path_or_url}")
        if path.suffix.lower() != ".pdf":
            _fail(f"Only .pdf files and http(s) URLs are supported, got: {path_or_url}")
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            records = load_pdf(path)
        for w in caught:
            click.echo(f"Warning: {w.message}", err=True)

    docs = to_documents(records)
    if not docs:
        _fail(f"{path_or_url} produced no extractable text (scanned/empty document?).")

    chunks = split_recursive(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    try:
        if store_exists():
            store = load_vectorstore()
            add_documents(store, chunks)
        else:
            store = build_vectorstore(chunks)
    except Exception as e:
        if "connect" in str(e).lower():
            _fail(f"Can't reach Ollama at {settings.OLLAMA_URL} to embed chunks: {e}")
        raise

    click.echo(f"Ingested {path_or_url}: {len(docs)} document(s) -> {len(chunks)} chunks.")


@cli.command()
@click.argument("question")
@click.option("-k", default=settings.K, show_default=True, help="Chunks to retrieve.")
@click.option("--threshold", default=settings.SCORE_THRESHOLD, show_default=True,
              help="Minimum top-result cosine similarity to attempt an answer.")
def ask(question: str, k: int, threshold: float):
    """Retrieve context for QUESTION and print a streamed, cited answer."""
    from chain import answer_question
    from vectorstore import load_vectorstore, store_exists

    _check_ollama_reachable()

    if not store_exists():
        _fail("Nothing has been ingested yet. Run `rag ingest <path-or-url>` first.")

    store = load_vectorstore()
    if store._collection.count() == 0:
        _fail("The vector store exists but is empty. Run `rag ingest <path-or-url>` first.")

    try:
        for chunk in answer_question(store, question, k=k, threshold=threshold):
            click.echo(chunk, nl=False)
    except Exception as e:
        if "connect" in str(e).lower():
            _fail(f"Can't reach Ollama at {settings.OLLAMA_URL}: {e}")
        raise
    click.echo()


@cli.command(name="list")
def list_sources_cmd():
    """Show every ingested source in the vector store, with its chunk count."""
    from vectorstore import list_sources, load_vectorstore, store_exists

    if not store_exists():
        _fail("Nothing has been ingested yet. Run `rag ingest <path-or-url>` first.")

    store = load_vectorstore()
    sources = list_sources(store)
    if not sources:
        click.echo("No sources ingested yet.")
        return

    total = sum(sources.values())
    click.echo(f"{len(sources)} source(s), {total} chunk(s) total:\n")
    for source, count in sorted(sources.items()):
        click.echo(f"  {count:4d} chunks  {source}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    cli()
