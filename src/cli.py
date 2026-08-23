"""rag: the entry point wiring ingest/chunk/store/chain into one tool.

    rag ingest <path-or-url>   load -> chunk -> embed -> store
    rag ask "<question>"      retrieve -> generate -> print streamed, cited answer
    rag list                  show ingested sources in the vectorstore

Config lives in config.py, one place. Example session: README.md.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import click
import httpx2

from chain import RagChain
from chunking import DocumentBuilder, RecursiveChunker
from config import settings
from ingest import PDFLoader, URLLoader
from vectorstore import VectorStoreService


def _fail(message: str) -> None:
    """One-line, non-traceback error + exit(1). For failures the user causes
    or can't fix from inside this process -- not for bugs, which still raise."""
    click.echo(f"Error: {message}", err=True)
    sys.exit(1)


def _check_ollama_reachable() -> None:
    try:
        httpx2.get(settings.OLLAMA_URL, timeout=3.0)
    except httpx2.HTTPError:
        _fail(f"Can't reach Ollama at {settings.OLLAMA_URL}. Is it running? Try: ollama serve")


def _require_store() -> VectorStoreService:
    service = VectorStoreService()
    if not service.exists:
        _fail("Nothing has been ingested yet. Run `rag ingest <path-or-url>` first.")
    return service


@click.group()
def cli():
    """rag: ingest documents and ask grounded, cited questions about them."""


@cli.command()
@click.argument("path_or_url")
@click.option("--chunk-size", default=settings.CHUNK_SIZE, show_default=True)
@click.option("--chunk-overlap", default=settings.CHUNK_OVERLAP, show_default=True)
def ingest(path_or_url: str, chunk_size: int, chunk_overlap: int):
    """Load PATH_OR_URL (a PDF file or a http(s) URL), chunk it, embed it, store it."""
    _check_ollama_reachable()

    is_url = path_or_url.startswith("http://") or path_or_url.startswith("https://")
    if is_url:
        try:
            records = URLLoader().load(path_or_url)
        except ConnectionError as e:
            _fail(str(e))
    else:
        path = Path(path_or_url)
        if not path.exists():
            _fail(f"No such file: {path_or_url}")
        if path.suffix.lower() != ".pdf":
            _fail(f"Only .pdf files and http(s) URLs are supported, got: {path_or_url}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            records = PDFLoader().load(path)
        for w in caught:
            click.echo(f"Warning: {w.message}", err=True)

    docs = DocumentBuilder.from_records(records)
    if not docs:
        _fail(f"{path_or_url} produced no extractable text (scanned/empty document?).")

    chunks = RecursiveChunker(chunk_size, chunk_overlap).split(docs)

    try:
        VectorStoreService().ingest(chunks)
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
    _check_ollama_reachable()
    service = _require_store()
    store = service.load()
    if store._collection.count() == 0:
        _fail("The vector store exists but is empty. Run `rag ingest <path-or-url>` first.")

    chain = RagChain(store, k=k, threshold=threshold)
    try:
        for chunk in chain.answer(question):
            click.echo(chunk, nl=False)
    except Exception as e:
        if "connect" in str(e).lower():
            _fail(f"Can't reach Ollama at {settings.OLLAMA_URL}: {e}")
        raise
    click.echo()


@cli.command(name="list")
def list_sources_cmd():
    """Show every ingested source in the vector store, with its chunk count."""
    service = _require_store()
    sources = service.list_sources()
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
