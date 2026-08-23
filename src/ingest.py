"""Document loaders -> PageRecord. Library choice and messy-PDF handling: README.md."""

from __future__ import annotations

import re
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import httpx2
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader


@dataclass
class PageRecord:
    source: str
    page: int
    text: str
    is_scanned: bool = False


class BoilerplateStripper:
    """Drops running headers/footers/page numbers repeated across pages."""

    EDGE_LINES = 3
    PAGE_RATIO = 0.5

    @classmethod
    def strip(cls, records: list[PageRecord]) -> list[PageRecord]:
        if len(records) < 3:
            return records

        counts = Counter()
        for record in records:
            head, _, tail = cls._split_edges(record.text.splitlines())
            counts.update({cls._mask(line) for line in head + tail if line.strip()})

        threshold = max(3, int(len(records) * cls.PAGE_RATIO))
        boilerplate = {key for key, count in counts.items() if count >= threshold}
        if not boilerplate:
            return records

        for record in records:
            head, middle, tail = cls._split_edges(record.text.splitlines())
            head = [line for line in head if cls._mask(line) not in boilerplate]
            tail = [line for line in tail if cls._mask(line) not in boilerplate]
            record.text = "\n".join(head + middle + tail).strip()
        return records

    @classmethod
    def _split_edges(cls, lines: list[str]) -> tuple[list[str], list[str], list[str]]:
        edge = cls.EDGE_LINES
        if len(lines) <= 2 * edge:
            half = len(lines) // 2
            return lines[:half], [], lines[half:]
        return lines[:edge], lines[edge:-edge], lines[-edge:]

    @staticmethod
    def _mask(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip())


class DocumentLoader(ABC):
    SCANNED_CHAR_THRESHOLD = 50

    @abstractmethod
    def load(self, source: str) -> list[PageRecord]: ...

    @staticmethod
    def clean(text: str) -> str:
        """Rejoin hyphenated line breaks; drop glyphs the parser couldn't map."""
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        return re.sub(r"\(cid:\d+\)", "", text)


class PDFLoader(DocumentLoader):
    def load(self, path: str | Path) -> list[PageRecord]:
        source = str(path)
        records = []
        for page, doc in enumerate(PyPDFLoader(source).lazy_load(), start=1):
            text = self.clean(doc.page_content)
            is_scanned = len(text.strip()) < self.SCANNED_CHAR_THRESHOLD
            if is_scanned:
                warnings.warn(
                    f"{source} page {page}: no extractable text, likely a scan. "
                    f"OCR it separately.",
                    stacklevel=2,
                )
            records.append(PageRecord(source=source, page=page, text=text, is_scanned=is_scanned))
        return BoilerplateStripper.strip(records)


class URLLoader(DocumentLoader):
    """Fetches a web page as a single PageRecord (page=1, no pagination)."""

    NOISE_TAGS = ("script", "style", "nav", "header", "footer", "aside", "form")

    def load(self, url: str, timeout: float = 15.0) -> list[PageRecord]:
        try:
            response = httpx2.get(
                url, timeout=timeout, follow_redirects=True,
                headers={"User-Agent": "rag-ai/0.1"},
            )
            response.raise_for_status()
        except httpx2.HTTPError as e:
            raise ConnectionError(f"Could not fetch {url}: {e}") from e

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup.find_all(self.NOISE_TAGS):
            tag.decompose()

        text = self.clean(soup.get_text(separator="\n"))
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        is_scanned = len(text) < self.SCANNED_CHAR_THRESHOLD
        if is_scanned:
            warnings.warn(f"{url}: page yielded almost no text after stripping markup.",
                          stacklevel=2)
        return [PageRecord(source=url, page=1, text=text, is_scanned=is_scanned)]


def _self_check():
    assert DocumentLoader.clean("informa-\ntion flow") == "information flow"
    assert DocumentLoader.clean("well-\nknown") == "wellknown"
    assert DocumentLoader.clean("end -\n5") == "end -\n5"
    assert DocumentLoader.clean("(cid:104)Question(cid:105)") == "Question"

    bodies = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    pages = [
        PageRecord(source="s", page=i,
                   text=f"Python Programming\n{word} opens the page\nsecond {word} line\n"
                        f"a repeated middle line\nthird {word} line\n{word} closes it\n"
                        f"CHAPTER FOOTER\nPage {i} of 10")
        for i, word in enumerate(bodies, start=1)
    ]
    first = BoilerplateStripper.strip(pages)[0].text
    assert "Python Programming" not in first, first
    assert "CHAPTER FOOTER" not in first, first
    assert "Page 1 of 10" not in first, first
    assert "alpha opens the page" in first and "alpha closes it" in first
    assert "a repeated middle line" in first

    short = [PageRecord(source="s", page=1, text="HEADER\nx"),
             PageRecord(source="s", page=2, text="HEADER\ny")]
    assert [r.text for r in BoilerplateStripper.strip(short)] == [r.text for r in short]

    tiny = [PageRecord(source="s", page=i, text=f"HEADER\nfirst {w}\nsecond {w}\nHEADER")
            for i, w in enumerate(bodies)]
    assert BoilerplateStripper.strip(tiny)[0].text == "first alpha\nsecond alpha"

    html = """
    <html><body>
      <nav>Home | About | Contact</nav>
      <script>trackPageView();</script>
      <article><p>LangChain is a framework for building context-aware apps.</p></article>
      <footer>Copyright 2026</footer>
    </body></html>
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(URLLoader.NOISE_TAGS):
        tag.decompose()
    stripped = soup.get_text()
    assert "LangChain is a framework" in stripped
    assert "Home | About | Contact" not in stripped
    assert "trackPageView" not in stripped
    assert "Copyright 2026" not in stripped

    sample = Path(__file__).resolve().parents[1] / "assets" / "Python Programming.pdf"
    if sample.exists():
        records = PDFLoader().load(sample)
        assert len(records) == 143, len(records)
        assert [r.page for r in records] == list(range(1, 144))
        assert all(r.source == str(sample) for r in records)
        assert sum(1 for r in records if r.text.strip()) > 100
        print(f"ok: {len(records)} pages")
    else:
        print("ok (unit checks only; no sample PDF)")


if __name__ == "__main__":
    import json
    import sys
    from dataclasses import asdict

    if len(sys.argv) > 1:
        records = PDFLoader().load(sys.argv[1])
        Path("output.json").write_text(json.dumps([asdict(r) for r in records], indent=2))
        print(f"{len(records)} page records -> output.json")
    else:
        _self_check()
