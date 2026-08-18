"""PDF -> page records, via LangChain's PyPDFLoader.

PyPDFLoader (pypdf underneath) gives us page text plus source/page metadata, so
that part is not rewritten here. What it does not do is clean the text: every
function below exists because a real PDF in assets/ broke without it. See
docs/pdf-extraction.md for the library comparison and the measured limits.

A page record is a plain dict:
    {
      "source_file": "assets/Python Programming.pdf",  # for citations (issue #10)
      "page_number": 12,                               # 1-based, as a reader sees it
      "text": "...",
      "is_scanned": False,   # no extractable text -> needs OCR
    }
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

# A page yielding fewer than this many characters has no usable text layer.
SCANNED_CHAR_THRESHOLD = 50

# Lines from the top/bottom of each page that are candidates for boilerplate.
BOILERPLATE_EDGE_LINES = 3

# A candidate line must repeat on at least this fraction of pages to be stripped.
BOILERPLATE_PAGE_RATIO = 0.5


def load_pdf(path: str | Path) -> list[dict]:
    """Extract one record per page, cleaned, with repeated boilerplate stripped."""
    path = Path(path)
    source_file = str(path)
    records = []

    for page_number, doc in enumerate(PyPDFLoader(source_file).lazy_load(), start=1):
        text = clean(doc.page_content)
        is_scanned = len(text.strip()) < SCANNED_CHAR_THRESHOLD
        if is_scanned:
            warnings.warn(
                f"{source_file} page {page_number}: no extractable text, likely a "
                f"scan. OCR it separately (see docs/pdf-extraction.md).",
                stacklevel=2,
            )
        records.append(
            {
                "source_file": source_file,
                "page_number": page_number,
                "text": text,
                "is_scanned": is_scanned,
            }
        )

    return strip_boilerplate(records)


def clean(text: str) -> str:
    """Rejoin hyphenated line breaks and drop glyphs pypdf could not map.

    "informa-\\ntion" -> "information". A font with no usable ToUnicode table
    yields "(cid:104)" placeholders; they are noise to an embedding model.
    """
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return re.sub(r"\(cid:\d+\)", "", text)


def strip_boilerplate(records: list[dict]) -> list[dict]:
    """Drop running headers/footers/page numbers that repeat across pages.

    Page numbers differ per page, so compare with digit runs masked out.
    """
    if len(records) < 3:
        return records

    counts = Counter()
    for record in records:
        head, _, tail = _split_edges(record["text"].splitlines())
        counts.update({_mask(line) for line in head + tail if line.strip()})

    threshold = max(3, int(len(records) * BOILERPLATE_PAGE_RATIO))
    boilerplate = {key for key, count in counts.items() if count >= threshold}
    if not boilerplate:
        return records

    for record in records:
        head, middle, tail = _split_edges(record["text"].splitlines())
        head = [line for line in head if _mask(line) not in boilerplate]
        tail = [line for line in tail if _mask(line) not in boilerplate]
        record["text"] = "\n".join(head + middle + tail).strip()
    return records


def _split_edges(lines: list[str]) -> tuple[list[str], list[str], list[str]]:
    """(head, middle, tail) with no overlap even on pages of very few lines."""
    edge = BOILERPLATE_EDGE_LINES
    if len(lines) <= 2 * edge:
        half = len(lines) // 2
        return lines[:half], [], lines[half:]
    return lines[:edge], lines[edge:-edge], lines[-edge:]


def _mask(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip())


def _self_check():
    assert clean("informa-\ntion flow") == "information flow"
    assert clean("well-\nknown") == "wellknown"  # can't tell real hyphens apart
    assert clean("end -\n5") == "end -\n5"  # not a word split, left alone
    assert clean("(cid:104)Question(cid:105)") == "Question"

    bodies = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    pages = [
        {"text": f"Python Programming\n{word} opens the page\nsecond {word} line\n"
                 f"a repeated middle line\nthird {word} line\n{word} closes it\n"
                 f"CHAPTER FOOTER\nPage {i} of 10"}
        for i, word in enumerate(bodies, start=1)
    ]
    first = strip_boilerplate(pages)[0]["text"]
    assert "Python Programming" not in first, first  # running head
    assert "CHAPTER FOOTER" not in first, first
    assert "Page 1 of 10" not in first, first  # page number, matched after masking
    assert "alpha opens the page" in first, first
    assert "alpha closes it" in first, first
    # only head/tail lines are candidates, so a repeat in the body survives
    assert "a repeated middle line" in first, first

    short = [{"text": "HEADER\nx"}, {"text": "HEADER\ny"}]
    assert strip_boilerplate(short) == short  # too few pages to be confident

    # a 4-line page: head and tail must not overlap and duplicate lines
    tiny = [{"text": f"HEADER\nfirst {w}\nsecond {w}\nHEADER"} for w in bodies]
    assert strip_boilerplate(tiny)[0]["text"] == "first alpha\nsecond alpha"

    sample = Path(__file__).resolve().parents[1] / "assets" / "Python Programming.pdf"
    if sample.exists():
        records = load_pdf(sample)
        assert len(records) == 143, len(records)
        assert [r["page_number"] for r in records] == list(range(1, 144))
        assert all(r["source_file"] == str(sample) for r in records)
        assert sum(1 for r in records if r["text"].strip()) > 100
        print(f"ok: {len(records)} pages")
    else:
        print("ok (unit checks only; no sample PDF)")


if __name__ == "__main__":
    # No argument: run the checks. With a PDF path: dump its page records to
    # output.json for eyeballing.
    import json
    import sys

    if len(sys.argv) > 1:
        records = load_pdf(sys.argv[1])
        Path("output.json").write_text(json.dumps(records, indent=2))
        print(f"{len(records)} page records -> output.json")
    else:
        _self_check()
