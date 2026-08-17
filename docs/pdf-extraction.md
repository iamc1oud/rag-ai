# PDF text extraction: library comparison and choice

**Decision: pdfplumber, with our own reading-order pass on top** (`ingest/pdf_loader.py`).

## What was measured

Two real PDFs, both already messy in different ways:

| File | Shape | Why it stresses the loader |
| --- | --- | --- |
| `assets/Python Programming.pdf` | 143pp LaTeX book, single column | Running header + page number on every page, 8 image-only pages, a borderless table, LaTeX fonts that emit no space characters |
| `assets/bert-two-column.pdf` | 16pp ACL paper (BERT, arXiv 1810.04805) | Two columns with a 17pt gutter, full-width figures mid-page, footnotes, `(cid:NNN)` unmapped glyphs |

Numbers below are from this machine (M-series Mac, Python 3.13), `python /tmp/bench.py`:

| Library | BERT (16pp) | Book (143pp) | Two-column reading order | Tables | Images |
| --- | --- | --- | --- | --- | --- |
| pypdf 6.x | 1.92s / 64k chars | 0.50s / 122k | Correct **by luck** — emits content-stream order | None | Can list/extract XObjects |
| PyMuPDF (pymupdf) | 1.44s / 64k | 0.12s / 120k | Correct by luck, same reason | `page.find_tables()` | Yes, with bboxes |
| PyMuPDF `get_text(sort=True)` | 0.22s / 89k | 0.46s / 144k | **Wrong** — sorts by y, interleaves the columns | same | same |
| pdfplumber 0.11 | 1.20s / 61k | 1.33s / 116k | **Wrong** by default — sorts by `(top, x0)` | `extract_tables()`, lines *or* text strategy | `page.images` metadata |
| unstructured 0.19 (`strategy="fast"`) | 6.59s / 64k, 654 elements | 6.15s / 117k, 2069 elements | Same pdfminer order as pdfplumber | With `hi_res` only | With `hi_res` only |
| **ours (pdfplumber + reading-order pass)** | 0.81s / 64k | 1.41s / 118k | **Correct by construction** | via pdfplumber | detects image-only pages |

## Why pdfplumber

1. **Word-level geometry is the whole job.** Every messy case in this issue is a
   geometry problem, not a text problem: columns need x-gaps, headers need y-position,
   scans need "are there glyphs at all". pdfplumber's `extract_words()` hands back
   `x0/x1/top/bottom` per word, which is exactly the input the reading-order pass needs.
   pypdf exposes no coordinates at all, so it is disqualified for anything past
   single-column prose.
2. **Do not trust order that works by accident.** pypdf and PyMuPDF read the BERT
   columns correctly — but only because that PDF's content stream happens to be
   written column by column. Nothing in the format guarantees that, and PyMuPDF's own
   `sort=True` (the flag you would reach for to *fix* order) actively breaks it. A
   detected gutter is a property of the page; content-stream order is a property of
   whichever tool wrote the file.
3. **Licence.** PyMuPDF is AGPL-3.0 or a paid commercial licence; pdfplumber is MIT.
   Not worth an AGPL dependency for a 3x speed win on a batch job that runs offline.
4. **unstructured is the wrong altitude.** Its useful part — layout-aware element
   typing (Title / NarrativeText / ListItem) — needs `strategy="hi_res"` and a
   detection model; the `fast` strategy is pdfminer underneath, i.e. pdfplumber with
   more dependencies and 5x the runtime. It also decides chunking for us, which is
   issue #3's job. Revisit if scanned-document layout detection becomes the bottleneck.

Speed is not the deciding factor: 1.4s for a 143-page book is ~10ms/page, and ingestion
runs once per document while embedding dominates the pipeline cost.

## What the loader does about each messy case

`load_pdf(path)` returns one record per page:

```python
{"source_file": "assets/bert-two-column.pdf", "page_number": 4,
 "text": "...", "is_scanned": False, "columns": 2}
```

`source_file` + `page_number` are the citation key for issue #10; page numbers are
1-based so they match what a reader sees.

### Multi-column reading order
`_find_gutter()` groups words into rows, then looks for a vertical gap that *most
full-width rows share*. Per-row instead of per-page because a single full-width figure
would otherwise bridge the gutter and hide it — that is exactly what happened on BERT
p4 during development. Guards against false positives, each one added after a real
misfire on these two files:

- gap ≥ 2% of page width (a real ACL gutter is 17pt/595pt ≈ 3%; word spacing is 3-5pt)
- only rows spanning ≥ 70% of the text width vote, so a narrow borderless table is not
  mistaken for a column split (book p82 was being split and reordered before this)
- the gap must appear in ≥ 8 rows and ≥ 35% of rows, and those positions must cluster

Result: 7 of 16 BERT pages detected as two-column, 0 of 143 book pages. Rows where a
single word straddles the gutter are treated as full-width and kept with the left column.

Known ceiling: two columns only. Three-column layouts, or columns that start partway
down the page, need a recursive XY-cut.

### Scanned / image-only pages
A page with `< 50` extractable characters *and* at least one image is flagged
`is_scanned: True` and warns rather than failing. Eight such pages in the book (chapter
dividers that are a figure plus a title). To plug in OCR, pass a callable:

```python
import pytesseract
load_pdf(path, ocr=lambda page: pytesseract.image_to_string(
    page.to_image(resolution=300).original))
```

That needs the `tesseract` binary plus `pytesseract`, so it stays optional and out of
`pyproject.toml` until the corpus actually contains scans.

### Repeated headers / footers / page numbers
`strip_boilerplate()` looks only at the first and last 3 lines of each page, masks digit
runs (`Page 12 of 143` → `Page # of #`) so page numbers collapse to one key, and drops
lines appearing on ≥ 50% of pages. Confirmed on the book: `Python Programming` and the
bare page number are gone, body text intact.

Ceiling: masking digits means two *different* short numeric lines can collide, so only
head/tail lines are ever candidates — a repeated line in the body survives on purpose.

### Broken hyphenation
`clean()` rejoins `informa-\ntion` → `information`. It cannot distinguish a
line-break hyphen from a real one, so `well-\nknown` becomes `wellknown`. Accepted:
mis-joining a genuine compound costs one token's accuracy, whereas leaving the split
costs the word entirely at embedding time.

`clean()` also strips `(cid:NNN)` placeholders, which pdfminer emits for fonts with no
usable ToUnicode table (BERT p4 has several).

### Missing spaces
LaTeX PDFs often encode no space characters, and pdfplumber's default 3pt word gap then
glues phrases together — `Throughoutthiswork`. `X_TOLERANCE = 1.5` fixes it on both
files; measured on BERT p4 across tolerances 3 → 0.5, everything ≤ 2 is correct.

### Tables
Not extracted as structure yet — table text lands in the page text. When it matters,
pdfplumber handles both kinds, but the strategy is not automatic: the borderless table
on book p82 needs the text strategy, since the default looks for ruling lines.

```python
page.extract_tables({"vertical_strategy": "text", "horizontal_strategy": "text"})
# -> [[['Time', 'Value'], ['1', '22'], ['2', '25'], ['3', '28']]]
```

## Running it

```bash
python ingest/pdf_loader.py                      # self-checks (asserts, no framework)
python ingest/pdf_loader.py assets/some.pdf      # dump page records to output.json
```
