### RAG

We will be using model `gemma4:31b-cloud` for agent.

# Embedding Model
`nomic-embed-text-v2-moe` is a multilingual MoE text embedding model that excels at multilingual retrieval.

- High Performance: SoTA Multilingual performance compared to ~300M parameter models, competitive with models 2x in size
- Multilinguality: Supports ~100 languages and trained on over 1.6B pairs
- Flexible Embedding Dimension: Trained with Matryoshka Embeddings with 3x reductions in storage cost with minimal performance degradations
- Fully Open-Source: Model weights, code, and training data

# PDF Ingestion

`ingest/lang_pdf_loader.py`. `load_pdf(path)` returns one record per page:

```python
{"source_file": "assets/bert-two-column.pdf", "page_number": 4,
 "text": "...", "is_scanned": False}
```

`source_file` + `page_number` are the citation key for issue #10. Page numbers are
1-based so they match what a reader sees.

```bash
python ingest/lang_pdf_loader.py                   # self-checks
python ingest/lang_pdf_loader.py assets/some.pdf   # dump page records to output.json
```

Text extraction is LangChain's `PyPDFLoader` (pypdf underneath) — no reason to
hand-roll page iteration and metadata. Library comparison with measured numbers:
`docs/pdf-extraction.md`.

## The three test PDFs

Deliberately different kinds of messy:

| File | What it is | Result |
| --- | --- | --- |
| `Python Programming.pdf` | 143pp LaTeX book, single column, clean | 121k chars, 0.56s, 16 pages flagged as having no text layer |
| `bert-two-column.pdf` | 16pp ACL paper (BERT, arXiv 1810.04805), two columns | 63k chars, 0.21s, columns read in the right order |
| `scanned-newspaper-1863.pdf` | 4pp scan of an 1863 newspaper (archive.org), ~6 columns of OCR'd newsprint | 229k chars, 1.00s, readable per story |

## Messy cases: what was found and what was done

| Case | Found in | Handled? |
| --- | --- | --- |
| Repeated running header + page number | book, every page | **Yes.** `strip_boilerplate()` |
| Hyphenation across line breaks | book, BERT | **Yes.** `clean()` |
| Unmapped glyphs (`(cid:NNN)`) | pdfminer output for BERT | **Yes**, stripped in `clean()` — though pypdf maps most of them properly anyway (`⟨Question, Answer⟩` came through intact) |
| Pages with no text layer | book, 16 pages (chapter dividers that are one figure) | **Detected**, not fixed: flagged `is_scanned` with a warning |
| Multi-column reading order | BERT (2 col), newspaper (~6 col) | **Yes, for free.** pypdf emits content-stream order, which in both files is column-by-column |
| OCR noise | newspaper: `casualty ocoarred`, `hie reid`, `&6.` | **No.** See below |
| Tables as structure | book p82 (borderless), BERT | **No.** Table text lands in the page text, unstructured |
| Multi-column when stream order is *wrong* | not seen in these three | **No.** See below |

### Not handled, and why

**OCR noise.** The 1863 scan's text layer is archive.org's OCR, and it is wrong in
ways nothing downstream can undo: `casualty ocoarred on our side`, `Colonel Kilpatrick
bas been entirely ceseful in hie reid`. Fixing this means re-OCRing at higher
resolution or spell-correcting against a period lexicon — a project of its own, and
retrieval degrades gracefully since embeddings still match on the correct words in the
same passage. Left as-is, deliberately.

**Pages with no text layer.** Flagged, not OCR'd. `pytesseract` plus the `tesseract`
binary would fix it, but 16 of 143 pages in one book are chapter dividers whose only
content is a figure and a title already in the ToC, so there is nothing to retrieve.
Add OCR when a document in the corpus is *entirely* scanned without a text layer.

**Tables.** pypdf has no table support at all. pdfplumber does, and the recipe is in
`docs/pdf-extraction.md` — the borderless table on book p82 needs its text strategy,
not the default line-detection one. Deferred until a question actually needs a cell
value; flowing table text into the page is fine for prose retrieval.

**Multi-column when the content stream is wrong.** Both multi-column files here read
correctly because their producers wrote the text column by column — that is a property
of the writing tool, not a guarantee of the format. pypdf exposes no word coordinates,
so if a PDF turns up whose stream order interleaves columns, this loader cannot fix it
and the text will be scrambled. That is the accepted cost of pypdf over a
geometry-aware extractor; `docs/pdf-extraction.md` records what the alternative buys.
