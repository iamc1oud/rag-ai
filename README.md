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

# Chunking

`chunking.py`. `to_documents(records)` turns `ingest.load_pdf()` output into
`langchain_core.documents.Document`s (dropping scanned/empty pages), then one of three
splitters cuts them into retrieval-sized chunks. `metadata={"source", "page"}` is set
once on the parent Document and LangChain's `split_documents()` copies it onto every
chunk unchanged — confirmed in `_self_check()`.

```bash
python chunking.py                              # self-checks
python chunking.py "assets/Python Programming.pdf"   # chunk a real PDF, print stats
```

## Splitters compared

Run over the book's 127 non-scanned pages (120,997 chars / 35,707 cl100k tokens):

| Splitter | chunk_size | count | avg chars | max chars | avg tokens | max tokens |
| --- | --- | --- | --- | --- | --- | --- |
| `CharacterTextSplitter` (naive, `separator=""`) | 1000 chars | 195 | 673 | 1000 | 200 | 465 |
| `RecursiveCharacterTextSplitter` (chosen) | 1000 chars | 195 | 659 | 1000 | 196 | 455 |
| `RecursiveCharacterTextSplitter` + `length_function=tiktoken` | 250 tokens | 223 | 585 | 1256 | 174 | 249 |

Chunk counts and average size are close between naive and recursive at the same
`chunk_size` — the difference isn't volume, it's *where* the cut falls. That's the
actual reason to prefer recursive: same budget, boundaries that respect the text.

Note the char-based row's max_tokens (455) is nearly double its average (196) — chunk
size in characters is a poor proxy for token count on real text (see "Why tokens"
below). The token-based row holds max_tokens to 249, just over its 250 budget, because
tiktoken is being measured directly instead of estimated from character count.

## Naive vs. recursive: a concrete mid-sentence cut

Page 23 of the book, `chunk_size=124`, both with `chunk_overlap=0`, same source text:

> "With interpreted languages, the code is saved in the same format that you entered.
> Compiled programs generally run faster than interpreted ones because interpreted
> programs must be reduced to machine instructions at run-time."

**`CharacterTextSplitter` (naive, `separator=""`)** — cuts at character 124 with no
regard for what's there:
```
chunk[0]: "...Compiled programs generally run faster th"
chunk[1]: "an interpreted ones because\ninterpreted programs must be red"
```
`th|an` and (looking at the next boundary) `red|uced` are both split mid-word.

**`RecursiveCharacterTextSplitter`** (`separators=["\n\n", "\n", ". ", " ", ""]`) — tries
a paragraph break, then a line break, then a sentence boundary, and only falls through
to a word or character cut if the piece still doesn't fit:
```
chunk[0]: "With interpreted languages, the code is saved in the same format that you entered"
chunk[1]: ". Compiled programs generally run faster than interpreted on..."
```
Falls back to the `" "` separator here (124 chars doesn't fit a full sentence), so the
cut lands between words, never inside one — because it will not use a finer separator
than it has to.

## Why tokens, not just characters

Chunk size ultimately exists to respect two token-based limits: the embedding model's
input window and the chat model's context window. Character count is only a proxy for
token count, and the ratio isn't fixed — it depends on the text. Measured on the book:
`With interpreted languages, the code is saved` is 47 chars / 11 tokens (~4.3
chars/token), but dense identifier- or LaTeX-heavy text runs closer to 2 chars/token.
A splitter with `chunk_size=1000` **characters** can silently produce a chunk over
double the intended token budget on the wrong kind of text (max_tokens=455 above, more
than double the average of 196) — invisible until an embedding call actually rejects
an oversized input.

`split_by_tokens()` passes `length_function=count_tokens` (backed by tiktoken's
`cl100k_base`) into `RecursiveCharacterTextSplitter`, so `chunk_size` and
`chunk_overlap` are measured in tokens directly rather than approximated from
characters. `cl100k_base` is OpenAI's tokenizer, not our Ollama models' — there's no
public tiktoken encoding for `llama3.1`/`qwen2.5`/`nomic-embed-text`. It's used anyway
as a consistent, fast token-count *estimate*: close enough to catch the
character-based blind spot above, without vendoring or calling out to a
model-specific tokenizer just to size chunks.

## Chunk size and overlap: what was picked and why

**Default (`chunking.py` constants): 1000 characters, 150 overlap, via
`RecursiveCharacterTextSplitter`.**

- **1000 chars (~200-250 tokens)** keeps each chunk small enough that a retrieved
  chunk is about one paragraph of the source — specific enough to answer a targeted
  question — while staying comfortably under `nomic-embed-text-v2-moe`'s per-input
  limit with room for several chunks plus the question in the chat model's context.
  Smaller chunks (e.g. 250-400 chars) fragment multi-sentence explanations across
  chunk boundaries and hurt recall; much larger chunks (2000+) dilute the embedding
  with unrelated sentences and hurt precision.
- **150 overlap (~15%)** so a sentence that lands right at a boundary is still
  fully present in at least one chunk, without the index bloat of a 50% overlap.
- **Recursive over naive**: demonstrated above — same chunk count and average size,
  but chunk boundaries fall on sentence/word breaks instead of mid-word, which
  matters because a chunk's *edges* are what a nearby chunk's retrieval competes
  against, and a broken word or sentence fragment embeds worse than a clean one.
- **Token-aware sizing** (`split_by_tokens`, 250 tokens / 40 overlap) is the one to
  reach for once a document's char-to-token ratio is known to be unusual (code,
  non-English text, heavy LaTeX) — the comparison above is the concrete case for
  needing it, but it isn't the default because it costs a tokenizer dependency and
  a slower splitting pass for a difference that doesn't show up on prose-heavy PDFs.
