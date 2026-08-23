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

# Vector store

`vectorstore.py`. Chunks from issue #3 -> embedded via `OllamaEmbeddings` -> stored in
a persistent `Chroma` collection (`langchain-chroma`).

```bash
python vectorstore.py                                          # wiring self-checks
python vectorstore.py "assets/Python Programming.pdf" "python"  # index + query, live
```

## Which endpoint OllamaEmbeddings calls

`embed_documents()` (used when adding chunks) calls `self._client.embed(...)`, and
`ollama.Client.embed` POSTs to **`/api/embed`** — confirmed by reading both source
files (`langchain_ollama/embeddings.py`, `ollama/_client.py`). That's the batch
endpoint: the whole list of chunk texts goes in one request, one response with an
`"embeddings"` list back — not the older single-text `/api/embeddings`, which
`OllamaEmbeddings` never calls. Matters for ingestion throughput: 195 chunks embed in
one round trip's worth of batching, not 195.

## Chroma vs. FAISS, for this project

| | Chroma (`langchain-chroma`) | FAISS (`langchain-community`) |
| --- | --- | --- |
| Persistence | Automatic — give it `persist_directory`, every write lands on disk immediately | In-memory only; `save_local()`/`load_local()` are calls *you* have to remember to make |
| Runs as | Embedded library, no separate process | Embedded library, no separate process |
| Metadata filtering | `similarity_search(query, filter={...})`, native | Supported but bolted on (post-filter or a metadata-aware index you configure) |
| What's stored on disk | Vectors + documents + metadata, one SQLite-backed directory | Just the index; you separately pickle/store the documents and metadata yourself |

Both are embedded (no server), so that's a wash. The deciding factor is persistence
model: Chroma treats "written = durable" as the default, so `build_vectorstore()` and
`load_vectorstore()` in this file are ~10 lines total. FAISS makes persistence and
metadata storage the caller's problem — doable, but it's exactly the kind of manual
plumbing this project is trying to avoid by using LangChain integrations at all
(see issue #1). Chosen: Chroma.

Note: current `langchain-chroma`/`chromadb` don't have a `.persist()` method — every
write already persists as it happens. That method existed in older Chroma releases;
`Chroma.from_documents(..., persist_directory=...)` is what replaces "create + embed +
add + persist" as one call now, and `add_documents()` persists on every subsequent add.

## Metadata filtering

Each chunk's metadata (`{"source": ..., "page": ...}`, set in `chunking.to_documents`)
survives into Chroma untouched, so `similarity_search` can filter by it directly:

```python
store.similarity_search(query, k=5, filter={"page": 21})
store.similarity_search(query, k=5, filter={"source": "assets/Python Programming.pdf"})
```

Verified: filtering by a page from a different, unindexed PDF (`filter={"source":
"assets/bert-two-column.pdf"}`) returns 0 results even though the index has 195
chunks — the filter is a real pre-search restriction, not ignored.

## Fresh-process reload + similarity_search(k=5)

```
$ python -c "from vectorstore import index_pdf; index_pdf('assets/Python Programming.pdf')"
indexed: 195

# separate python process, no import of index_pdf, only load_vectorstore()
$ python -c "from vectorstore import load_vectorstore; s = load_vectorstore(); print(s._collection.count())"
reloaded count: 195
```

`similarity_search("What is a Python dictionary?", k=5)` after that reload:

```
page 21  -> "Chapter 2 What is Python? 2.1 Introduction to Python..."
page 138 -> "[18] python.org, "The python standard library"..."
page 5   -> "Preface Python is a popular programming language..."
page 137 -> "Bibliography [1] H.-P. Halvorsen..."
page 15  -> "We need to find and learn Programming Languages..."
```

Page 21 (the actual "What is Python?" chapter) ranks first, as expected. The
bibliography pages ranking 2nd-4th is a real, informative miss: the book's index
citing `python.org`/`matplotlib.org` lexically overlaps "Python" heavily without being
about the topic — exactly the kind of result issue #6 (retrieval tuning /
re-ranking) exists to improve on.

## Sanity-checking a similarity score

Chroma's HNSW index defaults to **squared L2 distance**, not cosine — this was a real
bug caught while writing this check, not a hypothetical. First pass, without setting
the distance space:

```
chroma raw distance:                    0.6365
hand-computed cosine similarity:        0.6817   ->  1 - cosine = 0.3183
```

Those don't match — until you notice `0.6365 / 0.3183 ≈ 2`. Confirmed
`OllamaEmbeddings` output is unit-normalized (`norm(embed_query("test")) ≈ 0.99999986`),
and for unit vectors `L2² = 2·(1 − cosine_similarity)`. That's exactly the factor of 2
above: Chroma was returning squared-L2, correctly, just not the metric this check
expected.

Fix: `build_vectorstore()` now passes `collection_metadata={"hnsw:space": "cosine"}` to
`Chroma.from_documents`, since the embeddings are meant to be compared by cosine
similarity. Same query, same top result, after the fix:

```python
doc, chroma_distance = store.similarity_search_with_score(query, k=1)[0]
q_vec = embeddings.embed_query(query)
d_vec = embeddings.embed_query(doc.page_content)
hand_cosine = sum(a*b for a, b in zip(q_vec, d_vec)) / (norm(q_vec) * norm(d_vec))

# chroma reported distance:              0.318253
# chroma reported similarity (1 - dist): 0.681747
# hand-computed cosine similarity:       0.681746
# difference:                            6.3e-07
```

The residual `6.3e-07` is re-embedding float noise (the doc text was embedded twice,
once during indexing and once for this check — Ollama's embedding call isn't bitwise
deterministic across requests), not a metric mismatch. Ranking was identical before and
after the fix — squared L2 and cosine distance are monotonic transforms of each other
for unit vectors — but the *reported number* only means "cosine similarity" now.

# RAG chain

`chain.py`. LCEL pipeline wiring the Chroma retriever from issue #4 to `ChatOllama`:
retrieve -> relevance-threshold gate -> format numbered context -> prompt -> stream ->
append citations.

```bash
python chain.py                                          # wiring self-checks
python chain.py "How do I plot a sine function in matplotlib?"   # live, streamed
```

## How it's wired

```python
scored = store.similarity_search_with_relevance_scores(question, k=5)
if not scored or scored[0][1] < RELEVANCE_THRESHOLD:
    yield NO_CONTEXT_MESSAGE
    return                                    # no LLM call at all

chain = prompt | ChatOllama(...) | StrOutputParser()
for token in chain.stream({"context": format_docs(docs), "question": question}):
    yield token
yield format_citations(docs)
```

The threshold check runs *before* the LCEL chain, not as a step inside it: it has to be
able to veto generation entirely, and by the time a `RunnableLambda` inside a chain
sees the retrieved docs, the chain is already committed to calling the model. Keeping
it as a plain Python `if` ahead of `chain.stream(...)` is also the only way to make the
"no LLM call for out-of-scope questions" behavior fast and certain rather than a prompt
instruction the model could ignore.

The prompt instructs the model to answer only from the numbered context, cite the
block number inline after every sentence that uses it, and reply with the exact string
`I don't know -- no relevant context found for that question.` when the context doesn't
cover the question — a second, model-level guardrail behind the retrieval-score one
(see "A second guardrail" below).

## Picking the relevance threshold

`similarity_search_with_relevance_scores` returns Chroma's cosine similarity (see
issue #4 — `hnsw:space="cosine"` makes `1 - distance` a real cosine similarity, not an
arbitrary score). Measured on this corpus, top-1 score for 4 on-topic and 4
clearly-off-topic questions:

| Question | Top score |
| --- | --- |
| How do I plot a sine function with matplotlib? | 0.682 |
| What is a Python dictionary? | 0.623 |
| How do I install Python on Windows? | 0.683 |
| What is the difference between a compiled and interpreted language? | 0.698 |
| What is the capital of France? | 0.208 |
| How do I bake a chocolate cake? | 0.323 |
| What is the Transformer architecture in deep learning? | 0.346 |
| Who won the world cup in 2018? | 0.271 |

On-topic clusters at 0.62-0.70, off-topic at 0.21-0.35 — a clean gap. `0.45` sits in the
middle with margin on both sides. This is a property of `nomic-embed-text-v2-moe` and
this specific corpus (a Python textbook), not a universal constant — a corpus covering
broader or more overlapping topics would need this re-measured, not assumed.

## Manually verified: 3 grounded questions, citations checked against the source PDF

**"How do I plot a sine function with matplotlib?"** (top score 0.682) — answer cited
`[1]` (page 50) and `[4]` (page 51). Page 50 contains `Example 4.6.2. Plotting a Sine
Curve`; page 51 contains the exact `x = np.arange(xstart, xstop, increment)` /
`grid()` lines the answer describes. Correct.

**"What is the difference between a compiled and interpreted language?"** (top score
0.698) — cited `[1]`/`[3]` (page 23) and `[2]` (page 22). Page 22 literally reads "code
you enter is reduced to a set of machine-specific instructions before being saved as an
executable file" — matches the `[2]` claim word for word. Page 23 has the "must be
parsed, interpreted, and executed each time" and "ad hoc calculations" language cited
under `[1]`/`[3]`. Correct. (The chain also retrieved page 103, an unrelated VS Code
setup page that happens to contain the word "interpreted" — the model correctly never
cited it in the body, even though it's listed in the Sources block as retrieved.)

**"How do I install Python on Windows?"** (top score 0.683) — cited `[1]` (page 29,
Microsoft Store + Anaconda instructions) and `[2]` (page 28, the python.org link). Both
check out against the source text. Retrieved-but-uncited pages 33/87 (Command Prompt,
pip) were correctly left out of the answer body.

**Out-of-scope: "What is the capital of France?"** (top score 0.208, well under 0.45) —
returned `I don't know -- no relevant context found for that question.` immediately,
confirmed via timing that no generation call happened (0.7s total vs. 2+ s for the
grounded answers above, all of which include a full generation).

## A second guardrail: the threshold passing isn't the same as the answer being grounded

**"What is a Python dictionary?"** scored 0.623 — comfortably over threshold, so
generation went ahead — and the model still answered
`I don't know -- no relevant context found for that question.` The retrieved chunks
(the "What is Python?" chapter intro, the preface, the bibliography) all score
reasonably high because they're dense with the word "Python", but none of them actually
covers dictionaries; this book's dictionary content, if any, didn't make the top 5 for
this phrasing. This is exactly the failure mode a pure similarity-threshold check can't
catch — high lexical/topical overlap with genuinely wrong content — and it's why the
prompt also instructs the model to decline when *its* context is insufficient, not just
the retriever. Two independent checks, two different failure modes covered.

# Evaluation

`eval.py`. A labeled test set plus two kinds of metrics: hand-rolled retrieval metrics
(Hit Rate@k, MRR) and ragas LLM-as-judge generation metrics (faithfulness, answer
relevancy, context precision).

```bash
python eval.py                         # wiring self-checks (no live calls)
python eval.py live 5                  # full report at k=5: retrieval + ragas
python eval.py live 5 --no-ragas       # retrieval only (fast, no LLM judge calls)
```

## The test set

16 Q&A pairs against `assets/Python Programming.pdf` (`TEST_SET` in `eval.py`), each
with an `expected_pages` tuple. The pages were picked by reading the book's chapter
structure first (`Chapter N` headers, via `ingest.load_pdf`) and writing the question
from the chapter content — **not** by running retrieval and checking what came back.
Ground truth has to be independent of the system being measured, or the eval only
confirms the retriever agrees with itself.

## Hit Rate@k and MRR — a few lines, no library

```python
rank = next((i for i, p in enumerate(pages, start=1) if p in case.expected_pages), None)
hit_rate = sum(r is not None for r in ranks) / len(ranks)
mrr = sum(1 / r for r in ranks if r is not None) / len(ranks)
```

**Hit Rate@k**: fraction of questions where at least one expected page appears
somewhere in the top-k retrieved chunks — a coarse "did retrieval fail outright"
signal. **MRR** (Mean Reciprocal Rank): average of `1/rank` of the *first* correct
hit (0 if none in top-k) — rewards the correct chunk appearing early, which matters
because it's what the prompt puts first in context and what a user reads first in a
citation list.

## ragas: how the three metrics work, and their limits

All three are **LLM-as-judge**: an LLM (not a formula) reads the question, the
retrieved context, and the generated answer, and produces a verdict.

- **Faithfulness**: the judge extracts the individual factual claims in the answer,
  then checks each one against the retrieved context and reports the fraction that
  are supported. Measures hallucination, not correctness against the real world — an
  answer can be 100% faithful to context that is itself wrong.
- **Answer relevancy**: the judge generates several *hypothetical questions* that the
  given answer would be a good response to, embeds them, and averages their cosine
  similarity to the real question. A correct-but-incomplete answer (e.g. missing part
  of a multi-part question) can still score high, because the judge only sees "does
  this answer look like it's addressing something in this direction."
- **Context precision**: the judge scores each retrieved chunk as useful or not for
  producing the reference answer, then computes precision weighted so
  usefully-ranked-earlier chunks count more. This is the metric that overlaps most
  with Hit Rate/MRR, but scores *usefulness for the answer*, not just topical rank.

**Known biases/limitations, from ragas's own docs and observed here:**
- **Self-preference bias**: a model tends to rate its own outputs' phrasing more
  favorably than equally-correct answers worded differently. This is why the judge
  here (`ornith:9b`, local) is deliberately not `settings.CHAT_MODEL` (the generator,
  `gemma4:31b-cloud`) — grading your own homework is the textbook failure case.
- **Judge quality caps score reliability**: all three metrics are only as good as the
  judge's ability to extract claims / generate hypothetical questions / assess
  usefulness. A small local model judging a much larger generator's answers can
  itself misjudge, especially on claim decomposition for faithfulness.
- **Cost/latency**: each metric is its own LLM call (or several, for claim
  decomposition) per question. 16 questions x 3 metrics x 1 generation = 64 LLM calls
  for one report — this doesn't scale to CI running per-commit without a much smaller
  test set or a faster/cheaper judge.
- **Non-determinism**: LLM judges don't give bit-identical scores run to run,
  especially at nonzero temperature. `model_args=InstructorModelArgs(temperature=0.01)`
  (ragas's own default) keeps this small but doesn't eliminate it — treat single-run
  score deltas smaller than ~0.05 as noise, not signal.

### Two real bugs hit wiring this up (both are genuine upstream issues, not typos)

1. **`ragas` 0.4.3 fails to import at all** against the versions of `langchain-chroma`
   and `langchain-community` this project already needs: `ragas/llms/base.py`
   unconditionally does `from langchain_community.chat_models.vertexai import
   ChatVertexAI`, a submodule current `langchain-community` (0.4.x, required by
   `langchain-chroma>=1.1`) no longer ships — it moved to the standalone
   `langchain-google-vertexai` package with no back-compat shim left behind. There is
   no version of `langchain-community` that satisfies both: old enough to have that
   submodule means `langchain-core<1.0`, which `langchain-chroma>=1.1` rejects.
   Fixed with a placeholder module registered in `sys.modules` before importing ragas
   (`eval.py`, top) — `ChatVertexAI` is never instantiated since Ollama is the judge,
   so a class that's never used only needs to exist, not work.
2. **Ollama's judge calls truncated mid-response**: ragas's `InstructorModelArgs`
   defaults `max_tokens=1024`. `ornith:9b` is a reasoning model that spends tokens on
   a `<think>` block before its structured JSON verdict, so 1024 wasn't enough and
   every judge call raised `IncompleteOutputException`. `llm_factory()`'s own
   `InstructorAdapter` always constructs its own `InstructorModelArgs()` and then
   re-passes `**kwargs`, so passing `model_args=...` through `llm_factory` collides
   with itself (`TypeError: got multiple values for keyword argument 'model_args'`) —
   a real bug in ragas 0.4.3, not a usage error. Worked around by building
   `InstructorLLM` directly (`eval.py`, `_judge_clients()`), bypassing `llm_factory`,
   with `max_tokens=4096`.

## Baseline report (chunk_size=1000/overlap=150, k=5)

```
Hit Rate@5: 1.000
MRR@5:      0.830

avg faithfulness:       1.000
avg answer_relevancy:   0.903
avg context_precision:  0.871
```

Faithfulness at 1.000 across all 16 questions is expected, not suspicious: the
threshold-gated LCEL chain (issue #5) only generates when retrieval clears the
relevance bar, and the prompt hard-instructs "answer only from context" — faithfulness
specifically measures whether that instruction held, and on a textbook with
non-adversarial questions it should.

## Tuning iteration: k=3 -> k=5

**Failure found.** Retrieval-only report at `k=3` (`python eval.py live 3 --no-ragas`):

```
Hit Rate@3: 0.875
MRR@3:      0.802

4 question(s) not a rank-1 hit:
  rank=None  What is the recommended basic editor for writing your first Python programs?
  rank=   2  What is the Python Standard Library?
  rank=None  How do you print 'Hello World' in Python?
  rank=   3  What is Anaconda?
```

Two outright misses (page not in top 3 at all) out of 16 questions.

**Root cause.** Inspecting the actual retrieved chunks for the two misses:

```
"What is the recommended basic editor...?"  (expected page 30)
  1. page 97  score=0.739  "you can use Python in that editor..."
  2. page 25  score=0.715  "Which editor you should use depends on your background..."
  3. page 97  score=0.697  "Chapter 16 Python Editors..."
  ...
  5. page 30  score=0.670  "Chapter 3 Start using Python... IDLE"        <- expected page

"How do you print 'Hello World' in Python?"  (expected page 40)
  1. page 43  score=0.569  "Variables of numeric types are created..."
  2. page 31  score=0.563  "Example 3.2.1. Plotting in Python. Lets open your Python Editor and type"
  3. page 107 score=0.561  "Figure 19.4: Python Interactive. We start by creating a basic Hello World"
  4. page 40  score=0.549  "Chapter 4 Basic Python Programming..."        <- expected page
```

Both misses share the same cause: the book reuses near-identical example-intro
boilerplate ("Lets open your Python Editor and type...") across many chapters, and
discusses editors in three separate places (ch. 3's quick IDLE mention, the dedicated
ch. 16 "Python Editors", and the ch. 2 comparison page). That lexical/topical overlap
pushes the single specific chunk this test set names as "correct" to rank 4-5, not
because retrieval is wrong, but because multiple chunks are legitimately
about the same surface topic. At k=3, the correct chunk is cut off; it was never far
away.

**Fix.** Raise k from 3 to 5 — already what `chain.py`'s retriever defaults to
(issue #5's `search_kwargs={"k": 5}`), so this is confirmation the existing default is
load-bearing, not a new code change.

**Re-run, improvement shown:**

| | Hit Rate | MRR |
| --- | --- | --- |
| k=3 (before) | 0.875 (14/16) | 0.802 |
| k=5 (after) | **1.000 (16/16)** | **0.830** |

The generation-side metrics corroborate this independently: the "recommended basic
editor" question has the lowest `context_precision` of the whole test set (0.59, vs.
0.87 average) even at k=5 — ragas's judge is penalizing the same chunk-competition
problem the retrieval numbers found, from the generation side. That agreement across
two independently-computed metrics (one arithmetic, one LLM-judged) is itself a
sanity check that neither is measuring noise.

**What this doesn't fix**: k=5 recovers Hit Rate but MRR (0.830) is still short of a
hypothetical 1.0 (every answer at rank 1) — the four questions listed above still rank
their correct chunk 2nd-5th, they just now clear the "in top-5" bar. A chunk_size or
embedding-model change might close that gap further; k was the cheapest, evidence-backed
fix available and it was already correctly set.
