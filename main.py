import tempfile
from ingest import load_pdf
from embedding import EmbeddingService
from fastapi import FastAPI, UploadFile, File, Form
from langchain_core.vectorstores import InMemoryVectorStore

app = FastAPI()
embedding_service = EmbeddingService()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return {"filename": file.filename, "contents": contents}

@app.post("/ingest/")
async def ingest_file(file: UploadFile = File(...)):
    file_content = await file.read()

    output = None

    with tempfile.NamedTemporaryFile(delete=True, suffix='.pdf') as temp:
        temp.write(file_content)
        temp.seek(0)
        output = load_pdf(temp.name)

    return {"filename": file.filename, "output": output}


@app.post("/query")
@app.post("/query/", include_in_schema=False)
async def query(query: str = Form(...), k: int = Form(4), file: UploadFile = File(...)):
    """Embed one PDF into a throwaway in-memory store and retrieve against it.

    The store lives for the request only -- this is the testing endpoint for
    "do the embeddings retrieve anything sensible", not the real pipeline.
    """
    file_content = await file.read()

    with tempfile.NamedTemporaryFile(delete=True, suffix='.pdf') as temp:
        temp.write(file_content)
        temp.seek(0)
        records = load_pdf(temp.name)

    # One document per page, keeping the citation metadata (issue #10).
    # Pages flagged is_scanned are title/divider pages with no real text layer;
    # embedding them just puts "Python Programming" at the top of every result.
    pages = [r for r in records if r["text"].strip() and not r["is_scanned"]]
    texts = [r["text"] for r in pages]
    metadatas = [
        {"source_file": file.filename, "page_number": r["page_number"]} for r in pages
    ]

    embeddings = await embedding_service.get_embed()
    vectorstore = await InMemoryVectorStore.afrom_texts(
        texts, embedding=embeddings, metadatas=metadatas
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_documents = await retriever.ainvoke(query)

    return {
        "filename": file.filename,
        "query": query,
        "pages_indexed": len(texts),
        "results": [
            {"page_number": d.metadata["page_number"], "text": d.page_content}
            for d in retrieved_documents
        ],
    }
