import tempfile
from dataclasses import asdict

from fastapi import FastAPI, File, Form, UploadFile
from langchain_core.vectorstores import InMemoryVectorStore

from ingest import PDFLoader
from vectorstore import VectorStoreService

app = FastAPI()


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return {"filename": file.filename, "contents": contents}


@app.post("/ingest/")
async def ingest_file(file: UploadFile = File(...)):
    file_content = await file.read()
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp:
        temp.write(file_content)
        temp.seek(0)
        records = PDFLoader().load(temp.name)
    return {"filename": file.filename, "output": [asdict(r) for r in records]}


@app.post("/query")
@app.post("/query/", include_in_schema=False)
async def query(query: str = Form(...), k: int = Form(4), file: UploadFile = File(...)):
    """Embed one PDF into a throwaway in-memory store and retrieve against it.

    Test endpoint for "do the embeddings retrieve anything sensible", not the
    real pipeline (that's cli.py's persistent Chroma store).
    """
    file_content = await file.read()
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp:
        temp.write(file_content)
        temp.seek(0)
        records = PDFLoader().load(temp.name)

    pages = [r for r in records if r.text.strip() and not r.is_scanned]
    texts = [r.text for r in pages]
    metadatas = [{"source": file.filename, "page": r.page} for r in pages]

    vectorstore = await InMemoryVectorStore.afrom_texts(
        texts, embedding=VectorStoreService.embeddings(), metadatas=metadatas
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_documents = await retriever.ainvoke(query)

    return {
        "filename": file.filename,
        "query": query,
        "pages_indexed": len(texts),
        "results": [
            {"page": d.metadata["page"], "text": d.page_content} for d in retrieved_documents
        ],
    }
