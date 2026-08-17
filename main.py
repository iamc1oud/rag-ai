import tempfile
from ingest.lang_pdf_loader import PDFIngest
from fastapi import FastAPI, UploadFile, File

app = FastAPI()



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
        output = PDFIngest(temp.name).load()

    return {"filename": file.filename, "output": output}
