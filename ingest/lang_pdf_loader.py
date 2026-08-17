from langchain_community.document_loaders import PyPDFLoader

class PDFIngest:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list:
        loader = PyPDFLoader(self.file_path)
        return loader.load()

output = PDFIngest('/Users/ajay/Documents/Projects/AI Engineering/rag-ai/assets/Python Programming.pdf').load()
