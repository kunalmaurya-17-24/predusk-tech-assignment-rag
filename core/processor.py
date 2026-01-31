import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from constants import CHUNK_SIZE, CHUNK_OVERLAP

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )

    def load_document(self, file_path: str) -> List[Document]:
        """Loads a document based on its extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            loader = PyMuPDFLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        return loader.load()

    def process_text(self, text: str, source_name: str) -> List[Document]:
        """Processes raw text into chunks."""
        doc = Document(page_content=text, metadata={"source": source_name})
        return self.split_documents([doc])

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits documents into smaller chunks with metadata."""
        chunks = self.text_splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks),
                "title": chunk.metadata.get("source", "Unknown"),
                "section": f"Section {i+1}",
                "position": chunk.metadata.get("start_index", 0)
            })
        return chunks
