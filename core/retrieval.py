import os
from typing import List
from dotenv import load_dotenv
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from core.vector_store import VectorStoreManager
from constants import RERANK_MODEL

load_dotenv()

class RAGRetriever:
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vs_manager = vector_store_manager
        self.reranker = CohereRerank(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model=RERANK_MODEL,
            top_n=5
        )

    def retrieve(self, query: str) -> List[Document]:
        """Retrieves and reranks documents based on the query."""
        # Get base documents from vector store
        base_retriever = self.vs_manager.get_retriever()
        docs = base_retriever.invoke(query)
        
        # Rerank using Cohere
        if docs:
            reranked = self.reranker.compress_documents(docs, query)
            return list(reranked)
        return docs
