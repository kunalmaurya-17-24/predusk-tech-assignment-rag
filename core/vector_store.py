import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from constants import INDEX_NAME, EMBEDDING_MODEL, PINECONE_DIMENSION

load_dotenv()

class VectorStoreManager:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self._ensure_index_exists()
        self.vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=self.embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )

    def _ensure_index_exists(self):
        if INDEX_NAME not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=PINECONE_DIMENSION,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1' # Defaulting to us-east-1 for free tier
                )
            )

    def upsert_documents(self, documents):
        """Adds documents to the Pinecone vector store."""
        return self.vector_store.add_documents(documents)

    def get_retriever(self, search_kwargs=None):
        """Returns a retriever object."""
        if search_kwargs is None:
            search_kwargs = {"k": 10, "search_type": "mmr"}
        return self.vector_store.as_retriever(**search_kwargs)
