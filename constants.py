# Store chunking sizes (800-1200) and overlap (10-15%)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Model configurations
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "gemini-2.5-flash"
RERANK_MODEL = "rerank-english-v3.0"

# Pinecone configuration
INDEX_NAME = "mini-rag-index"
PINECONE_DIMENSION = 384 # For all-MiniLM-L6-v2
