import streamlit as st
import time
import os
from core.processor import DocumentProcessor
from core.vector_store import VectorStoreManager
from core.retrieval import RAGRetriever
from core.generator import RAGGenerator
from constants import GEN_MODEL

st.set_page_config(page_title="Predusk Mini RAG", layout="wide")

# Initialize components
@st.cache_resource
def init_components():
    processor = DocumentProcessor()
    vs_manager = VectorStoreManager()
    retriever = RAGRetriever(vs_manager)
    generator = RAGGenerator()
    return processor, vs_manager, retriever, generator

processor, vs_manager, retriever, generator = init_components()

st.title("🚀 Predusk AI - Mini RAG System")

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF or TXT files", accept_multiple_files=True)
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing..."):
                for uploaded_file in uploaded_files:
                    # Save locally temporarily
                    with open(uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    docs = processor.load_document(uploaded_file.name)
                    chunks = processor.split_documents(docs)
                    vs_manager.upsert_documents(chunks)
                    os.remove(uploaded_file.name)
                st.success("Documents processed and indexed!")
        else:
            st.warning("Please upload files first.")

# Query interface
query = st.text_input("Ask a question about your documents:")

if query:
    start_time = time.time()
    with st.spinner("Retrieving and Reranking..."):
        context_docs = retriever.retrieve(query)
    
    with st.spinner("Generating Answer..."):
        result = generator.generate_answer(query, context_docs)
    
    end_time = time.time()
    duration = end_time - start_time

    st.markdown("### Answer")
    st.write(result["answer"])

    with st.expander("Sources & Citations"):
        for source in result["sources"]:
            st.markdown(f"**[{source['id']}] {source['title']}** - {source['section']}")
            st.text(source["content"][:200] + "...")

    # Performance Metadata
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Response Time", f"{duration:.2f}s")
    
    # Simple cost estimation (rough)
    with col2:
        # Rough token estimate
        prompt_tokens = len(query.split()) + len(str(context_docs).split())
        completion_tokens = len(result["answer"].split())
        # Gemini 2.5 Flash pricing: ~$0.075/1M input, ~$0.30/1M output
        cost = (prompt_tokens * 0.000000075) + (completion_tokens * 0.0000003)
        st.metric("Est. Cost", f"${cost:.6f}")
    
    with col3:
        st.metric("Chunks Retrieved", len(context_docs))
