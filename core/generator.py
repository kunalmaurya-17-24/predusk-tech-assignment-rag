import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from constants import GEN_MODEL

load_dotenv()

class RAGGenerator:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=GEN_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an AI assistant that answers questions based on the provided context.
        Use only the provided context to answer the question. If the answer is not in the context, say that you don't know.
        
        Provide your answer with inline citations in the format [1], [2], etc., corresponding to the context snippets.
        List the sources at the end of your answer.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """)

    def generate_answer(self, query: str, context_docs: List[Document]) -> Dict[str, Any]:
        """Generates a grounded answer with citations."""
        # Format context for prompt
        context_text = ""
        sources = []
        for i, doc in enumerate(context_docs):
            context_text += f"\n[{i+1}] {doc.page_content}\n"
            sources.append({
                "id": i + 1,
                "title": doc.metadata.get("title", "Unknown"),
                "source": doc.metadata.get("source", "Unknown"),
                "section": doc.metadata.get("section", "N/A"),
                "content": doc.page_content
            })

        # Generate answer
        chain = self.prompt_template | self.llm
        response = chain.invoke({"context": context_text, "question": query})
        
        return {
            "answer": response.content,
            "sources": sources
        }
