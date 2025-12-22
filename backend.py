import os
import json
import re
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import config

class NustCopilot:
    def __init__(self, data_path="data/nust_corpus_preprocessed.json"):
        self.docs = self._load_docs(data_path)
        self.embeddings = HuggingFaceEmbeddings(model_name=config.MODEL_NAME)
        
        # Initialize Retrievers
        self.vector_db = FAISS.from_documents(self.docs, self.embeddings)
        self.dense_retriever = self.vector_db.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})
        self.bm25_retriever = BM25Retriever.from_documents(self.docs)
        self.bm25_retriever.k = config.RETRIEVAL_K
        
        # LLM
        self.llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            temperature=0,
            google_api_key=config.GOOGLE_API_KEY
            )
        
        # Prompt with Staleness Logic
        self.prompt = ChatPromptTemplate.from_template(
            """You are the NUST Policies Copilot. 
            Current Date: {date}
            
            Context:
            {context}

            Question: {question}

            INSTRUCTIONS:
            1. Answer based ONLY on the context.
            2. CITE SOURCES: Use format [doc_id] at the end of sentences.
            3. STALENESS CHECK: If the user asks about a specific year (e.g., 2023) and the context is from a different year, explicitly warn them.
            4. If the answer is not in the context, say "I cannot answer this based on available policies."
            """
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _load_docs(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [Document(page_content=d['page_content'], metadata=d['metadata']) for d in data]

    def get_hybrid_docs(self, query):
        # RRF Logic
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.bm25_retriever.invoke(query)
        
        # Simple deduplication for demo speed (Production would use RRF math)
        seen = set()
        final_docs = []
        for d in dense_docs + sparse_docs:
            if d.metadata['doc_id'] not in seen:
                final_docs.append(d)
                seen.add(d.metadata['doc_id'])
        return final_docs[:config.RETRIEVAL_K]

    def answer(self, query, strict_mode=True):
        docs = self.get_hybrid_docs(query)
        if not docs: return "I cannot answer this.", []
        
        context_str = "\n\n".join([f"Source [{d.metadata['doc_id']}]: {d.page_content}" for d in docs])
        
        response = self.chain.invoke({
            "context": context_str, 
            "question": query,
            "date": datetime.now().strftime("%Y-%m-%d")
        })
        
        if strict_mode:
            # Simple cite-checker: Ensure cited ID exists in retrieved docs
            retrieved_ids = [d.metadata['doc_id'] for d in docs]
            citations = re.findall(r'\[(.*?)\]', response)
            for cit in citations:
                if cit not in retrieved_ids:
                    response = response.replace(f"[{cit}]", "") # Strip hallucination
            
            if len(citations) == 0 and "cannot answer" not in response:
                 response += "\n\n(Note: No verifiable citations found for this generated answer.)"

        return response, docs