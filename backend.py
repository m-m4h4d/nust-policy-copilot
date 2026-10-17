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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class NustCopilot:
    def __init__(self, data_path=os.path.join(BASE_DIR, "data", "nust_corpus_preprocessed.json")):
        self.docs = self._load_docs(data_path)
        self.embeddings = HuggingFaceEmbeddings(model_name=config.MODEL_NAME)
        
        # Initialize Retrievers
        self.vector_db = FAISS.from_documents(self.docs, self.embeddings)
        self.dense_retriever = self.vector_db.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})
        self.bm25_retriever = BM25Retriever.from_documents(self.docs)
        self.bm25_retriever.k = config.RETRIEVAL_K
        
        # LLM (only when a key is set, so retrieval-only use such as the eval's Phase 1 works without one)
        api_key = os.environ.get("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI (
            model=config.LLM_MODEL,
            temperature=0,
            google_api_key=api_key
        ) if api_key else None
        
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
        self.chain = self.prompt | self.llm | StrOutputParser() if self.llm else None

    def _load_docs(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [Document(page_content=d['page_content'], metadata=d['metadata']) for d in data]

    def get_hybrid_docs(self, query):
        # Weighted Reciprocal Rank Fusion: score(d) = sum over retrievers of weight / (RRF_K + rank)
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.bm25_retriever.invoke(query)

        scores = {}
        docs_by_id = {}
        for ranked, weight in ((dense_docs, config.DENSE_WEIGHT), (sparse_docs, config.SPARSE_WEIGHT)):
            for rank, d in enumerate(ranked, start=1):
                doc_id = d.metadata['doc_id']
                scores[doc_id] = scores.get(doc_id, 0.0) + weight / (config.RRF_K + rank)
                docs_by_id[doc_id] = d

        fused_ids = sorted(scores, key=scores.get, reverse=True)
        return [docs_by_id[i] for i in fused_ids[:config.RETRIEVAL_K]]

    def generate(self, query, use_hybrid=True):
        """Retrieve and generate. Returns the raw LLM output (no cite-checking) and the docs."""
        docs = self.get_hybrid_docs(query) if use_hybrid else self.dense_retriever.invoke(query)
        if not docs: return "I cannot answer this.", []
        if self.chain is None:
            raise RuntimeError("GOOGLE_API_KEY is not set. Add it to .env or .streamlit/secrets.toml.")

        context_str = "\n\n".join([f"Source [{d.metadata['doc_id']}]: {d.page_content}" for d in docs])

        response = self.chain.invoke({
            "context": context_str,
            "question": query,
            "date": datetime.now().strftime("%Y-%m-%d")
        })
        return response, docs

    def apply_strict_mode(self, response, docs):
        # Simple cite-checker: Ensure cited ID exists in retrieved docs
        retrieved_ids = [d.metadata['doc_id'] for d in docs]
        citations = re.findall(r'\[(.*?)\]', response)
        for cit in citations:
            if cit not in retrieved_ids:
                response = response.replace(f"[{cit}]", "") # Strip hallucination

        if len(citations) == 0 and "cannot answer" not in response:
             response += "\n\n(Note: No verifiable citations found for this generated answer.)"
        return response

    def answer(self, query, strict_mode=True):
        response, docs = self.generate(query)
        if strict_mode and docs:
            response = self.apply_strict_mode(response, docs)
        return response, docs