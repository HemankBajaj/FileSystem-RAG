import logging
import time
from chromadb import HttpClient
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

class ChromaClient:
    def __init__(
        self,
        collection_name: str = "all_users_docs",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 2000,
        host: str = "127.0.0.1",
        port: int = 8000,
        tenant: str = "default_tenant",
        database: str = "default_database",
    ):
        self.collection_name = collection_name
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.batch_size = batch_size

        logging.info("[ChromaClient] Connecting to Chroma HTTP server...")
        client = HttpClient(host=host, port=port, tenant=tenant, database=database)
        self.vectordb = Chroma(
            client=client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
        )
        logging.info("[ChromaClient] Connected to Chroma server.")

    def add_documents(self, docs):
        if not docs:
            logging.warning("[ChromaClient] No documents to add.")
            return

        logging.info(f"[ChromaClient] Adding {len(docs)} documents to Chroma DB in batches of {self.batch_size}...")
        for i in range(0, len(docs), self.batch_size):
            start = time.time()
            batch = docs[i:i + self.batch_size]
            self.vectordb.add_documents(batch)
            end = time.time()
            logging.info(f"[ChromaClient] Adding batch {i//self.batch_size + 1}/{len(docs)//self.batch_size + 1} with {len(batch)} documents. Time Taken {end-start} seconds")
        # No persist() in HTTP mode
        logging.info("[ChromaClient] All documents added.")

    def get_user_qa(self, user_id: str, model_name="google/flan-t5-small", max_new_tokens=200, device=-1):
        logging.info(f"[ChromaClient] Creating QA chain for user_id: {user_id}")
        start = time.time()
        retriever = self.vectordb.as_retriever(
            search_kwargs={"k": 5, "filter": {"user_id": user_id}}
        )
        llm_pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            max_new_tokens=max_new_tokens,
            device=device
        )
        llm = HuggingFacePipeline(pipeline=llm_pipeline)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        logging.info(f"Time taken to make qa_chain : {time.time()-start:.2f}s")
        return qa_chain
