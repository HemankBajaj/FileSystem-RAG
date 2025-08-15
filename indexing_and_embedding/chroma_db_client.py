import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

class ChromaClient:
    def __init__(self, persist_directory="chroma_store", collection_name="all_users_docs",
                 embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

        logging.info("[ChromaClient] Initializing Chroma DB...")
        self.vectordb = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )
        logging.info("[ChromaClient] Chroma DB initialized.")

    def add_documents(self, docs):
        if docs:
            logging.info(f"[ChromaClient] Adding {len(docs)} documents to Chroma DB...")
            self.vectordb.add_documents(docs)
            self.vectordb.persist()
            logging.info("[ChromaClient] Documents persisted.")
        else:
            logging.warning("[ChromaClient] No documents to add.")

    def get_user_qa(self, user_id: str, model_name="google/flan-t5-small", max_new_tokens=200, device=-1):
        logging.info(f"[ChromaClient] Creating QA chain for user_id: {user_id}")
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
        logging.info(f"[ChromaClient] QA chain for user_id: {user_id} created.")
        return qa_chain
