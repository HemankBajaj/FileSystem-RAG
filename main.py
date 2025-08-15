import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import time
from file_processors.text_file_processor import TextFileProcessor

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Ensure INFO messages are shown

file_paths = [
    "data/user_a/text/facts.txt",
    "data/user_b/text/earth_facts.txt",
    "data/user_a/text/12.txt"
]

text_file_processor = TextFileProcessor()

docs = text_file_processor.process_files(file_paths)


# ---------- CONFIG ----------
persist_directory = "chroma_store"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize persistent vector DB
vectordb = Chroma(
    collection_name="all_users_docs",
    embedding_function=embedding_model,
    persist_directory=persist_directory,
)

vectordb.add_documents(docs)
vectordb.persist()

def get_user_qa(user_id: str):
    retriever = vectordb.as_retriever(
        search_kwargs={"k": 5, "filter": {"user_id": user_id}}
    )

    llm_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        tokenizer="google/flan-t5-small",
        max_new_tokens=200,
        device=-1  # CPU (set to 0 for GPU)
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )


qa_a = get_user_qa("user_a")
result_a = qa_a({"query": "What is the speed at which sneeze travels?"})
print("\n[User A Answer]:", result_a["result"])
print("[Sources]:", [doc.metadata for doc in result_a["source_documents"]])
del qa_a, result_a

qa_b = get_user_qa("user_b")
result_b = qa_b({"query": "How many trees are there on earth?"})
print("\n[User B Answer]:", result_b["result"])
print("[Sources]:", [doc.metadata for doc in result_b["source_documents"]])

