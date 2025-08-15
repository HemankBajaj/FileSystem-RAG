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
from indexing_and_embedding.chroma_db_client import ChromaClient

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
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize persistent vector DB
chroma_client = ChromaClient(persist_directory=persist_directory,
                             collection_name="all_users_docs", embedding_model_name=embedding_model)
chroma_client.add_documents(docs)

qa_a = chroma_client.get_user_qa("user_a")
result_a = qa_a({"query": "What is the speed at which sneeze travels?"})
print("\n[User A Answer]:", result_a["result"])
print("[Sources]:", [doc.metadata for doc in result_a["source_documents"]])
del qa_a, result_a

qa_b = chroma_client.get_user_qa("user_b")
result_b = qa_b({"query": "How many trees are there on earth?"})
print("\n[User B Answer]:", result_b["result"])
print("[Sources]:", [doc.metadata for doc in result_b["source_documents"]])
