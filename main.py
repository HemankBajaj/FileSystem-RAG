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
    "data/user_a/text/11.txt",
    "data/user_a/text/12.txt"
]

text_file_processor = TextFileProcessor()

docs = text_file_processor.process_files(file_paths)


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(docs, embedding_model, persist_directory="chroma_store")
retriever = vectordb.as_retriever()

llm_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    tokenizer="google/flan-t5-small",
    max_new_tokens=200,
    device=-1  # -1 for CPU, 0 for GPU
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

start_time = time.time()

query = "What unusual object did the White Rabbit take out of its waistcoat-pocket that made Alice curious enough to follow it?"
result = qa_chain({"query": query})

end_time = time.time()

print("\nAnswer:", result["result"])
print("\nSources and Chunks:")

for doc in result["source_documents"]:
    print("---- Chunk Text ----")
    print(doc.page_content[:500], "...")  # print first 500 chars of chunk for brevity
    print("---- Source Metadata ----")
    print(doc.metadata)
    print("\n")