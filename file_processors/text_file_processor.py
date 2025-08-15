import os
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from file_processors.file_processor import FileProcessor
import logging 

class TextFileProcessor(FileProcessor):
    """Processes multiple .txt files into chunked documents with metadata."""

    def __init__(self, chunk_size: int = 2048, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def process_files(self, file_paths_list: List[str]):
        all_chunks = []

        for file_path in file_paths_list:
            if not os.path.isfile(file_path) or not file_path.endswith(".txt"):
                logging.warning(f"Skipping invalid file: {file_path}")
                continue

            loader = TextLoader(file_path)
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)

            for chunk in chunks:
                if not hasattr(chunk, "metadata") or chunk.metadata is None:
                    chunk.metadata = {}
                chunk.metadata.update(FileProcessor._FileProcessor__get_file_metadata(file_path))
            logging.info(f"[FileProcessor] Text File {file_path} processed with {len(chunks)} chunks")
            all_chunks.extend(chunks)
        return all_chunks


if __name__ == "__main__":
    file_paths = [
        "data/user_a/text/11.txt",
    ]

    processor = TextFileProcessor()
    chunks = processor.process_files(file_paths)

    print(f"Total chunks created: {len(chunks)}")
    if chunks:
        print("Sample chunk metadata:", chunks[0].metadata)
        print("Sample chunk text (first 500 chars):", chunks[0].page_content[:500])
