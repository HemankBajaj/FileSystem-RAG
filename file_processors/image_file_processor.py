import os
import time
import pandas as pd
import random
import logging
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from file_processors.file_processor import FileProcessor

class ImageFileProcessor(FileProcessor):
    """
    Processes multiple image files by looking up descriptions
    from a CSV file and creating LangChain documents.
    """
    
    def __init__(self, description_file: str = "data/images/image_descriptions.csv"):
        self.description_file = description_file
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
        self.descriptions = self._load_descriptions()
        self.no_description_messages = [
            "A beautiful landscape image with vibrant colors.",
            "A black and white photo of a busy street.",
            "A close-up shot of a flower in bloom.",
            "A stunning architectural masterpiece.",
            "An abstract piece with bold shapes and colors."
        ]
    
    def _load_descriptions(self):
        """Loads image descriptions from the CSV file."""
        try:
            df = pd.read_csv(self.description_file)
            return df.set_index("file_name")["description"].to_dict()
        except FileNotFoundError:
            logging.error(f"Description file not found: {self.description_file}")
            return {}

    def process_files(self, file_paths_list: List[str]) -> List[Document]:
        """
        Processes image files by fetching descriptions and creating documents.
        
        Args:
            file_paths_list: A list of full file paths to image files.
            
        Returns:
            A list of LangChain Document objects.
        """
        all_documents = []
        for file_path in file_paths_list:
            if not os.path.isfile(file_path) or not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                logging.warning(f"Skipping invalid image file: {file_path}")
                continue
            
            file_name = os.path.basename(file_path)
            description = self.descriptions.get(file_name)
            
            if description is None:
                description = f"Image description: [NOT FOUND] {random.choice(self.no_description_messages)}"
                logging.warning(f"Description not found for {file_name}. Using a random description.")
            else:
                description = f"Image description: {description}"
            
            # Use text splitter to create documents (even if just one chunk)
            documents = self.text_splitter.create_documents([description])
            
            # Attach metadata and file path to each document
            for doc in documents:
                doc.page_content = f"{description} file_path: {file_path}"
                metadata = self.get_file_metadata(file_path)
                doc.metadata = metadata
                all_documents.append(doc)
                
            logging.info(f"[ImageFileProcessor] Processed image file: {file_path}")
            
        # Add a 30-second wait after all files are processed
        logging.info("Waiting for 30 seconds...")
        time.sleep(30)
        
        return all_documents

    def get_file_metadata(self, file_path: str):
        """
        Extracts metadata from the file path.
        Assumes the format: data/<user>/<image>/<file_name>
        """
        parts = file_path.split(os.sep)
        metadata = {
            'file_path': file_path,
            'user_id': parts[1] if len(parts) > 1 else 'unknown_user',
            'mime_type': 'image'
        }
        return metadata
