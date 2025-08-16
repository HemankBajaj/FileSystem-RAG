import os
import time
import logging
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from file_processors.file_processor import FileProcessor
from transformers import pipeline
from PIL import Image

class ImageFileProcessor(FileProcessor):
    """
    Processes multiple image files by captioning the image using Blip.
    """
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
        self.captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    def _caption_image(self, image_path):
        image = Image.open(image_path)
        caption = self.captioner(image)
        return caption[0]['generated_text']
    
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
            description = self._caption_image(file_path)
            
            if description is None:
                description = ""
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
        
        return all_documents

    def get_file_metadata(self, file_path: str):
        """
        Extracts metadata from the file path.
        Assumes the format: data/<user>/<image>/<file_name>
        """
        parts = file_path.split(os.sep)
        metadata = {
            'source' : file_path,
            'file_path': file_path,
            'user_id': parts[1] if len(parts) > 1 else 'unknown_user',
            'mime_type': 'image'
        }
        return metadata
    
# if __name__ == "__main__":
#     image_processor = ImageFileProcessor()
#     print(image_processor.process_files(["data/images/COCO_train2014_000000000009.jpg"]))
