"""
This is the base class for File Processing unit. This is used to prrocess the file data to forward to our indexing unit. 
"""

from typing import List


class FileProcessor:
    """Base class for processing files into chunked documents with metadata."""

    def process_files(self, file_paths_list : List[str]):
        """Abstract method: should return a list of chunked documents with metadata."""
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_file_metadata(file_path : str):
        raise NotImplementedError("Subclasses should implement this method.")
    
if __name__ == "__main__":
    file_processor = FileProcessor()
    print(file_processor.process_files(["./"]))
