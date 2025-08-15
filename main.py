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

text_file_processor.process_files(file_paths)