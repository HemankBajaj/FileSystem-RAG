import os
import multiprocessing
import logging
import time
import redis
from datetime import datetime
import sys
import json

# To resolve import issue
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_processors.text_file_processor import TextFileProcessor
from file_processors.image_file_processor import ImageFileProcessor

DATA_DIR = "data"
SKIP_DIR = "books"
NUM_PROCESSES = 5

# Redis Configuration
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
STREAM_KEY = 'ingestion_stream_chunks'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UserIngestionWorker:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.data_dir = os.path.join("data", user_id)
        self.skip_dir = "books"
        self.text_processor = TextFileProcessor()
        self.image_processor = ImageFileProcessor()

        # Connect to Redis
        self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        try:
            self.redis_client.ping()
            logger.info(f"[{multiprocessing.current_process().name}] Connected to Redis.")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"[{multiprocessing.current_process().name}] Could not connect to Redis: {e}")
            raise

    def get_published_files_key(self):
        """Return the Redis key for a user's set of published files."""
        return f"published_files:{self.user_id}"

    def has_file_been_published(self, file_path: str):
        """Check if a file has been published using a Redis Set."""
        key = self.get_published_files_key()
        return self.redis_client.sismember(key, file_path)

    def mark_file_as_published(self, file_path: str, total_chunks_published : int):
        """Add a file to the Redis Set of published files."""
        key = self.get_published_files_key()
        self.redis_client.sadd(key, file_path)
        logger.info(f"Marked '{file_path}' with {total_chunks_published} chunks as published for user '{self.user_id}'.")

    def _publish_chunk_to_stream(self, chunk, total_chunks):
        """Adds a new chunk's information to the Redis Stream."""
        # Include the total number of chunks for the file in the message metadata
        chunk.metadata['total_chunks'] = total_chunks
        
        message = {
            'page_content': chunk.page_content,
            'metadata': json.dumps(chunk.metadata),  # Serialize metadata
            'timestamp': datetime.now().isoformat()
        }
        # Removing max len limit here
        self.redis_client.xadd(STREAM_KEY, message)

    def ingest_files(self):
        """Checks a user's directory for new files, chunks them, and publishes the chunks."""
        total_chunks_published = 0
        if os.path.isdir(self.data_dir):
            for root, _, files in os.walk(self.data_dir):
                if self.skip_dir in os.path.relpath(root, self.data_dir).split(os.sep):
                    continue
                for file in files:
                    file_path = os.path.join(root, file)
                    if not self.has_file_been_published(file_path):
                        # Chunk the file
                        if file_path.endswith(".txt"):
                            chunks = self.text_processor.process_files([file_path])
                            total_chunks = len(chunks)
                            for chunk in chunks:
                                self._publish_chunk_to_stream(chunk, total_chunks)
                                total_chunks_published += 1

                            self.mark_file_as_published(file_path, total_chunks_published)
                        elif file_path.endswith(".png") or file_path.endswith(".jpg"):
                            chunks = self.image_processor.process_files([file_path])
                            total_chunks = len(chunks)
                            for chunk in chunks:
                                self._publish_chunk_to_stream(chunk, total_chunks)
                                total_chunks_published += 1

                            self.mark_file_as_published(file_path, total_chunks_published)

                        else:
                            logging.warning(f"[{multiprocessing.current_process().name}] Skipping unsupported file format: {file_path}")
        if total_chunks_published > 0:
            logger.info(f"[{multiprocessing.current_process().name}] Finished publishing new files. Total chunks: {total_chunks_published}")
        return total_chunks_published

def create_user_directory(user_id: str):
    """Creates a data directory for a user."""
    user_dir = os.path.join(DATA_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    logger.info(f"User '{user_id}' directory created.")

def ingestion_worker_process(user_id: str):
    """Function to create and run a single worker process."""
    worker = UserIngestionWorker(user_id)
    while True:
        try:
            worker.ingest_files()
        except Exception as e:
            logger.error(f"[{multiprocessing.current_process().name}] An error occurred: {e}")
        time.sleep(5) # Interval between checks

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    users_to_ingest = ["user_a", "user_b", "user_c", "user_d", "user_e"]
    processes = []
    
    # Create directories for each user before starting the processes
    for user_id in users_to_ingest:
        create_user_directory(user_id)

    logger.info("Starting ingestion worker processes... 🚀")
    
    for user_id in users_to_ingest:
        process = multiprocessing.Process(
            target=ingestion_worker_process,
            args=(user_id,),
            name=f"Worker-{user_id}"
        )
        processes.append(process)
        process.start()

    logger.info("All workers have been spawned. Press Ctrl+C to stop.")
    
    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        logger.info("Terminating all workers...")
        for process in processes:
            process.terminate()
        for process in processes:
            process.join()