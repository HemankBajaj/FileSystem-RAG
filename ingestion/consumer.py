import redis
import logging
import time
import os
import multiprocessing
import sys
import json
from datetime import datetime

# To resolve import issue
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexing_and_embedding.chroma_db_client import ChromaClient
from langchain.schema import Document

# Redis Stream Config
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
STREAM_KEY = 'ingestion_stream_chunks'
CONSUMER_GROUP = 'ingestion_workers_group'
CONSUMER_NAME_PREFIX = 'worker'
NUM_WORKERS = 4
# Chroma DB Config
persist_directory = "chroma_store"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
# Batch size for processing messages
BATCH_SIZE = 5000
# Timeout for claiming old messages (in milliseconds)
CLAIM_TIMEOUT_MS = 60000 


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IngestionConsumer:
    def __init__(self, consumer_name=None):
        self.consumer_name = consumer_name if consumer_name else f"{CONSUMER_NAME_PREFIX}-{os.getpid()}"

        self.redis_client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.chroma_db_client = ChromaClient(collection_name="all_users_docs", embedding_model_name=embedding_model)
        try:
            self.redis_client.ping()
            logger.info(
                f"[Consumer] '{self.consumer_name}' connected to Redis.")
        except redis.exceptions.ConnectionError as e:
            logger.error(
                f"[Consumer] '{self.consumer_name}' could not connect to Redis: {e}")
            raise

        self._create_consumer_group()

    def _create_consumer_group(self):
        try:
            self.redis_client.xgroup_create(
                STREAM_KEY, CONSUMER_GROUP, id='0', mkstream=True)
            logger.info(
                f"[Consumer] Created consumer group '{CONSUMER_GROUP}' on stream '{STREAM_KEY}'.")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise e
            else:
                logger.info(
                    f"[Consumer] Consumer group '{CONSUMER_GROUP}' already exists.")

    def _get_chunk_count_key(self, user_id: str, file_path: str):
        return f"file_chunks_processed:{user_id}:{os.path.basename(file_path)}"

    def _get_processed_files_key(self, user_id: str):
        return f"processed_files:{user_id}"

    def _process_chunk_batch(self, message_list, is_pending=False):
        documents_to_add = []
        message_ids_to_ack = []
        
        for message_id, message_data in message_list:
            try:
                page_content = message_data.get('page_content')
                metadata_str = message_data.get('metadata')
                if not page_content or not metadata_str:
                    logger.error(f"[Consumer] Invalid message format received: {message_data}. Skipping.")
                    continue
                metadata = json.loads(metadata_str)
                document = Document(page_content=page_content, metadata=metadata)
                documents_to_add.append(document)
                message_ids_to_ack.append(message_id)
            except Exception as e:
                logger.error(f"[Consumer] Failed to process message {message_id}: {e}")
                continue
        
        if documents_to_add:
            try:
                self.chroma_db_client.add_documents(documents_to_add)
                logger.info(f"[Consumer] Successfully added {len(documents_to_add)} documents to ChromaDB.")
                
                for doc in documents_to_add:
                    user_id = doc.metadata.get('user_id')
                    file_path = doc.metadata.get('file_path')
                    total_chunks = int(doc.metadata.get('total_chunks'))
                    if user_id and file_path and total_chunks:
                        chunk_count_key = self._get_chunk_count_key(user_id, file_path)
                        processed_count = self.redis_client.hincrby(chunk_count_key, 'processed_count', 1)
                        if processed_count == total_chunks:
                            processed_files_key = self._get_processed_files_key(user_id)
                            self.redis_client.sadd(processed_files_key, file_path)
                            logger.info(f"File '{file_path}' for user '{user_id}' is now COMPLETE (all {total_chunks} chunks processed).")
                            self.redis_client.delete(chunk_count_key)
            except Exception as e:
                logger.error(f"[Consumer] Failed to add documents to ChromaDB or update chunk counts: {e}")
                return []
        
        return message_ids_to_ack

    def _process_pending_messages(self):
        """Claims and processes pending messages from other consumers."""
        # Use '0-0' as the start ID to claim all old messages
        start_id = '0-0'
        # Claim up to BATCH_SIZE messages that are older than CLAIM_TIMEOUT_MS
        claimed_messages = self.redis_client.xautoclaim(
            STREAM_KEY,
            CONSUMER_GROUP,
            self.consumer_name,
            CLAIM_TIMEOUT_MS,
            start_id,
            count=BATCH_SIZE
        )

        if claimed_messages and claimed_messages[1]:
            logger.info(f"[Consumer] Found {len(claimed_messages[1])} pending messages. Processing...")
            message_ids_to_ack = self._process_chunk_batch(claimed_messages[1], is_pending=True)
            if message_ids_to_ack:
                self.redis_client.xack(STREAM_KEY, CONSUMER_GROUP, *message_ids_to_ack)
                logger.info(f"[Consumer] Acknowledged {len(message_ids_to_ack)} claimed messages.")
        else:
            logger.info("[Consumer] No pending messages to claim.")


    def run(self):
        loop_count = 0
        while True:
            try:
                # 1. Occasionally process pending messages (every ~1 min)
                if loop_count % 60 == 0:  
                    self._process_pending_messages()

                # 2. Read new messages with smaller batches
                messages = self.redis_client.xreadgroup(
                    CONSUMER_GROUP,
                    self.consumer_name,
                    {STREAM_KEY: '>'},
                    count=BATCH_SIZE,
                    block=1000
                )

                if messages and messages[0][1]:
                    message_ids_to_ack = self._process_chunk_batch(messages[0][1])
                    if message_ids_to_ack:
                        self.redis_client.xack(STREAM_KEY, CONSUMER_GROUP, *message_ids_to_ack)
                        logger.info(f"[Consumer] Acked {len(message_ids_to_ack)} new messages.")
                else:
                    logger.debug("[Consumer] No new messages, waiting...")

            except Exception as e:
                logger.error(f"[Consumer] Error in loop: {e}", exc_info=True)
                time.sleep(2)  # backoff

            loop_count += 1


# --------------------------------------------------------------------------------------------------

def start_worker(worker_id):
    worker_name = f"{CONSUMER_NAME_PREFIX}-{worker_id}"
    consumer = IngestionConsumer(consumer_name=worker_name)
    consumer.run()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    processes = []
    logger.info(f"Starting {NUM_WORKERS} consumer workers... 🚀")
    
    for i in range(NUM_WORKERS):
        process = multiprocessing.Process(target=start_worker, args=(i,))
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