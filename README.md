# FileSystem RAG

This project implements a FileSystem RAG capable of high throughput ingestion (via a fault tolerant event-driven architecture). 
Realtime ingestion trains stores indexes in a vector index database. This vector index database is used to augment LLM responses by providing information from files stored by a user.
## Setup

```bash
uv sync
```

## Data Setup
```bash
pip install foobar
```

## Data Setup

Download Books from Project Gutenburg.
```
make download_books
```
Download Images
```
<drive_link>
```

## Start File Ingestion Service
Start Redis (Please install redis first)
```bash
make start-redis-server
```
Start Chroma DB Server (Make sure you do `uv sync` before).
```bash
make start-chroma-server
```
Start Ingestion Consumers (Read Chunks written by producers)
```bash
make ingestion-consumer
```
Start Ingestion Producers (Poll for updates in FS and write to redis stream)
```bash
make ingestion-producer
```

Now, you are ready to add files. As of now, we support image and text files ingestion. :)
Add Test Files : `./data/user_x/text/`
Add Test Files : `./data/user_x/image/`

## Start Lookup Client
```bash
make run
```

This starts an interactive chat session for a user (Not supporting authentication as of now) to send queries.
top-k is set to 5 by default.

## Tests
```bash
make unittests
```
Integration Tests are not included as of now.

# Clean
To clean the vector DB and files.
Uncomment `clean-all` in `Makefile` and run it. After that, run `redis-cli flushall` to remove persistent data from redis.

