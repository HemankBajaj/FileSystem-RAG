PROJECT_ROOT=/home/hemankbajaj/Desktop/Projects/FileSystem-RAG
BOOKS_DATA_DIR=$(PROJECT_ROOT)/data/books
BOOKS_DOWNLOAD_SCRIPT=$(PROJECT_ROOT)/scripts/download_books.sh


.PHONY: install download_books


install:
	@echo "📦 Installing project dependencies..."
	uv sync

run:
	@echo "🚀 Running main file"
	uv run python main.py

unittests:
	@echo "🧪 Running Python Unittests..."
	pytest -v --maxfail=1 --disable-warnings
	@echo "✅ Unittests completed!"

# Download the 100 books from project Gutenberg
download_books:
	@echo "Downloading 100 books from Gutenberg"
	@bash $(BOOKS_DOWNLOAD_SCRIPT)

start-redis-server:
	@echo "🛑 Stopping Prviously Running Redis Server"
	systemctl stop redis-server
	@echo "🚀 Starting Redis Server"
	redis-server

start-chroma-server:
	@echo "🚀 Starting Chroma Server"
	uv run chroma run --path ./chroma_store

ingestion-producer:
	@echo "Starting Event Producer for Ingestion Files"
	uv run python ingestion/producer.py

ingestion-consumer:
	@echo "Starting Event Producer for Ingestion Files"
	uv run python ingestion/consumer.py

clean-file-distribution:
	@echo "🧹 Cleaning distributed files..."
	@rm -rf data/user_*/text/*
	@echo "✅ All user text directories cleaned!"


clean:
	@echo "🧹 Cleaning temporary files..."
	@find . -name "__pycache__" -type d -exec rm -rf {} +
	@find . -name "*.pyc" -exec rm -f {} +

clean-all: clean clean-file-distribution
	@echo "🧹 Cleaning Chroma DB Store"
	rm -rf chroma_store
	@echo "🧹 Cleaning Redis Data Dump"
	rm dump.rdb
