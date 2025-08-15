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


# Download the 100 books from project Gutenberg
download_books:
	@echo "Downloading 100 books from Gutenberg"
	@bash $(BOOKS_DOWNLOAD_SCRIPT)

clean: