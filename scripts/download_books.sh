#!/bin/bash

# Note : This script has been vibe-coded!!

# Output directory
OUTPUT_DIR="/home/hemankbajaj/Desktop/Projects/FileSystem-RAG/data/books"

# Make sure the directory exists
mkdir -p "$OUTPUT_DIR"

# Step 1: Fetch top 100 book IDs into book_ids.txt
curl -s https://www.gutenberg.org/browse/scores/top \
  | grep -o '/ebooks/[0-9]\+' \
  | sed 's/\/ebooks\///' \
  | head -n 100 > book_ids.txt

# Step 2: Loop through each ID and download the book
while read -r id; do
  echo "Downloading book $id..."

  # Try plain .txt first
  URL="https://www.gutenberg.org/cache/epub/$id/pg$id.txt"
  curl -s -f "$URL" -o "$OUTPUT_DIR/$id.txt"

  # If plain .txt failed, try .utf8
  if [ $? -ne 0 ]; then
    URL="https://www.gutenberg.org/cache/epub/$id/pg$id.txt.utf8"
    curl -s -f "$URL" -o "$OUTPUT_DIR/$id.txt"
  fi

  # Check final status
  if [ $? -eq 0 ]; then
    echo "✅ Saved $OUTPUT_DIR/$id.txt"
  else
    echo "❌ Failed for $id"
  fi
done < book_ids.txt
