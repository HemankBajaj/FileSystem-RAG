# This simulates the file uploading process for our files.
# Distribute books to each user and wait for 3 minutes before distributing the next book.

# Vibe-Coded 


import os
import shutil
import time
from math import ceil

# Source directory containing books
SOURCE_DIR = "data/books"

# Destination directories for users
USERS = ["user_a", "user_b", "user_c", "user_d", "user_e"]
DEST_BASE_DIR = "data"

# Interval between rounds (in seconds)
INTERVAL = 3 * 60  # 3 minutes

# Create user directories if they don't exist
for user in USERS:
    dest_dir = os.path.join(DEST_BASE_DIR, user, "text")
    os.makedirs(dest_dir, exist_ok=True)

# Get list of all books
books = [f for f in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, f))]
books.sort()  # optional: keep order stable

# Distribute books evenly into dictionary
user_books = {user: [] for user in USERS}
for i, book in enumerate(books):
    user = USERS[i % len(USERS)]
    user_books[user].append(book)

# Calculate number of rounds
rounds = ceil(len(books) / len(USERS))

# Distribute books round by round
for r in range(rounds):
    print(f"\n--- Round {r+1} ---")
    for user in USERS:
        if r < len(user_books[user]):
            book = user_books[user][r]
            src_path = os.path.join(SOURCE_DIR, book)
            dest_path = os.path.join(DEST_BASE_DIR, user, "text", book)
            
            shutil.copy2(src_path, dest_path)
            print(f"Copied '{book}' to {user}/text")
    
    # Wait only if more rounds left
    if r < rounds - 1:
        print(f"Waiting {INTERVAL/60} minutes before next round...")
        time.sleep(INTERVAL)

print("\n✅ All books distributed successfully!")
