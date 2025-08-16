import os
import time
import logging
from colorama import Fore, Style, init

from file_processors.text_file_processor import TextFileProcessor
from indexing_and_embedding.chroma_db_client import ChromaClient

# Initialize colorama for colored terminal output
init(autoreset=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize persistent vector DB
chroma_client = ChromaClient(
    collection_name="all_users_docs",
    embedding_model_name=embedding_model
)

def get_response_for_user(user_id: str, query: str, debug: bool = True):
    qa_user = chroma_client.get_user_qa(user_id)
    result = qa_user.invoke({"query": query})
    return result

# Terminal-based chat interface
def chat_interface():
    print(Fore.CYAN + Style.BRIGHT + "\n🚀 FS-RAG Chatbot (Terminal Edition) 🚀")
    print(Fore.YELLOW + "Type your questions below. Type 'exit' to quit.\n")

    print(Fore.GREEN + "Enter your user_id: ")
    user_id = input().strip()
    print(Fore.MAGENTA + f"\nHello, {user_id}! Let's start chatting...\n")
    qa_user = chroma_client.get_user_qa(user_id)

    while True:
        print(Fore.MAGENTA + f"{user_id}, Enter your query > ")
        query = input().strip()
        if query.lower() in ["exit", "quit", "q"]:
            print(Fore.CYAN + "\nGoodbye! 👋\n")
            break

        try:
            start = time.time()
            result = qa_user.invoke({"query": query})
            end = time.time()

            print(Fore.BLUE + "\n[Answer]:" + Fore.WHITE, result['result'])
            if "source_documents" in result:
                print(Fore.YELLOW + "\n[Sources]:")
                for i, doc in enumerate(result["source_documents"], 1):
                    snippet = (doc.page_content[:120] + "...") if len(doc.page_content) > 120 else doc.page_content
                    print(Fore.LIGHTBLACK_EX + f"  {i}. {snippet}")
                    print(Fore.LIGHTBLACK_EX + f"     Source: {doc.metadata.get('source', 'N/A')}")

            print(Fore.GREEN + f"\n⏱ Answered in {end-start:.2f} seconds")
            print(Fore.CYAN + "-" * 60)

        except Exception as e:
            logger.error(f"Error during query: {e}")
            print(Fore.RED + "⚠ Something went wrong. Please try again.")

if __name__ == "__main__":
    chat_interface()
