# indexing_and_embedding/chroma_db_client_test.py
import pytest
import sys
from unittest.mock import MagicMock, patch

# -----------------------------
# Mock external dependencies BEFORE importing ChromaClient
# -----------------------------
sys.modules["chromadb"] = MagicMock()
sys.modules["langchain_community.vectorstores"] = MagicMock()
sys.modules["langchain_community.embeddings"] = MagicMock()
sys.modules["langchain.chains"] = MagicMock()
sys.modules["langchain_huggingface"] = MagicMock()
sys.modules["transformers"] = MagicMock()

from indexing_and_embedding.chroma_db_client import ChromaClient

# -----------------------------
# Simple tests
# -----------------------------
def test_chroma_client_initialization_simple():
    with patch("indexing_and_embedding.chroma_db_client.HuggingFaceEmbeddings") as MockEmbeddings, \
         patch("indexing_and_embedding.chroma_db_client.HttpClient") as MockHttpClient, \
         patch("indexing_and_embedding.chroma_db_client.Chroma") as MockChroma:

        mock_embedding_instance = MagicMock()
        MockEmbeddings.return_value = mock_embedding_instance

        mock_vectordb = MagicMock()
        MockChroma.return_value = mock_vectordb
        MockHttpClient.return_value = MagicMock()

        client = ChromaClient(collection_name="test_collection")

        # Check that embedding and vectordb are assigned
        assert client.embedding_model == mock_embedding_instance
        assert client.vectordb == mock_vectordb

        # Check Chroma called correctly
        MockChroma.assert_called_once_with(
            client=MockHttpClient.return_value,
            collection_name="test_collection",
            embedding_function=mock_embedding_instance,
        )


def test_add_documents_simple():
    with patch("indexing_and_embedding.chroma_db_client.HuggingFaceEmbeddings"), \
         patch("indexing_and_embedding.chroma_db_client.HttpClient"), \
         patch("indexing_and_embedding.chroma_db_client.Chroma") as MockChroma:

        mock_vectordb = MagicMock()
        MockChroma.return_value = mock_vectordb

        client = ChromaClient(batch_size=2)
        docs = ["doc1", "doc2", "doc3"]
        client.add_documents(docs)

        # Should call add_documents in 2 batches
        assert mock_vectordb.add_documents.call_count == 2
        assert mock_vectordb.add_documents.call_args_list[0][0][0] == ["doc1", "doc2"]
        assert mock_vectordb.add_documents.call_args_list[1][0][0] == ["doc3"]


def test_add_documents_empty_simple(caplog):
    with patch("indexing_and_embedding.chroma_db_client.HuggingFaceEmbeddings"), \
         patch("indexing_and_embedding.chroma_db_client.HttpClient"), \
         patch("indexing_and_embedding.chroma_db_client.Chroma"):

        client = ChromaClient()
        with caplog.at_level("WARNING"):
            client.add_documents([])

        assert "[ChromaClient] No documents to add." in caplog.text


def test_get_user_retriever_simple():
    with patch("indexing_and_embedding.chroma_db_client.HuggingFaceEmbeddings"), \
         patch("indexing_and_embedding.chroma_db_client.HttpClient"), \
         patch("indexing_and_embedding.chroma_db_client.Chroma") as MockChroma:

        mock_vectordb = MagicMock()
        mock_retriever = MagicMock()
        mock_vectordb.as_retriever.return_value = mock_retriever
        MockChroma.return_value = mock_vectordb

        client = ChromaClient()
        retriever = client.get_user_retriever("user_a", top_k=3)

        mock_vectordb.as_retriever.assert_called_once_with(
            search_kwargs={"k": 3, "filter": {"user_id": "user_a"}}
        )
        assert retriever == mock_retriever
