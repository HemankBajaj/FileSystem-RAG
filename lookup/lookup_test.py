# lookup/lookup_test.py
import sys
import pytest
from unittest.mock import MagicMock, patch

# --- Patch heavy dependencies so imports don’t break ---
sys.modules["chromadb"] = MagicMock()
sys.modules["langchain.chains"] = MagicMock()
sys.modules["langchain_huggingface"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["langchain_community"] = MagicMock()
sys.modules["langchain_community.vectorstores"] = MagicMock()
sys.modules["langchain_community.embeddings"] = MagicMock()

from lookup.lookup import Lookup


@pytest.fixture
def mock_chroma_client():
    """Fixture for a fake ChromaClient."""
    return MagicMock()


def test_lookup_initialization(mock_chroma_client):
    """It should initialize Lookup with a mocked pipeline + chroma client."""
    with patch("lookup.lookup.pipeline", return_value=MagicMock()) as mock_pipeline, \
         patch("lookup.lookup.HuggingFacePipeline", return_value=MagicMock()) as mock_hf_pipeline:

        lookup = Lookup(mock_chroma_client)

        mock_pipeline.assert_called_once()
        mock_hf_pipeline.assert_called_once()
        assert lookup.chroma_db_client == mock_chroma_client


def test_get_qa_returns_chain(mock_chroma_client):
    """It should create a RetrievalQA chain with the retriever."""
    fake_retriever = MagicMock()
    fake_chain = MagicMock()

    with patch("lookup.lookup.RetrievalQA.from_chain_type", return_value=fake_chain) as mock_from_chain:
        lookup = Lookup(mock_chroma_client)
        chain = lookup.get_qa(fake_retriever)

        mock_from_chain.assert_called_once()
        assert chain == fake_chain


def test_generate_response_calls_retriever_and_chain(mock_chroma_client):
    """It should call retriever and return QA chain result."""
    fake_retriever = MagicMock()
    mock_chroma_client.get_user_retriever.return_value = fake_retriever
    fake_result = {"result": "answer"}
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = fake_result

    with patch("lookup.lookup.RetrievalQA.from_chain_type", return_value=fake_chain):
        lookup = Lookup(mock_chroma_client)
        result = lookup.generate_reponse("user_1", "test query")

        mock_chroma_client.get_user_retriever.assert_called_once_with("user_1", 5)
        fake_chain.invoke.assert_called_once_with({"query": "test query"})
        assert result == fake_result
