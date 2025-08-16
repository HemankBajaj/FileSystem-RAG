# Vibe-coded
import pytest
from unittest.mock import patch, MagicMock

# --- MOCK external dependencies before importing ChromaClient ---
patcher_chroma = patch.dict("sys.modules", {
    "chromadb": MagicMock(),
    "langchain_community.vectorstores": MagicMock(),
    "langchain_community.embeddings": MagicMock(),
    "langchain.chains": MagicMock(),
    "langchain_huggingface": MagicMock(),
    "transformers": MagicMock()
})
patcher_chroma.start()

# Import after patching
from indexing_and_embedding.chroma_db_client import ChromaClient

patcher_chroma.stop()


@pytest.fixture
def mock_chroma_and_client():
    with patch("indexing_and_embedding.chroma_db_client.Chroma") as MockChroma, \
         patch("indexing_and_embedding.chroma_db_client.HttpClient") as MockHttpClient, \
         patch("indexing_and_embedding.chroma_db_client.HuggingFaceEmbeddings") as MockEmbeddings:

        mock_vectordb = MagicMock()
        MockChroma.return_value = mock_vectordb
        MockHttpClient.return_value = MagicMock()
        MockEmbeddings.return_value = MagicMock()

        yield MockChroma, mock_vectordb


def test_chroma_client_initialization(mock_chroma_and_client):
    MockChroma, mock_vectordb = mock_chroma_and_client

    client = ChromaClient(collection_name="test_collection")

    MockChroma.assert_called_once()
    assert client.vectordb == mock_vectordb


def test_add_documents_batches(mock_chroma_and_client):
    _, mock_vectordb = mock_chroma_and_client
    client = ChromaClient(batch_size=2)

    docs = ["doc1", "doc2", "doc3"]
    client.add_documents(docs)

    # Should be called twice: [doc1, doc2], then [doc3]
    assert mock_vectordb.add_documents.call_count == 2
    first_call = mock_vectordb.add_documents.call_args_list[0][0][0]
    second_call = mock_vectordb.add_documents.call_args_list[1][0][0]

    assert first_call == ["doc1", "doc2"]
    assert second_call == ["doc3"]


def test_add_documents_empty(mock_chroma_and_client, caplog):
    _, mock_vectordb = mock_chroma_and_client
    client = ChromaClient()

    with caplog.at_level("WARNING"):
        client.add_documents([])

    assert "[ChromaClient] No documents to add." in caplog.text
    mock_vectordb.add_documents.assert_not_called()


def test_get_user_qa_creates_chain(mock_chroma_and_client):
    _, mock_vectordb = mock_chroma_and_client
    mock_retriever = MagicMock()
    mock_vectordb.as_retriever.return_value = mock_retriever

    with patch("indexing_and_embedding.chroma_db_client.pipeline") as MockPipeline, \
         patch("indexing_and_embedding.chroma_db_client.HuggingFacePipeline") as MockHFPipeline, \
         patch("indexing_and_embedding.chroma_db_client.RetrievalQA.from_chain_type") as MockRetrievalQA:

        mock_pipeline_instance = MagicMock()
        MockPipeline.return_value = mock_pipeline_instance

        mock_llm = MagicMock()
        MockHFPipeline.return_value = mock_llm

        mock_qa_chain = MagicMock()
        MockRetrievalQA.return_value = mock_qa_chain

        client = ChromaClient()
        qa = client.get_user_qa("user_a")

        mock_vectordb.as_retriever.assert_called_once_with(
            search_kwargs={"k": 5, "filter": {"user_id": "user_a"}}
        )
        MockPipeline.assert_called_once()
        MockHFPipeline.assert_called_once_with(pipeline=mock_pipeline_instance)
        MockRetrievalQA.assert_called_once_with(
            llm=mock_llm, retriever=mock_retriever, return_source_documents=True
        )
        assert qa == mock_qa_chain
