import pytest
from unittest.mock import patch, MagicMock

# --- MOCK external dependencies before importing ChromaClient ---
patcher_chroma = patch.dict("sys.modules", {
    "langchain_community.vectorstores": MagicMock(),
    "langchain_community.embeddings": MagicMock(),
    "langchain.chains": MagicMock(),
    "langchain_huggingface": MagicMock(),
    "transformers": MagicMock()
})
patcher_chroma.start()

# Now import ChromaClient safely
from indexing_and_embedding.chroma_db_client import ChromaClient

# Stop patching after import
patcher_chroma.stop()


@pytest.fixture
def mock_chroma_add_documents():
    with patch("indexing_and_embedding.chroma_db_client.Chroma") as MockChroma:
        mock_vectordb = MagicMock()
        MockChroma.return_value = mock_vectordb
        yield MockChroma, mock_vectordb

@pytest.fixture
def mock_huggingface_pipeline():
    with patch("indexing_and_embedding.chroma_db_client.pipeline") as MockPipeline:
        mock_pipeline_instance = MagicMock()
        MockPipeline.return_value = mock_pipeline_instance
        yield MockPipeline, mock_pipeline_instance

def test_chroma_client_initialization(mock_chroma_add_documents):
    MockChroma, mock_vectordb = mock_chroma_add_documents

    client = ChromaClient(persist_directory="test_dir", collection_name="test_collection")
    
    MockChroma.assert_called_once_with(
        collection_name="test_collection",
        embedding_function=client.embedding_model,
        persist_directory="test_dir"
    )
    assert client.vectordb == mock_vectordb

def test_add_documents_calls_vectordb_persist(mock_chroma_add_documents):
    _, mock_vectordb = mock_chroma_add_documents
    client = ChromaClient()
    
    docs = ["doc1", "doc2"]
    client.add_documents(docs)
    
    mock_vectordb.add_documents.assert_called_once_with(docs)
    mock_vectordb.persist.assert_called_once()

def test_add_documents_empty(mock_chroma_add_documents):
    _, _ = mock_chroma_add_documents
    client = ChromaClient()
    
    client.add_documents([])

def test_get_user_qa_creates_chain(mock_chroma_add_documents, mock_huggingface_pipeline):
    _, mock_vectordb = mock_chroma_add_documents
    _, mock_pipeline_instance = mock_huggingface_pipeline
    
    mock_retriever = MagicMock()
    mock_vectordb.as_retriever.return_value = mock_retriever
    
    with patch("indexing_and_embedding.chroma_db_client.HuggingFacePipeline") as MockHuggingFacePipeline:
        mock_llm = MagicMock()
        MockHuggingFacePipeline.return_value = mock_llm
        
        with patch("indexing_and_embedding.chroma_db_client.RetrievalQA.from_chain_type") as MockRetrievalQA:
            mock_qa_chain = MagicMock()
            MockRetrievalQA.return_value = mock_qa_chain
            
            client = ChromaClient()
            qa = client.get_user_qa("user_a")
            
            mock_vectordb.as_retriever.assert_called_once_with(
                search_kwargs={"k": 5, "filter": {"user_id": "user_a"}}
            )
            assert qa == mock_qa_chain
