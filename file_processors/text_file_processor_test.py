#Vibe-Coded
import pytest
import logging
from unittest.mock import patch, MagicMock

# Mock external dependencies before importing your module
with patch.dict('sys.modules', {
    'langchain_community.document_loaders': MagicMock(),
    'langchain.text_splitter': MagicMock()
}):
    from file_processors.text_file_processor import TextFileProcessor
def test_process_files_creates_chunks():
    fake_file_path = "data/fake_user/text/valid.txt"

    # Fake document and chunk objects
    fake_document = MagicMock()
    fake_document.page_content = "This is some test content for chunking."
    fake_document.metadata = {}

    fake_chunk = MagicMock()
    fake_chunk.page_content = "This is a chunk."
    fake_chunk.metadata = {}

    processor = TextFileProcessor(chunk_size=10, chunk_overlap=2)

    with patch("os.path.isfile", return_value=True), \
         patch("file_processors.text_file_processor.TextLoader") as mock_loader_class, \
         patch.object(processor.text_splitter, "split_documents", return_value=[fake_chunk]):

        mock_loader_instance = mock_loader_class.return_value
        mock_loader_instance.load.return_value = [fake_document]

        chunks = processor.process_files([fake_file_path])

        # Assertions updated to match __get_file_metadata
        assert len(chunks) == 1
        assert chunks[0].page_content == "This is a chunk."
        assert "file_path" in chunks[0].metadata
        assert chunks[0].metadata["file_path"] == fake_file_path
        assert "user" in chunks[0].metadata
        assert chunks[0].metadata["user"] == "fake_user"
        assert "mime_type" in chunks[0].metadata
        assert chunks[0].metadata["mime_type"] == "text"

def test_process_files_skips_invalid_files(caplog):
    processor = TextFileProcessor()
    invalid_file_path = "invalid_file.pdf"

    with patch("os.path.isfile", return_value=False):
        with caplog.at_level(logging.WARNING):
            chunks = processor.process_files([invalid_file_path])
            # Assertions
            assert len(chunks) == 0
            assert f"Skipping invalid file: {invalid_file_path}" in caplog.text
