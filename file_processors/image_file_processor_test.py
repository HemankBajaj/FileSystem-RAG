# file_processors/test_image_file_processor.py
# Vibe-Coded
import pytest
import logging
from unittest.mock import patch, MagicMock

# Mock external dependencies before importing your module
with patch.dict("sys.modules", {
    "langchain.docstore.document": MagicMock(),
    "langchain.text_splitter": MagicMock(),
}):
    from file_processors.image_file_processor import ImageFileProcessor


@pytest.fixture
def processor():
    # Patch out time.sleep to avoid waiting in tests
    with patch("time.sleep", return_value=None):
        yield ImageFileProcessor(description_file="tests/fake_image_descriptions.csv")


def test_process_files_with_description(processor):
    fake_file_path = "data/fake_user/images/cat.png"

    # Mock descriptions dictionary
    processor.descriptions = {"cat.png": "A cute cat sitting on a sofa."}

    fake_doc = MagicMock()
    fake_doc.page_content = ""
    fake_doc.metadata = {}

    with patch("os.path.isfile", return_value=True), \
         patch.object(processor.text_splitter, "create_documents", return_value=[fake_doc]):

        docs = processor.process_files([fake_file_path])

        assert len(docs) == 1
        assert "A cute cat sitting on a sofa." in docs[0].page_content
        assert docs[0].metadata["file_path"] == fake_file_path
        assert docs[0].metadata["user_id"] == "fake_user"
        assert docs[0].metadata["mime_type"] == "image"


def test_process_files_without_description(processor, caplog):
    fake_file_path = "data/fake_user/images/dog.png"

    # Empty descriptions dictionary (no entry for dog.png)
    processor.descriptions = {}

    fake_doc = MagicMock()
    fake_doc.page_content = ""
    fake_doc.metadata = {}

    with patch("os.path.isfile", return_value=True), \
         patch.object(processor.text_splitter, "create_documents", return_value=[fake_doc]):

        with caplog.at_level(logging.WARNING):
            docs = processor.process_files([fake_file_path])

            assert len(docs) == 1
            assert "[NOT FOUND]" in docs[0].page_content
            assert "Description not found for dog.png" in caplog.text
            assert docs[0].metadata["file_path"] == fake_file_path
            assert docs[0].metadata["user_id"] == "fake_user"


def test_process_files_skips_invalid_files(processor, caplog):
    invalid_file_path = "data/fake_user/images/not_an_image.txt"

    with patch("os.path.isfile", return_value=True):  # File exists but wrong extension
        with caplog.at_level(logging.WARNING):
            docs = processor.process_files([invalid_file_path])
            assert len(docs) == 0
            assert f"Skipping invalid image file: {invalid_file_path}" in caplog.text
