import asyncio
import json
import logging
import os
from queue import Empty
from unittest.mock import patch, MagicMock

import pytest

from inputs.plugins.rag import RagInput
from inputs.base import SensorConfig


@pytest.fixture
def dummy_config(tmp_path):
    """
    Provide a minimal SensorConfig to avoid using real directories/files.
    This sets the data directory to pytest's tmp_path.
    """
    return SensorConfig(
        data_path=str(tmp_path),
        embedding_model="text-embedding-3-large",
        openai_api_key="fake_openai_api_key",
        chunk_size=100,
        chunk_overlap=10
    )


@pytest.fixture
def rag_input(dummy_config):
    """
    Create a RagInput instance with the dummy config.
    We'll patch out actual network/disk operations.
    """
    with patch("inputs.plugins.rag.OpenAIEmbeddings") as mock_embeddings, \
         patch("inputs.plugins.rag.faiss") as mock_faiss, \
         patch("inputs.plugins.rag.FAISS") as mock_faiss_class:
        # Mock embeddings so we don't call real OpenAI
        mock_embeddings.return_value.embed_query.return_value = [0.0, 0.0, 0.0]

        # Mock FAISS to avoid real disk or memory usage
        mock_faiss.IndexFlatL2.return_value = MagicMock()
        mock_faiss_class.load_local.return_value = MagicMock()
        mock_faiss_class.return_value = MagicMock()

        instance = RagInput(config=dummy_config)

    return instance


def test_init(rag_input, dummy_config):
    """
    Test that RagInput initializes correctly:
    - buffer and message_queue are empty
    - config-driven attributes are set
    - embeddings and vector_store are not None
    """
    assert rag_input.buffer == []
    assert rag_input.message_buffer.empty()
    assert rag_input.context is None
    assert rag_input.session is None
    assert rag_input.embeddings is not None
    assert rag_input.vector_store is not None
    assert rag_input.doc_directory == dummy_config.data_path


@pytest.mark.asyncio
async def test_poll_with_message(rag_input):
    """
    Test that _poll returns a message if there's one in the queue.
    """
    test_message = "Test poll message"
    rag_input.message_buffer.put_nowait(test_message)
    result = await rag_input._poll()
    assert result == test_message


@pytest.mark.asyncio
async def test_poll_empty_queue(rag_input):
    """
    Test that _poll returns None if the queue is empty.
    """
    result = await rag_input._poll()
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_input(rag_input):
    """
    If we pass raw_input, raw_to_text should add it to the buffer
    and return it.
    """
    msg = "incoming message"
    result = await rag_input.raw_to_text(msg)
    assert result == msg
    assert rag_input.buffer[-1] == msg


@pytest.mark.asyncio
async def test_raw_to_text_no_input(rag_input):
    """
    If no raw_input is passed but the queue has a message, 
    raw_to_text should process that message.
    """
    queued_msg = "message in queue"
    rag_input.message_buffer.put_nowait(queued_msg)
    result = await rag_input.raw_to_text()
    assert result == queued_msg
    assert rag_input.buffer[-1] == queued_msg


@pytest.mark.asyncio
async def test__raw_to_text(rag_input):
    """
    Same as raw_to_text, but for the internal _raw_to_text method.
    """
    msg = "internal raw message"
    result = await rag_input._raw_to_text(msg)
    assert result == msg
    assert rag_input.buffer[-1] == msg


def test_formatted_latest_buffer(rag_input):
    """
    If no context is set, we use the last item in buffer.
    If context is set, we use that.
    """
    # Initially: no buffer, no context
    assert rag_input.formatted_latest_buffer() is None

    # Add something to buffer
    rag_input.buffer.append("buffer message")
    result = rag_input.formatted_latest_buffer()
    assert "buffer message" in result

    # If context is set, that takes precedence
    rag_input.context = "CONTEXT OVERRIDE"
    result = rag_input.formatted_latest_buffer()
    assert "CONTEXT OVERRIDE" in result


@pytest.mark.asyncio
async def test_async_context_manager(rag_input):
    """
    Test the async context manager __aenter__ / __aexit__
    to ensure it creates and closes an aiohttp.ClientSession.
    """
    # We'll mock aiohttp.ClientSession to verify usage
    with patch("inputs.plugins.rag.aiohttp.ClientSession") as mock_session:
        session_instance = mock_session.return_value
        async with rag_input as instance:
            assert instance.session == session_instance

        # After the 'with' block, we should have closed the session
        session_instance.close.assert_awaited_once()




def test_initialize_vector_store_(rag_input, tmp_path):
    """
    If checkpoint and index exist, and the file list hasn't changed,
    we load existing store instead of rebuilding.
    """
    checkpoint_path = tmp_path / "faiss_index_checkpoint.json"
    index_path = tmp_path / "faiss_index"
    
    # Write a trivial checkpoint
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(["myfile.pdf"], f)

    def mock_exists(path):
        # Both index and checkpoint exist
        return str(path) in [str(checkpoint_path), str(index_path)]

    # File list that matches "myfile.pdf"
    with patch.object(rag_input, 'doc_directory', str(tmp_path)), \
         patch("inputs.plugins.rag.os.path.exists", side_effect=mock_exists), \
         patch.object(rag_input, '_get_data_files', return_value=["myfile.pdf"]), \
         patch("inputs.plugins.rag.FAISS.load_local") as mock_load:

        store = rag_input._initialize_vector_store()
        mock_load.assert_called_once_with(
            str(index_path), rag_input.embeddings, allow_dangerous_deserialization=True
        )
        assert store == mock_load.return_value


def test_initialize_vector_store_changed_files(rag_input, tmp_path):
    """
    If checkpoint+index exist, but the file list changed, 
    we expect to rebuild the index.
    """
    checkpoint_path = tmp_path / "faiss_index_checkpoint.json"
    index_path = tmp_path / "faiss_index"

    # Old checkpoint
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(["old.pdf"], f)

    def mock_exists(path):
        return str(path) in [str(checkpoint_path), str(index_path)]

    # The new file list is different
    new_files = ["new.pdf"]

    with patch.object(rag_input, 'doc_directory', str(tmp_path)), \
         patch("inputs.plugins.rag.os.path.exists", side_effect=mock_exists), \
         patch.object(rag_input, '_get_data_files', return_value=new_files), \
         patch("inputs.plugins.rag.DirectoryLoader") as mock_loader, \
         patch("inputs.plugins.rag.RecursiveCharacterTextSplitter") as mock_splitter, \
         patch("inputs.plugins.rag.FAISS") as mock_faiss_class:

        mock_loader.return_value.load.return_value = []
        mock_splitter.return_value.split_documents.return_value = []

        store = rag_input._initialize_vector_store()
        # Rebuild => FAISS(...) is called
        mock_faiss_class.assert_called_once()
        assert store == mock_faiss_class.return_value
