import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from llm import LLMConfig
from providers.io_provider import IOProvider
from llm.plugins.rag_openai import OpenAILLMRag


class DummyOutputModel(BaseModel):
    test_field: str

@pytest.fixture
def config():
    """Fixture for a sample LLMConfig"""
    return LLMConfig(
        api_key="YOUR OPENMIND API KEY",
        openai_api_key= "YOUR OPENAI API KEY",
        base_url="https://api.openmind.org/api/core/openai",
        embedding_model="text-embedding-3-large",
        doc_directory="./test_docs",
    )


@pytest.fixture
def mock_response():
    """Mock API response"""
    return {
        "choices": [{"message": {"content": '{"test_field": "success"}'}}]
    }


@pytest.fixture
def llm(config):
    """Fixture to initialize OpenAILLMRag"""
    return OpenAILLMRag(DummyOutputModel, config)


def test_init_with_config(config):
    """Ensure OpenAILLMRag initializes correctly"""
    llm = OpenAILLMRag(DummyOutputModel, config)
    
    assert llm.api_key == "YOUR OPENMIND API KEY"
    assert llm.openai_api_key == "YOUR OPENAI API KEY"
    assert llm.doc_directory == "./test_docs"
    assert llm.embedding_model == "text-embedding-3-large"


def test_init_empty_key():
    """Ensure ValueError is raised when api_key is missing"""
    with pytest.raises(ValueError, match="config file missing api_key"):
        OpenAILLMRag(DummyOutputModel, LLMConfig(openai_api_key="YOUR OPENAI API KEY"))


@pytest.mark.asyncio
@patch("llm.plugins.openai_llm_rag.FAISS.load_local")
@patch("llm.plugins.openai_llm_rag.openai.AsyncClient")
async def test_ask_success(mock_openai_client, mock_faiss, llm, mock_response):
    """Ensure ask() returns parsed response on success"""

    # Mock FAISS document retrieval
    mock_doc = MagicMock()
    mock_doc.page_content = "Mocked document content"
    mock_faiss.return_value.as_retriever.return_value.invoke.return_value = [mock_doc]

    # Mock OpenAI API response
    mock_openai_client.return_value.beta.chat.completions.parse = AsyncMock(
        return_value=mock_response
    )

    # Mock IOProvider ASR input
    llm.io_provider.inputs = {"ASRInput": MagicMock(input="Sample question")}

    response = await llm.ask("Sample question")

    assert response is not None
    assert response.test_field == "success"


@pytest.mark.asyncio
@patch("llm.plugins.openai_llm_rag.openai.AsyncClient")
async def test_ask_invalid_json(mock_openai_client, llm):
    """Ensure ask() returns None when response JSON is invalid"""

    # Mock OpenAI API with invalid JSON
    mock_openai_client.return_value.beta.chat.completions.parse = AsyncMock(
        return_value={"choices": [{"message": {"content": "invalid_json"}}]}
    )

    llm.io_provider.inputs = {"ASRInput": MagicMock(input="Sample question")}

    response = await llm.ask("Sample question")

    assert response is None


@pytest.mark.asyncio
@patch("llm.plugins.openai_llm_rag.openai.AsyncClient")
async def test_ask_api_error(mock_openai_client, llm):
    """Ensure ask() returns None when OpenAI API call fails"""

    # Mock API raising an exception
    mock_openai_client.return_value.beta.chat.completions.parse.side_effect = Exception(
        "API Error"
    )

    llm.io_provider.inputs = {"ASRInput": MagicMock(input="Sample question")}

    response = await llm.ask("Sample question")

    assert response is None


@pytest.mark.asyncio
@patch("llm.plugins.openai_llm_rag.openai.AsyncClient")
async def test_io_provider_timing(mock_openai_client, llm, mock_response):
    """Ensure io_provider timing is set correctly"""

    mock_openai_client.return_value.beta.chat.completions.parse = AsyncMock(
        return_value=mock_response
    )

    llm.io_provider.inputs = {"ASRInput": MagicMock(input="Sample question")}

    start_time = time.time()
    await llm.ask("Sample question")
    end_time = time.time()

    assert llm.io_provider.llm_start_time >= start_time
    assert llm.io_provider.llm_end_time >= llm.io_provider.llm_start_time
    assert llm.io_provider.llm_end_time <= end_time
