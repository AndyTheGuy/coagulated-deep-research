import pytest
from unittest.mock import MagicMock, AsyncMock
from langchain_core.messages import AIMessage

@pytest.fixture
def mock_llm_calls(monkeypatch):
    """Fixture to mock ChatVertexAI and ChatOpenAI responses."""
    mock_vertex_instance = MagicMock()
    mock_vertex_instance.model = "gemini-1.5-flash"
    mock_vertex_instance.ainvoke = AsyncMock(return_value=AIMessage(
        content="Mocked Vertex AI response",
        usage_metadata={"input_tokens": 15, "output_tokens": 8, "total_tokens": 23}
    ))

    mock_openai_instance = MagicMock()
    mock_openai_instance.model_name = "gpt-4o-mini"
    mock_openai_instance.ainvoke = AsyncMock(return_value=AIMessage(
        content="Mocked FreeLLMAPI response",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    ))

    mock_vertex_cls = MagicMock(return_value=mock_vertex_instance)
    mock_openai_cls = MagicMock(return_value=mock_openai_instance)

    monkeypatch.setattr("core.llm_router.ChatVertexAI", mock_vertex_cls)
    monkeypatch.setattr("core.llm_router.ChatOpenAI", mock_openai_cls)

    return mock_vertex_instance, mock_openai_instance
