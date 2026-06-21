import pytest
from unittest.mock import AsyncMock
from langchain_core.messages import HumanMessage, AIMessage
from core.llm_router import LLMRouter, _extract_token_usage

@pytest.mark.asyncio
async def test_critical_routes_to_vertex(mock_llm_calls):
    """Test that CRITICAL tier directly invokes Vertex AI and tracks its tokens."""
    mock_vertex, mock_openai = mock_llm_calls
    router = LLMRouter()
    
    messages = [HumanMessage(content="Hello")]
    response = await router.ainvoke(messages, tier="CRITICAL", agent_name="test-agent")
    
    assert response.content == "Mocked Vertex AI response"
    mock_vertex.ainvoke.assert_called_once_with(messages)
    mock_openai.ainvoke.assert_not_called()
    
    # Check token tracking
    assert router.token_usage["vertex_ai"]["calls"] == 1
    assert router.token_usage["vertex_ai"]["input_tokens"] == 15
    assert router.token_usage["vertex_ai"]["output_tokens"] == 8
    assert router.token_usage["freellmapi"]["calls"] == 0
    assert router.token_usage["failovers"] == 0

@pytest.mark.asyncio
async def test_standard_routes_to_freellmapi(mock_llm_calls):
    """Test that STANDARD tier invokes FreeLLMAPI and tracks its tokens."""
    mock_vertex, mock_openai = mock_llm_calls
    router = LLMRouter()
    
    messages = [HumanMessage(content="Hello")]
    response = await router.ainvoke(messages, tier="STANDARD", agent_name="test-agent")
    
    assert response.content == "Mocked FreeLLMAPI response"
    mock_openai.ainvoke.assert_called_once_with(messages)
    mock_vertex.ainvoke.assert_not_called()
    
    # Check token tracking
    assert router.token_usage["freellmapi"]["calls"] == 1
    assert router.token_usage["freellmapi"]["input_tokens"] == 10
    assert router.token_usage["freellmapi"]["output_tokens"] == 5
    assert router.token_usage["vertex_ai"]["calls"] == 0
    assert router.token_usage["failovers"] == 0

@pytest.mark.asyncio
async def test_failover_to_vertex_on_failure(mock_llm_calls):
    """Test that a failure in FreeLLMAPI triggers a fallback to Vertex AI."""
    mock_vertex, mock_openai = mock_llm_calls
    
    # Make FreeLLMAPI fail
    mock_openai.ainvoke = AsyncMock(side_effect=RuntimeError("API Down"))
    
    router = LLMRouter()
    messages = [HumanMessage(content="Hello")]
    
    # Invoke standard call (which will fail and fall back)
    response = await router.ainvoke(messages, tier="STANDARD", agent_name="test-agent")
    
    assert response.content == "Mocked Vertex AI response"
    mock_openai.ainvoke.assert_called_once_with(messages)
    mock_vertex.ainvoke.assert_called_once_with(messages)
    
    # Check token tracking
    assert router.token_usage["failovers"] == 1
    assert router.token_usage["freellmapi"]["calls"] == 0
    assert router.token_usage["vertex_ai"]["calls"] == 1
    assert router.token_usage["vertex_ai"]["input_tokens"] == 15
    assert router.token_usage["vertex_ai"]["output_tokens"] == 8

def test_extract_token_usage():
    """Test that token extraction handles different response metadata formats."""
    # Test standard usage_metadata attribute
    msg1 = AIMessage(content="abc", usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30})
    assert _extract_token_usage(msg1) == (10, 20)
    
    # Test fallback token_usage dict in response_metadata
    msg2 = AIMessage(content="abc")
    msg2.response_metadata = {"token_usage": {"prompt_tokens": 5, "completion_tokens": 15}}
    assert _extract_token_usage(msg2) == (5, 15)
    
    # Test Vertex alternative usage_metadata dict in response_metadata
    msg3 = AIMessage(content="abc")
    msg3.response_metadata = {"usage_metadata": {"prompt_token_count": 8, "candidates_token_count": 12}}
    assert _extract_token_usage(msg3) == (8, 12)
    
    # Test fallback empty cases
    msg4 = AIMessage(content="abc")
    assert _extract_token_usage(msg4) == (0, 0)
