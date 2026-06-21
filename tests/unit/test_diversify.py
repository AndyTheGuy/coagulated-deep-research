import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage
from search.diversify import diversify_query

@pytest.mark.asyncio
async def test_diversify_query_success():
    # Mock LLMRouter
    mock_router = AsyncMock()
    
    # Return a mocked AIMessage containing valid JSON matching the QueryVariants schema
    mock_response = AIMessage(
        content='{"variants": ["gpt4o performance benchmarks", "gpt-4o vs gemini 1.5 pro", "openai gpt4o speed speed test"]}'
    )
    mock_router.ainvoke.return_value = mock_response
    
    question = "What are the latest performance benchmarks of GPT-4o?"
    variants = await diversify_query(question, num_variants=3, router=mock_router)
    
    assert len(variants) == 3
    assert "gpt4o performance benchmarks" in variants
    assert "gpt-4o vs gemini 1.5 pro" in variants
    assert "openai gpt4o speed speed test" in variants
    
    # Verify mock router invocation args
    mock_router.ainvoke.assert_called_once()
    call_kwargs = mock_router.ainvoke.call_args[1]
    assert call_kwargs["tier"] == "BULK"
    assert call_kwargs["agent_name"] == "SearchAgent"
    assert call_kwargs["node_name"] == "diversify_query"

@pytest.mark.asyncio
async def test_diversify_query_invalid_json_fallback():
    mock_router = AsyncMock()
    # Bad JSON content
    mock_response = AIMessage(content="This is not a JSON object, it's plain text.")
    mock_router.ainvoke.return_value = mock_response
    
    question = "How to use Reciprocal Rank Fusion?"
    variants = await diversify_query(question, router=mock_router)
    
    # Should fall back to the original question list on JSON parsing error
    assert len(variants) == 1
    assert variants[0] == question

@pytest.mark.asyncio
async def test_diversify_query_llm_failure_fallback():
    mock_router = AsyncMock()
    # Mock LLM raising a runtime error
    mock_router.ainvoke.side_effect = RuntimeError("API service overloaded")
    
    question = "How to implement search deduplication?"
    variants = await diversify_query(question, router=mock_router)
    
    # Should fall back to the original question list on LLM exception
    assert len(variants) == 1
    assert variants[0] == question

@pytest.mark.asyncio
async def test_diversify_query_empty_input():
    variants = await diversify_query("")
    assert variants == []
    
    variants = await diversify_query("   ")
    assert variants == []
