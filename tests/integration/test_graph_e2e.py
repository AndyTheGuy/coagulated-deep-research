from unittest.mock import AsyncMock
import pytest
from core.models import GraphState
from core.graph import compile_graph
from core.nodes import scoping  # to monkeypatch scoping.router

@pytest.mark.asyncio
async def test_compiled_graph_specific_query_run(monkeypatch):
    """Test that the compiled graph completes scoping on a specific query."""
    # 1. Compile the graph
    app = compile_graph()
    
    # 2. Mock the LLM router inside scoping
    mock_ainvoke = AsyncMock()
    mock_ainvoke.side_effect = [
        AsyncMock(
            content='{"clarification_needed": false, "clarifying_question": null}',
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        ),
        AsyncMock(
            content='{"topic": "Quantum Key Distribution", "scope": "Detailed scope of QKD", "constraints": ["use local papers"], "sub_questions": [{"id": "q1", "question": "What is QKD security?"}], "target_source_count": 20}',
            usage_metadata={"input_tokens": 15, "output_tokens": 15, "total_tokens": 30}
        )
    ]
    monkeypatch.setattr(scoping.router, "ainvoke", mock_ainvoke)
    
    # 3. Invoke graph
    initial_state = {
        "user_query": "Explain how Quantum Key Distribution (QKD) works.",
        "topic": "Quantum Key Distribution"
    }
    final_state = await app.ainvoke(initial_state)
    
    assert final_state["clarification_needed"] is False
    assert final_state["research_brief"] is not None
    assert final_state["research_brief"].topic == "Quantum Key Distribution"
    assert len(final_state["sub_questions_state"]) == 1
    assert final_state["sub_questions_state"][0].id == "q1"

@pytest.mark.asyncio
async def test_compiled_graph_ambiguous_query_run(monkeypatch):
    """Test that the compiled graph exits early for clarification on an ambiguous query."""
    # 1. Compile the graph
    app = compile_graph()
    
    # 2. Mock the LLM router
    mock_ainvoke = AsyncMock()
    mock_ainvoke.side_effect = [
        AsyncMock(
            content='{"clarification_needed": true, "clarifying_question": "What specific aspect of Quantum Computing?"}',
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        )
    ]
    monkeypatch.setattr(scoping.router, "ainvoke", mock_ainvoke)
    
    # 3. Invoke graph
    initial_state = {
        "user_query": "Quantum Computing",
        "topic": "Quantum Computing"
    }
    final_state = await app.ainvoke(initial_state)
    
    assert final_state["clarification_needed"] is True
    assert final_state["clarification_question"] == "What specific aspect of Quantum Computing?"
    assert final_state.get("research_brief") is None
