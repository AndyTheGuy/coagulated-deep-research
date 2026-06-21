from unittest.mock import AsyncMock
import pytest
from core.models import GraphState
from core.nodes.scoping import clarify_with_user_node, write_research_brief_node
from core.nodes import scoping  # import to monkeypatch the singleton router

@pytest.mark.asyncio
async def test_scoping_flow_with_ambiguity(monkeypatch):
    """Test scoping flow when user query is ambiguous, requiring a clarification step."""
    mock_ainvoke = AsyncMock()
    # First call is for clarify_with_user_node, second for write_research_brief_node
    mock_ainvoke.side_effect = [
        AsyncMock(
            content='{"clarification_needed": true, "clarifying_question": "Which quantum key distribution protocols?"}',
            usage_metadata={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}
        ),
        AsyncMock(
            content='{"topic": "Quantum Key Distribution", "scope": "Detailed scope of QKD", "constraints": ["use local papers"], "sub_questions": [{"id": "q1", "question": "What is QKD security?"}], "target_source_count": 20}',
            usage_metadata={"input_tokens": 15, "output_tokens": 15, "total_tokens": 30}
        )
    ]
    monkeypatch.setattr(scoping.router, "ainvoke", mock_ainvoke)

    # 1. Ambiguity detection
    state = GraphState(user_query="Quantum Cryptography")
    res1 = await clarify_with_user_node(state)
    assert res1["clarification_needed"] is True
    assert res1["clarification_question"] == "Which quantum key distribution protocols?"

    # Update state
    state.clarification_needed = res1["clarification_needed"]
    state.clarification_question = res1["clarification_question"]
    state.clarification_response = "Focus on BB84 and Decoy State protocols."

    # 2. Brief generation
    res2 = await write_research_brief_node(state)
    brief = res2["research_brief"]
    sub_questions = res2["sub_questions_state"]

    assert brief.topic == "Quantum Key Distribution"
    assert brief.target_source_count == 20
    assert len(sub_questions) == 1
    assert sub_questions[0].id == "q1"
    assert sub_questions[0].question == "What is QKD security?"

@pytest.mark.asyncio
async def test_scoping_flow_specific_query(monkeypatch):
    """Test scoping flow when user query is specific, skipping the clarification loop."""
    mock_ainvoke = AsyncMock()
    mock_ainvoke.side_effect = [
        AsyncMock(
            content='{"clarification_needed": false, "clarifying_question": null}',
            usage_metadata={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}
        ),
        AsyncMock(
            content='{"topic": "Quantum BB84 Protocol", "scope": "In-scope: BB84 protocol details", "constraints": [], "sub_questions": [{"id": "q1", "question": "Explain BB84 protocol."}], "target_source_count": 15}',
            usage_metadata={"input_tokens": 15, "output_tokens": 15, "total_tokens": 30}
        )
    ]
    monkeypatch.setattr(scoping.router, "ainvoke", mock_ainvoke)

    # 1. Ambiguity detection
    state = GraphState(user_query="How does the BB84 protocol work?")
    res1 = await clarify_with_user_node(state)
    assert res1["clarification_needed"] is False

    # Update state
    state.clarification_needed = res1["clarification_needed"]

    # 2. Brief generation
    res2 = await write_research_brief_node(state)
    brief = res2["research_brief"]
    sub_questions = res2["sub_questions_state"]

    assert brief.topic == "Quantum BB84 Protocol"
    assert len(sub_questions) == 1
    assert sub_questions[0].question == "Explain BB84 protocol."

@pytest.mark.asyncio
async def test_scoping_flow_fallback_on_error(monkeypatch):
    """Test that a parsing error in brief generation triggers a fallback brief instead of crashing."""
    mock_ainvoke = AsyncMock()
    # Return invalid JSON for the brief generation to trigger parser exception
    mock_ainvoke.side_effect = [
        AsyncMock(
            content='{"clarification_needed": false}',
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        ),
        AsyncMock(
            content='invalid json output that fails parser',
            usage_metadata={"input_tokens": 15, "output_tokens": 15, "total_tokens": 30}
        )
    ]
    monkeypatch.setattr(scoping.router, "ainvoke", mock_ainvoke)

    state = GraphState(user_query="BB84 Protocol", topic="BB84 Protocol")
    
    # Ambiguity check
    res1 = await clarify_with_user_node(state)
    assert res1["clarification_needed"] is False
    
    # Brief generation (with parsing error)
    res2 = await write_research_brief_node(state)
    
    assert "research_brief" in res2
    assert "sub_questions_state" in res2
    assert res2["research_brief"].topic == "BB84 Protocol"
    assert "Fallback scope" in res2["research_brief"].scope
    assert len(res2["errors"]) == 1
    assert "Scoping failed" in res2["errors"][0]
