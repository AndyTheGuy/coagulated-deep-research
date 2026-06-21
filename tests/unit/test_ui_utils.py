import pytest
from core.models import GraphState, SubQuestion, SearchResult, VerifiedSource
from ui.utils import (
    update_graph_state_with_chunk,
    update_cost_stats_from_state,
    estimate_remaining_cost
)

def test_update_graph_state_with_chunk_basic_and_lists():
    """Verify state update from chunks for single fields, standard lists, and custom merged attributes."""
    state = GraphState()
    
    # 1. Update basic and list fields
    chunk1 = {
        "scoping_ambiguity_check": {
            "topic": "Superconductivity Research",
            "logs": ["Scoping start", "Ambiguity analyzed"],
            "errors": ["Warning: check timeout"]
        }
    }
    
    update_graph_state_with_chunk(state, chunk1)
    
    assert state.topic == "Superconductivity Research"
    assert state.logs == ["Scoping start", "Ambiguity analyzed"]
    assert state.errors == ["Warning: check timeout"]
    
    # 2. Update with additional chunk, appending to lists
    chunk2 = {
        "supervisor_node": {
            "logs": ["Supervisor assigned tasks"],
            "errors": ["Warning: retry search"]
        }
    }
    
    update_graph_state_with_chunk(state, chunk2)
    
    assert state.topic == "Superconductivity Research"  # Unchanged
    assert state.logs == ["Scoping start", "Ambiguity analyzed", "Supervisor assigned tasks"]
    assert state.errors == ["Warning: check timeout", "Warning: retry search"]

def test_update_graph_state_sub_questions_reducer():
    """Verify that sub_questions_state uses custom reducers, updating existing and appending new ones."""
    state = GraphState()
    
    q1 = SubQuestion(id="q1", question="What is critical temperature?", status="pending")
    q2 = SubQuestion(id="q2", question="Is mercury a superconductor?", status="pending")
    
    chunk1 = {
        "supervisor_node": {
            "sub_questions_state": [q1, q2]
        }
    }
    
    update_graph_state_with_chunk(state, chunk1)
    assert len(state.sub_questions_state) == 2
    assert state.sub_questions_state[0].status == "pending"
    
    # Update q1 status to completed, and add a new sub-question q3
    q1_updated = SubQuestion(id="q1", question="What is critical temperature?", status="completed")
    q3 = SubQuestion(id="q3", question="History of cuprates", status="pending")
    
    chunk2 = {
        "researcher_node": {
            "sub_questions_state": [q1_updated, q3]
        }
    }
    
    update_graph_state_with_chunk(state, chunk2)
    
    # Should merge them based on ID
    sub_qs = {q.id: q for q in state.sub_questions_state}
    assert len(sub_qs) == 3
    assert sub_qs["q1"].status == "completed"
    assert sub_qs["q2"].status == "pending"
    assert sub_qs["q3"].status == "pending"

def test_update_graph_state_token_usage_reducer():
    """Verify token usage accumulations via dict merge reducer."""
    state = GraphState()
    state.token_usage = {
        "vertex_ai": {"input_tokens": 100, "output_tokens": 50, "calls": 1},
        "freellmapi": {"input_tokens": 500, "output_tokens": 200, "calls": 2},
        "failovers": 0
    }
    
    chunk = {
        "verifier_node": {
            "token_usage": {
                "vertex_ai": {"input_tokens": 200, "output_tokens": 100, "calls": 1},
                "failovers": 1
            }
        }
    }
    
    update_graph_state_with_chunk(state, chunk)
    
    assert state.token_usage["vertex_ai"]["input_tokens"] == 300
    assert state.token_usage["vertex_ai"]["output_tokens"] == 150
    assert state.token_usage["vertex_ai"]["calls"] == 2
    assert state.token_usage["freellmapi"]["input_tokens"] == 500  # Unchanged
    assert state.token_usage["failovers"] == 1

def test_update_cost_stats_from_state():
    """Verify conversion of token usage counts to costs using 2026 Gemini 3.5 Flash pricing."""
    state = GraphState()
    state.token_usage = {
        "vertex_ai": {
            "input_tokens": 2000000,   # 2M input
            "output_tokens": 1000000,  # 1M output
            "calls": 5
        },
        "freellmapi": {
            "input_tokens": 5000000,
            "output_tokens": 2000000,
            "calls": 12
        },
        "failovers": 2
    }
    
    cost_stats = {}
    update_cost_stats_from_state(state, cost_stats)
    
    assert cost_stats["vertex_calls"] == 5
    assert cost_stats["vertex_input"] == 2000000
    assert cost_stats["vertex_output"] == 1000000
    
    # Vertex cost: 2M * $0.0375 / 1M + 1M * $0.15 / 1M
    # Input cost: 2 * 0.0375 = 0.075
    # Output cost: 1 * 0.15 = 0.150
    # Total expected cost: 0.225
    assert pytest.approx(cost_stats["vertex_cost"], 1e-6) == 0.225
    
    assert cost_stats["freellm_calls"] == 12
    assert cost_stats["freellm_input"] == 5000000
    assert cost_stats["freellm_output"] == 2000000
    assert cost_stats["failovers"] == 2

def test_estimate_remaining_cost():
    """Verify cost projection and fallback estimates for remaining tasks."""
    cost_stats = {"vertex_cost": 0.10}
    state_empty = GraphState()
    
    # 1. Empty subquestions list fallback
    assert estimate_remaining_cost(state_empty, cost_stats) == 0.0015
    
    # 2. 0 completed subquestions, estimates $0.0005 per remaining subquestion
    state_init = GraphState()
    state_init.sub_questions_state = [
        SubQuestion(id="q1", question="Q1", status="pending"),
        SubQuestion(id="q2", question="Q2", status="in_progress")
    ]
    # 2 remaining * 0.0005 = 0.0010
    assert estimate_remaining_cost(state_init, cost_stats) == 0.0010
    
    # 3. Partially completed subquestions
    state_partial = GraphState()
    state_partial.sub_questions_state = [
        SubQuestion(id="q1", question="Q1", status="completed"),
        SubQuestion(id="q2", question="Q2", status="failed"),
        SubQuestion(id="q3", question="Q3", status="pending"),
        SubQuestion(id="q4", question="Q4", status="in_progress")
    ]
    # Completed count = 2 (q1, q2)
    # Remaining count = 2 (q3, q4)
    # Current cost = 0.10
    # Average cost per subquestion = 0.10 / 2 = 0.05
    # Projected remaining cost = 0.05 * 2 = 0.10
    assert estimate_remaining_cost(state_partial, cost_stats) == 0.10
