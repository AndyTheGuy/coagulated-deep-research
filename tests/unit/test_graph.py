import pytest
from core.models import GraphState, SubQuestion
from core.graph import compile_graph, route_after_verification

def test_graph_compiles():
    """Verify that the StateGraph compiles successfully without errors."""
    app = compile_graph()
    assert app is not None

def test_route_after_verification_gaps():
    """Verify that we route back to the supervisor when there are pending sub-questions (gaps)."""
    state_with_gaps = GraphState(
        topic="AI Safety",
        sub_questions_state=[
            SubQuestion(id="q1", question="What is safety?", status="completed"),
            SubQuestion(id="gap_1", question="What about alignment?", status="pending")
        ]
    )
    route = route_after_verification(state_with_gaps)
    assert route == "supervisor_node"

def test_route_after_verification_no_gaps():
    """Verify that we route to the report writer when there are no pending sub-questions."""
    state_no_gaps = GraphState(
        topic="AI Safety",
        sub_questions_state=[
            SubQuestion(id="q1", question="What is safety?", status="completed")
        ]
    )
    route = route_after_verification(state_no_gaps)
    assert route == "report_writer_node"
