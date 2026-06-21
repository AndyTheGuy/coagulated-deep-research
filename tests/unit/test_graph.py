import pytest
from core.models import GraphState, SubQuestion, DREMEvaluation, MetricScore
from core.graph import compile_graph, route_after_verification, route_after_evaluation

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

def test_route_after_evaluation_pass():
    """Verify routing after evaluation when overall_passed is True."""
    kic = MetricScore(metric_name="KIC", score=0.9, threshold=0.8, passed=True)
    rq = MetricScore(metric_name="RQ", score=0.85, threshold=0.8, passed=True)
    fact = MetricScore(metric_name="Factuality", score=0.95, threshold=0.9, passed=True)
    evaluation = DREMEvaluation(
        key_information_coverage=kic,
        reasoning_quality=rq,
        factuality=fact,
        overall_passed=True
    )
    state = GraphState(
        topic="AI Safety",
        evaluation=evaluation
    )
    route = route_after_evaluation(state)
    assert route == "__end__"

def test_route_after_evaluation_fail_with_pending():
    """Verify routing after evaluation when overall_passed is False and there are pending questions."""
    kic = MetricScore(metric_name="KIC", score=0.7, threshold=0.8, passed=False)
    rq = MetricScore(metric_name="RQ", score=0.85, threshold=0.8, passed=True)
    fact = MetricScore(metric_name="Factuality", score=0.95, threshold=0.9, passed=True)
    evaluation = DREMEvaluation(
        key_information_coverage=kic,
        reasoning_quality=rq,
        factuality=fact,
        overall_passed=False
    )
    state = GraphState(
        topic="AI Safety",
        evaluation=evaluation,
        sub_questions_state=[
            SubQuestion(id="q1", question="What is safety?", status="completed"),
            SubQuestion(id="eval_gap_1", question="remediation query", status="pending")
        ]
    )
    route = route_after_evaluation(state)
    assert route == "supervisor_node"

def test_route_after_evaluation_fail_no_pending():
    """Verify routing after evaluation when overall_passed is False but there are no pending questions."""
    kic = MetricScore(metric_name="KIC", score=0.7, threshold=0.8, passed=False)
    rq = MetricScore(metric_name="RQ", score=0.85, threshold=0.8, passed=True)
    fact = MetricScore(metric_name="Factuality", score=0.95, threshold=0.9, passed=True)
    evaluation = DREMEvaluation(
        key_information_coverage=kic,
        reasoning_quality=rq,
        factuality=fact,
        overall_passed=False
    )
    state = GraphState(
        topic="AI Safety",
        evaluation=evaluation,
        sub_questions_state=[
            SubQuestion(id="q1", question="What is safety?", status="completed")
        ]
    )
    route = route_after_evaluation(state)
    assert route == "__end__"

