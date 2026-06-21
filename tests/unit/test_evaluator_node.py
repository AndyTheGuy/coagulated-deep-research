import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from core.models import GraphState, Report, SubQuestion, DREMEvaluation, MetricScore, Claim
from core.nodes.evaluator import evaluator_node

def make_router_ainvoke_mock(
    key_facts=None,
    coverage=None,
    reasoning=None,
    factuality=None,
    remediation=None,
    raise_for_node=None
):
    async def side_effect(*args, **kwargs):
        node_name = kwargs.get("node_name")
        if raise_for_node == node_name:
            raise ValueError(f"Simulated error in {node_name}")
            
        if node_name == "extract_key_facts":
            content = key_facts or '{"key_facts": ["fact1", "fact2"]}'
            return AIMessage(content=content)
        elif node_name == "check_coverage":
            content = coverage or '{"coverage_items": [{"fact": "fact1", "covered": true, "explanation": "ok"}, {"fact": "fact2", "covered": false, "explanation": "missing"}]}'
            return AIMessage(content=content)
        elif node_name == "evaluate_reasoning":
            content = reasoning or '{"score": 0.8, "explanation": "good reasoning"}'
            return AIMessage(content=content)
        elif node_name == "evaluate_factuality":
            content = factuality or '{"score": 0.95, "explanation": "correct citations"}'
            return AIMessage(content=content)
        elif node_name == "remediation_formulation":
            content = remediation or '{"gaps_found": true, "remediation_queries": ["What about fact2?"], "remediation_notes": "failed coverage"}'
            return AIMessage(content=content)
        else:
            return AIMessage(content="{}")
            
    return side_effect

@pytest.mark.asyncio
async def test_evaluator_node_no_final_report():
    """Verify that evaluator_node bypasses evaluation when no final report is available."""
    state = GraphState(topic="AI Safety", final_report=None)
    updates = await evaluator_node(state)
    assert "logs" in updates
    assert any("Bypassed evaluation" in log for log in updates["logs"])
    assert "evaluation" not in updates

@pytest.mark.asyncio
async def test_evaluator_node_passing_scenario():
    """Verify evaluator_node passes when all metrics exceed thresholds."""
    state = GraphState(
        topic="AI Safety",
        final_report=Report(
            title="AI Safety Report",
            content="This is a fully compliant AI safety report with outstanding citation formatting and reasoning.",
            claims=[Claim(claim_id="c1", claim_text="AI safety is crucial", section="Introduction", confidence_score=0.95)],
            citations=["https://safety.org"]
        )
    )

    # All metrics should pass:
    # KIC = 1.0 (both fact1 and fact2 covered)
    # RQ = 0.85
    # Factuality = 0.7 * 0.95 + 0.3 * 0.95 = 0.95
    # Thresholds: KIC >= 0.8, RQ >= 0.75, Factuality >= 0.90
    mock_ainvoke = AsyncMock(side_effect=make_router_ainvoke_mock(
        coverage='{"coverage_items": [{"fact": "fact1", "covered": true, "explanation": "ok"}, {"fact": "fact2", "covered": true, "explanation": "ok"}]}',
        reasoning='{"score": 0.85, "explanation": "excellent reasoning"}',
        factuality='{"score": 0.95, "explanation": "perfect citations"}'
    ))

    with patch("core.nodes.scoping.router.ainvoke", new=mock_ainvoke):
        updates = await evaluator_node(state)

    assert "evaluation" in updates
    evaluation = updates["evaluation"]
    assert isinstance(evaluation, DREMEvaluation)
    assert evaluation.overall_passed is True
    assert evaluation.key_information_coverage.passed is True
    assert evaluation.key_information_coverage.score == 1.0
    assert evaluation.reasoning_quality.passed is True
    assert evaluation.reasoning_quality.score == 0.85
    assert evaluation.factuality.passed is True
    assert evaluation.factuality.score == 0.95

    # No new sub-questions should be returned
    assert len(updates.get("sub_questions_state", [])) == 0

@pytest.mark.asyncio
async def test_evaluator_node_failing_remediation_scenario():
    """Verify evaluator_node fails and adds remediation questions when metrics are below thresholds."""
    state = GraphState(
        topic="AI Safety",
        final_report=Report(
            title="AI Safety Report",
            content="This is a partial report.",
            claims=[Claim(claim_id="c1", claim_text="AI safety is crucial", section="Introduction", confidence_score=0.5)],
            citations=["https://safety.org"]
        )
    )

    # KIC fails: 1 of 2 covered = 0.5 (< 0.8)
    mock_ainvoke = AsyncMock(side_effect=make_router_ainvoke_mock(
        coverage='{"coverage_items": [{"fact": "fact1", "covered": true, "explanation": "ok"}, {"fact": "fact2", "covered": false, "explanation": "missing"}]}',
        remediation='{"gaps_found": true, "remediation_queries": ["What about fact2?", "How does fact2 compare?"], "remediation_notes": "failed coverage"}'
    ))

    with patch("core.nodes.scoping.router.ainvoke", new=mock_ainvoke):
        updates = await evaluator_node(state)

    assert "evaluation" in updates
    evaluation = updates["evaluation"]
    assert evaluation.overall_passed is False
    assert evaluation.key_information_coverage.passed is False
    assert evaluation.key_information_coverage.score == 0.5

    # Remediations should be populated with new sub-questions with IDs starting with eval_gap_
    assert "sub_questions_state" in updates
    new_qs = updates["sub_questions_state"]
    assert len(new_qs) == 2
    assert new_qs[0].id == "eval_gap_1"
    assert new_qs[0].question == "What about fact2?"
    assert new_qs[0].status == "pending"
    assert new_qs[1].id == "eval_gap_2"
    assert new_qs[1].question == "How does fact2 compare?"
    assert new_qs[1].status == "pending"

@pytest.mark.asyncio
async def test_evaluator_node_max_loop_threshold():
    """Verify that evaluator_node forces approval when max loop threshold is reached to prevent infinite loops."""
    existing_gaps = [
        SubQuestion(id="eval_gap_1", question="gap 1", status="completed"),
        SubQuestion(id="eval_gap_2", question="gap 2", status="completed"),
        SubQuestion(id="eval_gap_3", question="gap 3", status="completed"),
        SubQuestion(id="eval_gap_4", question="gap 4", status="completed")
    ]
    state = GraphState(
        topic="AI Safety",
        sub_questions_state=existing_gaps,
        final_report=Report(
            title="AI Safety Report",
            content="This is a partial report.",
            claims=[Claim(claim_id="c1", claim_text="AI safety is crucial", section="Introduction", confidence_score=0.5)],
            citations=["https://safety.org"]
        )
    )

    # Coverage score would fail (0.5), but 4 existing eval gap sub-questions should force a pass
    mock_ainvoke = AsyncMock(side_effect=make_router_ainvoke_mock(
        coverage='{"coverage_items": [{"fact": "fact1", "covered": true, "explanation": "ok"}, {"fact": "fact2", "covered": false, "explanation": "missing"}]}'
    ))

    with patch("core.nodes.scoping.router.ainvoke", new=mock_ainvoke):
        updates = await evaluator_node(state)

    assert "evaluation" in updates
    evaluation = updates["evaluation"]
    assert evaluation.overall_passed is True  # Forced to pass
    assert len(updates.get("sub_questions_state", [])) == 0  # No new subquestions formulated

@pytest.mark.asyncio
async def test_evaluator_node_json_parsing_fallbacks():
    """Verify evaluator_node's robust fallback behaviors when LLM responses return invalid JSON."""
    state = GraphState(
        topic="AI Safety",
        final_report=Report(
            title="AI Safety Report",
            content="Partial report.",
            claims=[Claim(claim_id="c1", claim_text="AI safety is crucial", section="Introduction", confidence_score=0.5)],
            citations=["https://safety.org"]
        )
    )

    # 1. Test fallback when extract_key_facts returns invalid json
    # It should fallback to default facts, continue check_coverage, and still proceed.
    mock_ainvoke_facts_err = AsyncMock(side_effect=make_router_ainvoke_mock(
        key_facts="invalid_json_completely",
        coverage='{"coverage_items": [{"fact": "fallback fact", "covered": true, "explanation": "ok"}]}'
    ))

    with patch("core.nodes.scoping.router.ainvoke", new=mock_ainvoke_facts_err):
        updates = await evaluator_node(state)
    assert updates["evaluation"].key_information_coverage.passed is True
    assert updates["evaluation"].key_information_coverage.score == 1.0

    # 2. Test fallback when check_coverage returns invalid json
    # It should fallback to score 0.85 (passed=True)
    mock_ainvoke_cov_err = AsyncMock(side_effect=make_router_ainvoke_mock(
        coverage="invalid_json"
    ))
    with patch("core.nodes.scoping.router.ainvoke", new=mock_ainvoke_cov_err):
        updates = await evaluator_node(state)
    assert updates["evaluation"].key_information_coverage.passed is True
    assert updates["evaluation"].key_information_coverage.score == 0.85

    # 3. Test fallback when evaluate_reasoning returns invalid json
    # It should fallback to score 0.80 (passed=True)
    mock_ainvoke_rq_err = AsyncMock(side_effect=make_router_ainvoke_mock(
        reasoning="invalid_json"
    ))
    with patch("core.nodes.scoping.router.ainvoke", new=mock_ainvoke_rq_err):
        updates = await evaluator_node(state)
    assert updates["evaluation"].reasoning_quality.passed is True
    assert updates["evaluation"].reasoning_quality.score == 0.80

    # 4. Test fallback when evaluate_factuality returns invalid json
    # It should fallback to llm_fact_score 0.90
    mock_ainvoke_fact_err = AsyncMock(side_effect=make_router_ainvoke_mock(
        factuality="invalid_json"
    ))
    with patch("core.nodes.scoping.router.ainvoke", new=mock_ainvoke_fact_err):
        updates = await evaluator_node(state)
    expected_combined_factuality = (0.7 * 0.5) + (0.3 * 0.90)  # = 0.35 + 0.27 = 0.62
    assert updates["evaluation"].factuality.score == pytest.approx(expected_combined_factuality)

    # 5. Test fallback when remediation_formulation returns invalid json
    # It should fallback to adding a single default evaluator remediation sub-question
    mock_ainvoke_remediat_err = AsyncMock(side_effect=make_router_ainvoke_mock(
        coverage='{"coverage_items": [{"fact": "fact1", "covered": false, "explanation": "missing"}]}', # KIC fails -> 0.0
        remediation="invalid_json"
    ))
    with patch("core.nodes.scoping.router.ainvoke", new=mock_ainvoke_remediat_err):
        updates = await evaluator_node(state)
    
    assert updates["evaluation"].overall_passed is False
    assert len(updates["sub_questions_state"]) == 1
    default_q = updates["sub_questions_state"][0]
    assert default_q.id == "eval_gap_1"
    assert "Gather further evidence and elaborate on the core findings" in default_q.question
    assert default_q.status == "pending"
