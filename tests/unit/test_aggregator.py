import pytest
from core.models import GraphState, SubQuestion, VerifiedSource, Report
from core.nodes.aggregator import context_aggregator_node

@pytest.mark.asyncio
async def test_context_aggregator_node_success():
    state = GraphState(
        topic="AI Safety",
        sub_questions_state=[
            SubQuestion(
                id="q1", 
                question="What is RLHF?", 
                status="completed", 
                results_summary="RLHF stands for reinforcement learning from human feedback."
            ),
            SubQuestion(
                id="q2", 
                question="What is RLAIF?", 
                status="completed", 
                results_summary="RLAIF stands for reinforcement learning from AI feedback."
            ),
            SubQuestion(
                id="q3", 
                question="What is DPO?", 
                status="pending", 
                results_summary=None
            )
        ],
        verified_sources=[
            VerifiedSource(url="https://example.com/rlhf", title="RLHF paper", content="Some text...", accessible=True),
            VerifiedSource(url="https://example.com/rlhf/", title="RLHF paper dup", content="Some text...", accessible=True),
            VerifiedSource(url="https://example.com/rlaif", title="RLAIF paper", content="Some text...", accessible=True),
            VerifiedSource(url="https://example.com/inaccessible", title="Failed link", content="Failed", accessible=False),
        ]
    )

    result = await context_aggregator_node(state)

    assert "draft_report" in result
    assert "logs" in result

    report = result["draft_report"]
    assert isinstance(report, Report)
    assert "Aggregated Findings: AI Safety" in report.title
    assert "RLHF" in report.content
    assert "RLAIF" in report.content
    assert "DPO" not in report.content  # Pending, not completed

    # Citations verification (unique and accessible only)
    assert "https://example.com/rlhf" in report.citations
    assert "https://example.com/rlaif" in report.citations
    assert "https://example.com/inaccessible" not in report.citations
    assert len(report.citations) == 2

    # Bibliography formatting check
    assert "[RLHF paper](https://example.com/rlhf)" in report.content
    assert "[RLAIF paper](https://example.com/rlaif)" in report.content

@pytest.mark.asyncio
async def test_context_aggregator_node_no_completed():
    state = GraphState(
        topic="Incomplete Research",
        sub_questions_state=[
            SubQuestion(id="q1", question="What is RLHF?", status="pending")
        ]
    )

    result = await context_aggregator_node(state)

    assert "draft_report" in result
    report = result["draft_report"]
    assert isinstance(report, Report)
    assert "No completed researcher agent findings" in report.content
    assert len(report.citations) == 0
