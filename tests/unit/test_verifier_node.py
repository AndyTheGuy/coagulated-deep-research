import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.messages import AIMessage

from core.models import GraphState, Report, SubQuestion, ReportConfidenceScore
from core.nodes.verifier import verifier_node

@pytest.mark.asyncio
async def test_verifier_node_no_draft_report():
    # Arrange
    state = GraphState(
        topic="Post-quantum cryptography",
        draft_report=None
    )
    
    # Act
    updates = await verifier_node(state)
    
    # Assert
    assert "logs" in updates
    assert "No draft report available" in updates["logs"][0]

@pytest.mark.asyncio
async def test_verifier_node_gaps_found():
    # Arrange
    state = GraphState(
        topic="Company X Strategy",
        draft_report=Report(
            title="Aggregated Findings: Company X",
            content="Some draft text",
            citations=["https://finance.com"]
        )
    )
    
    # Mock VerificationPipeline
    mock_pipeline_inst = MagicMock()
    mock_pipeline_inst.run_verification_pipeline = AsyncMock(return_value=(
        Report(
            title="Aggregated Findings: Company X",
            content="Some draft text",
            citations=["https://finance.com"]
        ),
        ReportConfidenceScore(
            overall_score=0.30,
            verified_claims_count=0,
            total_claims_count=1,
            unverified_claims_count=0,
            failed_claims_count=0,
            gaps_count=1
        )
    ))
    
    # Mock LLM critique response
    mock_critique_json = """{
        "gaps_found": true,
        "critique_text": "The report lacks detail on competitors.",
        "suggested_queries": ["Who are Company X's top competitors?"]
    }"""
    mock_response = AIMessage(content=mock_critique_json)
    
    with patch("core.nodes.verifier.VerificationPipeline", return_value=mock_pipeline_inst), \
         patch("core.nodes.scoping.router.ainvoke", new_callable=AsyncMock) as mock_ainvoke:
         
        mock_ainvoke.return_value = mock_response
        
        # Act
        updates = await verifier_node(state)
        
        # Assert
        assert "draft_report" in updates
        assert "sub_questions_state" in updates
        assert len(updates["sub_questions_state"]) == 1
        
        # Check newly created pending sub-question
        new_q = updates["sub_questions_state"][0]
        assert new_q.id == "gap_1"
        assert new_q.question == "Who are Company X's top competitors?"
        assert new_q.status == "pending"
        
        assert "logs" in updates
        assert any("Gaps found" in log for log in updates["logs"])
        
        mock_ainvoke.assert_called_once()

@pytest.mark.asyncio
async def test_verifier_node_no_gaps_found():
    # Arrange
    state = GraphState(
        topic="Company X Strategy",
        draft_report=Report(
            title="Aggregated Findings: Company X",
            content="Some draft text",
            citations=["https://finance.com"]
        )
    )
    
    mock_pipeline_inst = MagicMock()
    mock_pipeline_inst.run_verification_pipeline = AsyncMock(return_value=(
        Report(
            title="Aggregated Findings: Company X",
            content="Some draft text",
            citations=["https://finance.com"]
        ),
        ReportConfidenceScore(
            overall_score=0.95,
            verified_claims_count=5,
            total_claims_count=5,
            unverified_claims_count=0,
            failed_claims_count=0,
            gaps_count=0
        )
    ))
    
    mock_critique_json = """{
        "gaps_found": false,
        "critique_text": "The report looks great and is fully verified.",
        "suggested_queries": []
    }"""
    mock_response = AIMessage(content=mock_critique_json)
    
    with patch("core.nodes.verifier.VerificationPipeline", return_value=mock_pipeline_inst), \
         patch("core.nodes.scoping.router.ainvoke", new_callable=AsyncMock) as mock_ainvoke:
         
        mock_ainvoke.return_value = mock_response
        
        # Act
        updates = await verifier_node(state)
        
        # Assert
        assert "draft_report" in updates
        assert "sub_questions_state" in updates
        assert len(updates["sub_questions_state"]) == 0
        
        assert "logs" in updates
        assert any("No gaps found" in log for log in updates["logs"])
        
        mock_ainvoke.assert_called_once()
