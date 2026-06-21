import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import AIMessage

from core.models import GraphState, Report, Claim
from core.nodes.writer import report_writer_node

@pytest.mark.asyncio
async def test_writer_node_no_draft():
    state = GraphState(
        topic="Company X Strategy",
        draft_report=None
    )
    
    mock_json = """{
        "title": "Final Research Report: Company X",
        "content": "# Final Research Report: Company X\\n\\nSome final markdown report details."
    }"""
    mock_response = AIMessage(content=mock_json)
    
    with patch("core.nodes.scoping.router.ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value = mock_response
        
        updates = await report_writer_node(state)
        
        assert "final_report" in updates
        assert isinstance(updates["final_report"], Report)
        assert updates["final_report"].title == "Final Research Report: Company X"
        assert "# Final Research Report" in updates["final_report"].content

@pytest.mark.asyncio
async def test_writer_node_success():
    draft_claims = [
        Claim(
            claim_id="c1",
            claim_text="Company X grew revenue by 15%",
            section="Executive Summary",
            supporting_quotes=["revenue by 15%"],
            source_urls=["https://finance.com"],
            verification_status="verified",
            confidence_score=0.70
        )
    ]
    
    state = GraphState(
        topic="Company X Strategy",
        draft_report=Report(
            title="Aggregated Findings: Company X",
            content="Aggregated draft findings",
            claims=draft_claims,
            citations=["https://finance.com"],
            confidence_score=0.70
        )
    )
    
    mock_json = """{
        "title": "Final Academic Report: Company X",
        "content": "# Final Academic Report: Company X\\n\\nVerified content here."
    }"""
    mock_response = AIMessage(content=mock_json)
    
    with patch("core.nodes.scoping.router.ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value = mock_response
        
        updates = await report_writer_node(state)
        
        assert "final_report" in updates
        final = updates["final_report"]
        assert final.title == "Final Academic Report: Company X"
        assert final.confidence_score == 0.70
        assert len(final.claims) == 1
        assert final.claims[0].claim_id == "c1"

@pytest.mark.asyncio
async def test_writer_node_parsing_failure():
    state = GraphState(
        topic="Company X Strategy",
        draft_report=Report(
            title="Aggregated Findings: Company X",
            content="Aggregated draft findings",
            claims=[],
            citations=[],
            confidence_score=1.0
        )
    )
    
    # Invalid JSON that will cause parsing to fail
    bad_response = AIMessage(content="Sorry, I cannot format this as JSON.")
    
    with patch("core.nodes.scoping.router.ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value = bad_response
        
        updates = await report_writer_node(state)
        
        assert "final_report" in updates
        final = updates["final_report"]
        assert final.title == "Aggregated Findings: Company X" # Fallback title from draft
        assert final.content == "Aggregated draft findings" # Fallback content
        assert "errors" in updates
        assert any("parsing failed" in err for err in updates["errors"])
