import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import AIMessage
from core.models import Report, Claim
from verification.claim_extractor import ClaimExtractor

@pytest.mark.asyncio
async def test_claim_extractor_success():
    extractor = ClaimExtractor()
    
    mock_json_content = """{
        "claims": [
            {
                "claim_id": "c1",
                "claim_text": "Company X grew revenue by 15% in Q3",
                "section": "Executive Summary",
                "supporting_quotes": ["Company X grew revenue by 15% in Q3, outperforming expectations."],
                "source_urls": ["https://companyx.com/report"]
            },
            {
                "claim_id": "c2",
                "claim_text": "Post-quantum cryptography deployment is expected by 2028",
                "section": "Security Assessment",
                "supporting_quotes": ["NIST expects full deployment of PQC by 2028."],
                "source_urls": ["https://nist.gov/pqc"]
            }
        ]
    }"""
    
    mock_response = AIMessage(content=mock_json_content)
    
    with patch("core.nodes.scoping.router.ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value = mock_response
        
        report = Report(
            title="Q3 Security Report",
            content="# Executive Summary\nCompany X grew revenue by 15% in Q3, outperforming expectations.\n\n# Security Assessment\nNIST expects full deployment of PQC by 2028.",
            claims=[],
            citations=["https://companyx.com/report", "https://nist.gov/pqc"]
        )
        
        claims = await extractor.extract_claims(report)
        
        assert len(claims) == 2
        assert claims[0].claim_id == "c1"
        assert claims[0].claim_text == "Company X grew revenue by 15% in Q3"
        assert claims[0].section == "Executive Summary"
        assert claims[0].supporting_quotes == ["Company X grew revenue by 15% in Q3, outperforming expectations."]
        assert claims[0].source_urls == ["https://companyx.com/report"]
        assert claims[0].verification_status == "unverified"
        
        assert claims[1].claim_id == "c2"
        assert claims[1].claim_text == "Post-quantum cryptography deployment is expected by 2028"
        assert claims[1].section == "Security Assessment"
        assert claims[1].supporting_quotes == ["NIST expects full deployment of PQC by 2028."]
        assert claims[1].source_urls == ["https://nist.gov/pqc"]
        
        mock_ainvoke.assert_called_once()

@pytest.mark.asyncio
async def test_claim_extractor_failure_fallback():
    extractor = ClaimExtractor()
    
    # Simulating LLM exception
    with patch("core.nodes.scoping.router.ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.side_effect = Exception("LLM connection timed out")
        
        report = Report(
            title="Q3 Security Report",
            content="# Executive Summary\nSome content.",
            claims=[]
        )
        
        claims = await extractor.extract_claims(report)
        assert claims == []
