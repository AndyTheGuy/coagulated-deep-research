import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.messages import AIMessage

from core.models import Report, Claim, VerifiedSource, ReportConfidenceScore
from verification.pipeline import VerificationPipeline

@pytest.mark.asyncio
async def test_verification_pipeline_integration():
    """Integration test verifying that the verification pipeline correctly orchestrates 
    Claim Extraction, Source Checking, Quote Verification, and Confidence Scoring.
    """
    
    # 1. Prepare sample report
    sample_report = Report(
        title="AI Trends 2026",
        content=(
            "# AI Trends 2026\n\n"
            "## Trends\n"
            "AI investments will hit $200B by 2025 as per industry tracking.\n\n"
            "## Risks\n"
            "There is slower adoption due to regulatory hurdles in the EU.\n"
        )
    )
    
    # 2. Mock standard-tier LLM claim extraction response
    # The JSON response contains two claims: c1 (verified) and c2 (unaccessible source -> gap)
    mock_extractor_json = """{
        "claims": [
            {
                "claim_id": "c1",
                "claim_text": "AI investments will hit $200B by 2025",
                "section": "Trends",
                "supporting_quotes": ["hit $200B by 2025"],
                "source_urls": ["https://research.com/ai-trends"]
            },
            {
                "claim_id": "c2",
                "claim_text": "Slower adoption due to regulatory hurdles",
                "section": "Risks",
                "supporting_quotes": ["regulatory hurdles"],
                "source_urls": ["https://failures.com/gap"]
            }
        ]
    }"""
    mock_response = AIMessage(content=mock_extractor_json)
    
    # 3. Prepare cache mock
    # https://research.com/ai-trends is in cache -> accessible
    # https://failures.com/gap is NOT in cache -> will trigger live HTTP fallback (which we will mock to return 404)
    mock_cache = MagicMock()
    async def mock_get_url(url: str):
        if url == "https://research.com/ai-trends":
            return {
                "title": "AI Trends Page",
                "content": "A report tracking investments. AI investments will hit $200B by 2025 in major markets."
            }
        return None
        
    mock_cache.get_url = AsyncMock(side_effect=mock_get_url)
    mock_cache.set_url = AsyncMock()
    
    # 4. Mock Live HTTP GET client for the cache miss on failures.com/gap (return 404)
    mock_http_response = MagicMock()
    mock_http_response.status_code = 404
    mock_http_client = MagicMock()
    mock_http_client.get = AsyncMock(return_value=mock_http_response)
    
    # Initialize pipeline with our mocked cache
    pipeline = VerificationPipeline(cache=mock_cache)
    
    # 5. Execute pipeline while patching LLM router and httpx AsyncClient
    with patch("core.nodes.scoping.router.ainvoke", new_callable=AsyncMock) as mock_ainvoke, \
         patch("httpx.AsyncClient") as mock_client_class:
         
        mock_ainvoke.return_value = mock_response
        
        # Patch httpx.AsyncClient context manager
        mock_client_ctx = AsyncMock()
        mock_client_ctx.__aenter__.return_value = mock_http_client
        mock_client_class.return_value = mock_client_ctx
        
        # Run pipeline
        verified_report, stats = await pipeline.run_verification_pipeline(sample_report)
        
        # 6. Verify pipeline integration results
        assert isinstance(verified_report, Report)
        assert isinstance(stats, ReportConfidenceScore)
        
        # Claims list size check
        assert len(verified_report.claims) == 2
        
        # Verify first claim (c1) status and score
        c1 = verified_report.claims[0]
        assert c1.claim_id == "c1"
        assert c1.verification_status == "verified"
        assert c1.confidence_score == 0.70 # Medium confidence (1 verified source)
        
        # Verify second claim (c2) status and score
        c2 = verified_report.claims[1]
        assert c2.claim_id == "c2"
        assert c2.verification_status == "gap" # Inaccessible source -> gap
        assert c2.confidence_score == 0.0 # 0 accessible sources
        
        # Verify aggregated statistics
        assert stats.total_claims_count == 2
        assert stats.verified_claims_count == 1
        assert stats.gaps_count == 1
        assert stats.failed_claims_count == 0
        assert stats.overall_score == 0.35 # Average of 0.70 and 0.0
        
        # Verify report overall score carried over
        assert verified_report.confidence_score == 0.35
        
        # Verify cache lookup was performed
        assert mock_cache.get_url.call_count >= 2
