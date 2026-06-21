import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from core.models import Report, Claim, VerifiedSource, ReportConfidenceScore
from verification.pipeline import VerificationPipeline

@pytest.mark.asyncio
async def test_verification_pipeline_success():
    # Arrange
    pipeline = VerificationPipeline()
    
    mock_claims = [
        # Claim 1: High confidence (backed by 2 verified sources)
        Claim(
            claim_id="c1",
            claim_text="Company X grew revenue by 15%",
            section="Finance",
            supporting_quotes=["grew revenue by 15 percent"],
            source_urls=["https://finance.com", "https://sec.gov"]
        ),
        # Claim 2: Medium confidence (backed by 1 verified source)
        Claim(
            claim_id="c2",
            claim_text="Post-quantum cryptography is coming",
            section="Security",
            supporting_quotes=["PQC by 2028"],
            source_urls=["https://nist.gov"]
        ),
        # Claim 3: Information gap claim (0 verified sources)
        Claim(
            claim_id="c3",
            claim_text="Information gap claim",
            section="Summary",
            supporting_quotes=["no quotes match this"],
            source_urls=["https://unreachable.com"]
        )
    ]
    
    # Mock ClaimExtractor to return our mock claims
    mock_extract_claims = AsyncMock(return_value=mock_claims)
    
    # Mock SourceChecker to return VerifiedSources
    async def mock_check_source(url):
        if url == "https://finance.com":
            return VerifiedSource(
                url=url,
                title="Finance Page",
                content="Company X grew revenue by 15 percent in Q3.",
                accessible=True,
                status_code=200
            )
        elif url == "https://sec.gov":
            return VerifiedSource(
                url=url,
                title="SEC Filing",
                content="Our revenue grew by 15 percent as reported.",
                accessible=True,
                status_code=200
            )
        elif url == "https://nist.gov":
            return VerifiedSource(
                url=url,
                title="NIST Security",
                content="NIST standards will require PQC by 2028.",
                accessible=True,
                status_code=200
            )
        else:
            return VerifiedSource(
                url=url,
                title="",
                content="",
                accessible=False,
                status_code=404,
                error_message="HTTP Error 404"
            )
            
    mock_check = AsyncMock(side_effect=mock_check_source)
    
    with patch.object(pipeline.extractor, "extract_claims", mock_extract_claims), \
         patch.object(pipeline.checker, "check_source", mock_check):
         
        report = Report(
            title="Q3 Business Overview",
            content="Draft report content...",
            claims=[],
            citations=["https://finance.com", "https://sec.gov", "https://nist.gov", "https://unreachable.com"]
        )
        
        # Act
        verified_report, stats = await pipeline.run_verification_pipeline(report)
        
        # Assert
        assert isinstance(verified_report, Report)
        assert isinstance(stats, ReportConfidenceScore)
        
        # Report claims list updated
        assert len(verified_report.claims) == 3
        
        # Claim 1 should be High confidence (1.0)
        c1 = verified_report.claims[0]
        assert c1.claim_id == "c1"
        assert c1.verification_status == "verified"
        assert c1.confidence_score == 1.0  # High confidence (2 verified sources, perfect quote match)
        
        # Claim 2 should be Medium confidence (0.7)
        c2 = verified_report.claims[1]
        assert c2.claim_id == "c2"
        assert c2.verification_status == "verified"
        assert c2.confidence_score == 0.7  # Medium confidence (1 verified source, perfect quote match)
        
        # Claim 3 should be a gap (0 verified sources)
        c3 = verified_report.claims[2]
        assert c3.claim_id == "c3"
        assert c3.verification_status == "gap"
        assert c3.confidence_score == 0.0
        
        # Aggregate statistics checking
        assert stats.total_claims_count == 3
        assert stats.verified_claims_count == 2
        assert stats.gaps_count == 1
        assert stats.failed_claims_count == 0
        assert stats.overall_score == 0.57  # round((1.0 + 0.7 + 0.0) / 3, 2) = 0.57
        assert verified_report.confidence_score == 0.57
