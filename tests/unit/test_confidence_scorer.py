import pytest
from core.models import Claim, QuoteVerification, VerificationResult
from verification.confidence_scorer import ConfidenceScorer

def test_confidence_scorer_high_confidence():
    scorer = ConfidenceScorer()
    
    claim = Claim(
        claim_id="c1",
        claim_text="Company X grew revenue by 15%",
        section="Finance",
        source_urls=["https://source1.com", "https://source2.com"]
    )
    
    quotes_verification = [
        QuoteVerification(quote="grew revenue by 15%", is_verified=True, score=1.0, matched_text="grew revenue by 15%")
    ]
    
    source_statuses = {
        "https://source1.com": True,
        "https://source2.com": True
    }
    
    result = scorer.score_claim(claim, quotes_verification, source_statuses)
    
    assert isinstance(result, VerificationResult)
    assert result.status == "verified"
    assert result.confidence_score == 1.0  # 0.8 + 0.2 * 1.0
    assert claim.verification_status == "verified"
    assert claim.confidence_score == 1.0
    assert "High confidence" in claim.remediation_notes

def test_confidence_scorer_medium_confidence():
    scorer = ConfidenceScorer()
    
    claim = Claim(
        claim_id="c2",
        claim_text="NIST expects PQC deployment by 2028",
        section="Security",
        source_urls=["https://nist.gov", "https://inaccessible.gov"]
    )
    
    quotes_verification = [
        QuoteVerification(quote="deployment by 2028", is_verified=True, score=0.90, matched_text="deployment in 2028")
    ]
    
    source_statuses = {
        "https://nist.gov": True,
        "https://inaccessible.gov": False
    }
    
    result = scorer.score_claim(claim, quotes_verification, source_statuses)
    
    assert result.status == "verified"
    assert result.confidence_score == 0.68  # round(0.5 + 0.2 * 0.9, 2)
    assert claim.verification_status == "verified"
    assert claim.confidence_score == 0.68
    assert "Medium confidence" in claim.remediation_notes

def test_confidence_scorer_quote_failed():
    scorer = ConfidenceScorer()
    
    claim = Claim(
        claim_id="c3",
        claim_text="A hallucinated claim",
        section="Executive Summary",
        source_urls=["https://source1.com"]
    )
    
    quotes_verification = [
        # One quote is verified, one fails
        QuoteVerification(quote="some text", is_verified=True, score=1.0, matched_text="some text"),
        QuoteVerification(quote="fake quote", is_verified=False, score=0.2, matched_text=None)
    ]
    
    source_statuses = {
        "https://source1.com": True
    }
    
    result = scorer.score_claim(claim, quotes_verification, source_statuses)
    
    assert result.status == "failed"
    assert result.confidence_score == 0.0
    assert claim.verification_status == "failed"
    assert claim.confidence_score == 0.0
    assert "Quote verification failed" in claim.remediation_notes

def test_confidence_scorer_information_gap():
    scorer = ConfidenceScorer()
    
    claim = Claim(
        claim_id="c4",
        claim_text="No sources accessible",
        section="Summary",
        source_urls=["https://source1.com"]
    )
    
    quotes_verification = [
        QuoteVerification(quote="some text", is_verified=True, score=1.0, matched_text="some text")
    ]
    
    source_statuses = {
        "https://source1.com": False
    }
    
    result = scorer.score_claim(claim, quotes_verification, source_statuses)
    
    assert result.status == "gap"
    assert result.confidence_score == 0.0
    assert claim.verification_status == "gap"
    assert claim.confidence_score == 0.0
    assert "Information gap" in claim.remediation_notes
