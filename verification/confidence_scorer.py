import structlog
from typing import List, Dict
from core.models import Claim, QuoteVerification, VerificationResult

logger = structlog.get_logger("deep-research")

class ConfidenceScorer:
    """Confidence scorer that evaluates claim correctness, source accessibility, and quote verification details."""

    def __init__(self) -> None:
        pass

    def score_claim(
        self, 
        claim: Claim, 
        quotes_verification: List[QuoteVerification], 
        source_statuses: Dict[str, bool]
    ) -> VerificationResult:
        """Calculate confidence score and determine verification status for a single claim."""
        logger.info("Scoring claim confidence", claim_id=claim.claim_id)
        
        # 1. Count verified sources (accessible and listed in claim's source_urls)
        verified_sources_count = 0
        for url in claim.source_urls:
            if source_statuses.get(url, False):
                verified_sources_count += 1
                
        # 2. Check quote verifications
        quote_failed = False
        avg_quote_score = 1.0
        
        if quotes_verification:
            total_score = 0.0
            for q_ver in quotes_verification:
                total_score += q_ver.score
                if not q_ver.is_verified:
                    quote_failed = True
            avg_quote_score = total_score / len(quotes_verification)
        else:
            # If no quotes are provided, default to average score of 1.0
            avg_quote_score = 1.0

        # 3. Determine status and calculate score
        status = "unverified"
        confidence_score = 0.0
        remediation_notes = None

        if verified_sources_count == 0:
            status = "gap"
            confidence_score = 0.0
            remediation_notes = "Information gap: no accessible and verified sources found to back this claim."
        elif quote_failed:
            status = "failed"
            confidence_score = 0.0
            remediation_notes = "Quote verification failed: one or more quotes do not match cited source content."
        elif verified_sources_count == 1:
            status = "verified"
            # Medium confidence: Base of 0.5, plus up to 0.2 from quote quality (ranges from 0.5 to 0.7)
            confidence_score = round(0.5 + 0.2 * avg_quote_score, 2)
            remediation_notes = "Verified (Medium confidence): claim is supported by exactly one verified source."
        else:  # verified_sources_count >= 2
            status = "verified"
            # High confidence: Base of 0.8, plus up to 0.2 from quote quality (ranges from 0.8 to 1.0)
            confidence_score = round(0.8 + 0.2 * avg_quote_score, 2)
            remediation_notes = f"Verified (High confidence): claim is supported by {verified_sources_count} verified sources."

        logger.info(
            "Claim scored", 
            claim_id=claim.claim_id, 
            status=status, 
            score=confidence_score, 
            verified_sources=verified_sources_count
        )

        # Update the claim object itself in place
        claim.verification_status = status
        claim.confidence_score = confidence_score
        claim.remediation_notes = remediation_notes

        return VerificationResult(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            status=status,
            confidence_score=confidence_score,
            quotes_verification=quotes_verification,
            source_status={url: source_statuses.get(url, False) for url in claim.source_urls}
        )
