import asyncio
import structlog
from typing import List, Dict, Optional, Tuple

from core.models import Report, Claim, VerifiedSource, ReportConfidenceScore
from db.cache import SemanticCache
from verification.claim_extractor import ClaimExtractor
from verification.source_checker import SourceChecker
from verification.quote_verifier import QuoteVerifier
from verification.confidence_scorer import ConfidenceScorer

logger = structlog.get_logger("deep-research")

class VerificationPipeline:
    """Orchestrates the entire verification pipeline: extraction, source checking, quote verification, and scoring."""

    def __init__(self, cache: Optional[SemanticCache] = None) -> None:
        self.cache = cache or SemanticCache()
        self.extractor = ClaimExtractor()
        self.checker = SourceChecker(cache=self.cache)
        self.verifier = QuoteVerifier()
        self.scorer = ConfidenceScorer()

    async def run_verification_pipeline(self, report: Report) -> Tuple[Report, ReportConfidenceScore]:
        """Run all 4 verification stages sequentially on a report and return the annotated report and confidence stats."""
        logger.info("Starting verification pipeline for report", title=report.title)
        
        # Stage 1: Extract claims
        logger.info("Verification Stage 1: Extracting claims from report")
        claims = await self.extractor.extract_claims(report)
        if not claims:
            logger.info("No claims extracted from report")
            report.claims = []
            report.confidence_score = 1.0
            stats = ReportConfidenceScore(
                overall_score=1.0,
                verified_claims_count=0,
                total_claims_count=0,
                unverified_claims_count=0,
                failed_claims_count=0,
                gaps_count=0
            )
            return report, stats

        logger.info("Extracted claims", count=len(claims))
        
        # Gather all unique cited source URLs
        unique_urls = set()
        for claim in claims:
            for url in claim.source_urls:
                if url:
                    unique_urls.add(url)
                    
        # Stage 2: Check all cited URLs concurrently
        logger.info("Verification Stage 2: Checking source accessibility and fetching content", unique_urls_count=len(unique_urls))
        checker_tasks = [self.checker.check_source(url) for url in unique_urls]
        verified_sources_list = await asyncio.gather(*checker_tasks, return_exceptions=True)
        
        # Build source mappings
        url_to_source: Dict[str, VerifiedSource] = {}
        source_statuses: Dict[str, bool] = {}
        
        for url, src in zip(unique_urls, verified_sources_list):
            if isinstance(src, Exception):
                logger.error("Failed to check source URL due to exception", url=url, error=str(src))
                url_to_source[url] = VerifiedSource(url=url, title="", content="", accessible=False, error_message=str(src))
                source_statuses[url] = False
            else:
                url_to_source[url] = src
                source_statuses[url] = src.accessible

        # Stage 3 & 4: Quote verification and Confidence scoring for each claim
        logger.info("Verification Stages 3 & 4: Verifying quotes and scoring confidence for each claim")
        
        verified_claims_count = 0
        unverified_claims_count = 0
        failed_claims_count = 0
        gaps_count = 0
        total_score_sum = 0.0
        
        for claim in claims:
            # Map claim's source URLs to their VerifiedSource objects
            claim_sources = [url_to_source[url] for url in claim.source_urls if url in url_to_source]
            
            # Stage 3: Verify quotes
            quote_verifications = self.verifier.verify_claim_quotes(claim, claim_sources)
            
            # Stage 4: Score claim confidence (which updates claim in-place)
            self.scorer.score_claim(claim, quote_verifications, source_statuses)
            
            # Aggregate stats
            total_score_sum += claim.confidence_score
            if claim.verification_status == "verified":
                verified_claims_count += 1
            elif claim.verification_status == "failed":
                failed_claims_count += 1
            elif claim.verification_status == "gap":
                gaps_count += 1
            else:
                unverified_claims_count += 1

        total_claims_count = len(claims)
        overall_score = round(total_score_sum / total_claims_count, 2) if total_claims_count > 0 else 1.0
        
        report.claims = claims
        report.confidence_score = overall_score
        
        stats = ReportConfidenceScore(
            overall_score=overall_score,
            verified_claims_count=verified_claims_count,
            total_claims_count=total_claims_count,
            unverified_claims_count=unverified_claims_count,
            failed_claims_count=failed_claims_count,
            gaps_count=gaps_count
        )
        
        logger.info(
            "Verification pipeline complete", 
            overall_score=overall_score, 
            total_claims=total_claims_count, 
            verified=verified_claims_count,
            failed=failed_claims_count,
            gaps=gaps_count
        )
        
        return report, stats
