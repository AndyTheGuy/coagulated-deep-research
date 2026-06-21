import re
import structlog
from typing import List, Optional
from rapidfuzz import fuzz

from core.models import Claim, VerifiedSource, QuoteVerification

logger = structlog.get_logger("deep-research")

class QuoteVerifier:
    """Fuzzy quote verifier that checks whether each literal quote from a claim actually appears in its cited source document."""

    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = threshold

    def find_exact_normalized_match(self, quote: str, document: str) -> Optional[str]:
        """Find an exact match where casing and extra whitespaces are normalized, returning original text from document."""
        q_words = quote.split()
        d_words = document.split()
        if not q_words or not d_words:
            return None
        
        q_norm = " ".join(q_words).lower()
        w_size = len(q_words)
        for i in range(len(d_words) - w_size + 1):
            window_words = d_words[i:i+w_size]
            window_text = " ".join(window_words)
            if window_text.lower() == q_norm:
                return window_text
        return None

    def verify_quote(self, quote: str, document_content: str) -> QuoteVerification:
        """Verify a single quote against document content using exact and fuzzy matching."""
        if not quote or not document_content:
            return QuoteVerification(quote=quote, is_verified=False, score=0.0, matched_text=None)

        # 1. Try exact match first (whitespace-normalized, case-insensitive)
        exact_match = self.find_exact_normalized_match(quote, document_content)
        if exact_match:
            logger.debug("Exact quote match found", quote=quote)
            return QuoteVerification(
                quote=quote,
                is_verified=True,
                score=1.0,
                matched_text=exact_match
            )

        # 2. Fuzzy match using rapidfuzz partial ratio
        score = fuzz.partial_ratio(quote, document_content) / 100.0
        is_verified = score >= self.threshold

        if not is_verified:
            logger.debug("Quote failed verification", quote=quote, score=score)
            return QuoteVerification(
                quote=quote,
                is_verified=False,
                score=score,
                matched_text=None
            )

        # 3. Find the best matching segment (matched_text) in the document
        # We split the document into sentences and scan sliding windows
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', document_content) if s.strip()]
        best_segment = ""
        best_segment_score = 0.0

        # Check sliding windows of sentences of size 1, 2, and 3
        for size in range(1, 4):
            for i in range(len(sentences) - size + 1):
                segment = " ".join(sentences[i:i+size])
                segment_score = fuzz.ratio(quote, segment) / 100.0
                if segment_score > best_segment_score:
                    best_segment_score = segment_score
                    best_segment = segment

        matched_text = best_segment if best_segment else None

        logger.debug("Fuzzy quote match found", quote=quote, score=score, matched_text=matched_text)
        return QuoteVerification(
            quote=quote,
            is_verified=is_verified,
            score=score,
            matched_text=matched_text
        )

    def verify_claim_quotes(self, claim: Claim, sources: List[VerifiedSource]) -> List[QuoteVerification]:
        """Verify all supporting quotes of a claim against available verified sources."""
        verifications = []
        combined_content = " ".join([src.content for src in sources if src.accessible and src.content])
        
        for quote in claim.supporting_quotes:
            verification = self.verify_quote(quote, combined_content)
            verifications.append(verification)
            
        return verifications
